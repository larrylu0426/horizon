import copy
import gc
import os
import sys

from abc import abstractmethod
from numpy import inf
import torch
import wandb

import horizon.utils.const as const
from horizon.utils.logger import Logger
import horizon.utils.loss as module_loss
import horizon.utils.metric as module_metric
from horizon.utils.util import DataPrefetcher
from horizon.utils.util import MetricTracker
from horizon.utils.util import timer


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device

        self.start_epoch = 1

        if self.config.mode == 'train':
            run_dir = os.path.join(const.SAVE_DIR, wandb.run.name)
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
            os.system('cp {} {}'.format(
                os.path.join(const.PROJECT_DIR, self.config.config_file),
                run_dir + "/config.yaml"))
            sys.stdout = Logger(run_dir + "/log.txt")

            if type(self.model) == torch.nn.DataParallel:
                path = os.path.join(
                    const.PROJECT_DIR,
                    self.model.module.__module__.replace('.', '/') + '.py')
            else:
                path = os.path.join(
                    const.PROJECT_DIR,
                    self.model.__module__.replace('.', '/') + '.py')
            os.system('cp {} {}'.format(
                path, os.path.join(const.SAVE_DIR, wandb.run.name,
                                   'model.py')))

        _data_loaders = self._add_data_loaders()
        self.train_data, self.val_data, self.test_data = _data_loaders
        if device.type == 'cuda':
            self.train_data = DataPrefetcher(self.train_data)
            self.val_data = DataPrefetcher(self.val_data)
            self.test_data = DataPrefetcher(self.test_data)

        self.do_validation = True if self.val_data else False
        self.do_test = True if self.test_data else False

        # get function handles of loss functions and metrics
        if type(config['loss']) == list:
            criterion = {}
            for loss in config['loss']:
                criterion[loss] = getattr(module_loss, loss)
        else:
            criterion = getattr(module_loss, config['loss'])

        # build optimizer, learning rate scheduler
        self.criterion = criterion
        self.metric_ftns = [
            getattr(module_metric, met) for met in config['metrics']
        ]
        trainable_params = filter(lambda p: p.requires_grad,
                                  model.parameters())
        self.optimizer = config.init_obj('optimizer', torch.optim,
                                         trainable_params)
        self.lr_scheduler = config.init_obj('lr_scheduler',
                                            torch.optim.lr_scheduler,
                                            self.optimizer)

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            if not self.mnt_mode in ['min', 'max']:
                raise ValueError(
                    "Error: The program monitor mode {} is not supported.".
                    format(self.mnt_mode))
            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.train_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns])
        self.val_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns])
        self.test_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns])

        self.not_improved_count = 0
        self.best = {
            'epoch': 0,
            'model': None,
            'result': {},
        }
        if self.config.resume is not None:
            self.resume_model(self.config.resume)

    def run(self):
        if not self.config.mode in ['train', 'test']:
            raise ValueError(
                "Error: The program mode {} is not supported.".format(
                    self.mode))

        if self.config.mode == 'train':
            for epoch in range(self.start_epoch, self.epochs + 1):
                if self.not_improved_count == self.early_stop:
                    print(
                        "Metric didn\'t improve for {} epochs, Training stops."
                        .format(self.early_stop))
                    break

                # save logged informations into log dict
                log = {'epoch': '{}/{}'.format(epoch, self.epochs)}
                wandb_log = {}
                result, wandb_result = self.train()
                log.update(result)
                wandb_log.update(wandb_result)

                if self.do_validation:
                    result, wandb_result = self.valid(self.model)
                    log.update(result)
                    wandb_log.update(wandb_result)

                # evaluate model performance according to configured metric
                if self.mnt_mode != 'off':
                    self.set_best(epoch, log, wandb_log)

                # print logged informations to the screen
                for key, value in log.items():
                    print('    {:15s}: {}'.format(str(key), value))
                wandb.log({**wandb_log})

                self.save_model(self.model, wandb_log, epoch)

            for k, v in self.best['result'].items():
                wandb.summary[k] = v
            wandb.summary['best_epoch'] = self.best['epoch']

            if self.mnt_mode != 'off':
                self.save_model(self.best['model'],
                                self.best['result'],
                                best=True)

            if self.do_test:
                print('Loading best model of {} for testing...'.format(
                    self.best['epoch']))
                self.test_model(self.best['model'], best=True)

        elif self.config.mode == 'test':
            self.test_model(self.model)

    def set_best(self, epoch, log, result):
        try:
            # check whether model performance improved or not
            improved = ((self.mnt_mode == 'min' and
                         (log[self.mnt_metric] <= self.mnt_best))
                        or (self.mnt_mode == 'max' and
                            (log[self.mnt_metric] >= self.mnt_best)))
        except KeyError:
            raise KeyError("Error: Metric '{}' is not found. ".format(
                self.mnt_metric))

        if improved:
            self.not_improved_count = 0
            self.best['epoch'] = epoch
            self.mnt_best = log[self.mnt_metric]
            for key, value in result.items():
                self.best['result'][key] = value
            self.best['model'] = None
            gc.collect()
            self.best['model'] = copy.deepcopy(self.model)
        else:
            self.not_improved_count += 1

    def save_model(self, model, result, epoch=None, best=False):
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'metric': result,
            'not_improved_count': self.not_improved_count,
            'best': self.best
        }
        ckpt_dir = os.path.join(const.SAVE_DIR, wandb.run.name, 'checkpoints')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        filename = str('{}_{}.pt'.format('model',
                                         epoch if not best else 'best'))
        path = os.path.join(ckpt_dir, filename)
        torch.save(state, path)

    def save_wandb_model(self, path):
        art = wandb.Artifact(wandb.run.name, type="model")
        art.add_file(path)
        wandb.log_artifact(art)

    def resume_model(self, path):
        state = torch.load(path)
        self.model.load_state_dict(state['state_dict'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])
        self.start_epoch = state['epoch'] + 1 if state[
            'epoch'] else state['best']['epoch'] + 1
        self.not_improved_count = state['not_improved_count']
        self.best = state['best']

    def test_model(self, model, best=False):
        result, wandb_result = self.test(model)
        for k, v in wandb_result.items():
            wandb.summary[k] = v
        # print logged informations to the screen
        for key, value in result.items():
            print('    {:15s}: {}'.format(str(key), value))
        if self.config.wandb:
            if best:
                path = os.path.join(const.SAVE_DIR, wandb.run.name,
                                    'checkpoints', 'model_best.pt')
                self.save_wandb_model(path)

    @timer(phase="train")
    def train(self):
        return self._train()

    @timer(phase="val")
    def valid(self, model):
        return self._valid(model)

    @timer(phase="test")
    def test(self, model):
        return self._test(model)

    @abstractmethod
    def _add_data_loaders(self):
        """
        Define the data loaders
        :return: train_data_loader, val_data_loader, test_data_loader
        """
        raise NotImplementedError

    @abstractmethod
    def _train(self):
        """
        Training logic for an epoch

        :return: A log that contains average loss and metric in this epoch.
        """
        raise NotImplementedError

    @abstractmethod
    def _valid(self, model):
        """
        Validation logic for the trained model

        :return: A log that contains average loss and metric of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def _test(self, model):
        """
        Test logic for the trained model

        :return: A log that contains average loss and metric of the model.
        """
        raise NotImplementedError
