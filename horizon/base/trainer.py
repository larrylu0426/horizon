import copy
import os

from abc import abstractmethod
from numpy import inf
import torch
import wandb

from horizon.utils.util import MetricTracker
from horizon.utils.util import timer


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, lr_scheduler,
                 config, device):
        self.config = config
        self.device = device
        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        _data_loaders = self._add_data_loaders()
        self.train_data, self.val_data, self.test_data = _data_loaders
        self.do_validation = True if self.val_data else False
        self.do_test = True if self.test_data else False

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.train_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns])
        self.test_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns])

    @abstractmethod
    def _add_data_loaders(self):
        """
        Define the data loaders
        :return: train_data_loader, valid_data_loader, test_data_loader
        """
        raise NotImplementedError

    @abstractmethod
    def _train_epoch(self):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        raise NotImplementedError

    @timer(phase="train")
    def train_epoch(self):
        return self._train_epoch()

    @abstractmethod
    def _valid_epoch(self):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        raise NotImplementedError

    @timer(phase="val")
    def valid_epoch(self):
        return self._valid_epoch()

    @abstractmethod
    def _test_model(self):
        """
        Test logic for the trained model

        :return: A log that contains average loss and metric of the model.
        """
        raise NotImplementedError

    @timer(phase="test")
    def test_model(self):
        return self._test_model()

    def run(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(1, self.epochs + 1):

            # save logged informations into log dict
            log = {'epoch': '{}/{}'.format(epoch, self.epochs)}
            wandb_log = {}
            result, wandb_result = self.train_epoch()
            log.update(result)
            wandb_log.update(wandb_result)

            if self.do_validation:
                result, wandb_result = self.valid_epoch()
                log.update(result)
                wandb_log.update(wandb_result)
            wandb.log({**wandb_log})

            # print logged informations to the screen
            for key, value in log.items():
                print('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not
                    improved = ((self.mnt_mode == 'min' and
                                 (log[self.mnt_metric] <= self.mnt_best))
                                or (self.mnt_mode == 'max' and
                                    (log[self.mnt_metric] >= self.mnt_best)))
                except KeyError:
                    print("Warning: Metric '{}' is not found. "
                          "Disable the monitoring.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    self.best_model = copy.deepcopy(self.model)
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation result didn\'t improve for {} epochs. "
                          "Training stops.".format(self.early_stop))
                    break
            else:
                self.best_model = copy.deepcopy(self.model)

            self.save_model()

        if self.do_test:
            result, wandb_result = self.test_model()
            for k, v in wandb_result.items():
                wandb.summary[k] = v
            # print logged informations to the screen
            for key, value in result.items():
                print('    {:15s}: {}'.format(str(key), value))

    def save_model(self):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str('{}.pt'.format(arch))
        torch.save(state, filename)
        art = wandb.Artifact(arch, type="model")
        art.add_file(filename)
        wandb.log_artifact(art)
        os.remove(filename)
