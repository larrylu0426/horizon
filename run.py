#!/usr/bin/env python3
import argparse
import os

import torch
import wandb

import horizon.model as module_arch
import horizon.trainer as module_trainer
from horizon.utils.config import ConfigParser
import horizon.utils.loss as module_loss
import horizon.utils.metric as module_metric
from horizon.utils.util import init_seeds, prepare_device


def main(args):

    init_seeds(0)

    config = ConfigParser(args)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['trainer']['n_gpu'])

    # build model architecture
    model = config.init_obj('arch', module_arch)
    if model:
        model = model.to(device)
    else:
        raise ModuleNotFoundError
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss functions and metrics
    if type(config['loss']) == list:
        criterion = {}
        for loss in config['loss']:
            criterion[loss] = getattr(module_loss, loss)
    else:
        criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler,
                                   optimizer)

    trainer = config.init_obj('trainer', module_trainer, model, criterion,
                              metrics, optimizer, lr_scheduler, config, device)

    trainer.run()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Horizon')
    args.add_argument('-w',
                      '--wandb',
                      default=False,
                      type=bool,
                      help='flag of opening wandb (default: False)')
    args.add_argument('-n',
                      '--name',
                      default=None,
                      type=str,
                      help='project name used in wandb (default: Folder Name)')
    args.add_argument(
        '-c',
        '--configs',
        default="config/config-defaults.yaml",
        type=str,
        help='config file used in wandb (default: config-defaults.yaml)')
    args, unknown = args.parse_known_args()
    if not args.wandb:
        os.environ['WANDB_MODE'] = "disabled"
    wandb.login()
    main(args)
    wandb.finish()
