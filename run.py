#!/usr/bin/env python3
import argparse
import os

import torch
import wandb

import horizon.model as module_arch
import horizon.trainer as module_trainer
from horizon.utils.config import ConfigParser
from horizon.utils.util import init_seeds, prepare_device


def main(args):
    # parse config and args
    config = ConfigParser(args)

    # set random seed
    init_seeds(config['trainer']['seed'])

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['trainer']['gpu_ids'],
                                        config['trainer']['use_gpu'])

    # build model architecture
    model = config.init_obj('arch', module_arch)
    if model:
        model = model.to(device)
    else:
        raise ModuleNotFoundError("model cannot be initialized.")
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    trainer = config.init_obj('trainer', module_trainer, config, model, device)
    trainer.run()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Horizon')
    args.add_argument('-c',
                      '--config_file',
                      default="config/defaults.yaml",
                      type=str,
                      help='config file(default: defaults.yaml)')
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
    args.add_argument('-m',
                      '--mode',
                      default='train',
                      type=str,
                      help='mode: train or test (default: train)')
    args.add_argument('-r',
                      '--resume',
                      default=None,
                      type=str,
                      help='path to specific checkpoint (default: None)')
    args, _ = args.parse_known_args()

    if not args.wandb:
        os.environ['WANDB_MODE'] = "disabled"

    wandb.login()
    try:
        main(args)
    except Exception as e:
        print(e)
    wandb.finish()
