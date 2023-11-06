from functools import wraps
import os
import random
import time

import numpy as np
import pandas as pd
import torch


def init_seeds(seed=0):
    """
    Fix random seeds for reproducibility.

    Args:
        seed (int, optional): _description_. Defaults to 0.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed(seed)
    # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_device(n_gpu):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        list_ids = list(range(1))  # [0]
    elif torch.cuda.is_available():
        n_gpu_all = torch.cuda.device_count()
        if n_gpu > 0 and n_gpu_all == 0:
            raise ValueError(
                "Error: GPU not available but n_gpu is {}".format(n_gpu))
        if n_gpu > n_gpu_all:
            raise ValueError(
                "Error: GPU not enough, we can only use {} GPU at most".format(
                    n_gpu_all))
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        list_ids = list(range(n_gpu))
    else:
        device = torch.device('cpu')
        list_ids = list(range(1))  # [0]
    return device, list_ids


def timer(phase):

    def decorate(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            t = time.time()
            ret = func(*args, **kwargs)
            t = time.time() - t
            wandb_ret = {phase + '/' + k: v for k, v in ret.items()}
            ret = {phase + '_' + k: v for k, v in ret.items()}
            t = '{:.0f}m {:.0f}s'.format(t // 60, t % 60)
            ret.update({phase + "_time": t})
            return ret, wandb_ret

        return wrapper

    return decorate


class MetricTracker:

    def __init__(self, *keys):
        self._data = pd.DataFrame(index=keys,
                                  columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / \
            self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
