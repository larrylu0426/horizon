from datetime import datetime
from functools import partial
from functools import reduce
from operator import getitem
import pkgutil

import wandb


class ConfigParser:

    def __init__(self, args):
        # add configs from args
        for k, v in args.__dict__.items():
            setattr(self, k, v)
        # create a run of wandb
        self.wandb_run = wandb.init(project=args.name, config=args.config_file)
        self._config = _update_config(wandb.config.as_dict())

        wandb.run.name = self._config['dataset']['name'] + \
            '_' + self._config['arch']['name'] + \
            '_' + datetime.now().strftime(r'%m%d_%H%M_%S')
        
        if args.mode == "test":
            wandb.run.name = "test_" + wandb.run.name

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config,
        and returns the instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['name']
        if 'args' in self[name]:
            module_args = dict(self[name]['args'])
        else:
            module_args = dict()
        assert all([
            k not in module_args for k in kwargs
        ]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        try:
            return getattr(module, module_name)(*args, **module_args)
        except AttributeError:
            # given module is a package
            for m, n, _ in pkgutil.iter_modules(module.__path__):
                n = module.__name__ + "." + n
                m = m.find_module(n).load_module(n)
                if module_name in dir(m):
                    return getattr(m, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['name']
        module_args = dict(self[name]['args'])
        assert all([
            k not in module_args for k in kwargs
        ]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self._config[name]


def _update_config(config):
    # helper functions to update config dict with custom cli options
    modification = {}
    removed = []
    for k in config.keys():
        if len(k.split(".")) > 1:
            modification[k] = config[k]
            removed.append(k)
    if modification is None:
        return config
    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    for k in removed:
        del config[k]
    return config


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split('.')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
