"""
Configuration system for Point Cloud Completion Boilerplate.

Uses YAML files with EasyDict for attribute-style access.
Supports hierarchical config: default.yaml -> model-specific.yaml -> CLI overrides.

Usage:
    from configs.config import load_config, cfg_from_args
    
    # Load from YAML
    cfg = load_config('configs/pcn.yaml')
    
    # Or from argparse
    cfg = cfg_from_args(args)
"""

import os
import yaml
import copy
from easydict import EasyDict as edict


def _dict_to_edict(d):
    """Recursively convert dict to EasyDict."""
    if isinstance(d, dict):
        return edict({k: _dict_to_edict(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_edict(v) for v in d]
    return d


def _merge_dict(base, override):
    """Recursively merge override into base dict."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge_dict(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_config(config_path):
    """
    Load a YAML config file. If `_base_` key exists, recursively loads the
    base config first and merges.

    Args:
        config_path: path to YAML config file

    Returns:
        EasyDict configuration object
    """
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    # Handle base config inheritance
    if '_base_' in cfg_dict:
        base_path = cfg_dict.pop('_base_')
        # Resolve relative to current config file's directory
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(config_path), base_path)
        base_cfg = load_config(base_path)
        base_dict = dict(base_cfg)
        cfg_dict = _merge_dict(base_dict, cfg_dict)

    return _dict_to_edict(cfg_dict)


def merge_configs(base_cfg, override_cfg):
    """Merge two config dicts, override takes precedence."""
    base_dict = dict(base_cfg) if isinstance(base_cfg, edict) else base_cfg
    override_dict = dict(override_cfg) if isinstance(override_cfg, edict) else override_cfg
    merged = _merge_dict(base_dict, override_dict)
    return _dict_to_edict(merged)


def cfg_from_args(args, config_path=None):
    """
    Build config from argparse args and optional YAML file.
    CLI args override YAML values.
    
    Args:
        args: argparse namespace  
        config_path: path to YAML config (or args.config if available)

    Returns:
        EasyDict configuration object
    """
    # Load YAML config
    path = config_path or getattr(args, 'config', None)
    if path and os.path.exists(path):
        cfg = load_config(path)
    else:
        cfg = load_config(os.path.join(os.path.dirname(__file__), 'default.yaml'))

    # Apply CLI overrides
    args_dict = vars(args)
    
    cli_mappings = {
        'model_name': ('model', 'name'),
        'batch_size': ('train', 'batch_size'),
        'epochs': ('train', 'n_epochs'),
        'lr': ('train', 'learning_rate'),
        'num_workers': ('train', 'num_workers'),
        'dataset': ('dataset', 'train'),
        'gpu': ('general', 'gpu'),
        'seed': ('general', 'seed'),
    }

    for arg_name, cfg_path in cli_mappings.items():
        if arg_name in args_dict and args_dict[arg_name] is not None:
            obj = cfg
            for key in cfg_path[:-1]:
                if key not in obj:
                    obj[key] = edict()
                obj = obj[key]
            obj[cfg_path[-1]] = args_dict[arg_name]

    return cfg
