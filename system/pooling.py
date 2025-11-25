import torch
import torch.nn as nn
from typing import Dict, Any, Type, List, Optional, Union, Tuple
from torch.nn import Module

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS


class PoolingFactory:
    POOLINGS: Dict[str, Type[nn.Module]] = {
        'max_pool1d': nn.MaxPool1d,
        'max_pool2d': nn.MaxPool2d,
        'max_pool3d': nn.MaxPool3d,
        'max_unpool1d': nn.MaxUnpool1d,
        'max_unpool2d': nn.MaxUnpool2d,
        'max_unpool3d': nn.MaxUnpool3d,
        'adaptive_max_pool1d': nn.AdaptiveMaxPool1d,
        'adaptive_max_pool2d': nn.AdaptiveMaxPool2d,
        'adaptive_max_pool3d': nn.AdaptiveMaxPool3d,
        'avg_pool1d': nn.AvgPool1d,
        'avg_pool2d': nn.AvgPool2d,
        'avg_pool3d': nn.AvgPool3d,
        'adaptive_avg_pool1d': nn.AdaptiveAvgPool1d,
        'adaptive_avg_pool2d': nn.AdaptiveAvgPool2d,
        'adaptive_avg_pool3d': nn.AdaptiveAvgPool3d,
        'global_max_pool1d': nn.AdaptiveMaxPool1d,
        'global_max_pool2d': nn.AdaptiveMaxPool2d,
        'global_max_pool3d': nn.AdaptiveMaxPool3d,
        'global_avg_pool1d': nn.AdaptiveAvgPool1d,
        'global_avg_pool2d': nn.AdaptiveAvgPool2d,
        'global_avg_pool3d': nn.AdaptiveAvgPool3d,
        'lppool1d': nn.LPPool1d,
        'lppool2d': nn.LPPool2d,
        'fractional_max_pool2d': nn.FractionalMaxPool2d,
    }

    DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {}


    @classmethod
    def get_available_poolings(cls) -> List[str]:
        return list(cls.POOLINGS.keys())

    @classmethod
    def get_pooling_info(cls, pool_name: str) -> Dict[str, Any]:
        if pool_name not in cls.POOLINGS:
            return {}

        defaults = cls._get_default_params()
        if pool_name not in defaults:
            raise ValueError(f"Missing default parameters for pooling layer '{pool_name}' in config/config.json - check pooling.defaults")
        return {
            'name': pool_name,
            'class': cls.POOLINGS[pool_name].__name__,
            'module': cls.POOLINGS[pool_name].__module__,
            'default_params': defaults.get(pool_name, {}),
            'description': cls._get_pooling_description(pool_name),
            'properties': cls._get_pooling_properties(pool_name),
        }

    @classmethod
    def create_pooling(cls,
                      pool_name: str,
                      custom_params: Optional[Dict[str, Any]] = None) -> nn.Module:
        if pool_name not in cls.POOLINGS:
            available = ', '.join(cls.get_available_poolings())
            raise ValueError(f"Unsupported pooling layer: {pool_name}. Available options: {available}")

        defaults = cls._get_default_params()
        if pool_name not in defaults:
            raise ValueError(f"Missing default parameters for pooling layer '{pool_name}' in config/config.json - check pooling.defaults")
        default_params = defaults.get(pool_name, {}).copy()

        if custom_params:
            default_params.update(custom_params)

        try:
            pool_class = cls.POOLINGS[pool_name]
            pooling = pool_class(**default_params)

            info(f"Pooling layer created successfully: {pool_name} with parameters {default_params}")
            return pooling

        except Exception as e:
            error(f"Failed to create pooling layer: {str(e)}")
            raise TypeError(f"Invalid parameters for pooling layer: {str(e)}")

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> nn.Module:
        if 'type' not in config:
            raise ValueError("Missing 'type' in pooling configuration")

        pool_name = config['type'].lower()
        custom_params = config.get('params', {})

        return cls.create_pooling(pool_name, custom_params)

    @classmethod
    def _get_pooling_description(cls, pool_name: str) -> str:
        desc_key = f"pooling.{pool_name}_desc"
        return ""

    @classmethod
    def _get_pooling_properties(cls, pool_name: str) -> Dict[str, Any]:
        try:
            cfg_props = SETTINGS['pooling']['properties']
        except Exception:
            raise ValueError("pooling.properties not found in config/config.json - check your config")
        if not isinstance(cfg_props, dict) or pool_name not in cfg_props:
            raise ValueError(f"Missing properties for pooling layer '{pool_name}' in config/config.json - check pooling.properties")
        return cfg_props.get(pool_name, {})

    @classmethod
    def _get_default_params(cls) -> Dict[str, Dict[str, Any]]:
        try:
            defaults = SETTINGS['pooling']['defaults']
        except Exception:
            raise ValueError("pooling.defaults not found in config/config.json - check your config")
        if not isinstance(defaults, dict) or not defaults:
            raise ValueError("pooling.defaults not found in config.json - check config/config.json")
        return defaults


def get_pooling_for_network(pool_config: Optional[Dict[str, Any]] = None,
                          use_case: str = 'general') -> nn.Module:
    if pool_config is None:
        try:
            training_config = SETTINGS['training']
        except Exception:
            raise ValueError("Missing required 'training' section in config/config.json")
        pool_config = training_config.get('pooling', {})

    if not pool_config:
        raise ValueError("No pooling configuration found. Add 'training.pooling' to config/config.json")

    return PoolingFactory.create_from_config(pool_config)


def list_all_poolings() -> Dict[str, Any]:
    result = {}
    for name in PoolingFactory.get_available_poolings():
        result[name] = PoolingFactory.get_pooling_info(name)
    return result


def get_pooling_recommendations_for_case(use_case: str) -> Dict[str, Any]:
    return {"recommended": [], "description": "Pooling recommendations have been removed from the system"}