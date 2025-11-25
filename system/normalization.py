import torch
import torch.nn as nn
from typing import Dict, Any, Type, List, Optional, Union, Tuple
from torch.nn import Module

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS


class NormalizationFactory:
    NORMALIZATIONS: Dict[str, Type[nn.Module]] = {
        'batch_norm1d': nn.BatchNorm1d,
        'batch_norm2d': nn.BatchNorm2d,
        'batch_norm3d': nn.BatchNorm3d,
        'sync_batch_norm': nn.SyncBatchNorm,
        'layer_norm': nn.LayerNorm,
        'group_norm': nn.GroupNorm,
        'instance_norm1d': nn.InstanceNorm1d,
        'instance_norm2d': nn.InstanceNorm2d,
        'instance_norm3d': nn.InstanceNorm3d,
        'local_response_norm': nn.LocalResponseNorm,
        'cross_map_lr_norm': nn.CrossMapLRN2d,
    }

    DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_available_normalizations(cls) -> List[str]:
        return list(cls.NORMALIZATIONS.keys())

    @classmethod
    def get_normalization_info(cls, norm_name: str) -> Dict[str, Any]:
        if norm_name not in cls.NORMALIZATIONS:
            return {}

        defaults = cls._get_default_params()
        if norm_name not in defaults:
            raise ValueError(f"Missing default parameters for normalization '{norm_name}' in config/config.json - check normalization.defaults")
        return {
            'name': norm_name,
            'class': cls.NORMALIZATIONS[norm_name].__name__,
            'module': cls.NORMALIZATIONS[norm_name].__module__,
            'default_params': defaults.get(norm_name, {}),
            'description': cls._get_normalization_description(norm_name),
            'properties': cls._get_normalization_properties(norm_name),
        }

    @classmethod
    def create_normalization(cls,
                           norm_name: str,
                           custom_params: Optional[Dict[str, Any]] = None) -> nn.Module:
        if norm_name not in cls.NORMALIZATIONS:
            available = ', '.join(cls.get_available_normalizations())
            raise ValueError(f"Unsupported normalization: {norm_name}. Available options: {available}")

        defaults = cls._get_default_params()
        if norm_name not in defaults:
            raise ValueError(f"Missing default parameters for normalization '{norm_name}' in config/config.json - check normalization.defaults")
        default_params = defaults.get(norm_name, {}).copy()

        if custom_params:
            default_params.update(custom_params)

        if norm_name in ['batch_norm1d', 'batch_norm2d', 'batch_norm3d', 'sync_batch_norm',
                        'instance_norm1d', 'instance_norm2d', 'instance_norm3d']:
            if default_params.get('num_features') is None:
                raise ValueError(f"Missing 'num_features' parameter for {norm_name}")

        if norm_name == 'group_norm':
            if default_params.get('num_channels') is None:
                raise ValueError(f"Missing 'num_channels' parameter for {norm_name}")

        if norm_name == 'layer_norm':
            if default_params.get('normalized_shape') is None:
                raise ValueError(f"Missing 'normalized_shape' parameter for {norm_name}")

        try:
            norm_class = cls.NORMALIZATIONS[norm_name]
            normalization = norm_class(**default_params)

            info(f"Normalization layer created successfully: {norm_name} with parameters {default_params}")
            return normalization

        except Exception as e:
            error(f"Failed to create normalization layer: {str(e)}")
            raise TypeError(f"Invalid parameters for normalization layer: {str(e)}")

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> nn.Module:
        if 'type' not in config:
            raise ValueError("Missing 'type' in normalization configuration")

        norm_name = config['type'].lower()
        custom_params = config.get('params', {})

        return cls.create_normalization(norm_name, custom_params)

    @classmethod
    def _get_normalization_description(cls, norm_name: str) -> str:
        desc_key = f"normalization.{norm_name}_desc"
        return ""

    @classmethod
    def _get_normalization_properties(cls, norm_name: str) -> Dict[str, Any]:
        try:
            cfg_props = SETTINGS['normalization']['properties']
        except Exception:
            raise ValueError("normalization.properties not found in config/config.json - check your config")
        if not isinstance(cfg_props, dict) or norm_name not in cfg_props:
            raise ValueError(f"Missing properties for normalization '{norm_name}' in config/config.json - check normalization.properties")
        return cfg_props.get(norm_name, {})

    @classmethod
    def _get_default_params(cls) -> Dict[str, Dict[str, Any]]:
        try:
            defaults = SETTINGS['normalization']['defaults']
        except Exception:
            raise ValueError("normalization.defaults not found in config/config.json - check your config")
        if not isinstance(defaults, dict) or not defaults:
            raise ValueError("normalization.defaults not found in config.json - check config/config.json")
        return defaults


def get_normalization_for_network(norm_config: Optional[Dict[str, Any]] = None,
                                use_case: str = 'general') -> nn.Module:
    if norm_config is None:
        try:
            training_config = SETTINGS['training']
        except Exception:
            raise ValueError("Missing required 'training' section in config/config.json")
        norm_config = training_config.get('normalization', {})

        if not norm_config:
            raise ValueError("No normalization configuration found. Add 'training.normalization' to config/config.json")

    return NormalizationFactory.create_from_config(norm_config)


def list_all_normalizations() -> Dict[str, Any]:
    result = {}
    for name in NormalizationFactory.get_available_normalizations():
        result[name] = NormalizationFactory.get_normalization_info(name)
    return result


def get_normalization_recommendations_for_case(use_case: str) -> Dict[str, Any]:
    return {"recommended": [], "description": "Normalization recommendations have been removed from the system"}