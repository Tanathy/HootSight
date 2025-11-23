import torch
import torch.nn as nn
from typing import Dict, Any, Type, List, Optional, Union
from torch.nn import Module

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS, lang


class RegularizationFactory:
    REGULARIZATIONS: Dict[str, Type[nn.Module]] = {
        'dropout': nn.Dropout,
        'dropout2d': nn.Dropout2d,
        'dropout3d': nn.Dropout3d,
        'alphadropout': nn.AlphaDropout,
        'featurealphadropout': nn.FeatureAlphaDropout,
        'batchnorm1d': nn.BatchNorm1d,
        'batchnorm2d': nn.BatchNorm2d,
        'batchnorm3d': nn.BatchNorm3d,
        'layer_norm': nn.LayerNorm,
        'group_norm': nn.GroupNorm,
        'instance_norm1d': nn.InstanceNorm1d,
        'instance_norm2d': nn.InstanceNorm2d,
        'instance_norm3d': nn.InstanceNorm3d,
    }

    DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_available_regularizations(cls) -> List[str]:
        return list(cls.REGULARIZATIONS.keys())

    @classmethod
    def get_regularization_info(cls, reg_name: str) -> Dict[str, Any]:
        if reg_name not in cls.REGULARIZATIONS:
            return {}

        defaults = cls._get_default_params()
        if reg_name not in defaults:
            raise ValueError(lang("regularization.missing_default", reg=reg_name))
        return {
            'name': reg_name,
            'class': cls.REGULARIZATIONS[reg_name].__name__,
            'module': cls.REGULARIZATIONS[reg_name].__module__,
            'default_params': defaults.get(reg_name, {}),
            'description': cls._get_regularization_description(reg_name),
            'properties': cls._get_regularization_properties(reg_name),
        }

    @classmethod
    def create_regularization(cls,
                            reg_name: str,
                            custom_params: Optional[Dict[str, Any]] = None) -> nn.Module:
        if reg_name not in cls.REGULARIZATIONS:
            available = ', '.join(cls.get_available_regularizations())
            raise ValueError(lang("regularization.not_supported", reg=reg_name, available=available))

        defaults = cls._get_default_params()
        if reg_name not in defaults:
            raise ValueError(lang("regularization.missing_default", reg=reg_name))
        default_params = defaults.get(reg_name, {}).copy()

        if custom_params:
            default_params.update(custom_params)

        try:
            reg_class = cls.REGULARIZATIONS[reg_name]
            regularization = reg_class(**default_params)

            info(lang("regularization.created", reg=reg_name, params=default_params))
            return regularization

        except Exception as e:
            error(lang("regularization.creation_failed", reg=reg_name, error=str(e)))
            raise TypeError(lang("regularization.invalid_params", reg=reg_name, error=str(e)))

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> nn.Module:
        if 'type' not in config:
            raise ValueError(lang("regularization.config_missing_type"))

        reg_name = config['type'].lower()
        custom_params = config.get('params', {})

        return cls.create_regularization(reg_name, custom_params)

    @classmethod
    def _get_regularization_description(cls, reg_name: str) -> str:
        desc_key = f"regularization.{reg_name}_desc"
        return lang(desc_key)

    @classmethod
    def _get_regularization_properties(cls, reg_name: str) -> Dict[str, Any]:
        try:
            cfg_props = SETTINGS['regularization']['properties']
        except Exception:
            raise ValueError("regularization.properties not found in config/config.json - check your config")
        if not isinstance(cfg_props, dict) or reg_name not in cfg_props:
            raise ValueError(lang("regularization.missing_properties", reg=reg_name))
        return cfg_props.get(reg_name, {})

    @classmethod
    def _get_default_params(cls) -> Dict[str, Dict[str, Any]]:
        try:
            defaults = SETTINGS['regularization']['defaults']
        except Exception:
            raise ValueError("regularization.defaults not found in config/config.json - check your config")
        if not isinstance(defaults, dict) or not defaults:
            raise ValueError("regularization.defaults not found in config.json - check config/config.json")
        return defaults


def get_regularization_for_model(reg_config: Optional[Dict[str, Any]] = None,
                               use_case: str = 'general') -> nn.Module:
    if reg_config is None:
        try:
            training_config = SETTINGS['training']
        except Exception:
            raise ValueError("Missing required 'training' section in config/config.json")
        reg_config = training_config.get('regularization', {})

    if not reg_config:
        raise ValueError(lang("regularization.no_config_for_model"))

    return RegularizationFactory.create_from_config(reg_config)


def list_all_regularizations() -> Dict[str, Any]:
    result = {}
    for name in RegularizationFactory.get_available_regularizations():
        result[name] = RegularizationFactory.get_regularization_info(name)
    return result


def get_regularization_recommendations_for_case(use_case: str) -> Dict[str, Any]:
    return {"recommended": [], "description": lang("regularization.recommendations_removed")}