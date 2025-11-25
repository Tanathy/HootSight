import torch
import torch.nn as nn
from typing import Dict, Any, Type, List, Optional, Union
from torch.nn import Module

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS


class ActivationFactory:
    ACTIVATIONS: Dict[str, Type[nn.Module]] = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'prelu': nn.PReLU,
        'rrelu': nn.RReLU,
        'elu': nn.ELU,
        'celu': nn.CELU,
        'selu': nn.SELU,
        'gelu': nn.GELU,
        'silu': nn.SiLU,
        'mish': nn.Mish,
        'hardtanh': nn.Hardtanh,
        'hardshrink': nn.Hardshrink,
        'softshrink': nn.Softshrink,
        'threshold': nn.Threshold,
        'sigmoid': nn.Sigmoid,
        'logsigmoid': nn.LogSigmoid,
        'tanh': nn.Tanh,
        'tanhshrink': nn.Tanhshrink,
        'softmax': nn.Softmax,
        'logsoftmax': nn.LogSoftmax,
        'softmin': nn.Softmin,
        'softplus': nn.Softplus,
        'softsign': nn.Softsign,
        'identity': nn.Identity,
    }
    
    @classmethod
    def _get_default_params(cls) -> Dict[str, Dict[str, Any]]:
        try:
            cfg = SETTINGS['activations']['defaults']
        except Exception:
            raise ValueError("activations.defaults not found in config/config.json - check config/config.json")
        if not isinstance(cfg, dict):
            raise ValueError("activations.defaults in config/config.json must be an object/dict")
        return cfg

    DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_available_activations(cls) -> List[str]:
        return list(cls.ACTIVATIONS.keys())

    @classmethod
    def get_activation_info(cls, activation_name: str) -> Dict[str, Any]:
        if activation_name not in cls.ACTIVATIONS:
            return {}
        
        return {
            'name': activation_name,
            'class': cls.ACTIVATIONS[activation_name].__name__,
            'module': cls.ACTIVATIONS[activation_name].__module__,
            'default_params': cls._get_default_params().get(activation_name, {}),
            'description': cls._get_activation_description(activation_name),
            'properties': cls._get_activation_properties(activation_name),
        }

    @classmethod
    def create_activation(cls, 
                         activation_name: str, 
                         custom_params: Optional[Dict[str, Any]] = None) -> nn.Module:
        if activation_name not in cls.ACTIVATIONS:
            available = ', '.join(cls.get_available_activations())
            raise ValueError(f"Unsupported activation: {activation_name}. Available options: {available}")
        default_params = cls._get_default_params().get(activation_name, {}).copy()

        
        if custom_params:
            default_params.update(custom_params)
        
        try:
            activation_class = cls.ACTIVATIONS[activation_name]
            activation = activation_class(**default_params)
            
            info(f"Activation created successfully: {activation_name} with parameters {default_params}")
            return activation
            
        except Exception as e:
            error(f"Failed to create activation: {str(e)}")
            raise TypeError(f"Invalid parameters for activation: {str(e)}")

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> nn.Module:
        if 'type' not in config:
            raise ValueError("Missing 'type' in activation configuration")
        
        activation_name = config['type'].lower()
        custom_params = config.get('params', {})
        
        return cls.create_activation(activation_name, custom_params)

    @classmethod
    def _get_activation_description(cls, activation_name: str) -> str:
        desc_key = f"activations.{activation_name}_desc"
        return ""

    @classmethod
    def _get_activation_properties(cls, activation_name: str) -> Dict[str, Any]:
        try:
            cfg_props = SETTINGS['activations']['properties']
        except Exception:
            raise ValueError("activations.properties not found in config/config.json - check your config file")
        if not isinstance(cfg_props, dict) or activation_name not in cfg_props:
            raise ValueError(f"Activation properties for '{activation_name}' must be defined in config.config.json under 'activations.properties'.")
        return cfg_props[activation_name]


def get_activation_for_layer(activation_config: Optional[Dict[str, Any]] = None,
                           use_case: str = 'general') -> nn.Module:
    if activation_config is None:
        try:
            training_config = SETTINGS['training']
        except Exception:
            raise ValueError("Missing required 'training' section in config/config.json")
        activation_config = training_config.get('activation')
        if not activation_config:
            raise ValueError("No activation configuration found. Add 'training.activation' to config/config.json")
    
    return ActivationFactory.create_from_config(activation_config)


def list_all_activations() -> Dict[str, Any]:
    result = {}
    for name in ActivationFactory.get_available_activations():
        result[name] = ActivationFactory.get_activation_info(name)
    return result


def get_activation_recommendations_for_task(task_type: str) -> Dict[str, Any]:
    return {"recommended": [], "description": "Activation recommendations removed"}