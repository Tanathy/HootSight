import torch
import torch.nn as nn
from typing import Dict, Any, Callable, List, Optional, Union
from torch.nn import Module

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS


class WeightInitFactory:

    INITIALIZERS: Dict[str, Callable] = {
        'xavier_uniform': nn.init.xavier_uniform_,
        'xavier_normal': nn.init.xavier_normal_,
        'kaiming_uniform': nn.init.kaiming_uniform_,
        'kaiming_normal': nn.init.kaiming_normal_,
        'uniform': nn.init.uniform_,
        'normal': nn.init.normal_,
        'trunc_normal': nn.init.trunc_normal_,
        'constant': nn.init.constant_,
        'zeros': nn.init.zeros_,
        'ones': nn.init.ones_,
        'orthogonal': nn.init.orthogonal_,
        'sparse': nn.init.sparse_,
        'eye': nn.init.eye_,
        'dirac': nn.init.dirac_,
    }

    @classmethod
    def _get_default_params(cls) -> Dict[str, Dict[str, Any]]:
        try:
            training_config = SETTINGS['training']
        except Exception:
            raise ValueError("Missing required 'training' section in config/config.json")
        weight_init_config = training_config.get('weight_init', {})
        
        if isinstance(weight_init_config, dict) and weight_init_config.get('defaults'):
            return weight_init_config.get('defaults', {})
        raise ValueError("training.weight_init.defaults not found in config/config.json - check training.weight_init")

    @classmethod
    def get_available_initializers(cls) -> List[str]:
        return list(cls.INITIALIZERS.keys())

    @classmethod
    def get_initializer_info(cls, init_name: str) -> Dict[str, Any]:
        if init_name not in cls.INITIALIZERS:
            return {}

        return {
            'name': init_name,
            'function': cls.INITIALIZERS[init_name].__name__,
            'module': cls.INITIALIZERS[init_name].__module__,
            'default_params': cls._get_default_params().get(init_name, {}),
            'description': cls._get_initializer_description(init_name),
            'properties': cls._get_initializer_properties(init_name),
        }

    @classmethod
    def apply_initialization(cls,
                           module: nn.Module,
                           init_name: str,
                           custom_params: Optional[Dict[str, Any]] = None) -> None:
        if init_name not in cls.INITIALIZERS:
            available = ', '.join(cls.get_available_initializers())
            raise ValueError(f"Unsupported weight initialization: {init_name}. Available options: {available}")

        default_params = cls._get_default_params().get(init_name, {}).copy()
        
        if custom_params:
            default_params.update(custom_params)
        
        try:
            init_func = cls.INITIALIZERS[init_name]

            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                init_func(module.weight, **default_params)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.Linear):
                init_func(module.weight, **default_params)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, (nn.LSTM, nn.GRU)):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        init_func(param, **default_params)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

            else:
                for param in module.parameters():
                    if param.dim() > 1:
                        init_func(param, **default_params)
                    else:
                        nn.init.constant_(param, 0)

        except Exception as e:
            error(f"Failed to apply weight initialization: {str(e)}")
            raise TypeError(f"Invalid parameters for weight initialization: {str(e)}")

    @classmethod
    def apply_to_model(cls,
                      model: nn.Module,
                      init_config: Optional[Dict[str, Any]] = None,
                      layer_type: str = 'default') -> None:
        if init_config is None:
            try:
                training_config = SETTINGS['training']
            except Exception:
                raise ValueError("Missing required 'training' section in config/config.json")
            init_config = training_config.get('weight_init')
            if not init_config:
                raise ValueError("No weight_init configuration found. Add 'training.weight_init' to config/config.json")

        if 'type' not in init_config or not init_config['type']:
            raise ValueError("weight_init.type missing. Add 'training.weight_init.type' to config/config.json")
        init_name = init_config['type']
        custom_params = init_config.get('params', {})

        initialized_counts = {}
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear, nn.LSTM, nn.GRU)):
                cls.apply_initialization(module, init_name, custom_params)
                layer_type = type(module).__name__
                initialized_counts[layer_type] = initialized_counts.get(layer_type, 0) + 1
        
        # Log summary once
        if initialized_counts:
            summary = ', '.join(f"{count} {ltype}" for ltype, count in initialized_counts.items())
            info(f"Weight initialization '{init_name}' applied to {summary}")

    @classmethod
    def _get_initializer_description(cls, init_name: str) -> str:
        desc_key = f"weight_init.{init_name}_desc"
        return ""

    @classmethod
    def _get_initializer_properties(cls, init_name: str) -> Dict[str, Any]:
        try:
            training_config = SETTINGS['training']
        except Exception:
            raise ValueError("Missing required 'training' section in config/config.json")
        weight_init_config = training_config.get('weight_init', {})
        if not isinstance(weight_init_config, dict) or 'properties' not in weight_init_config:
            raise ValueError("training.weight_init.properties not found in config/config.json - check training.weight_init")
        cfg_props = weight_init_config.get('properties', {})
        if not isinstance(cfg_props, dict) or init_name not in cfg_props:
            raise ValueError(f"Missing properties for weight initialization '{init_name}' in config/config.json - check training.weight_init.properties")
        return cfg_props.get(init_name, {})


def initialize_model_weights(model: nn.Module,
                           init_config: Optional[Dict[str, Any]] = None,
                           layer_type: str = 'default') -> None:
    WeightInitFactory.apply_to_model(model, init_config, layer_type)


def list_all_initializers() -> Dict[str, Any]:
    result = {}
    for name in WeightInitFactory.get_available_initializers():
        result[name] = WeightInitFactory.get_initializer_info(name)
    return result


def get_initializer_recommendations_for_layer(layer_type: str) -> Dict[str, Any]:
    return {"recommended": [], "description": "Weight initialization recommendations have been removed from the system"}