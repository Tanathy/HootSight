"""Weight initialization module for Hootsight.

Provides a unified interface for all PyTorch weight initialization methods with configuration support.
Supports all major initialization methods: Xavier, He, uniform, normal, etc.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Callable, List, Optional, Union
from torch.nn import Module

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS, lang


class WeightInitFactory:
    """Factory class for creating PyTorch weight initialization functions with unified configuration."""

    # Registry of all available weight initialization methods
    INITIALIZERS: Dict[str, Callable] = {
        # Xavier/Glorot initialization
        'xavier_uniform': nn.init.xavier_uniform_,
        'xavier_normal': nn.init.xavier_normal_,

        # He/Kaiming initialization
        'kaiming_uniform': nn.init.kaiming_uniform_,
        'kaiming_normal': nn.init.kaiming_normal_,

        # Uniform and normal distributions
        'uniform': nn.init.uniform_,
        'normal': nn.init.normal_,
        'trunc_normal': nn.init.trunc_normal_,

        # Constant and zeros
        'constant': nn.init.constant_,
        'zeros': nn.init.zeros_,
        'ones': nn.init.ones_,

        # Orthogonal and sparse
        'orthogonal': nn.init.orthogonal_,
        'sparse': nn.init.sparse_,

        # Special initializations
        'eye': nn.init.eye_,
        'dirac': nn.init.dirac_,
    }

    # Default parameters for each initialization method
    DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
        'xavier_uniform': {'gain': 1.0},
        'xavier_normal': {'gain': 1.0},
        'kaiming_uniform': {'a': 0, 'mode': 'fan_in', 'nonlinearity': 'leaky_relu'},
        'kaiming_normal': {'a': 0, 'mode': 'fan_in', 'nonlinearity': 'leaky_relu'},
        'uniform': {'a': -0.1, 'b': 0.1},
        'normal': {'mean': 0.0, 'std': 0.02},
        'trunc_normal': {'mean': 0.0, 'std': 0.02, 'a': -2.0, 'b': 2.0},
        'constant': {'val': 0.0},
        'zeros': {},
        'ones': {},
        'orthogonal': {'gain': 1.0},
        'sparse': {'sparsity': 0.1, 'std': 0.01},
        'eye': {},
        'dirac': {},
    }

    # Note: Recommendations removed from system by request.

    @classmethod
    def get_available_initializers(cls) -> List[str]:
        """Get list of all available weight initialization methods."""
        return list(cls.INITIALIZERS.keys())

    @classmethod
    def get_initializer_info(cls, init_name: str) -> Dict[str, Any]:
        """Get information about a specific weight initialization method.

        Args:
            init_name: Name of the initialization method

        Returns:
            Dictionary with initialization method information including default parameters
        """
        if init_name not in cls.INITIALIZERS:
            return {}

        return {
            'name': init_name,
            'function': cls.INITIALIZERS[init_name].__name__,
            'module': cls.INITIALIZERS[init_name].__module__,
            'default_params': cls.DEFAULT_PARAMS.get(init_name, {}),
            'description': cls._get_initializer_description(init_name),
            'properties': cls._get_initializer_properties(init_name),
        }

    # get_recommendations removed

    @classmethod
    def apply_initialization(cls,
                           module: nn.Module,
                           init_name: str,
                           custom_params: Optional[Dict[str, Any]] = None) -> None:
        """Apply weight initialization to a module.

        Args:
            module: PyTorch module to initialize
            init_name: Name of the initialization method
            custom_params: Custom parameters to override defaults

        Raises:
            ValueError: If initialization method is not supported
            TypeError: If parameters are invalid
        """
        if init_name not in cls.INITIALIZERS:
            available = ', '.join(cls.get_available_initializers())
            raise ValueError(lang("weight_init.not_supported", init=init_name, available=available))

        # Get default parameters
        default_params = cls.DEFAULT_PARAMS.get(init_name, {}).copy()

        # Override with custom parameters
        if custom_params:
            default_params.update(custom_params)

        try:
            init_func = cls.INITIALIZERS[init_name]

            # Apply initialization based on module type
            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                # Convolutional layers
                init_func(module.weight, **default_params)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.Linear):
                # Linear layers
                init_func(module.weight, **default_params)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, (nn.LSTM, nn.GRU)):
                # Recurrent layers
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        init_func(param, **default_params)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

            else:
                # Generic initialization for other layers
                for param in module.parameters():
                    if param.dim() > 1:  # Weight parameters
                        init_func(param, **default_params)
                    else:  # Bias parameters
                        nn.init.constant_(param, 0)

            info(lang("weight_init.applied", init=init_name, module=type(module).__name__, params=default_params))

        except Exception as e:
            error(lang("weight_init.application_failed", init=init_name, error=str(e)))
            raise TypeError(lang("weight_init.invalid_params", init=init_name, error=str(e)))

    @classmethod
    def apply_to_model(cls,
                      model: nn.Module,
                      init_config: Optional[Dict[str, Any]] = None,
                      layer_type: str = 'default') -> None:
        """Apply weight initialization to an entire model.

        Args:
            model: PyTorch model to initialize
            init_config: Optional configuration dict, if None uses settings
            layer_type: Layer type parameter (deprecated, kept for compatibility)
        """
        # Get config from settings if not provided
        if init_config is None:
            training_config = SETTINGS.get('training', {})
            init_config = training_config.get('weight_init', {})

            # Use default if no config
            if not init_config:
                init_config = {
                    'type': 'kaiming_normal',  # Default initialization
                    'params': {}
                }

        init_name = init_config.get('type', 'kaiming_normal')
        custom_params = init_config.get('params', {})

        # Apply initialization to all submodules
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear, nn.LSTM, nn.GRU)):
                cls.apply_initialization(module, init_name, custom_params)

    @classmethod
    def _get_initializer_description(cls, init_name: str) -> str:
        """Get description for a weight initialization method."""
        desc_key = f"weight_init.{init_name}_desc"
        return lang(desc_key)

    @classmethod
    def _get_initializer_properties(cls, init_name: str) -> Dict[str, Any]:
        """Get properties of a weight initialization method."""
        properties = {
            'xavier_uniform': {'distribution': 'uniform', 'scale': 'xavier', 'bias_init': 'zeros'},
            'xavier_normal': {'distribution': 'normal', 'scale': 'xavier', 'bias_init': 'zeros'},
            'kaiming_uniform': {'distribution': 'uniform', 'scale': 'he', 'bias_init': 'zeros'},
            'kaiming_normal': {'distribution': 'normal', 'scale': 'he', 'bias_init': 'zeros'},
            'uniform': {'distribution': 'uniform', 'scale': 'custom', 'bias_init': 'zeros'},
            'normal': {'distribution': 'normal', 'scale': 'custom', 'bias_init': 'zeros'},
            'trunc_normal': {'distribution': 'truncated_normal', 'scale': 'custom', 'bias_init': 'zeros'},
            'constant': {'distribution': 'constant', 'scale': 'none', 'bias_init': 'none'},
            'zeros': {'distribution': 'zeros', 'scale': 'none', 'bias_init': 'none'},
            'ones': {'distribution': 'ones', 'scale': 'none', 'bias_init': 'none'},
            'orthogonal': {'distribution': 'orthogonal', 'scale': 'custom', 'bias_init': 'zeros'},
            'sparse': {'distribution': 'sparse', 'scale': 'custom', 'bias_init': 'zeros'},
            'eye': {'distribution': 'identity', 'scale': 'none', 'bias_init': 'none'},
            'dirac': {'distribution': 'dirac', 'scale': 'none', 'bias_init': 'none'},
        }
        return properties.get(init_name, {})


def initialize_model_weights(model: nn.Module,
                           init_config: Optional[Dict[str, Any]] = None,
                           layer_type: str = 'default') -> None:
    """Convenience function to initialize model weights.

    Args:
        model: PyTorch model to initialize
        init_config: Optional configuration dict
        layer_type: Layer type for recommendations
    """
    WeightInitFactory.apply_to_model(model, init_config, layer_type)


def list_all_initializers() -> Dict[str, Any]:
    """Get comprehensive list of all available weight initialization methods with details.

    Returns:
        Dictionary mapping initializer names to their information
    """
    result = {}
    for name in WeightInitFactory.get_available_initializers():
        result[name] = WeightInitFactory.get_initializer_info(name)
    return result


def get_initializer_recommendations_for_layer(layer_type: str) -> Dict[str, Any]:
    """Deprecated: weight initialization recommendations removed. Kept for compatibility returning empty structure.

    Args:
        layer_type: Type of layer parameter (ignored)

    Returns:
        Empty recommendations structure
    """
    return {"recommended": [], "description": lang("weight_init.recommendations_removed")}