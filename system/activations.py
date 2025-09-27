"""Activations module for Hootsight.

Provides a unified interface for all PyTorch activation functions with configuration support.
Supports all major activation functions: ReLU, Sigmoid, Tanh, LeakyReLU, Softmax, etc.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Type, List, Optional, Union
from torch.nn import Module

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS, lang


class ActivationFactory:
    """Factory class for creating PyTorch activation functions with unified configuration."""
    
    # Registry of all available activation functions
    ACTIVATIONS: Dict[str, Type[nn.Module]] = {
        # Basic activations
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'prelu': nn.PReLU,
        'rrelu': nn.RReLU,
        'elu': nn.ELU,
        'celu': nn.CELU,
        'selu': nn.SELU,
        'gelu': nn.GELU,
        'silu': nn.SiLU,  # Swish
        'mish': nn.Mish,
        'hardtanh': nn.Hardtanh,
        'hardshrink': nn.Hardshrink,
        'softshrink': nn.Softshrink,
        'threshold': nn.Threshold,
        
        # Sigmoid variants
        'sigmoid': nn.Sigmoid,
        'logsigmoid': nn.LogSigmoid,
        
        # Tanh variants
        'tanh': nn.Tanh,
        'tanhshrink': nn.Tanhshrink,
        
        # Softmax variants
        'softmax': nn.Softmax,
        'logsoftmax': nn.LogSoftmax,
        'softmin': nn.Softmin,
        
        # Specialized
        'softplus': nn.Softplus,
        'softsign': nn.Softsign,
        'identity': nn.Identity,
    }
    
    # Default parameters for each activation function
    DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
        'relu': {},
        'leaky_relu': {'negative_slope': 0.01},
        'prelu': {'num_parameters': 1, 'init': 0.25},
        'rrelu': {'lower': 1./8, 'upper': 1./3},
        'elu': {'alpha': 1.0},
        'celu': {'alpha': 1.0},
        'selu': {},
        'gelu': {},
        'silu': {},
        'mish': {},
        'hardtanh': {'min_val': -1.0, 'max_val': 1.0},
        'hardshrink': {'lambd': 0.5},
        'softshrink': {'lambd': 0.5},
        'threshold': {'threshold': 0.5, 'value': 0.0},
        'sigmoid': {},
        'logsigmoid': {},
        'tanh': {},
        'tanhshrink': {},
        'softmax': {'dim': 1},
        'logsoftmax': {'dim': 1},
        'softmin': {'dim': 1},
        'softplus': {'beta': 1, 'threshold': 20},
        'softsign': {},
        'identity': {},
    }
    
    # Note: Recommendations removed from system by request.

    @classmethod
    def get_available_activations(cls) -> List[str]:
        """Get list of all available activation function names."""
        return list(cls.ACTIVATIONS.keys())

    @classmethod
    def get_activation_info(cls, activation_name: str) -> Dict[str, Any]:
        """Get information about a specific activation function.
        
        Args:
            activation_name: Name of the activation function
            
        Returns:
            Dictionary with activation function information including default parameters
        """
        if activation_name not in cls.ACTIVATIONS:
            return {}
        
        return {
            'name': activation_name,
            'class': cls.ACTIVATIONS[activation_name].__name__,
            'module': cls.ACTIVATIONS[activation_name].__module__,
            'default_params': cls.DEFAULT_PARAMS.get(activation_name, {}),
            'description': cls._get_activation_description(activation_name),
            'properties': cls._get_activation_properties(activation_name),
        }

    # get_recommendations removed

    @classmethod
    def create_activation(cls, 
                         activation_name: str, 
                         custom_params: Optional[Dict[str, Any]] = None) -> nn.Module:
        """Create an activation function instance.
        
        Args:
            activation_name: Name of the activation function to create
            custom_params: Custom parameters to override defaults
            
        Returns:
            Configured activation function instance
            
        Raises:
            ValueError: If activation function name is not supported
            TypeError: If parameters are invalid
        """
        if activation_name not in cls.ACTIVATIONS:
            available = ', '.join(cls.get_available_activations())
            raise ValueError(lang("activations.not_supported", activation=activation_name, available=available))
        
        # Get default parameters
        default_params = cls.DEFAULT_PARAMS.get(activation_name, {}).copy()
        
        # Override with custom parameters
        if custom_params:
            default_params.update(custom_params)
        
        try:
            activation_class = cls.ACTIVATIONS[activation_name]
            activation = activation_class(**default_params)
            
            info(lang("activations.created", activation=activation_name, params=default_params))
            return activation
            
        except Exception as e:
            error(lang("activations.creation_failed", activation=activation_name, error=str(e)))
            raise TypeError(lang("activations.invalid_params", activation=activation_name, error=str(e)))

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> nn.Module:
        """Create activation function from configuration dictionary.
        
        Args:
            config: Configuration dictionary with 'type' and 'params' keys
            
        Returns:
            Configured activation function instance
            
        Example config:
            {
                "type": "relu",
                "params": {}
            }
        """
        if 'type' not in config:
            raise ValueError(lang("activations.config_missing_type"))
        
        activation_name = config['type'].lower()
        custom_params = config.get('params', {})
        
        return cls.create_activation(activation_name, custom_params)

    @classmethod
    def _get_activation_description(cls, activation_name: str) -> str:
        """Get description for an activation function."""
        desc_key = f"activations.{activation_name}_desc"
        return lang(desc_key)

    @classmethod
    def _get_activation_properties(cls, activation_name: str) -> Dict[str, Any]:
        """Get properties of an activation function."""
        properties = {
            'relu': {'range': '[0, ∞)', 'differentiable': True, 'monotonic': True, 'zero_at_zero': True},
            'leaky_relu': {'range': '(-∞, ∞)', 'differentiable': True, 'monotonic': True, 'zero_at_zero': False},
            'prelu': {'range': '(-∞, ∞)', 'differentiable': True, 'monotonic': True, 'zero_at_zero': False},
            'rrelu': {'range': '(-∞, ∞)', 'differentiable': True, 'monotonic': True, 'zero_at_zero': False},
            'elu': {'range': '(-α, ∞)', 'differentiable': True, 'monotonic': True, 'zero_at_zero': False},
            'celu': {'range': '(-α, ∞)', 'differentiable': True, 'monotonic': True, 'zero_at_zero': False},
            'selu': {'range': '(-∞, ∞)', 'differentiable': True, 'monotonic': True, 'zero_at_zero': False},
            'gelu': {'range': '(-∞, ∞)', 'differentiable': True, 'monotonic': True, 'zero_at_zero': False},
            'silu': {'range': '(-∞, ∞)', 'differentiable': True, 'monotonic': True, 'zero_at_zero': False},
            'mish': {'range': '(-∞, ∞)', 'differentiable': True, 'monotonic': True, 'zero_at_zero': False},
            'hardtanh': {'range': '[-1, 1]', 'differentiable': False, 'monotonic': True, 'zero_at_zero': True},
            'sigmoid': {'range': '(0, 1)', 'differentiable': True, 'monotonic': True, 'zero_at_zero': False},
            'tanh': {'range': '(-1, 1)', 'differentiable': True, 'monotonic': True, 'zero_at_zero': True},
            'softmax': {'range': '(0, 1)', 'differentiable': True, 'monotonic': False, 'zero_at_zero': False},
        }
        return properties.get(activation_name, {})


def get_activation_for_layer(activation_config: Optional[Dict[str, Any]] = None,
                           use_case: str = 'general') -> nn.Module:
    """Convenience function to get an activation function for a layer.
    
    Args:
        activation_config: Optional configuration dict, if None uses settings
        use_case: Use case for recommendations if no config provided
        
    Returns:
        Configured activation function instance
    """
    # Get config from settings if not provided
    if activation_config is None:
        training_config = SETTINGS.get('training', {})
        activation_config = training_config.get('activation', {})
        
        # Use default if no config
        if not activation_config:
            activation_config = {
                'type': 'relu',  # Default activation
                'params': {}
            }
    
    return ActivationFactory.create_from_config(activation_config)


def list_all_activations() -> Dict[str, Any]:
    """Get comprehensive list of all available activation functions with details.
    
    Returns:
        Dictionary mapping activation names to their information
    """
    result = {}
    for name in ActivationFactory.get_available_activations():
        result[name] = ActivationFactory.get_activation_info(name)
    return result


def get_activation_recommendations_for_task(task_type: str) -> Dict[str, Any]:
    """Deprecated: activation recommendations removed. Kept for compatibility returning empty structure.
    
    Args:
        task_type: Type of task (ignored)
        
    Returns:
        Empty recommendations structure
    """
    return {"recommended": [], "description": lang("activations.recommendations_removed")}