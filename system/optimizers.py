"""Optimizers module for Hootsight.

Provides a unified interface for all PyTorch optimizers with configuration support.
Supports all major optimizers: SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, Adamax, etc.
"""
import torch
import torch.optim as optim
from typing import Dict, Any, Type, List, Optional
from torch.nn import Module

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS, lang


class OptimizerFactory:
    """Factory class for creating PyTorch optimizers with unified configuration."""
    
    # Registry of all available optimizers
    OPTIMIZERS: Dict[str, Type[optim.Optimizer]] = {
        # First-order gradient-based optimizers
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'adamax': optim.Adamax,
        'nadam': optim.NAdam,
        'radam': optim.RAdam,
        'rmsprop': optim.RMSprop,
        'rprop': optim.Rprop,
        'adagrad': optim.Adagrad,
        'adadelta': optim.Adadelta,
        'sparse_adam': optim.SparseAdam,
        
        # Second-order optimizers
        'lbfgs': optim.LBFGS,
        
        # Learning rate schedulers (technically not optimizers but often grouped)
        'asgd': optim.ASGD,  # Averaged Stochastic Gradient Descent
    }
    
    # Default parameters for each optimizer type (built-in fallback)
    DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
        'sgd': {
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'nesterov': True,
            'dampening': 0,
        },
        'adam': {
            'lr': 0.001,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0,
            'amsgrad': False,
        },
        'adamw': {
            'lr': 0.001,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.01,
            'amsgrad': False,
        },
        'adamax': {
            'lr': 0.002,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0,
        },
        'nadam': {
            'lr': 0.002,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0,
            'momentum_decay': 4e-3,
        },
        'radam': {
            'lr': 0.001,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0,
        },
        'rmsprop': {
            'lr': 0.01,
            'alpha': 0.99,
            'eps': 1e-8,
            'weight_decay': 0,
            'momentum': 0,
            'centered': False,
        },
        'rprop': {
            'lr': 0.01,
            'etas': (0.5, 1.2),
            'step_sizes': (1e-6, 50),
        },
        'adagrad': {
            'lr': 0.01,
            'lr_decay': 0,
            'weight_decay': 0,
            'initial_accumulator_value': 0,
            'eps': 1e-10,
        },
        'adadelta': {
            'lr': 1.0,
            'rho': 0.9,
            'eps': 1e-6,
            'weight_decay': 0,
        },
        'sparse_adam': {
            'lr': 0.001,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
        },
        'lbfgs': {
            'lr': 1,
            'max_iter': 20,
            'max_eval': None,
            'tolerance_grad': 1e-7,
            'tolerance_change': 1e-9,
            'history_size': 100,
            'line_search_fn': None,
        },
        'asgd': {
            'lr': 0.01,
            'lambd': 1e-4,
            'alpha': 0.75,
            't0': 1e6,
            'weight_decay': 0,
        },
    }

    # Override defaults from config if provided
    _cfg_defaults = SETTINGS.get('optimizers', {}).get('defaults')
    if isinstance(_cfg_defaults, dict) and _cfg_defaults:
        DEFAULT_PARAMS = _cfg_defaults  # type: ignore[assignment]
    
    # Note: Recommendations removed from system by request.

    @classmethod
    def get_available_optimizers(cls) -> List[str]:
        """Get list of all available optimizer names."""
        return list(cls.OPTIMIZERS.keys())

    @classmethod
    def get_optimizer_info(cls, optimizer_name: str) -> Dict[str, Any]:
        """Get information about a specific optimizer.
        
        Args:
            optimizer_name: Name of the optimizer
            
        Returns:
            Dictionary with optimizer information including default parameters
        """
        if optimizer_name not in cls.OPTIMIZERS:
            return {}
        
        return {
            'name': optimizer_name,
            'class': cls.OPTIMIZERS[optimizer_name].__name__,
            'module': cls.OPTIMIZERS[optimizer_name].__module__,
            'default_params': cls.DEFAULT_PARAMS.get(optimizer_name, {}),
            'description': cls._get_optimizer_description(optimizer_name),
        }

    # get_recommendations removed

    @classmethod
    def create_optimizer(cls, 
                        optimizer_name: str, 
                        model_parameters, 
                        custom_params: Optional[Dict[str, Any]] = None) -> optim.Optimizer:
        """Create an optimizer instance.
        
        Args:
            optimizer_name: Name of the optimizer to create
            model_parameters: Model parameters to optimize (model.parameters())
            custom_params: Custom parameters to override defaults
            
        Returns:
            Configured optimizer instance
            
        Raises:
            ValueError: If optimizer name is not supported
            TypeError: If parameters are invalid
        """
        if optimizer_name not in cls.OPTIMIZERS:
            available = ', '.join(cls.get_available_optimizers())
            raise ValueError(lang("optimizers.not_supported", optimizer=optimizer_name, available=available))
        
        # Get default parameters
        default_params = cls.DEFAULT_PARAMS.get(optimizer_name, {}).copy()
        
        # Override with custom parameters
        if custom_params:
            default_params.update(custom_params)
        
        try:
            optimizer_class = cls.OPTIMIZERS[optimizer_name]
            optimizer = optimizer_class(model_parameters, **default_params)
            
            info(lang("optimizers.created", optimizer=optimizer_name, params=default_params))
            return optimizer
            
        except Exception as e:
            error(lang("optimizers.creation_failed", optimizer=optimizer_name, error=str(e)))
            raise TypeError(lang("optimizers.invalid_params", optimizer=optimizer_name, error=str(e)))

    @classmethod
    def create_from_config(cls, 
                          model_parameters, 
                          config: Dict[str, Any]) -> optim.Optimizer:
        """Create optimizer from configuration dictionary.
        
        Args:
            model_parameters: Model parameters to optimize
            config: Configuration dictionary with 'type' and 'params' keys
            
        Returns:
            Configured optimizer instance
            
        Example config:
            {
                "type": "adam",
                "params": {
                    "lr": 0.001,
                    "weight_decay": 1e-4
                }
            }
        """
        if 'type' not in config:
            raise ValueError(lang("optimizers.config_missing_type"))
        
        optimizer_name = config['type'].lower()
        # Support flattened configs: if 'params' missing, use all top-level keys except 'type'
        if 'params' in config and isinstance(config['params'], dict):
            custom_params = config.get('params', {})
        else:
            custom_params = {k: v for k, v in config.items() if k != 'type'}
        
        return cls.create_optimizer(optimizer_name, model_parameters, custom_params)

    @classmethod
    def _get_optimizer_description(cls, optimizer_name: str) -> str:
        """Get description for an optimizer."""
        desc_key = f"optimizers.{optimizer_name}_desc"
        return lang(desc_key)


def get_optimizer_for_model(model: Module, 
                           optimizer_config: Optional[Dict[str, Any]] = None,
                           use_case: str = 'general') -> optim.Optimizer:
    """Convenience function to get an optimizer for a model.
    
    Args:
        model: PyTorch model
        optimizer_config: Optional configuration dict, if None uses settings
        use_case: Use case for recommendations if no config provided
        
    Returns:
        Configured optimizer instance
    """
    # Get config from settings if not provided
    if optimizer_config is None:
        training_config = SETTINGS.get('training', {})
        optimizer_config = training_config.get('optimizer', {})
        
        # Use default if no config
        if not optimizer_config:
            optimizer_config = {
                'type': 'adam',  # Default optimizer
                'params': {}
            }
    
    return OptimizerFactory.create_from_config(model.parameters(), optimizer_config)


def list_all_optimizers() -> Dict[str, Any]:
    """Get comprehensive list of all available optimizers with details.
    
    Returns:
        Dictionary mapping optimizer names to their information
    """
    result = {}
    for name in OptimizerFactory.get_available_optimizers():
        result[name] = OptimizerFactory.get_optimizer_info(name)
    return result


def get_optimizer_recommendations_for_task(task_type: str) -> Dict[str, Any]:
    """Deprecated: optimizer recommendations removed. Kept for compatibility returning empty structure.
    
    Args:
        task_type: Type of task (ignored)
        
    Returns:
        Empty recommendations structure
    """
    return {"recommended": [], "description": lang("optimizers.recommendations_removed")}