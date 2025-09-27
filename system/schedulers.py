"""Learning rate schedulers module for Hootsight.

Provides a unified interface for all PyTorch learning rate schedulers with configuration support.
Supports all major schedulers: StepLR, ReduceLROnPlateau, CosineAnnealingLR, etc.
"""
import torch
from torch.optim import lr_scheduler
from typing import Dict, Any, Type, List, Optional, Union
from torch.optim import Optimizer

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS, lang


class LRSchedulerFactory:
    """Factory class for creating PyTorch learning rate schedulers with unified configuration."""

    # Registry of all available learning rate schedulers
    SCHEDULERS: Dict[str, Type] = {
        # Step-based schedulers
        'step_lr': lr_scheduler.StepLR,
        'multi_step_lr': lr_scheduler.MultiStepLR,
        'exponential_lr': lr_scheduler.ExponentialLR,

        # Cosine-based schedulers
        'cosine_annealing_lr': lr_scheduler.CosineAnnealingLR,
        'cosine_annealing_warm_restarts': lr_scheduler.CosineAnnealingWarmRestarts,

        # Plateau-based schedulers
        'reduce_lr_on_plateau': lr_scheduler.ReduceLROnPlateau,

        # Cyclic schedulers
        'cyclic_lr': lr_scheduler.CyclicLR,
        'one_cycle_lr': lr_scheduler.OneCycleLR,

        # Polynomial schedulers
        'polynomial_lr': lr_scheduler.PolynomialLR,

        # Linear schedulers
        'linear_lr': lr_scheduler.LinearLR,

        # Lambda-based schedulers
        'lambda_lr': lr_scheduler.LambdaLR,
        'multiplicative_lr': lr_scheduler.MultiplicativeLR,
    }

    # Default parameters for each scheduler (built-in fallback)
    DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
        'step_lr': {'step_size': 30, 'gamma': 0.1},
        'multi_step_lr': {'milestones': [30, 60, 90], 'gamma': 0.1},
        'exponential_lr': {'gamma': 0.95},
        'cosine_annealing_lr': {'T_max': 100, 'eta_min': 0},
        'cosine_annealing_warm_restarts': {'T_0': 100, 'T_mult': 2, 'eta_min': 0},
        'reduce_lr_on_plateau': {'mode': 'min', 'factor': 0.1, 'patience': 10, 'threshold': 1e-4, 'threshold_mode': 'rel', 'cooldown': 0, 'min_lr': 0, 'eps': 1e-8},
        'cyclic_lr': {'base_lr': 0.001, 'max_lr': 0.01, 'step_size_up': 2000, 'step_size_down': None, 'mode': 'triangular', 'gamma': 1.0, 'scale_fn': None, 'scale_mode': 'cycle', 'cycle_momentum': True, 'base_momentum': 0.8, 'max_momentum': 0.9},
        'one_cycle_lr': {'max_lr': 0.01, 'total_steps': None, 'epochs': None, 'steps_per_epoch': None, 'pct_start': 0.3, 'anneal_strategy': 'cos', 'cycle_momentum': True, 'base_momentum': 0.85, 'max_momentum': 0.95, 'div_factor': 25.0, 'final_div_factor': 10000.0, 'three_phase': False},
        'polynomial_lr': {'total_iters': 100, 'power': 1.0},
        'linear_lr': {'start_factor': 1.0/3, 'end_factor': 1.0, 'total_iters': 100},
        'lambda_lr': {'lr_lambda': lambda epoch: 0.95 ** epoch},
        'multiplicative_lr': {'lr_lambda': lambda epoch: 0.95},
    }

    # Override defaults from config if provided
    _cfg_defaults = SETTINGS.get('schedulers', {}).get('defaults')
    if isinstance(_cfg_defaults, dict) and _cfg_defaults:
        DEFAULT_PARAMS = _cfg_defaults  # type: ignore[assignment]

    # Note: Recommendations removed from system by request.

    @classmethod
    def get_available_schedulers(cls) -> List[str]:
        """Get list of all available learning rate scheduler names."""
        return list(cls.SCHEDULERS.keys())

    @classmethod
    def get_scheduler_info(cls, scheduler_name: str) -> Dict[str, Any]:
        """Get information about a specific learning rate scheduler.

        Args:
            scheduler_name: Name of the scheduler

        Returns:
            Dictionary with scheduler information including default parameters
        """
        if scheduler_name not in cls.SCHEDULERS:
            return {}

        return {
            'name': scheduler_name,
            'class': cls.SCHEDULERS[scheduler_name].__name__,
            'module': cls.SCHEDULERS[scheduler_name].__module__,
            'default_params': cls.DEFAULT_PARAMS.get(scheduler_name, {}),
            'description': cls._get_scheduler_description(scheduler_name),
            'properties': cls._get_scheduler_properties(scheduler_name),
        }

    # get_recommendations removed

    @classmethod
    def create_scheduler(cls,
                        scheduler_name: str,
                        optimizer: Optimizer,
                        custom_params: Optional[Dict[str, Any]] = None):
        """Create a learning rate scheduler instance.

        Args:
            scheduler_name: Name of the scheduler to create
            optimizer: PyTorch optimizer instance
            custom_params: Custom parameters to override defaults

        Returns:
            Configured scheduler instance

        Raises:
            ValueError: If scheduler name is not supported
            TypeError: If parameters are invalid
        """
        if scheduler_name not in cls.SCHEDULERS:
            available = ', '.join(cls.get_available_schedulers())
            raise ValueError(lang("schedulers.not_supported", scheduler=scheduler_name, available=available))

        # Get default parameters
        default_params = cls.DEFAULT_PARAMS.get(scheduler_name, {}).copy()

        # Override with custom parameters
        if custom_params:
            default_params.update(custom_params)

        try:
            scheduler_class = cls.SCHEDULERS[scheduler_name]
            scheduler = scheduler_class(optimizer, **default_params)

            info(lang("schedulers.created", scheduler=scheduler_name, params=default_params))
            return scheduler

        except Exception as e:
            error(lang("schedulers.creation_failed", scheduler=scheduler_name, error=str(e)))
            raise TypeError(lang("schedulers.invalid_params", scheduler=scheduler_name, error=str(e)))

    @classmethod
    def create_from_config(cls,
                          optimizer: Optimizer,
                          config: Dict[str, Any]):
        """Create scheduler from configuration dictionary.

        Args:
            optimizer: PyTorch optimizer instance
            config: Configuration dictionary with 'type' and 'params' keys

        Returns:
            Configured scheduler instance

        Example config:
            {
                "type": "step_lr",
                "params": {"step_size": 30, "gamma": 0.1}
            }
        """
        if 'type' not in config:
            raise ValueError(lang("schedulers.config_missing_type"))

        scheduler_name = config['type'].lower()
        # Support flattened configs: if 'params' missing, use all top-level keys except 'type'
        if 'params' in config and isinstance(config['params'], dict):
            custom_params = config.get('params', {})
        else:
            custom_params = {k: v for k, v in config.items() if k != 'type'}

        return cls.create_scheduler(scheduler_name, optimizer, custom_params)

    @classmethod
    def _get_scheduler_description(cls, scheduler_name: str) -> str:
        """Get description for a scheduler."""
        desc_key = f"schedulers.{scheduler_name}_desc"
        return lang(desc_key)

    @classmethod
    def _get_scheduler_properties(cls, scheduler_name: str) -> Dict[str, Any]:
        """Get properties of a scheduler."""
        properties = {
            'step_lr': {'type': 'step_based', 'adaptive': False, 'requires_metric': False},
            'multi_step_lr': {'type': 'step_based', 'adaptive': False, 'requires_metric': False},
            'exponential_lr': {'type': 'exponential_decay', 'adaptive': False, 'requires_metric': False},
            'cosine_annealing_lr': {'type': 'cosine', 'adaptive': False, 'requires_metric': False},
            'cosine_annealing_warm_restarts': {'type': 'cosine', 'adaptive': False, 'requires_metric': False},
            'reduce_lr_on_plateau': {'type': 'plateau_based', 'adaptive': True, 'requires_metric': True},
            'cyclic_lr': {'type': 'cyclic', 'adaptive': False, 'requires_metric': False},
            'one_cycle_lr': {'type': 'cyclic', 'adaptive': False, 'requires_metric': False},
            'polynomial_lr': {'type': 'polynomial', 'adaptive': False, 'requires_metric': False},
            'linear_lr': {'type': 'linear', 'adaptive': False, 'requires_metric': False},
            'lambda_lr': {'type': 'custom_function', 'adaptive': False, 'requires_metric': False},
            'multiplicative_lr': {'type': 'custom_function', 'adaptive': False, 'requires_metric': False},
        }
        return properties.get(scheduler_name, {})


def get_scheduler_for_training(optimizer: Optimizer,
                              scheduler_config: Optional[Dict[str, Any]] = None,
                              scenario: str = 'standard_training') -> lr_scheduler._LRScheduler:
    """Convenience function to get a scheduler for training.

    Args:
        optimizer: PyTorch optimizer instance
        scheduler_config: Optional configuration dict, if None uses settings
        scenario: Training scenario for recommendations if no config provided

    Returns:
        Configured scheduler instance
    """
    # Get config from settings if not provided
    if scheduler_config is None:
        training_config = SETTINGS.get('training', {})
        scheduler_config = training_config.get('scheduler', {})

        # Use default if no config
        if not scheduler_config:
            scheduler_config = {
                'type': 'step_lr',  # Default scheduler
                'params': {}
            }

    return LRSchedulerFactory.create_from_config(optimizer, scheduler_config)


def list_all_schedulers() -> Dict[str, Any]:
    """Get comprehensive list of all available schedulers with details.

    Returns:
        Dictionary mapping scheduler names to their information
    """
    result = {}
    for name in LRSchedulerFactory.get_available_schedulers():
        result[name] = LRSchedulerFactory.get_scheduler_info(name)
    return result


def get_scheduler_recommendations_for_scenario(scenario: str) -> Dict[str, Any]:
    """Deprecated: scheduler recommendations removed. Kept for compatibility returning empty structure.

    Args:
        scenario: Training scenario parameter (ignored)

    Returns:
        Empty recommendations structure
    """
    return {"recommended": [], "description": lang("schedulers.recommendations_removed")}