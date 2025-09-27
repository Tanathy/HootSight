"""Loss functions module for Hootsight.

Provides a unified interface for all PyTorch loss functions with configuration support.
Supports all major loss functions: CrossEntropyLoss, MSELoss, L1Loss, BCEWithLogitsLoss, etc.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Type, List, Optional, Union
from torch.nn import Module

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS, lang


class LossFactory:
    """Factory class for creating PyTorch loss functions with unified configuration."""

    # Registry of all available loss functions
    LOSSES: Dict[str, Type[nn.Module]] = {
        # Classification losses
        'cross_entropy': nn.CrossEntropyLoss,
        'nll_loss': nn.NLLLoss,
        'bce_loss': nn.BCELoss,
        'bce_with_logits': nn.BCEWithLogitsLoss,
        'multi_margin': nn.MultiMarginLoss,
        'multi_label_margin': nn.MultiLabelMarginLoss,
        'multi_label_soft_margin': nn.MultiLabelSoftMarginLoss,

        # Regression losses
        'mse_loss': nn.MSELoss,
        'l1_loss': nn.L1Loss,
        'smooth_l1': nn.SmoothL1Loss,
        'huber_loss': nn.HuberLoss,

        # Distribution losses
        'kl_div': nn.KLDivLoss,

        # Ranking losses
        'margin_ranking': nn.MarginRankingLoss,
        'hinge_embedding': nn.HingeEmbeddingLoss,
        'triplet_margin': nn.TripletMarginLoss,
        'cosine_embedding': nn.CosineEmbeddingLoss,

        # Other losses
        'ctc_loss': nn.CTCLoss,
        'poisson_nll': nn.PoissonNLLLoss,
        'gaussian_nll': nn.GaussianNLLLoss,
    }

    # Default parameters for each loss function (built-in fallback)
    DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
        'cross_entropy': {'reduction': 'mean'},
        'nll_loss': {'reduction': 'mean'},
        'bce_loss': {'reduction': 'mean'},
        'bce_with_logits': {'reduction': 'mean'},
        'multi_margin': {'p': 1, 'margin': 1.0, 'weight': None, 'size_average': None, 'reduce': None, 'reduction': 'mean'},
        'multi_label_margin': {'reduction': 'mean'},
        'multi_label_soft_margin': {'reduction': 'mean'},
        'mse_loss': {'reduction': 'mean'},
        'l1_loss': {'reduction': 'mean'},
        'smooth_l1': {'reduction': 'mean', 'beta': 1.0},
        'huber_loss': {'reduction': 'mean', 'delta': 1.0},
        'kl_div': {'reduction': 'batchmean'},
        'margin_ranking': {'margin': 0.0, 'reduction': 'mean'},
        'hinge_embedding': {'margin': 1.0, 'reduction': 'mean'},
        'triplet_margin': {'margin': 1.0, 'p': 2.0, 'eps': 1e-6, 'swap': False, 'reduction': 'mean'},
        'cosine_embedding': {'margin': 0.0, 'reduction': 'mean'},
        'ctc_loss': {'blank': 0, 'reduction': 'mean', 'zero_infinity': False},
        'poisson_nll': {'log_input': True, 'full': False, 'eps': 1e-8, 'reduction': 'mean'},
        'gaussian_nll': {'full': False, 'eps': 1e-8, 'reduction': 'mean'},
    }

    # Override defaults from config if provided
    _cfg_defaults = SETTINGS.get('losses', {}).get('defaults')
    if isinstance(_cfg_defaults, dict) and _cfg_defaults:
        DEFAULT_PARAMS = _cfg_defaults  # type: ignore[assignment]

    # Note: Recommendations removed from system by request.

    @classmethod
    def get_available_losses(cls) -> List[str]:
        """Get list of all available loss function names."""
        return list(cls.LOSSES.keys())

    @classmethod
    def get_loss_info(cls, loss_name: str) -> Dict[str, Any]:
        """Get information about a specific loss function.

        Args:
            loss_name: Name of the loss function

        Returns:
            Dictionary with loss function information including default parameters
        """
        if loss_name not in cls.LOSSES:
            return {}

        return {
            'name': loss_name,
            'class': cls.LOSSES[loss_name].__name__,
            'module': cls.LOSSES[loss_name].__module__,
            'default_params': cls.DEFAULT_PARAMS.get(loss_name, {}),
            'description': cls._get_loss_description(loss_name),
            'properties': cls._get_loss_properties(loss_name),
        }

    # get_recommendations removed

    @classmethod
    def create_loss(cls,
                   loss_name: str,
                   custom_params: Optional[Dict[str, Any]] = None) -> nn.Module:
        """Create a loss function instance.

        Args:
            loss_name: Name of the loss function to create
            custom_params: Custom parameters to override defaults

        Returns:
            Configured loss function instance

        Raises:
            ValueError: If loss function name is not supported
            TypeError: If parameters are invalid
        """
        if loss_name not in cls.LOSSES:
            available = ', '.join(cls.get_available_losses())
            raise ValueError(lang("losses.not_supported", loss=loss_name, available=available))

        # Get default parameters
        default_params = cls.DEFAULT_PARAMS.get(loss_name, {}).copy()

        # Override with custom parameters
        if custom_params:
            default_params.update(custom_params)

        try:
            loss_class = cls.LOSSES[loss_name]
            loss = loss_class(**default_params)

            info(lang("losses.created", loss=loss_name, params=default_params))
            return loss

        except Exception as e:
            error(lang("losses.creation_failed", loss=loss_name, error=str(e)))
            raise TypeError(lang("losses.invalid_params", loss=loss_name, error=str(e)))

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> nn.Module:
        """Create loss function from configuration dictionary.

        Args:
            config: Configuration dictionary with 'type' and 'params' keys

        Returns:
            Configured loss function instance

        Example config:
            {
                "type": "cross_entropy",
                "params": {"reduction": "mean"}
            }
        """
        if 'type' not in config:
            raise ValueError(lang("losses.config_missing_type"))

        loss_name = config['type'].lower()
        # Support flattened configs: if 'params' missing, use all top-level keys except 'type'
        if 'params' in config and isinstance(config['params'], dict):
            custom_params = config.get('params', {})
        else:
            custom_params = {k: v for k, v in config.items() if k != 'type'}
        
        return cls.create_loss(loss_name, custom_params)

    @classmethod
    def _get_loss_description(cls, loss_name: str) -> str:
        """Get description for a loss function."""
        desc_key = f"losses.{loss_name}_desc"
        return lang(desc_key)

    @classmethod
    def _get_loss_properties(cls, loss_name: str) -> Dict[str, Any]:
        """Get properties of a loss function."""
        properties = {
            'cross_entropy': {'type': 'classification', 'range': '[0, ∞)', 'differentiable': True},
            'nll_loss': {'type': 'classification', 'range': '(-∞, ∞)', 'differentiable': True},
            'bce_loss': {'type': 'binary_classification', 'range': '[0, ∞)', 'differentiable': True},
            'bce_with_logits': {'type': 'binary_classification', 'range': '[0, ∞)', 'differentiable': True},
            'mse_loss': {'type': 'regression', 'range': '[0, ∞)', 'differentiable': True},
            'l1_loss': {'type': 'regression', 'range': '[0, ∞)', 'differentiable': True},
            'smooth_l1': {'type': 'regression', 'range': '[0, ∞)', 'differentiable': True},
            'huber_loss': {'type': 'regression', 'range': '[0, ∞)', 'differentiable': True},
            'kl_div': {'type': 'distribution', 'range': '[0, ∞)', 'differentiable': True},
            'margin_ranking': {'type': 'ranking', 'range': '(-∞, ∞)', 'differentiable': True},
            'triplet_margin': {'type': 'ranking', 'range': '[0, ∞)', 'differentiable': True},
        }
        return properties.get(loss_name, {})


def get_loss_for_task(loss_config: Optional[Dict[str, Any]] = None,
                     task_type: str = 'classification') -> Optional[nn.Module]:
    """Convenience function to get a loss function for a task.

    Args:
        loss_config: Optional configuration dict, if None uses settings
        task_type: Task type for recommendations if no config provided

    Returns:
        Configured loss function instance, or None for detection
    """
    if task_type == 'detection':
        return None

    # Get config from settings if not provided
    if loss_config is None:
        training_config = SETTINGS.get('training', {})

        # Build loss config from flattened keys
        loss_type = training_config.get('loss_type', 'cross_entropy')
        loss_params_source = training_config.get('loss_params', {})
        loss_params: Dict[str, Any] = {}
        if isinstance(loss_params_source, dict):
            selected_params = loss_params_source.get(loss_type, {})
            if isinstance(selected_params, dict):
                loss_params.update(selected_params)

        reduction_override = training_config.get('loss_reduction', 'mean')
        if isinstance(reduction_override, str):
            loss_params['reduction'] = reduction_override

        loss_config = {
            'type': loss_type,
            'params': loss_params
        }

    return LossFactory.create_from_config(loss_config)


def list_all_losses() -> Dict[str, Any]:
    """Get comprehensive list of all available loss functions with details.

    Returns:
        Dictionary mapping loss names to their information
    """
    result = {}
    for name in LossFactory.get_available_losses():
        result[name] = LossFactory.get_loss_info(name)
    return result


def get_loss_recommendations_for_task(task_type: str) -> Dict[str, Any]:
    """Deprecated: loss recommendations removed. Kept for compatibility returning empty structure.

    Args:
        task_type: Type of task parameter (ignored)

    Returns:
        Empty recommendations structure
    """
    return {"recommended": [], "description": lang("losses.recommendations_removed")}