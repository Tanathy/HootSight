"""Regularization module for Hootsight.

Provides a unified interface for all PyTorch regularization techniques with configuration support.
Supports Dropout, L1/L2 regularization, and other regularization methods.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Type, List, Optional, Union
from torch.nn import Module

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS, lang


class RegularizationFactory:
    """Factory class for creating PyTorch regularization techniques with unified configuration."""

    # Registry of all available regularization techniques
    REGULARIZATIONS: Dict[str, Type[nn.Module]] = {
        # Dropout variants
        'dropout': nn.Dropout,
        'dropout2d': nn.Dropout2d,
        'dropout3d': nn.Dropout3d,
        'alphadropout': nn.AlphaDropout,
        'featurealphadropout': nn.FeatureAlphaDropout,

        # Batch normalization (also acts as regularization)
        'batchnorm1d': nn.BatchNorm1d,
        'batchnorm2d': nn.BatchNorm2d,
        'batchnorm3d': nn.BatchNorm3d,

        # Other regularization layers
        'layer_norm': nn.LayerNorm,
        'group_norm': nn.GroupNorm,
        'instance_norm1d': nn.InstanceNorm1d,
        'instance_norm2d': nn.InstanceNorm2d,
        'instance_norm3d': nn.InstanceNorm3d,
    }

    # Default parameters for each regularization technique
    DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
        'dropout': {'p': 0.5, 'inplace': False},
        'dropout2d': {'p': 0.5, 'inplace': False},
        'dropout3d': {'p': 0.5, 'inplace': False},
        'alphadropout': {'p': 0.5, 'inplace': False},
        'featurealphadropout': {'p': 0.5, 'inplace': False},
        'batchnorm1d': {'eps': 1e-5, 'momentum': 0.1, 'affine': True, 'track_running_stats': True},
        'batchnorm2d': {'eps': 1e-5, 'momentum': 0.1, 'affine': True, 'track_running_stats': True},
        'batchnorm3d': {'eps': 1e-5, 'momentum': 0.1, 'affine': True, 'track_running_stats': True},
        'layer_norm': {'eps': 1e-5, 'elementwise_affine': True},
        'group_norm': {'num_groups': 32, 'eps': 1e-5, 'affine': True},
        'instance_norm1d': {'eps': 1e-5, 'momentum': 0.1, 'affine': False, 'track_running_stats': False},
        'instance_norm2d': {'eps': 1e-5, 'momentum': 0.1, 'affine': False, 'track_running_stats': False},
        'instance_norm3d': {'eps': 1e-5, 'momentum': 0.1, 'affine': False, 'track_running_stats': False},
    }

    # Note: Recommendations removed from system by request.

    @classmethod
    def get_available_regularizations(cls) -> List[str]:
        """Get list of all available regularization techniques."""
        return list(cls.REGULARIZATIONS.keys())

    @classmethod
    def get_regularization_info(cls, reg_name: str) -> Dict[str, Any]:
        """Get information about a specific regularization technique.

        Args:
            reg_name: Name of the regularization technique

        Returns:
            Dictionary with regularization technique information including default parameters
        """
        if reg_name not in cls.REGULARIZATIONS:
            return {}

        return {
            'name': reg_name,
            'class': cls.REGULARIZATIONS[reg_name].__name__,
            'module': cls.REGULARIZATIONS[reg_name].__module__,
            'default_params': cls.DEFAULT_PARAMS.get(reg_name, {}),
            'description': cls._get_regularization_description(reg_name),
            'properties': cls._get_regularization_properties(reg_name),
        }

    # get_recommendations removed

    @classmethod
    def create_regularization(cls,
                            reg_name: str,
                            custom_params: Optional[Dict[str, Any]] = None) -> nn.Module:
        """Create a regularization layer instance.

        Args:
            reg_name: Name of the regularization technique
            custom_params: Custom parameters to override defaults

        Returns:
            Configured regularization layer instance

        Raises:
            ValueError: If regularization technique is not supported
            TypeError: If parameters are invalid
        """
        if reg_name not in cls.REGULARIZATIONS:
            available = ', '.join(cls.get_available_regularizations())
            raise ValueError(lang("regularization.not_supported", reg=reg_name, available=available))

        # Get default parameters
        default_params = cls.DEFAULT_PARAMS.get(reg_name, {}).copy()

        # Override with custom parameters
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
        """Create regularization layer from configuration dictionary.

        Args:
            config: Configuration dictionary with 'type' and 'params' keys

        Returns:
            Configured regularization layer instance

        Example config:
            {
                "type": "dropout",
                "params": {"p": 0.3}
            }
        """
        if 'type' not in config:
            raise ValueError(lang("regularization.config_missing_type"))

        reg_name = config['type'].lower()
        custom_params = config.get('params', {})

        return cls.create_regularization(reg_name, custom_params)

    @classmethod
    def _get_regularization_description(cls, reg_name: str) -> str:
        """Get description for a regularization technique."""
        desc_key = f"regularization.{reg_name}_desc"
        return lang(desc_key)

    @classmethod
    def _get_regularization_properties(cls, reg_name: str) -> Dict[str, Any]:
        """Get properties of a regularization technique."""
        properties = {
            'dropout': {'type': 'dropout', 'train_only': True, 'normalization': False},
            'dropout2d': {'type': 'dropout', 'train_only': True, 'normalization': False},
            'dropout3d': {'type': 'dropout', 'train_only': True, 'normalization': False},
            'alphadropout': {'type': 'dropout', 'train_only': True, 'normalization': False},
            'featurealphadropout': {'type': 'dropout', 'train_only': True, 'normalization': False},
            'batchnorm1d': {'type': 'normalization', 'train_only': False, 'normalization': True},
            'batchnorm2d': {'type': 'normalization', 'train_only': False, 'normalization': True},
            'batchnorm3d': {'type': 'normalization', 'train_only': False, 'normalization': True},
            'layer_norm': {'type': 'normalization', 'train_only': False, 'normalization': True},
            'group_norm': {'type': 'normalization', 'train_only': False, 'normalization': True},
            'instance_norm1d': {'type': 'normalization', 'train_only': False, 'normalization': True},
            'instance_norm2d': {'type': 'normalization', 'train_only': False, 'normalization': True},
            'instance_norm3d': {'type': 'normalization', 'train_only': False, 'normalization': True},
        }
        return properties.get(reg_name, {})


def get_regularization_for_model(reg_config: Optional[Dict[str, Any]] = None,
                               use_case: str = 'general') -> nn.Module:
    """Convenience function to get a regularization layer.

    Args:
        reg_config: Optional configuration dict, if None uses settings
        use_case: Use case for recommendations if no config provided

    Returns:
        Configured regularization layer instance
    """
    # Get config from settings if not provided
    if reg_config is None:
        training_config = SETTINGS.get('training', {})
        reg_config = training_config.get('regularization', {})

        # Use default if no config
        if not reg_config:
            reg_config = {
                'type': 'dropout',  # Default regularization
                'params': {}
            }

    return RegularizationFactory.create_from_config(reg_config)


def list_all_regularizations() -> Dict[str, Any]:
    """Get comprehensive list of all available regularization techniques with details.

    Returns:
        Dictionary mapping regularization names to their information
    """
    result = {}
    for name in RegularizationFactory.get_available_regularizations():
        result[name] = RegularizationFactory.get_regularization_info(name)
    return result


def get_regularization_recommendations_for_case(use_case: str) -> Dict[str, Any]:
    """Deprecated: regularization recommendations removed. Kept for compatibility returning empty structure.

    Args:
        use_case: Use case parameter (ignored)

    Returns:
        Empty recommendations structure
    """
    return {"recommended": [], "description": lang("regularization.recommendations_removed")}