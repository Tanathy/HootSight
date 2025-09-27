"""Normalization layers module for Hootsight.

Provides a unified interface for all PyTorch normalization layers with configuration support.
Supports BatchNorm, LayerNorm, GroupNorm, InstanceNorm, and other normalization operations.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Type, List, Optional, Union, Tuple
from torch.nn import Module

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS, lang


class NormalizationFactory:
    """Factory class for creating PyTorch normalization layers with unified configuration."""

    # Registry of all available normalization layers
    NORMALIZATIONS: Dict[str, Type[nn.Module]] = {
        # Batch normalization variants
        'batch_norm1d': nn.BatchNorm1d,
        'batch_norm2d': nn.BatchNorm2d,
        'batch_norm3d': nn.BatchNorm3d,
        'sync_batch_norm': nn.SyncBatchNorm,

        # Layer normalization
        'layer_norm': nn.LayerNorm,
        'group_norm': nn.GroupNorm,
        'instance_norm1d': nn.InstanceNorm1d,
        'instance_norm2d': nn.InstanceNorm2d,
        'instance_norm3d': nn.InstanceNorm3d,

        # Other normalization variants
        'local_response_norm': nn.LocalResponseNorm,
        'cross_map_lr_norm': nn.CrossMapLRN2d,
    }

    # Default parameters for each normalization layer
    DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
        'batch_norm1d': {'num_features': None, 'eps': 1e-5, 'momentum': 0.1, 'affine': True, 'track_running_stats': True},
        'batch_norm2d': {'num_features': None, 'eps': 1e-5, 'momentum': 0.1, 'affine': True, 'track_running_stats': True},
        'batch_norm3d': {'num_features': None, 'eps': 1e-5, 'momentum': 0.1, 'affine': True, 'track_running_stats': True},
        'sync_batch_norm': {'num_features': None, 'eps': 1e-5, 'momentum': 0.1, 'affine': True, 'track_running_stats': True, 'process_group': None},
        'layer_norm': {'normalized_shape': None, 'eps': 1e-5, 'elementwise_affine': True},
        'group_norm': {'num_groups': 32, 'num_channels': None, 'eps': 1e-5, 'affine': True},
        'instance_norm1d': {'num_features': None, 'eps': 1e-5, 'momentum': 0.1, 'affine': False, 'track_running_stats': False},
        'instance_norm2d': {'num_features': None, 'eps': 1e-5, 'momentum': 0.1, 'affine': False, 'track_running_stats': False},
        'instance_norm3d': {'num_features': None, 'eps': 1e-5, 'momentum': 0.1, 'affine': False, 'track_running_stats': False},
        'local_response_norm': {'size': 5, 'alpha': 0.0001, 'beta': 0.75, 'k': 1.0},
        'cross_map_lr_norm': {'size': 5, 'alpha': 0.0001, 'beta': 0.75, 'k': 1.0},
    }

    # Note: Recommendations removed from system by request.

    @classmethod
    def get_available_normalizations(cls) -> List[str]:
        """Get list of all available normalization layers."""
        return list(cls.NORMALIZATIONS.keys())

    @classmethod
    def get_normalization_info(cls, norm_name: str) -> Dict[str, Any]:
        """Get information about a specific normalization layer.

        Args:
            norm_name: Name of the normalization layer

        Returns:
            Dictionary with normalization layer information including default parameters
        """
        if norm_name not in cls.NORMALIZATIONS:
            return {}

        return {
            'name': norm_name,
            'class': cls.NORMALIZATIONS[norm_name].__name__,
            'module': cls.NORMALIZATIONS[norm_name].__module__,
            'default_params': cls.DEFAULT_PARAMS.get(norm_name, {}),
            'description': cls._get_normalization_description(norm_name),
            'properties': cls._get_normalization_properties(norm_name),
        }

    # get_recommendations removed

    @classmethod
    def create_normalization(cls,
                           norm_name: str,
                           custom_params: Optional[Dict[str, Any]] = None) -> nn.Module:
        """Create a normalization layer instance.

        Args:
            norm_name: Name of the normalization layer
            custom_params: Custom parameters to override defaults

        Returns:
            Configured normalization layer instance

        Raises:
            ValueError: If normalization layer is not supported
            TypeError: If parameters are invalid
        """
        if norm_name not in cls.NORMALIZATIONS:
            available = ', '.join(cls.get_available_normalizations())
            raise ValueError(lang("normalization.not_supported", norm=norm_name, available=available))

        # Get default parameters
        default_params = cls.DEFAULT_PARAMS.get(norm_name, {}).copy()

        # Override with custom parameters
        if custom_params:
            default_params.update(custom_params)

        # Handle special cases for required parameters
        if norm_name in ['batch_norm1d', 'batch_norm2d', 'batch_norm3d', 'sync_batch_norm',
                        'instance_norm1d', 'instance_norm2d', 'instance_norm3d']:
            if default_params.get('num_features') is None:
                raise ValueError(lang("normalization.missing_num_features", norm=norm_name))

        if norm_name == 'group_norm':
            if default_params.get('num_channels') is None:
                raise ValueError(lang("normalization.missing_num_channels", norm=norm_name))

        if norm_name == 'layer_norm':
            if default_params.get('normalized_shape') is None:
                raise ValueError(lang("normalization.missing_normalized_shape", norm=norm_name))

        try:
            norm_class = cls.NORMALIZATIONS[norm_name]
            normalization = norm_class(**default_params)

            info(lang("normalization.created", norm=norm_name, params=default_params))
            return normalization

        except Exception as e:
            error(lang("normalization.creation_failed", norm=norm_name, error=str(e)))
            raise TypeError(lang("normalization.invalid_params", norm=norm_name, error=str(e)))

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> nn.Module:
        """Create normalization layer from configuration dictionary.

        Args:
            config: Configuration dictionary with 'type' and 'params' keys

        Returns:
            Configured normalization layer instance

        Example config:
            {
                "type": "batch_norm2d",
                "params": {"num_features": 64}
            }
        """
        if 'type' not in config:
            raise ValueError(lang("normalization.config_missing_type"))

        norm_name = config['type'].lower()
        custom_params = config.get('params', {})

        return cls.create_normalization(norm_name, custom_params)

    @classmethod
    def _get_normalization_description(cls, norm_name: str) -> str:
        """Get description for a normalization layer."""
        desc_key = f"normalization.{norm_name}_desc"
        return lang(desc_key)

    @classmethod
    def _get_normalization_properties(cls, norm_name: str) -> Dict[str, Any]:
        """Get properties of a normalization layer."""
        properties = {
            'batch_norm1d': {'type': 'batch', 'learnable': True, 'dimensions': 1, 'requires_stats': True},
            'batch_norm2d': {'type': 'batch', 'learnable': True, 'dimensions': 2, 'requires_stats': True},
            'batch_norm3d': {'type': 'batch', 'learnable': True, 'dimensions': 3, 'requires_stats': True},
            'sync_batch_norm': {'type': 'batch', 'learnable': True, 'dimensions': None, 'requires_stats': True, 'distributed': True},
            'layer_norm': {'type': 'layer', 'learnable': True, 'dimensions': None, 'requires_stats': False},
            'group_norm': {'type': 'group', 'learnable': True, 'dimensions': None, 'requires_stats': False},
            'instance_norm1d': {'type': 'instance', 'learnable': False, 'dimensions': 1, 'requires_stats': False},
            'instance_norm2d': {'type': 'instance', 'learnable': False, 'dimensions': 2, 'requires_stats': False},
            'instance_norm3d': {'type': 'instance', 'learnable': False, 'dimensions': 3, 'requires_stats': False},
            'local_response_norm': {'type': 'local_response', 'learnable': False, 'dimensions': None, 'requires_stats': False},
            'cross_map_lr_norm': {'type': 'cross_map_lrn', 'learnable': False, 'dimensions': 2, 'requires_stats': False},
        }
        return properties.get(norm_name, {})


def get_normalization_for_network(norm_config: Optional[Dict[str, Any]] = None,
                                use_case: str = 'general') -> nn.Module:
    """Convenience function to get a normalization layer for a network.

    Args:
        norm_config: Optional configuration dict, if None uses settings
        use_case: Use case parameter (deprecated, kept for compatibility)

    Returns:
        Configured normalization layer instance
    """
    # Get config from settings if not provided
    if norm_config is None:
        training_config = SETTINGS.get('training', {})
        norm_config = training_config.get('normalization', {})

        # Use default if no config
        if not norm_config:
            norm_config = {
                'type': 'batch_norm2d',
                'params': {}
            }

    return NormalizationFactory.create_from_config(norm_config)


def list_all_normalizations() -> Dict[str, Any]:
    """Get comprehensive list of all available normalization layers with details.

    Returns:
        Dictionary mapping normalization names to their information
    """
    result = {}
    for name in NormalizationFactory.get_available_normalizations():
        result[name] = NormalizationFactory.get_normalization_info(name)
    return result


def get_normalization_recommendations_for_case(use_case: str) -> Dict[str, Any]:
    """Deprecated: recommendations removed. Kept for compatibility returning empty structure.

    Args:
        use_case: Use case parameter (ignored)

    Returns:
        Empty recommendations structure
    """
    return {"recommended": [], "description": lang("normalization.recommendations_removed")}