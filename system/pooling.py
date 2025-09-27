"""Pooling layers module for Hootsight.

Provides a unified interface for all PyTorch pooling layers with configuration support.
Supports MaxPool, AvgPool, GlobalAvgPool, and other pooling operations.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Type, List, Optional, Union, Tuple
from torch.nn import Module

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS, lang


class PoolingFactory:
    """Factory class for creating PyTorch pooling layers with unified configuration."""

    # Registry of all available pooling layers
    POOLINGS: Dict[str, Type[nn.Module]] = {
        # Max pooling variants
        'max_pool1d': nn.MaxPool1d,
        'max_pool2d': nn.MaxPool2d,
        'max_pool3d': nn.MaxPool3d,
        'max_unpool1d': nn.MaxUnpool1d,
        'max_unpool2d': nn.MaxUnpool2d,
        'max_unpool3d': nn.MaxUnpool3d,
        'adaptive_max_pool1d': nn.AdaptiveMaxPool1d,
        'adaptive_max_pool2d': nn.AdaptiveMaxPool2d,
        'adaptive_max_pool3d': nn.AdaptiveMaxPool3d,

        # Average pooling variants
        'avg_pool1d': nn.AvgPool1d,
        'avg_pool2d': nn.AvgPool2d,
        'avg_pool3d': nn.AvgPool3d,
        'adaptive_avg_pool1d': nn.AdaptiveAvgPool1d,
        'adaptive_avg_pool2d': nn.AdaptiveAvgPool2d,
        'adaptive_avg_pool3d': nn.AdaptiveAvgPool3d,

        # Global pooling (special case of adaptive pooling)
        'global_max_pool1d': nn.AdaptiveMaxPool1d,
        'global_max_pool2d': nn.AdaptiveMaxPool2d,
        'global_max_pool3d': nn.AdaptiveMaxPool3d,
        'global_avg_pool1d': nn.AdaptiveAvgPool1d,
        'global_avg_pool2d': nn.AdaptiveAvgPool2d,
        'global_avg_pool3d': nn.AdaptiveAvgPool3d,

        # Other pooling variants
        'lppool1d': nn.LPPool1d,
        'lppool2d': nn.LPPool2d,
        'fractional_max_pool2d': nn.FractionalMaxPool2d,
    }

    # Default parameters for each pooling layer
    DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
        'max_pool1d': {'kernel_size': 2, 'stride': None, 'padding': 0, 'dilation': 1, 'return_indices': False, 'ceil_mode': False},
        'max_pool2d': {'kernel_size': 2, 'stride': None, 'padding': 0, 'dilation': 1, 'return_indices': False, 'ceil_mode': False},
        'max_pool3d': {'kernel_size': 2, 'stride': None, 'padding': 0, 'dilation': 1, 'return_indices': False, 'ceil_mode': False},
        'max_unpool1d': {'kernel_size': 2, 'stride': None, 'padding': 0},
        'max_unpool2d': {'kernel_size': 2, 'stride': None, 'padding': 0},
        'max_unpool3d': {'kernel_size': 2, 'stride': None, 'padding': 0},
        'adaptive_max_pool1d': {'output_size': 1},
        'adaptive_max_pool2d': {'output_size': 1},
        'adaptive_max_pool3d': {'output_size': 1},
        'avg_pool1d': {'kernel_size': 2, 'stride': None, 'padding': 0, 'ceil_mode': False, 'count_include_pad': True},
        'avg_pool2d': {'kernel_size': 2, 'stride': None, 'padding': 0, 'ceil_mode': False, 'count_include_pad': True, 'divisor_override': None},
        'avg_pool3d': {'kernel_size': 2, 'stride': None, 'padding': 0, 'ceil_mode': False, 'count_include_pad': True, 'divisor_override': None},
        'adaptive_avg_pool1d': {'output_size': 1},
        'adaptive_avg_pool2d': {'output_size': 1},
        'adaptive_avg_pool3d': {'output_size': 1},
        'global_max_pool1d': {'output_size': 1},
        'global_max_pool2d': {'output_size': 1},
        'global_max_pool3d': {'output_size': 1},
        'global_avg_pool1d': {'output_size': 1},
        'global_avg_pool2d': {'output_size': 1},
        'global_avg_pool3d': {'output_size': 1},
        'lppool1d': {'norm_type': 2, 'kernel_size': 2, 'stride': None, 'ceil_mode': False},
        'lppool2d': {'norm_type': 2, 'kernel_size': 2, 'stride': None, 'ceil_mode': False},
        'fractional_max_pool2d': {'kernel_size': 2, 'output_size': None, 'output_ratio': None, 'return_indices': False, '_random_samples': None},
    }

    # Note: Recommendations removed from system by request.

    @classmethod
    def get_available_poolings(cls) -> List[str]:
        """Get list of all available pooling layers."""
        return list(cls.POOLINGS.keys())

    @classmethod
    def get_pooling_info(cls, pool_name: str) -> Dict[str, Any]:
        """Get information about a specific pooling layer.

        Args:
            pool_name: Name of the pooling layer

        Returns:
            Dictionary with pooling layer information including default parameters
        """
        if pool_name not in cls.POOLINGS:
            return {}

        return {
            'name': pool_name,
            'class': cls.POOLINGS[pool_name].__name__,
            'module': cls.POOLINGS[pool_name].__module__,
            'default_params': cls.DEFAULT_PARAMS.get(pool_name, {}),
            'description': cls._get_pooling_description(pool_name),
            'properties': cls._get_pooling_properties(pool_name),
        }

    # get_recommendations removed

    @classmethod
    def create_pooling(cls,
                      pool_name: str,
                      custom_params: Optional[Dict[str, Any]] = None) -> nn.Module:
        """Create a pooling layer instance.

        Args:
            pool_name: Name of the pooling layer
            custom_params: Custom parameters to override defaults

        Returns:
            Configured pooling layer instance

        Raises:
            ValueError: If pooling layer is not supported
            TypeError: If parameters are invalid
        """
        if pool_name not in cls.POOLINGS:
            available = ', '.join(cls.get_available_poolings())
            raise ValueError(lang("pooling.not_supported", pool=pool_name, available=available))

        # Get default parameters
        default_params = cls.DEFAULT_PARAMS.get(pool_name, {}).copy()

        # Override with custom parameters
        if custom_params:
            default_params.update(custom_params)

        try:
            pool_class = cls.POOLINGS[pool_name]
            pooling = pool_class(**default_params)

            info(lang("pooling.created", pool=pool_name, params=default_params))
            return pooling

        except Exception as e:
            error(lang("pooling.creation_failed", pool=pool_name, error=str(e)))
            raise TypeError(lang("pooling.invalid_params", pool=pool_name, error=str(e)))

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> nn.Module:
        """Create pooling layer from configuration dictionary.

        Args:
            config: Configuration dictionary with 'type' and 'params' keys

        Returns:
            Configured pooling layer instance

        Example config:
            {
                "type": "max_pool2d",
                "params": {"kernel_size": 2, "stride": 2}
            }
        """
        if 'type' not in config:
            raise ValueError(lang("pooling.config_missing_type"))

        pool_name = config['type'].lower()
        custom_params = config.get('params', {})

        return cls.create_pooling(pool_name, custom_params)

    @classmethod
    def _get_pooling_description(cls, pool_name: str) -> str:
        """Get description for a pooling layer."""
        desc_key = f"pooling.{pool_name}_desc"
        return lang(desc_key)

    @classmethod
    def _get_pooling_properties(cls, pool_name: str) -> Dict[str, Any]:
        """Get properties of a pooling layer."""
        properties = {
            'max_pool1d': {'type': 'max', 'adaptive': False, 'dimensions': 1},
            'max_pool2d': {'type': 'max', 'adaptive': False, 'dimensions': 2},
            'max_pool3d': {'type': 'max', 'adaptive': False, 'dimensions': 3},
            'max_unpool1d': {'type': 'max_unpool', 'adaptive': False, 'dimensions': 1},
            'max_unpool2d': {'type': 'max_unpool', 'adaptive': False, 'dimensions': 2},
            'max_unpool3d': {'type': 'max_unpool', 'adaptive': False, 'dimensions': 3},
            'adaptive_max_pool1d': {'type': 'max', 'adaptive': True, 'dimensions': 1},
            'adaptive_max_pool2d': {'type': 'max', 'adaptive': True, 'dimensions': 2},
            'adaptive_max_pool3d': {'type': 'max', 'adaptive': True, 'dimensions': 3},
            'avg_pool1d': {'type': 'avg', 'adaptive': False, 'dimensions': 1},
            'avg_pool2d': {'type': 'avg', 'adaptive': False, 'dimensions': 2},
            'avg_pool3d': {'type': 'avg', 'adaptive': False, 'dimensions': 3},
            'adaptive_avg_pool1d': {'type': 'avg', 'adaptive': True, 'dimensions': 1},
            'adaptive_avg_pool2d': {'type': 'avg', 'adaptive': True, 'dimensions': 2},
            'adaptive_avg_pool3d': {'type': 'avg', 'adaptive': True, 'dimensions': 3},
            'global_max_pool1d': {'type': 'max', 'adaptive': True, 'dimensions': 1, 'global': True},
            'global_max_pool2d': {'type': 'max', 'adaptive': True, 'dimensions': 2, 'global': True},
            'global_max_pool3d': {'type': 'max', 'adaptive': True, 'dimensions': 3, 'global': True},
            'global_avg_pool1d': {'type': 'avg', 'adaptive': True, 'dimensions': 1, 'global': True},
            'global_avg_pool2d': {'type': 'avg', 'adaptive': True, 'dimensions': 2, 'global': True},
            'global_avg_pool3d': {'type': 'avg', 'adaptive': True, 'dimensions': 3, 'global': True},
            'lppool1d': {'type': 'lp', 'adaptive': False, 'dimensions': 1},
            'lppool2d': {'type': 'lp', 'adaptive': False, 'dimensions': 2},
            'fractional_max_pool2d': {'type': 'fractional_max', 'adaptive': False, 'dimensions': 2},
        }
        return properties.get(pool_name, {})


def get_pooling_for_network(pool_config: Optional[Dict[str, Any]] = None,
                          use_case: str = 'general') -> nn.Module:
    """Convenience function to get a pooling layer for a network.

    Args:
        pool_config: Optional configuration dict, if None uses settings
        use_case: Use case for recommendations if no config provided

    Returns:
        Configured pooling layer instance
    """
    # Get config from settings if not provided
    if pool_config is None:
        training_config = SETTINGS.get('training', {})
        pool_config = training_config.get('pooling', {})

        # Use default if no config
        if not pool_config:
            pool_config = {
                'type': 'max_pool2d',  # Default pooling
                'params': {}
            }

    return PoolingFactory.create_from_config(pool_config)


def list_all_poolings() -> Dict[str, Any]:
    """Get comprehensive list of all available pooling layers with details.

    Returns:
        Dictionary mapping pooling names to their information
    """
    result = {}
    for name in PoolingFactory.get_available_poolings():
        result[name] = PoolingFactory.get_pooling_info(name)
    return result


def get_pooling_recommendations_for_case(use_case: str) -> Dict[str, Any]:
    """Deprecated: pooling recommendations removed. Kept for compatibility returning empty structure.

    Args:
        use_case: Use case parameter (ignored)

    Returns:
        Empty recommendations structure
    """
    return {"recommended": [], "description": lang("pooling.recommendations_removed")}