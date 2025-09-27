"""Data augmentation module for Hootsight.

Provides a unified interface for all torchvision data augmentation transforms with configuration support.
Supports random crop, flip, rotation, color jitter, and other augmentation techniques.
"""
import torch
from torchvision import transforms
from typing import Dict, Any, Type, List, Optional, Union
from torch.nn import Module

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS, lang


class DataAugmentationFactory:
    """Factory class for creating torchvision data augmentation transforms with unified configuration."""

    # Registry of all available data augmentation transforms
    AUGMENTATIONS: Dict[str, Type] = {
        # Geometric transforms
        'random_crop': transforms.RandomCrop,
        'random_resized_crop': transforms.RandomResizedCrop,
        'center_crop': transforms.CenterCrop,
        'random_horizontal_flip': transforms.RandomHorizontalFlip,
        'random_vertical_flip': transforms.RandomVerticalFlip,
        'random_rotation': transforms.RandomRotation,
        'random_affine': transforms.RandomAffine,
        'random_perspective': transforms.RandomPerspective,

        # Color transforms
        'color_jitter': transforms.ColorJitter,
        'random_grayscale': transforms.RandomGrayscale,
        'random_erasing': transforms.RandomErasing,

        # Other transforms
        'to_tensor': transforms.ToTensor,
        'normalize': transforms.Normalize,
        'random_invert': transforms.RandomInvert,
        'random_posterize': transforms.RandomPosterize,
        'random_solarize': transforms.RandomSolarize,
        'random_adjust_sharpness': transforms.RandomAdjustSharpness,
        'random_autocontrast': transforms.RandomAutocontrast,
        'random_equalize': transforms.RandomEqualize,

        # Compose (for combining multiple transforms)
        'compose': transforms.Compose,
    }

    # Default parameters for each augmentation transform
    DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
        'random_crop': {'size': 224, 'padding': None, 'pad_if_needed': False, 'fill': 0, 'padding_mode': 'constant'},
        'random_resized_crop': {'size': 224, 'scale': (0.08, 1.0), 'ratio': (3./4., 4./3.), 'interpolation': transforms.InterpolationMode.BILINEAR},
        'center_crop': {'size': 224},
        'random_horizontal_flip': {'p': 0.5},
        'random_vertical_flip': {'p': 0.5},
        'random_rotation': {'degrees': (-30, 30), 'interpolation': transforms.InterpolationMode.NEAREST, 'expand': False, 'center': None, 'fill': 0},
        'random_affine': {'degrees': (-30, 30), 'translate': None, 'scale': None, 'shear': None, 'interpolation': transforms.InterpolationMode.NEAREST, 'fill': 0, 'center': None},
        'random_perspective': {'distortion_scale': 0.5, 'p': 0.5, 'interpolation': transforms.InterpolationMode.BILINEAR, 'fill': 0},
        'color_jitter': {'brightness': 0.1, 'contrast': 0.1, 'saturation': 0.1, 'hue': 0.1},
        'random_grayscale': {'p': 0.1},
        'random_erasing': {'p': 0.5, 'scale': (0.02, 0.33), 'ratio': (0.3, 3.3), 'value': 0, 'inplace': False},
        'to_tensor': {},
        'normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        'random_invert': {'p': 0.5},
        'random_posterize': {'bits': 4, 'p': 0.5},
        'random_solarize': {'threshold': 128, 'p': 0.5},
        'random_adjust_sharpness': {'sharpness_factor': 2, 'p': 0.5},
        'random_autocontrast': {'p': 0.5},
        'random_equalize': {'p': 0.5},
        'compose': {'transforms': []},
    }

    # Note: Recommendations removed from system by request.

    @classmethod
    def get_available_augmentations(cls) -> List[str]:
        """Get list of all available data augmentation transforms."""
        return list(cls.AUGMENTATIONS.keys())

    @classmethod
    def get_augmentation_info(cls, aug_name: str) -> Dict[str, Any]:
        """Get information about a specific data augmentation transform.

        Args:
            aug_name: Name of the augmentation transform

        Returns:
            Dictionary with augmentation transform information including default parameters
        """
        if aug_name not in cls.AUGMENTATIONS:
            return {}

        return {
            'name': aug_name,
            'class': cls.AUGMENTATIONS[aug_name].__name__,
            'module': cls.AUGMENTATIONS[aug_name].__module__,
            'default_params': cls.DEFAULT_PARAMS.get(aug_name, {}),
            'description': cls._get_augmentation_description(aug_name),
            'properties': cls._get_augmentation_properties(aug_name),
        }

    @classmethod
    # get_recommendations removed

    @classmethod
    def create_augmentation(cls,
                          aug_name: str,
                          custom_params: Optional[Dict[str, Any]] = None):
        """Create a data augmentation transform instance.

        Args:
            aug_name: Name of the augmentation transform
            custom_params: Custom parameters to override defaults

        Returns:
            Configured augmentation transform instance

        Raises:
            ValueError: If augmentation transform is not supported
            TypeError: If parameters are invalid
        """
        if aug_name not in cls.AUGMENTATIONS:
            available = ', '.join(cls.get_available_augmentations())
            raise ValueError(lang("augmentation.not_supported", aug=aug_name, available=available))

        # Get default parameters
        default_params = cls.DEFAULT_PARAMS.get(aug_name, {}).copy()

        # Override with custom parameters
        if custom_params:
            default_params.update(custom_params)

        try:
            aug_class = cls.AUGMENTATIONS[aug_name]

            # Special handling for Compose
            if aug_name == 'compose':
                transform_list = default_params.get('transforms', [])
                if not transform_list:
                    # Create default composition
                    transform_list = [
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]
                augmentation = aug_class(transform_list)
            else:
                augmentation = aug_class(**default_params)

            info(lang("augmentation.created", aug=aug_name, params=default_params))
            return augmentation

        except Exception as e:
            error(lang("augmentation.creation_failed", aug=aug_name, error=str(e)))
            raise TypeError(lang("augmentation.invalid_params", aug=aug_name, error=str(e)))

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]):
        """Create augmentation transform from configuration dictionary.

        Args:
            config: Configuration dictionary with 'type' and 'params' keys

        Returns:
            Configured augmentation transform instance

        Example config:
            {
                "type": "color_jitter",
                "params": {"brightness": 0.2, "contrast": 0.2}
            }
        """
        if 'type' not in config:
            raise ValueError(lang("augmentation.config_missing_type"))

        aug_name = config['type'].lower()
        custom_params = config.get('params', {})

        return cls.create_augmentation(aug_name, custom_params)

    @classmethod
    def create_composition(cls, transform_configs: List[Dict[str, Any]]):
        """Create a composition of multiple augmentation transforms.

        Args:
            transform_configs: List of transform configuration dictionaries

        Returns:
            Composed transform

        Example:
            configs = [
                {"type": "random_horizontal_flip", "params": {"p": 0.5}},
                {"type": "color_jitter", "params": {"brightness": 0.1}}
            ]
        """
        ordered_configs: List[Dict[str, Any]] = []
        deferred_resize: List[Dict[str, Any]] = []
        trailing_tensors: List[Dict[str, Any]] = []

        for config in transform_configs:
            if not isinstance(config, dict):
                continue
            aug_type = (config.get('type') or '').lower()
            if aug_type in {'to_tensor', 'normalize'}:
                trailing_tensors.append(config)
            elif aug_type == 'random_resized_crop':
                deferred_resize.append(config)
            else:
                ordered_configs.append(config)

        normalized_order = ordered_configs + deferred_resize + trailing_tensors

        transforms_list: List[Module] = []
        for config in normalized_order:
            transform = cls.create_from_config(config)
            transforms_list.append(transform)

        return transforms.Compose(transforms_list)

    @classmethod
    def _get_augmentation_description(cls, aug_name: str) -> str:
        """Get description for a data augmentation transform."""
        desc_key = f"augmentation.{aug_name}_desc"
        return lang(desc_key)

    @classmethod
    def _get_augmentation_properties(cls, aug_name: str) -> Dict[str, Any]:
        """Get properties of a data augmentation transform."""
        properties = {
            'random_crop': {'type': 'geometric', 'random': True, 'spatial': True},
            'random_resized_crop': {'type': 'geometric', 'random': True, 'spatial': True},
            'center_crop': {'type': 'geometric', 'random': False, 'spatial': True},
            'random_horizontal_flip': {'type': 'geometric', 'random': True, 'spatial': True},
            'random_vertical_flip': {'type': 'geometric', 'random': True, 'spatial': True},
            'random_rotation': {'type': 'geometric', 'random': True, 'spatial': True},
            'random_affine': {'type': 'geometric', 'random': True, 'spatial': True},
            'random_perspective': {'type': 'geometric', 'random': True, 'spatial': True},
            'color_jitter': {'type': 'color', 'random': True, 'spatial': False},
            'random_grayscale': {'type': 'color', 'random': True, 'spatial': False},
            'random_erasing': {'type': 'erasing', 'random': True, 'spatial': True},
            'normalize': {'type': 'normalization', 'random': False, 'spatial': False},
            'random_invert': {'type': 'color', 'random': True, 'spatial': False},
            'random_posterize': {'type': 'color', 'random': True, 'spatial': False},
            'random_solarize': {'type': 'color', 'random': True, 'spatial': False},
            'random_adjust_sharpness': {'type': 'color', 'random': True, 'spatial': False},
            'random_autocontrast': {'type': 'color', 'random': True, 'spatial': False},
            'random_equalize': {'type': 'color', 'random': True, 'spatial': False},
            'compose': {'type': 'composition', 'random': True, 'spatial': True},
        }
        return properties.get(aug_name, {})


def get_augmentation_for_dataset(aug_config: Optional[Dict[str, Any]] = None,
                               dataset_type: str = 'general'):
    """Convenience function to get data augmentation for a dataset.

    Args:
        aug_config: Optional configuration dict, if None uses settings
        dataset_type: Dataset type for recommendations if no config provided

    Returns:
        Configured augmentation transform instance
    """
    # Get config from settings if not provided
    if aug_config is None:
        training_config = SETTINGS.get('training', {})
        aug_config = training_config.get('augmentation', {})

        # Fallback to default compose pipeline if no config provided
        if not aug_config:
            return DataAugmentationFactory.create_augmentation('compose', {})

    return DataAugmentationFactory.create_from_config(aug_config)


def list_all_augmentations() -> Dict[str, Any]:
    """Get comprehensive list of all available data augmentation transforms with details.

    Returns:
        Dictionary mapping augmentation names to their information
    """
    result = {}
    for name in DataAugmentationFactory.get_available_augmentations():
        result[name] = DataAugmentationFactory.get_augmentation_info(name)
    return result


def get_augmentation_recommendations_for_dataset(dataset_type: str) -> Dict[str, Any]:
    """Deprecated: recommendations removed. Kept for compatibility returning empty structure."""
    return {"recommended": [], "description": lang("augmentation.recommendations_removed")}