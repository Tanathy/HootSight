import torch
from torchvision import transforms
from typing import Dict, Any, Type, List, Optional, Union
from torch.nn import Module

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS


class DataAugmentationFactory:

    AUGMENTATIONS: Dict[str, Type] = {
        'resize': transforms.Resize,
        'random_crop': transforms.RandomCrop,
        'random_resized_crop': transforms.RandomResizedCrop,
        'center_crop': transforms.CenterCrop,
        'random_horizontal_flip': transforms.RandomHorizontalFlip,
        'random_vertical_flip': transforms.RandomVerticalFlip,
        'random_rotation': transforms.RandomRotation,
        'random_affine': transforms.RandomAffine,
        'random_perspective': transforms.RandomPerspective,
        'color_jitter': transforms.ColorJitter,
        'random_grayscale': transforms.RandomGrayscale,
        'random_erasing': transforms.RandomErasing,
        'to_tensor': transforms.ToTensor,
        'normalize': transforms.Normalize,
        'random_invert': transforms.RandomInvert,
        'random_posterize': transforms.RandomPosterize,
        'random_solarize': transforms.RandomSolarize,
        'random_adjust_sharpness': transforms.RandomAdjustSharpness,
        'random_autocontrast': transforms.RandomAutocontrast,
        'random_equalize': transforms.RandomEqualize,
        'compose': transforms.Compose,
    }

    DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_available_augmentations(cls) -> List[str]:
        return list(cls.AUGMENTATIONS.keys())

    @classmethod
    def get_augmentation_info(cls, aug_name: str) -> Dict[str, Any]:
        if aug_name not in cls.AUGMENTATIONS:
            return {}

        defaults = cls._get_default_params()
        return {
            'name': aug_name,
            'class': cls.AUGMENTATIONS[aug_name].__name__,
            'module': cls.AUGMENTATIONS[aug_name].__module__,
            'default_params': defaults.get(aug_name, {}),
            'description': cls._get_augmentation_description(aug_name),
            'properties': cls._get_augmentation_properties(aug_name),
        }

    @classmethod
    def create_augmentation(cls,
                          aug_name: str,
                          custom_params: Optional[Dict[str, Any]] = None):
        if aug_name not in cls.AUGMENTATIONS:
            available = ', '.join(cls.get_available_augmentations())
            raise ValueError(f"Unsupported augmentation: {aug_name}. Available options: {available}")

        
        default_params = cls._get_default_params().get(aug_name, {}).copy()

        
        if custom_params:
            default_params.update(custom_params)

        try:
            aug_class = cls.AUGMENTATIONS[aug_name]

            
            if aug_name == 'compose':
                transform_list = default_params.get('transforms', [])
                if not transform_list:
                    raise ValueError("augmentation.compose must define 'transforms' in config under augmentations.defaults.compose or the specific augmentation config.")
                augmentation = cls.create_composition(transform_list)
            else:
                
                normalised_params = cls._convert_param_types(aug_name, default_params)
                augmentation = aug_class(**normalised_params)

            info(f"Augmentation created successfully: {aug_name} with parameters {default_params}")
            return augmentation

        except Exception as e:
            error(f"Failed to create augmentation: {str(e)}")
            raise TypeError(f"Invalid parameters for augmentation: {str(e)}")

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]):
        if 'type' not in config:
            raise ValueError("Missing 'type' in augmentation configuration")

        aug_name = config['type'].lower()
        custom_params = config.get('params', {})

        return cls.create_augmentation(aug_name, custom_params)

    @classmethod
    def create_composition(cls, transform_configs: List[Dict[str, Any]]):
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

        transforms_list: List[Any] = []
        for config in normalized_order:
            transform = cls.create_from_config(config)
            transforms_list.append(transform)

        return transforms.Compose(transforms_list)

    @classmethod
    def _get_augmentation_description(cls, aug_name: str) -> str:
        desc_key = f"augmentation.{aug_name}_desc"
        return ""

    @classmethod
    def _get_augmentation_properties(cls, aug_name: str) -> Dict[str, Any]:
        
        try:
            cfg_props = SETTINGS['augmentations']['properties']
        except Exception:
            raise ValueError("augmentations.properties not found in config/config.json - check your config")
        if not isinstance(cfg_props, dict) or aug_name not in cfg_props:
            raise ValueError(f"Augmentation properties for '{aug_name}' must be defined under 'augmentations.properties' in config/config.json.")
        return cfg_props[aug_name]

    @classmethod
    def _get_default_params(cls) -> Dict[str, Dict[str, Any]]:
        try:
            defaults = SETTINGS['augmentations']['defaults']
        except Exception:
            raise ValueError("augmentations.defaults not found in config/config.json - check your config")
        if not isinstance(defaults, dict):
            raise ValueError("augmentations.defaults in config/config.json must be an object/dict")
        return defaults

    @classmethod
    def _convert_param_types(cls, aug_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        converted = dict(params)
        
        interp_key = 'interpolation'
        if interp_key in converted:
            val = converted[interp_key]
            
            if isinstance(val, str):
                enum_name = val.upper()
                try:
                    converted[interp_key] = getattr(transforms.InterpolationMode, enum_name)
                except Exception:
                    
                    try:
                        converted[interp_key] = getattr(transforms.InterpolationMode, enum_name.replace('-', '_'))
                    except Exception:
                        
                        converted[interp_key] = val
        return converted


def get_augmentation_for_dataset(aug_config: Optional[Dict[str, Any]] = None,
                               dataset_type: str = 'general'):
    
    if aug_config is None:
        try:
            training_config = SETTINGS['training']
        except Exception:
            raise ValueError("Missing required 'training' section in config/config.json")
        if 'augmentation' not in training_config:
            raise ValueError("Missing required 'training.augmentation' in config/config.json")
        aug_config = training_config['augmentation']

    
    if isinstance(aug_config, dict) and ('train' in aug_config or 'val' in aug_config):
        
        key = 'train' if dataset_type in ('train', 'general', 'training') else 'val'
        pipeline = aug_config.get(key, [])
        
        if isinstance(pipeline, list) and pipeline:
            return DataAugmentationFactory.create_composition(pipeline)
        
        try:
            defaults = SETTINGS['augmentations']['defaults']
        except Exception:
            raise ValueError("augmentations.defaults not found in config/config.json - check your config")
        compose_defaults = defaults.get('compose') if isinstance(defaults, dict) else None
        if compose_defaults and isinstance(compose_defaults, dict) and 'transforms' in compose_defaults:
            return DataAugmentationFactory.create_augmentation('compose', compose_defaults)
        raise ValueError("No augmentation pipeline found in training.augmentation and no augmentations.defaults.compose is defined in config.")

    
    if isinstance(aug_config, dict) and 'type' in aug_config:
        return DataAugmentationFactory.create_from_config(aug_config)

    
    raise ValueError("Invalid augmentation config provided. Must be either a pipeline dict with 'train'/'val' or a single transform dict with 'type'.")


def list_all_augmentations() -> Dict[str, Any]:
    result = {}
    for name in DataAugmentationFactory.get_available_augmentations():
        result[name] = DataAugmentationFactory.get_augmentation_info(name)
    return result


def get_augmentation_recommendations_for_dataset(dataset_type: str) -> Dict[str, Any]:
    return {"recommended": [], "description": "Augmentation recommendations removed"}