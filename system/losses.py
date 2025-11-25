import torch
import torch.nn as nn
from typing import Dict, Any, Type, List, Optional, Union
from torch.nn import Module

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS


class LossFactory:
    LOSSES: Dict[str, Type[nn.Module]] = {
        'cross_entropy': nn.CrossEntropyLoss,
        'nll_loss': nn.NLLLoss,
        'bce_loss': nn.BCELoss,
        'bce_with_logits': nn.BCEWithLogitsLoss,
        'multi_margin': nn.MultiMarginLoss,
        'multi_label_margin': nn.MultiLabelMarginLoss,
        'multi_label_soft_margin': nn.MultiLabelSoftMarginLoss,
        'mse_loss': nn.MSELoss,
        'l1_loss': nn.L1Loss,
        'smooth_l1': nn.SmoothL1Loss,
        'huber_loss': nn.HuberLoss,
        'kl_div': nn.KLDivLoss,
        'margin_ranking': nn.MarginRankingLoss,
        'hinge_embedding': nn.HingeEmbeddingLoss,
        'triplet_margin': nn.TripletMarginLoss,
        'cosine_embedding': nn.CosineEmbeddingLoss,
        'ctc_loss': nn.CTCLoss,
        'poisson_nll': nn.PoissonNLLLoss,
        'gaussian_nll': nn.GaussianNLLLoss,
    }

    @classmethod
    def _get_default_params(cls) -> Dict[str, Dict[str, Any]]:
        try:
            defaults = SETTINGS['losses']['defaults']
        except Exception:
            raise ValueError("losses.defaults not found in config.json - check config/config.json")
        if not isinstance(defaults, dict):
            raise ValueError("losses.defaults in config/config.json must be an object/dict")
        return defaults

    @classmethod
    def get_available_losses(cls) -> List[str]:
        return list(cls.LOSSES.keys())

    @classmethod
    def get_loss_info(cls, loss_name: str) -> Dict[str, Any]:
        if loss_name not in cls.LOSSES:
            return {}

        return {
            'name': loss_name,
            'class': cls.LOSSES[loss_name].__name__,
            'module': cls.LOSSES[loss_name].__module__,
            'default_params': cls._get_default_params().get(loss_name, {}),
            'description': cls._get_loss_description(loss_name),
            'properties': cls._get_loss_properties(loss_name),
        }

    # get_recommendations removed

    @classmethod
    def create_loss(cls,
                   loss_name: str,
                   custom_params: Optional[Dict[str, Any]] = None) -> nn.Module:
        if loss_name not in cls.LOSSES:
            available = ', '.join(cls.get_available_losses())
            raise ValueError(f"Unsupported loss function: {loss_name}. Available options: {available}")

        # Get default parameters from config
        default_params = cls._get_default_params().get(loss_name, {}).copy()

        # Override with custom parameters
        if custom_params:
            default_params.update(custom_params)

        try:
            loss_class = cls.LOSSES[loss_name]
            loss = loss_class(**default_params)

            info(f"Loss function created successfully: {loss_name} with parameters {default_params}")
            return loss

        except Exception as e:
            error(f"Failed to create loss function: {str(e)}")
            raise TypeError(f"Invalid parameters for loss function: {str(e)}")

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> nn.Module:
        if 'type' not in config:
            raise ValueError("Missing 'type' in loss configuration")

        loss_name = config['type'].lower()
        # Support flattened configs: if 'params' missing, use all top-level keys except 'type'
        if 'params' in config and isinstance(config['params'], dict):
            custom_params = config.get('params', {})
        else:
            custom_params = {k: v for k, v in config.items() if k != 'type'}
        
        return cls.create_loss(loss_name, custom_params)

    @classmethod
    def _get_loss_description(cls, loss_name: str) -> str:
        desc_key = f"losses.{loss_name}_desc"
        return ""

    @classmethod
    def _get_loss_properties(cls, loss_name: str) -> Dict[str, Any]:
        try:
            cfg_props = SETTINGS['losses']['properties']
        except Exception:
            raise ValueError("losses.properties not found in config/config.json - check your config")
        if not isinstance(cfg_props, dict) or loss_name not in cfg_props:
            raise ValueError(f"Missing properties for loss function '{loss_name}' in config/config.json - check losses.properties")
        return cfg_props.get(loss_name, {})


def get_loss_for_task(loss_config: Optional[Dict[str, Any]] = None,
                     task_type: str = 'classification') -> Optional[nn.Module]:
    if task_type == 'detection':
        return None

    # Get config from settings if not provided
    if loss_config is None:
        try:
            training_config = SETTINGS['training']
        except Exception:
            raise ValueError("Missing required 'training' section in config/config.json")

        # Build loss config from flattened keys
        if 'loss_type' not in training_config:
            raise ValueError("Missing required 'training.loss_type' in config/config.json")
        loss_type = training_config['loss_type']
        loss_params_source = training_config.get('loss_params', {})
        loss_params: Dict[str, Any] = {}
        if isinstance(loss_params_source, dict):
            selected_params = loss_params_source.get(loss_type, {})
            if isinstance(selected_params, dict):
                loss_params.update(selected_params)

        reduction_override = training_config.get('loss_reduction')
        if isinstance(reduction_override, str):
            loss_params['reduction'] = reduction_override

        if not loss_type:
            raise ValueError("No loss configuration found. Add 'training.loss' or 'training.loss_type' to config/config.json")

        loss_config = {
            'type': loss_type,
            'params': loss_params
        }

    return LossFactory.create_from_config(loss_config)


def list_all_losses() -> Dict[str, Any]:
    result = {}
    for name in LossFactory.get_available_losses():
        result[name] = LossFactory.get_loss_info(name)
    return result


def get_loss_recommendations_for_task(task_type: str) -> Dict[str, Any]:
    return {"recommended": [], "description": "Loss recommendations have been removed from the system"}