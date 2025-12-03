import torch
from torch.optim import lr_scheduler
from typing import Dict, Any, Type, List, Optional, Union
from torch.optim import Optimizer

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS


class LRSchedulerFactory:

    SCHEDULERS: Dict[str, Type] = {
        'step_lr': lr_scheduler.StepLR,
        'multi_step_lr': lr_scheduler.MultiStepLR,
        'exponential_lr': lr_scheduler.ExponentialLR,
        'cosine_annealing_lr': lr_scheduler.CosineAnnealingLR,
        'cosine_annealing_warm_restarts': lr_scheduler.CosineAnnealingWarmRestarts,
        'reduce_lr_on_plateau': lr_scheduler.ReduceLROnPlateau,
        'cyclic_lr': lr_scheduler.CyclicLR,
        'one_cycle_lr': lr_scheduler.OneCycleLR,
        'polynomial_lr': lr_scheduler.PolynomialLR,
        'linear_lr': lr_scheduler.LinearLR,
        'lambda_lr': lr_scheduler.LambdaLR,
        'multiplicative_lr': lr_scheduler.MultiplicativeLR,
    }

    @classmethod
    def _get_default_params(cls) -> Dict[str, Dict[str, Any]]:
        try:
            defaults = SETTINGS['schedulers']['defaults']
        except Exception:
            raise ValueError("schedulers.defaults not found in config/config.json - check config/config.json")
        if not isinstance(defaults, dict):
            raise ValueError("schedulers.defaults in config/config.json must be an object/dict")
        return defaults

    @classmethod
    def get_available_schedulers(cls) -> List[str]:
        return list(cls.SCHEDULERS.keys())

    @classmethod
    def get_scheduler_info(cls, scheduler_name: str) -> Dict[str, Any]:
        if scheduler_name not in cls.SCHEDULERS:
            return {}

        return {
            'name': scheduler_name,
            'class': cls.SCHEDULERS[scheduler_name].__name__,
            'module': cls.SCHEDULERS[scheduler_name].__module__,
            'default_params': cls._get_default_params().get(scheduler_name, {}),
            'description': cls._get_scheduler_description(scheduler_name),
            'properties': cls._get_scheduler_properties(scheduler_name),
        }

    @classmethod
    def create_scheduler(cls,
                        scheduler_name: str,
                        optimizer: Optimizer,
                        custom_params: Optional[Dict[str, Any]] = None):
        if scheduler_name not in cls.SCHEDULERS:
            available = ', '.join(cls.get_available_schedulers())
            raise ValueError(f"Unsupported scheduler: {scheduler_name}. Available options: {available}")

        default_params = cls._get_default_params().get(scheduler_name, {}).copy()
        
        if custom_params:
            default_params.update(custom_params)
        
        try:
            scheduler_class = cls.SCHEDULERS[scheduler_name]
            scheduler = scheduler_class(optimizer, **default_params)

            info(f"Scheduler created successfully: {scheduler_name} with parameters {default_params}")
            return scheduler

        except Exception as e:
            error(f"Failed to create scheduler: {str(e)}")
            raise TypeError(f"Invalid parameters for scheduler: {str(e)}")

    @classmethod
    def create_from_config(cls,
                          optimizer: Optimizer,
                          config: Dict[str, Any]):
        if 'type' not in config:
            raise ValueError("Missing 'type' in scheduler configuration")

        scheduler_name = config['type'].lower()
        if 'params' in config and isinstance(config['params'], dict):
            custom_params = config.get('params', {})
        else:
            custom_params = {k: v for k, v in config.items() if k != 'type'}

        return cls.create_scheduler(scheduler_name, optimizer, custom_params)

    @classmethod
    def _get_scheduler_description(cls, scheduler_name: str) -> str:
        desc_key = f"schedulers.{scheduler_name}_desc"
        return ""

    @classmethod
    def _get_scheduler_properties(cls, scheduler_name: str) -> Dict[str, Any]:
        try:
            cfg_props = SETTINGS['schedulers']['properties']
        except Exception:
            raise ValueError("schedulers.properties not found in config/config.json - check your config")
        if not isinstance(cfg_props, dict) or scheduler_name not in cfg_props:
            raise ValueError(f"Missing properties for scheduler '{scheduler_name}' in config/config.json - check schedulers.properties")
        return cfg_props.get(scheduler_name, {})


def get_scheduler_for_training(optimizer: Optimizer,
                              scheduler_config: Optional[Dict[str, Any]] = None,
                              scenario: str = 'standard_training') -> lr_scheduler._LRScheduler:
    if scheduler_config is None:
        try:
            training_config = SETTINGS['training']
        except Exception:
            raise ValueError("Missing required 'training' section in config/config.json")
        cfg = training_config.get('scheduler')
        if isinstance(cfg, dict) and cfg:
            scheduler_config = cfg
        else:
            if 'scheduler_type' not in training_config:
                raise ValueError("Missing required 'training.scheduler_type' in config/config.json")
            scheduler_type = training_config['scheduler_type']
            if scheduler_type:
                scheduler_params_source = training_config.get('scheduler_params', {})
                scheduler_params = {}
                if isinstance(scheduler_params_source, dict):
                    selected = scheduler_params_source.get(scheduler_type, {})
                    if isinstance(selected, dict):
                        scheduler_params.update(selected)
                
                # Only apply step_size/gamma overrides for schedulers that use them
                schedulers_with_step_size = ('step_lr', 'multi_step_lr')
                schedulers_with_gamma = ('step_lr', 'multi_step_lr', 'exponential_lr')
                
                if scheduler_type in schedulers_with_step_size:
                    step_size_override = training_config.get('scheduler_step_size')
                    if isinstance(step_size_override, int):
                        scheduler_params['step_size'] = step_size_override
                
                if scheduler_type in schedulers_with_gamma:
                    gamma_override = training_config.get('scheduler_gamma')
                    if isinstance(gamma_override, (int, float)):
                        scheduler_params['gamma'] = gamma_override
                
                scheduler_config = {'type': scheduler_type, 'params': scheduler_params}
            else:
                raise ValueError("No scheduler configuration found. Add 'training.scheduler' or 'training.scheduler_type' to config/config.json")

    return LRSchedulerFactory.create_from_config(optimizer, scheduler_config)


def list_all_schedulers() -> Dict[str, Any]:
    result = {}
    for name in LRSchedulerFactory.get_available_schedulers():
        result[name] = LRSchedulerFactory.get_scheduler_info(name)
    return result


def get_scheduler_recommendations_for_scenario(scenario: str) -> Dict[str, Any]:
    return {"recommended": [], "description": "Scheduler recommendations have been removed from the system"}