import torch
import torch.optim as optim
from typing import Dict, Any, Type, List, Optional
from torch.nn import Module

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS


class OptimizerFactory:
    
    OPTIMIZERS: Dict[str, Type[optim.Optimizer]] = {
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
        'lbfgs': optim.LBFGS,
        'asgd': optim.ASGD,
    }
    
    @classmethod
    def _get_default_params(cls) -> Dict[str, Dict[str, Any]]:
        try:
            defaults = SETTINGS['optimizers']['defaults']
        except Exception:
            raise ValueError("optimizers.defaults not found in config.json - check config/config.json")
        if not isinstance(defaults, dict):
            raise ValueError("optimizers.defaults in config/config.json must be an object/dict")
        return defaults

    @classmethod
    def get_available_optimizers(cls) -> List[str]:
        return list(cls.OPTIMIZERS.keys())

    @classmethod
    def get_optimizer_info(cls, optimizer_name: str) -> Dict[str, Any]:
        if optimizer_name not in cls.OPTIMIZERS:
            return {}
        
        return {
            'name': optimizer_name,
            'class': cls.OPTIMIZERS[optimizer_name].__name__,
            'module': cls.OPTIMIZERS[optimizer_name].__module__,
            'default_params': cls._get_default_params().get(optimizer_name, {}),
            'description': cls._get_optimizer_description(optimizer_name),
            'properties': cls._get_optimizer_properties(optimizer_name),
        }

    @classmethod
    def create_optimizer(cls, 
                        optimizer_name: str, 
                        model_parameters, 
                        custom_params: Optional[Dict[str, Any]] = None) -> optim.Optimizer:
        if optimizer_name not in cls.OPTIMIZERS:
            available = ', '.join(cls.get_available_optimizers())
            raise ValueError(f"Unsupported optimizer: {optimizer_name}. Available options: {available}")
        
        default_params = cls._get_default_params().get(optimizer_name, {}).copy()
        
        if custom_params:
            default_params.update(custom_params)
        
        try:
            optimizer_class = cls.OPTIMIZERS[optimizer_name]
            optimizer = optimizer_class(model_parameters, **default_params)
            
            info(f"Optimizer created successfully: {optimizer_name} with parameters {default_params}")
            return optimizer
            
        except Exception as e:
            error(f"Failed to create optimizer: {str(e)}")
            raise TypeError(f"Invalid parameters for optimizer: {str(e)}")

    @classmethod
    def create_from_config(cls, 
                          model_parameters, 
                          config: Dict[str, Any]) -> optim.Optimizer:
        if 'type' not in config:
            raise ValueError("Missing 'type' in optimizer configuration")
        
        optimizer_name = config['type'].lower()
        if 'params' in config and isinstance(config['params'], dict):
            custom_params = config.get('params', {})
        else:
            custom_params = {k: v for k, v in config.items() if k != 'type'}
        
        return cls.create_optimizer(optimizer_name, model_parameters, custom_params)

    @classmethod
    def _get_optimizer_description(cls, optimizer_name: str) -> str:
        desc_key = f"optimizers.{optimizer_name}_desc"
        return ""

    @classmethod
    def _get_optimizer_properties(cls, optimizer_name: str) -> Dict[str, Any]:
        try:
            cfg_props = SETTINGS['optimizers']['properties']
        except Exception:
            raise ValueError("optimizers.properties not found in config/config.json - check your config file")
        if not isinstance(cfg_props, dict) or optimizer_name not in cfg_props:
            raise ValueError(f"Optimizer properties for '{optimizer_name}' must be defined in config.config.json under 'optimizers.properties'.")
        return cfg_props[optimizer_name]


def get_optimizer_for_model(model: Module, 
                           optimizer_config: Optional[Dict[str, Any]] = None,
                           use_case: str = 'general') -> optim.Optimizer:
    if optimizer_config is None:
        try:
            training_config = SETTINGS['training']
        except Exception:
            raise ValueError("Missing required 'training' section in config/config.json")
        cfg = training_config.get('optimizer')
        if isinstance(cfg, dict) and cfg:
            optimizer_config = cfg
        else:
            if 'optimizer_type' not in training_config:
                raise ValueError("Missing required 'training.optimizer_type' in config/config.json")
            optimizer_type = training_config['optimizer_type']
            if optimizer_type:
                optimizer_params_source = training_config.get('optimizer_params', {})
                optimizer_params = {}
                if isinstance(optimizer_params_source, dict):
                    selected_params = optimizer_params_source.get(optimizer_type, {})
                    if isinstance(selected_params, dict):
                        optimizer_params.update(selected_params)
                lr_override = training_config.get('optimizer_lr', training_config.get('learning_rate'))
                if isinstance(lr_override, (int, float)):
                    optimizer_params['lr'] = lr_override
                wd_override = training_config.get('optimizer_weight_decay', training_config.get('weight_decay'))
                if isinstance(wd_override, (int, float)):
                    optimizer_params['weight_decay'] = wd_override
                optimizer_config = {'type': optimizer_type, 'params': optimizer_params}
            else:
                raise ValueError("No optimizer configuration found. Add 'training.optimizer' or 'training.optimizer_type' to config/config.json")
    
    return OptimizerFactory.create_from_config(model.parameters(), optimizer_config)


def list_all_optimizers() -> Dict[str, Any]:
    result = {}
    for name in OptimizerFactory.get_available_optimizers():
        result[name] = OptimizerFactory.get_optimizer_info(name)
    return result


def get_optimizer_recommendations_for_task(task_type: str) -> Dict[str, Any]:
    return {"recommended": [], "description": "Optimizer recommendations have been removed from the system"}