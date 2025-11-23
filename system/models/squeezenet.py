"""SqueezeNet module for Hootsight.

Provides comprehensive SqueezeNet model support including training, evaluation,
and model-specific utilities for image classification tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from typing import Dict, Any, Optional, Tuple, List, Callable
import os
from pathlib import Path

from system.log import info, success, warning, error
from system.coordinator_settings import SETTINGS


class SqueezeNetModel:
    """SqueezeNet model wrapper with training and evaluation capabilities."""

    def __init__(self, model_name: str = 'squeezenet1_0', num_classes: int = 10, pretrained: bool = True, task: str = 'classification'):
        """Initialize SqueezeNet model.

        Args:
            model_name: Name of SqueezeNet variant ('squeezenet1_0', 'squeezenet1_1')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            task: Task type ('classification', 'detection', 'segmentation')
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.task = task
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Runtime performance settings - require presence in config
        try:
            runtime_cfg = SETTINGS['training']['runtime']
        except Exception:
            raise ValueError("Missing required 'training.runtime' configuration in config/config.json")
        self.use_amp = bool(runtime_cfg['mixed_precision'])
        self.channels_last = bool(runtime_cfg['channels_last'])
        self._scaler = torch.cuda.amp.GradScaler() if self.use_amp and self.device.type == 'cuda' else None

        # Initialize model
        self.model = self._create_model()
        self.model = self.model.to(self.device)

        info(f"Initialized {model_name} model for {task} with {num_classes} classes on {self.device}")

    def _create_model(self) -> nn.Module:
        """Create SqueezeNet model based on model_name and task."""
        if self.task in ('classification', 'multi_label'):
            model_map = {
                'squeezenet1_0': models.squeezenet1_0,
                'squeezenet1_1': models.squeezenet1_1
            }

            if self.model_name not in model_map:
                raise ValueError(f"Unsupported SqueezeNet variant: {self.model_name}")

            model = model_map[self.model_name](pretrained=self.pretrained)

            # Modify final classifier for custom number of classes
            # SqueezeNet uses Conv2d(512, 1000, kernel_size=(1, 1)) as final layer
            model.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1, 1), stride=(1, 1))

        else:
            raise ValueError(f"SqueezeNet currently only supports classification task, got: {self.task}")

        return model

    def get_optimizer(self, lr: float = 0.001, weight_decay: float = 1e-4) -> optim.Optimizer:
        """Get AdamW optimizer configured for SqueezeNet."""
        # Try to get from config first
        try:
            training_config = SETTINGS['training']
        except Exception:
            raise ValueError("Missing required 'training' section in config/config.json")
        optimizer_config = training_config.get('optimizer', {})
        if optimizer_config.get('type') == 'adamw':
            params = optimizer_config.get('params', {})
            lr = params.get('lr', lr)
            weight_decay = params.get('weight_decay', weight_decay)

        return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def get_scheduler(self, optimizer: optim.Optimizer, step_size: int = 7, gamma: float = 0.1) -> optim.lr_scheduler.StepLR:
        """Get StepLR scheduler configured for SqueezeNet."""
        # Try to get from config first
        try:
            training_config = SETTINGS['training']
        except Exception:
            raise ValueError("Missing required 'training' section in config/config.json")
        scheduler_config = training_config.get('scheduler', {})
        if scheduler_config.get('type') == 'step_lr':
            params = scheduler_config.get('params', {})
            step_size = params.get('step_size', step_size)
            gamma = params.get('gamma', gamma)

        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    def get_criterion(self) -> nn.Module:
        """Get criterion based on task for SqueezeNet."""
        if self.task == 'multi_label':
            return nn.BCEWithLogitsLoss()
        return nn.CrossEntropyLoss()

    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer,
                   criterion: nn.Module, scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                   progress: Optional[Callable[[str, int, int, int, Dict[str, Any]], None]] = None,
                   epoch_index: Optional[int] = None, should_stop: Optional[Callable[[], bool]] = None) -> Dict[str, float]:
        """Train model for one epoch.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            scheduler: Learning rate scheduler (optional)
            progress: Optional progress callback
            epoch_index: Current epoch index

        Returns:
            dict: Training metrics (loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        steps_total = len(train_loader)
        steps_completed = 0

        for step_idx, (inputs, targets) in enumerate(train_loader, start=1):
            if should_stop and should_stop():
                break
            inputs = inputs.to(self.device, non_blocking=True)
            if self.channels_last:
                try:
                    inputs = inputs.to(memory_format=torch.channels_last)
                except Exception:
                    pass
            targets = targets.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            if self.use_amp and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets.float() if self.task == 'multi_label' else targets)
                assert self._scaler is not None
                self._scaler.scale(loss).backward()
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets.float() if self.task == 'multi_label' else targets)
                loss.backward()
                optimizer.step()

            loss_value = float(loss.item())
            total_loss += loss_value
            if self.task != 'multi_label':
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            if progress:
                try:
                    running_loss = total_loss / step_idx if step_idx else loss_value
                    metrics_payload: Dict[str, Any] = {
                        'loss': loss_value,
                        'step_loss': loss_value,
                        'epoch_loss': running_loss,
                        'train_loss': running_loss
                    }
                    if self.task != 'multi_label' and total > 0:
                        running_acc = 100.0 * correct / total
                        metrics_payload['epoch_accuracy'] = running_acc
                        metrics_payload['train_accuracy'] = running_acc
                    progress('train', epoch_index or 0, step_idx, steps_total, metrics_payload)
                except Exception:
                    pass

            steps_completed = step_idx
            if should_stop and should_stop():
                break

        if scheduler:
            scheduler.step()

        avg_loss = total_loss / steps_completed if steps_completed else 0.0
        if self.task == 'multi_label':
            return {'train_loss': avg_loss}
        accuracy = 100. * correct / total if total > 0 else 0.0
        return {'train_loss': avg_loss, 'train_accuracy': accuracy}

    def validate(self, val_loader: DataLoader, criterion: nn.Module,
                progress: Optional[Callable[[str, int, int, int, Dict[str, Any]], None]] = None,
                epoch_index: Optional[int] = None, should_stop: Optional[Callable[[], bool]] = None) -> Dict[str, float]:
        """Validate model on validation set.

        Args:
            val_loader: Validation data loader
            criterion: Loss function
            progress: Optional progress callback
            epoch_index: Current epoch index

        Returns:
            dict: Validation metrics (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        steps_total = len(val_loader)
        steps_completed = 0

        with torch.no_grad():
            for step_idx, (inputs, targets) in enumerate(val_loader, start=1):
                if should_stop and should_stop():
                    break
                inputs = inputs.to(self.device, non_blocking=True)
                if self.channels_last:
                    try:
                        inputs = inputs.to(memory_format=torch.channels_last)
                    except Exception:
                        pass
                targets = targets.to(self.device, non_blocking=True)

                if self.use_amp and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets.float() if self.task == 'multi_label' else targets)
                else:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets.float() if self.task == 'multi_label' else targets)

                loss_value = float(loss.item())
                total_loss += loss_value
                if self.task != 'multi_label':
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                if progress:
                    try:
                        running_loss = total_loss / step_idx if step_idx else loss_value
                        metrics_payload: Dict[str, Any] = {
                            'loss': loss_value,
                            'step_loss': loss_value,
                            'epoch_loss': running_loss,
                            'val_loss': running_loss
                        }
                        if self.task != 'multi_label' and total > 0:
                            running_acc = 100.0 * correct / total
                            metrics_payload['epoch_accuracy'] = running_acc
                            metrics_payload['val_accuracy'] = running_acc
                        progress('val', epoch_index or 0, step_idx, steps_total, metrics_payload)
                    except Exception:
                        pass
                steps_completed = step_idx
                if should_stop and should_stop():
                    break

        avg_loss = total_loss / steps_completed if steps_completed else 0.0
        if self.task == 'multi_label':
            return {'val_loss': avg_loss}
        accuracy = 100. * correct / total if total > 0 else 0.0
        return {'val_loss': avg_loss, 'val_accuracy': accuracy}

    def save_checkpoint(self, path: str, epoch: int, best_accuracy: float):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            best_accuracy: Best validation accuracy achieved
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_accuracy': best_accuracy,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'task': self.task
        }
        torch.save(checkpoint, path)
        info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file

        Returns:
            dict: Checkpoint information
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        info(f"Checkpoint loaded from {path}")
        return checkpoint

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            dict: Model information including parameters count
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'task': self.task,
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


def create_squeezenet_model(model_name: str = 'squeezenet1_0', num_classes: int = 10, pretrained: bool = True, task: str = 'classification') -> SqueezeNetModel:
    """Create SqueezeNet model instance.

    Args:
        model_name: Name of SqueezeNet variant
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        task: Task type

    Returns:
        SqueezeNetModel: Initialized model instance
    """
    return SqueezeNetModel(model_name, num_classes, pretrained, task)


def get_supported_squeezenet_variants() -> List[str]:
    """Get list of supported SqueezeNet variants."""
    return ['squeezenet1_0', 'squeezenet1_1']


def get_squeezenet_config(model_name: str) -> Dict[str, Any]:
    """Get default configuration for SqueezeNet variant.

    Args:
        model_name: Name of SqueezeNet variant

    Returns:
        dict: Default configuration
    """
    # Get from config first
    try:
        models_config = SETTINGS['models']['squeezenet']['variants']
    except Exception:
        raise ValueError("Missing required 'models.squeezenet.variants' in config/config.json")
    if model_name in models_config:
        return models_config[model_name]
    configs = {
        'squeezenet1_0': {'params': 1.25e6, 'recommended_batch_size': 64},
        'squeezenet1_1': {'params': 1.24e6, 'recommended_batch_size': 64}
    }

    raise ValueError(f"Unsupported SqueezeNet variant (not found in config): {model_name}")