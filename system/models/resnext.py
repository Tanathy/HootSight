"""ResNeXt module for Hootsight.

Provides comprehensive ResNeXt model support including training, evaluation,
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
from system.device import get_device, get_device_type, create_grad_scaler, autocast_context
from system.common.training_metrics import build_step_metrics, build_epoch_result


class ResNeXtModel:
    """ResNeXt model wrapper with training and evaluation capabilities."""

    def __init__(self, model_name: str = 'resnext50_32x4d', num_classes: int = 10, pretrained: bool = True, task: str = 'classification'):
        """Initialize ResNeXt model.

        Args:
            model_name: Name of ResNeXt variant ('resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            task: Task type ('classification', 'detection', 'segmentation')
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.task = task
        self.device = get_device()
        self._device_type = get_device_type()

        # Runtime performance settings - require presence in config
        try:
            runtime_cfg = SETTINGS['training']['runtime']
        except Exception:
            raise ValueError("Missing required 'training.runtime' configuration in config/config.json")
        self.use_amp = bool(runtime_cfg['mixed_precision'])
        self.channels_last = bool(runtime_cfg['channels_last'])
        self._scaler = create_grad_scaler(self.use_amp and self._device_type == 'cuda')

        # Initialize model
        self.model = self._create_model()
        self.model = self.model.to(self.device)

        info(f"Initialized {model_name} model for {task} with {num_classes} classes on {self.device}")

    def _create_model(self) -> nn.Module:
        """Create ResNeXt model based on model_name and task."""
        if self.task in ('classification', 'multi_label'):
            # Map model names to their weight classes for the new torchvision API
            weights_map = {
                'resnext50_32x4d': models.ResNeXt50_32X4D_Weights.DEFAULT,
                'resnext101_32x8d': models.ResNeXt101_32X8D_Weights.DEFAULT,
                'resnext101_64x4d': models.ResNeXt101_64X4D_Weights.DEFAULT
            }
            
            model_map = {
                'resnext50_32x4d': models.resnext50_32x4d,
                'resnext101_32x8d': models.resnext101_32x8d,
                'resnext101_64x4d': models.resnext101_64x4d
            }

            if self.model_name not in model_map:
                raise ValueError(f"Unsupported ResNeXt variant: {self.model_name}")

            weights = weights_map.get(self.model_name) if self.pretrained else None
            model = model_map[self.model_name](weights=weights)

            # Modify final layer to match num_classes for both tasks
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)

        else:
            raise ValueError(f"ResNeXt currently only supports classification task, got: {self.task}")

        return model

    def get_optimizer(self, lr: float = 0.001, weight_decay: float = 1e-4) -> optim.Optimizer:
        """Get AdamW optimizer configured for ResNeXt."""
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
        """Get StepLR scheduler configured for ResNeXt."""
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
        """Get criterion based on task for ResNeXt."""
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
            if self.use_amp and self._scaler is not None:
                with autocast_context(True):
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets.float() if self.task == 'multi_label' else targets)
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
                    running_acc = 100.0 * correct / total if self.task != 'multi_label' and total > 0 else None
                    metrics_payload = build_step_metrics(
                        loss=loss_value,
                        running_loss=running_loss,
                        phase='train',
                        optimizer=optimizer,
                        accuracy=running_acc
                    )
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
            return build_epoch_result(avg_loss=avg_loss, phase='train', optimizer=optimizer)
        accuracy = 100. * correct / total if total > 0 else 0.0
        return build_epoch_result(avg_loss=avg_loss, phase='train', optimizer=optimizer, accuracy=accuracy)

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

                if self.use_amp and self._scaler is not None:
                    with autocast_context(True):
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
                        running_acc = 100.0 * correct / total if self.task != 'multi_label' and total > 0 else None
                        metrics_payload = build_step_metrics(
                            loss=loss_value,
                            running_loss=running_loss,
                            phase='val',
                            accuracy=running_acc
                        )
                        progress('val', epoch_index or 0, step_idx, steps_total, metrics_payload)
                    except Exception:
                        pass
                steps_completed = step_idx
                if should_stop and should_stop():
                    break

        avg_loss = total_loss / steps_completed if steps_completed else 0.0
        if self.task == 'multi_label':
            return build_epoch_result(avg_loss=avg_loss, phase='val')
        accuracy = 100. * correct / total if total > 0 else 0.0
        return build_epoch_result(avg_loss=avg_loss, phase='val', accuracy=accuracy)

    def save_checkpoint(self, path: str, epoch: int, optimizer: optim.Optimizer,
                       scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                       metrics: Optional[Dict[str, float]] = None,
                       labels: Optional[List[str]] = None) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            optimizer: Optimizer state
            scheduler: Scheduler state (optional)
            metrics: Training metrics (optional)
            labels: Class labels as dict {index: name} for deterministic mapping
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'labels': labels or {}
        }

        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if metrics:
            checkpoint['metrics'] = metrics

        torch.save(checkpoint, path)
        info(f"Checkpoint saved to {path} with {len(labels or {})} labels")

    def load_checkpoint(self, path: str) -> Tuple[int, Dict[str, Any]]:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file

        Returns:
            tuple: (epoch, checkpoint_data)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)

        info(f"Checkpoint loaded from {path} (epoch {epoch})")
        return epoch, checkpoint

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


def create_resnext_model(model_name: str = 'resnext50_32x4d', num_classes: int = 10, pretrained: bool = True, task: str = 'classification') -> ResNeXtModel:
    """Factory function to create ResNeXt model.

    Args:
        model_name: Name of ResNeXt variant
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        task: Task type ('classification')

    Returns:
        ResNeXtModel: Configured ResNeXt model instance
    """
    return ResNeXtModel(model_name, num_classes, pretrained, task)


def get_supported_resnext_variants() -> List[str]:
    """Get list of supported ResNeXt variants."""
    return ['resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d']


def get_resnext_config(model_name: str) -> Dict[str, Any]:
    """Get default configuration for ResNeXt variant.

    Args:
        model_name: Name of ResNeXt variant

    Returns:
        dict: Default configuration
    """
    # Get from config first
    try:
        models_config = SETTINGS['models']['resnext']['variants']
    except Exception:
        raise ValueError("Missing required 'models.resnext.variants' in config/config.json")
    if model_name in models_config:
        return models_config[model_name]
    # No baked-in fallback; require config to contain the variant
    configs = {
        'resnext50_32x4d': {'params': 25.0e6, 'recommended_batch_size': 32},
        'resnext101_32x8d': {'params': 88.8e6, 'recommended_batch_size': 16},
        'resnext101_64x4d': {'params': 83.5e6, 'recommended_batch_size': 16}
    }

    raise ValueError(f"Unsupported ResNeXt variant (not found in config): {model_name}")