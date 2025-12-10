"""ConvNeXt module for Hootsight.

Provides comprehensive ConvNeXt model support including training, evaluation,
and model-specific utilities for image classification tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from typing import Dict, Any, Optional, Tuple, List, Callable, cast
import os
from pathlib import Path

from system.log import info, success, warning, error
from system.coordinator_settings import SETTINGS
from system.device import get_device, get_device_type, create_grad_scaler, autocast_context
from system.common.training_metrics import build_step_metrics, build_epoch_result


class ConvNeXtModel:
    """ConvNeXt model wrapper with training and evaluation capabilities."""

    def __init__(self, model_name: str = 'convnext_tiny', num_classes: int = 10, pretrained: bool = True, task: str = 'classification'):
        """Initialize ConvNeXt model.

        Args:
            model_name: Name of ConvNeXt variant ('convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            task: Task type ('classification', 'multi_label')
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
        """Create ConvNeXt model based on model_name and task."""
        if self.task in ('classification', 'multi_label'):
            weights_map = {
                'convnext_tiny': models.ConvNeXt_Tiny_Weights.DEFAULT,
                'convnext_small': models.ConvNeXt_Small_Weights.DEFAULT,
                'convnext_base': models.ConvNeXt_Base_Weights.DEFAULT,
                'convnext_large': models.ConvNeXt_Large_Weights.DEFAULT
            }
            
            model_map = {
                'convnext_tiny': models.convnext_tiny,
                'convnext_small': models.convnext_small,
                'convnext_base': models.convnext_base,
                'convnext_large': models.convnext_large
            }

            if self.model_name not in model_map:
                raise ValueError(f"Unsupported ConvNeXt variant: {self.model_name}")

            weights = weights_map.get(self.model_name) if self.pretrained else None
            model = model_map[self.model_name](weights=weights)

            # Modify final classifier for custom number of classes
            # ConvNeXt uses a Sequential classifier where the last layer (index 2) is Linear
            # (classifier): Sequential(
            #   (0): LayerNorm2d((768,), eps=1e-06, elementwise_affine=True)
            #   (1): Flatten(start_dim=1, end_dim=-1)
            #   (2): Linear(in_features=768, out_features=1000, bias=True)
            # )
            if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
                last_layer_idx = len(model.classifier) - 1
                if isinstance(model.classifier[last_layer_idx], nn.Linear):
                    in_features = cast(nn.Linear, model.classifier[last_layer_idx]).in_features
                    model.classifier[last_layer_idx] = nn.Linear(in_features, self.num_classes)
                else:
                     # Fallback if structure is different than expected
                     warning(f"Unexpected ConvNeXt classifier structure, attempting to replace last layer.")
                     in_features = cast(nn.Linear, model.classifier[last_layer_idx]).in_features
                     model.classifier[last_layer_idx] = nn.Linear(in_features, self.num_classes)
            else:
                raise ValueError(f"Could not find classifier head in {self.model_name}")

        else:
            raise ValueError(f"Unsupported task: {self.task}. ConvNeXt currently supports 'classification' and 'multi_label'.")

        return model

    def get_optimizer(self, lr: float = 0.001, weight_decay: float = 1e-4) -> optim.Optimizer:
        """Get AdamW optimizer configured for ConvNeXt."""
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
        
        # ConvNeXt typically benefits from AdamW
        return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def get_scheduler(self, optimizer: optim.Optimizer, step_size: int = 7, gamma: float = 0.1) -> optim.lr_scheduler.StepLR:
        """Get StepLR scheduler configured for ConvNeXt."""
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

    def get_criterion(self) -> Optional[nn.Module]:
        """Get loss criterion based on task."""
        # Try to get from config first
        try:
            training_config = SETTINGS['training']
            loss_type = training_config.get('loss_type')
            if loss_type:
                from system.losses import LossFactory
                loss_params = training_config.get('loss_params', {})
                reduction = training_config.get('loss_reduction')
                if reduction:
                    loss_params['reduction'] = reduction
                
                return LossFactory.create_loss(loss_type, loss_params)
        except Exception:
            pass

        if self.task == 'classification':
            return nn.CrossEntropyLoss()
        elif self.task == 'multi_label':
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported task for criterion: {self.task}")

    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer,
                   criterion: Optional[nn.Module], scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                   progress: Optional[Callable[[str, int, int, int, Dict[str, Any]], None]] = None,
                   epoch_index: Optional[int] = None, should_stop: Optional[Callable[[], bool]] = None) -> Dict[str, float]:
        """Train model for one epoch.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            scheduler: Learning rate scheduler (optional)

        Returns:
            dict: Training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        steps_total = len(train_loader)
        steps_completed = 0

        for step_idx, batch in enumerate(train_loader, start=1):
            if should_stop and should_stop():
                break
            
            inputs, targets = batch
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
                    if criterion is not None:
                        if self.task == 'multi_label':
                             loss = criterion(outputs, targets.float())
                        else:
                             loss = criterion(outputs, targets)
                    else:
                        raise ValueError("Criterion required")
                self._scaler.scale(loss).backward()
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                outputs = self.model(inputs)
                if criterion is not None:
                    if self.task == 'multi_label':
                         loss = criterion(outputs, targets.float())
                    else:
                         loss = criterion(outputs, targets)
                else:
                    raise ValueError("Criterion required")
                loss.backward()
                optimizer.step()

            loss_value = float(loss.item())
            total_loss += loss_value
            
            if self.task == 'classification':
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            if progress:
                try:
                    running_loss = total_loss / step_idx
                    accuracy = 100.0 * correct / total if total > 0 else 0.0
                    metrics_payload = build_step_metrics(
                        loss=loss_value,
                        running_loss=running_loss,
                        phase='train',
                        optimizer=optimizer,
                        accuracy=accuracy if self.task == 'classification' else None
                    )
                    progress('train', epoch_index or 0, step_idx, steps_total, metrics_payload)
                except Exception:
                    pass

            steps_completed = step_idx

        if scheduler:
            scheduler.step()

        avg_loss = total_loss / steps_completed if steps_completed else 0.0

        if self.task == 'classification':
            accuracy = 100. * correct / total if total else 0.0
            return build_epoch_result(avg_loss, 'train', optimizer, accuracy)
        else:
            return build_epoch_result(avg_loss, 'train', optimizer)

    def validate(self, val_loader: DataLoader, criterion: Optional[nn.Module],
                 progress: Optional[Callable[[str, int, int, int, Dict[str, Any]], None]] = None,
                 epoch_index: Optional[int] = None, should_stop: Optional[Callable[[], bool]] = None) -> Dict[str, float]:
        """Validate model on validation set.

        Args:
            val_loader: Validation data loader
            criterion: Loss function
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        steps_total = len(val_loader)
        steps_completed = 0

        with torch.no_grad():
            for step_idx, batch in enumerate(val_loader, start=1):
                if should_stop and should_stop():
                    break
                
                inputs, targets = batch
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
                        if criterion is not None:
                            if self.task == 'multi_label':
                                loss = criterion(outputs, targets.float())
                            else:
                                loss = criterion(outputs, targets)
                        else:
                            raise ValueError("Criterion required")
                else:
                    outputs = self.model(inputs)
                    if criterion is not None:
                        if self.task == 'multi_label':
                            loss = criterion(outputs, targets.float())
                        else:
                            loss = criterion(outputs, targets)
                    else:
                        raise ValueError("Criterion required")

                loss_value = float(loss.item())
                total_loss += loss_value

                if self.task == 'classification':
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                if progress:
                    try:
                        running_loss = total_loss / step_idx
                        accuracy = 100.0 * correct / total if total > 0 else 0.0
                        metrics_payload = build_step_metrics(
                            loss=loss_value,
                            running_loss=running_loss,
                            phase='val',
                            optimizer=None,
                            accuracy=accuracy if self.task == 'classification' else None
                        )
                        progress('val', epoch_index or 0, step_idx, steps_total, metrics_payload)
                    except Exception:
                        pass
                
                steps_completed = step_idx

        avg_loss = total_loss / steps_completed if steps_completed else 0.0

        if self.task == 'classification':
            accuracy = 100. * correct / total if total else 0.0
            return build_epoch_result(avg_loss, 'val', None, accuracy)
        else:
            return build_epoch_result(avg_loss, 'val', None)

    def save_checkpoint(self, path: str, epoch: int, optimizer: optim.Optimizer,
                       scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                       metrics: Optional[Dict[str, float]] = None,
                       labels: Optional[Dict[int, str]] = None) -> None:
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
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Fallback for older checkpoints or raw weights
            self.model.load_state_dict(checkpoint)
            
        epoch = checkpoint.get('epoch', 0)
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


def create_convnext_model(model_name: str = 'convnext_tiny', num_classes: int = 10, pretrained: bool = True, task: str = 'classification') -> ConvNeXtModel:
    """Factory function to create ConvNeXt model."""
    return ConvNeXtModel(model_name, num_classes, pretrained, task)
