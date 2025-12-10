"""AlexNet module for Hootsight.

Provides AlexNet model support for classification and multi-label tasks.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from system.coordinator_settings import SETTINGS
from system.device import autocast_context, create_grad_scaler, get_device, get_device_type
from system.log import info, warning
from system.common.training_metrics import build_epoch_result, build_step_metrics
from system.losses import LossFactory


class AlexNetModel:
    """AlexNet model wrapper with training and evaluation capabilities."""

    def __init__(self, model_name: str = 'alexnet', num_classes: int = 10, pretrained: bool = True, task: str = 'classification'):
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.task = task
        self.device = get_device()
        self._device_type = get_device_type()

        try:
            runtime_cfg = SETTINGS['training']['runtime']
        except Exception:
            raise ValueError("Missing required 'training.runtime' configuration in config/config.json")
        self.use_amp = bool(runtime_cfg['mixed_precision'])
        self.channels_last = bool(runtime_cfg['channels_last'])
        self._scaler = create_grad_scaler(self.use_amp and self._device_type == 'cuda')

        self.model = self._create_model()
        self.model = self.model.to(self.device)

        info(f"Initialized {model_name} model for {task} with {num_classes} classes on {self.device}")

    def _create_model(self) -> nn.Module:
        if self.task not in ('classification', 'multi_label'):
            raise ValueError(f"AlexNet currently supports classification/multi_label tasks, got: {self.task}")

        weights_map = {'alexnet': models.AlexNet_Weights.DEFAULT}
        model_fn = models.alexnet

        if self.model_name not in weights_map:
            raise ValueError(f"Unsupported AlexNet variant: {self.model_name}")

        weights = weights_map[self.model_name] if self.pretrained else None
        model = model_fn(weights=weights)

        classifier = model.classifier[6]
        if isinstance(classifier, nn.Linear):
            model.classifier[6] = nn.Linear(classifier.in_features, self.num_classes)

        return model

    def get_optimizer(self, lr: float = 0.001, weight_decay: float = 1e-4) -> optim.Optimizer:
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
        """Get criterion based on config or task for AlexNet."""
        try:
            training_config = SETTINGS['training']
            loss_type = training_config.get('loss_type', 'cross_entropy')
            
            # Use the factory to create the loss function
            return LossFactory.create(loss_type, device=self.device)
        except Exception as e:
            # Fallback to default behavior if config fails
            if self.task == 'multi_label':
                return nn.BCEWithLogitsLoss()
            return nn.CrossEntropyLoss()

    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer,
                   criterion: nn.Module, scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                   progress: Optional[Callable[[str, int, int, int, Dict[str, Any]], None]] = None,
                   epoch_index: Optional[int] = None, should_stop: Optional[Callable[[], bool]] = None) -> Dict[str, float]:
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
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return build_epoch_result(avg_loss=avg_loss, phase='train', optimizer=optimizer, accuracy=accuracy)

    def validate(self, val_loader: DataLoader, criterion: nn.Module,
                 progress: Optional[Callable[[str, int, int, int, Dict[str, Any]], None]] = None,
                 epoch_index: Optional[int] = None, should_stop: Optional[Callable[[], bool]] = None) -> Dict[str, float]:
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
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return build_epoch_result(avg_loss=avg_loss, phase='val', accuracy=accuracy)

    def save_checkpoint(self, path: str, epoch: int, optimizer: optim.Optimizer,
                       scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                       metrics: Optional[Dict[str, float]] = None,
                       labels: Optional[Dict[int, str]] = None) -> None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'task': self.task,
            'labels': labels or {}
        }
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if metrics:
            checkpoint['metrics'] = metrics

        torch.save(checkpoint, path)
        info(f"Checkpoint saved to {path} with {len(labels or {})} labels")

    def load_checkpoint(self, path: str) -> Tuple[int, Dict[str, Any]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        info(f"Checkpoint loaded from {path} (epoch {epoch})")
        return epoch, checkpoint

    def get_model_info(self) -> Dict[str, Any]:
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


def create_alexnet_model(model_name: str = 'alexnet', num_classes: int = 10, pretrained: bool = True, task: str = 'classification') -> AlexNetModel:
    return AlexNetModel(model_name, num_classes, pretrained, task)


def get_supported_alexnet_variants() -> List[str]:
    return ['alexnet']


def get_alexnet_config(model_name: str) -> Dict[str, Any]:
    try:
        models_config = SETTINGS['models']['alexnet']['variants']
    except Exception:
        raise ValueError("Missing required 'models.alexnet.variants' in config/config.json")
    if model_name in models_config:
        return models_config[model_name]
    raise ValueError(f"Unsupported AlexNet variant (not found in config): {model_name}")
