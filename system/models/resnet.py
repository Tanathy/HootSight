"""ResNet module for Hootsight.

Provides comprehensive ResNet model support including training, evaluation,
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


class ResNetModel:
    """ResNet model wrapper with training and evaluation capabilities."""

    def __init__(self, model_name: str = 'resnet50', num_classes: int = 10, pretrained: bool = True, task: str = 'classification'):
        """Initialize ResNet model.

        Args:
            model_name: Name of ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            task: Task type ('classification', 'detection', 'segmentation')
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.task = task
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Runtime performance settings
        self.use_amp = SETTINGS.get('training.runtime.mixed_precision', True)
        self.channels_last = SETTINGS.get('training.runtime.channels_last', True)
        self._scaler = torch.cuda.amp.GradScaler() if self.use_amp and self.device.type == 'cuda' else None

        # Initialize model
        self.model = self._create_model()
        self.model = self.model.to(self.device)

        info(f"Initialized {model_name} model for {task} with {num_classes} classes on {self.device}")

    def _create_model(self) -> nn.Module:
        """Create ResNet model based on model_name and task."""
        if self.task == 'classification':
            model_map = {
                'resnet18': models.resnet18,
                'resnet34': models.resnet34,
                'resnet50': models.resnet50,
                'resnet101': models.resnet101,
                'resnet152': models.resnet152
            }

            if self.model_name not in model_map:
                raise ValueError(f"Unsupported ResNet variant: {self.model_name}")

            model = model_map[self.model_name](pretrained=self.pretrained)

            # Modify final layer for custom number of classes
            if self.num_classes != 1000:  # Default ImageNet classes
                num_features = model.fc.in_features
                model.fc = nn.Linear(num_features, self.num_classes)

        elif self.task == 'detection':
            if self.model_name != 'resnet50':
                warning(f"Detection currently optimized for resnet50, using {self.model_name}")
            # For detection, use pretrained weights but adjust for custom classes
            if self.pretrained:
                model = models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
                # Replace the classifier head for custom number of classes
                in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
                model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, self.num_classes)
            else:
                model = models.detection.fasterrcnn_resnet50_fpn(num_classes=self.num_classes)

        elif self.task == 'segmentation':
            if self.model_name != 'resnet50':
                warning(f"Segmentation currently optimized for resnet50, using {self.model_name}")
            # For segmentation, use pretrained weights but adjust for custom classes
            if self.pretrained:
                model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
                # Replace the classifier for custom number of classes
                model.classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
                model.aux_classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
            else:
                model = models.segmentation.deeplabv3_resnet50(num_classes=self.num_classes)

        elif self.task == 'multi_label':
            # Multi-label classification: use standard ResNet but with sigmoid output
            model_map = {
                'resnet18': models.resnet18,
                'resnet34': models.resnet34,
                'resnet50': models.resnet50,
                'resnet101': models.resnet101,
                'resnet152': models.resnet152
            }

            if self.model_name not in model_map:
                raise ValueError(f"Unsupported ResNet variant for multi-label: {self.model_name}")

            model = model_map[self.model_name](pretrained=self.pretrained)
            # For multi-label, ensure classifier outputs num_classes logits
            if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, self.num_classes)

        else:
            raise ValueError(f"Unsupported task: {self.task}")

        return model

    def get_optimizer(self, lr: float = 0.001, weight_decay: float = 1e-4) -> optim.Optimizer:
        """Get AdamW optimizer configured for ResNet."""
        # Try to get from config first
        training_config = SETTINGS.get('training', {})
        optimizer_config = training_config.get('optimizer', {})
        if optimizer_config.get('type') == 'adamw':
            params = optimizer_config.get('params', {})
            lr = params.get('lr', lr)
            weight_decay = params.get('weight_decay', weight_decay)
        
        return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def get_scheduler(self, optimizer: optim.Optimizer, step_size: int = 7, gamma: float = 0.1) -> optim.lr_scheduler.StepLR:
        """Get StepLR scheduler configured for ResNet."""
        # Try to get from config first
        training_config = SETTINGS.get('training', {})
        scheduler_config = training_config.get('scheduler', {})
        if scheduler_config.get('type') == 'step_lr':
            params = scheduler_config.get('params', {})
            step_size = params.get('step_size', step_size)
            gamma = params.get('gamma', gamma)
        
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    def get_criterion(self) -> Optional[nn.Module]:
        """Get loss criterion based on task."""
        if self.task == 'classification':
            return nn.CrossEntropyLoss()
        elif self.task == 'multi_label':
            return nn.BCEWithLogitsLoss()
        elif self.task == 'detection':
            # Detection models have built-in loss, return None or dummy
            return None
        elif self.task == 'segmentation':
            return nn.CrossEntropyLoss()
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
            criterion: Loss function (None for detection)
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
            if self.task == 'classification':
                inputs, targets = batch
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
                        if criterion is not None:
                            loss = criterion(outputs, targets)
                        else:
                            raise ValueError("Criterion required for classification")
                    assert self._scaler is not None
                    self._scaler.scale(loss).backward()
                    self._scaler.step(optimizer)
                    self._scaler.update()
                else:
                    outputs = self.model(inputs)
                    if criterion is not None:
                        loss = criterion(outputs, targets)
                    else:
                        raise ValueError("Criterion required for classification")
                    loss.backward()
                    optimizer.step()

                loss_value = float(loss.item())
                total_loss += loss_value
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                if progress:
                    try:
                        running_loss = total_loss / step_idx
                        running_acc = 100.0 * correct / total
                        metrics_payload: Dict[str, Any] = {
                            'loss': loss_value,
                            'step_loss': loss_value,
                            'epoch_loss': running_loss,
                            'train_loss': running_loss,
                            'epoch_accuracy': running_acc,
                            'train_accuracy': running_acc
                        }
                        progress('train', epoch_index or 0, step_idx, steps_total, metrics_payload)
                    except Exception:
                        pass

            elif self.task == 'detection':
                images, targets = batch
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses = torch.tensor(losses) if not isinstance(losses, torch.Tensor) else losses
                losses.backward()
                optimizer.step()

                loss_value = float(losses.detach().item() if isinstance(losses, torch.Tensor) else float(losses))
                total_loss += loss_value
                if progress:
                    try:
                        running_loss = total_loss / step_idx
                        metrics_payload = {
                            'loss': loss_value,
                            'step_loss': loss_value,
                            'epoch_loss': running_loss,
                            'train_loss': running_loss
                        }
                        progress('train', epoch_index or 0, step_idx, steps_total, metrics_payload)
                    except Exception:
                        pass

            elif self.task == 'segmentation':
                inputs, targets = batch
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
                        outputs = self.model(inputs)['out']
                        if criterion is not None:
                            loss = criterion(outputs, targets)
                        else:
                            raise ValueError("Criterion required for segmentation")
                    assert self._scaler is not None
                    self._scaler.scale(loss).backward()
                    self._scaler.step(optimizer)
                    self._scaler.update()
                else:
                    outputs = self.model(inputs)['out']
                    if criterion is not None:
                        loss = criterion(outputs, targets)
                    else:
                        raise ValueError("Criterion required for segmentation")
                    loss.backward()
                    optimizer.step()

                loss_value = float(loss.item())
                total_loss += loss_value
                if progress:
                    try:
                        running_loss = total_loss / step_idx
                        metrics_payload = {
                            'loss': loss_value,
                            'step_loss': loss_value,
                            'epoch_loss': running_loss,
                            'train_loss': running_loss
                        }
                        progress('train', epoch_index or 0, step_idx, steps_total, metrics_payload)
                    except Exception:
                        pass

            elif self.task == 'multi_label':
                inputs, targets = batch
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
                        if criterion is not None:
                            loss = criterion(outputs, targets.float())  # BCEWithLogitsLoss expects float targets
                        else:
                            raise ValueError("Criterion required for multi-label")
                    assert self._scaler is not None
                    self._scaler.scale(loss).backward()
                    self._scaler.step(optimizer)
                    self._scaler.update()
                else:
                    outputs = self.model(inputs)
                    if criterion is not None:
                        loss = criterion(outputs, targets.float())
                    else:
                        raise ValueError("Criterion required for multi-label")
                    loss.backward()
                    optimizer.step()

                loss_value = float(loss.item())
                total_loss += loss_value
                # For multi-label, we don't calculate accuracy the same way
                # Could add F1, precision, recall metrics here if needed
                if progress:
                    try:
                        running_loss = total_loss / step_idx
                        metrics_payload = {
                            'loss': loss_value,
                            'step_loss': loss_value,
                            'epoch_loss': running_loss,
                            'train_loss': running_loss
                        }
                        progress('train', epoch_index or 0, step_idx, steps_total, metrics_payload)
                    except Exception:
                        pass

            steps_completed = step_idx
            if should_stop and should_stop():
                break

        if scheduler:
            scheduler.step()

        avg_loss = total_loss / steps_completed if steps_completed else 0.0

        if self.task == 'classification':
            accuracy = 100. * correct / total if total else 0.0
            return {
                'train_loss': avg_loss,
                'train_accuracy': accuracy
            }
        elif self.task == 'multi_label':
            # Multi-label accuracy is calculated differently, return only loss
            return {
                'train_loss': avg_loss
            }
        else:
            return {
                'train_loss': avg_loss
            }

    def validate(self, val_loader: DataLoader, criterion: Optional[nn.Module],
                 progress: Optional[Callable[[str, int, int, int, Dict[str, Any]], None]] = None,
                 epoch_index: Optional[int] = None, should_stop: Optional[Callable[[], bool]] = None) -> Dict[str, float]:
        """Validate model on validation set.

        Args:
            val_loader: Validation data loader
            criterion: Loss function (None for detection)

        Returns:
            dict: Validation metrics
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
                if self.task == 'classification':
                    inputs, targets = batch
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
                            loss = criterion(outputs, targets) if criterion is not None else None
                    else:
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets) if criterion is not None else None
                    if loss is None:
                        raise ValueError("Criterion required for classification")

                    loss_value = float(loss.item())
                    total_loss += loss_value
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    if progress:
                        try:
                            running_loss = total_loss / step_idx
                            running_acc = 100.0 * correct / total
                            metrics_payload: Dict[str, Any] = {
                                'loss': loss_value,
                                'step_loss': loss_value,
                                'epoch_loss': running_loss,
                                'val_loss': running_loss,
                                'epoch_accuracy': running_acc,
                                'val_accuracy': running_acc
                            }
                            progress('val', epoch_index or 0, step_idx, steps_total, metrics_payload)
                        except Exception:
                            pass

                elif self.task == 'detection':
                    images, targets = batch
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    losses = torch.tensor(losses) if not isinstance(losses, torch.Tensor) else losses
                    loss_value = float(losses.detach().item() if isinstance(losses, torch.Tensor) else float(losses))
                    total_loss += loss_value
                    if progress:
                        try:
                            running_loss = total_loss / step_idx
                            metrics_payload = {
                                'loss': loss_value,
                                'step_loss': loss_value,
                                'epoch_loss': running_loss,
                                'val_loss': running_loss
                            }
                            progress('val', epoch_index or 0, step_idx, steps_total, metrics_payload)
                        except Exception:
                            pass

                elif self.task == 'segmentation':
                    inputs, targets = batch
                    inputs = inputs.to(self.device, non_blocking=True)
                    if self.channels_last:
                        try:
                            inputs = inputs.to(memory_format=torch.channels_last)
                        except Exception:
                            pass
                    targets = targets.to(self.device, non_blocking=True)

                    if self.use_amp and self.device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            outputs = self.model(inputs)['out']
                            loss = criterion(outputs, targets) if criterion is not None else None
                    else:
                        outputs = self.model(inputs)['out']
                        loss = criterion(outputs, targets) if criterion is not None else None
                    if loss is None:
                        raise ValueError("Criterion required for segmentation")

                    loss_value = float(loss.item())
                    total_loss += loss_value
                    if progress:
                        try:
                            running_loss = total_loss / step_idx
                            metrics_payload = {
                                'loss': loss_value,
                                'step_loss': loss_value,
                                'epoch_loss': running_loss,
                                'val_loss': running_loss
                            }
                            progress('val', epoch_index or 0, step_idx, steps_total, metrics_payload)
                        except Exception:
                            pass

                elif self.task == 'multi_label':
                    inputs, targets = batch
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
                            loss = criterion(outputs, targets.float()) if criterion is not None else None
                    else:
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets.float()) if criterion is not None else None
                    if loss is None:
                        raise ValueError("Criterion required for multi-label")

                    loss_value = float(loss.item())
                    total_loss += loss_value
                    # For multi-label validation, could compute F1, precision, recall per class
                    if progress:
                        try:
                            running_loss = total_loss / step_idx
                            metrics_payload = {
                                'loss': loss_value,
                                'step_loss': loss_value,
                                'epoch_loss': running_loss,
                                'val_loss': running_loss
                            }
                            progress('val', epoch_index or 0, step_idx, steps_total, metrics_payload)
                        except Exception:
                            pass

                    steps_completed = step_idx
                    if should_stop and should_stop():
                        break

        avg_loss = total_loss / steps_completed if steps_completed else 0.0

        if self.task == 'classification':
            accuracy = 100. * correct / total if total else 0.0
            return {
                'val_loss': avg_loss,
                'val_accuracy': accuracy
            }
        elif self.task == 'multi_label':
            # For multi-label, return loss only (could add more metrics)
            return {
                'val_loss': avg_loss
            }
        else:
            return {
                'val_loss': avg_loss
            }

    def save_checkpoint(self, path: str, epoch: int, optimizer: optim.Optimizer,
                       scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                       metrics: Optional[Dict[str, float]] = None) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            optimizer: Optimizer state
            scheduler: Scheduler state (optional)
            metrics: Training metrics (optional)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes
        }

        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if metrics:
            checkpoint['metrics'] = metrics

        torch.save(checkpoint, path)
        info(f"Checkpoint saved to {path}")

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


def create_resnet_model(model_name: str = 'resnet50', num_classes: int = 10, pretrained: bool = True, task: str = 'classification') -> ResNetModel:
    """Factory function to create ResNet model.

    Args:
        model_name: Name of ResNet variant
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        task: Task type ('classification', 'detection', 'segmentation')

    Returns:
        ResNetModel: Configured ResNet model instance
    """
    return ResNetModel(model_name, num_classes, pretrained, task)


def get_supported_resnet_variants() -> List[str]:
    """Get list of supported ResNet variants."""
    return ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def get_resnet_config(model_name: str) -> Dict[str, Any]:
    """Get default configuration for ResNet variant.

    Args:
        model_name: Name of ResNet variant

    Returns:
        dict: Default configuration
    """
    # Get from config first
    models_config = SETTINGS.get('models', {}).get('resnet', {}).get('variants', {})
    if model_name in models_config:
        return models_config[model_name]
    
    # Fallback to hardcoded values
    configs = {
        'resnet18': {'params': 11.7e6, 'recommended_batch_size': 64},
        'resnet34': {'params': 21.8e6, 'recommended_batch_size': 32},
        'resnet50': {'params': 25.6e6, 'recommended_batch_size': 32},
        'resnet101': {'params': 44.5e6, 'recommended_batch_size': 16},
        'resnet152': {'params': 60.2e6, 'recommended_batch_size': 8}
    }

    if model_name not in configs:
        raise ValueError(f"Unsupported ResNet variant: {model_name}")

    return configs[model_name]