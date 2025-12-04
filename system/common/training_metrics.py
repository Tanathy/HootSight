"""
Common training metrics payload builder.

Ensures consistent metric structure across all model implementations.
"""

from typing import Any, Dict, Optional
import torch.optim as optim


def build_step_metrics(
    loss: float,
    running_loss: float,
    phase: str,
    optimizer: Optional[optim.Optimizer] = None,
    accuracy: Optional[float] = None
) -> Dict[str, Any]:
    """
    Build consistent metrics payload for step-level progress reporting.
    
    Args:
        loss: Current step loss value
        running_loss: Running average loss for the epoch
        phase: 'train' or 'val'
        optimizer: Optimizer instance (for learning rate extraction, train phase only)
        accuracy: Running accuracy percentage (if applicable)
    
    Returns:
        Consistent metrics dictionary
    """
    metrics: Dict[str, Any] = {
        'loss': loss,
        'step_loss': loss,
        'epoch_loss': running_loss,
    }
    
    # Phase-specific loss key
    if phase == 'train':
        metrics['train_loss'] = running_loss
    else:
        metrics['val_loss'] = running_loss
    
    # Accuracy (if provided)
    if accuracy is not None:
        metrics['epoch_accuracy'] = accuracy
        metrics['step_accuracy'] = accuracy
        metrics['running_accuracy'] = accuracy
        if phase == 'train':
            metrics['train_accuracy'] = accuracy
            metrics['train_step_accuracy'] = accuracy
        else:
            metrics['val_accuracy'] = accuracy
            metrics['val_step_accuracy'] = accuracy
    
    # Learning rate (train phase only, requires optimizer)
    if phase == 'train' and optimizer is not None:
        try:
            metrics['learning_rate'] = optimizer.param_groups[0]['lr']
        except (IndexError, KeyError):
            pass
    
    return metrics


def build_epoch_result(
    avg_loss: float,
    phase: str,
    optimizer: Optional[optim.Optimizer] = None,
    accuracy: Optional[float] = None
) -> Dict[str, Any]:
    """
    Build consistent result dictionary for epoch completion.
    
    Args:
        avg_loss: Average loss for the epoch
        phase: 'train' or 'val'
        optimizer: Optimizer instance (for learning rate, train phase only)
        accuracy: Final accuracy percentage (if applicable)
    
    Returns:
        Consistent epoch result dictionary
    """
    result: Dict[str, Any] = {}
    
    if phase == 'train':
        result['train_loss'] = avg_loss
        if accuracy is not None:
            result['train_accuracy'] = accuracy
        if optimizer is not None:
            try:
                result['learning_rate'] = optimizer.param_groups[0]['lr']
            except (IndexError, KeyError):
                pass
    else:
        result['val_loss'] = avg_loss
        if accuracy is not None:
            result['val_accuracy'] = accuracy
    
    return result
