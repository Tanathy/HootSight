"""Training module for Hootsight.

Handles the complete training process for models, including background execution,
progress tracking, and model checkpoint management.
"""

from typing import Dict, Any, Optional, Callable, List
import threading
import json
from pathlib import Path
import copy

from system.log import info, success, error
from system.coordinator import create_coordinator
from system.coordinator_settings import SETTINGS


class TrainingManager:
    """Manages training processes and their lifecycle."""

    def __init__(self):
        self.active_trainings: Dict[str, Dict[str, Any]] = {}
        # Live progress map: {id: {epoch, total_epochs, phase, step, total_steps}}
        self.progress: Dict[str, Dict[str, Any]] = {}
        self.progress_history: Dict[str, List[Dict[str, Any]]] = {}
        self._progress_updates: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def _reset_progress_tracking(self, training_id: str) -> None:
        with self._lock:
            self.progress[training_id] = {}
            self.progress_history[training_id] = []
            self._progress_updates[training_id] = []

    def _record_event(self, training_id: str, event: Dict[str, Any]) -> None:
        with self._lock:
            history = self.progress_history.setdefault(training_id, [])
            history.append(event)
            updates = self._progress_updates.setdefault(training_id, [])
            updates.append(event)

            if event.get('type') == 'step':
                self.progress[training_id] = {
                    'phase': event.get('phase'),
                    'epoch': event.get('epoch'),
                    'step': event.get('step'),
                    'total_steps': event.get('total_steps'),
                    'metrics': event.get('metrics', {})
                }
            elif event.get('type') == 'epoch':
                latest = self.progress.setdefault(training_id, {})
                latest.update({
                    'phase': event.get('phase'),
                    'epoch': event.get('epoch'),
                    'metrics': event.get('metrics', {})
                })

    def _record_step_event(self, training_id: str, phase: str, epoch: int,
                           step: int, total_steps: int, metrics: Dict[str, Any]) -> None:
        event = {
            'type': 'step',
            'phase': phase,
            'epoch': epoch,
            'step': step,
            'total_steps': total_steps,
            'metrics': metrics
        }
        self._record_event(training_id, event)

    def _record_epoch_summary(self, training_id: str, phase: str, epoch: int,
                               metrics: Dict[str, Any]) -> None:
        event = {
            'type': 'epoch',
            'phase': phase,
            'epoch': epoch,
            'metrics': metrics
        }
        self._record_event(training_id, event)

    def _consume_updates(self, training_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            pending = self._progress_updates.get(training_id, [])
            self._progress_updates[training_id] = []
        return copy.deepcopy(pending)

    def _get_history(self, training_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            history = self.progress_history.get(training_id, [])
        return copy.deepcopy(history)

    def get_training_history(self, training_id: Optional[str] = None) -> Dict[str, Any]:
        if training_id:
            return {
                "training_id": training_id,
                "events": self._get_history(training_id)
            }

        histories: Dict[str, List[Dict[str, Any]]] = {}
        with self._lock:
            training_ids = list(self.progress_history.keys())

        for tid in training_ids:
            histories[tid] = self._get_history(tid)

        return {
            "trainings": histories,
            "count": len(histories)
        }

    def start_training(self, project_name: str, model_type: str = "resnet",
                      model_name: str = "resnet50", epochs: Optional[int] = None,
                      callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Start a training process.

        Args:
            project_name: Name of the project to train on
            model_type: Type of model to use
            model_name: Specific model name
            epochs: Number of epochs to train
            callback: Optional callback function for progress updates

        Returns:
            dict: Training start status
        """
        try:
            # Create unique training ID
            training_id = f"{project_name}_{model_type}_{model_name}_{threading.current_thread().ident}"

            # Reset progress trackers for fresh run
            self._reset_progress_tracking(training_id)

            # Create coordinator
            coordinator = create_coordinator(model_type, model_name)

            # Prepare training
            training_config = coordinator.prepare_training(project_name)

            # Override epochs if provided
            if epochs:
                training_config['config']['epochs'] = epochs

            stop_event = threading.Event()

            # Start training in background thread
            training_thread = threading.Thread(
                target=self._run_training,
                args=(training_config, project_name, model_type, model_name, training_id, callback, stop_event),
                name=f"Training-{training_id}"
            )
            training_thread.daemon = True

            # Store training info
            self.active_trainings[training_id] = {
                'thread': training_thread,
                'config': training_config,
                'coordinator': coordinator,
                'start_time': threading.current_thread().ident,
                'status': 'starting',
                'stop_event': stop_event
            }

            training_thread.start()

            info(f"Training started for project {project_name} with ID: {training_id}")

            return {
                "started": True,
                "training_id": training_id,
                "project": project_name,
                "model_type": model_type,
                "model_name": model_name,
                "config": coordinator.get_training_summary(training_config),
                "message": f"Training started successfully for {project_name}"
            }

        except Exception as e:
            error(f"Failed to start training: {e}")
            return {
                "started": False,
                "error": str(e),
                "message": "Failed to start training"
            }

    def stop_training(self, training_id: str) -> Dict[str, Any]:
        """Stop a training process.

        Args:
            training_id: ID of the training to stop

        Returns:
            dict: Stop status
        """
        if training_id not in self.active_trainings:
            return {"stopped": False, "message": f"Training {training_id} not found"}

        training_info = self.active_trainings[training_id]
        training_info['status'] = 'stopping'
        stop_event = training_info.get('stop_event')
        if stop_event:
            stop_event.set()

        info(f"Training {training_id} stop signal dispatched")

        return {"stopped": True, "message": f"Training {training_id} stop initiated"}

    def get_training_status(self, training_id: Optional[str] = None) -> Dict[str, Any]:
        """Get training status.

        Args:
            training_id: Specific training ID, or None for all

        Returns:
            dict: Training status information
        """
        if training_id:
            if training_id not in self.active_trainings:
                return {"error": f"Training {training_id} not found"}

            training_info = self.active_trainings[training_id]
            # Robust project name extraction from path (handles Windows/Unix separators)
            ds_path = training_info['config']['config']['dataset_path'] or ""
            proj_name = Path(ds_path).parts[-2] if ds_path else "unknown"
            prog = self.progress.get(training_id, {})
            updates = self._consume_updates(training_id)
            # Derive helpful runtime stats
            try:
                train_loader = training_info['config']['train_loader']
                val_loader = training_info['config']['val_loader']
                train_steps = len(train_loader)
                val_steps = len(val_loader)
                train_samples = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else None
                val_samples = len(val_loader.dataset) if hasattr(val_loader, 'dataset') else None
            except Exception:
                train_steps = None
                val_steps = None
                train_samples = None
                val_samples = None
            batch_size = training_info['config']['config'].get('batch_size')
            latest_metrics = prog.get('metrics', {})

            return {
                "training_id": training_id,
                "status": training_info['status'],
                "project": proj_name,
                "model_type": training_info['coordinator'].model_type,
                "model_name": training_info['coordinator'].model_name,
                "current_epoch": prog.get('epoch', 0),
                "total_epochs": training_info['config']['config']['epochs'],
                "phase": prog.get('phase'),
                "current_step": prog.get('step'),
                "total_steps": prog.get('total_steps'),
                "batch_size": batch_size,
                "train_steps_per_epoch": train_steps,
                "val_steps_per_epoch": val_steps,
                "train_samples": train_samples,
                "val_samples": val_samples,
                "best_accuracy": 0.0,  # TODO: Implement accuracy tracking
                "latest_metrics": latest_metrics,
                "updates": updates
            }

        # Return all active trainings
        return {
            "active_trainings": list(self.active_trainings.keys()),
            "count": len(self.active_trainings)
        }

    def _run_training(self, training_config: Dict[str, Any], project_name: str,
                     model_type: str, model_name: str, training_id: str,
                     callback: Optional[Callable] = None, stop_event: Optional[threading.Event] = None):
        """Run training process in background thread.

        Args:
            training_config: Prepared training configuration
            project_name: Name of the project
            model_type: Type of model
            model_name: Specific model name
            training_id: Unique training identifier
            callback: Optional progress callback
        """
        try:
            # Update status
            if training_id in self.active_trainings:
                self.active_trainings[training_id]['status'] = 'running'

            info(f"Starting background training for {project_name} (ID: {training_id})")

            config = training_config['config']
            model = training_config['model']
            train_loader = training_config['train_loader']
            val_loader = training_config['val_loader']
            optimizer = training_config['optimizer']
            scheduler = training_config['scheduler']
            criterion = training_config['criterion']

            num_epochs = config['epochs']
            best_accuracy = 0.0
            best_val_loss = float('inf')

            # Create output directory (use coordinator-prepared path when available)
            output_dir = Path(config.get('output_dir') or f"models/{project_name}/model")
            output_dir.mkdir(parents=True, exist_ok=True)

            training_history = []

            stop_signal = stop_event or threading.Event()
            if training_id in self.active_trainings:
                self.active_trainings[training_id]['stop_event'] = stop_signal

            def should_stop() -> bool:
                if stop_signal.is_set():
                    return True
                if training_id not in self.active_trainings:
                    return True
                return self.active_trainings[training_id].get('status') == 'stopping'

            stopped_early = False

            for epoch in range(num_epochs):
                if should_stop():
                    stopped_early = True
                    info(f"Training {training_id} stop detected before epoch {epoch + 1}")
                    break

                info(f"Epoch {epoch + 1}/{num_epochs} (ID: {training_id})")

                def _progress(phase: str, epoch_index: int, step: int, total_steps: int, metrics: Dict[str, Any]):
                    if should_stop():
                        return
                    self._record_step_event(
                        training_id,
                        phase=phase,
                        epoch=epoch_index + 1,
                        step=step,
                        total_steps=total_steps,
                        metrics=metrics
                    )

                train_metrics = model.train_epoch(
                    train_loader,
                    optimizer,
                    criterion,
                    scheduler,
                    progress=_progress,
                    epoch_index=epoch,
                    should_stop=should_stop
                )

                train_info_parts: List[str] = []
                if 'train_loss' in train_metrics and train_metrics['train_loss'] is not None:
                    train_info_parts.append(f"Train loss: {train_metrics['train_loss']:.4f}")
                if 'train_accuracy' in train_metrics and train_metrics['train_accuracy'] is not None:
                    train_info_parts.append(f"acc: {train_metrics['train_accuracy']:.2f}%")
                info(', '.join(train_info_parts) if train_info_parts else "Train phase complete")

                train_summary: Dict[str, Any] = {}
                if 'train_loss' in train_metrics and train_metrics['train_loss'] is not None:
                    train_summary['epoch_loss'] = train_metrics['train_loss']
                    train_summary['train_loss'] = train_metrics['train_loss']
                if 'train_accuracy' in train_metrics and train_metrics['train_accuracy'] is not None:
                    train_summary['epoch_accuracy'] = train_metrics['train_accuracy']
                    train_summary['train_accuracy'] = train_metrics['train_accuracy']
                if train_summary:
                    self._record_epoch_summary(
                        training_id,
                        phase='train',
                        epoch=epoch + 1,
                        metrics=train_summary
                    )

                if should_stop():
                    stopped_early = True
                    info(f"Training {training_id} stop detected after train phase of epoch {epoch + 1}")
                    break

                val_metrics = model.validate(
                    val_loader,
                    criterion,
                    progress=_progress,
                    epoch_index=epoch,
                    should_stop=should_stop
                )

                val_info_parts: List[str] = []
                if 'val_loss' in val_metrics and val_metrics['val_loss'] is not None:
                    val_info_parts.append(f"Val loss: {val_metrics['val_loss']:.4f}")
                if 'val_accuracy' in val_metrics and val_metrics['val_accuracy'] is not None:
                    val_info_parts.append(f"acc: {val_metrics['val_accuracy']:.2f}%")
                info(', '.join(val_info_parts) if val_info_parts else "Validation phase complete")

                val_summary: Dict[str, Any] = {}
                if 'val_loss' in val_metrics and val_metrics['val_loss'] is not None:
                    val_summary['epoch_loss'] = val_metrics['val_loss']
                    val_summary['val_loss'] = val_metrics['val_loss']
                if 'val_accuracy' in val_metrics and val_metrics['val_accuracy'] is not None:
                    val_summary['epoch_accuracy'] = val_metrics['val_accuracy']
                    val_summary['val_accuracy'] = val_metrics['val_accuracy']
                if val_summary:
                    self._record_epoch_summary(
                        training_id,
                        phase='val',
                        epoch=epoch + 1,
                        metrics=val_summary
                    )

                if should_stop():
                    stopped_early = True
                    info(f"Training {training_id} stop detected after validation phase of epoch {epoch + 1}")
                    break

                epoch_metrics = {'epoch': epoch + 1}
                if 'train_loss' in train_metrics and train_metrics['train_loss'] is not None:
                    epoch_metrics['train_loss'] = train_metrics['train_loss']
                if 'val_loss' in val_metrics and val_metrics['val_loss'] is not None:
                    epoch_metrics['val_loss'] = val_metrics['val_loss']
                if 'train_accuracy' in train_metrics and train_metrics['train_accuracy'] is not None:
                    epoch_metrics['train_accuracy'] = train_metrics['train_accuracy']
                if 'val_accuracy' in val_metrics and val_metrics['val_accuracy'] is not None:
                    epoch_metrics['val_accuracy'] = val_metrics['val_accuracy']
                training_history.append(epoch_metrics)

                if training_id in self.active_trainings:
                    self.active_trainings[training_id]['current_epoch'] = epoch + 1
                    if 'val_accuracy' in val_metrics and val_metrics['val_accuracy'] is not None:
                        current_best = max(best_accuracy, val_metrics['val_accuracy'])
                        self.active_trainings[training_id]['best_accuracy'] = current_best
                    elif 'val_loss' in val_metrics and val_metrics['val_loss'] is not None:
                        current_best_loss = min(best_val_loss, val_metrics['val_loss'])
                        self.active_trainings[training_id]['best_val_loss'] = current_best_loss

                improved = False
                save_reason = ""
                if 'val_accuracy' in val_metrics and val_metrics['val_accuracy'] is not None:
                    if val_metrics['val_accuracy'] > best_accuracy:
                        best_accuracy = val_metrics['val_accuracy']
                        improved = True
                        save_reason = f"accuracy {best_accuracy:.2f}%"
                elif 'val_loss' in val_metrics and val_metrics['val_loss'] is not None:
                    if val_metrics['val_loss'] < best_val_loss:
                        best_val_loss = val_metrics['val_loss']
                        improved = True
                        save_reason = f"val_loss {best_val_loss:.4f}"

                if improved:
                    checkpoint_config = SETTINGS.get('training', {}).get('checkpoint', {})
                    checkpoint_path = output_dir / checkpoint_config.get('best_model_filename', 'best_model.pth')
                    model.save_checkpoint(
                        str(checkpoint_path),
                        epoch + 1,
                        optimizer,
                        scheduler,
                        epoch_metrics
                    )
                    info(f"New best model by {save_reason} -> checkpoint saved")

                if callback:
                    callback(epoch + 1, num_epochs, epoch_metrics)

            # Save final training history
            history_filename = SETTINGS.get('training', {}).get('checkpoint', {}).get('training_history_filename', 'training_history.json')
            history_path = output_dir / history_filename
            with open(history_path, 'w') as f:
                json.dump(training_history, f, indent=2)

            finished_normally = not stopped_early and not stop_signal.is_set()

            if training_id in self.active_trainings:
                if finished_normally:
                    self.active_trainings[training_id]['status'] = 'completed'
                    self.active_trainings[training_id]['final_accuracy'] = best_accuracy
                else:
                    self.active_trainings[training_id]['status'] = 'stopped'

            if finished_normally:
                if best_accuracy > 0:
                    success(f"Training completed for {project_name}! Best validation accuracy: {best_accuracy:.2f}%")
                elif best_val_loss < float('inf'):
                    success(f"Training completed for {project_name}! Best validation loss: {best_val_loss:.4f}")
                else:
                    success(f"Training completed for {project_name}!")
            else:
                info(f"Training interrupted for {project_name} (ID: {training_id})")

        except Exception as e:
            error(f"Training failed for {project_name} (ID: {training_id}): {e}")
            if training_id in self.active_trainings:
                self.active_trainings[training_id]['status'] = 'failed'
                self.active_trainings[training_id]['error'] = str(e)


# Global training manager instance
training_manager = TrainingManager()


def start_training(project_name: str, model_type: str = "resnet",
                  model_name: str = "resnet50", epochs: Optional[int] = None,
                  callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Convenience function to start training.

    Args:
        project_name: Name of the project to train on
        model_type: Type of model to use
        model_name: Specific model name
        epochs: Number of epochs to train
        callback: Optional callback function for progress updates

    Returns:
        dict: Training start status
    """
    return training_manager.start_training(project_name, model_type, model_name, epochs, callback)


def stop_training(training_id: str) -> Dict[str, Any]:
    """Convenience function to stop training.

    Args:
        training_id: ID of the training to stop

    Returns:
        dict: Stop status
    """
    return training_manager.stop_training(training_id)


def get_training_status(training_id: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to get training status.

    Args:
        training_id: Specific training ID, or None for all

    Returns:
        dict: Training status information
    """
    return training_manager.get_training_status(training_id)


def get_training_status_all(training_id: Optional[str] = None) -> Dict[str, Any]:
    """Get full training status history.

    Args:
        training_id: Specific training ID, or None for all

    Returns:
        dict: Training history information
    """
    return training_manager.get_training_history(training_id)


def get_supported_training_models() -> Dict[str, Any]:
    """Get list of supported models for training.

    Returns:
        dict: Supported model types and their variants
    """
    from system.coordinator import get_supported_models
    return get_supported_models()