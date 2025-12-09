from typing import Dict, Any, Optional, Callable, List
import threading
import json
from pathlib import Path
import copy

from system.log import info, success, error, warning
from system.coordinator import create_coordinator
from system.coordinator_settings import SETTINGS
from system.common.checkpoint import find_checkpoint
from system import project_db


class TrainingManager:
    def __init__(self):
        self.active_trainings: Dict[str, Dict[str, Any]] = {}
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

    def _find_latest_checkpoint(self, output_dir: Path, best_filename: str, model_filename: str) -> Optional[Path]:
        """Find a checkpoint preferring best model, then regular model saves."""
        found, path, _ = find_checkpoint(str(output_dir), best_filename, model_filename)
        return path if found else None

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
                      callback: Optional[Callable] = None,
                      resume: bool = False, mode: str = "new") -> Dict[str, Any]:
        try:
            training_id = f"{project_name}_{model_type}_{model_name}_{threading.current_thread().ident}"

            self._reset_progress_tracking(training_id)

            coordinator = create_coordinator(model_type, model_name)

            training_config = coordinator.prepare_training(project_name)

            if epochs:
                training_config['config']['epochs'] = epochs

            stop_event = threading.Event()

            # Determine effective mode from resume flag if mode not explicitly set
            effective_mode = mode
            if mode == "new" and resume:
                effective_mode = "resume"

            training_thread = threading.Thread(
                target=self._run_training,
                args=(training_config, project_name, model_type, model_name, training_id, callback, stop_event, effective_mode),
                name=f"Training-{training_id}"
            )
            training_thread.daemon = True

            self.active_trainings[training_id] = {
                'thread': training_thread,
                'config': training_config,
                'coordinator': coordinator,
                'start_time': threading.current_thread().ident,
                'status': 'starting',
                'stop_event': stop_event,
                'mode': effective_mode
            }

            training_thread.start()

            mode_msgs = {
                'new': '',
                'resume': ' (resuming from checkpoint)',
                'finetune': ' (fine-tuning from checkpoint)'
            }
            mode_msg = mode_msgs.get(effective_mode, '')
            info(f"Training started for project {project_name} with ID: {training_id}{mode_msg}")

            return {
                "started": True,
                "training_id": training_id,
                "project": project_name,
                "model_type": model_type,
                "model_name": model_name,
                "mode": effective_mode,
                "config": coordinator.get_training_summary(training_config),
                "message": f"Training started successfully for {project_name}{mode_msg}"
            }

        except Exception as e:
            error(f"Failed to start training: {e}")
            return {
                "started": False,
                "error": str(e),
                "message": "Failed to start training"
            }

    def stop_training(self, training_id: str) -> Dict[str, Any]:
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
        if training_id:
            if training_id not in self.active_trainings:
                return {"error": f"Training {training_id} not found"}

            training_info = self.active_trainings[training_id]
            ds_path = training_info['config']['config']['dataset_path'] or ""
            proj_name = Path(ds_path).parts[-2] if ds_path else "unknown"
            prog = self.progress.get(training_id, {})
            updates = self._consume_updates(training_id)
            best_accuracy = training_info.get('best_accuracy')
            if best_accuracy is None:
                best_accuracy = 0.0
            best_val_loss = training_info.get('best_val_loss')
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
                "best_accuracy": best_accuracy,
                "best_val_loss": best_val_loss,
                "latest_metrics": latest_metrics,
                "updates": updates
            }

        return {
            "active_trainings": list(self.active_trainings.keys()),
            "count": len(self.active_trainings)
        }

    def _run_training(self, training_config: Dict[str, Any], project_name: str,
                     model_type: str, model_name: str, training_id: str,
                     callback: Optional[Callable] = None, stop_event: Optional[threading.Event] = None,
                     mode: str = "new"):
        try:
            if training_id in self.active_trainings:
                self.active_trainings[training_id]['status'] = 'running'

            info(f"Starting background training for {project_name} (ID: {training_id}, mode: {mode})")

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
            start_epoch = 0

            # Resolve checkpoint directory inside the configured projects folder so UI detection works
            try:
                projects_dir = SETTINGS['paths'].get('projects_dir', 'projects')
            except Exception:
                projects_dir = 'projects'
            root_path = Path(__file__).resolve().parents[1]
            output_dir = Path(config.get('output_dir') or root_path / projects_dir / project_name / 'model')
            output_dir = output_dir.resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            # Persist back for downstream components that rely on config
            config['output_dir'] = str(output_dir)

            try:
                base_checkpoint_cfg = SETTINGS['training']['checkpoint']
            except Exception:
                raise ValueError("Missing required 'training.checkpoint' section in config/config.json")

            checkpoint_config = dict(base_checkpoint_cfg) if isinstance(base_checkpoint_cfg, dict) else {}
            project_checkpoint_cfg = training_config.get('checkpoint_config')
            if isinstance(project_checkpoint_cfg, dict):
                checkpoint_config.update(project_checkpoint_cfg)

            best_model_filename = checkpoint_config.get('best_model_filename') or 'best_model.pth'
            model_filename = checkpoint_config.get('model_filename') or 'model.pth'
            save_best_only = checkpoint_config.get('save_best_only', True)
            save_frequency = checkpoint_config.get('save_frequency', 1)

            # Load checkpoint based on mode
            if mode in ('resume', 'finetune'):
                checkpoint_path = self._find_latest_checkpoint(output_dir, best_model_filename, model_filename)
                if checkpoint_path:
                    try:
                        loaded_epoch, checkpoint_data = model.load_checkpoint(str(checkpoint_path))
                        
                        if mode == 'resume':
                            # Resume: restore everything and continue from where we left off
                            start_epoch = loaded_epoch
                            
                            # Restore optimizer state
                            if 'optimizer_state_dict' in checkpoint_data:
                                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                                info(f"Restored optimizer state from checkpoint")
                            
                            # Restore scheduler state
                            if scheduler and 'scheduler_state_dict' in checkpoint_data:
                                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                                info(f"Restored scheduler state from checkpoint")
                            
                            # Restore best metrics
                            if 'metrics' in checkpoint_data:
                                metrics = checkpoint_data['metrics']
                                if 'val_accuracy' in metrics and metrics['val_accuracy']:
                                    best_accuracy = metrics['val_accuracy']
                                if 'val_loss' in metrics and metrics['val_loss']:
                                    best_val_loss = metrics['val_loss']
                            
                            info(f"Resuming training from epoch {start_epoch + 1} (checkpoint: {checkpoint_path.name})")
                        
                        elif mode == 'finetune':
                            # Fine-tune: load model weights only, reset optimizer/scheduler, start from epoch 1
                            # Model weights are already loaded by load_checkpoint
                            # Do NOT restore optimizer, scheduler, or metrics
                            start_epoch = 0
                            info(f"Fine-tuning from checkpoint: {checkpoint_path.name} (starting fresh from epoch 1)")
                        
                    except Exception as ex:
                        error(f"Failed to load checkpoint, starting from scratch: {ex}")
                        start_epoch = 0
                else:
                    warning(f"No checkpoint found for {mode} request in {output_dir}, starting new training")

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

            for epoch in range(start_epoch, num_epochs):
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
                if 'learning_rate' in train_metrics and train_metrics['learning_rate'] is not None:
                    train_info_parts.append(f"lr: {train_metrics['learning_rate']:.2e}")
                info(', '.join(train_info_parts) if train_info_parts else "Train phase complete")

                train_summary: Dict[str, Any] = {}
                if 'train_loss' in train_metrics and train_metrics['train_loss'] is not None:
                    train_summary['epoch_loss'] = train_metrics['train_loss']
                    train_summary['train_loss'] = train_metrics['train_loss']
                if 'train_accuracy' in train_metrics and train_metrics['train_accuracy'] is not None:
                    train_summary['epoch_accuracy'] = train_metrics['train_accuracy']
                    train_summary['train_accuracy'] = train_metrics['train_accuracy']
                if 'learning_rate' in train_metrics and train_metrics['learning_rate'] is not None:
                    train_summary['learning_rate'] = train_metrics['learning_rate']
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
                if 'learning_rate' in train_metrics and train_metrics['learning_rate'] is not None:
                    epoch_metrics['learning_rate'] = train_metrics['learning_rate']
                training_history.append(epoch_metrics)

                # Save to project.db training_history table
                project_db.training_history_record(
                    project_name=project_name,
                    training_id=training_id,
                    epoch=epoch + 1,
                    train_loss=epoch_metrics.get('train_loss'),
                    val_loss=epoch_metrics.get('val_loss'),
                    train_accuracy=epoch_metrics.get('train_accuracy'),
                    val_accuracy=epoch_metrics.get('val_accuracy'),
                    learning_rate=epoch_metrics.get('learning_rate')
                )

                if training_id in self.active_trainings:
                    self.active_trainings[training_id]['current_epoch'] = epoch + 1
                    if 'val_accuracy' in val_metrics and val_metrics['val_accuracy'] is not None:
                        current_best = max(best_accuracy, val_metrics['val_accuracy'])
                        self.active_trainings[training_id]['best_accuracy'] = current_best
                    elif 'val_loss' in val_metrics and val_metrics['val_loss'] is not None:
                        current_best_loss = min(best_val_loss, val_metrics['val_loss'])
                        self.active_trainings[training_id]['best_val_loss'] = current_best_loss

                # Check if model improved
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

                if 'labels' not in config or not config['labels']:
                    raise ValueError("Training config is missing the label list required for deterministic checkpoints")
                # labels is now a dict {idx: name} - save as-is for checkpoint
                labels_dict = config['labels']
                if not isinstance(labels_dict, dict):
                    # Backwards compat: convert list to dict if needed
                    labels_dict = {idx: name for idx, name in enumerate(labels_dict)}

                # Determine if we should save checkpoints this epoch
                save_requests: List[tuple[str, str]] = []  # (filename, reason)

                if save_best_only:
                    if improved:
                        save_requests.append((best_model_filename, save_reason or "improved"))
                else:
                    if (epoch + 1) % save_frequency == 0:
                        save_requests.append((model_filename, f"epoch {epoch + 1} (periodic)"))
                    if improved:
                        save_requests.append((best_model_filename, save_reason or "improved"))

                for checkpoint_filename, reason in save_requests:
                    checkpoint_path = output_dir / checkpoint_filename
                    info(f"[DEBUG] Saving checkpoint with labels: {labels_dict} (count: {len(labels_dict)})")
                    model.save_checkpoint(
                        str(checkpoint_path),
                        epoch + 1,
                        optimizer,
                        scheduler,
                        epoch_metrics,
                        labels=labels_dict
                    )
                    info(f"Checkpoint saved: {checkpoint_filename} ({reason})")

                if callback:
                    callback(epoch + 1, num_epochs, epoch_metrics)

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


training_manager = TrainingManager()


def start_training(project_name: str, model_type: str = "resnet",
                  model_name: str = "resnet50", epochs: Optional[int] = None,
                  callback: Optional[Callable] = None, resume: bool = False,
                  mode: str = "new") -> Dict[str, Any]:
    return training_manager.start_training(project_name, model_type, model_name, epochs, callback, resume, mode)


def stop_training(training_id: str) -> Dict[str, Any]:
    return training_manager.stop_training(training_id)


def get_training_status(training_id: Optional[str] = None) -> Dict[str, Any]:
    return training_manager.get_training_status(training_id)


def get_training_status_all(training_id: Optional[str] = None) -> Dict[str, Any]:
    return training_manager.get_training_history(training_id)


def get_supported_training_models() -> Dict[str, Any]:
    from system.coordinator import get_supported_models
    return get_supported_models()