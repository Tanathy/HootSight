from typing import Dict, Any, Optional, Tuple, List, Set
import os
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from system.log import info, success, warning, error
from system.coordinator_settings import SETTINGS
from system.memory import get_optimal_batch_size, get_memory_status
from system.dataset_discovery import get_project_info
from system.augmentation import DataAugmentationFactory
from system.optimizers import get_optimizer_for_model
from system.schedulers import get_scheduler_for_training
from system.losses import get_loss_for_task
from system.weight_init import WeightInitFactory
from system.project_db import list_metadata, flat_to_nested
from system.common.deep_merge import deep_merge_json
from system.project_labels import persist_project_labels


def _format_path_sample(paths: List[str], limit: int = 5) -> str:
    sample = ', '.join(Path(p).name for p in paths[:limit])
    if len(paths) > limit:
        sample += ', ...'
    return sample


def _read_annotation_file(txt_path: str) -> List[str]:
    labels: List[str] = []
    try:
        with open(txt_path, 'r', encoding='utf-8') as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                parts = [segment.strip() for segment in line.split(',') if segment.strip()]
                labels.extend(parts)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to read labels from '{txt_path}': {exc}") from exc

    # Preserve order but drop duplicates
    seen: Set[str] = set()
    deduped: List[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            deduped.append(label)
    return deduped


def _read_labels_for_image(img_path: str) -> List[str]:
    """Read mandatory comma-separated labels from the sibling .txt file."""
    txt_path = os.path.splitext(img_path)[0] + '.txt'
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Missing annotation txt for '{img_path}' (expected '{txt_path}')")

    labels = _read_annotation_file(txt_path)
    if not labels:
        raise ValueError(f"No labels defined inside '{txt_path}'. Provide at least one comma-separated tag.")
    return labels


def _build_label_cache(image_paths: List[str], labels_map: Dict[str, int]) -> Dict[str, List[str]]:
    cache: Dict[str, List[str]] = {}
    missing_files: List[str] = []
    empty_or_invalid: List[str] = []
    skipped_unknown_images: List[str] = []
    unknown_labels: Set[str] = set()

    for img_path in image_paths:
        try:
            labels = _read_labels_for_image(img_path)
        except FileNotFoundError:
            missing_files.append(img_path)
            continue
        except ValueError:
            empty_or_invalid.append(img_path)
            continue

        filtered_labels = [label for label in labels if label in labels_map]
        unknown = [label for label in labels if label not in labels_map]
        if unknown:
            unknown_labels.update(unknown)
        if not filtered_labels:
            skipped_unknown_images.append(img_path)
            continue

        cache[img_path] = filtered_labels

    if missing_files:
        warning(
            f"Skipped {len(missing_files)} images missing annotation files (e.g., {_format_path_sample(missing_files)})."
        )

    if empty_or_invalid:
        warning(
            f"Skipped {len(empty_or_invalid)} annotation files because they were empty or malformed (e.g., {_format_path_sample(empty_or_invalid)})."
        )

    if skipped_unknown_images:
        warning(
            f"Skipped {len(skipped_unknown_images)} images referencing labels that are not registered in project metadata (e.g., {_format_path_sample(skipped_unknown_images)})."
        )

    if unknown_labels:
        sample_labels = ', '.join(sorted(list(unknown_labels))[:5])
        if len(unknown_labels) > 5:
            sample_labels += ', ...'
        warning(
            f"Unknown labels detected: {sample_labels}. Refresh project statistics to sync the label space."
        )

    if not cache:
        raise ValueError("No valid annotated samples remain after filtering invalid files.")

    info(f"Using {len(cache)} of {len(image_paths)} images after removing invalid annotations.")

    return cache


class CustomImageDataset(Dataset):
    """Dataset for single-label image classification."""
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        p = self.image_paths[idx]
        img = Image.open(p).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class MultiLabelDataset(Dataset):
    """Dataset for multi-label image classification."""
    def __init__(self, paths: List[str], labels_map: Dict[str, int], label_count: int, transform=None):
        self.paths = paths
        self.labels_map = labels_map
        self.label_count = label_count
        self.transform = transform
        self._label_cache = _build_label_cache(self.paths, labels_map)
        self.paths = [p for p in self.paths if p in self._label_cache]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        p = self.paths[i]
        img = Image.open(p).convert('RGB')
        labs = self._label_cache[p]
        target = torch.zeros(self.label_count, dtype=torch.float32)
        for lab in labs:
            if lab in self.labels_map:
                target[self.labels_map[lab]] = 1.0
        if self.transform:
            img = self.transform(img)
        return img, target


class MultiLabelFolderDataset(Dataset):
    """Dataset for multi-label classification with train/val folder structure."""
    def __init__(self, root: str, labels_map: Dict[str, int], label_count: int, transform=None):
        import glob
        self.transform = transform
        self.labels_map = labels_map
        self.label_count = label_count
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']
        self.images: List[str] = []
        for ext in exts:
            self.images.extend(glob.glob(os.path.join(root, '**', ext), recursive=True))
        self.images = [p for p in self.images if os.path.isfile(p)]
        self._label_cache = _build_label_cache(self.images, labels_map)
        self.images = [p for p in self.images if p in self._label_cache]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        path = self.images[idx]
        img = Image.open(path).convert('RGB')
        labs = self._label_cache[path]
        target = torch.zeros(self.label_count, dtype=torch.float32)
        for lab in labs:
            if lab in self.labels_map:
                target[self.labels_map[lab]] = 1.0
        if self.transform:
            img = self.transform(img)
        return img, target


class TrainingCoordinator:

    def __init__(self, model_type: str = 'resnet', model_name: str = 'resnet50'):
        self.model_type = model_type
        self.model_name = model_name
        self.settings = SETTINGS.copy() if SETTINGS else {}
        self.config = {}  # Will be set in prepare_training
        self.memory_config = {}  # Will be set in prepare_training
        self.model_config = {}  # Will be set in prepare_training

        info(f"Initialized coordinator for {model_type}/{model_name}")

    def _load_config(self) -> Dict[str, Any]:
        try:
            training = self.settings['training']
        except Exception:
            raise ValueError("Missing required 'training' section in config/config.json")
        input_block = training.get('input') if isinstance(training.get('input'), dict) else None
        if input_block is not None:
            if 'image_size' not in input_block:
                raise ValueError("Missing required 'training.input.image_size' in config/config.json")
            input_size = input_block['image_size']
            if 'normalize' not in input_block:
                raise ValueError("Missing required 'training.input.normalize' in config/config.json")
            normalize_cfg = input_block['normalize']
        else:
            if 'input_size' not in training:
                raise ValueError("Missing required 'training.input_size' in config/config.json")
            input_size = training['input_size']
            if 'normalize' not in training:
                raise ValueError("Missing required 'training.normalize' in config/config.json")
            normalize_cfg = training['normalize']
        if 'dataloader' not in training or not isinstance(training.get('dataloader'), dict):
            raise ValueError("Missing required 'training.dataloader' configuration in config/config.json")
        dataloader_cfg = training['dataloader']
        if 'num_workers' not in dataloader_cfg:
            raise ValueError("Missing required 'training.dataloader.num_workers' in config/config.json")
        raw_workers = dataloader_cfg['num_workers']
        if isinstance(raw_workers, str) and raw_workers.lower() == 'auto':
            try:
                import os
                cpu_count = os.cpu_count() or 2
                resolved_workers = max(2, cpu_count - 1)
            except Exception:
                resolved_workers = 2
        else:
            try:
                resolved_workers = int(raw_workers)
            except Exception:
                resolved_workers = 0
        if 'runtime' not in training or not isinstance(training.get('runtime'), dict):
            raise ValueError("Missing required 'training.runtime' configuration in config/config.json")
        runtime_cfg = training['runtime']
        runtime_flags = {
            'mixed_precision': bool(runtime_cfg['mixed_precision']),
            'channels_last': bool(runtime_cfg['channels_last']),
            'allow_tf32': bool(runtime_cfg['allow_tf32']),
            'cudnn_benchmark': bool(runtime_cfg['cudnn_benchmark'])
        }
        if 'epochs' not in training:
            raise ValueError("Missing required 'training.epochs' in config/config.json")
        raw_epochs = training['epochs']
        try:
            epochs = int(raw_epochs)
            if epochs <= 0:
                epochs = 10
        except Exception:
            epochs = 10
        split_block = training.get('split') if isinstance(training.get('split'), dict) else None
        if split_block is not None and 'val_ratio' in split_block:
            val_ratio = split_block['val_ratio']
        elif 'val_ratio' in training:
            val_ratio = training['val_ratio']
        else:
            raise ValueError("Missing required 'training.val_ratio' or 'training.split.val_ratio' in config/config.json")

        missing_training_keys = [k for k in ['batch_size', 'learning_rate', 'weight_decay', 'task'] if k not in training]
        if 'pretrained' not in training and not (isinstance(training.get('model'), dict) and 'pretrained' in training['model']):
            missing_training_keys.append('pretrained')
        if missing_training_keys:
            raise ValueError(f"Missing required training keys in config/config.json: {', '.join(missing_training_keys)}")

        return {
            'batch_size': training['batch_size'],
            'epochs': epochs,
            'learning_rate': training['learning_rate'],
            'weight_decay': training['weight_decay'],
            'task': training['task'],
            'model_type': self.model_type,
            'pretrained': (bool(training['pretrained']) if 'pretrained' in training else bool(training['model']['pretrained'])),
            'projects_base_dir': (self.settings['paths']['projects_dir'] if 'paths' in self.settings and 'projects_dir' in self.settings['paths'] else (self.settings['paths']['models_dir'] if 'paths' in self.settings and 'models_dir' in self.settings['paths'] else None)),
            'dataset_path': None,
            'output_dir': None,
            'input': {
                'channels': 3,
                'image_size': int(input_size),
                'normalize': normalize_cfg
            },
            'dataloader': {
                'num_workers': resolved_workers,
                'pin_memory': bool(dataloader_cfg['pin_memory'])
            },
            'runtime': runtime_flags,
            'split': {'val_ratio': float(val_ratio)}
        }

    def _apply_runtime_settings(self) -> None:
        runtime = self.config['runtime']
        
        if runtime['allow_tf32']:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                info("Enabled TF32 for faster computations")
            except Exception as e:
                warning(f"Could not enable TF32: {e}")
        
        if runtime['cudnn_benchmark']:
            try:
                torch.backends.cudnn.benchmark = True
                info("Enabled cuDNN benchmark for optimized convolutions")
            except Exception as e:
                warning(f"Could not enable cuDNN benchmark: {e}")

    def _get_memory_config(self) -> Dict[str, Any]:
        try:
            memory_settings = self.settings['memory']
        except Exception:
            raise ValueError("Missing required 'memory' section in config/config.json")
        missing = [k for k in ('target_memory_usage', 'safety_margin', 'augmentation_threads') if k not in memory_settings]
        if missing:
            raise ValueError(f"Missing required memory keys in config/config.json: {', '.join(missing)}")
        return {
            'target_usage': memory_settings['target_memory_usage'],
            'safety_margin': memory_settings['safety_margin'],
            'augmentation_threads': memory_settings['augmentation_threads']
        }

    def _get_model_config(self) -> Dict[str, Any]:
        model_type = self.config['model_type']
        if model_type == 'resnet':
            from system.models.resnet import get_resnet_config
            return get_resnet_config(self.model_name)
        elif model_type == 'resnext':
            from system.models.resnext import get_resnext_config
            return get_resnext_config(self.model_name)
        elif model_type == 'mobilenet':
            from system.models.mobilenet import get_mobilenet_config
            return get_mobilenet_config(self.model_name)
        elif model_type == 'shufflenet':
            from system.models.shufflenet import get_shufflenet_config
            return get_shufflenet_config(self.model_name)
        elif model_type == 'squeezenet':
            from system.models.squeezenet import get_squeezenet_config
            return get_squeezenet_config(self.model_name)
        elif model_type == 'efficientnet':
            from system.models.efficientnet import get_efficientnet_config
            return get_efficientnet_config(self.model_name)
        else:
            return {}  # Default empty config

    def prepare_training(self, project_name: str) -> Dict[str, Any]:
        info(f"Preparing training for project: {project_name}")

        try:
            paths_cfg = self.settings['paths']
        except Exception:
            raise ValueError("Missing required 'paths' section in config/config.json")
        if 'projects_dir' not in paths_cfg:
            raise ValueError("Missing required 'paths.projects_dir' in config/config.json")
        
        # STEP 1: Load project-level config.json overrides (legacy method)
        project_config_path = os.path.join(paths_cfg['projects_dir'], project_name, 'config.json')
        if os.path.exists(project_config_path):
            try:
                with open(project_config_path, 'r', encoding='utf-8') as f:
                    project_settings = json.load(f)
                self.settings = deep_merge_json(self.settings, project_settings)
                info(f"Loaded project settings from {project_config_path}")
            except Exception as e:
                warning(f"Failed to load project settings: {e}")
        
        # STEP 2: Load project.db metadata overrides (priority!)
        # These are set via the UI and override everything else
        db_overrides = list_metadata(project_name)
        if db_overrides:
            # Convert flat keys like "training.task" to nested {"training": {"task": ...}}
            nested_overrides = flat_to_nested(db_overrides)
            self.settings = deep_merge_json(self.settings, nested_overrides)
            info(f"Applied {len(db_overrides)} project overrides from project.db")
            
            # Override model_type and model_name if specified in metadata
            if 'training.model_type' in db_overrides:
                self.model_type = db_overrides['training.model_type']
                info(f"Using model type override from project settings: {self.model_type}")
            if 'training.model_name' in db_overrides:
                self.model_name = db_overrides['training.model_name']
                info(f"Using model from project settings: {self.model_name}")

        self.config = self._load_config()
        self.memory_config = self._get_memory_config()
        self.model_config = self._get_model_config()

        self._apply_runtime_settings()

        # compute_stats=True is REQUIRED to get labels for training
        project_info = get_project_info(project_name, compute_stats=True)
        if not project_info:
            raise ValueError(f"Project not found: {project_name}")

        self.config['dataset_path'] = project_info.dataset_path
        base_labels = [lbl for lbl in project_info.labels if not str(lbl).startswith('folder:')]
        label_list = sorted(base_labels if base_labels else project_info.labels)
        # Store labels as dict with index keys for deterministic mapping
        # {0: "cat", 1: "dog", 2: "bird"} - index is the model output index
        labels_dict = {idx: name for idx, name in enumerate(label_list)}
        self.config['labels'] = labels_dict
        self.config['num_classes'] = len(label_list)

        persist_project_labels(
            project_name,
            labels_dict,  # Pass dict, not list!
            self.config['task'],
            self.config.get('projects_base_dir')
        )
        
        # Sanity check - training with 0 classes makes no sense
        if self.config['num_classes'] == 0:
            raise ValueError(f"No labels found for project '{project_name}'. Cannot train with 0 classes.")
        
        ckpt_dir_name = 'model'
        self.config['output_dir'] = os.path.join(self.config['projects_base_dir'], project_name, ckpt_dir_name)

        optimal_batch = self._calculate_optimal_batch_size()
        self.config['batch_size'] = optimal_batch['optimal_batch_size']

        model = self._prepare_model()

        train_loader, val_loader = self._prepare_data_loaders()

        optimizer, scheduler, criterion = self._prepare_training_components(model)

        training_config = {
            'model': model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'criterion': criterion,
            'config': self.config,
            'project_info': project_info.to_dict(),
            'memory_info': get_memory_status(),
            'batch_calculation': optimal_batch
        }

        success(f"Training preparation completed for {project_name}")
        return training_config

    def _calculate_optimal_batch_size(self) -> Dict[str, Any]:
        model_type = self.config['model_type']
        if model_type == 'resnet':
            from system.models.resnet import create_resnet_model, get_resnet_config
            dummy_model = create_resnet_model(self.model_name, self.config['num_classes'], pretrained=False, task=self.config['task'])
            config_func = get_resnet_config
        elif model_type == 'resnext':
            from system.models.resnext import create_resnext_model, get_resnext_config
            dummy_model = create_resnext_model(self.model_name, self.config['num_classes'], pretrained=False, task=self.config['task'])
            config_func = get_resnext_config
        elif model_type == 'mobilenet':
            from system.models.mobilenet import create_mobilenet_model, get_mobilenet_config
            dummy_model = create_mobilenet_model(self.model_name, self.config['num_classes'], pretrained=False, task=self.config['task'])
            config_func = get_mobilenet_config
        elif model_type == 'shufflenet':
            from system.models.shufflenet import create_shufflenet_model, get_shufflenet_config
            dummy_model = create_shufflenet_model(self.model_name, self.config['num_classes'], pretrained=False, task=self.config['task'])
            config_func = get_shufflenet_config
        elif model_type == 'squeezenet':
            from system.models.squeezenet import create_squeezenet_model, get_squeezenet_config
            dummy_model = create_squeezenet_model(self.model_name, self.config['num_classes'], pretrained=False, task=self.config['task'])
            config_func = get_squeezenet_config
        elif model_type == 'efficientnet':
            from system.models.efficientnet import create_efficientnet_model, get_efficientnet_config
            dummy_model = create_efficientnet_model(self.model_name, self.config['num_classes'], pretrained=False, task=self.config['task'])
            config_func = get_efficientnet_config
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        try:
            model_config = config_func(self.model_name)
            if 'recommended_batch_size' not in model_config:
                raise ValueError(f"recommended_batch_size missing for model '{self.model_name}' in config/config.json")
            recommended_batch = model_config['recommended_batch_size']
        except:
            recommended_batch = 32

        channels = int(self.config['input']['channels'])
        size = int(self.config['input']['image_size'])
        input_shape = (channels, size, size)

        optimal_batch = get_optimal_batch_size(
            dummy_model.model,
            input_shape,
            target_memory_usage=self.memory_config['target_usage'],
            safety_margin=self.memory_config['safety_margin']
        )

        info(f"Calculated optimal batch size: {optimal_batch['optimal_batch_size']}")
        return optimal_batch

    def _prepare_model(self):
        model_type = self.config['model_type']
        model_name = self.model_name
        num_classes = self.config['num_classes']
        pretrained = bool(self.config['pretrained'])
        task = self.config['task']
        runtime = self.config['runtime']

        if model_type == 'resnet':
            from system.models.resnet import create_resnet_model
            model = create_resnet_model(model_name, num_classes, pretrained, task)
        elif model_type == 'resnext':
            from system.models.resnext import create_resnext_model
            model = create_resnext_model(model_name, num_classes, pretrained, task)
        elif model_type == 'mobilenet':
            from system.models.mobilenet import create_mobilenet_model
            model = create_mobilenet_model(model_name, num_classes, pretrained, task)
        elif model_type == 'shufflenet':
            from system.models.shufflenet import create_shufflenet_model
            model = create_shufflenet_model(model_name, num_classes, pretrained, task)
        elif model_type == 'squeezenet':
            from system.models.squeezenet import create_squeezenet_model
            model = create_squeezenet_model(model_name, num_classes, pretrained, task)
        elif model_type == 'efficientnet':
            from system.models.efficientnet import create_efficientnet_model
            model = create_efficientnet_model(model_name, num_classes, pretrained, task)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        try:
            import torch
            if runtime['allow_tf32']:
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
                    torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
                except Exception:
                    pass
            if runtime['cudnn_benchmark']:
                try:
                    torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
                except Exception:
                    pass
            if runtime['channels_last']:
                try:
                    if hasattr(model, 'model'):
                        from system.device import get_device
                        self_device = model.device if hasattr(model, 'device') else get_device()
                        model.model = model.model.to(self_device)
                        for p in model.model.parameters():
                            try:
                                p.data = p.data.contiguous(memory_format=torch.channels_last)
                            except Exception:
                                break
                except Exception:
                    pass
        except Exception:
            pass

        try:
            try:
                train_cfg = self.settings['training']
            except Exception:
                raise ValueError("Missing required 'training' section in config/config.json")
            init_cfg = train_cfg.get('weight_init')
            if init_cfg:
                WeightInitFactory.apply_to_model(model.model, init_cfg)
        except Exception as ex:
            warning(f"Weight initialization skipped: {ex}")

        try:
            from system.models.resnet import ResNetModel  # type: ignore
            from system.device import get_device_type, create_grad_scaler
            runtime = self.config['runtime']
            if isinstance(model, ResNetModel):
                device_type = get_device_type()
                cuda_enabled = device_type == 'cuda'
                model.use_amp = bool(runtime['mixed_precision'] and cuda_enabled)
                model._scaler = create_grad_scaler(model.use_amp)
                model.channels_last = bool(runtime['channels_last'])
        except Exception:
            pass

        info(f"Prepared {self.model_type} model: {model.get_model_info()}")
        return model

    def _prepare_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        from torchvision import datasets, transforms
        import glob
        import copy

        image_size = int(self.config['input']['image_size'])
        mean = self.config['input']['normalize']['mean']
        std = self.config['input']['normalize']['std']

        try:
            training_cfg = self.settings['training']
        except Exception:
            raise ValueError("Missing required 'training' section in config/config.json")
        if 'augmentation' not in training_cfg or not isinstance(training_cfg['augmentation'], dict):
            raise ValueError("Missing required 'training.augmentation' in config/config.json")
        
        # Deep copy augmentation configs to avoid modifying original settings
        train_aug_cfg = copy.deepcopy(training_cfg['augmentation'].get('train'))
        val_aug_cfg = copy.deepcopy(training_cfg['augmentation'].get('val'))
        
        # Override size parameters in augmentation configs with project's input_size
        def _override_sizes(aug_list, target_size):
            if not isinstance(aug_list, list):
                return
            for aug in aug_list:
                if not isinstance(aug, dict):
                    continue
                aug_type = (aug.get('type') or '').lower()
                # These augmentation types have a 'size' parameter
                if aug_type in ('random_resized_crop', 'center_crop', 'resize', 'random_crop'):
                    if 'params' not in aug:
                        aug['params'] = {}
                    aug['params']['size'] = target_size
        
        _override_sizes(train_aug_cfg, image_size)
        _override_sizes(val_aug_cfg, image_size)

        if isinstance(train_aug_cfg, list) and train_aug_cfg:
            train_transform = DataAugmentationFactory.create_composition(train_aug_cfg)
        else:
            train_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        if isinstance(val_aug_cfg, list) and val_aug_cfg:
            val_transform = DataAugmentationFactory.create_composition(val_aug_cfg)
        else:
            val_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        if 'task' not in self.config:
            raise ValueError("Missing required 'config.task' value initialized from config/config.json")
        is_multi_label = (self.config['task'] == 'multi_label')
        if 'labels' not in self.config:
            raise ValueError("Missing required 'config.labels' initialized from dataset discovery")
        # labels is now dict {idx: name} - create name->idx mapping for datasets
        labels_dict: Dict[int, str] = self.config['labels']
        labels_map: Dict[str, int] = {name: idx for idx, name in labels_dict.items()}
        label_count = len(labels_dict)

        if self.config['dataset_path'] and os.path.exists(self.config['dataset_path']):
            train_path = os.path.join(self.config['dataset_path'], 'train')
            val_path = os.path.join(self.config['dataset_path'], 'val')

            if os.path.exists(train_path) and os.path.exists(val_path):
                if not is_multi_label:
                    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
                    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
                    # CRITICAL: Sync our labels_dict with ImageFolder's class_to_idx
                    # ImageFolder determines indices from sorted folder names
                    # We must use THE SAME mapping for checkpoint labels!
                    folder_class_to_idx = train_dataset.class_to_idx  # {"cat": 0, "dog": 1}
                    labels_dict = {idx: name for name, idx in folder_class_to_idx.items()}
                    labels_map = folder_class_to_idx
                    label_count = len(labels_dict)
                    self.config['labels'] = labels_dict
                    self.config['num_classes'] = label_count
                    info(f"Single-label mode: synced labels with ImageFolder class_to_idx: {labels_dict}")
                else:
                    train_dataset = MultiLabelFolderDataset(train_path, labels_map, label_count, transform=train_transform)
                    val_dataset = MultiLabelFolderDataset(val_path, labels_map, label_count, transform=val_transform)
            else:
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.bmp', '*.pgm', '*.tif', '*.tiff', '*.webp']
                all_images: List[str] = []
                for ext in image_extensions:
                    all_images.extend(glob.glob(os.path.join(self.config['dataset_path'], '**', ext), recursive=True))
                if not all_images:
                    raise ValueError(f"No valid images found in dataset: {self.config['dataset_path']}")

                from collections import defaultdict
                class_images: Dict[str, List[str]] = defaultdict(list)
                for img_path in all_images:
                    class_name = os.path.basename(os.path.dirname(img_path))
                    class_images[class_name].append(img_path)

                try:
                    min_images = self.settings['dataset']['discovery']['balance_analysis']['min_images_per_class']
                except Exception:
                    raise ValueError("Missing 'dataset.discovery.balance_analysis.min_images_per_class' in config/config.json")
                valid_classes = {cls: imgs for cls, imgs in class_images.items() if len(imgs) >= min_images}
                if not valid_classes:
                    raise ValueError("No classes found with sufficient images (minimum 5 images per class)")

                # CRITICAL: Sort class names for deterministic index assignment
                sorted_class_names = sorted(valid_classes.keys())
                
                all_image_paths: List[str] = []
                all_labels: List[int] = []
                class_to_idx: Dict[str, int] = {}
                for idx, cls_name in enumerate(sorted_class_names):
                    class_to_idx[cls_name] = idx
                    for p in valid_classes[cls_name]:
                        all_image_paths.append(p)
                        all_labels.append(idx)

                if not is_multi_label:
                    # Update labels_dict to match class_to_idx (deterministic sorted order)
                    labels_dict = {idx: name for name, idx in class_to_idx.items()}
                    labels_map = class_to_idx
                    label_count = len(labels_dict)
                    self.config['labels'] = labels_dict
                    self.config['num_classes'] = label_count
                    info(f"Single-label flat mode: using sorted class order: {labels_dict}")

                if not is_multi_label:
                    base_dataset = CustomImageDataset(all_image_paths, all_labels, train_transform)
                    base_dataset_val = CustomImageDataset(all_image_paths, all_labels, val_transform)
                else:
                    base_dataset = MultiLabelDataset(all_image_paths, labels_map, label_count, transform=train_transform)
                    base_dataset_val = MultiLabelDataset(all_image_paths, labels_map, label_count, transform=val_transform)

                if 'split' not in self.config or 'val_ratio' not in self.config['split']:
                    raise ValueError("Missing required 'config.split.val_ratio' to split dataset. Ensure 'training.val_ratio' is provided in config/config.json")
                val_ratio = float(self.config['split']['val_ratio'])
                total_size = len(base_dataset)
                train_size = int((1.0 - val_ratio) * total_size)
                val_size = total_size - train_size
                indices = torch.randperm(total_size).tolist()
                train_indices = indices[:train_size]
                val_indices = indices[train_size:]

                from torch.utils.data import Subset
                train_dataset = Subset(base_dataset, train_indices)
                val_dataset = Subset(base_dataset_val, val_indices)
        else:
            raise ValueError(f"No valid dataset found at path: {self.config['dataset_path']}. Please ensure the dataset path is correctly configured and contains image files.")

        try:
            dl_cfg = self.settings['training']['dataloader']
        except Exception:
            raise ValueError("Missing 'training.dataloader' in config/config.json")
        nw = int(self.config['dataloader']['num_workers'])
        if nw > 0:
            if 'prefetch_factor' not in dl_cfg:
                raise ValueError("Missing 'training.dataloader.prefetch_factor' in config/config.json")
            if 'persistent_workers' not in dl_cfg:
                raise ValueError("Missing 'training.dataloader.persistent_workers' in config/config.json")
            prefetch_factor = int(dl_cfg['prefetch_factor'])
            persistent_workers = bool(dl_cfg['persistent_workers'])
        else:
            prefetch_factor = None
            persistent_workers = False

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=nw,
            pin_memory=self.config['dataloader']['pin_memory'],
            prefetch_factor=prefetch_factor if prefetch_factor else None,
            persistent_workers=persistent_workers
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=nw,
            pin_memory=self.config['dataloader']['pin_memory'],
            prefetch_factor=prefetch_factor if prefetch_factor else None,
            persistent_workers=persistent_workers
        )

        info(f"Prepared data loaders - Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
        return train_loader, val_loader

    def _prepare_training_components(self, model):
        try:
            training_cfg = self.settings['training']
        except Exception:
            raise ValueError("Missing required 'training' section in config/config.json")
        
        if 'optimizer_type' not in training_cfg:
            raise ValueError("Missing required 'training.optimizer_type' in config/config.json")
        optimizer_type = training_cfg['optimizer_type']
        optimizer_params_source = training_cfg.get('optimizer_params', {})
        optimizer_params = {}
        if isinstance(optimizer_params_source, dict):
            selected_params = optimizer_params_source.get(optimizer_type, {})
            if isinstance(selected_params, dict):
                optimizer_params.update(selected_params)
        lr_override = training_cfg.get('optimizer_lr', training_cfg.get('learning_rate'))
        if isinstance(lr_override, (int, float)):
            optimizer_params['lr'] = lr_override
        weight_decay_override = training_cfg.get('optimizer_weight_decay', training_cfg.get('weight_decay'))
        if isinstance(weight_decay_override, (int, float)):
            optimizer_params['weight_decay'] = weight_decay_override

        optimizer_config = None
        if optimizer_type:
            optimizer_config = {
                'type': optimizer_type,
                'params': optimizer_params
            }

        if 'scheduler_type' not in training_cfg:
            raise ValueError("Missing required 'training.scheduler_type' in config/config.json")
        scheduler_type = training_cfg['scheduler_type']
        scheduler_params_source = training_cfg.get('scheduler_params', {})
        scheduler_params = {}
        if isinstance(scheduler_params_source, dict):
            selected_scheduler = scheduler_params_source.get(scheduler_type, {})
            if isinstance(selected_scheduler, dict):
                scheduler_params.update(selected_scheduler)
        
        # Only apply step_size/gamma overrides for schedulers that use them
        schedulers_with_step_size = ('step_lr', 'multi_step_lr')
        schedulers_with_gamma = ('step_lr', 'multi_step_lr', 'exponential_lr')
        
        if scheduler_type in schedulers_with_step_size:
            step_size_override = training_cfg.get('scheduler_step_size')
            if isinstance(step_size_override, int):
                scheduler_params['step_size'] = step_size_override
        
        if scheduler_type in schedulers_with_gamma:
            gamma_override = training_cfg.get('scheduler_gamma')
            if isinstance(gamma_override, (int, float)):
                scheduler_params['gamma'] = gamma_override

        scheduler_config = None
        if scheduler_type:
            scheduler_config = {
                'type': scheduler_type,
                'params': scheduler_params
            }

        if 'loss_type' not in training_cfg:
            raise ValueError("Missing required 'training.loss_type' in config/config.json")
        loss_type = training_cfg['loss_type']
        loss_params_source = training_cfg.get('loss_params', {})
        loss_params = {}
        if isinstance(loss_params_source, dict):
            selected_loss = loss_params_source.get(loss_type, {})
            if isinstance(selected_loss, dict):
                loss_params.update(selected_loss)
        reduction_override = training_cfg.get('loss_reduction')
        if isinstance(reduction_override, str):
            loss_params['reduction'] = reduction_override

        loss_config = None
        if loss_type:
            loss_config = {'type': loss_type, 'params': loss_params}
        
        optimizer = get_optimizer_for_model(
            model.model if hasattr(model, 'model') else model,
            optimizer_config=optimizer_config,
            use_case='computer_vision'
        )

        scheduler = get_scheduler_for_training(
            optimizer,
            scheduler_config=scheduler_config,
            scenario='standard_training'
        )

        criterion = get_loss_for_task(
            loss_config=loss_config,
            task_type=self.config['task']
        )

        info("Prepared training components (optimizer, scheduler, criterion)")
        return optimizer, scheduler, criterion

    def get_training_summary(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'batch_size': training_config['config']['batch_size'],
            'epochs': training_config['config']['epochs'],
            'learning_rate': training_config['config']['learning_rate'],
            'dataset_path': training_config['config']['dataset_path'],
            'num_classes': training_config['config']['num_classes'],
            'memory_status': training_config['memory_info'],
            'batch_calculation': training_config['batch_calculation'],
            'model_info': training_config['model'].get_model_info()
        }


def create_coordinator(model_type: str = 'resnet', model_name: str = 'resnet50') -> TrainingCoordinator:
    return TrainingCoordinator(model_type, model_name)


def get_supported_models() -> Dict[str, List[str]]:
    from system.models.resnet import get_supported_resnet_variants

    return {
        'resnet': get_supported_resnet_variants()
    }