"""Coordinator module for Hootsight.

Coordinates all components needed for model training including memory management,
data loading, model configuration, and training setup.
"""

from typing import Dict, Any, Optional, Tuple, List
import os
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from system.log import info, success, warning, error
from system.coordinator_settings import SETTINGS
from system.memory import get_optimal_batch_size, get_memory_status
from system.dataset_discovery import get_project_info
from system.augmentation import DataAugmentationFactory
from system.optimizers import get_optimizer_for_model
from system.schedulers import get_scheduler_for_training
from system.losses import get_loss_for_task
from system.weight_init import WeightInitFactory


class TrainingCoordinator:
    """Coordinates all components for model training."""

    def __init__(self, model_type: str = 'resnet', model_name: str = 'resnet50'):
        """Initialize training coordinator.

        Args:
            model_type: Type of model ('resnet', etc.)
            model_name: Specific model name ('resnet50', etc.)
        """
        self.model_type = model_type
        self.model_name = model_name
        self.settings = SETTINGS.copy() if SETTINGS else {}
        self.config = {}  # Will be set in prepare_training
        self.memory_config = {}  # Will be set in prepare_training
        self.model_config = {}  # Will be set in prepare_training

        info(f"Initialized coordinator for {model_type}/{model_name}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from settings."""
        training = self.settings.get('training', {})
        # Support both nested training.input.* and flattened training.input_size/normalize
        input_block = training.get('input', {}) if isinstance(training.get('input'), dict) else {}
        input_size = input_block.get('image_size', training.get('input_size', 224))
        normalize_cfg = input_block.get('normalize', training.get('normalize', {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }))
        dataloader_cfg = training.get('dataloader', {})
        # Resolve num_workers (supports "auto")
        raw_workers = dataloader_cfg.get('num_workers', 0)
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
        # Runtime/performance flags
        runtime_cfg = training.get('runtime', {}) if isinstance(training.get('runtime'), dict) else {}
        runtime_flags = {
            'mixed_precision': bool(runtime_cfg.get('mixed_precision', True)),
            'channels_last': bool(runtime_cfg.get('channels_last', True)),
            'allow_tf32': bool(runtime_cfg.get('allow_tf32', True)),
            'cudnn_benchmark': bool(runtime_cfg.get('cudnn_benchmark', True))
        }
        # Resolve epochs to an integer (handle "auto" or invalid)
        raw_epochs = training.get('epochs', 10)
        try:
            epochs = int(raw_epochs)
            if epochs <= 0:
                epochs = 10
        except Exception:
            epochs = 10
        # Support val_ratio at training root or nested under split
        split_block = training.get('split', {}) if isinstance(training.get('split'), dict) else {}
        val_ratio = training.get('val_ratio', split_block.get('val_ratio', 0.2))

        return {
            'batch_size': training.get('batch_size', 32),
            'epochs': epochs,
            'learning_rate': training.get('learning_rate', 0.001),
            'weight_decay': training.get('weight_decay', 1e-4),
            'task': training.get('task', 'classification'),
            'model_type': self.model_type,  # Use instance variable instead of SETTINGS
            'pretrained': training.get('model', {}).get('pretrained', True),
            'projects_base_dir': self.settings.get('paths', {}).get('projects_dir') or self.settings.get('paths', {}).get('models_dir', 'projects'),
            'dataset_path': None,  # Will be set by prepare_training
            'output_dir': None,  # Will be set by prepare_training
            'input': {
                'channels': 3,
                'image_size': int(input_size),
                'normalize': normalize_cfg
            },
            'dataloader': {
                'num_workers': resolved_workers,
                'pin_memory': bool(dataloader_cfg.get('pin_memory', True))
            },
            'runtime': runtime_flags,
            'split': {'val_ratio': float(val_ratio)}
        }

    def _apply_runtime_settings(self) -> None:
        """Apply global PyTorch runtime performance settings."""
        runtime = self.config.get('runtime', {})
        
        # TF32 settings for faster matrix multiplications on Ampere+ GPUs
        if runtime.get('allow_tf32', True):
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                info("Enabled TF32 for faster computations")
            except Exception as e:
                warning(f"Could not enable TF32: {e}")
        
        # cuDNN benchmark for optimized convolution algorithms
        if runtime.get('cudnn_benchmark', True):
            try:
                torch.backends.cudnn.benchmark = True
                info("Enabled cuDNN benchmark for optimized convolutions")
            except Exception as e:
                warning(f"Could not enable cuDNN benchmark: {e}")

    def _get_memory_config(self) -> Dict[str, Any]:
        """Get memory management configuration."""
        memory_settings = self.settings.get('memory', {})
        return {
            'target_usage': memory_settings.get('target_memory_usage', 0.8),
            'safety_margin': memory_settings.get('safety_margin', 0.9),
            'augmentation_threads': memory_settings.get('augmentation_threads', 'auto')
        }

    def _get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        model_type = self.config.get('model_type', 'resnet')
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
        """Prepare all components for training.

        Args:
            project_name: Name of the project/dataset to use

        Returns:
            dict: Prepared training configuration
        """
        info(f"Preparing training for project: {project_name}")

        # Load project-specific settings if exist
        project_config_path = os.path.join(self.settings.get('paths', {}).get('projects_dir', 'projects'), project_name, 'config.json')
        if os.path.exists(project_config_path):
            try:
                with open(project_config_path, 'r', encoding='utf-8') as f:
                    project_settings = json.load(f)
                # Merge project settings with global settings (project overrides)
                from system.common.deep_merge import deep_merge_json
                self.settings = deep_merge_json(self.settings, project_settings)
                info(f"Loaded project settings from {project_config_path}")
            except Exception as e:
                warning(f"Failed to load project settings: {e}")
        else:
            # Save current settings to project
            try:
                os.makedirs(os.path.dirname(project_config_path), exist_ok=True)
                with open(project_config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.settings, f, indent=2, ensure_ascii=False)
                info(f"Saved current settings to {project_config_path}")
            except Exception as e:
                warning(f"Failed to save project settings: {e}")

        # Load configuration from settings
        self.config = self._load_config()
        self.memory_config = self._get_memory_config()
        self.model_config = self._get_model_config()

        # Apply global PyTorch performance settings
        self._apply_runtime_settings()

        # Get project information
        project_info = get_project_info(project_name)
        if not project_info:
            raise ValueError(f"Project not found: {project_name}")

        self.config['dataset_path'] = project_info.dataset_path
        # Only include base labels (exclude folder: prefixed ones)
        base_labels = [lbl for lbl in project_info.labels if not str(lbl).startswith('folder:')]
        labels = base_labels if base_labels else project_info.labels
        self.config['labels'] = labels
        self.config['num_classes'] = len(labels)
        # Output dir under project (fixed to 'model' to keep it simple/minimal)
        ckpt_dir_name = 'model'
        self.config['output_dir'] = os.path.join(self.config['projects_base_dir'], project_name, ckpt_dir_name)

        # Calculate optimal batch size
        optimal_batch = self._calculate_optimal_batch_size()
        self.config['batch_size'] = optimal_batch['optimal_batch_size']

        # Prepare model
        model = self._prepare_model()

        # Prepare data loaders
        train_loader, val_loader = self._prepare_data_loaders()

        # Prepare training components
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
        """Calculate optimal batch size based on memory constraints."""
        # Create a dummy model for batch size calculation
        model_type = self.config.get('model_type', 'resnet')
        if model_type == 'resnet':
            from system.models.resnet import create_resnet_model, get_resnet_config
            dummy_model = create_resnet_model(self.model_name, self.config['num_classes'], pretrained=False, task=self.config.get('task', 'classification'))
            config_func = get_resnet_config
        elif model_type == 'resnext':
            from system.models.resnext import create_resnext_model, get_resnext_config
            dummy_model = create_resnext_model(self.model_name, self.config['num_classes'], pretrained=False, task=self.config.get('task', 'classification'))
            config_func = get_resnext_config
        elif model_type == 'mobilenet':
            from system.models.mobilenet import create_mobilenet_model, get_mobilenet_config
            dummy_model = create_mobilenet_model(self.model_name, self.config['num_classes'], pretrained=False, task=self.config.get('task', 'classification'))
            config_func = get_mobilenet_config
        elif model_type == 'shufflenet':
            from system.models.shufflenet import create_shufflenet_model, get_shufflenet_config
            dummy_model = create_shufflenet_model(self.model_name, self.config['num_classes'], pretrained=False, task=self.config.get('task', 'classification'))
            config_func = get_shufflenet_config
        elif model_type == 'squeezenet':
            from system.models.squeezenet import create_squeezenet_model, get_squeezenet_config
            dummy_model = create_squeezenet_model(self.model_name, self.config['num_classes'], pretrained=False, task=self.config.get('task', 'classification'))
            config_func = get_squeezenet_config
        elif model_type == 'efficientnet':
            from system.models.efficientnet import create_efficientnet_model, get_efficientnet_config
            dummy_model = create_efficientnet_model(self.model_name, self.config['num_classes'], pretrained=False, task=self.config.get('task', 'classification'))
            config_func = get_efficientnet_config
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Get recommended batch size from config
        try:
            model_config = config_func(self.model_name)
            recommended_batch = model_config.get('recommended_batch_size', 32)
        except:
            recommended_batch = 32

        # Estimate input shape from config
        channels = int(self.config['input'].get('channels', 3))
        size = int(self.config['input'].get('image_size', 224))
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
        """Prepare model for training."""
        model_type = self.config.get('model_type', 'resnet')
        model_name = self.model_name
        num_classes = self.config['num_classes']
        pretrained = bool(self.config.get('pretrained', True))
        task = self.config.get('task', 'classification')
        runtime = self.config.get('runtime', {})

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

        # Apply runtime/performance flags
        try:
            import torch
            if runtime.get('allow_tf32', False):
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
                    torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
                except Exception:
                    pass
            if runtime.get('cudnn_benchmark', False):
                try:
                    torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
                except Exception:
                    pass
            if runtime.get('channels_last', False):
                try:
                    if hasattr(model, 'model'):
                        # Convert parameters and buffers to channels_last memory format
                        model.model = model.model.to(self_device := (model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
                        for p in model.model.parameters():
                            try:
                                p.data = p.data.contiguous(memory_format=torch.channels_last)
                            except Exception:
                                break
                except Exception:
                    pass
        except Exception:
            pass

        # Optional weight initialization
        try:
            train_cfg = self.settings.get('training', {})
            init_cfg = train_cfg.get('weight_init')
            if init_cfg:
                # ResNetModel wrapper exposes underlying nn.Module as 'model'
                WeightInitFactory.apply_to_model(model.model, init_cfg)
        except Exception as ex:
            warning(f"Weight initialization skipped: {ex}")

        # Apply coordinator runtime toggles to the model wrapper if available (ResNet)
        try:
            from system.models.resnet import ResNetModel  # type: ignore
            runtime = self.config.get('runtime', {})
            if isinstance(model, ResNetModel):
                cuda_enabled = bool(getattr(model, 'device', None) and getattr(model, 'device').type == 'cuda')
                model.use_amp = bool(runtime.get('mixed_precision', True) and cuda_enabled)
                if cuda_enabled:
                    import torch
                    model._scaler = torch.cuda.amp.GradScaler(enabled=bool(model.use_amp))
                model.channels_last = bool(runtime.get('channels_last', True))
        except Exception:
            pass

        info(f"Prepared {self.model_type} model: {model.get_model_info()}")
        return model

    def _prepare_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders for training and validation."""
        # Local imports to keep global namespace clean
        from torchvision import datasets, transforms
        from torch.utils.data import random_split, Dataset
        from PIL import Image
        import glob, os
        import torch

        # Define transforms from config when provided
        image_size = int(self.config['input']['image_size'])
        mean = self.config['input']['normalize']['mean']
        std = self.config['input']['normalize']['std']

        train_aug_cfg = self.settings.get('training', {}).get('augmentation', {}).get('train')
        val_aug_cfg = self.settings.get('training', {}).get('augmentation', {}).get('val')

        if isinstance(train_aug_cfg, list) and train_aug_cfg:
            train_transform = DataAugmentationFactory.create_composition(train_aug_cfg)
        else:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        if isinstance(val_aug_cfg, list) and val_aug_cfg:
            val_transform = DataAugmentationFactory.create_composition(val_aug_cfg)
        else:
            resize_size = max(image_size + 32, image_size)
            val_transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        # Decide whether to build multi-label dataset
        is_multi_label = (self.config.get('task') == 'multi_label')
        label_list: List[str] = list(self.config.get('labels', []))

        # Dataset resolution
        if self.config['dataset_path'] and os.path.exists(self.config['dataset_path']):
            train_path = os.path.join(self.config['dataset_path'], 'train')
            val_path = os.path.join(self.config['dataset_path'], 'val')

            if os.path.exists(train_path) and os.path.exists(val_path):
                if not is_multi_label:
                    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
                    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
                else:
                    def read_labels_for_image(img_path: str) -> List[str]:
                        txt_path = os.path.splitext(img_path)[0] + '.txt'
                        labels_for_img: List[str] = []
                        if os.path.exists(txt_path):
                            try:
                                with open(txt_path, 'r', encoding='utf-8') as f:
                                    for line in f:
                                        line = line.strip()
                                        if line:
                                            # Split comma-separated tags and strip whitespace
                                            parts = [p.strip() for p in line.split(',') if p.strip()]
                                            labels_for_img.extend(parts)
                            except Exception:
                                pass
                        if not labels_for_img:
                            labels_for_img = [os.path.basename(os.path.dirname(img_path))]
                        return labels_for_img

                    class MultiLabelFolderDataset(Dataset):
                        def __init__(self, root: str, transform=None):
                            self.transform = transform
                            exts = ['*.jpg','*.jpeg','*.png','*.bmp','*.gif','*.tiff','*.webp']
                            self.images: List[str] = []
                            for ext in exts:
                                self.images.extend(glob.glob(os.path.join(root, '**', ext), recursive=True))
                            self.images = [p for p in self.images if os.path.isfile(p)]
                            self.labels_map = {name: idx for idx, name in enumerate(label_list)}

                        def __len__(self) -> int:
                            return len(self.images)

                        def __getitem__(self, idx: int):
                            path = self.images[idx]
                            img = Image.open(path).convert('RGB')
                            labs = read_labels_for_image(path)
                            target = torch.zeros(len(label_list), dtype=torch.float32)
                            for lab in labs:
                                if lab in self.labels_map:
                                    target[self.labels_map[lab]] = 1.0
                            if self.transform:
                                img = self.transform(img)
                            return img, target

                    train_dataset = MultiLabelFolderDataset(train_path, transform=train_transform)
                    val_dataset = MultiLabelFolderDataset(val_path, transform=val_transform)
            else:
                # No explicit split; scan and split
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

                min_images = self.settings.get('dataset', {}).get('discovery', {}).get('balance_analysis', {}).get('min_images_per_class', 5)
                valid_classes = {cls: imgs for cls, imgs in class_images.items() if len(imgs) >= min_images}
                if not valid_classes:
                    raise ValueError("No classes found with sufficient images (minimum 5 images per class)")

                class CustomImageDataset(Dataset):
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

                all_image_paths: List[str] = []
                all_labels: List[int] = []
                class_to_idx: Dict[str, int] = {}
                for idx, (cls_name, images) in enumerate(valid_classes.items()):
                    class_to_idx[cls_name] = idx
                    for p in images:
                        all_image_paths.append(p)
                        all_labels.append(idx)

                if not is_multi_label:
                    self.config['num_classes'] = len(valid_classes)

                if not is_multi_label:
                    base_dataset = CustomImageDataset(all_image_paths, all_labels, train_transform)
                    base_dataset_val = CustomImageDataset(all_image_paths, all_labels, val_transform)
                else:
                    labels_map = {name: idx for idx, name in enumerate(label_list)}

                    class MultiLabelDataset(Dataset):
                        def __init__(self, paths: List[str], transform=None):
                            self.paths = paths
                            self.transform = transform
                        def __len__(self) -> int:
                            return len(self.paths)
                        def __getitem__(self, i: int):
                            p = self.paths[i]
                            img = Image.open(p).convert('RGB')
                            txt = os.path.splitext(p)[0] + '.txt'
                            labs: List[str] = []
                            if os.path.exists(txt):
                                try:
                                    with open(txt, 'r', encoding='utf-8') as f:
                                        for line in f:
                                            line = line.strip()
                                            if line:
                                                parts = [s.strip() for s in line.split(',') if s.strip()]
                                                labs.extend(parts)
                                except Exception:
                                    pass
                            if not labs:
                                labs = [os.path.basename(os.path.dirname(p))]
                            target = torch.zeros(len(label_list), dtype=torch.float32)
                            for lab in labs:
                                if lab in labels_map:
                                    target[labels_map[lab]] = 1.0
                            if self.transform:
                                img = self.transform(img)
                            return img, target

                    base_dataset = MultiLabelDataset(all_image_paths, transform=train_transform)
                    base_dataset_val = MultiLabelDataset(all_image_paths, transform=val_transform)

                # Split for train/val using a single index split applied consistently to both views
                val_ratio = float(self.config.get('split', {}).get('val_ratio', 0.2))
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
            # No dataset available - raise error instead of using dummy data
            raise ValueError(f"No valid dataset found at path: {self.config['dataset_path']}. Please ensure the dataset path is correctly configured and contains image files.")

        # Create data loaders
        dl_cfg = self.settings.get('training', {}).get('dataloader', {})
        nw = int(self.config['dataloader']['num_workers'])
        prefetch_factor = int(dl_cfg.get('prefetch_factor', 2)) if nw > 0 else None
        persistent_workers = bool(dl_cfg.get('persistent_workers', False)) if nw > 0 else False

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
        """Prepare optimizer, scheduler, and criterion."""
        training_cfg = self.settings.get('training', {})
        
        optimizer_type = training_cfg.get('optimizer_type', 'adamw')
        optimizer_params_source = training_cfg.get('optimizer_params', {})
        optimizer_params = {}
        if isinstance(optimizer_params_source, dict):
            selected_params = optimizer_params_source.get(optimizer_type, {})
            if isinstance(selected_params, dict):
                optimizer_params.update(selected_params)
        # Apply flattened overrides for compatibility
        lr_override = training_cfg.get('optimizer_lr', training_cfg.get('learning_rate'))
        if isinstance(lr_override, (int, float)):
            optimizer_params['lr'] = lr_override
        weight_decay_override = training_cfg.get('optimizer_weight_decay', training_cfg.get('weight_decay'))
        if isinstance(weight_decay_override, (int, float)):
            optimizer_params['weight_decay'] = weight_decay_override

        optimizer_config = {
            'type': optimizer_type,
            'params': optimizer_params
        }

        scheduler_type = training_cfg.get('scheduler_type', 'step_lr')
        scheduler_params_source = training_cfg.get('scheduler_params', {})
        scheduler_params = {}
        if isinstance(scheduler_params_source, dict):
            selected_scheduler = scheduler_params_source.get(scheduler_type, {})
            if isinstance(selected_scheduler, dict):
                scheduler_params.update(selected_scheduler)
        step_size_override = training_cfg.get('scheduler_step_size')
        if isinstance(step_size_override, int):
            scheduler_params['step_size'] = step_size_override
        gamma_override = training_cfg.get('scheduler_gamma')
        if isinstance(gamma_override, (int, float)):
            scheduler_params['gamma'] = gamma_override

        scheduler_config = {
            'type': scheduler_type,
            'params': scheduler_params
        }

        loss_type = training_cfg.get('loss_type', 'cross_entropy')
        loss_params_source = training_cfg.get('loss_params', {})
        loss_params = {}
        if isinstance(loss_params_source, dict):
            selected_loss = loss_params_source.get(loss_type, {})
            if isinstance(selected_loss, dict):
                loss_params.update(selected_loss)
        reduction_override = training_cfg.get('loss_reduction')
        if isinstance(reduction_override, str):
            loss_params['reduction'] = reduction_override

        loss_config = {
            'type': loss_type,
            'params': loss_params
        }
        
        # Optimizer from config, fallback to model defaults
        try:
            optimizer = get_optimizer_for_model(
                model.model if hasattr(model, 'model') else model,
                optimizer_config=optimizer_config,
                use_case='computer_vision'
            )
        except Exception as ex:
            warning(f"Optimizer from config failed, falling back: {ex}")
            optimizer = model.get_optimizer(lr=self.config['learning_rate'])

        # Scheduler from config
        try:
            scheduler = get_scheduler_for_training(
                optimizer,
                scheduler_config=scheduler_config,
                scenario='standard_training'
            )
        except Exception as ex:
            warning(f"Scheduler from config failed, falling back: {ex}")
            scheduler = model.get_scheduler(optimizer)

        # Loss from config
        try:
            criterion = get_loss_for_task(
                loss_config=loss_config,
                task_type=self.config.get('task', 'classification')
            )
        except Exception as ex:
            warning(f"Loss from config failed, falling back: {ex}")
            criterion = model.get_criterion()

        info("Prepared training components (optimizer, scheduler, criterion)")
        return optimizer, scheduler, criterion

    def get_training_summary(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of training configuration."""
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
    """Factory function to create training coordinator.

    Args:
        model_type: Type of model ('resnet', etc.)
        model_name: Specific model name ('resnet50', etc.)

    Returns:
        TrainingCoordinator: Configured coordinator instance
    """
    return TrainingCoordinator(model_type, model_name)


def get_supported_models() -> Dict[str, List[str]]:
    """Get dictionary of supported model types and their variants."""
    from system.models.resnet import get_supported_resnet_variants

    return {
        'resnet': get_supported_resnet_variants()
    }