# Configuration File Guide

This guide provides a comprehensive overview of Hootsight's configuration system. The `config/config.json` file contains all system settings, organized hierarchically for clarity and ease of modification.

## Overview

Hootsight uses a single configuration file (`config/config.json`) for all settings. This file is divided into logical sections, each controlling different aspects of the system. You can override settings on a per-project basis by placing a `config.json` file in your project directory.

**Important Note**: This default configuration serves as both an initialization template and a reference guide for new systems. While the defaults are optimized for most use cases, understanding these settings allows for fine-tuning Hootsight to specific hardware configurations and training requirements.

## Configuration Sections

### General Settings

```json
"general": {
    "language": "en"
}
```

- **`language`**: The default language for the user interface. 
  - **Available values**: Currently only `"en"` (English) is supported in the schema
  - **Default**: `"en"`
  - **System impact**: Controls all UI text, button labels, messages, and documentation language. Changes take effect immediately without restart.
  - **When to change**: If you need a different language interface (though only English is currently available in the schema)

### API Settings

```json
"api": {
    "host": "127.0.0.1",
    "port": 8000
}
```

- **`host`**: The IP address where the FastAPI server binds.
  - **Available values**: Any valid hostname or IP address
  - **Default**: `"127.0.0.1"` (localhost only)
  - **Common options**: `"127.0.0.1"` (local only), `"0.0.0.0"` (all interfaces), specific IP addresses
  - **System impact**: Controls which network interfaces can access the API. Localhost restricts to same machine only.
  - **When to change**: To allow remote access from other computers on your network. **Security note**: Opening to all interfaces requires firewall configuration.

- **`port`**: The port number for the API server.
  - **Available values**: Integer from 1 to 65535
  - **Default**: `8000`
  - **System impact**: Determines the URL endpoint (http://localhost:PORT). All API calls and web interface access use this port.
  - **When to change**: If port 8000 is already in use by another application, or for security reasons to use a non-standard port.

### UI Settings

```json
"ui": {
    "title": "Hootsight",
    "width": 1200,
    "height": 800,
    "resizable": true
}
```

- **`title`**: The window title displayed in the application window.
  - **Available values**: Any string between 1-100 characters
  - **Default**: `"Hootsight"`
  - **System impact**: Appears in window title bar, taskbar, and system window lists. Purely cosmetic.
  - **When to change**: For branding or to distinguish multiple instances.

- **`width`**: Initial window width in pixels.
  - **Available values**: Integer from 800 to 3840 pixels
  - **Default**: `1200`
  - **System impact**: Determines initial window size when application starts. Must fit within screen resolution.
  - **When to change**: For different screen sizes or user preferences. Large displays benefit from larger windows.

- **`height`**: Initial window height in pixels.
  - **Available values**: Integer from 600 to 2160 pixels
  - **Default**: `800`
  - **System impact**: Sets vertical window size at startup. Affects how much content is visible initially.
  - **When to change**: Match your monitor resolution or workflow needs.

- **`resizable`**: Whether users can resize the window.
  - **Available values**: `true` or `false`
  - **Default**: `true`
  - **System impact**: Controls window resize handles and behavior. False creates fixed-size windows.
  - **When to change**: Set to false for kiosk mode or consistent UI layouts across users.

### Path Settings

```json
"paths": {
    "projects_dir": "projects",
    "ui_dir": "ui",
    "config_dir": "config",
    "localizations_dir": "config/localizations",
    "packages_file": "config/packages.jsonc",
    "mappings_file": "config/mappings.json",
    "cache_dir": "cache"
}
```

These define relative paths to various directories and files. **Do not change these unless you understand the system structure**, as incorrect paths will break functionality. System impact: Critical for file system operations, package management, and resource loading across all system components.

### System Settings

```json
"system": {
    "max_threads": "auto",
    "fallback_batch_size": 8,
    "memory_cleanup_interval": 300,
    "thread_pool_timeout": 30,
    "startup_wait_seconds": 2
}
```

- **`max_threads`**: Maximum number of threads for concurrent system operations.
  - **Available values**: `"auto"` or integer from 1 to 128
  - **Default**: `"auto"` (uses CPU cores - 1)
  - **System impact**: Controls parallel processing capacity. More threads can improve performance but consume more memory and CPU scheduling overhead.
  - **When to change**: Reduce for stability on older systems, increase for high-core-count CPUs with heavy workloads.

- **`fallback_batch_size`**: Safety batch size when automatic calculation fails.
  - **Available values**: Integer from 1 to 512
  - **Default**: `8`
  - **System impact**: Used when memory-based batch size calculation cannot determine optimal size. Prevents training from failing due to memory constraints.
  - **When to change**: Lower for very limited memory systems, higher for systems with abundant memory.

- **`memory_cleanup_interval`**: Seconds between automatic garbage collection cycles.
  - **Available values**: Integer from 60 to 3600 seconds (1 minute to 1 hour)
  - **Default**: `300` (5 minutes)
  - **System impact**: Balances memory efficiency vs performance overhead. More frequent cleanup uses more CPU but frees memory sooner.
  - **When to change**: Shorter intervals for memory-constrained systems, longer for performance-critical scenarios.

- **`thread_pool_timeout`**: Maximum wait time for background operations.
  - **Available values**: Integer from 1 to 300 seconds
  - **Default**: `30` seconds
  - **System impact**: Prevents hung background tasks from blocking system responsiveness. Affects reliability of concurrent operations.
  - **When to change**: Increase for slow storage systems, decrease for faster response requirements.

- **`startup_wait_seconds`**: Initialization delay before starting services.
  - **Available values**: Integer from 0 to 60 seconds
  - **Default**: `2` seconds
  - **System impact**: Allows components to initialize properly and prevents startup race conditions.
  - **When to change**: Increase on slow systems with startup issues, set to 0 for faster development cycles.

### Memory Management

```json
"memory": {
    "target_memory_usage": 0.8,
    "safety_margin": 0.9,
    "augmentation_threads": "auto"
}
```

- **`target_memory_usage`**: Target memory utilization percentage for training operations.
  - **Available values**: Number from 0.1 to 1.0 (10% to 100%)
  - **Default**: `0.8` (80%)
  - **System impact**: Controls how aggressively the system uses available GPU/CPU memory. Higher values allow larger batch sizes but risk out-of-memory errors.
  - **When to change**: Lower (0.6-0.7) for systems with limited memory or background processes, higher (0.85-0.9) for dedicated training systems.

- **`safety_margin`**: Safety buffer for memory calculations.
  - **Available values**: Number from 0.1 to 1.0 (10% to 100%)
  - **Default**: `0.9` (90% - assumes 10% overhead)
  - **System impact**: Reduces effective memory calculations to account for system overhead and memory fragmentation. Lower values are more aggressive.
  - **When to change**: Lower (0.8) for well-controlled environments, higher (0.95) for systems with many background processes.

- **`augmentation_threads`**: Thread count for parallel data preprocessing.
  - **Available values**: `"auto"` or integer from 1 to 64
  - **Default**: `"auto"` (uses CPU cores - 1)
  - **System impact**: Controls parallel image preprocessing during training. More threads speed up data loading but compete with training for CPU resources.
  - **When to change**: Lower for CPU-intensive models or limited cores, higher for I/O-bound scenarios with fast storage.

### Training Configuration

The training section is the most complex and controls all aspects of model training.

#### Basic Training Settings

```json
"training": {
    "model_type": "resnet",
    "model_name": "resnet50",
    "pretrained": true,
    "task": "multi_label",
    "batch_size": "auto",
    "epochs": "auto",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "input_size": 224
}
```

- **`model_type`**: Architecture family. Options: "resnet", "resnext", "mobilenet", "shufflenet", "squeezenet", "efficientnet". Affects available model variants. System impact: Determines computational requirements, memory usage patterns, and inference performance characteristics.
- **`model_name`**: Specific model variant. Must match the selected model_type. Larger models are more accurate but slower and use more memory. System impact: Directly controls model complexity, parameter count, and computational requirements.
- **`pretrained`**: Whether to use ImageNet-pretrained weights. true for better accuracy, false for training from scratch. System impact: Affects training convergence speed and final model accuracy through transfer learning.
- **`task`**: Training task type. "multi_label" for multi-label classification. This affects how the model processes outputs. System impact: Determines loss function selection and output processing pipeline.
- **`batch_size`**: Number of images processed simultaneously during training.
  - **Available values**: `"auto"` or integer from 1 to 512
  - **Default**: `"auto"` (memory-based calculation)
  - **System impact**: Larger batches provide more stable gradients and faster training but require more memory. Affects training speed and convergence quality.
  - **When to change**: Use specific numbers to override memory calculations, reduce for memory constraints, increase for stable training on large datasets.

- **`epochs`**: Number of complete passes through the training dataset.
  - **Available values**: `"auto"` or integer from 1 to 1000
  - **Default**: `"auto"` (uses early stopping)
  - **System impact**: More epochs allow better learning but risk overfitting. Training time scales linearly with epoch count.
  - **When to change**: Set specific numbers to override early stopping, reduce for quick experiments, increase for complex datasets.

- **`learning_rate`**: Step size for parameter updates during optimization.
  - **Available values**: Number from 1e-8 to 1.0
  - **Default**: `0.001`
  - **System impact**: Higher values speed up learning but may cause instability. Lower values train slower but more reliably.
  - **When to change**: Increase for faster convergence (with caution), decrease for fine-tuning or unstable training.

- **`weight_decay`**: L2 regularization strength to prevent overfitting.
  - **Available values**: Number from 0.0 to 1.0
  - **Default**: `0.0001`
  - **System impact**: Penalizes large weights, encouraging simpler models. Higher values provide stronger regularization but may hurt performance.
  - **When to change**: Increase for overfitting datasets, decrease for underfitting or small datasets.

- **`input_size`**: Size in pixels for input images (square images assumed).
  - **Available values**: Integer from 32 to 1024 pixels
  - **Default**: `224` (standard ImageNet size)
  - **System impact**: Larger sizes capture more detail but use significantly more memory and processing time. Must be compatible with model architecture.
  - **When to change**: Increase for high-detail tasks, decrease for memory constraints or simple images.

#### Data Normalization

```json
"normalize": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
}
```

Standard ImageNet normalization values. **Do not change unless you have specific requirements**, as this affects pretrained model compatibility. System impact: Ensures proper input distribution for pretrained models and maintains feature scaling consistency.

#### Data Loading

```json
"dataloader": {
    "num_workers": 0,
    "pin_memory": true,
    "persistent_workers": false,
    "prefetch_factor": 2
}
```

- **`num_workers`**: Number of background processes for loading and preprocessing data.
  - **Available values**: Integer from 0 to 32
  - **Default**: `0` (uses main training process)
  - **System impact**: More workers parallelize data loading but use more CPU cores and memory. Can significantly speed up training if data loading is a bottleneck.
  - **When to change**: Increase (2-8) for large datasets or slow storage, keep at 0 for simple datasets or limited CPU cores.

- **`pin_memory`**: Whether to allocate data tensors in pinned (page-locked) memory.
  - **Available values**: `true` or `false`
  - **Default**: `true`
  - **System impact**: Eliminates memory copying during CPU-to-GPU transfers, speeding up training. Uses slightly more system memory.
  - **When to change**: Set to false only if experiencing memory pressure or CPU-only training.

- **`persistent_workers`**: Whether to keep data loading workers alive between epochs.
  - **Available values**: `true` or `false`
  - **Default**: `false`
  - **System impact**: Reduces worker initialization overhead in multi-epoch training but consumes memory continuously.
  - **When to change**: Set to true for long training runs with many epochs, false for single-epoch or memory-constrained scenarios.

- **`prefetch_factor`**: Number of batches each worker loads ahead of time.
  - **Available values**: Integer from 1 to 10
  - **Default**: `2`
  - **System impact**: Higher values reduce data loading latency by preparing batches in advance but use more memory per worker.
  - **When to change**: Increase for slow data loading, decrease to reduce memory usage with many workers.

#### Data Augmentation

```json
"augmentation": {
    "train": [...],
    "val": [...]
}
```

Arrays of transformation steps for training and validation data. Each step has "type" and "params". Available transforms include:
- `random_resized_crop`: Random cropping with resizing
- `random_horizontal_flip`: Random horizontal flipping
- `to_tensor`: Convert to PyTorch tensor
- `normalize`: Apply normalization

**Modifying augmentation affects data diversity and model generalization.** System impact: Controls training data variability and model robustness to input transformations.

#### Optimizer Configuration

```json
"optimizer_type": "adamw",
"optimizer_lr": 0.001,
"optimizer_weight_decay": 0.01,
"optimizer_params": {
    "adamw": {
        "lr": 0.001,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "amsgrad": false
    }
}
```

- **`optimizer_type`**: Optimizer algorithm. Options include "adam", "adamw", "sgd", etc. adamw is generally recommended. System impact: Determines parameter update strategy and convergence characteristics.
- **`optimizer_lr`** and **`optimizer_weight_decay`**: Convenience fields, overridden by optimizer_params. System impact: Provides quick access to common optimization parameters.
- **`optimizer_params`**: Detailed parameters for each optimizer type. **Refer to PyTorch documentation for parameter meanings.** System impact: Fine-tunes optimization behavior for specific model architectures and datasets.

#### Learning Rate Scheduler

```json
"scheduler_type": "step_lr",
"scheduler_step_size": 7,
"scheduler_gamma": 0.1,
"scheduler_params": {
    "step_lr": {
        "step_size": 7,
        "gamma": 0.1
    }
}
```

- **`scheduler_type`**: Learning rate schedule type. Options include "step_lr", "cosine_annealing", etc. System impact: Controls learning rate decay pattern and training convergence trajectory.
- **`scheduler_step_size`**: How often to reduce the learning rate (in epochs). System impact: Determines learning rate adjustment frequency.
- **`scheduler_gamma`**: How much to multiply the learning rate by (0.1 = divide by 10). System impact: Controls the magnitude of learning rate reductions.
- **`scheduler_params`**: Detailed parameters for each scheduler type. System impact: Enables precise control over learning rate scheduling behavior.

#### Loss Function

```json
"loss_type": "bce_with_logits",
"loss_reduction": "mean",
"loss_params": {
    "bce_with_logits": {
        "reduction": "mean"
    }
}
```

- **`loss_type`**: Loss function for training. "bce_with_logits" for multi-label classification. System impact: Defines the training objective and gradient computation.
- **`loss_reduction`**: How to reduce loss across batch. "mean" or "sum". System impact: Affects gradient scaling and optimization stability.
- **`loss_params`**: Parameters for the loss function. System impact: Fine-tunes loss computation for specific task requirements.

#### Weight Initialization

```json
"weight_init": {
    "type": "kaiming_normal",
    "params": {
        "a": 0,
        "mode": "fan_in",
        "nonlinearity": "leaky_relu"
    }
}
```

- **`type`**: Weight initialization method. "kaiming_normal" is recommended for ReLU networks. System impact: Affects initial parameter distribution and training stability.
- **`params`**: Parameters for the initialization method. System impact: Optimizes parameter initialization for specific activation functions.

#### Checkpoint Settings

```json
"checkpoint": {
    "save_best_only": true,
    "save_frequency": 1,
    "max_checkpoints": 5,
    "checkpoint_dir": "model",
    "best_model_filename": "best_model.pth",
    "training_history_filename": "training_history.json"
}
```

- **`save_best_only`**: Only save the best model. true saves disk space. System impact: Controls storage efficiency and model versioning strategy.
- **`save_frequency`**: Save every N epochs. 1 saves after every epoch. System impact: Balances recovery granularity with storage overhead.
- **`max_checkpoints`**: Maximum number of checkpoints to keep. System impact: Manages disk space usage in long training runs.
- **`checkpoint_dir`**: Directory for saving models. System impact: Determines file system organization for model artifacts.
- **`best_model_filename`** and **`training_history_filename`**: Filenames for saved models and training metrics. System impact: Establishes naming conventions for model persistence.

#### Early Stopping

```json
"early_stopping": {
    "enabled": false,
    "patience": 10,
    "min_delta": 0.001,
    "monitor": "val_loss"
}
```

- **`enabled`**: Whether to use early stopping. true stops training when validation loss stops improving. System impact: Prevents overfitting and optimizes training resource utilization.
- **`patience`**: Epochs to wait before stopping. System impact: Controls sensitivity to training fluctuations.
- **`min_delta`**: Minimum change to qualify as improvement. System impact: Sets threshold for meaningful performance improvements.
- **`monitor`**: Metric to monitor. "val_loss" monitors validation loss. System impact: Determines the criterion for training termination.

#### Gradient Settings

```json
"gradient": {
    "clip_norm": null,
    "clip_value": null,
    "accumulation_steps": 1
}
```

- **`clip_norm`**: Clip gradients by global norm. Prevents exploding gradients. System impact: Stabilizes training in deep networks with large parameter counts.
- **`clip_value`**: Clip gradients by value. System impact: Provides alternative gradient control mechanism.
- **`accumulation_steps`**: Accumulate gradients over multiple steps. Effective batch size = batch_size * accumulation_steps. System impact: Enables larger effective batch sizes on memory-constrained systems.

#### Runtime Optimizations

```json
"runtime": {
    "mixed_precision": true,
    "channels_last": true,
    "allow_tf32": true,
    "cudnn_benchmark": true
}
```

- **`mixed_precision`**: Use automatic mixed precision training. true speeds up training with minimal accuracy loss. System impact: Dramatically improves training throughput on compatible hardware.
- **`channels_last`**: Use channels-last memory format. May improve performance on some GPUs. System impact: Optimizes memory access patterns for specific GPU architectures.
- **`allow_tf32`**: Allow TensorFloat-32 operations. true speeds up math on Ampere+ GPUs. System impact: Accelerates matrix operations on modern NVIDIA GPUs.
- **`cudnn_benchmark`**: Enable cuDNN benchmarking. true finds optimal algorithms but increases startup time. System impact: Maximizes convolutional neural network performance through algorithm optimization.

### Dataset Settings

```json
"dataset": {
    "image_extensions": [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".tiff",
        ".webp"
    ]
}
```

- **`image_extensions`**: File extensions recognized as images. Add custom formats if needed. System impact: Controls which files are processed during dataset discovery and loading operations.

### Optimizer Defaults

The `optimizers` section provides default parameters for various optimizer types. These are used when configuring training optimizers. **Advanced users can modify these, but defaults are generally optimal.** System impact: Establishes baseline optimization parameters for different algorithms.

### Scheduler Defaults

Similar to optimizers, provides default parameters for learning rate schedulers. System impact: Defines standard scheduling behaviors for various decay patterns.

### Loss Function Defaults

Default parameters for loss functions. System impact: Provides sensible defaults for different loss computation requirements.

### Model Configurations

The `models` section contains information about supported model architectures, including:
- Available variants for each model type
- Parameter counts and recommended batch sizes
- Default optimizer and scheduler settings

**Do not modify this section unless adding new model support.** System impact: Defines architectural constraints and optimization recommendations for each model family.

## UI Available Options

This section lists the options available in the Hootsight user interface for configuration. These are the values you can select from dropdown menus and configuration panels.

### Dataset Options

**Image Extensions** (supported file formats for dataset images):
- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.gif`
- `.tiff`
- `.webp`

### Training Setup Options

**Model Types** (neural network architectures):
- `resnet` - Residual Networks (reliable, widely used)
- `resnext` - ResNeXt (improved ResNet with cardinality)
- `mobilenet` - MobileNet (efficient for mobile/edge devices)
- `shufflenet` - ShuffleNet (lightweight architecture)
- `squeezenet` - SqueezeNet (very compact models)
- `efficientnet` - EfficientNet (state-of-the-art efficiency)

**Model Variants** (specific implementations within each type):
- ResNet: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- ResNeXt: `resnext50_32x4d`, `resnext101_32x8d`, `resnext101_64x4d`
- MobileNet: `mobilenet_v2`, `mobilenet_v3_small`, `mobilenet_v3_large`
- ShuffleNet: `shufflenet_v2_x0_5`, `shufflenet_v2_x1_0`, `shufflenet_v2_x1_5`, `shufflenet_v2_x2_0`
- SqueezeNet: `squeezenet1_0`, `squeezenet1_1`
- EfficientNet: `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`, `efficientnet_b3`, `efficientnet_b4`, `efficientnet_b5`, `efficientnet_b6`, `efficientnet_b7`, `efficientnet_v2_s`, `efficientnet_v2_m`, `efficientnet_v2_l`

**Task Types** (training objectives):
- `classification` - Single-label classification
- `multi_label` - Multi-label classification
- `detection` - Object detection
- `segmentation` - Image segmentation

**Optimizer Types** (parameter update algorithms):
- `sgd` - Stochastic Gradient Descent
- `adam` - Adaptive Moment Estimation
- `adamw` - Adam with weight decay
- `adamax` - Adam variant with infinity norm
- `nadam` - Nesterov-accelerated Adam
- `radam` - Rectified Adam
- `rmsprop` - Root Mean Square Propagation
- `rprop` - Resilient Backpropagation
- `adagrad` - Adaptive Gradient
- `adadelta` - Adaptive Delta
- `sparse_adam` - Sparse Adam
- `lbfgs` - Limited-memory BFGS
- `asgd` - Averaged Stochastic Gradient Descent

**Scheduler Types** (learning rate adjustment strategies):
- `step_lr` - Step-wise learning rate decay
- `multi_step_lr` - Multi-step decay at specified epochs
- `exponential_lr` - Exponential decay
- `cosine_annealing_lr` - Cosine annealing
- `cosine_annealing_warm_restarts` - Cosine annealing with warm restarts
- `reduce_lr_on_plateau` - Reduce on plateau detection
- `cyclic_lr` - Cyclical learning rates
- `one_cycle_lr` - One cycle policy
- `polynomial_lr` - Polynomial decay
- `linear_lr` - Linear learning rate schedule
- `lambda_lr` - Custom lambda function
- `multiplicative_lr` - Multiplicative decay

**Loss Function Types** (training objectives):
- `cross_entropy` - Cross-entropy loss
- `nll_loss` - Negative log likelihood loss
- `bce_loss` - Binary cross-entropy loss
- `bce_with_logits` - BCE with logits (recommended for multi-label)
- `multi_margin` - Multi-margin loss
- `multi_label_margin` - Multi-label margin loss
- `multi_label_soft_margin` - Multi-label soft margin loss
- `mse_loss` - Mean squared error loss
- `l1_loss` - L1 loss
- `smooth_l1` - Smooth L1 loss
- `huber_loss` - Huber loss
- `kl_div` - KL divergence loss
- `margin_ranking` - Margin ranking loss
- `hinge_embedding` - Hinge embedding loss
- `triplet_margin` - Triplet margin loss
- `cosine_embedding` - Cosine embedding loss
- `ctc_loss` - Connectionist temporal classification loss
- `poisson_nll` - Poisson negative log likelihood loss
- `gaussian_nll` - Gaussian negative log likelihood loss

### Augmentation Options

**Available Transform Types** (data augmentation operations):

**Training Phase Transforms**:
- `random_resized_crop` - Random cropping with resizing
- `random_horizontal_flip` - Random horizontal flipping
- `random_vertical_flip` - Random vertical flipping
- `random_rotation` - Random rotation
- `color_jitter` - Random color jittering
- `random_grayscale` - Random grayscale conversion
- `random_erasing` - Random erasing
- `random_perspective` - Random perspective transformation

**Validation Phase Transforms**:
- `center_crop` - Center cropping
- `random_horizontal_flip` - Random horizontal flipping
- `random_rotation` - Random rotation

**Note**: The available transforms are defined in the UI's `AUGMENTATION_PRESETS` configuration in `actions.js`. Only these specific transforms are supported in the current UI implementation.

## Best Practices

1. **Start with defaults**: The default configuration is optimized for most use cases.
2. **Test changes incrementally**: Change one setting at a time and verify the impact.
3. **Monitor resources**: Watch memory usage and training speed when adjusting batch sizes and threads.
4. **Use project overrides**: Place project-specific settings in `projects/{project_name}/config.json`.
5. **Backup before major changes**: Save a copy of your working config before experimenting.

## Troubleshooting

- **Out of memory errors**: Reduce batch_size, target_memory_usage, or input_size.
- **Slow training**: Increase batch_size, num_workers, or enable mixed_precision.
- **Poor accuracy**: Adjust learning_rate, epochs, or augmentation settings.
- **Unstable training**: Enable gradient clipping or reduce learning_rate.

For more detailed information about specific components, refer to the relevant system documentation.

_Page created by Roxxy (AI) – 2025-10-01._