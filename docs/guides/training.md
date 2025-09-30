# Training Guide

The Training page is where you shape the configuration that `system/coordinator.py` uses to launch a run. Every control on this page edits the active project's `config.json`, so what you see in the UI is what the trainer consumes.

## Before you begin

1. **Load a project** from the Projects page so the header shows an active project badge. Training settings are saved per project.
2. **Check dataset layout** under `projects/<name>/dataset/` (`train/` and `val/` folders are optional but recommended). The coordinator will discover labels from this tree.
3. **Refresh configuration** if you changed anything on disk. Opening the Training page triggers a schema fetch; wait for the cards to render before editing fields.
4. **Mind unsaved changes** – the footer status will flip to "Training config saved" only after you press the **Save Training Config** action in the page header.

## Page anatomy

The Training page is rendered as a stack of cards. Each card corresponds to a group in `config_schema.json`. Controls are schema-driven and include inline validation, so illegal values never reach the configuration.

### Model settings

Pick the architecture family (`model_type`), specific backbone (`model_name`), and whether to load ImageNet weights (`pretrained`). Families available out of the box are ResNet, ResNeXt, MobileNet, ShuffleNet, SqueezeNet, and EfficientNet. Larger variants consume more memory; the memory planner accounts for the selection when it estimates batch size.

### Task configuration

Choose the task (`classification` or `multi_label`) and the image size. Multi-label mode is fully supported across the data pipeline. Detection and segmentation appear in the schema for future work but require custom dataset loaders; the UI labels them accordingly so you know they are experimental.

### Training parameters

Configure `batch_size`, `epochs`, `learning_rate`, `weight_decay`, and `val_ratio`.

- Setting `batch_size` or `epochs` to `auto` delegates to the coordinator. `batch_size` will run through `system.memory.get_optimal_batch_size` before every training session using the model you selected and the memory targets from `config.memory`.
- Numeric fields enforce ranges from the schema. The UI presents an "auto" vs "value" selector for mixed-mode fields.

### Optimizer, scheduler, and loss cards

These three cards are dynamic:

- Select the type from the dropdown at the top of the card.
- The parameter section underneath rebuilds itself using the schema defaults in `optimizers.defaults`, `schedulers.defaults`, or `losses.defaults`.
- Key parameters stay synchronized with the simpler controls. For example, changing the optimizer's `lr` updates `training.optimizer_lr` and `training.learning_rate` so the plan is self-consistent.

All parameters save into `training.optimizer_params.<type>`, `training.scheduler_params.<type>`, and `training.loss_params.<type>` inside your project config.

### Data loading

The `dataloader` object collects `num_workers`, `pin_memory`, `persistent_workers`, and `prefetch_factor`. These map straight to the PyTorch `DataLoader` arguments. Increase `num_workers` when heavy augmentations start starving the GPU; keep it at zero on Windows if you hit spawn limitations.

### Normalization

`training.normalize.mean` and `training.normalize.std` control the normalization layer appended by the Augmentation page. Stick with the ImageNet statistics when you use pre-trained weights; customize them only if you trained a backbone from scratch on different data.

### Checkpointing and weight initialization

- **Checkpointing** covers `save_best_only`, `save_frequency`, `max_checkpoints`, and file naming. These values are read directly by `TrainingManager` when it writes checkpoints.
- **Weight initialization** exposes the method (`kaiming_normal`, `xavier_uniform`, and so on) plus method-specific parameters. Leave the defaults unless you know the architecture requires a custom initializer.

## Saving and validation

The header action **Save Training Config** runs schema validation before writing anything. If a field violates the schema, the footer displays "Fix validation errors before saving" and the validation summary highlights each failing path. On success the button re-enables automatically and your project `config.json` is updated in place.

Unsaved changes keep the app in a "dirty" state; switching pages or projects will warn you by leaving the save button active. Saving clears the dirty flag.

## Relationship to augmentation

The augmentation toggles live on their own page, but the Training page owns the canonical configuration. When you change normalization statistics or the task type here, the augmentation preview panel picks up the new values immediately. See the [Augmentation guide](augmentation.md) for the full pipeline details.

## Launching training

Once the configuration is saved:

1. Switch to the **Projects** page and use the active project's **Start Training** button. That action posts to `/training/start` for the project you selected; the Status page does not launch jobs.
2. The coordinator will reload the project config, calculate the optimal batch size if `batch_size` was left on `auto`, instantiate the model, build the augmentation pipelines, and spin up the optimizer/scheduler/loss using the parameters you defined.
3. The **Status** page is your telemetry dashboard. It surfaces running- and historical-session metrics, including the per-epoch graphs and live counters for phase, epoch, and step.

Need to cancel a run? Hit **Stop Training** on the same project card. The backend raises a stop flag, lets the current batch finish so gradients stay consistent, then tears down the loaders and workers before returning control. The status panel flips back to idle as soon as the shutdown completes.

## Troubleshooting

- **Schema not loaded** – the Training page shows an error card when the schema fetch fails. Reload the app; without the schema the UI refuses to edit the config.
- **Saving disabled** – ensure a project is active. The button is only enabled when `state.currentProject` is populated.
- **Batch size too aggressive** – override `batch_size` with a concrete value if you know your hardware limits, or lower the `memory.target_memory_usage` in the global config.
- **Parameters reverting** – the UI respects the schema defaults. If a value keeps snapping back, verify the type: integers vs strings (`"auto"`) are enforced strictly.

_Page created by Roxxy (AI) – 2025-10-01._
