# Augmentation Guide

The Augmentation page lets you configure the image preprocessing pipeline that feeds the trainer. It mirrors the real transform objects created in `system/augmentation.py`, so whatever you toggle in the UI is exactly what the coordinator executes before every batch.

## Phase layout

Augmentations are separated into two phases:

- **Training phase (`train`)** – usually more aggressive to improve generalization.
- **Validation phase (`val`)** – typically conservative to keep evaluation stable.

Each phase has its own toggle list. Turning an option on immediately updates the in-memory configuration for the active project and queues the preview section for refresh. Turning it off removes the transform from that phase while preserving any parameter edits you made.

### Preset catalogue

| Phase | Transform | Default state | Tunable fields |
|-------|-----------|---------------|----------------|
| train | `random_resized_crop` | Enabled | `size`, `scale[0..1]`, `ratio[0..1]`
| train | `random_horizontal_flip` | Enabled | `p`
| train | `random_vertical_flip` | Disabled | `p`
| train | `random_rotation` | Disabled | `degrees[0..1]`
| train | `color_jitter` | Disabled | `brightness`, `contrast`, `saturation`, `hue`
| train | `random_grayscale` | Disabled | `p`
| train | `random_erasing` | Disabled | `p`, `scale[0..1]`, `ratio[0..1]`, `value`, `inplace`
| train | `random_perspective` | Disabled | `distortion_scale`, `p`
| val | `center_crop` | Enabled | `size`
| val | `random_horizontal_flip` | Disabled | `p`
| val | `random_rotation` | Disabled | `degrees[0..1]`

Parameter widgets enforce the same bounds that the UI schema defines (for example, probabilities are clamped to \[0, 1\] and numeric fields respect their `min`, `max`, and `step` values). When you re-enable a preset the UI restores your last saved values.

## How configuration is stored

Active selections are serialized into your project configuration under `training.augmentation.train` and `training.augmentation.val`. The UI always appends the tensor conversion and normalization steps based on `training.normalize`, so every phase ends with:

```jsonc
{
  "type": "to_tensor"
},
{
  "type": "normalize",
  "params": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  }
}
```

`random_resized_crop` is deferred to the tail of the transform list (just before `to_tensor`) so geometric warps apply in a predictable order. Any additional transforms that you inject manually into `config.json` are kept verbatim, surfaced in the "custom transforms" notice, and appended in the same order you defined.

### Manual edits

You can add transforms that are not exposed in the toggle list by editing your project configuration directly:

```json
{
  "type": "random_autocontrast",
  "params": { "p": 0.3 }
}
```

The UI will acknowledge them as read-only custom entries. Do not remove `to_tensor` or `normalize`; they are compulsory for the trainer to receive normalized tensors.

## Preview workflow

The preview button posts `{ phase, transforms }` to `/projects/{project}/augmentation/preview`. The backend:

1. Chooses a random image from `projects/<name>/dataset/<phase>` when it exists, otherwise it falls back to the dataset root.
2. Builds a torchvision pipeline with `DataAugmentationFactory.create_from_config` for every transform you supplied, reusing normalization settings so the preview reflects the true training pipeline.
3. Returns a JSON payload with base64-encoded PNGs for the original and augmented images plus the relative image path.

The UI caches the latest response per phase until you edit the pipeline again. Typical failure cases:

- **No project loaded** – the preview panel switches to an error state immediately.
- **Empty pipeline** – you will be prompted to enable at least one transform.
- **Dataset has no images** – the API returns a localized `preview_no_images` message.
- **Parameter error** – invalid settings bubble up from `DataAugmentationFactory` with details in the message and the system log.

## Training integration

When you start training, `system/coordinator.py` reads the serialized lists and feeds them through `DataAugmentationFactory.create_composition`. If you leave a phase empty, the coordinator falls back to a default torchvision recipe (`RandomResizedCrop` + flip for training, `Resize` + `CenterCrop` for validation). Multi-label projects reuse the exact same augmentation pipeline, so you can rely on the preview to represent what both classification and multi-label loaders will see.

Because batch-size calculation happens before the data loaders spin up, you can tune augmentations without invalidating the memory planner. Augmentations execute on the CPU using the worker count defined in `training.dataloader.num_workers`, so consider increasing that value if heavy transforms (such as random perspective or erasing) become a bottleneck.

_Page created by Roxxy (AI) – 2025-10-01._

## Troubleshooting checklist

- **Preview always idle** – ensure a project is loaded and the dataset folders contain at least one supported image extension.
- **Transforms not saving** – the "Save Training Config" header action must succeed; watch the footer for the "Training config saved" message.
- **Runtime mismatch** – double-check for manual edits that conflict with the presets. The UI regenerates the transform order on every interaction, so keep custom transforms minimal and well tested.
- **Unexpected normalization** – adjust `training.normalize.mean/std` in the Training guide if you need non-ImageNet statistics. The augmentation page simply mirrors those values.
