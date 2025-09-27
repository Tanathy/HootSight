# HootSight

A modular, configuration‑driven image recognition and training system built on PyTorch + FastAPI with a simple web UI (pywebview). It auto‑discovers projects, supports multiple model families and multi‑label datasets, includes memory‑aware batch‑size calculation to prevent OOM, and aims to be “run and go.”


## Requirements

- Windows 10/11 64‑bit
- Python 3.12 (recommended; 3.10–3.12 is fine)
- NVIDIA GPU recommended
- Internet connection on first run to install dependencies
- ESSENTIAL for GPU acceleration: install the NVIDIA CUDA Toolkit matching your GPU/driver from the official CUDA Downloads page:
	https://developer.nvidia.com/cuda-downloads

Notes:
- No global installs. On first run, a local virtual environment (venv) is created and all packages are installed there.
- PyTorch and xFormers CUDA wheels are auto‑selected either by autodetection or via `config/packages.jsonc` → `pytorch_cuda_version` (default: `cu129`). If you’re CPU‑only, remove that key (or leave it empty) and install a CPU build of torch.
- If your installed CUDA runtime differs from the configured one, adjust `pytorch_cuda_version` accordingly; otherwise CUDA‑specific wheels (xFormers) may fail to install.
- Platform/testing status: we have not yet been able to test on Linux environments or with AMD GPUs. We aim for multi‑compatibility and a friendly UX, but this is still very early; issues may occur.


## Quick start

1) Download/clone the repo.
2) In PowerShell, from the repo root, run:

```powershell
py -3.12 run.py
```

What happens:
- A `.\venv` is created and pip is upgraded.
- PyTorch (CUDA build via autodetection or config) and all required packages from `config/packages.jsonc` are installed.
- The FastAPI server and the pywebview app start.
- If the desktop window doesn’t show, open the UI in your browser at http://127.0.0.1:8000


## Project layout and dataset rules

Typical project structure:

```
projects/{project_name}/
	├─ config.json            # Project‑specific overrides (merged with the global config)
	├─ dataset/               # Dataset root
	│   ├─ train/ ...         # (optional) training split
	│   └─ val/ ...           # (optional) validation split
	├─ model/                 # Outputs (checkpoints, history)
	│   ├─ best_model.pth
	│   └─ training_history.json
	└─ heatmaps/              # (optional) Grad‑CAM images
```

Dataset modes and tips:
- Two main modes:
	1) Folder‑based classification
		 - Recommended: `dataset/train/{label}/images...` and `dataset/val/{label}/images...`
		 - If there’s no `train/val`, a flat `dataset/{label}/images...` layout is auto‑split into train/val using `training.split.val_ratio` from the global config.
	2) Multi‑label
		 - Place a `.txt` file next to each image with the same basename. First line: comma‑separated tags, e.g. `dog, animal, cute`.
		 - If ≥90% of images have a `.txt` pair, the dataset is treated as multi‑label. For images without `.txt`, the folder name is used as a fallback label.

Good to know:
- Minimum images per class: around 5. Fewer than that may be dropped by the loader.
- Supported image formats: `.jpg, .jpeg, .png, .bmp, .gif, .tiff, .webp` (configurable in `config/config.json` → `dataset.image_extensions`).
- Label normalization for reports: folder names get normalized (space → `_`, slash `/` → `.`). Example: `awesome fursuits/tania` → `awesome_fursuits.tania`. For multi‑label `.txt` tags, no normalization is applied—be consistent in how you write tags.
- Task type: global default is `"training.task": "multi_label"`. For single‑label classification, set it to `classification`.


## Supported model families and variants

The following model families are supported; see `config/config.json` → `models` for the full variant list and recommended input/batch sizes:

- ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
- ResNeXt: resnext50_32x4d, resnext101_32x8d, resnext101_64x4d
- MobileNet: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
- ShuffleNet: shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
- SqueezeNet: squeezenet1_0, squeezenet1_1
- EfficientNet: efficientnet_b0…b7, efficientnet_v2_s/m/l

Note: At startup, HootSight computes a memory‑aware batch size to avoid out‑of‑memory crashes. It can differ from the “recommended” size depending on your hardware, and that’s intentional.


## Configuration center

- Global: `config/config.json` — API port, UI size, training parameters (optimizer/scheduler/loss), augmentations, and more. The web UI reads/writes this file.
- Environment/packages: `config/packages.jsonc` — if you need a specific CUDA build for PyTorch/xFormers (`pytorch_cuda_version`), set it here. For CPU‑only, remove or leave it empty and install a CPU torch.
- Per‑project: `projects/{project}/config.json` — overrides global settings just for that project.


## Troubleshooting (quick tips)

- Port 8000 in use: change `config/config.json` → `api.port` and restart.
- xFormers install fails: not fatal; the app runs without it, but some GPU speed‑ups will be missing.
- No UI window: open http://127.0.0.1:8000 in your browser.
- CUDA mismatch: if you installed a different CUDA than what `config/packages.jsonc` uses (e.g. `cu129`), update `pytorch_cuda_version` to match, or remove it to let autodetection pick a compatible wheel.


## License and contributions

Actively evolving project—built by humans and AI (Roxxy, my in‑house dev AI). To keep the project alive and healthy, please report any issues you run into; that feedback is a big help. PRs and ideas are welcome.

