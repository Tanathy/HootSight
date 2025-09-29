# HootSight

A modular, configuration‑driven image recognition and training system built on PyTorch + FastAPI with a simple web UI (pywebview). It auto‑discovers projects, supports multiple model families and multi‑label datasets, includes memory‑aware batch‑size calculation to prevent OOM, and aims to be “run and go.”

![App Screenshot](https://raw.githubusercontent.com/Tanathy/HootSight/refs/heads/main/docs/imgs/app.png)


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
	config.json             # Optional project overrides; omit to inherit the global config
	dataset/                # Images + optional .txt tag files (organise either by train/val or nested themes)
		...
	model/                  # Created by the trainer when you run a job. Here comes the trained model with its settings also.
		best_model.pth
		training_history.json
	heatmaps/               # Grad-CAM renders land here when you request them
```

Dataset layout and annotations (sample scenario):
- Create your own project under `projects/{project_name}/dataset`. For example, you might build `projects/reference_library/dataset` with thematic folders such as `creatures/avian/`, `gear/harness/`, or `poses/on_four_limbs/`. The loader walks the tree recursively, so you can nest folders as deep as you like as long as the leaves contain images.
- To unlock multi-label behaviour, drop a `.txt` file next to each image with the same basename. Put your comma-separated tags on the first non-empty line, trimming whitespace. Example: `glider, winged, nocturnal`.
- When at least 90% of images in a dataset have these `.txt` companions, HootSight automatically switches to multi-label mode and builds the class list from the union of all discovered tags. Images missing a `.txt` fall back to the name of their leaf folder so nothing is discarded.
- Explicit `train/` and `val/` folders remain optional. If they’re absent, the coordinator compiles the full image list twice (for train and validation views) and performs a randomized split using `training.val_ratio` (or the legacy `training.split.val_ratio`)—20% by default.
- Keep a minimum of five samples per label. Anything below the configured threshold (`dataset.discovery.balance_analysis.min_images_per_class`, default 5) is ignored when the training split is generated.
- Supported image formats stay the same: `.jpg, .jpeg, .png, .bmp, .gif, .tiff, .webp`. Add more via `config/config.json → dataset.image_extensions` if you need other formats.
- The UI’s dataset analysis may display helper entries prefixed with `folder:`. Those are only for balance insights; they’re stripped before training so the class list contains just your real tags.

Good to know:
- The global default is multi-label (`"training.task": "multi_label"`). If your dataset is classic single-label classification (one folder = one class, no tag mixing), flip that value to `classification` before launching a run so the coordinator emits integer class indices instead of multi-hot vectors.
- Images without a `.txt` companion inherit their leaf folder name as a fallback tag. That keeps stragglers in play, but make sure the folder naming actually reflects the concept you want learned.
- Folder names in analytics are normalized for readability: spaces become `_`, nested folders are joined with `.`. Example: `creatures skyborne/long glider` is rendered as `creatures_skyborne.long_glider`. The raw tag strings inside `.txt` files are left exactly as you typed them, so stay consistent with casing and punctuation.
- Random train/validation splits are re-generated on every run when you rely on auto splitting.

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

