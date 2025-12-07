from __future__ import annotations

import os
import io
import random
import base64
import time
import copy
from typing import Optional, Tuple, Any, cast, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
import cv2
import numpy as np

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS
from system.dataset_discovery import get_project_info
from system.models.resnet import ResNetModel
from system.project_labels import load_project_labels


# Model cache: stores loaded models to avoid redundant loading
# Key: (project_name, checkpoint_path, checkpoint_mtime)
# Value: (wrapper, checkpoint_path, checkpoint_data)
_model_cache: Dict[Tuple[str, str, float], Tuple[Any, str, dict]] = {}


def clear_model_cache() -> None:
    """Clear the model cache to force reload on next evaluation."""
    global _model_cache
    _model_cache.clear()
    info("Heatmap model cache cleared")


def _get_project_paths(project_name: str) -> Tuple[str, str, str, str]:
    """Get project paths: (dataset_path, model_dir, validation_path, data_source_path)"""
    try:
        paths_cfg = SETTINGS['paths']
    except Exception:
        raise ValueError("Missing required 'paths' section in config/config.json")
    base_rel = paths_cfg.get('projects_dir')
    if base_rel is None:
        raise ValueError("paths.projects_dir must be defined in config/config.json")
    base_path = os.path.dirname(os.path.dirname(__file__))
    base_projects = os.path.join(base_path, base_rel)
    project_root = os.path.join(base_projects, project_name)
    dataset_path = os.path.join(project_root, 'dataset')
    model_dir = os.path.join(project_root, 'model')
    validation_path = os.path.join(project_root, 'validation')
    data_source_path = os.path.join(project_root, 'data_source')
    return dataset_path, model_dir, validation_path, data_source_path


def _collect_images(folder: str) -> list[str]:
    try:
        dataset_cfg = SETTINGS['dataset']
    except Exception:
        raise ValueError("Missing required 'dataset' section in config/config.json")
    exts = dataset_cfg.get('image_extensions')
    if not isinstance(exts, list) or not exts:
        raise ValueError("dataset.image_extensions must be a non-empty list in config/config.json")
    results: list[str] = []
    if not os.path.isdir(folder):
        return results
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if any(f.lower().endswith(ext) for ext in exts):
                results.append(os.path.join(root, f))
    return results


def pick_sample_image(project_name: str, preferred_split: str = 'validation') -> Optional[str]:
    """Pick a random sample image from project folders.
    
    Priority: validation > data_source > dataset
    Returns the full absolute path to the image file.
    """
    _dataset_path, _model_dir, validation_path, data_source_path = _get_project_paths(project_name)
    
    # Try validation folder first
    candidates = _collect_images(validation_path)
    if candidates:
        return random.choice(candidates)
    
    # Fallback to data_source
    candidates = _collect_images(data_source_path)
    if candidates:
        return random.choice(candidates)
    
    # Final fallback to dataset
    candidates = _collect_images(_dataset_path)
    if candidates:
        return random.choice(candidates)
    
    warning(f"No images found for project {project_name}")
    return None


def _get_preprocess_params() -> dict:
    try:
        tr_cfg = SETTINGS['training']
    except Exception:
        raise ValueError("Missing required 'training' section in config/config.json")
    input_block = tr_cfg.get('input', {}) if isinstance(tr_cfg.get('input'), dict) else {}
    size = input_block.get('image_size', tr_cfg.get('input_size'))
    if size is None:
        raise ValueError("training.input_size or training.input.image_size must be provided in config/config.json")
    size = int(size)
    resize_target = int(max(size + 32, size))
    norm = input_block.get('normalize', tr_cfg.get('normalize'))
    if not isinstance(norm, dict):
        raise ValueError("training.normalize must be provided in config/config.json")
    mean = norm.get('mean', [0.485, 0.456, 0.406])
    std = norm.get('std', [0.229, 0.224, 0.225])
    return {
        'input_size': size,
        'resize_target': resize_target,
        'mean': mean,
        'std': std,
    }


def _prepare_model_input(image: Image.Image) -> Tuple[torch.Tensor, dict]:
    params = _get_preprocess_params()
    resize = transforms.Resize(params['resize_target'])
    crop = transforms.CenterCrop(params['input_size'])
    resized = resize(image)
    cropped = crop(resized)
    tensor = transforms.ToTensor()(cropped)
    tensor = transforms.Normalize(mean=params['mean'], std=params['std'])(tensor)
    tensor = tensor.unsqueeze(0)

    crop_left = max((resized.width - params['input_size']) // 2, 0)
    crop_top = max((resized.height - params['input_size']) // 2, 0)

    meta = {
        'original_size': image.size,
        'resized_size': (resized.width, resized.height),
        'crop_offset': (crop_left, crop_top),
        'input_size': params['input_size'],
    }
    return tensor, meta


def _create_model_from_checkpoint(checkpoint_path: str, map_location: Optional[str] = None):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    from system.device import get_device
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=map_location or device)
    model_name = checkpoint.get('model_name', 'resnet50')
    num_classes = int(checkpoint.get('num_classes', 1000))

    model_type = _infer_model_type(model_name)

    if model_type == 'resnet':
        from system.models.resnet import create_resnet_model
        wrapper = create_resnet_model(model_name=model_name, num_classes=num_classes, pretrained=False, task='classification')
    elif model_type == 'resnext':
        from system.models.resnext import create_resnext_model
        wrapper = create_resnext_model(model_name=model_name, num_classes=num_classes, pretrained=False, task='classification')
    elif model_type == 'mobilenet':
        from system.models.mobilenet import create_mobilenet_model
        wrapper = create_mobilenet_model(model_name=model_name, num_classes=num_classes, pretrained=False, task='classification')
    elif model_type == 'shufflenet':
        from system.models.shufflenet import create_shufflenet_model
        wrapper = create_shufflenet_model(model_name=model_name, num_classes=num_classes, pretrained=False, task='classification')
    elif model_type == 'squeezenet':
        from system.models.squeezenet import create_squeezenet_model
        wrapper = create_squeezenet_model(model_name=model_name, num_classes=num_classes, pretrained=False, task='classification')
    elif model_type == 'efficientnet':
        from system.models.efficientnet import create_efficientnet_model
        wrapper = create_efficientnet_model(model_name=model_name, num_classes=num_classes, pretrained=False, task='classification')
    else:
        warning(f"Unknown model type for {model_name}, falling back to ResNet")
        from system.models.resnet import create_resnet_model
        wrapper = create_resnet_model(model_name='resnet50', num_classes=num_classes, pretrained=False, task='classification')

    wrapper.load_checkpoint(checkpoint_path)
    wrapper.model.eval()
    return wrapper, checkpoint_path, checkpoint


def _infer_model_type(model_name: str) -> str:
    model_name_lower = model_name.lower()
    if model_name_lower.startswith('resnext'):
        return 'resnext'
    elif model_name_lower.startswith('resnet'):
        return 'resnet'
    elif model_name_lower.startswith('mobilenet'):
        return 'mobilenet'
    elif model_name_lower.startswith('shufflenet'):
        return 'shufflenet'
    elif model_name_lower.startswith('squeezenet'):
        return 'squeezenet'
    elif model_name_lower.startswith('efficientnet'):
        return 'efficientnet'
    else:
        return 'resnet'  # default


def _load_model_from_checkpoint(project_name: str, map_location: Optional[str] = None, use_cache: bool = True):
    """Load checkpoint from project model directory using config filename.
    
    Uses caching to avoid redundant model loads when the checkpoint file hasn't changed.
    Set use_cache=False to force a fresh load (for live model updates).
    """
    global _model_cache
    
    _dataset_path, model_dir, _validation_path, _data_source_path = _get_project_paths(project_name)
    
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    try:
        checkpoint_cfg = SETTINGS['training']['checkpoint']
    except Exception:
        raise ValueError("Missing required 'training.checkpoint' section in config/config.json")
    
    best_name = checkpoint_cfg.get('best_model_filename', 'best_model.pth')
    ckpt_path = os.path.join(model_dir, best_name)
    
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # Get file modification time for cache validation
    ckpt_mtime = os.path.getmtime(ckpt_path)
    cache_key = (project_name, ckpt_path, ckpt_mtime)
    
    # Check cache if enabled
    if use_cache and cache_key in _model_cache:
        info(f"Using cached model for {project_name}")
        return _model_cache[cache_key]
    
    # Load fresh model
    result = _create_model_from_checkpoint(ckpt_path, map_location)
    
    # Cache the loaded model (only if caching is enabled)
    if use_cache:
        # Clear old cache entries for this project to prevent memory bloat
        keys_to_remove = [k for k in _model_cache if k[0] == project_name]
        for k in keys_to_remove:
            del _model_cache[k]
        _model_cache[cache_key] = result
        info(f"Cached model for {project_name}")
    
    return result


def _find_target_layer(module: nn.Module) -> nn.Module:
    try:
        layer4 = getattr(module, 'layer4', None)
        if layer4 is not None and hasattr(layer4, '__getitem__'):
            last_block = layer4[-1]
            if hasattr(last_block, 'conv3'):
                return last_block.conv3
            if hasattr(last_block, 'conv2'):
                return last_block.conv2
    except Exception:
        pass

    target = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            target = m
    if target is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM")
    return target


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor) -> Tuple[int, torch.Tensor]:
    device = next(model.parameters()).device
    x = x.to(device)
    logits = model(x)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    pred = int(torch.argmax(logits, dim=1).item())
    return pred, logits


def compute_gradcam(model: nn.Module, x: torch.Tensor, target_class: Optional[int] = None,
                    target_layer: Optional[nn.Module] = None) -> torch.Tensor:
    assert x.ndim == 4 and x.size(0) == 1, "Grad-CAM expects a single image (N=1)"

    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)

    layer = target_layer or _find_target_layer(model)

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def fwd_hook(_m: nn.Module, _inp: Any, out: torch.Tensor):
        activations.append(out.detach())

    def bwd_hook(_m: nn.Module, grad_in: Any, grad_out: Any):
        gradients.append(grad_out[0].detach())

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    logits = model(x)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    if target_class is None:
        target_class = int(torch.argmax(logits, dim=1).item())

    one_hot = torch.zeros_like(logits)
    one_hot[0, target_class] = 1.0

    model.zero_grad(set_to_none=True)
    (logits * one_hot).sum().backward()

    act = activations[-1]
    grad = gradients[-1]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=False)
    cam = F.relu(cam)

    cam_min = cam.min(dim=1, keepdim=False)[0].min(dim=1, keepdim=False)[0]
    cam_max = cam.max(dim=1, keepdim=False)[0].max(dim=1, keepdim=False)[0]
    cam = (cam - cam_min[:, None, None]) / (cam_max[:, None, None] - cam_min[:, None, None] + 1e-8)

    h1.remove(); h2.remove()
    return cam[0].detach().cpu()


def overlay_heatmap_on_image(
    heatmap: torch.Tensor,
    image_rgb: Image.Image,
    preprocess_meta: dict,
    alpha: float = 0.5
) -> Image.Image:
    orig_w, orig_h = image_rgb.size
    resized_w, resized_h = preprocess_meta['resized_size']
    crop_left, crop_top = preprocess_meta['crop_offset']
    input_size = preprocess_meta['input_size']

    heat_np = heatmap.numpy()
    heat_square = cv2.resize(heat_np, (input_size, input_size))

    full_resized = np.zeros((resized_h, resized_w), dtype=np.float32)
    end_y = min(resized_h, crop_top + input_size)
    end_x = min(resized_w, crop_left + input_size)
    patch = heat_square[: end_y - crop_top, : end_x - crop_left]
    full_resized[crop_top:end_y, crop_left:end_x] = patch

    heat_original = cv2.resize(full_resized, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    heat_uint8 = np.clip(heat_original * 255, 0, 255).astype(np.uint8)

    heatmap_colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    original_np = np.array(image_rgb)
    output = cv2.addWeighted(original_np, 1 - alpha, heatmap_colored, alpha, 0)

    return Image.fromarray(output)


def generate_heatmap(project_name: str, image_path: Optional[str] = None,
                     target_class: Optional[int] = None, alpha: float = 0.5) -> Tuple[bytes, dict]:
    proj = get_project_info(project_name, compute_stats=True)
    if not proj or not proj.has_dataset:
        raise FileNotFoundError(f"Project {project_name} or its dataset not found")

    if not image_path:
        image_path = pick_sample_image(project_name)
    if not image_path or not os.path.isfile(image_path):
        raise FileNotFoundError("No valid image found for heatmap generation")

    pil_img = Image.open(image_path).convert('RGB')
    x, preprocess_meta = _prepare_model_input(pil_img)

    wrapper, ckpt_path, _ckpt_data = _load_model_from_checkpoint(project_name)

    with torch.no_grad():
        pred_class, _ = predict(wrapper.model, x.clone())

    tclass = pred_class if target_class is None else int(target_class)

    x_grad = x.clone().requires_grad_(True)
    heat = compute_gradcam(wrapper.model, x_grad, target_class=tclass, target_layer=None)

    overlay = overlay_heatmap_on_image(heat, pil_img, preprocess_meta, alpha=alpha)
    buf = io.BytesIO()
    overlay.save(buf, format='PNG')
    data = buf.getvalue()

    meta = {
        'project': project_name,
        'image_path': image_path,
        'predicted_class': int(pred_class),
        'used_class': int(tclass),
        'checkpoint': ckpt_path
    }
    info(f"Generated heatmap for {project_name} using image {os.path.basename(image_path)} (class {tclass})")
    return data, meta


def evaluate_with_heatmap(project_name: str, image_path: Optional[str] = None,
                          checkpoint_path: Optional[str] = None,
                          use_live_model: bool = False) -> dict:
    """Evaluate an image with Grad-CAM heatmap visualization.
    
    Args:
        project_name: Name of the project
        image_path: Full path to image, or None to pick random from validation/data_source
        checkpoint_path: Optional specific checkpoint path (overrides project default)
        use_live_model: If True, skip cache and always load fresh model from disk
    
    Returns:
        Dict with heatmap (base64), predictions, and image/checkpoint info
    """
    proj = get_project_info(project_name, compute_stats=True)
    if not proj or not proj.has_dataset:
        raise FileNotFoundError(f"Project {project_name} or its dataset not found")

    if not image_path:
        image_path = pick_sample_image(project_name)
    if not image_path or not os.path.isfile(image_path):
        raise FileNotFoundError("No valid image found for evaluation")

    pil_img = Image.open(image_path).convert('RGB')
    x, preprocess_meta = _prepare_model_input(pil_img)

    if checkpoint_path:
        wrapper, ckpt_path, ckpt_data = _create_model_from_checkpoint(checkpoint_path)
    else:
        # use_cache=False when live model is requested (forces fresh load)
        wrapper, ckpt_path, ckpt_data = _load_model_from_checkpoint(project_name, use_cache=not use_live_model)

    with torch.no_grad():
        pred_class, logits = predict(wrapper.model, x.clone())

    try:
        task = SETTINGS['training']['task']
    except Exception:
        raise ValueError("Missing required 'training.task' in config/config.json")
    if not task:
        raise ValueError("training.task must be defined in config/config.json")

    # Prefer labels from checkpoint (embedded during training), fallback to persisted manifest then discovery
    labels = ckpt_data.get('labels') or []
    if not labels:
        labels = load_project_labels(project_name)
    if not labels:
        labels = proj.labels or []

    if task == 'multi_label':
        probabilities = torch.sigmoid(logits)
        # Show top 5 predictions regardless of threshold (more useful for debugging)
        top_k = min(5, len(labels)) if labels else 0
        if top_k > 0:
            top_values, top_indices = torch.topk(probabilities[0], top_k)
            class_names = [labels[idx] for idx in top_indices.tolist() if idx < len(labels)]
            confidence_values = top_values.tolist()
        else:
            class_names = []
            confidence_values = []
    else:
        probabilities = F.softmax(logits, dim=1)[0]
        # Show top 5 predictions regardless of threshold
        top_k = min(5, len(labels)) if labels else 0
        if top_k > 0:
            top_values, top_indices = torch.topk(probabilities, top_k)
            class_names = [labels[idx] for idx in top_indices.tolist() if idx < len(labels)]
            confidence_values = top_values.tolist()
        else:
            class_names = []
            confidence_values = []

    x_grad = x.clone().requires_grad_(True)
    heat = compute_gradcam(wrapper.model, x_grad, target_class=pred_class, target_layer=None)

    overlay = overlay_heatmap_on_image(heat, pil_img, preprocess_meta, alpha=0.5)
    buf = io.BytesIO()
    overlay.save(buf, format='PNG')
    png_data = buf.getvalue()

    heatmaps_dir = os.path.join(_get_project_paths(project_name)[1], '..', 'heatmaps')
    os.makedirs(heatmaps_dir, exist_ok=True)
    timestamp = int(time.time() * 1000)
    heatmap_filename = f"heatmap_{timestamp}.png"
    heatmap_path = os.path.join(heatmaps_dir, heatmap_filename)
    with open(heatmap_path, 'wb') as f:
        f.write(png_data)
    info(f"Heatmap saved to {heatmap_path}")

    heatmap_b64 = base64.b64encode(png_data).decode('utf-8')

    predictions = {
        "predicted_classes": class_names,
        "confidence_values": confidence_values
    }

    result = {
        "heatmap": heatmap_b64,
        "predictions": predictions,
        "image_path": os.path.basename(image_path),
        "full_image_path": image_path,  # Full path for Refresh functionality
        "checkpoint": os.path.basename(ckpt_path)
    }

    info(f"Generated heatmap for {project_name} using image {os.path.basename(image_path)} (class {pred_class})")
    return result
