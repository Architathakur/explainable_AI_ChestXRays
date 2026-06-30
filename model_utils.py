import io
import json
import sys
import time
import types
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


IMAGE_SIZE = 224


def _require_torch():
    import torch
    from torch import nn

    return torch, nn


def _normalize_np(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image -= float(np.min(image))
    max_value = float(np.max(image))
    if max_value > 0:
        image /= max_value
    return image


def _resize_image(image: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def _overlay_heatmap(display_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import cv2

    if display_np.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"display_np must have shape (224, 224), got {display_np.shape}")
    if heatmap.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"heatmap must have shape (224, 224), got {heatmap.shape}")

    heatmap = np.nan_to_num(heatmap.astype(np.float32))
    heatmap -= float(np.min(heatmap))
    denom = float(np.max(heatmap))
    if denom > 0:
        heatmap /= denom

    base = np.stack([display_np, display_np, display_np], axis=-1).astype(np.float32)
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32)
    overlay = (1.0 - alpha) * base + alpha * colored
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _assert_tensor_shape(tensor, name: str = "tensor") -> None:
    if tuple(tensor.shape) != (1, 1, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"{name} must have shape [1, 1, 224, 224], got {tuple(tensor.shape)}")


def _assert_logits_shape(logits) -> None:
    if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != 2:
        raise ValueError(f"Model output must have shape [1, 2], got {tuple(logits.shape)}")


def _ensure_lzma_stub() -> None:
    if "lzma" in sys.modules:
        return
    try:
        import lzma  # noqa: F401
        return
    except Exception:
        pass
    lzma_stub = types.ModuleType("lzma")

    class _UnavailableLZMAFile:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("lzma is unavailable in this Python build.")

    lzma_stub.LZMAFile = _UnavailableLZMAFile
    lzma_stub.open = _UnavailableLZMAFile
    sys.modules["lzma"] = lzma_stub


def _load_threshold(threshold_path: str) -> float:
    with open(threshold_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "best_threshold" in data:
        return float(data["best_threshold"])
    if "best_f1" in data and "threshold" in data["best_f1"]:
        return float(data["best_f1"]["threshold"])
    return 0.731


def load_model(checkpoint_path: str, threshold_path: str):
    """
    Load TorchXRayVision DenseNet121 encoder (densenet121-res224-all) + MLP head.
    MLP: Linear(1024->512), ReLU, Dropout(0.4), Linear(512->2)
    Load checkpoint weights into the head.
    Load threshold from threshold_tuning.json (key: 'best_threshold').
    Return (full_model, threshold) both on CPU.
    """
    torch, nn = _require_torch()
    _ensure_lzma_stub()
    import torchxrayvision as xrv

    checkpoint_file = Path(checkpoint_path)
    threshold_file = Path(threshold_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    if not threshold_file.exists():
        raise FileNotFoundError(f"Threshold file not found: {threshold_file}")

    class PneumoXAIModel(nn.Module):
        def __init__(self, encoder, feature_dim: int = 1024):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(512, 2),
            )

        def forward(self, images):
            features = self.encoder.features(images)
            features = torch.relu(features)
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
            logits = self.head(features)
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError(f"Model head must return shape [batch, 2], got {tuple(logits.shape)}")
            return logits

    base_dir = Path(__file__).parent
    cache_dir = base_dir / "outputs" / "torchxrayvision"
    cache_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(str(checkpoint_file), map_location="cpu", weights_only=False)
    weights = checkpoint.get("weights", "densenet121-res224-all") if isinstance(checkpoint, dict) else "densenet121-res224-all"
    encoder = xrv.models.DenseNet(weights=weights, cache_dir=str(cache_dir))
    feature_dim = int(checkpoint.get("feature_dim", getattr(encoder.classifier, "in_features", 1024)))
    model = PneumoXAIModel(encoder=encoder, feature_dim=feature_dim)

    if isinstance(checkpoint, dict) and "head_state_dict" in checkpoint:
        head_state = checkpoint["head_state_dict"]
        if all(key.startswith("net.") for key in head_state):
            head_state = {key.replace("net.", "", 1): value for key, value in head_state.items()}
        model.head.load_state_dict(head_state)
        if "encoder_state_dict" in checkpoint:
            model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("Unsupported checkpoint format; expected a state dict or checkpoint dictionary.")

    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    model.to("cpu")
    model.eval()
    return model, _load_threshold(str(threshold_file))


def preprocess_image(uploaded_file) -> tuple:
    """
    Handle .dcm, .png, .jpg, .jpeg uploads.
    DICOM: pydicom read -> pixel array -> handle MONOCHROME1 inversion -> normalize [0,1] -> resize 224x224
    PNG/JPG: PIL open -> convert grayscale -> normalize [0,1] -> resize 224x224
    Apply TorchXRayVision normalization: image = (2 * image - 1) * 1024
    Return (tensor [1,1,224,224] float32 CPU, display_np [224,224] uint8 for visualization)
    """
    torch, _ = _require_torch()
    name = getattr(uploaded_file, "name", "").lower()
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()

    image = None
    if name.endswith(".dcm"):
        try:
            import pydicom

            ds = pydicom.dcmread(io.BytesIO(raw), force=True)
            image = ds.pixel_array.astype(np.float32)
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                image = np.max(image) - image
        except Exception:
            try:
                import streamlit as st

                st.warning("Could not parse as DICOM, trying as standard image...")
            except Exception:
                pass

    if image is None:
        pil_image = Image.open(io.BytesIO(raw)).convert("L")
        image = np.asarray(pil_image, dtype=np.float32)

    image = _resize_image(_normalize_np(image))
    display_np = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    model_input = (2.0 * image - 1.0) * 1024.0
    tensor = torch.from_numpy(model_input[None, None, :, :].astype(np.float32)).to("cpu")
    _assert_tensor_shape(tensor)
    return tensor, display_np


def run_inference(model, tensor, threshold) -> tuple:
    """
    Forward pass through full model (encoder + head).
    Apply softmax -> pneumonia probability = output[:,1].
    Apply threshold.
    Return (label: str, confidence: float, prob_pneumonia: float, prob_normal: float)
    label is 'Pneumonia' or 'Normal'
    """
    torch, _ = _require_torch()
    _assert_tensor_shape(tensor)
    model.eval()
    with torch.no_grad():
        logits = model(tensor.to("cpu"))
        _assert_logits_shape(logits)
        probs = torch.softmax(logits, dim=1)[0]
    prob_normal = float(probs[0].detach().cpu())
    prob_pneumonia = float(probs[1].detach().cpu())
    label = "Pneumonia" if prob_pneumonia >= float(threshold) else "Normal"
    confidence = prob_pneumonia if label == "Pneumonia" else prob_normal
    return label, confidence, prob_pneumonia, prob_normal


def _manual_gradcam_heatmap(model, tensor, class_index: int = 1) -> np.ndarray:
    torch, _ = _require_torch()
    _assert_tensor_shape(tensor)
    activations = []
    gradients = []
    target_layer = model.encoder.features.denseblock4
    x = tensor.clone().detach().to("cpu").requires_grad_(True)

    def forward_hook(_, __, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    handle = target_layer.register_forward_hook(forward_hook)
    model.zero_grad(set_to_none=True)
    try:
        logits = model(x)
        _assert_logits_shape(logits)
        logits[:, class_index].sum().backward()
        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")
        feature_map = activations[-1].detach()
        gradient = gradients[-1].detach()
    finally:
        handle.remove()

    if feature_map.ndim != 4:
        raise ValueError(f"Grad-CAM activation must be [batch, channels, h, w], got {tuple(feature_map.shape)}")
    if gradient.shape != feature_map.shape:
        raise ValueError(f"Grad-CAM gradient shape {tuple(gradient.shape)} does not match activation {tuple(feature_map.shape)}")

    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feature_map).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    heatmap = cam[0, 0].detach().cpu().numpy()
    return _normalize_np(heatmap)


def generate_gradcam(model, tensor, display_np) -> np.ndarray:
    """
    Hook into model.encoder.features.denseblock4
    Enable requires_grad on tensor.
    Compute Grad-CAM: global avg pool gradients as channel weights,
    weighted sum feature maps, ReLU, resize to 224x224.
    Overlay jet colormap at alpha=0.45 on grayscale X-ray.
    Return overlay as [224,224,3] uint8 RGB.
    """
    start = time.perf_counter()
    heatmap = _manual_gradcam_heatmap(model, tensor, class_index=1)
    overlay = _overlay_heatmap(display_np, heatmap, alpha=0.45)
    print(f"XAI timing Grad-CAM: {(time.perf_counter() - start) * 1000:.1f} ms")
    return overlay


def _manual_gradcampp_heatmap(model, tensor, class_index: int = 1) -> np.ndarray:
    torch, _ = _require_torch()
    _assert_tensor_shape(tensor)
    activations = []
    gradients = []
    target_layer = model.encoder.features.denseblock4
    x = tensor.clone().detach().to("cpu").requires_grad_(True)

    def forward_hook(_, __, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    handle = target_layer.register_forward_hook(forward_hook)
    model.zero_grad(set_to_none=True)
    try:
        logits = model(x)
        _assert_logits_shape(logits)
        logits[:, class_index].sum().backward()
        if not activations or not gradients:
            raise RuntimeError("Grad-CAM++ hooks did not capture activations and gradients.")
        feature_map = activations[-1].detach()
        gradient = gradients[-1].detach()
    finally:
        handle.remove()

    grad2 = gradient.pow(2)
    grad3 = gradient.pow(3)
    denom = 2.0 * grad2 + (feature_map * grad3).sum(dim=(2, 3), keepdim=True)
    denom = torch.where(denom != 0.0, denom, torch.ones_like(denom) * 1e-8)
    alphas = grad2 / denom
    weights = (alphas * torch.relu(gradient)).sum(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feature_map).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    heatmap = cam[0, 0].detach().cpu().numpy()
    return _normalize_np(heatmap)


def generate_gradcampp(model, tensor, display_np) -> np.ndarray:
    """
    Use pytorch_grad_cam.GradCAMPlusPlus with target layer model.encoder.features.denseblock4.
    If import fails, fall back to manual Grad-CAM++ implementation.
    Return overlay as [224,224,3] uint8 RGB.
    """
    start = time.perf_counter()
    _assert_tensor_shape(tensor)
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        with GradCAMPlusPlus(model=model, target_layers=[model.encoder.features.denseblock4]) as cam:
            grayscale_cam = cam(input_tensor=tensor.clone().detach().to("cpu"), targets=[ClassifierOutputTarget(1)])
        heatmap = np.asarray(grayscale_cam[0], dtype=np.float32)
        heatmap = _normalize_np(heatmap)
    except Exception:
        heatmap = _manual_gradcampp_heatmap(model, tensor, class_index=1)
    overlay = _overlay_heatmap(display_np, heatmap, alpha=0.45)
    print(f"XAI timing Grad-CAM++: {(time.perf_counter() - start) * 1000:.1f} ms")
    return overlay


def generate_ig(model, tensor, display_np, n_steps=20) -> np.ndarray:
    """
    Use captum.attr.IntegratedGradients.
    Baseline: zero tensor same shape as input.
    n_steps=20 for deployment speed.
    Sum absolute attributions across channel dim, normalize [0,1].
    Apply jet colormap overlay at alpha=0.45.
    Return overlay as [224,224,3] uint8 RGB.
    """
    start = time.perf_counter()
    torch, _ = _require_torch()
    from captum.attr import IntegratedGradients

    _assert_tensor_shape(tensor)
    model.eval()
    x = tensor.clone().detach().to("cpu").requires_grad_(True)
    baseline = torch.zeros_like(x)
    ig = IntegratedGradients(model)
    attributions = ig.attribute(x, baselines=baseline, target=1, n_steps=n_steps, method="gausslegendre")
    if attributions.ndim != 4 or attributions.shape[2:] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Integrated Gradients attribution must be [batch, channels, 224, 224], got {tuple(attributions.shape)}")
    heatmap = attributions.detach().abs().sum(dim=1)[0].cpu().numpy()
    heatmap = _normalize_np(heatmap)
    overlay = _overlay_heatmap(display_np, heatmap, alpha=0.45)
    print(f"XAI timing Integrated Gradients: {(time.perf_counter() - start) * 1000:.1f} ms")
    return overlay
