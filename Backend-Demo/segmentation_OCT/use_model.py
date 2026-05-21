import os
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import models
from scipy import ndimage

try:
    import torch_directml
except ImportError:
    torch_directml = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEGMENTATION_MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")
CLASSIFIER_MODEL_PATH = os.path.join(
    BASE_DIR,
    "stratified_5fold",
    "latest_best",
    "best_resnet18_classifier_fold_1.pth"
)
THICKNESS_MODEL_PATH = os.path.join(BASE_DIR, "thickness_classifier.joblib")

TARGET_SIZE_SEG = (256, 256)
TARGET_SIZE_CLS = (256, 256)
NUM_CLASSES = 8

LAYER_NAMES = {
    0: "Background",
    1: "Layer 1",
    2: "Layer 2",
    3: "Layer 3",
    4: "Layer 4",
    5: "Layer 5",
    6: "Layer 6",
    7: "Layer 7",
}

_CLASSIFIER = None
_SEGMENTATION_MODEL = None
_THICKNESS_MODEL = None
_DEVICE = None
_BACKEND = None


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    elif torch_directml is not None:
        try:
            return torch_directml.device(), "directml"
        except Exception:
            pass
    return torch.device("cpu"), "cpu"


def build_segmentation_model():
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=1,
        classes=NUM_CLASSES
    )
    return model


def build_classifier_model():
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model


def get_models():
    global _CLASSIFIER, _SEGMENTATION_MODEL, _THICKNESS_MODEL, _DEVICE, _BACKEND

    if _DEVICE is None:
        _DEVICE, _BACKEND = get_device()

    if _SEGMENTATION_MODEL is None:
        seg_model = build_segmentation_model()
        seg_state = torch.load(SEGMENTATION_MODEL_PATH, map_location="cpu", weights_only=False)
        seg_model.load_state_dict(seg_state)
        seg_model = seg_model.to(_DEVICE)
        seg_model.eval()
        _SEGMENTATION_MODEL = seg_model

    if _CLASSIFIER is None:
        cls_model = build_classifier_model()
        cls_state = torch.load(CLASSIFIER_MODEL_PATH, map_location="cpu", weights_only=False)
        cls_model.load_state_dict(cls_state)
        cls_model = cls_model.to(_DEVICE)
        cls_model.eval()
        _CLASSIFIER = cls_model

    if _THICKNESS_MODEL is None:
        _THICKNESS_MODEL = joblib.load(THICKNESS_MODEL_PATH)

    return _CLASSIFIER, _SEGMENTATION_MODEL, _THICKNESS_MODEL, _DEVICE, _BACKEND


def load_input_image(image_source):
    if isinstance(image_source, (str, os.PathLike)):
        img = Image.open(image_source).convert("L")
    elif hasattr(image_source, "getvalue"):
        img = Image.open(BytesIO(image_source.getvalue())).convert("L")
    elif hasattr(image_source, "read"):
        if hasattr(image_source, "seek"):
            image_source.seek(0)
        img = Image.open(image_source).convert("L")
    else:
        raise TypeError("Unsupported image input type.")
    return img


def preprocess_for_segmentation(image_pil, target_size=TARGET_SIZE_SEG):
    original = image_pil.copy()
    resized = image_pil.resize(target_size, Image.BILINEAR)

    img_np = np.array(resized).astype(np.float32) / 255.0
    img_np = np.expand_dims(img_np, axis=0)
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).float()

    return original, img_tensor


def preprocess_for_classifier(image_pil, target_size=TARGET_SIZE_CLS):
    resized = image_pil.resize(target_size, Image.BILINEAR)
    img_np = np.array(resized).astype(np.float32) / 255.0

    img_np = np.stack([img_np, img_np, img_np], axis=0)
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).float()

    return img_tensor


def colorize_mask(mask, num_classes=NUM_CLASSES, cmap_name="tab10"):
    cmap = plt.get_cmap(cmap_name, num_classes)
    colored = cmap(mask)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    return Image.fromarray(colored)


def create_overlay(base_gray_pil, pred_mask_np, alpha=0.45):
    base = base_gray_pil.convert("RGB")
    colored_mask = colorize_mask(pred_mask_np).resize(base.size, Image.NEAREST)
    overlay = Image.blend(base, colored_mask, alpha=alpha)
    return overlay


def create_mask_with_legend(mask, layer_names=LAYER_NAMES, cmap_name="tab10"):
    mask_img = colorize_mask(mask, num_classes=NUM_CLASSES, cmap_name=cmap_name).convert("RGB")
    mask_w, mask_h = mask_img.size

    legend_width = 260
    canvas = Image.new("RGB", (mask_w + legend_width, mask_h), color=(20, 24, 35))
    canvas.paste(mask_img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    title_x = mask_w + 15
    y = 15
    draw.text((title_x, y), "Layer legend", fill="white", font=font)
    y += 25

    cmap = plt.get_cmap(cmap_name, NUM_CLASSES)

    for class_id, layer_name in layer_names.items():
        if class_id == 0:
            continue

        color = tuple((np.array(cmap(class_id)[:3]) * 255).astype(np.uint8))
        draw.rectangle([title_x, y, title_x + 18, y + 18], fill=color, outline="white")
        draw.text((title_x + 28, y + 2), f"{layer_name}", fill="white", font=font)
        y += 28

    return canvas


def get_layer_thickness_by_column(mask, class_id):
    _, w = mask.shape
    thicknesses = np.full(w, np.nan, dtype=np.float32)

    for x in range(w):
        ys = np.where(mask[:, x] == class_id)[0]
        if ys.size > 0:
            thicknesses[x] = ys.max() - ys.min() + 1

    return thicknesses


def summarize_layer_thickness(mask, class_id, layer_name):
    thicknesses = get_layer_thickness_by_column(mask, class_id)
    valid = thicknesses[~np.isnan(thicknesses)]

    if valid.size == 0:
        return {
            "class_id": int(class_id),
            "layer_name": layer_name,
            "mean_thickness_px": None,
            "std_thickness_px": None,
            "min_thickness_px": None,
            "max_thickness_px": None,
            "coverage_percent": 0.0,
        }

    return {
        "class_id": int(class_id),
        "layer_name": layer_name,
        "mean_thickness_px": float(np.mean(valid)),
        "std_thickness_px": float(np.std(valid)),
        "min_thickness_px": float(np.min(valid)),
        "max_thickness_px": float(np.max(valid)),
        "coverage_percent": float((valid.size / mask.shape[1]) * 100.0),
    }


def compute_all_layer_metrics(mask, layer_names=LAYER_NAMES, include_background=False):
    rows = []
    for class_id, layer_name in layer_names.items():
        if not include_background and class_id == 0:
            continue
        rows.append(summarize_layer_thickness(mask, class_id, layer_name))
    return pd.DataFrame(rows)


def compute_total_retinal_thickness(mask, layer_ids=None):
    if layer_ids is None:
        layer_ids = [class_id for class_id in LAYER_NAMES.keys() if class_id != 0]

    _, w = mask.shape
    total_thickness = np.full(w, np.nan, dtype=np.float32)

    for x in range(w):
        ys = np.where(np.isin(mask[:, x], layer_ids))[0]
        if ys.size > 0:
            total_thickness[x] = ys.max() - ys.min() + 1

    valid = total_thickness[~np.isnan(total_thickness)]

    if valid.size == 0:
        return {
            "mean_total_retinal_thickness_px": None,
            "std_total_retinal_thickness_px": None,
            "min_total_retinal_thickness_px": None,
            "max_total_retinal_thickness_px": None,
            "coverage_percent": 0.0,
        }

    return {
        "mean_total_retinal_thickness_px": float(np.mean(valid)),
        "std_total_retinal_thickness_px": float(np.std(valid)),
        "min_total_retinal_thickness_px": float(np.min(valid)),
        "max_total_retinal_thickness_px": float(np.max(valid)),
        "coverage_percent": float((valid.size / w) * 100.0),
    }


def combined_pathology_score(image, mask):
    score = 0.0

    for cls in range(1, 8):
        layer = (mask == cls).astype(np.uint8)
        if layer.sum() == 0:
            continue

        _, num_features = ndimage.label(layer)
        score += num_features

        eroded = ndimage.binary_erosion(layer)
        edge_pixels = layer.sum() - eroded.sum()
        if layer.sum() > 0:
            score += (edge_pixels / layer.sum()) * 10

        filled = ndimage.binary_fill_holes(layer)
        holes = filled.sum() - layer.sum()
        score += holes / 100.0

    for cls in range(1, 8):
        layer = (mask == cls)
        col_thickness = layer.sum(axis=0)
        col_thickness = col_thickness[col_thickness > 0]

        if len(col_thickness) > 5:
            diffs = np.abs(np.diff(col_thickness))
            score += np.mean(diffs) * 2.0
            score += np.std(col_thickness) * 0.3

    rpe_classes = [6, 7]
    for cls in rpe_classes:
        rpe_mask = (mask == cls)
        if rpe_mask.sum() < 50:
            continue

        rpe_intensities = image[rpe_mask].astype(np.float32)
        score += np.std(rpe_intensities) * 0.3

        threshold_intensity = np.percentile(rpe_intensities, 95)
        bright_outliers = np.sum(rpe_intensities > threshold_intensity * 1.2)
        score += bright_outliers / 50.0

    return float(score)


def extract_thickness_features_from_mask(mask):
    features = {}

    total_present = (mask > 0)
    total_col = total_present.sum(axis=0).astype(np.float32)
    total_valid = total_col[total_col > 0]

    features["total_mean"] = float(np.mean(total_valid)) if len(total_valid) else np.nan
    features["total_std"] = float(np.std(total_valid)) if len(total_valid) else np.nan
    features["total_min"] = float(np.min(total_valid)) if len(total_valid) else np.nan
    features["total_max"] = float(np.max(total_valid)) if len(total_valid) else np.nan
    features["total_coverage"] = float(len(total_valid) / mask.shape[1])

    for cls in range(1, NUM_CLASSES):
        thickness = get_layer_thickness_by_column(mask, cls)
        valid = thickness[~np.isnan(thickness)]

        prefix = f"layer_{cls}"
        features[f"{prefix}_mean"] = float(np.mean(valid)) if len(valid) else np.nan
        features[f"{prefix}_std"] = float(np.std(valid)) if len(valid) else np.nan
        features[f"{prefix}_min"] = float(np.min(valid)) if len(valid) else np.nan
        features[f"{prefix}_max"] = float(np.max(valid)) if len(valid) else np.nan
        features[f"{prefix}_coverage"] = float(len(valid) / mask.shape[1])

        if len(valid) > 1:
            diffs = np.abs(np.diff(valid))
            features[f"{prefix}_diff_mean"] = float(np.mean(diffs))
            features[f"{prefix}_diff_std"] = float(np.std(diffs))
        else:
            features[f"{prefix}_diff_mean"] = np.nan
            features[f"{prefix}_diff_std"] = np.nan

    return pd.DataFrame([features])


def analyze_image(image_source):
    classifier_model, segmentation_model, thickness_model, device, backend = get_models()

    image_pil = load_input_image(image_source)

    original_img, seg_tensor = preprocess_for_segmentation(image_pil)
    cls_tensor = preprocess_for_classifier(image_pil)

    seg_tensor = seg_tensor.to(device)
    cls_tensor = cls_tensor.to(device)

    with torch.no_grad():
        seg_output = segmentation_model(seg_tensor)
        seg_probs = torch.softmax(seg_output, dim=1)
        pred_mask = torch.argmax(seg_probs, dim=1).cpu().squeeze().numpy()

        cls_output = classifier_model(cls_tensor)
        cls_probs = torch.softmax(cls_output, dim=1)[0].cpu().numpy()

    colored_mask = colorize_mask(pred_mask)
    overlay_image = create_overlay(original_img, pred_mask)
    mask_with_legend = create_mask_with_legend(pred_mask)

    layer_metrics_df = compute_all_layer_metrics(pred_mask)
    layer_metrics = layer_metrics_df.to_dict(orient="records")
    total_retinal_metrics = compute_total_retinal_thickness(pred_mask)
    pathology_score = combined_pathology_score(
        np.array(image_pil.resize(TARGET_SIZE_SEG, Image.BILINEAR)),
        pred_mask
    )

    thickness_features = extract_thickness_features_from_mask(pred_mask)
    thickness_probs = thickness_model.predict_proba(thickness_features)[0]

    combined_healthy_prob = (float(cls_probs[0]) + float(thickness_probs[0])) / 2.0
    combined_unhealthy_prob = (float(cls_probs[1]) + float(thickness_probs[1])) / 2.0

    image_prediction = "Unhealthy" if float(cls_probs[1]) >= 0.5 else "Healthy"
    thickness_prediction = "Unhealthy" if float(thickness_probs[1]) >= 0.5 else "Healthy"

    if combined_unhealthy_prob >= 0.60:
        final_prediction = "Unhealthy"
        cls_conf = combined_unhealthy_prob
    elif combined_unhealthy_prob <= 0.40:
        final_prediction = "Healthy"
        cls_conf = combined_healthy_prob
    else:
        final_prediction = "Uncertain"
        cls_conf = max(combined_healthy_prob, combined_unhealthy_prob)

    return {
        "prediction": final_prediction,
        "final_prediction": final_prediction,
        "image_prediction": image_prediction,
        "thickness_prediction": thickness_prediction,
        "confidence": cls_conf,
        "class_probabilities": {
            "Healthy": combined_healthy_prob,
            "Unhealthy": combined_unhealthy_prob,
        },
        "image_model_probabilities": {
            "Healthy": float(cls_probs[0]),
            "Unhealthy": float(cls_probs[1]),
        },
        "thickness_probabilities": {
            "Healthy": float(thickness_probs[0]),
            "Unhealthy": float(thickness_probs[1]),
        },
        "mask": pred_mask,
        "colored_mask": colored_mask,
        "overlay": overlay_image,
        "mask_with_legend": mask_with_legend,
        "backend": backend,
        "original_image": original_img,
        "layer_metrics": layer_metrics,
        "total_retinal_metrics": total_retinal_metrics,
        "pathology_score": pathology_score,
    }