import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split
from torchvision import models

try:
    import torch_directml
except ImportError:
    torch_directml = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = "gradcam_occlusion_outputs"
STRATIFIED_DIR = os.path.join(BASE_DIR, "stratified_5fold")
LATEST_BEST_DIR = os.path.join(STRATIFIED_DIR, "latest_best")
DEFAULT_FOLD = 1


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda_or_rocm"
    elif torch_directml is not None:
        try:
            return torch_directml.device(), "directml"
        except Exception:
            pass
    return torch.device("cpu"), "cpu"


def build_model(device):
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)
    return model


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        score = output[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], device=activations.device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
        pred = int(output.argmax(dim=1).item())
        conf = float(probs[pred])

        return cam, pred, probs, conf

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def preprocess_image(img):
    img = img.astype(np.float32)
    img = img / 255.0
    img = np.stack([img, img, img], axis=0)
    return torch.from_numpy(img.copy()).unsqueeze(0)


def resize_cam(cam, target_hw):
    cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
    cam_tensor = torch.nn.functional.interpolate(
        cam_tensor,
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    return cam_tensor.squeeze().numpy()


def overlay_heatmap(img, cam, alpha=0.4):
    img_norm = img.astype(np.float32)
    img_norm -= img_norm.min()
    if img_norm.max() > 0:
        img_norm /= img_norm.max()

    heatmap = cm.jet(cam)[..., :3]
    base = np.stack([img_norm, img_norm, img_norm], axis=-1)
    overlay = (1 - alpha) * base + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)

    return base, heatmap, overlay


def split_pool(labels, use_val_split=True, random_state=42):
    all_indices = np.arange(len(labels))

    if use_val_split:
        _, val_idx = train_test_split(
            all_indices,
            test_size=0.2,
            random_state=random_state,
            stratify=labels,
        )
        return val_idx

    return all_indices


def collect_predictions(model, images, labels, pool_indices, device, gradcam, target_mode):
    records = []

    for idx in pool_indices:
        img = images[idx].astype(np.float32)
        label = int(labels[idx])
        input_tensor = preprocess_image(img).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            pred = int(logits.argmax(dim=1).item())
            conf = float(probs[pred])

        target_class = label if target_mode == "true" else pred
        cam, pred2, probs2, conf2 = gradcam.generate(input_tensor, class_idx=target_class)

        records.append({
            "idx": int(idx),
            "label": label,
            "pred": pred2,
            "correct": int(pred2 == label),
            "p0": float(probs2[0]),
            "p1": float(probs2[1]),
            "conf": float(conf2),
            "target_class": int(target_class),
            "img": img,
            "cam": cam,
        })

    return records


def pick_balanced_examples(records, per_class=6):
    selected = []

    for cls in [0, 1]:
        group = [r for r in records if r["label"] == cls]
        correct = [r for r in group if r["correct"] == 1]
        wrong = [r for r in group if r["correct"] == 0]

        correct_sorted = sorted(correct, key=lambda x: x["conf"], reverse=True)

        easy_n = min(max(per_class // 2, 1), len(correct_sorted))
        easy = correct_sorted[:easy_n]

        remaining_correct = [r for r in correct_sorted if r not in easy]
        ambiguous_sorted = sorted(remaining_correct, key=lambda x: x["conf"])
        ambiguous_n = per_class - len(easy)
        ambiguous = ambiguous_sorted[:ambiguous_n]

        chosen = easy + ambiguous

        if len(chosen) < per_class:
            leftovers = [r for r in wrong if r not in chosen] + [r for r in remaining_correct if r not in chosen]
            leftovers = sorted(leftovers, key=lambda x: x["conf"])
            for r in leftovers:
                if len(chosen) >= per_class:
                    break
                chosen.append(r)

        chosen = sorted(chosen, key=lambda x: (x["correct"], x["conf"]), reverse=True)
        selected.extend(chosen[:per_class])

    return selected


def make_occlusion_map(model, input_tensor, target_class, patch_size=24, stride=12, occlusion_value=0.0):
    model.eval()

    with torch.no_grad():
        base_logits = model(input_tensor)
        base_probs = torch.softmax(base_logits, dim=1)[0]
        base_score = float(base_probs[target_class].item())

    _, _, h, w = input_tensor.shape
    score_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + patch_size, h)
            x2 = min(x + patch_size, w)

            occluded = input_tensor.clone()
            occluded[:, :, y:y2, x:x2] = occlusion_value

            with torch.no_grad():
                logits = model(occluded)
                probs = torch.softmax(logits, dim=1)[0]
                occluded_score = float(probs[target_class].item())

            drop = base_score - occluded_score
            score_map[y:y2, x:x2] += drop
            count_map[y:y2, x:x2] += 1.0

    count_map[count_map == 0] = 1.0
    score_map = score_map / count_map
    score_map = np.maximum(score_map, 0)

    if score_map.max() > 0:
        score_map = score_map / score_map.max()

    return score_map, base_score


def save_single_gradcam_figure(record, out_path):
    cam_resized = resize_cam(record["cam"], record["img"].shape)
    base, heatmap, overlay = overlay_heatmap(record["img"], cam_resized)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(base, cmap="gray")
    axes[0].set_title(f"Original\nidx={record['idx']} true={record['label']}")
    axes[0].axis("off")

    axes[1].imshow(heatmap)
    axes[1].set_title(f"GradCAM\nclass={record['target_class']}")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay\npred={record['pred']} | p0={record['p0']:.3f}, p1={record['p1']:.3f}"
    )
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_single_occlusion_figure(record, occ_map, out_path):
    base, heatmap, overlay = overlay_heatmap(record["img"], occ_map)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(base, cmap="gray")
    axes[0].set_title(f"Original\nidx={record['idx']} true={record['label']}")
    axes[0].axis("off")

    axes[1].imshow(heatmap)
    axes[1].set_title(f"Occlusion\nclass={record['target_class']}")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay\npred={record['pred']} | p0={record['p0']:.3f}, p1={record['p1']:.3f}"
    )
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_grid_figure(records, maps, out_path, title, map_name="GradCAM"):
    n = len(records)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.2 * n))

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, record in enumerate(records):
        current_map = maps[r]

        if map_name.lower() == "gradcam":
            current_map = resize_cam(current_map, record["img"].shape)

        base, heatmap, overlay = overlay_heatmap(record["img"], current_map)

        axes[r, 0].imshow(base, cmap="gray")
        axes[r, 0].set_title(f"Original | idx={record['idx']} true={record['label']}")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(heatmap)
        axes[r, 1].set_title(f"{map_name} | class={record['target_class']}")
        axes[r, 1].axis("off")

        axes[r, 2].imshow(overlay)
        axes[r, 2].set_title(
            f"Overlay | pred={record['pred']} | p0={record['p0']:.3f}, p1={record['p1']:.3f}"
        )
        axes[r, 2].axis("off")

    fig.suptitle(title, fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.992])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="GradCAM + Occlusion analysis for ResNet18 OCT classifier")
    parser.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--images-path", type=str, default="images_clean.npy")
    parser.add_argument("--labels-path", type=str, default="binary_labels.npy")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--use-val-split", action="store_true")
    parser.add_argument("--target-mode", type=str, default="pred", choices=["pred", "true"])
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=24)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--occlusion-value", type=float, default=0.0)
    args = parser.parse_args()

    device, backend = get_device()
    print(f"Using backend: {backend}, device: {device}")

    images_path = args.images_path if os.path.isabs(args.images_path) else os.path.join(BASE_DIR, args.images_path)
    labels_path = args.labels_path if os.path.isabs(args.labels_path) else os.path.join(BASE_DIR, args.labels_path)
    if args.model_path is None:
        model_path = os.path.join(
            LATEST_BEST_DIR,
            f"best_resnet18_classifier_fold_{args.fold}.pth"
        )
    else:
        model_path = args.model_path if os.path.isabs(args.model_path) else os.path.join(BASE_DIR, args.model_path)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(BASE_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    images = np.load(images_path)
    labels = np.load(labels_path)

    if not os.path.exists(model_path):
       raise FileNotFoundError(f"Fant ikke modellfil: {model_path}")

    model = build_model(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    target_layer = model.layer4[-1].conv2
    gradcam = GradCAM(model, target_layer)

    pool_indices = split_pool(labels, use_val_split=args.use_val_split, random_state=42)
    records = collect_predictions(model, images, labels, pool_indices, device, gradcam, args.target_mode)
    selected = pick_balanced_examples(records, per_class=args.per_class)

    gradcam_maps = [record["cam"] for record in selected]
    occlusion_maps = []
    summary_lines = []

    for record in selected:
        gradcam_file_name = (
            f"gradcam_idx{record['idx']}_true{record['label']}_pred{record['pred']}_"
            f"class{record['target_class']}.png"
        )
        gradcam_out_path = os.path.join(output_dir, gradcam_file_name)
        save_single_gradcam_figure(record, gradcam_out_path)

        input_tensor = preprocess_image(record["img"]).to(device)
        occ_map, base_score = make_occlusion_map(
            model=model,
            input_tensor=input_tensor,
            target_class=record["target_class"],
            patch_size=args.patch_size,
            stride=args.stride,
            occlusion_value=args.occlusion_value,
        )
        occlusion_maps.append(occ_map)

        occlusion_file_name = (
            f"occlusion_idx{record['idx']}_true{record['label']}_pred{record['pred']}_"
            f"class{record['target_class']}.png"
        )
        occlusion_out_path = os.path.join(output_dir, occlusion_file_name)
        save_single_occlusion_figure(record, occ_map, occlusion_out_path)

        line = (
            f"idx={record['idx']} | true={record['label']} | pred={record['pred']} | "
            f"correct={record['correct']} | target_class={record['target_class']} | "
            f"p0={record['p0']:.4f} | p1={record['p1']:.4f} | conf={record['conf']:.4f} | "
            f"base_score={base_score:.4f} | gradcam_file={gradcam_file_name} | "
            f"occlusion_file={occlusion_file_name}"
        )
        summary_lines.append(line)
        print(line)

    gradcam_grid_path = os.path.join(output_dir, "gradcam_grid_12_examples.png")
    save_grid_figure(
        selected,
        gradcam_maps,
        gradcam_grid_path,
        title="GradCAM examples for ResNet18 OCT classifier",
        map_name="GradCAM",
    )

    occlusion_grid_path = os.path.join(output_dir, "occlusion_grid_12_examples.png")
    save_grid_figure(
        selected,
        occlusion_maps,
        occlusion_grid_path,
        title="Occlusion sensitivity examples for ResNet18 OCT classifier",
        map_name="Occlusion",
    )

    summary_path = os.path.join(output_dir, "gradcam_occlusion_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("GradCAM + Occlusion summary\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Images path: {images_path}\n")
        f.write(f"Labels path: {labels_path}\n")
        f.write("Target layer: model.layer4[-1].conv2\n")
        f.write(f"Target mode: {args.target_mode}\n")
        f.write(f"Use val split: {args.use_val_split}\n")
        f.write(f"Per class: {args.per_class}\n")
        f.write(f"Patch size: {args.patch_size}\n")
        f.write(f"Stride: {args.stride}\n")
        f.write(f"Occlusion value: {args.occlusion_value}\n")
        f.write(f"Total selected: {len(selected)}\n")
        f.write(f"Selected indices: {[r['idx'] for r in selected]}\n\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write(f"\nCombined GradCAM grid: {gradcam_grid_path}\n")
        f.write(f"Combined occlusion grid: {occlusion_grid_path}\n")

    gradcam.close()

    print(f"Saved results to: {output_dir}")
    print(f"Saved GradCAM grid to: {gradcam_grid_path}")
    print(f"Saved occlusion grid to: {occlusion_grid_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()