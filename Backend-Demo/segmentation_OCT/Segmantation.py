import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader, random_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_IMAGES_PATH = os.path.join(BASE_DIR, "resized_images.npy")
INPUT_MASKS_PATH = os.path.join(BASE_DIR, "resized_labeledimages.npy")

OUTPUT_DIR = os.path.join(BASE_DIR, "segmentation_outputs")

LOW_PCT = 20
HIGH_PCT = 80
NUM_CLASSES = 8
BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
VAL_RATIO = 0.2
SEED = 42
TARGET_SIZE = (256, 256)


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def per_class_dice_score(preds, targets, num_classes=NUM_CLASSES, eps=1e-7):
    dice_scores = []

    for cls in range(num_classes):
        pred_c = (preds == cls).float()
        target_c = (targets == cls).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice = (2 * intersection + eps) / (union + eps)
        dice_scores.append(dice.item())

    return dice_scores


def val_dice_score(preds, targets, num_classes=NUM_CLASSES, eps=1e-7):
    dice_scores = []

    for cls in range(num_classes):
        pred_c = (preds == cls).float()
        target_c = (targets == cls).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice = (2 * intersection + eps) / (union + eps)
        dice_scores.append(dice)

    return torch.mean(torch.stack(dice_scores))


class NPYSegDataset(Dataset):
    def __init__(self, images_path, masks_path):
        self.images = np.load(images_path)
        self.masks = np.load(masks_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        mask = self.masks[idx].astype(np.int64)

        return torch.from_numpy(img.copy()).float(), torch.from_numpy(mask.copy()).long()


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

    return score


def get_device():
    try:
        import torch_directml
    except ImportError:
        torch_directml = None

    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda_or_rocm"
    elif torch_directml is not None:
        try:
            return torch_directml.device(), "directml"
        except Exception:
            pass

    return torch.device("cpu"), "cpu"


def save_threshold_analysis(scores):
    threshold_tests = [(30, 70), (25, 75), (20, 80), (15, 85), (10, 90)]
    rows = []

    for low_pct, high_pct in threshold_tests:
        low_t = np.percentile(scores, low_pct)
        high_t = np.percentile(scores, high_pct)

        low_mask = scores <= low_t
        high_mask = scores >= high_t
        keep_mask = low_mask | high_mask

        low_scores = scores[low_mask]
        high_scores = scores[high_mask]

        rows.append({
            "low_pct": low_pct,
            "high_pct": high_pct,
            "low_threshold": low_t,
            "high_threshold": high_t,
            "kept_total": int(keep_mask.sum()),
            "kept_percent": 100 * keep_mask.sum() / len(scores),
            "low_count": int(low_mask.sum()),
            "high_count": int(high_mask.sum()),
            "discarded": int(len(scores) - keep_mask.sum()),
            "low_mean": float(low_scores.mean()),
            "high_mean": float(high_scores.mean()),
            "score_gap": float(high_scores.mean() - low_scores.mean())
        })

    csv_path = os.path.join(OUTPUT_DIR, "threshold_hardness_test.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    x = [f"{r['low_pct']}/{r['high_pct']}" for r in rows]
    gaps = [r["score_gap"] for r in rows]
    kept = [r["kept_total"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(x, gaps, marker="o", color="tab:blue")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Mean score gap", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(x, kept, marker="s", color="tab:red")
    ax2.set_ylabel("Kept images", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.suptitle("Confident labeling hardness test")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUTPUT_DIR, "threshold_hardness_gap.png"), dpi=150)
    plt.close(fig)


def save_label_visualizations(images_clean, scores_clean, binary_labels_clean):
    sorted_idx = np.argsort(scores_clean)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, idx in enumerate(sorted_idx[:5]):
        axes[0, i].imshow(images_clean[idx], cmap="gray")
        axes[0, i].set_title(f"Lav score\nscore={scores_clean[idx]:.1f}")
        axes[0, i].axis("off")

    for i, idx in enumerate(sorted_idx[-5:]):
        axes[1, i].imshow(images_clean[idx], cmap="gray")
        axes[1, i].set_title(f"Høy score\nscore={scores_clean[idx]:.1f}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "label_validation.png"), dpi=150)
    plt.close(fig)

    low_idx = np.where(binary_labels_clean == 0)[0]
    high_idx = np.where(binary_labels_clean == 1)[0]

    if len(low_idx) >= 5 and len(high_idx) >= 5:
        low_border = low_idx[np.argsort(scores_clean[low_idx])[-5:]]
        high_border = high_idx[np.argsort(scores_clean[high_idx])[:5]]

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i, idx in enumerate(low_border):
            axes[0, i].imshow(images_clean[idx], cmap="gray")
            axes[0, i].set_title(f"Lav grense\nscore={scores_clean[idx]:.1f}")
            axes[0, i].axis("off")

        for i, idx in enumerate(high_border):
            axes[1, i].imshow(images_clean[idx], cmap="gray")
            axes[1, i].set_title(f"Høy grense\nscore={scores_clean[idx]:.1f}")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "label_borderline.png"), dpi=150)
        plt.close(fig)


def save_pseudo_label_outputs():
    images_raw = np.load(INPUT_IMAGES_PATH)
    masks_raw = np.load(INPUT_MASKS_PATH)

    scores = np.array([
        combined_pathology_score(images_raw[i], masks_raw[i])
        for i in range(len(masks_raw))
    ])

    save_threshold_analysis(scores)

    low_threshold = np.percentile(scores, LOW_PCT)
    high_threshold = np.percentile(scores, HIGH_PCT)

    confident_mask = (scores <= low_threshold) | (scores >= high_threshold)
    confident_idx = np.where(confident_mask)[0]

    images_clean = images_raw[confident_idx]
    masks_clean = masks_raw[confident_idx]
    scores_clean = scores[confident_idx]
    binary_labels_clean = (scores_clean >= high_threshold).astype(np.int64)

    np.save(os.path.join(BASE_DIR, "binary_labels.npy"), binary_labels_clean)
    np.save(os.path.join(BASE_DIR, "pathology_scores.npy"), scores_clean)
    np.save(os.path.join(BASE_DIR, "pathology_scores_all.npy"), scores)
    np.save(os.path.join(BASE_DIR, "images_clean.npy"), images_clean)
    np.save(os.path.join(BASE_DIR, "masks_clean.npy"), masks_clean)
    np.save(os.path.join(BASE_DIR, "confident_indices.npy"), confident_idx)

    report_path = os.path.join(OUTPUT_DIR, "pseudo_label_report.csv")
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original_index", "pathology_score", "pseudo_label", "label_name"])
        for idx, score, label in zip(confident_idx, scores_clean, binary_labels_clean):
            label_name = "low_score" if label == 0 else "high_score"
            writer.writerow([int(idx), float(score), int(label), label_name])

    summary_path = os.path.join(OUTPUT_DIR, "pseudo_label_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["low_pct", LOW_PCT])
        writer.writerow(["high_pct", HIGH_PCT])
        writer.writerow(["original_count", len(scores)])
        writer.writerow(["kept_count", len(confident_idx)])
        writer.writerow(["kept_percent", 100 * len(confident_idx) / len(scores)])
        writer.writerow(["discarded_count", len(scores) - len(confident_idx)])
        writer.writerow(["low_threshold", float(low_threshold)])
        writer.writerow(["high_threshold", float(high_threshold)])
        writer.writerow(["label_0_count", int((binary_labels_clean == 0).sum())])
        writer.writerow(["label_1_count", int((binary_labels_clean == 1).sum())])

    save_label_visualizations(images_clean, scores_clean, binary_labels_clean)

    print("=== PSEUDO LABELING ===")
    print(f"Originalt antall bilder: {len(scores)}")
    print(f"Beholdt antall bilder:   {len(confident_idx)}")
    print(f"Label 0 count:          {(binary_labels_clean == 0).sum()}")
    print(f"Label 1 count:          {(binary_labels_clean == 1).sum()}")
    print(f"Lav threshold:          {low_threshold:.2f}")
    print(f"Høy threshold:          {high_threshold:.2f}")
    print("=======================\n")


def plot_training_curves(epochs, train_losses, val_losses, val_dices):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train Loss")
    ax.plot(epochs, val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, val_dices, label="Val Dice", color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice Score")
    ax.set_title("Validation Dice")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "dice_curve.png"), dpi=300)
    plt.close(fig)


def plot_per_class_dice(best_class_dice):
    classes = [f"Class {i}" for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(classes, best_class_dice, color="skyblue")
    ax.set_xlabel("Class")
    ax.set_ylabel("Dice Score")
    ax.set_title("Per-Class Dice at Best Epoch")
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "per_class_dice.png"), dpi=300)
    plt.close(fig)


def save_prediction_preview(model, val_loader, device):
    model.eval()
    imgs, masks = next(iter(val_loader))
    imgs_device = imgs.to(device)

    with torch.no_grad():
        out = model(imgs_device)

    preds = out.argmax(dim=1).cpu()

    n_show = min(3, imgs.shape[0])
    fig, axes = plt.subplots(n_show, 3, figsize=(10, 3 * n_show))

    if n_show == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n_show):
        axes[i, 0].imshow(imgs[i, 0].cpu().numpy(), cmap="gray")
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(masks[i].cpu().numpy(), cmap="nipy_spectral", vmin=0, vmax=NUM_CLASSES - 1)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(preds[i].cpu().numpy(), cmap="nipy_spectral", vmin=0, vmax=NUM_CLASSES - 1)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "prediction_preview.png"), dpi=200)
    plt.close(fig)


def train_segmentation():
    dataset = NPYSegDataset(INPUT_IMAGES_PATH, INPUT_MASKS_PATH)

    n_total = len(dataset)
    n_val = int(VAL_RATIO * n_total)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device, backend = get_device()
    print(f"Using backend: {backend}, device: {device}")

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=1,
        classes=NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    log_path = os.path.join(OUTPUT_DIR, "training_log.csv")
    best_model_path = os.path.join(BASE_DIR, "best_model.pth")
    summary_path = os.path.join(OUTPUT_DIR, "training_summary.csv")

    history_epochs = []
    history_train_loss = []
    history_val_loss = []
    history_val_dice = []

    best_epoch = 1
    best_val_loss = float("inf")
    best_dice = 0.0
    best_class_dice = np.zeros(NUM_CLASSES, dtype=np.float64)

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Epoch", "Train Loss", "Val Loss", "Val Dice", "Best Model",
            "Dice Class 0", "Dice Class 1", "Dice Class 2", "Dice Class 3",
            "Dice Class 4", "Dice Class 5", "Dice Class 6", "Dice Class 7"
        ])

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_class_dice = np.zeros(NUM_CLASSES, dtype=np.float64)

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                dice = val_dice_score(preds, masks, num_classes=NUM_CLASSES)
                val_dice += dice.item()

                class_dice = per_class_dice_score(preds, masks, num_classes=NUM_CLASSES)
                val_class_dice += np.array(class_dice)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_class_dice /= len(val_loader)

        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_dice = val_dice
            best_class_dice = val_class_dice.copy()
            torch.save(model.state_dict(), best_model_path)

        history_epochs.append(epoch + 1)
        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        history_val_dice.append(val_dice)

        print(
            f"Epoch {epoch + 1:>3}/{NUM_EPOCHS} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Dice: {val_dice:.4f} | "
            f"{'BEST' if is_best else ''}"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{train_loss:.4f}",
                f"{val_loss:.4f}",
                f"{val_dice:.4f}",
                "YES" if is_best else "",
                *[f"{x:.4f}" for x in val_class_dice]
            ])

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["best_epoch", best_epoch])
        writer.writerow(["best_val_loss", f"{best_val_loss:.4f}"])
        writer.writerow(["best_val_dice", f"{best_dice:.4f}"])
        writer.writerow(["architecture", "U-Net"])
        writer.writerow(["encoder", "resnet18"])
        writer.writerow(["encoder_weights", "imagenet"])
        writer.writerow(["input_channels", 1])
        writer.writerow(["output_classes", NUM_CLASSES])
        writer.writerow(["batch_size", BATCH_SIZE])
        writer.writerow(["learning_rate", LEARNING_RATE])
        writer.writerow(["num_epochs", NUM_EPOCHS])
        writer.writerow(["train_samples", n_train])
        writer.writerow(["val_samples", n_val])
        writer.writerow(["total_samples", n_total])
        writer.writerow(["backend", backend])
        for cls in range(NUM_CLASSES):
            writer.writerow([f"best_dice_class_{cls}", f"{best_class_dice[cls]:.4f}"])

    plot_training_curves(history_epochs, history_train_loss, history_val_loss, history_val_dice)
    plot_per_class_dice(best_class_dice)
    save_prediction_preview(model, val_loader, device)

    print("\n=== TRAINING SUMMARY ===")
    print(f"Best epoch:    {best_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val dice: {best_dice:.4f}")
    print(f"Saved model:   {best_model_path}")
    print("========================")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ensure_output_dir()
    save_pseudo_label_outputs()
    train_segmentation()
    print(f"\nRapporter og grafer ligger i: {OUTPUT_DIR}")
    print(f"Delte datafiler og best_model.pth ligger i: {BASE_DIR}")


if __name__ == "__main__":
    main()