import os
import csv
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from torchvision import models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRATIFIED_DIR = os.path.join(BASE_DIR, "stratified_5fold")
LATEST_BEST_DIR = os.path.join(STRATIFIED_DIR, "latest_best")


class OCTClassificationDataset(Dataset):
    def __init__(self, images_path, labels_path, indices=None, augment=False):
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)
        self.indices = indices
        self.augment = augment

        if self.indices is None:
            self.indices = np.arange(len(self.images))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        img = self.images[real_idx].astype(np.float32)
        label = int(self.labels[real_idx])

        if self.augment:
          img = self.apply_augmentation(img)

        img = img / 255.0
        img = np.stack([img, img, img], axis=0).astype(np.float32)

        return torch.from_numpy(img.copy()).float(), torch.tensor(label, dtype=torch.long)

# lager augmentering av dataset klassen
    def apply_augmentation(self, img):
        if np.random.rand() < 0.5:
            img = np.fliplr(img)

        if np.random.rand() < 0.4:
            alpha = np.random.uniform(0.9, 1.1)
            beta = np.random.uniform(-8, 8)
            img = img * alpha + beta
            img = np.clip(img, 0, 255)

        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 4, img.shape)
            img = img + noise
            img = np.clip(img, 0, 255)

        return img.astype(np.float32)


try:
    import torch_directml
except ImportError:
    torch_directml = None


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda_or_rocm"
    elif torch_directml is not None:
        try:
            return torch_directml.device(), "directml"
        except Exception:
            pass
    return torch.device("cpu"), "cpu"


def create_resnet18_model(device):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    return model.to(device)

#train loop
def train_one_epoch(model, train_loader, criterion, optimizer, device): 
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for imgs, labels_batch in train_loader:
        imgs = imgs.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels_batch).sum().item()
        train_total += labels_batch.size(0)

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    return train_loss, train_acc


def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels_batch in val_loader:
            imgs = imgs.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels_batch)

            val_loss += loss.item()
            preds = outputs.argmax(dim=1)

            val_correct += (preds == labels_batch).sum().item()
            val_total += labels_batch.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    return val_loss, val_acc, np.array(all_preds), np.array(all_labels)


# ── LOAD DATA ─────────────────────────────────────────────
images_path = os.path.join(BASE_DIR, "images_clean.npy")
labels_path = os.path.join(BASE_DIR, "binary_labels.npy")

images = np.load(images_path)
labels = np.load(labels_path).astype(np.int64)
all_indices = np.arange(len(labels))

print("=== CLASSIFICATION DATASET ===")
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Label counts: {np.bincount(labels)}")
print("==============================")

del images


# ── DEVICE ────────────────────────────────────────────────
device, backend = get_device()
print(f"Using backend: {backend}, device: {device}")


# ── 5-FOLD STRATIFIED K-FOLD ──────────────────────────────
num_epochs = 30
batch_size = 8
n_splits = 5

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

run_dir = os.path.join(STRATIFIED_DIR, f"run_{run_id}")
os.makedirs(run_dir, exist_ok=True)
os.makedirs(LATEST_BEST_DIR, exist_ok=True)

log_path = os.path.join(run_dir, f"resnet18_5fold_log_{run_id}.csv")
summary_path = os.path.join(run_dir, f"resnet18_5fold_summary_{run_id}.csv")
latest_summary_path = os.path.join(LATEST_BEST_DIR, "latest_5fold_summary.csv")

with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Fold", "Epoch",
        "Train Loss", "Train Acc",
        "Val Loss", "Val Acc",
        "Best Model"
    ])

#stratifiedkfold
skf = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42
)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(all_indices, labels), start=1):
    print(f"\n========== FOLD {fold}/{n_splits} ==========")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")
    print(f"Train labels: {np.bincount(labels[train_idx], minlength=2)}")
    print(f"Val labels:   {np.bincount(labels[val_idx], minlength=2)}")

    train_ds = OCTClassificationDataset(
        images_path,
        labels_path,
        indices=train_idx,
        augment=True
    )

    val_ds = OCTClassificationDataset(
        images_path,
        labels_path,
        indices=val_idx,
        augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    imgs_check, labels_check = next(iter(train_loader))
    print("Batch imgs:", imgs_check.shape, imgs_check.dtype)
    print("Batch labels:", labels_check.shape, labels_check.dtype)
    print("Batch labels unique:", torch.unique(labels_check))

 #Resnet 18 bytter siste lag til 2 klasser med lav-score og høy-score patologi
    model = create_resnet18_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 1
    best_model_path = os.path.join(run_dir, f"best_resnet18_classifier_fold_{fold}.pth")
    latest_best_model_path = os.path.join(LATEST_BEST_DIR, f"best_resnet18_classifier_fold_{fold}.pth")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc, val_preds, val_true = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        is_best = val_acc > best_val_acc

        if is_best:
         best_val_acc = val_acc
         best_val_loss = val_loss
         best_epoch = epoch + 1
         torch.save(model.state_dict(), best_model_path)
         torch.save(model.state_dict(), latest_best_model_path)

        print(
            f"Fold {fold} | Epoch {epoch+1:>3}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"{' BEST ' if is_best else ''}"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                fold,
                epoch + 1,
                f"{train_loss:.4f}",
                f"{train_acc:.4f}",
                f"{val_loss:.4f}",
                f"{val_acc:.4f}",
                "YES" if is_best else ""
            ])

    fold_results.append({
    "fold": fold,
    "best_epoch": best_epoch,
    "best_val_loss": best_val_loss,
    "best_val_acc": best_val_acc,
    "run_model_path": best_model_path,
    "latest_model_path": latest_best_model_path
     })


# ── SAVE SUMMARY ──────────────────────────────────────────
results_df = pd.DataFrame(fold_results)
results_df.to_csv(summary_path, index=False)
results_df.to_csv(latest_summary_path, index=False)

mean_acc = results_df["best_val_acc"].mean()
std_acc = results_df["best_val_acc"].std()
mean_loss = results_df["best_val_loss"].mean()
std_loss = results_df["best_val_loss"].std()

print("\n========== 5-FOLD SUMMARY ==========")
print(results_df)
print(f"Mean val acc:  {mean_acc:.4f}")
print(f"Std val acc:   {std_acc:.4f}")
print(f"Mean val loss: {mean_loss:.4f}")
print(f"Std val loss:  {std_loss:.4f}")
print(f"Log file:      {log_path}")
print(f"Summary file:  {summary_path}")
print("====================================")