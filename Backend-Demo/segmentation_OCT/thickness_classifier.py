import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MASKS_PATH = os.path.join(BASE_DIR, "masks_clean.npy")
LABELS_PATH = os.path.join(BASE_DIR, "binary_labels.npy")
OUTPUT_MODEL_PATH = os.path.join(BASE_DIR, "thickness_classifier.joblib")
OUTPUT_FEATURES_PATH = os.path.join(BASE_DIR, "thickness_features.csv")

NUM_CLASSES = 8


def get_layer_thickness_by_column(mask, class_id):
    _, w = mask.shape
    thicknesses = np.full(w, np.nan, dtype=np.float32)

    for x in range(w):
        ys = np.where(mask[:, x] == class_id)[0]
        if ys.size > 0:
            thicknesses[x] = ys.max() - ys.min() + 1

    return thicknesses


def extract_features_from_mask(mask):
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

    return features


def main():
    masks = np.load(MASKS_PATH)
    labels = np.load(LABELS_PATH)

    rows = [extract_features_from_mask(mask) for mask in masks]
    df = pd.DataFrame(rows)
    df["label"] = labels
    df.to_csv(OUTPUT_FEATURES_PATH, index=False)

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("\nConfusion matrix:\n", confusion_matrix(y_val, y_pred))
    print("\nClassification report:\n", classification_report(y_val, y_pred, digits=4))

    joblib.dump(model, OUTPUT_MODEL_PATH)
    print(f"\nSaved model to: {OUTPUT_MODEL_PATH}")
    print(f"Saved features to: {OUTPUT_FEATURES_PATH}")


if __name__ == "__main__":
    main()