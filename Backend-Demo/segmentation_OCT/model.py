import os
import glob
import torch
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "best_model.pth")
test_image_folder = os.path.join(script_dir, "test_img")
mask_folder = os.path.join(script_dir, "exported_masks")

print(f"Looking for model at: {model_path}")
print(f"Test image folder: {test_image_folder}")
print(f"Mask folder: {mask_folder}")

try:
    import torch_directml
except ImportError:
    torch_directml = None


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    elif torch_directml is not None:
        try:
            return torch_directml.device(), "directml"
        except Exception:
            pass
    return torch.device("cpu"), "cpu"


device, backend = get_device()
print(f"Using device: {device} ({backend})")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Could not find model file: {model_path}")

print("\nLoading trained model...")

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=1,
    classes=8
)

state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print("Model loaded successfully!")

TARGET_SIZE = (256, 256)


def load_and_preprocess(image_path, target_size=TARGET_SIZE):
    img = Image.open(image_path).convert("L")
    img = img.resize(target_size, Image.BILINEAR)
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def load_mask(mask_path, target_size=TARGET_SIZE):
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(target_size, Image.NEAREST)
    mask = np.array(mask).astype(np.int64)

    value_to_class = {
        0: 0,
        36: 1,
        73: 2,
        109: 3,
        146: 4,
        182: 5,
        219: 6,
        255: 7
    }

    class_mask = np.zeros_like(mask, dtype=np.int64)
    for pixel_value, class_id in value_to_class.items():
        class_mask[mask == pixel_value] = class_id

    return class_mask


def dice_score(pred, target, num_classes=8, eps=1e-7):
    dice_scores = []
    for cls in range(num_classes):
        pred_c = (pred == cls).float()
        target_c = (target == cls).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2 * intersection + eps) / (union + eps)
        dice_scores.append(dice)
    return torch.mean(torch.stack(dice_scores)).item()


def colorize_mask(mask, num_classes=8, cmap_name="tab10"):
    cmap = plt.get_cmap(cmap_name, num_classes)
    colored = cmap(mask)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    return Image.fromarray(colored)


def create_overlay(base_gray_np, pred_mask_np, alpha=0.45):
    base = Image.fromarray((base_gray_np * 255).astype(np.uint8)).convert("RGB")
    pred_color = colorize_mask(pred_mask_np).resize(base.size)
    overlay = Image.blend(base, pred_color, alpha=alpha)
    return overlay


def analyze_image(image_input):
    if hasattr(image_input, "read"):
        img_pil = Image.open(image_input).convert("L")
        img_pil = img_pil.resize(TARGET_SIZE, Image.BILINEAR)
        img = np.array(img_pil).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
    else:
        img = load_and_preprocess(image_input)
        img_pil = Image.fromarray((img.squeeze() * 255).astype(np.uint8))

    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).cpu().squeeze().numpy()

    mask_image = colorize_mask(pred)
    overlay_image = create_overlay(img.squeeze(), pred)

    return {
        "pred": pred,
        "mask_image": mask_image,
        "overlay_image": overlay_image
    }


def main():
    if not os.path.exists(test_image_folder):
        raise FileNotFoundError(f"Could not find test_img folder: {test_image_folder}")

    if not os.path.exists(mask_folder):
        print(f"Warning: mask folder not found: {mask_folder}")

    image_files = sorted(glob.glob(os.path.join(test_image_folder, "*.png")))
    print(f"\nFound {len(image_files)} images in test_img")

    if not image_files:
        raise FileNotFoundError("No PNG images found in test_img")

    mask_files = []
    for img_file in image_files:
        img_name = os.path.basename(img_file)

        possible_masks = [
            os.path.join(mask_folder, img_name),
            os.path.join(mask_folder, img_name.replace("img_", "mask_")),
            os.path.join(mask_folder, img_name.replace(".png", "_mask.png")),
        ]

        found_mask = None
        for candidate in possible_masks:
            if os.path.exists(candidate):
                found_mask = candidate
                break

        mask_files.append(found_mask)

    num_masks = sum(1 for m in mask_files if m)
    print(f"Found {num_masks} corresponding masks")

    print("\nAvailable test images:")
    for idx, path in enumerate(image_files):
        print(f"[{idx}] {os.path.basename(path)}")

    selection = input("\nEnter indices (e.g. 0 or 0,2,3) OR press Enter for all: ").strip()

    if selection:
        try:
            selected_indices = [int(x) for x in selection.split(",")]
            image_files = [image_files[i] for i in selected_indices if i < len(image_files)]
            mask_files = [mask_files[i] for i in selected_indices if i < len(mask_files)]
        except Exception as e:
            print(f"Invalid input, using all images. Error: {e}")

    print("\n" + "=" * 50)
    print("Testing selected images...")
    print("=" * 50)

    results = []
    sample_data = []

    for i in range(len(image_files)):
        img_path = image_files[i]
        img = load_and_preprocess(img_path)
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).cpu().squeeze().numpy()

        print(f"\nImage {i + 1}: {os.path.basename(img_path)}")
        print(f"  Prediction shape: {pred.shape}")
        print(f"  Unique classes in prediction: {np.unique(pred)}")

        dice = None
        mask = None

        if mask_files[i] and os.path.exists(mask_files[i]):
            mask = load_mask(mask_files[i])

            if pred.shape == mask.shape:
                dice = dice_score(torch.from_numpy(pred), torch.from_numpy(mask))
                print(f"  Dice Score: {dice:.4f} ({dice * 100:.2f}%)")
                results.append(dice)
            else:
                print(f"  Shape mismatch: pred {pred.shape} vs mask {mask.shape}")
        else:
            print("  No mask found for this image")

        sample_data.append({
            "filename": os.path.basename(img_path),
            "img": img.squeeze(),
            "pred": pred,
            "mask": mask,
            "dice": dice
        })

    if results:
        avg_dice = np.mean(results)
        print("\n" + "=" * 50)
        print(f"Average Dice Score: {avg_dice:.4f} ({avg_dice * 100:.2f}%)")
        print("=" * 50)

    num_viz = min(4, len(sample_data))
    fig, axes = plt.subplots(3, num_viz, figsize=(5 * num_viz, 9))

    if num_viz == 1:
        axes = np.array(axes).reshape(3, 1)

    for i in range(num_viz):
        item = sample_data[i]

        axes[0, i].imshow(item["img"], cmap="gray")
        axes[0, i].set_title(f'Input\n{item["filename"]}')
        axes[0, i].axis("off")

        if item["mask"] is not None:
            axes[1, i].imshow(item["mask"], cmap="tab10", vmin=0, vmax=7)
        else:
            axes[1, i].text(0.5, 0.5, "No Mask", ha="center", va="center")

        axes[1, i].set_title("Ground Truth")
        axes[1, i].axis("off")

        axes[2, i].imshow(item["pred"], cmap="tab10", vmin=0, vmax=7)
        title = f'Prediction\nDice: {item["dice"]:.3f}' if item["dice"] is not None else "Prediction"
        axes[2, i].set_title(title)
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.show()

    print("\nTesting complete!")


if __name__ == "__main__":
    main()