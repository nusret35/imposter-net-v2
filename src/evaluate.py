"""
Evaluation and inference script for SwinDeepfakeDetector.

Usage:
    # Evaluate on validation split (default)
    python src/evaluate.py --checkpoint checkpoints/best.pth

    # Evaluate on held-out test split
    python src/evaluate.py --checkpoint checkpoints/best.pth --split test

    # Single-image inference
    python src/evaluate.py --checkpoint checkpoints/best.pth --predict path/to/face.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset import FaceDetector, build_splits, build_transforms
from model import SwinDeepfakeDetector
from utils import compute_metrics


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, device: torch.device) -> tuple[SwinDeepfakeDetector, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SwinDeepfakeDetector(
        backbone=cfg["model"]["backbone"],
        pretrained=False,
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint — epoch {ckpt['epoch']}, best val AUC={ckpt['best_auc']:.4f}")
    return model, cfg


def evaluate(
    model: SwinDeepfakeDetector,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Video-level evaluation: averages frame logits within each video.
    Loader batches have shape (B, N_frames, C, H, W).
    """
    all_probs: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for frames, labels in tqdm(loader, desc="Evaluating"):
            B, N, C, H, W = frames.shape
            frames = frames.view(B * N, C, H, W).to(device)
            logits = model(frames)
            logits = logits.view(B, N).mean(dim=1)   # video-level mean
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    return compute_metrics(np.array(all_labels), np.array(all_probs))


def predict(
    image_path: str,
    checkpoint_path: str,
    device: torch.device | None = None,
    use_mtcnn: bool = False,
    mtcnn_margin: int = 30,
) -> float:
    """
    Run inference on a single face image.

    Returns:
        probability (0–1) that the image is a deepfake
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = load_model(checkpoint_path, device)

    img = cv2.imread(image_path)
    if img is None:
        raise IOError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Optional face crop
    if use_mtcnn or cfg["model"].get("use_mtcnn", False):
        margin = cfg["model"].get("mtcnn_margin", mtcnn_margin)
        detector = FaceDetector(margin=margin, device=str(device))
        img = detector.crop(img)

    transform = build_transforms(train=False)
    tensor = transform(image=img)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()

    return prob


def main():
    parser = argparse.ArgumentParser(description="Evaluate SwinDeepfakeDetector")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument(
        "--split", choices=["val", "test"], default="val",
        help="Which split to evaluate (default: val)"
    )
    parser.add_argument("--predict", default=None, help="Path to a single image for inference")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Single-image inference ---
    if args.predict:
        prob = predict(args.predict, args.checkpoint, device)
        verdict = "FAKE" if prob >= 0.5 else "REAL"
        print(f"\nImage: {args.predict}")
        print(f"Deepfake probability: {prob:.4f}  →  {verdict}")
        return

    # --- Dataset evaluation ---
    model, cfg = load_model(args.checkpoint, device)

    face_detector = None
    if cfg["model"].get("use_mtcnn", False):
        face_detector = FaceDetector(
            margin=cfg["model"].get("mtcnn_margin", 30),
            device=str(device),
        )

    _train_ds, val_ds, test_ds = build_splits(
        csv_path=cfg["data"]["labels_csv"],
        frames_dir=cfg["data"]["frames_dir"],
        val_split=cfg["data"]["val_split"],
        test_split=cfg["data"]["test_split"],
        image_size=224,
        seed=cfg["training"]["seed"],
        face_detector=face_detector,
    )

    eval_ds = val_ds if args.split == "val" else test_ds
    loader = DataLoader(
        eval_ds,
        batch_size=cfg["training"]["batch_size"] // 2,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    metrics = evaluate(model, loader, device)
    split_label = args.split.capitalize()
    print(f"\n--- {split_label} Results (video-level aggregation) ---")
    print(f"  AUC:      {metrics['auc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1:       {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
