"""
Training script for SwinDeepfakeDetector.

Usage:
    python src/train.py
    python src/train.py --config config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Allow imports from src/ when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from dataset import FaceDetector, build_splits
from model import SwinDeepfakeDetector
from utils import AverageMeter, compute_metrics, set_seed


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_val(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """
    Video-level evaluation: averages frame logits within each video.
    The val DataLoader returns batches of shape (B, N_frames, C, H, W).
    """
    model.eval()
    all_probs: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for frames, labels in tqdm(loader, desc="Val", leave=False):
            # frames: (B, N_frames, C, H, W)
            B, N, C, H, W = frames.shape
            frames = frames.view(B * N, C, H, W).to(device)
            logits = model(frames)                      # (B*N,)
            logits = logits.view(B, N).mean(dim=1)     # (B,)  video-level mean
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    return compute_metrics(np.array(all_labels), np.array(all_probs))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    loss_meter = AverageMeter()

    for frames, labels in tqdm(loader, desc="Train", leave=False):
        frames = frames.to(device)          # (B, C, H, W)
        labels = labels.to(device)          # (B,)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(frames)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), frames.size(0))

    return loss_meter.avg


def main(cfg: dict):
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    face_detector = None
    if cfg["model"].get("use_mtcnn", False):
        face_detector = FaceDetector(
            margin=cfg["model"].get("mtcnn_margin", 30),
            device=str(device),
        )
        print("MTCNN face detection enabled.")

    train_ds, val_ds, _test_ds = build_splits(
        csv_path=cfg["data"]["labels_csv"],
        frames_dir=cfg["data"]["frames_dir"],
        val_split=cfg["data"]["val_split"],
        test_split=cfg["data"]["test_split"],
        image_size=224,
        seed=cfg["training"]["seed"],
        face_detector=face_detector,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"] // 2,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    # --- Model ---
    model = SwinDeepfakeDetector(
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    # --- Loss ---
    # Compute pos_weight from training label distribution
    labels_list = [s[1] for s in train_ds.samples]
    n_real = labels_list.count(0)
    n_fake = labels_list.count(1)
    pos_weight = torch.tensor([n_real / max(n_fake, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # --- Checkpoint dir ---
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_auc = 0.0
    freeze_epochs = cfg["training"].get("freeze_epochs", 0)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        # Backbone freeze schedule
        if epoch == 1 and freeze_epochs > 0:
            model.freeze_backbone()
            print(f"Epoch {epoch}: backbone frozen for {freeze_epochs} epochs")
        elif epoch == freeze_epochs + 1:
            model.unfreeze_backbone()
            print(f"Epoch {epoch}: backbone unfrozen")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            cfg["training"]["grad_clip"],
        )
        val_metrics = evaluate_val(model, val_loader, device)
        scheduler.step()

        auc = val_metrics["auc"]
        print(
            f"Epoch {epoch:03d}/{cfg['training']['epochs']} | "
            f"Loss: {train_loss:.4f} | "
            f"AUC: {auc:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if auc > best_auc:
            best_auc = auc
            save_path = ckpt_dir / "best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auc": best_auc,
                    "config": cfg,
                },
                save_path,
            )
            print(f"  ✓ Saved best checkpoint (AUC={best_auc:.4f}) → {save_path}")

    print(f"\nTraining complete. Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
