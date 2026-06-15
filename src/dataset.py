"""
Dataset for FaceForensics++ with video-level train/val/test splitting.

Key guarantee: No video that appears in training is present in
validation or test.

Directory name mapping (CSV path → jpegs folder):
    "Face2Face/944_032.mp4"                  → Face2Face_944_032
    "original/500.mp4"                       → original_500
    "DeepFakeDetection/06_18__walking.mp4"   → DeepFakeDetection_06_18__walking
"""

import os
import random
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
LABEL_MAP = {"REAL": 0, "FAKE": 1}

# Type aliases
Sample = Tuple[Path, int]


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def path_to_dir_name(file_path: str) -> str:
    """
    Convert a CSV file path to the corresponding jpegs directory name.
    Replaces '/' with '_' and strips the file extension.

    e.g. "Face2Face/944_032.mp4" → "Face2Face_944_032"
    """
    return Path(file_path).with_suffix("").as_posix().replace("/", "_")


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def _get_frames(video_dir: Path) -> List[Path]:
    return sorted(
        f for f in video_dir.iterdir()
        if f.suffix.lower() in IMG_EXTENSIONS
    )


def _load_frame(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# MTCNN face detector
# ---------------------------------------------------------------------------

# Per-process MTCNN registry — avoids re-initialising CUDA in forked workers.
# Each DataLoader worker process gets its own MTCNN instance (always on CPU).
_mtcnn_registry: dict = {}


class FaceDetector:
    """
    MTCNN-based face detector that crops the largest face from a frame
    before it is passed through the augmentation pipeline.

    Falls back to returning the full image if no face is detected.

    MTCNN is initialised lazily on first use inside each worker process
    (always on CPU) to avoid the "Cannot re-initialize CUDA in forked
    subprocess" error that occurs when using num_workers > 0 with DataLoader.

    Requires: pip install facenet-pytorch
    """

    def __init__(self, margin: int = 30, min_face_size: int = 40):
        self.margin = margin
        self.min_face_size = min_face_size
        # Intentionally NOT initialising MTCNN here — see _get_mtcnn().

    def _get_mtcnn(self):
        """Return this process's MTCNN instance, creating it if needed."""
        pid = os.getpid()
        if pid not in _mtcnn_registry:
            try:
                from facenet_pytorch import MTCNN
            except ImportError as e:
                raise ImportError(
                    "facenet-pytorch is required for face detection. "
                    "Install it with: pip install facenet-pytorch"
                ) from e
            # Always CPU: MTCNN is lightweight and GPU can't be used in
            # forked DataLoader worker processes.
            _mtcnn_registry[pid] = MTCNN(
                keep_all=False,
                select_largest=True,
                min_face_size=self.min_face_size,
                post_process=False,
                device="cpu",
            )
        return _mtcnn_registry[pid]

    def crop(self, img: np.ndarray) -> np.ndarray:
        """
        Detect the largest face and return the cropped region (RGB numpy).
        Returns the original image if no face is detected.
        """
        from PIL import Image

        boxes, _ = self._get_mtcnn().detect(Image.fromarray(img))
        if boxes is None or len(boxes) == 0:
            return img

        h, w = img.shape[:2]
        x1, y1, x2, y2 = boxes[0]
        x1 = max(0, int(x1) - self.margin)
        y1 = max(0, int(y1) - self.margin)
        x2 = min(w, int(x2) + self.margin)
        y2 = min(h, int(y2) + self.margin)

        crop = img[y1:y2, x1:x2]
        return crop if crop.size > 0 else img


# ---------------------------------------------------------------------------
# Albumentations transforms
# ---------------------------------------------------------------------------

def build_transforms(train: bool, image_size: int = 224) -> A.Compose:
    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.ImageCompression(quality_range=(50, 90), p=0.5),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FFDataset(Dataset):
    """
    FaceForensics++ dataset where each video is a directory of frames.

    Train mode : returns one randomly sampled frame  → (C, H, W), label
    Eval mode  : returns all frames stacked           → (N_frames, C, H, W), label
                 Use video-level aggregation (mean logits) at inference time.

    Args:
        samples:       List of (video_dir, label) pairs.
        train:         If True, samples one frame randomly; else returns all frames.
        image_size:    Target H/W after resize.
        transform:     Albumentations pipeline. Defaults to build_transforms(train).
        face_detector: Optional FaceDetector. When provided, crops the face region
                       from each frame before applying transforms.
    """

    def __init__(
        self,
        samples: List[Sample],
        train: bool = True,
        image_size: int = 224,
        transform: Optional[A.Compose] = None,
        face_detector: Optional[FaceDetector] = None,
    ):
        self.samples = samples
        self.train = train
        self.transform = transform or build_transforms(train, image_size)
        self.face_detector = face_detector

    def __len__(self) -> int:
        return len(self.samples)

    def _process_frame(self, path: Path) -> torch.Tensor:
        img = _load_frame(path)
        if self.face_detector is not None:
            img = self.face_detector.crop(img)
        return self.transform(image=img)["image"]

    def __getitem__(self, idx: int):
        video_dir, label = self.samples[idx]
        frames = _get_frames(video_dir)

        if not frames:
            raise RuntimeError(f"No frames found in: {video_dir}")

        if self.train:
            tensor = self._process_frame(random.choice(frames))   # (C, H, W)
            return tensor, torch.tensor(label, dtype=torch.float32)
        else:
            tensors = [self._process_frame(fp) for fp in frames]
            return torch.stack(tensors), torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Video-level split helpers
# ---------------------------------------------------------------------------

def _parse_csv_to_samples(csv_path: str, frames_dir: Path) -> List[Sample]:
    """Read CSV and resolve each row to (dir_path, label)."""
    df = pd.read_csv(csv_path, index_col=0)
    samples: List[Sample] = []
    missing = 0

    for _, row in df.iterrows():
        label = LABEL_MAP.get(str(row["Label"]).upper())
        if label is None:
            continue

        dir_path = frames_dir / path_to_dir_name(str(row["File Path"]))
        if not dir_path.exists() or not _get_frames(dir_path):
            missing += 1
            continue

        samples.append((dir_path, label))

    if missing:
        warnings.warn(
            f"{missing} CSV rows skipped — directory not found or empty under '{frames_dir}'. "
            "Check that path_to_dir_name() matches your folder naming."
        )
    return samples


def _split_samples(
    samples: List[Sample],
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Randomly split videos into train / val / test."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(samples))

    n = len(indices)
    n_test = max(1, int(n * test_split))
    n_val  = max(1, int(n * val_split))

    test_samples  = [samples[i] for i in indices[:n_test]]
    val_samples   = [samples[i] for i in indices[n_test : n_test + n_val]]
    train_samples = [samples[i] for i in indices[n_test + n_val :]]
    return train_samples, val_samples, test_samples


def _print_split_summary(
    train: List[Sample],
    val: List[Sample],
    test: List[Sample],
) -> None:
    for name, samples in [("Train", train), ("Val", val), ("Test", test)]:
        n_real = sum(1 for _, l in samples if l == 0)
        n_fake = sum(1 for _, l in samples if l == 1)
        print(f"  {name:5s}: {len(samples):5d} videos  (real={n_real}, fake={n_fake})")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_splits(
    csv_path: str,
    frames_dir: str,
    val_split: float = 0.15,
    test_split: float = 0.15,
    image_size: int = 224,
    seed: int = 42,
    face_detector: Optional[FaceDetector] = None,
) -> Tuple[FFDataset, FFDataset, FFDataset]:
    """
    Build train / val / test datasets from labels.csv using video-level splitting.

    Each video is randomly assigned to exactly one split so that no video
    appears in both training and evaluation.

    Args:
        csv_path:       Path to labels.csv.
        frames_dir:     Root directory containing per-video frame directories.
        val_split:      Fraction of videos for validation.
        test_split:     Fraction of videos for testing.
        image_size:     Resize target for transforms.
        seed:           Random seed for reproducible splits.
        face_detector:  Optional FaceDetector passed to each FFDataset.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    frames_root = Path(frames_dir)

    samples = _parse_csv_to_samples(csv_path, frames_root)

    train_samples, val_samples, test_samples = _split_samples(
        samples, val_split, test_split, seed
    )

    print("Dataset split summary:")
    _print_split_summary(train_samples, val_samples, test_samples)

    return (
        FFDataset(train_samples, train=True,  image_size=image_size, face_detector=face_detector),
        FFDataset(val_samples,   train=False, image_size=image_size, face_detector=face_detector),
        FFDataset(test_samples,  train=False, image_size=image_size, face_detector=face_detector),
    )
