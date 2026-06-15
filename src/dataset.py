"""
Dataset for FaceForensics++ with identity-based train/val/test splitting.

Key guarantee: No face identity that appears in training is present in
validation or test — whether via original or manipulated videos.

Directory name mapping (CSV path → jpegs folder):
    "Face2Face/944_032.mp4"                  → Face2Face_944_032
    "original/500.mp4"                       → original_500
    "DeepFakeDetection/06_18__walking.mp4"   → DeepFakeDetection_06_18__walking

Identity extraction:
    "original/500.mp4"          → ["500"]
    "Face2Face/944_032.mp4"     → ["944", "032"]
    "DeepFakeDetection/06_18__" → ["06", "18"]   (split on __ first)
"""

import random
import warnings
from pathlib import Path
from typing import List, Optional, Set, Tuple

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
Record = Tuple[Path, int, List[str]]


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


def extract_identities(file_path: str) -> List[str]:
    """
    Extract numeric identity IDs from a video file path.

    Handles all FF++ naming conventions:
      original/500.mp4                         -> ["500"]
      Face2Face/944_032.mp4                    -> ["944", "032"]
      DeepFakeDetection/06_18__walking....mp4  -> ["06", "18"]
    """
    stem = Path(file_path).stem
    first_part = stem.split("__")[0]   # drop description after double underscore
    parts = first_part.split("_")
    return [p for p in parts if p.isdigit()]


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

class FaceDetector:
    """
    MTCNN-based face detector that crops the largest face from a frame
    before it is passed through the augmentation pipeline.

    Falls back to returning the full image if no face is detected.

    Requires: pip install facenet-pytorch
    """

    def __init__(self, margin: int = 30, min_face_size: int = 40, device: str = "cpu"):
        try:
            from facenet_pytorch import MTCNN
        except ImportError as e:
            raise ImportError(
                "facenet-pytorch is required for face detection. "
                "Install it with: pip install facenet-pytorch"
            ) from e

        self.mtcnn = MTCNN(
            keep_all=False,
            select_largest=True,
            min_face_size=min_face_size,
            post_process=False,
            device=device,
        )
        self.margin = margin

    def crop(self, img: np.ndarray) -> np.ndarray:
        """
        Detect the largest face and return the cropped region (RGB numpy).
        Returns the original image if no face is detected.
        """
        from PIL import Image

        boxes, _ = self.mtcnn.detect(Image.fromarray(img))
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
            A.ImageCompression(quality_lower=50, quality_upper=90, p=0.5),
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

class FF_Dataset(Dataset):
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
# Identity-based split helpers  (extracted to keep build_splits simple)
# ---------------------------------------------------------------------------

def _parse_csv_to_records(csv_path: str, frames_dir: Path) -> List[Record]:
    """Read CSV and resolve each row to (dir_path, label, identity_ids)."""
    df = pd.read_csv(csv_path, index_col=0)
    records: List[Record] = []
    missing = 0

    for _, row in df.iterrows():
        label = LABEL_MAP.get(str(row["Label"]).upper())
        if label is None:
            continue

        dir_path = frames_dir / path_to_dir_name(str(row["File Path"]))
        if not dir_path.exists() or not _get_frames(dir_path):
            missing += 1
            continue

        records.append((dir_path, label, extract_identities(str(row["File Path"]))))

    if missing:
        warnings.warn(
            f"{missing} CSV rows skipped — directory not found or empty under '{frames_dir}'. "
            "Check that path_to_dir_name() matches your folder naming."
        )
    return records


def _partition_ids(
    all_ids: Set[str],
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Randomly partition identity IDs into (train_ids, val_ids, test_ids)."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(sorted(all_ids))

    n = len(shuffled)
    n_test = max(1, int(n * test_split))
    n_val  = max(1, int(n * val_split))

    test_ids  = set(shuffled[:n_test])
    val_ids   = set(shuffled[n_test : n_test + n_val])
    train_ids = set(shuffled[n_test + n_val :])
    return train_ids, val_ids, test_ids


def _assign_to_splits(
    records: List[Record],
    train_ids: Set[str],
    val_ids: Set[str],
    test_ids: Set[str],
) -> Tuple[List[Sample], List[Sample], List[Sample], int]:
    """Assign each video to exactly one split; skip if IDs span multiple splits."""
    train: List[Sample] = []
    val:   List[Sample] = []
    test:  List[Sample] = []
    skipped = 0

    for dir_path, label, ids in records:
        if not ids:
            skipped += 1
            continue
        id_set = set(ids)
        if id_set <= train_ids:
            train.append((dir_path, label))
        elif id_set <= val_ids:
            val.append((dir_path, label))
        elif id_set <= test_ids:
            test.append((dir_path, label))
        else:
            skipped += 1   # IDs span multiple splits → skip

    return train, val, test, skipped


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
) -> Tuple[FF_Dataset, FF_Dataset, FF_Dataset]:
    """
    Build train / val / test datasets from labels.csv using identity-based splitting.

    Why identity-based?
    -------------------
    In FF++, fake videos are derived from real (original) videos.
    For example, Face2Face/944_032.mp4 uses faces from original/944.mp4
    and original/032.mp4. If identity 944 were in training while
    Face2Face/944_032.mp4 were in validation, the model could exploit
    memorised texture/identity cues — a data leak.

    Strategy:
    ---------
    1. Collect all unique numeric IDs across every video path.
    2. Randomly partition IDs → train_ids / val_ids / test_ids.
    3. Assign a video to a split only if ALL its IDs belong to that split.
    4. Skip videos whose IDs span multiple splits (prevents leakage).

    Args:
        csv_path:       Path to labels.csv.
        frames_dir:     Root directory containing per-video frame directories.
        val_split:      Fraction of identities for validation.
        test_split:     Fraction of identities for testing.
        image_size:     Resize target for transforms.
        seed:           Random seed for reproducible splits.
        face_detector:  Optional FaceDetector passed to each FF_Dataset.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    frames_root = Path(frames_dir)

    records = _parse_csv_to_records(csv_path, frames_root)

    all_ids: Set[str] = {i for _, _, ids in records for i in ids}
    train_ids, val_ids, test_ids = _partition_ids(all_ids, val_split, test_split, seed)

    train_samples, val_samples, test_samples, skipped = _assign_to_splits(
        records, train_ids, val_ids, test_ids
    )

    if skipped:
        warnings.warn(
            f"{skipped} videos skipped (identity spans multiple splits or unknown identity)."
        )

    print("Dataset split summary:")
    _print_split_summary(train_samples, val_samples, test_samples)

    # Sanity check: no identity overlap between train and eval sets
    eval_ids = val_ids | test_ids
    assert not (train_ids & eval_ids), "BUG: identity leak between train and eval sets!"

    return (
        FF_Dataset(train_samples, train=True,  image_size=image_size, face_detector=face_detector),
        FF_Dataset(val_samples,   train=False, image_size=image_size, face_detector=face_detector),
        FF_Dataset(test_samples,  train=False, image_size=image_size, face_detector=face_detector),
    )
