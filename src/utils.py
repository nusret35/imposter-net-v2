import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Tracks a running mean of a scalar value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """
    Args:
        labels: ground-truth binary labels (0=real, 1=fake)
        probs:  predicted probabilities of being fake
    Returns:
        dict with auc, accuracy, f1
    """
    preds = (probs >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(labels, probs),
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
    }
