import torch
import torch.nn as nn
import timm


class FrequencyBranch(nn.Module):
    """
    Extracts frequency-domain features via 2D FFT.
    Input:  (B, C, H, W) RGB image
    Output: (B, out_dim) feature vector
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to grayscale: (B, 1, H, W)
        gray = x.mean(dim=1, keepdim=True)
        # FFT magnitude spectrum (log-compressed): (B, 1, H, W)
        fft_mag = torch.fft.fft2(gray).abs().log1p()
        # Shift zero-frequency to center
        fft_mag = torch.fft.fftshift(fft_mag, dim=(-2, -1))
        feat = self.conv(fft_mag)           # (B, 128, 1, 1)
        feat = feat.flatten(1)              # (B, 128)
        return self.proj(feat)              # (B, out_dim)


class SwinDeepfakeDetector(nn.Module):
    """
    Deepfake detection model with:
      - Swin-Tiny backbone (4-stage multi-scale features)
      - Frequency branch (FFT magnitude)
      - Multi-scale feature projectors
      - Binary classifier head
    """

    def __init__(
        self,
        backbone: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        # --- Backbone ---
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        stage_channels = self.backbone.feature_info.channels()
        # swin_tiny: [96, 192, 384, 768]

        # --- Per-stage projectors ---
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c, hidden_dim),
                nn.GELU(),
            )
            for c in stage_channels
        ])

        # --- Frequency branch ---
        self.freq_branch = FrequencyBranch(out_dim=hidden_dim)

        # --- Classifier ---
        fused_dim = hidden_dim * (len(stage_channels) + 1)  # 256 * 5 = 1280
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),  # raw logit; apply sigmoid outside
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224)
        Returns:
            logits: (B,) — use BCEWithLogitsLoss during training
        """
        stage_feats = self.backbone(x)  # list of 4 feature maps: (B, H, W, C)
        # timm's Swin Transformer outputs NHWC; permute to NCHW for AdaptiveAvgPool2d
        stage_feats = [f.permute(0, 3, 1, 2).contiguous() for f in stage_feats]
        proj_feats = [proj(f) for proj, f in zip(self.projectors, stage_feats)]

        freq_feat = self.freq_branch(x)

        fused = torch.cat(proj_feats + [freq_feat], dim=1)  # (B, 1280)
        logits = self.classifier(fused)                      # (B, 1)
        return logits.squeeze(1)                             # (B,)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
