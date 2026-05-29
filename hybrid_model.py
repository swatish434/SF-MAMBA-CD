import os
import sys
import torch
import torch.nn as nn
from torch.fft import fft2, ifft2
import torch.utils.checkpoint as cp
import torch.nn.functional as F

# ---------------------------------------------------------
# VMamba import (local repo)
# ---------------------------------------------------------
ROOT = os.path.dirname(__file__)
VMAMBA_ROOT = os.path.join(ROOT, "VMamba")

# Prefer importing directly from your local VMamba/vmamba.py (this is what worked)
# Ensure ROOT is on path so `VMamba` is importable as a package/module.
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if VMAMBA_ROOT not in sys.path:
    sys.path.insert(0, VMAMBA_ROOT)

from VMamba.vmamba import vanilla_vmamba_small


# =========================================================
# VMamba Encoder (semantic stream)
# =========================================================
class VMambaEncoder(nn.Module):
    """
    Produces a feature map (NOT classification logits).

    In your repo, vanilla_vmamba_small() returns logits by default (e.g., [B, 1000]).
    The correct way (verified by your test) is to set:
        m.classifier = nn.Identity()
    which makes forward return a feature map like:
        [B, H, W, C] (NHWC), e.g. [1, 8, 8, 768] for img_size=256.

    This encoder also converts NHWC -> NCHW so downstream Conv2d works.
    """
    def __init__(self, config):
        super().__init__()
        self.backbone = vanilla_vmamba_small(
            img_size=config.get("img_size", 256),
            in_chans=config["in_channels"],
        )

        # Force feature-map output (remove classification head)
        if hasattr(self.backbone, "classifier"):
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, "head"):
            self.backbone.head = nn.Identity()
        elif hasattr(self.backbone, "fc"):
            self.backbone.fc = nn.Identity()

        self.use_checkpoint = bool(config.get("checkpoint_vmamba", False))

    def forward(self, x):
        if self.use_checkpoint and self.training:
            feat = cp.checkpoint(self.backbone, x, use_reentrant=False)
        else:
            feat = self.backbone(x)

        # VMamba feature map is NHWC -> convert to NCHW
        # Expected: [B, H, W, C]
        if feat.dim() == 4:
            # if last dim looks like channels, do NHWC->NCHW
            feat = feat.permute(0, 3, 1, 2).contiguous()

        return feat


# =========================================================
# ChangeFFT Module (FFT stream)
# =========================================================
class ChangeFFTModule(nn.Module):
    """
    FFT applied safely in input spatial domain.

    Rules for AMP:
    - FFT does NOT support bf16 reliably.
    - torch.complex() does NOT accept bf16 inputs.
    So we do: FFT + complex packing + iFFT in float32, then cast back.
    """
    def __init__(self, config):
        super().__init__()
        in_ch = config.get("in_channels", 3)
        fft_ch = int(config.get("fft_channels", 48))

        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, fft_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fft_ch),
            nn.ReLU(inplace=True)
        )

        # Operate on concatenated [real, imag] => 2*fft_ch
        self.fd_conv = nn.Sequential(
            nn.Conv2d(2 * fft_ch, 2 * fft_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(2 * fft_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * fft_ch, 2 * fft_ch, kernel_size=1, bias=True),
        )

        self.fft_ch = fft_ch

    def forward(self, x1, x2):
        f1 = self.proj(x1)
        f2 = self.proj(x2)
        diff = f2 - f1  # (B, fft_ch, H, W)

        orig_dtype = diff.dtype  # e.g., bf16 under AMP

        # Force FFT + complex + iFFT to FP32
        with torch.autocast(device_type="cuda", enabled=False):
            diff_f32 = diff.float()
            freq = fft2(diff_f32, dim=(-2, -1))  # complex64

            freq_cat = torch.cat([freq.real, freq.imag], dim=1)  # float32
            freq_out = self.fd_conv(freq_cat)                    # float32

            real = freq_out[:, :self.fft_ch].float()
            imag = freq_out[:, self.fft_ch:].float()

            freq_complex = torch.complex(real, imag)             # complex64
            change = ifft2(freq_complex, dim=(-2, -1)).real      # float32

        return change.to(orig_dtype)


# =========================================================
# Fusion Adapter: VMamba features -> match FFT resolution/channels
# =========================================================
class SemanticAdapter(nn.Module):
    def __init__(self, out_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LazyConv2d(out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat, target_hw):
        H, W = target_hw
        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
        return self.proj(feat)


# =========================================================
# Decoder (channel-safe via LazyConv2d)
# =========================================================
class Decoder(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.head(x)


# =========================================================
# Full Model
# =========================================================
class VMambaChangeFFTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = VMambaEncoder(config)
        self.change_fft = ChangeFFTModule(config)

        fft_ch = int(config.get("fft_channels", 48))
        self.semantic_adapter = SemanticAdapter(out_ch=fft_ch)

        self.fuse = nn.Sequential(
            nn.Conv2d(2 * fft_ch, fft_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(fft_ch),
            nn.ReLU(inplace=True)
        )

        self.decoder_change = Decoder(out_channels=1)
        self.decoder_boundary = Decoder(out_channels=1)

    def forward(self, x1, x2):
        # FFT stream (B, fft_ch, H, W)
        fft_feat = self.change_fft(x1, x2)
        H, W = fft_feat.shape[-2:]

        # Semantic stream (feature maps)
        f1 = self.encoder(x1)  # (B, C, h, w)
        f2 = self.encoder(x2)  # (B, C, h, w)

        sem_diff = (f2 - f1).abs()  # (B, C, h, w)

        # Match to FFT spatial size + compress channels to fft_ch
        sem_feat = self.semantic_adapter(sem_diff, target_hw=(H, W))  # (B, fft_ch, H, W)

        # Fuse
        fused = self.fuse(torch.cat([fft_feat, sem_feat], dim=1))     # (B, fft_ch, H, W)

        # Decode
        change_map = self.decoder_change(fused)       # (B, 1, H, W)
        boundary_map = self.decoder_boundary(fused)   # (B, 1, H, W)

        return change_map, boundary_map

    def forward_with_features(self, x1, x2):
        """Forward pass that also returns intermediate feature maps for visualization."""
        # FFT stream
        fft_feat = self.change_fft(x1, x2)
        H, W = fft_feat.shape[-2:]

        # Semantic stream
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        sem_diff = (f2 - f1).abs()
        sem_feat = self.semantic_adapter(sem_diff, target_hw=(H, W))

        # Fuse
        fused = self.fuse(torch.cat([fft_feat, sem_feat], dim=1))

        # Decode
        change_map = self.decoder_change(fused)
        boundary_map = self.decoder_boundary(fused)

        features = {
            'fft_feat': fft_feat.detach().cpu(),
            'semantic_diff': sem_diff.detach().cpu(),
            'semantic_adapted': sem_feat.detach().cpu(),
            'fused': fused.detach().cpu(),
        }

        return change_map, boundary_map, features


# =========================================================
# Factory helpers
# =========================================================
def create_model(config):
    config = dict(config)
    config.setdefault("img_size", 256)
    config.setdefault("fft_channels", 48)
    config.setdefault("checkpoint_vmamba", False)
    model = VMambaChangeFFTModel(config)
    return model, None


def model_summary(model):
    print("VMambaChangeFFTModel")
    print("• VMamba semantic stream + ChangeFFT frequency stream")
    print("• VMamba classifier removed -> returns feature maps")
    print("• VMamba NHWC -> NCHW conversion applied")
    print("• FFT/complex/iFFT forced to FP32 for bf16 AMP compatibility")
    print("• Semantic diff fused with FFT features")
    print("• Channel-safe decoders via LazyConv2d")
