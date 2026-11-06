from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torchvision.models import resnet18


Norm = Literal["bn", "gn"]


def _norm1d(norm: Norm, dim: int) -> nn.Module:
    return nn.BatchNorm1d(dim) if norm == "bn" else nn.LayerNorm(dim)


def _norm2d(norm: Norm, channels: int) -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    return nn.GroupNorm(min(32, channels // 4), channels)


class ProjectionHead(nn.Module):
    """MLP projector. ``norm="gn"`` swaps BatchNorm for LayerNorm so that
    outputs do not depend on the micro-batch (needed for small-batch runs)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, norm: Norm = "bn") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            _norm1d(norm, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            _norm1d(norm, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SmallConvEncoder(nn.Module):
    def __init__(self, width: int = 64, emb_dim: int = 256, norm: Norm = "bn") -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, width, norm),
            self._block(width, width * 2, norm),
            self._block(width * 2, width * 4, norm),
            self._block(width * 4, width * 4, norm),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(width * 4, emb_dim)
        self.out_dim = emb_dim

    @staticmethod
    def _block(in_ch: int, out_ch: int, norm: Norm) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm2d(norm, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm2d(norm, out_ch),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.features(x).flatten(1)
        return self.fc(y)


class ResNet18Encoder(nn.Module):
    def __init__(self, emb_dim: int = 512, norm: Norm = "bn", cifar_stem: bool = False) -> None:
        super().__init__()
        norm_layer = None if norm == "bn" else (lambda c: _norm2d("gn", c))
        model = resnet18(weights=None, norm_layer=norm_layer)
        if cifar_stem:
            # 3x3 stride-1 stem without max-pool: the ImageNet stem shrinks 32 px inputs to 1x1 by the last stage.
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        model.fc = nn.Identity()
        self.backbone = model
        self.proj = nn.Linear(512, emb_dim) if emb_dim != 512 else nn.Identity()
        self.out_dim = emb_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.backbone(x))


class TinyViTEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        emb_dim: int = 256,
        depth: int = 4,
        heads: int = 4,
        mlp_dim: int = 512,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch = nn.Conv2d(3, emb_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(emb_dim)
        self.out_dim = emb_dim
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        y = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(y.shape[0], -1, -1)
        y = torch.cat([cls, y], dim=1) + self.pos
        y = self.blocks(y)
        return self.norm(y[:, 0])


class SSLModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        image_size: int,
        emb_dim: int,
        proj_dim: int,
        proj_hidden_dim: int,
        num_classes: int,
        norm: Norm = "bn",
    ) -> None:
        super().__init__()
        if backbone == "cnn":
            self.encoder = SmallConvEncoder(emb_dim=emb_dim, norm=norm)
        elif backbone in ("resnet18", "resnet18_cifar"):
            self.encoder = ResNet18Encoder(emb_dim=emb_dim, norm=norm, cifar_stem=backbone == "resnet18_cifar")
        elif backbone == "tiny_vit":
            self.encoder = TinyViTEncoder(image_size=image_size, emb_dim=emb_dim)
        else:
            raise ValueError(f"unknown backbone: {backbone}")
        self.projector = ProjectionHead(self.encoder.out_dim, proj_hidden_dim, proj_dim, norm=norm)
        self.probe = nn.Sequential(nn.LayerNorm(self.encoder.out_dim), nn.Linear(self.encoder.out_dim, num_classes))

    def forward(self, views: Tensor) -> tuple[Tensor, Tensor]:
        batch, num_views = views.shape[:2]
        x = views.flatten(0, 1)
        emb = self.encoder(x)
        proj = self.projector(emb)
        return emb, proj.reshape(batch, num_views, -1)
