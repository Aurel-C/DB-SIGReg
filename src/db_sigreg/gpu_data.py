"""GPU-resident CIFAR loader with batched augmentation.

The whole uint8 dataset (150 MB for CIFAR) lives in VRAM, and the SSL
augmentations run as batched tensor ops, so the training loop never waits on
CPU workers or host-to-device copies. It mirrors data.build_transforms:
RandomResizedCrop(scale=(0.2, 1)), horizontal flip, ColorJitter(0.4, 0.4, 0.4,
0.1) with p=0.8, grayscale with p=0.2, then ImageNet normalization. The jitter
ops are applied in a fixed order (torchvision randomizes it), and the crop's
aspect ratio is clamped instead of resampled; both are standard GPU-pipeline
simplifications.

Batches have the same shape as the CPU DataLoader: (views, labels) with views
of shape (B, V, 3, S, S).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
_GRAY = (0.299, 0.587, 0.114)


def load_arrays(name: str, root: Path, train: bool, download: bool) -> tuple[Tensor, Tensor]:
    """uint8 images (N, H, W, 3) and int64 labels, from .npz or torchvision."""

    import numpy as np

    split = "train" if train else "test"
    npz = root / f"{name}-{split}.npz"
    if npz.exists():
        data = np.load(npz)
        return torch.from_numpy(data["images"]), torch.from_numpy(data["labels"]).long()
    from torchvision import datasets

    cls = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}[name]
    ds = cls(root, train=train, download=download)
    return torch.from_numpy(ds.data), torch.as_tensor(ds.targets, dtype=torch.long)


class GPUViewLoader:
    def __init__(
        self,
        images: Tensor,
        labels: Tensor,
        *,
        batch_size: int,
        views: int,
        image_size: int,
        train: bool,
        device: torch.device,
        seed: int = 0,
    ) -> None:
        self.images = images.to(device).permute(0, 3, 1, 2).contiguous()  # (N, 3, H, W) uint8
        self.labels = labels.to(device)
        self.batch_size = batch_size
        self.views = views
        self.image_size = image_size
        self.train = train
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        self.gray = torch.tensor(_GRAY, device=device).view(1, 3, 1, 1)

    def __len__(self) -> int:
        n = self.labels.numel()
        return n // self.batch_size if self.train else math.ceil(n / self.batch_size)

    def __iter__(self):
        n = self.labels.numel()
        if self.train:
            order = torch.randperm(n, device=self.device, generator=self.generator)
        else:
            order = torch.arange(n, device=self.device)
        for i in range(len(self)):
            idx = order[i * self.batch_size : (i + 1) * self.batch_size]
            x = self.images[idx].float().div_(255.0)
            if self.train:
                x = x.repeat_interleave(self.views, dim=0)
                x = self._augment(x)
                x = x.view(idx.numel(), self.views, 3, self.image_size, self.image_size)
            else:
                if x.shape[-1] != self.image_size:
                    x = F.interpolate(x, size=self.image_size, mode="bilinear", antialias=True, align_corners=False)
                x = ((x - self.mean) / self.std).unsqueeze(1)
            yield x, self.labels[idx]

    def _rand(self, n: int, low: float = 0.0, high: float = 1.0) -> Tensor:
        return torch.rand(n, device=self.device, generator=self.generator) * (high - low) + low

    def _augment(self, x: Tensor) -> Tensor:
        n = x.shape[0]
        # RandomResizedCrop(scale=(0.2, 1.0), ratio=(3/4, 4/3)) + horizontal flip, as one affine warp.
        area = self._rand(n, 0.2, 1.0)
        log_ratio = self._rand(n, math.log(3 / 4), math.log(4 / 3))
        ratio = log_ratio.exp()
        w = (area * ratio).sqrt().clamp(max=1.0)
        h = (area / ratio).sqrt().clamp(max=1.0)
        cx = (self._rand(n) * (1 - w) + w / 2) * 2 - 1
        cy = (self._rand(n) * (1 - h) + h / 2) * 2 - 1
        flip = torch.where(self._rand(n) < 0.5, -1.0, 1.0)
        theta = torch.zeros(n, 2, 3, device=self.device)
        theta[:, 0, 0] = w * flip
        theta[:, 0, 2] = cx
        theta[:, 1, 1] = h
        theta[:, 1, 2] = cy
        grid = F.affine_grid(theta, [n, 3, self.image_size, self.image_size], align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

        # ColorJitter(0.4, 0.4, 0.4, 0.1) with p=0.8.
        jitter = (self._rand(n) < 0.8).view(n, 1, 1, 1)
        y = x * self._rand(n, 0.6, 1.4).view(n, 1, 1, 1)
        gray = (y * self.gray).sum(1, keepdim=True)
        mean = gray.mean(dim=(2, 3), keepdim=True)
        y = (y - mean) * self._rand(n, 0.6, 1.4).view(n, 1, 1, 1) + mean
        gray = (y * self.gray).sum(1, keepdim=True)
        y = (y - gray) * self._rand(n, 0.6, 1.4).view(n, 1, 1, 1) + gray
        y = self._hue(y.clamp(0, 1), self._rand(n, -0.1, 0.1))
        x = torch.where(jitter, y.clamp(0, 1), x)

        # RandomGrayscale(p=0.2).
        gray = (x * self.gray).sum(1, keepdim=True).expand_as(x)
        x = torch.where((self._rand(n) < 0.2).view(n, 1, 1, 1), gray, x)
        return (x - self.mean) / self.std

    @staticmethod
    def _hue(x: Tensor, shift: Tensor) -> Tensor:
        """Rotate chroma in YIQ space by shift * 2*pi (close to torchvision's HSV hue shift)."""
        to_yiq = x.new_tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
        to_rgb = torch.linalg.inv(to_yiq)
        yiq = torch.einsum("ij,njhw->nihw", to_yiq, x)
        angle = shift * 2 * math.pi
        cos, sin = angle.cos().view(-1, 1, 1), angle.sin().view(-1, 1, 1)
        i, q = yiq[:, 1], yiq[:, 2]
        yiq = torch.stack([yiq[:, 0], cos * i - sin * q, sin * i + cos * q], dim=1)
        return torch.einsum("ij,njhw->nihw", to_rgb, yiq)


def build_gpu_loaders(
    name: str,
    root: Path,
    *,
    batch_size: int,
    eval_batch_size: int,
    views: int,
    image_size: int,
    limit_train: int | None,
    limit_eval: int | None,
    download: bool,
    device: torch.device,
    seed: int,
) -> tuple[GPUViewLoader, GPUViewLoader, int, int, int]:
    """Returns train loader, eval loader, num classes, train size, eval size."""

    train_x, train_y = load_arrays(name, Path(root), True, download)
    eval_x, eval_y = load_arrays(name, Path(root), False, download)
    if limit_train:
        train_x, train_y = train_x[:limit_train], train_y[:limit_train]
    if limit_eval:
        eval_x, eval_y = eval_x[:limit_eval], eval_y[:limit_eval]
    classes = 10 if name == "cifar10" else 100
    train = GPUViewLoader(
        train_x, train_y, batch_size=batch_size, views=views, image_size=image_size, train=True, device=device, seed=seed
    )
    evaluation = GPUViewLoader(
        eval_x, eval_y, batch_size=eval_batch_size, views=1, image_size=image_size, train=False, device=device
    )
    return train, evaluation, classes, len(train_y), len(eval_y)
