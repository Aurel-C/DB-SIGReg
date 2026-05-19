from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


class MultiViewDataset(Dataset):
    def __init__(self, base: Dataset, train_transform: Callable, eval_transform: Callable, views: int) -> None:
        self.base = base
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.views = views

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image, label = self.base[index]
        transform = self.train_transform if self.views > 1 else self.eval_transform
        return torch.stack([transform(image) for _ in range(self.views)]), int(label)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def _limit(ds: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit <= 0 or limit >= len(ds):
        return ds
    return Subset(ds, range(limit))


def build_datasets(
    name: str,
    root: str | Path,
    image_size: int,
    views: int,
    limit_train: int | None = None,
    limit_eval: int | None = None,
    download: bool = True,
) -> tuple[Dataset, Dataset, int]:
    root = Path(root)
    train_transform, eval_transform = build_transforms(image_size)
    if name == "cifar10":
        train_base = datasets.CIFAR10(root, train=True, download=download)
        eval_base = datasets.CIFAR10(root, train=False, download=download)
        classes = 10
    elif name == "cifar100":
        train_base = datasets.CIFAR100(root, train=True, download=download)
        eval_base = datasets.CIFAR100(root, train=False, download=download)
        classes = 100
    elif name == "stl10":
        train_base = datasets.STL10(root, split="train", download=download)
        eval_base = datasets.STL10(root, split="test", download=download)
        classes = 10
    elif name == "fake":
        train_base = datasets.FakeData(size=limit_train or 512, image_size=(3, image_size, image_size), num_classes=10)
        eval_base = datasets.FakeData(size=limit_eval or 128, image_size=(3, image_size, image_size), num_classes=10)
        classes = 10
    else:
        raise ValueError(f"unknown dataset: {name}")

    train = MultiViewDataset(_limit(train_base, limit_train), train_transform, eval_transform, views=views)
    eval_ds = MultiViewDataset(_limit(eval_base, limit_eval), train_transform, eval_transform, views=1)
    return train, eval_ds, classes
