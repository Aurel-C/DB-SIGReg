"""Download CIFAR-10/100 from the Hugging Face Hub and save them as .npz files
that db_sigreg.data loads directly (faster than the Toronto mirror).

    uv run --with pyarrow python experiments/fetch_cifar_hf.py cifar100 cifar10

Writes data/<name>-train.npz and data/<name>-test.npz with uint8 images
(N, 32, 32, 3) and int64 labels (fine labels for CIFAR-100).
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

REPOS = {"cifar10": ("uoft-cs/cifar10", "label"), "cifar100": ("uoft-cs/cifar100", "fine_label")}
DATA = Path(__file__).resolve().parents[1] / "data"


def list_parquet(repo: str) -> list[str]:
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true"
    with urllib.request.urlopen(url) as resp:
        return [f["path"] for f in json.load(resp) if f["path"].endswith(".parquet")]


def fetch(name: str) -> None:
    repo, label_col = REPOS[name]
    files = list_parquet(repo)
    for split in ("train", "test"):
        out = DATA / f"{name}-{split}.npz"
        if out.exists():
            print(f"{out} exists, skipping")
            continue
        parts = sorted(f for f in files if Path(f).name.startswith(split))
        if not parts:
            raise SystemExit(f"no {split} parquet in {repo}: {files}")
        images, labels = [], []
        for path in parts:
            url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
            print(f"downloading {url}")
            with urllib.request.urlopen(url) as resp:
                table = pq.read_table(io.BytesIO(resp.read()))
            img_col = next(c for c in table.column_names if c in ("img", "image"))
            for img in table.column(img_col).to_pylist():
                images.append(np.asarray(Image.open(io.BytesIO(img["bytes"])).convert("RGB"), dtype=np.uint8))
            labels.extend(table.column(label_col).to_pylist())
        DATA.mkdir(parents=True, exist_ok=True)
        np.savez(out, images=np.stack(images), labels=np.asarray(labels, dtype=np.int64))
        print(f"wrote {out}: {len(labels)} images, {len(set(labels))} classes")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["cifar100", "cifar10"]:
        fetch(name)
