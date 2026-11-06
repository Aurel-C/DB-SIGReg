"""Experiment suites for the DB-SIGReg paper.

Usage (from the repo root):

    uv run python experiments/suites.py --list
    uv run python experiments/suites.py fidelity --dry-run
    uv run python experiments/suites.py fidelity premise --epochs 20
    uv run python experiments/suites.py main -- --backbone resnet18   # extra flags go to every run

Each run appends a summary line to results/runs.jsonl (tracked in git).
Runs whose name already appears there are skipped, so a suite can be
interrupted and restarted. Summarize with experiments/summarize.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "runs.jsonl"

# Shared defaults: native 32 px resolution, small CNN, 5 knots. The dataset is
# chosen per suite (DEFAULT_DATASET) or with --dataset, and is part of run names.
# Training recipe from the 2026-11-06 recipe search (CIFAR-100, 20 epochs):
# warmup+cosine lr, 2 global + 6 local crops, projector output 16 and 1024
# axes took the probe from 16.9% to 35.4% (random-init encoder: 14.4%).
RECIPE = "r2"  # part of run names, so results from older recipes are never skipped as done
BASE = [
    "--lr-schedule", "cosine",
    "--views", "2",
    "--local-views", "6",
    "--global-scale", "0.3", "1.0",
    "--proj-dim", "16",
    "--axes", "1024",
    "--backbone", "cnn",
    "--image-size", "32",
    "--knots", "5",
]


def premise(seeds: list[int]) -> list[tuple[str, list[str]]]:
    """Does plain SIGReg degrade as the batch shrinks? (no accumulation)"""
    runs = []
    for seed in seeds:
        for bs in (32, 64, 128, 256, 512):
            runs.append((f"premise-sigreg-bs{bs}-s{seed}", ["--loss", "sigreg", "--batch-size", str(bs), "--seed", str(seed)]))
    return runs


def premise_steps(seeds: list[int]) -> list[tuple[str, list[str]]]:
    """Premise with a fixed optimizer-step budget (7800 steps, bs128's 20 epochs) to remove the step-count confound."""
    runs = []
    for seed in seeds:
        for bs in (32, 64, 128, 256, 512):
            runs.append(
                (
                    f"premise-steps-sigreg-bs{bs}-s{seed}",
                    ["--loss", "sigreg", "--batch-size", str(bs), "--total-steps", "7800", "--seed", str(seed)],
                )
            )
    return runs


def fidelity(seeds: list[int]) -> list[tuple[str, list[str]]]:
    """Cosine similarity of DB / DB-fresh / mini-batch gradients to the exact window gradient."""
    runs = []
    for seed in seeds:
        configs = [(bs, k, "bn") for bs in (16, 64) for k in (1, 4, 16)]
        configs += [(4, k, "gn") for k in (16, 64)]  # small micro-batches, batch-independent norms
        for bs, k, norm in configs:
            suffix = "" if norm == "bn" else f"-{norm}"
            runs.append(
                (
                    f"fidelity-db-bs{bs}-k{k}{suffix}-s{seed}",
                    ["--loss", "dbsigreg", "--batch-size", str(bs), "--accum-steps", str(k), "--norm", norm,
                     "--fidelity-every", "20", "--seed", str(seed)],
                )
            )
    return runs


def smallbatch(seeds: list[int]) -> list[tuple[str, list[str]]]:
    """Tiny micro-batches at a fixed 256 images per optimizer step, batch-independent norms."""
    runs = []
    for seed in seeds:
        for bs in (2, 4, 8, 32):
            k = 256 // bs
            for loss in ("sigreg", "dbsigreg"):
                extra = ["--sigreg-refresh", str(k)] if loss == "sigreg" else []
                runs.append(
                    (
                        f"smallbatch-{loss}-bs{bs}-k{k}-s{seed}",
                        ["--loss", loss, "--norm", "gn", "--batch-size", str(bs), "--accum-steps", str(k),
                         "--seed", str(seed), *extra],
                    )
                )
    return runs


def main_comparison(seeds: list[int]) -> list[tuple[str, list[str]]]:
    """Equal memory (micro-batch 64, 4 accumulation steps) vs the true batch-256 ceiling."""
    runs = []
    for seed in seeds:
        common = ["--seed", str(seed)]
        runs += [
            (f"main-sigreg-accum-s{seed}", ["--loss", "sigreg", "--batch-size", "64", "--accum-steps", "4", *common]),
            (f"main-db-detached-s{seed}", ["--loss", "dbsigreg", "--batch-size", "64", "--accum-steps", "4", *common]),
            (
                f"main-db-include-current-s{seed}",
                ["--loss", "dbsigreg", "--db-stat-mode", "include_current", "--batch-size", "64", "--accum-steps", "4", *common],
            ),
            (f"main-sigreg-bs256-s{seed}", ["--loss", "sigreg", "--batch-size", "256", *common]),
        ]
    return runs


def nullspace(seeds: list[int]) -> list[tuple[str, list[str]]]:
    """Fixed vs rotating axes, with 4 and 64 axes on the 16-dim projector output.

    A null space only exists when there are fewer axes than dimensions.
    """
    runs = []
    for seed in seeds:
        for axes in (4, 64):
            common = ["--axes", str(axes), "--batch-size", "64", "--accum-steps", "4", "--seed", str(seed)]
            runs += [
                (f"nullspace-sigreg-rotate-a{axes}-s{seed}", ["--loss", "sigreg", "--sigreg-refresh", "4", *common]),
                (f"nullspace-sigreg-fixed-a{axes}-s{seed}", ["--loss", "sigreg", "--sigreg-refresh", "0", *common]),
            ]
            for m in (1, 4, 16):
                runs.append(
                    (f"nullspace-db-swap{4 * m}-a{axes}-s{seed}", ["--loss", "dbsigreg", "--swap-steps", str(4 * m), *common])
                )
    return runs


SUITES = {
    "premise": premise,
    "premise_steps": premise_steps,
    "fidelity": fidelity,
    "smallbatch": smallbatch,
    "main": main_comparison,
    "nullspace": nullspace,
}
DEFAULT_SEEDS = {"premise": [0], "premise_steps": [0], "fidelity": [0], "smallbatch": [0], "main": [0, 1, 2], "nullspace": [0]}
# Accuracy suites use CIFAR-100 (less saturated probe). Fidelity measures gradient
# agreement, which does not depend on the label set, so CIFAR-10 is enough there.
DEFAULT_DATASET = {"premise": "cifar100", "premise_steps": "cifar100", "fidelity": "cifar10", "smallbatch": "cifar100", "main": "cifar100", "nullspace": "cifar100"}
SHORT = {"cifar10": "c10", "cifar100": "c100", "stl10": "stl10", "fake": "fake"}
DEFAULT_EPOCHS = {"premise": 20, "premise_steps": 1, "fidelity": 5, "smallbatch": 20, "main": 20, "nullspace": 20}


def done_runs() -> set[str]:
    if not RESULTS.exists():
        return set()
    names = set()
    for line in RESULTS.read_text().splitlines():
        if line.strip():
            names.add(json.loads(line)["config"]["name"])
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suites", nargs="*", help=f"One or more of: {', '.join(SUITES)}")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="Override the suite's default epoch count.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--dataset", choices=list(SHORT), default=None, help="Override the suite's default dataset.")
    parser.add_argument("--num-workers", type=int, default=4)
    args, extra = parser.parse_known_args()  # unknown flags are passed to every run

    if args.list or not args.suites:
        for name, fn in SUITES.items():
            print(f"{name:11s} {len(fn(DEFAULT_SEEDS[name])):3d} runs, {DEFAULT_EPOCHS[name]} epochs  {fn.__doc__.strip().splitlines()[0]}")
        return

    unknown = [s for s in args.suites if s not in SUITES]
    if unknown:
        parser.error(f"unknown suite(s): {', '.join(unknown)}; choose from {', '.join(SUITES)}")
    extra = [a for a in extra if a != "--"]
    finished = done_runs()
    for suite in args.suites:
        seeds = args.seeds or DEFAULT_SEEDS[suite]
        epochs = args.epochs or DEFAULT_EPOCHS[suite]
        dataset = args.dataset or DEFAULT_DATASET[suite]
        for name, run_args in SUITES[suite](seeds):
            name = f"{name}-{SHORT[dataset]}-{RECIPE}"
            if name in finished:
                print(f"skip {name} (already in {RESULTS.relative_to(ROOT)})")
                continue
            cmd = [
                sys.executable, "-m", "db_sigreg.train", "--dataset", dataset, *BASE, *run_args,
                "--epochs", str(epochs), "--name", name, "--tag", suite,
                "--num-workers", str(args.num_workers), *extra,
            ]
            print(" ".join(cmd[1:]), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
