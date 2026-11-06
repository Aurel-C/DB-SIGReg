"""Print results/runs.jsonl as Markdown tables, one per suite.

    uv run python experiments/summarize.py            # all suites
    uv run python experiments/summarize.py fidelity   # one suite

Seeds are averaged (mean ± std when there is more than one).
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results" / "runs.jsonl"

COLUMNS = {
    "fidelity": ["fidelity_cos_db_mean", "fidelity_cos_db_fresh_mean", "fidelity_cos_minibatch_mean",
                 "fidelity_relerr_db_mean", "fidelity_relerr_minibatch_mean", "final_probe_acc"],
}
DEFAULT = ["final_probe_acc", "best_probe_acc", "final_ecf_error", "final_axis_std", "samples_per_sec", "peak_memory_mb"]


def fmt(values: list[float]) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return ""
    mean = statistics.fmean(values)
    if len(values) == 1:
        return f"{mean:.4g}"
    return f"{mean:.4g} ± {statistics.stdev(values):.2g}"


def main() -> None:
    wanted = set(sys.argv[1:])
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for line in RESULTS.read_text().splitlines():
        if not line.strip():
            continue
        run = json.loads(line)
        tag = run.get("tag") or "untagged"
        if wanted and tag not in wanted:
            continue
        variant = re.sub(r"-s\d+(?=-|$)", "", run["config"]["name"] or run["run"])
        groups[tag][variant].append(run)

    for tag, variants in groups.items():
        cols = COLUMNS.get(tag, DEFAULT)
        print(f"\n### {tag}\n")
        print("| variant | n | " + " | ".join(cols) + " |")
        print("| --- | --- | " + " | ".join("---" for _ in cols) + " |")
        for variant, runs in variants.items():
            cells = [fmt([r.get(c) for r in runs]) for c in cols]
            print(f"| {variant} | {len(runs)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
