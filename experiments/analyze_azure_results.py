"""
Azure Live Results — Table & Analysis
-------------------------------------
Reads the per-query CSV produced by run_azure_validation.py and prints a
results table in the project's familiar style, broken down by query bucket
(short / medium / long) plus an overall row. Real Azure data only — no AWS.

Usage:
  python -m experiments.analyze_azure_results                 # latest CSV
  python -m experiments.analyze_azure_results <path-to.csv>
"""

from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

RESULTS_DIR = Path(__file__).parent / "results"
BUCKETS = ("short", "medium", "long")


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap_ci(data, n_boot=10_000, seed=0):
    arr = np.asarray([x for x in data if np.isfinite(x)], float)
    if arr.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(arr, arr.size, replace=True).mean() for _ in range(n_boot)])
    return float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def latest_csv() -> Path:
    files = sorted(RESULTS_DIR.glob("azure_validation_*.csv"))
    if not files:
        raise SystemExit(f"No azure_validation_*.csv in {RESULTS_DIR}")
    return files[-1]


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _cost(row: dict) -> float:
    """Per-query measured USD = measured_cost_per_1k * total_tokens / 1000."""
    try:
        return float(row["measured_cost_per_1k"]) * int(row["total_tokens"]) / 1000.0
    except (ValueError, KeyError):
        return float("nan")


def summarize(rows: list[dict], which: list[str]) -> dict:
    sub = [r for r in rows if r["bucket"] in which]
    ok = [r for r in sub if r["success"] == "True"]
    lat = [float(r["latency_ms"]) for r in ok]
    cost = [c for c in (_cost(r) for r in ok) if np.isfinite(c)]
    cpt = [float(r["measured_cost_per_1k"]) for r in ok if r["measured_cost_per_1k"]]
    return {
        "n": len(sub),
        "errors": len(sub) - len(ok),
        "total_cost": float(np.sum(cost)) if cost else 0.0,
        "cost_per_1k": float(np.mean(cpt)) if cpt else float("nan"),
        "mean_ms": float(np.mean(lat)) if lat else float("nan"),
        "p95_ms": float(np.percentile(lat, 95)) if lat else float("nan"),
        "overhead_ms": float(np.mean([float(r["overhead_ms"]) for r in sub])) if sub else float("nan"),
        "lat": lat,
    }


def main(path: Path) -> None:
    rows = load_rows(path)
    n = len(rows)

    print("\n" + "=" * 72)
    print("  Multi-Cloud LLM Router — Azure Live Results [REAL]")
    print(f"  Endpoint: azure:tinyllama (AKS CPU)   Queries: {n}   Source: {path.name}")
    print("=" * 72)
    print(f"\n{'Bucket':<10}{'Queries':>9}{'Total cost ($)':>16}{'Mean (ms)':>12}{'P95 (ms)':>11}{'Errors':>8}")
    print("-" * 66)

    for b in BUCKETS:
        s = summarize(rows, [b])
        if s["n"] == 0:
            print(f"{b:<10}{0:>9}{'—':>16}{'—':>12}{'—':>11}{0:>8}")
            continue
        print(f"{b:<10}{s['n']:>9}{s['total_cost']:>16.6f}{s['mean_ms']:>12.0f}{s['p95_ms']:>11.0f}{s['errors']:>8}")

    alls = summarize(rows, list(BUCKETS))
    print("-" * 66)
    print(f"{'ALL':<10}{alls['n']:>9}{alls['total_cost']:>16.6f}{alls['mean_ms']:>12.0f}{alls['p95_ms']:>11.0f}{alls['errors']:>8}")

    # ---- Analysis ----
    c_pt, c_lo, c_hi = bootstrap_ci([float(r["measured_cost_per_1k"]) for r in rows
                                     if r["success"] == "True" and r["measured_cost_per_1k"]])
    l_pt, l_lo, l_hi = bootstrap_ci(alls["lat"])
    succ = sum(1 for r in rows if r["success"] == "True")
    w_lo, w_hi = wilson_ci(succ, n)

    print("\nAnalysis")
    print("-" * 66)
    print(f"  H1  cost/1k tokens (measured) : ${c_pt:.6f}  95% CI [${c_lo:.6f}, ${c_hi:.6f}]")
    print(f"  H2  end-to-end latency (mean) : {l_pt:.0f}ms  95% CI [{l_lo:.0f}, {l_hi:.0f}]   P95={alls['p95_ms']:.0f}ms")
    print(f"  H2  router overhead           : {alls['overhead_ms']:.3f}ms  ({'PASSED' if alls['overhead_ms'] < 5 else 'NOT MET'} <5ms)")
    print(f"  H3  completion (this CSV)     : {succ}/{n} = {succ/n*100:.2f}%  Wilson 95% CI [{w_lo*100:.2f}%, {w_hi*100:.2f}%]")
    print(f"  H4  methods                   : bootstrap 95% CIs, Wilson interval (see run_azure_validation for permutation test)")
    print("\n  NOTE: single live cloud — no AWS/cross-cloud cost reduction claimed here.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_csv()
    main(p)
