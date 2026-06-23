"""
Azure-Only Real Validation Harness
----------------------------------
Exercises the LIVE Azure AKS endpoint (Ollama / tinyllama) end-to-end through
the real PredictiveRouter and reports the project hypotheses with proper
statistical rigour (H4 = statistical significance, per supervisor feedback).

This is the honest single-cloud validation: only the Azure endpoint is
deployed, so every number below is a REAL measurement against live inference.
The cross-cloud cost comparison in H1 (Azure vs AWS) is reported as a measured
Azure cost basis with a confidence interval; the AWS arm is pending a second
live cloud and is explicitly NOT simulated here.

Hypotheses (from docs/Model_Paper_Metrics_Mapping):
  H1  Cost per 1k tokens + cost variance   -> measured throughput cost, bootstrap CI
  H2  P95 latency + router overhead (<5ms)  -> measured latency, overhead vs SLO
  H3  Request completion rate (>=99.5%)     -> measured under concurrent load, Wilson CI
  H4  Statistical significance              -> bootstrap CIs, Wilson CI, permutation test

Usage:
  python -m experiments.run_azure_validation                 # defaults
  python -m experiments.run_azure_validation --n 80 --max-tokens 128 --concurrency 8
"""

from __future__ import annotations
import argparse
import csv
import logging
import sys

# Windows consoles default to cp1252, which can't encode the report's α/≥/•/×
# characters and crashes the final print after a long run. Force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from router.query_analyzer import QueryAnalyzer, QueryFeatures, LengthBucket
from router.predictive_router import PredictiveRouter
from router.telemetry import TelemetryStore, EndpointTelemetry
from router.endpoint_client import load_clients, EndpointClient, ON_DEMAND_COST_PER_1K

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("azure_validation")

RESULTS_DIR = Path(__file__).parent / "results"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "endpoints.yaml"

# Azure D2ds_v7 on-demand price — used to derive a REAL, throughput-based cost
# per 1k tokens from each request's measured wall latency (see endpoint_client).
AZURE_D2DS_V7_USD_PER_HR = 0.096
ROUTER_OVERHEAD_SLO_MS = 5.0   # H2: router decision must add <5ms


# ---------------------------------------------------------------------------
# Router that knows about the live Azure CPU model (tinyllama).
# The shipped PredictiveRouter only lists GPU mistral/llama; we extend the
# capability prior so the real router actually selects the live endpoint.
# ---------------------------------------------------------------------------
class AzureCPURouter(PredictiveRouter):
    VIABLE_MODELS = {
        LengthBucket.SHORT:  {"tinyllama"},
        LengthBucket.MEDIUM: {"tinyllama"},
        LengthBucket.LONG:   {"tinyllama"},
    }


# ---------------------------------------------------------------------------
# Statistics helpers (numpy only — no scipy dependency)
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (better than normal at extremes)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(data, stat=np.mean, n_boot: int = 10_000, ci: float = 95.0,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI. Returns (point_estimate, lo, hi)."""
    arr = np.asarray([x for x in data if np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.array([
        stat(rng.choice(arr, size=arr.size, replace=True)) for _ in range(n_boot)
    ])
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    return (float(stat(arr)), float(lo), float(hi))


def permutation_test_diff_means(a, b, n_perm: int = 20_000, seed: int = 0) -> float:
    """Two-sided permutation test for difference in means. Returns p-value."""
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    b = np.asarray([x for x in b if np.isfinite(x)], dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    na = a.size
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(pooled[:na].mean() - pooled[na:].mean()) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def pctl(data, q: float) -> float:
    arr = np.asarray([x for x in data if np.isfinite(x)], dtype=float)
    return float(np.percentile(arr, q)) if arr.size else float("nan")


# ---------------------------------------------------------------------------
# Query loading (real dataset, with a built-in fallback bank)
# ---------------------------------------------------------------------------

_FALLBACK = {
    LengthBucket.SHORT: [
        "What is Kubernetes?", "Define LLM inference.", "What is a REST API?",
        "Explain microservices briefly.", "What is cloud computing?",
        "What does vLLM stand for?", "What is a GPU?", "Define token in NLP.",
    ],
    LengthBucket.MEDIUM: [
        "Explain how transformer attention works and why it matters for modern NLP.",
        "Compare SQL and NoSQL databases. When should you choose one over the other?",
        "Describe the CAP theorem and its implications for distributed systems.",
        "Difference between supervised and unsupervised learning? Give examples.",
        "Explain Docker containerisation vs virtual machines in isolation and overhead.",
    ],
    LengthBucket.LONG: [
        "Write a Python distributed rate limiter using Redis with a sliding window, "
        "supporting multiple tiers, race-condition handling via Lua, graceful Redis-down "
        "degradation, and unit tests for boundary conditions and concurrency. Explain the "
        "time complexity of each operation and how to extend it to Redis Cluster.",
        "Design a fault-tolerant event-driven microservices architecture for an e-commerce "
        "platform handling 100k orders/day: service decomposition, Kafka vs RabbitMQ, saga "
        "pattern for distributed transactions, circuit breakers, OpenTelemetry tracing, and "
        "solving the dual-write problem between database and broker without losing consistency.",
    ],
}


def load_queries(n: int, seed: int) -> list[QueryFeatures]:
    """Real dataset queries with the 50/35/15 bucket split; fall back to bank on failure."""
    try:
        from data.dataset_loader import load_experiment_queries
        qs = load_experiment_queries(n_total=n, seed=seed)
        if qs:
            return qs
    except Exception as exc:  # network / datasets issues shouldn't block the live test
        log.warning("Dataset load failed (%s) — using built-in fallback queries", exc)

    qa = QueryAnalyzer(tokenizer="cl100k_base")
    import random
    rng = random.Random(seed)
    split = [(LengthBucket.SHORT, 0.50), (LengthBucket.MEDIUM, 0.35), (LengthBucket.LONG, 0.15)]
    out: list[QueryFeatures] = []
    for bucket, frac in split:
        for _ in range(max(1, round(n * frac))):
            out.append(qa.analyze(rng.choice(_FALLBACK[bucket])))
    rng.shuffle(out)
    return out[:n]


# ---------------------------------------------------------------------------
# Result row
# ---------------------------------------------------------------------------
@dataclass
class Row:
    idx: int
    bucket: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    overhead_ms: float
    latency_ms: float
    measured_cost_per_1k: float
    success: bool


def _measured_cost_per_1k(latency_ms: float, total_tokens: int) -> float:
    """Real throughput-derived cost: (instance $/s * wall seconds) / tokens * 1000."""
    if total_tokens <= 0:
        return float("nan")
    usd_per_s = AZURE_D2DS_V7_USD_PER_HR / 3600.0
    cost = usd_per_s * (latency_ms / 1000.0)
    return cost / total_tokens * 1000.0


# ---------------------------------------------------------------------------
# Phase A — sequential routed run (H1 cost, H2 latency + overhead)
# ---------------------------------------------------------------------------

def phase_a(queries, router, telemetry: TelemetryStore, model_key: str,
            client: EndpointClient, max_tokens: int) -> list[Row]:
    rows: list[Row] = []
    rate = ON_DEMAND_COST_PER_1K.get(("azure", model_key), 0.0002)
    log.info("Phase A: %d sequential routed live calls (max_tokens=%d)…",
             len(queries), max_tokens)
    for i, q in enumerate(queries):
        # Refresh telemetry before each decision (mirrors the live 30s-TTL feed
        # and prevents the seeded entry from ageing out during slow CPU runs).
        ok_lat = [r.latency_ms for r in rows if r.success]
        p95 = float(np.percentile(ok_lat, 95)) if ok_lat else 1500.0
        telemetry.set(EndpointTelemetry("azure", model_key, cost_per_1k_tokens=rate,
                                        p95_latency_ms=p95, healthy=True))
        t0 = time.perf_counter()
        decision = router.route(q)
        overhead_ms = (time.perf_counter() - t0) * 1000.0

        res = client.infer(q.text, max_new_tokens=max_tokens)
        cost1k = _measured_cost_per_1k(res.latency_ms, res.total_tokens) if res.success else float("nan")
        rows.append(Row(
            idx=i, bucket=q.bucket.value, input_tokens=res.input_tokens,
            output_tokens=res.output_tokens, total_tokens=res.total_tokens,
            overhead_ms=overhead_ms, latency_ms=res.latency_ms if res.success else float("inf"),
            measured_cost_per_1k=cost1k, success=res.success,
        ))
        if (i + 1) % 10 == 0:
            ok = sum(1 for r in rows if r.success)
            log.info("  %d/%d done (%d ok)  last=%.0fms %s",
                     i + 1, len(queries), ok, rows[-1].latency_ms, decision.cloud + ":" + decision.model)
    return rows


# ---------------------------------------------------------------------------
# Phase B — concurrent load (H3 completion rate / reliability)
# ---------------------------------------------------------------------------

def phase_b(queries, client: EndpointClient, concurrency: int, max_tokens: int,
            total: int) -> tuple[int, int, list[float]]:
    # Cycle the query pool up to `total` requests so the reliability sample is
    # large enough for a tight Wilson CI (need ~760 perfect calls to clear 99.5%).
    load = [queries[i % len(queries)] for i in range(total)]
    log.info("Phase B: %d concurrent requests at concurrency=%d, max_tokens=%d (H3 reliability)…",
             total, concurrency, max_tokens)
    lat: list[float] = []
    ok = 0

    def _call(q: QueryFeatures):
        r = client.infer(q.text, max_new_tokens=max_tokens)
        return r.success, r.latency_ms

    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_call, q) for q in load]
        for fut in as_completed(futures):
            success, latency = fut.result()
            done += 1
            if success:
                ok += 1
                lat.append(latency)
            if done % 50 == 0:
                log.info("  load %d/%d  (%d ok)", done, total, ok)
    return ok, total, lat


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(rows: list[Row], b_ok: int, b_total: int, b_lat: list[float], elapsed: float) -> None:
    ok_rows = [r for r in rows if r.success]
    lat = [r.latency_ms for r in ok_rows]
    overhead = [r.overhead_ms for r in rows]
    cost1k = [r.measured_cost_per_1k for r in ok_rows]

    print("\n" + "=" * 66)
    print("  Azure-Only Real Validation — LIVE endpoint (tinyllama on AKS)")
    print(f"  Phase A queries: {len(rows)}   Elapsed: {elapsed:.0f}s")
    print("=" * 66)

    # ---- H1: cost ----
    c_pt, c_lo, c_hi = bootstrap_ci(cost1k)
    c_std = float(np.std([c for c in cost1k if np.isfinite(c)])) if cost1k else float("nan")
    print("\n[H1] Measured cost per 1k tokens (throughput-derived, real)")
    print(f"     mean = ${c_pt:.6f}  95% CI [${c_lo:.6f}, ${c_hi:.6f}]  std=${c_std:.6f}")
    print( "     NOTE: cross-cloud reduction vs AWS pending a second live cloud (not simulated).")

    # ---- H2: latency + router overhead ----
    l_pt, l_lo, l_hi = bootstrap_ci(lat)
    p95 = pctl(lat, 95)
    o_pt, o_lo, o_hi = bootstrap_ci(overhead)
    o_p95 = pctl(overhead, 95)
    overhead_pass = o_hi < ROUTER_OVERHEAD_SLO_MS
    print("\n[H2] End-to-end latency (real inference)")
    print(f"     mean = {l_pt:.0f}ms  95% CI [{l_lo:.0f}, {l_hi:.0f}]   P50={pctl(lat,50):.0f}  P95={p95:.0f}ms")
    print(f"     Router decision overhead: mean={o_pt:.3f}ms  95% CI [{o_lo:.3f}, {o_hi:.3f}]  P95={o_p95:.3f}ms")
    print(f"     H2 router overhead <{ROUTER_OVERHEAD_SLO_MS}ms: "
          f"{'PASSED' if overhead_pass else 'NOT MET'} (95% CI upper {o_hi:.3f}ms)")

    # ---- per-bucket latency + significance (H4) ----
    by_bucket = {b: [r.latency_ms for r in ok_rows if r.bucket == b] for b in ("short", "medium", "long")}
    print("\n[H2/H4] Latency by bucket (mean [95% CI], n):")
    for b in ("short", "medium", "long"):
        d = by_bucket[b]
        if d:
            pt, lo, hi = bootstrap_ci(d)
            print(f"     {b:7s} mean={pt:7.0f}ms  95% CI [{lo:.0f}, {hi:.0f}]  n={len(d)}")
        else:
            print(f"     {b:7s} (no samples)")
    # Permutation test on the two best-populated buckets (long is often n<=1).
    populated = sorted(((b, d) for b, d in by_bucket.items() if len(d) >= 2),
                       key=lambda kv: len(kv[1]), reverse=True)
    if len(populated) >= 2:
        (b1, d1), (b2, d2) = populated[0], populated[1]
        p = permutation_test_diff_means(d1, d2)
        print(f"     Permutation test {b1} vs {b2} latency: p={p:.4f} "
              f"({'significant' if p < 0.05 else 'not significant'} at alpha=0.05)")

    # ---- H3: completion rate ----
    rate = b_ok / b_total if b_total else 0.0
    w_lo, w_hi = wilson_ci(b_ok, b_total)
    h3_pass = w_lo >= 0.995
    print("\n[H3] Completion rate under concurrent load (reliability)")
    print(f"     {b_ok}/{b_total} = {rate*100:.2f}%   Wilson 95% CI [{w_lo*100:.2f}%, {w_hi*100:.2f}%]")
    if b_lat:
        print(f"     Under-load P95 latency = {pctl(b_lat,95):.0f}ms")
    print(f"     H3 completion >=99.5%: {'PASSED' if h3_pass else 'NOT MET'} "
          f"(95% CI lower {w_lo*100:.2f}%)")

    # ---- H4 summary ----
    print("\n[H4] Statistical significance — methods applied:")
    print("     • Bootstrap 95% CIs (10k resamples) on cost, latency, overhead")
    print("     • Wilson score 95% CI on completion proportion")
    print("     • Two-sided permutation test (20k) on bucket latency difference")
    print("=" * 66 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(n: int, seed: int, max_tokens: int, concurrency: int, model_key: str,
        load_n: int, load_max_tokens: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- live client for the Azure endpoint ---
    clients = load_clients(CONFIG_PATH)
    key = ("azure", model_key)
    if key not in clients:
        raise SystemExit(
            f"No live client for azure:{model_key}. Configured: {sorted(clients)}.\n"
            f"Check config/endpoints.yaml (URL must not contain REPLACE_)."
        )
    client = clients[key]

    # --- preflight health/inference check ---
    log.info("Preflight: probing azure:%s …", model_key)
    probe = client.infer("ping", max_new_tokens=4)
    if not probe.success:
        raise SystemExit(f"Live endpoint not serving: {probe.error}")
    log.info("Preflight OK — endpoint returned %d tokens in %.0fms",
             probe.total_tokens, probe.latency_ms)

    # --- real router wired to the live model ---
    telemetry = TelemetryStore()
    rate = ON_DEMAND_COST_PER_1K.get(key, 0.0002)
    telemetry.set(EndpointTelemetry("azure", model_key, cost_per_1k_tokens=rate,
                                    p95_latency_ms=1500.0, healthy=True))
    router = AzureCPURouter(telemetry)

    # --- queries ---
    queries = load_queries(n, seed)
    log.info("Loaded %d queries (%s)", len(queries),
             {b: sum(1 for q in queries if q.bucket.value == b) for b in ("short", "medium", "long")})

    t_start = time.monotonic()
    rows = phase_a(queries, router, telemetry, model_key, client, max_tokens)
    b_ok, b_total, b_lat = phase_b(queries, client, concurrency, load_max_tokens, load_n)
    elapsed = time.monotonic() - t_start

    # --- CSV ---
    csv_out = RESULTS_DIR / f"azure_validation_{ts}.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "bucket", "input_tokens", "output_tokens", "total_tokens",
                    "overhead_ms", "latency_ms", "measured_cost_per_1k", "success"])
        for r in rows:
            w.writerow([r.idx, r.bucket, r.input_tokens, r.output_tokens, r.total_tokens,
                        round(r.overhead_ms, 4), round(r.latency_ms, 1),
                        round(r.measured_cost_per_1k, 8) if np.isfinite(r.measured_cost_per_1k) else "",
                        r.success])
    log.info("Per-query results → %s", csv_out)

    report(rows, b_ok, b_total, b_lat, elapsed)
    client.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Azure-only real validation of H1–H4")
    p.add_argument("--n", type=int, default=100, help="Phase A routed queries (latency/cost, default 100)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=128, help="Phase A max_new_tokens (default 128)")
    p.add_argument("--concurrency", type=int, default=8, help="Phase B concurrent workers (default 8)")
    p.add_argument("--load-n", type=int, default=800, help="Phase B total requests for H3 (default 800)")
    p.add_argument("--load-max-tokens", type=int, default=32,
                   help="Phase B max_new_tokens — small keeps throughput up (default 32)")
    p.add_argument("--model-key", type=str, default="tinyllama", help="Azure model key in endpoints.yaml")
    a = p.parse_args()
    run(a.n, a.seed, a.max_tokens, a.concurrency, a.model_key, a.load_n, a.load_max_tokens)
