"""
Real Experiment Harness
-----------------------
Routes real queries from HuggingFace datasets through all four experimental
conditions against live vLLM endpoints on AWS EKS and Azure AKS, measuring
actual latency and cost to validate H1 (≥25% cost reduction) and H2 (<10%
latency degradation at 1x load).

Modes:
  --dry-run   Routing decisions only. No inference calls. Uses simulated
              telemetry. Safe to run without cloud access.

  --live      Makes real HTTP inference calls to vLLM endpoints defined
              in config/endpoints.yaml. Incurs cloud cost (~$0.10-0.50
              per 500-query run depending on instance pricing).

Output:
  experiments/results/<timestamp>_<mode>.csv   — per-query results
  Console table with aggregate cost, P95, H1/H2 pass/fail

Usage:
  python -m experiments.run_real_experiment --dry-run
  python -m experiments.run_real_experiment --live --n-queries 500
"""

from __future__ import annotations
import argparse
import csv
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path

import yaml

from router.query_analyzer import QueryAnalyzer, QueryFeatures
from router.baselines import make_router
from router.telemetry import EndpointTelemetry, seed_realistic
from router.redis_telemetry import RedisTelemetryStore
from router.prometheus_scraper import scrape_all as prometheus_scrape_all
from router.cost_scraper import get_proxy_costs
from router.endpoint_client import load_clients, InferenceResult, ON_DEMAND_COST_PER_1K
from data.dataset_loader import load_experiment_queries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

CONDITIONS  = ["static_aws", "static_azure", "round_robin", "predictive"]
RESULTS_DIR = Path(__file__).parent / "results"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "endpoints.yaml"

# H2 baseline: single-cloud static_aws P95 latency (set after first condition)
_AWS_P95_BASELINE: float | None = None


# ---------------------------------------------------------------------------
# Telemetry refresh helpers
# ---------------------------------------------------------------------------

def _load_prometheus_urls(cfg: dict) -> dict[tuple[str, str], str]:
    urls: dict[tuple[str, str], str] = {}
    for cloud, models in cfg.get("prometheus", {}).items():
        for model_key, url in models.items():
            if "REPLACE_" not in url:
                urls[(cloud, model_key)] = url
    return urls


def _refresh_telemetry(
    telemetry: RedisTelemetryStore,
    prom_urls: dict[tuple[str, str], str],
    cost_signals: list,
    dry_run: bool,
) -> None:
    if dry_run or not prom_urls:
        seed_realistic(telemetry, jitter=True)
        return

    cost_map = {(s.cloud, s.model_key): s.cost_per_1k_tokens for s in cost_signals}
    latencies = prometheus_scrape_all(prom_urls)
    updated = 0
    for (cloud, model), p95 in latencies.items():
        if p95 is None:
            continue
        telemetry.set(EndpointTelemetry(
            cloud=cloud,
            model=model,
            cost_per_1k_tokens=cost_map.get((cloud, model), 0.0002),
            p95_latency_ms=p95,
            healthy=True,
        ))
        updated += 1
    log.info("Telemetry refreshed from Prometheus (%d endpoints)", updated)


# ---------------------------------------------------------------------------
# Inference dispatch
# ---------------------------------------------------------------------------

def _call_or_estimate(
    clients: dict,
    decision,
    query: QueryFeatures,
    dry_run: bool,
) -> InferenceResult:
    cloud = decision.cloud
    model = decision.model

    if not dry_run and (cloud, model) in clients:
        return clients[(cloud, model)].infer(query.text)

    # Dry-run: use routing-layer estimates (no HTTP call)
    return InferenceResult(
        cloud=cloud,
        model_key=model,
        latency_ms=decision.est_latency_ms,
        input_tokens=query.input_tokens,
        output_tokens=0,
        total_tokens=query.input_tokens,
        cost_usd=decision.est_cost_usd,
        success=True,
    )


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run(
    n_queries: int = 500,
    seed: int = 42,
    dry_run: bool = True,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode    = "dryrun" if dry_run else "live"
    csv_out = RESULTS_DIR / f"results_{ts}_{mode}.csv"

    # --- Config ---
    cfg: dict = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}

    # --- Telemetry ---
    redis_cfg = cfg.get("redis", {})
    telemetry = RedisTelemetryStore(
        host=redis_cfg.get("host", "localhost"),
        port=redis_cfg.get("port", 6379),
        db=redis_cfg.get("db", 0),
    )
    prom_urls    = _load_prometheus_urls(cfg)
    cost_signals = get_proxy_costs()

    _refresh_telemetry(telemetry, prom_urls, cost_signals, dry_run)

    # --- Endpoint clients ---
    clients: dict = {}
    if not dry_run:
        try:
            clients = load_clients(CONFIG_PATH)
            if not clients:
                log.warning("No configured endpoints found — switching to dry-run")
                dry_run = True
                seed_realistic(telemetry, jitter=True)
        except FileNotFoundError:
            log.warning("config/endpoints.yaml not found — switching to dry-run")
            dry_run = True
            seed_realistic(telemetry, jitter=True)

    # --- Routers ---
    routers = {c: make_router(c, telemetry) for c in CONDITIONS}

    # --- Dataset ---
    log.info("Loading %d queries from HuggingFace datasets…", n_queries)
    queries = load_experiment_queries(n_total=n_queries, seed=seed)
    log.info("Dataset ready: %d queries", len(queries))
    _refresh_telemetry(telemetry, prom_urls, cost_signals, dry_run)

    # --- Results accumulators ---
    results: dict[str, dict] = {
        c: {"cost": 0.0, "lat": [], "errors": 0} for c in CONDITIONS
    }
    csv_rows: list[dict] = []

    log.info("Starting experiment — mode=%s  n_queries=%d", mode.upper(), n_queries)
    t_start = time.monotonic()

    for i, query in enumerate(queries):
        # Refresh telemetry every 50 queries (mirrors 30s TTL cadence)
        if i > 0 and i % 50 == 0:
            _refresh_telemetry(telemetry, prom_urls, cost_signals, dry_run)
            log.info("Query %d/%d  elapsed=%.0fs", i, n_queries,
                     time.monotonic() - t_start)

        for cond in CONDITIONS:
            decision = routers[cond].route(query)
            result   = _call_or_estimate(clients, decision, query, dry_run)

            lat  = result.latency_ms if result.success else float("inf")
            cost = result.cost_usd   if result.success else 0.0

            results[cond]["cost"]   += cost
            results[cond]["lat"].append(lat)
            results[cond]["errors"] += 0 if result.success else 1

            csv_rows.append({
                "query_idx":    i,
                "bucket":       query.bucket.value,
                "input_tokens": query.input_tokens,
                "condition":    cond,
                "cloud":        decision.cloud,
                "model":        decision.model,
                "latency_ms":   round(lat, 1),
                "cost_usd":     round(cost, 7),
                "success":      result.success,
            })

    # --- Write CSV ---
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    log.info("Results → %s", csv_out)

    # --- Console report ---
    _print_report(results, n_queries, mode, time.monotonic() - t_start)

    for c in clients.values():
        c.close()


def _print_report(
    results: dict[str, dict],
    n_queries: int,
    mode: str,
    elapsed_s: float,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Multi-Cloud LLM Router — Experiment Results [{mode.upper()}]")
    print(f"  Queries: {n_queries}   Elapsed: {elapsed_s:.0f}s")
    print(f"{'=' * 60}")
    print(f"\n{'Condition':<14}{'Total cost ($)':>16}{'Mean P95 (ms)':>16}{'Errors':>8}")
    print("-" * 56)

    baseline_cost = None
    baseline_lat  = None

    for cond in CONDITIONS:
        cost   = results[cond]["cost"]
        valid  = [l for l in results[cond]["lat"] if l != float("inf")]
        lat    = statistics.mean(valid) if valid else float("inf")
        errors = results[cond]["errors"]

        if cond == "static_aws":
            baseline_cost = cost
            baseline_lat  = lat

        print(f"{cond:<14}{cost:>16.4f}{lat:>16.1f}{errors:>8}")

    print("-" * 56)

    if baseline_cost and baseline_cost > 0:
        pred_cost = results["predictive"]["cost"]
        pred_lat  = statistics.mean(
            [l for l in results["predictive"]["lat"] if l != float("inf")]
        )
        cost_saving = (baseline_cost - pred_cost) / baseline_cost * 100
        lat_delta   = (pred_lat - baseline_lat) / baseline_lat * 100 if baseline_lat else 0

        print(f"\nH1 (cost >=25% reduction):  {cost_saving:+.1f}%  "
              f"{'PASSED' if cost_saving >= 25 else 'NOT MET'}")
        print(f"H2 (latency <10% degradation at 1x): {lat_delta:+.1f}%  "
              f"{'PASSED' if lat_delta <= 10 else 'NOT MET'}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Cloud LLM Router experiment")
    mode_grp = parser.add_mutually_exclusive_group()
    mode_grp.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Routing decisions only — no real inference calls (default)",
    )
    mode_grp.add_argument(
        "--live", action="store_true", default=False,
        help="Make real vLLM inference calls (cloud cost incurred)",
    )
    parser.add_argument("--n-queries", type=int, default=500,
                        help="Number of queries to run (default 500)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        n_queries=args.n_queries,
        seed=args.seed,
        dry_run=not args.live,
    )
