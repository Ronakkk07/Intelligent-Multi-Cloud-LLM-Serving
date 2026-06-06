"""
Offline Comparison Harness
--------------------------
Runs a stream of queries through all four experimental conditions and
reports total cost and mean P95 latency per condition. This is a dry
run of the H1/H2 evaluation -- no cloud needed. When real endpoints are
live, only telemetry.py changes; this harness stays the same.

Run:  python -m experiments.run_comparison
"""

import random
import statistics
from router.query_analyzer import QueryAnalyzer
from router.telemetry import TelemetryStore, seed_realistic
from router.baselines import make_router

CONDITIONS = ["static_aws", "static_azure", "round_robin", "predictive"]


def synthetic_query(rng: random.Random) -> str:
    bucket = rng.choices(["short", "medium", "long"], weights=[0.5, 0.35, 0.15])[0]
    n = {"short": rng.randint(5, 60),
         "medium": rng.randint(100, 350),
         "long": rng.randint(550, 1200)}[bucket]
    return "word " * n


def run(n_queries: int = 500, seed: int = 42):
    rng = random.Random(seed)
    qa = QueryAnalyzer()

    # Shared telemetry, re-jittered every 50 queries to mimic price drift
    telemetry = TelemetryStore()
    seed_realistic(telemetry, jitter=True)

    routers = {c: make_router(c, telemetry) for c in CONDITIONS}
    results = {c: {"cost": 0.0, "lat": []} for c in CONDITIONS}

    queries = [qa.analyze(synthetic_query(rng)) for _ in range(n_queries)]

    for i, f in enumerate(queries):
        if i % 50 == 0:
            seed_realistic(telemetry, jitter=True)   # provider prices shift
        for c in CONDITIONS:
            d = routers[c].route(f)
            results[c]["cost"] += d.est_cost_usd
            results[c]["lat"].append(d.est_latency_ms)

    print(f"\n{'Condition':<14}{'Total cost ($)':>16}{'Mean P95 (ms)':>16}")
    print("-" * 46)
    baseline_cost = None
    for c in CONDITIONS:
        cost = results[c]["cost"]
        lat = statistics.mean(results[c]["lat"])
        if c == "static_aws":
            baseline_cost = cost
        print(f"{c:<14}{cost:>16.3f}{lat:>16.1f}")

    pred = results["predictive"]["cost"]
    if baseline_cost:
        saving = (baseline_cost - pred) / baseline_cost * 100
        print("-" * 46)
        print(f"Predictive vs static AWS baseline: {saving:+.1f}% cost  "
              f"(H1 target: >=25% reduction)")


if __name__ == "__main__":
    run()
