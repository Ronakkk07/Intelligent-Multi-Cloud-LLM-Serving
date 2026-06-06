"""
Router Microservice
-------------------
FastAPI HTTP service implementing the Predictive Router entry point.
This is the "Predictive Router" box in Figure 1 of the paper.

Endpoints:
  POST /v1/route           — analyze query, pick endpoint, run inference
  GET  /health             — liveness probe (used by K8s)
  GET  /metrics            — Prometheus metrics for Grafana

Usage (development):
  uvicorn router.router_service:app --host 0.0.0.0 --port 8080 --reload

Usage (production via Dockerfile):
  CMD ["uvicorn", "router.router_service:app", "--host", "0.0.0.0", "--port", "8080"]
"""

from __future__ import annotations
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import yaml
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from starlette.responses import Response

from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST,
)

from router.query_analyzer import QueryAnalyzer
from router.predictive_router import PredictiveRouter, RoutingDecision
from router.redis_telemetry import RedisTelemetryStore
from router.cost_scraper import get_proxy_costs
from router.endpoint_client import load_clients, InferenceResult
from router.telemetry import EndpointTelemetry, seed_realistic

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "endpoints.yaml"

# ---------------------------------------------------------------------------
# Prometheus metrics (H2: router decision latency must be <5ms)
# ---------------------------------------------------------------------------
REQUEST_TOTAL = Counter(
    "router_requests_total",
    "Total routing requests",
    ["cloud", "model", "bucket"],
)
DECISION_LATENCY = Histogram(
    "router_decision_latency_seconds",
    "Time to make a routing decision (target <5ms)",
    buckets=[0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1],
)
E2E_LATENCY = Histogram(
    "router_e2e_latency_seconds",
    "End-to-end latency including inference",
    buckets=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, float("inf")],
)
INFERENCE_ERRORS = Counter(
    "router_inference_errors_total",
    "Failed inference calls",
    ["cloud", "model"],
)
HEALTHY_ENDPOINTS = Gauge(
    "router_healthy_endpoints",
    "Number of healthy endpoints in telemetry",
)

# ---------------------------------------------------------------------------
# Application state (initialised in lifespan)
# ---------------------------------------------------------------------------
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg: dict = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}

    redis_cfg = cfg.get("redis", {})
    telemetry = RedisTelemetryStore(
        host=redis_cfg.get("host", "localhost"),
        port=redis_cfg.get("port", 6379),
        db=redis_cfg.get("db", 0),
    )

    dry_run = os.environ.get("DRY_RUN", "true").lower() in ("1", "true", "yes")

    clients: dict = {}
    if not dry_run:
        try:
            clients = load_clients(CONFIG_PATH)
        except FileNotFoundError:
            log.warning("endpoints.yaml not found — running in dry-run mode")
            dry_run = True

    if dry_run or not clients:
        seed_realistic(telemetry, jitter=True)
        log.info("Seeded telemetry with simulated values (dry-run=%s)", dry_run)
    else:
        # Refresh telemetry from real Prometheus on startup
        from router.prometheus_scraper import scrape_all
        prom_cfg = cfg.get("prometheus", {})
        prom_urls = {
            (cloud, model): url
            for cloud, models in prom_cfg.items()
            for model, url in models.items()
        }
        if prom_urls:
            latencies = scrape_all(prom_urls)
            costs = {(s.cloud, s.model_key): s.cost_per_1k_tokens
                     for s in get_proxy_costs()}
            for (cloud, model), p95 in latencies.items():
                if p95 is not None:
                    telemetry.set(EndpointTelemetry(
                        cloud=cloud, model=model,
                        cost_per_1k_tokens=costs.get((cloud, model), 0.0002),
                        p95_latency_ms=p95, healthy=True,
                    ))

    _state["telemetry"] = telemetry
    _state["router"]    = PredictiveRouter(telemetry)
    _state["analyzer"]  = QueryAnalyzer(tokenizer="cl100k_base")
    _state["clients"]   = clients
    _state["dry_run"]   = dry_run

    HEALTHY_ENDPOINTS.set(len(telemetry.all_healthy()))
    log.info("Router service ready (dry_run=%s, endpoints=%d)", dry_run, len(clients))

    yield

    for c in clients.values():
        c.close()


app = FastAPI(
    title="Multi-Cloud LLM Router",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RouteRequest(BaseModel):
    query: str
    compliance_cloud: str | None = None   # "aws" | "azure" | None
    max_new_tokens: int = 512


class RouteResponse(BaseModel):
    cloud: str
    model: str
    est_cost_usd: float
    est_latency_ms: float
    reason: str
    # Populated only in live mode
    actual_latency_ms: float | None = None
    actual_cost_usd: float | None = None
    output_tokens: int | None = None
    success: bool = True
    error: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    n = len(_state.get("telemetry", RedisTelemetryStore()).all_healthy())
    HEALTHY_ENDPOINTS.set(n)
    return {"status": "ok", "healthy_endpoints": n}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/route", response_model=RouteResponse)
def route(req: RouteRequest):
    analyzer: QueryAnalyzer    = _state["analyzer"]
    router:   PredictiveRouter = _state["router"]
    clients:  dict             = _state["clients"]
    dry_run:  bool             = _state["dry_run"]

    # --- Routing decision (must be <5ms, H2) ---
    t_decision = time.monotonic()
    features  = analyzer.analyze(req.query)
    allowed   = {req.compliance_cloud} if req.compliance_cloud else None
    try:
        decision: RoutingDecision = router.route(features, allowed_clouds=allowed)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    decision_ms = (time.monotonic() - t_decision) * 1_000.0

    DECISION_LATENCY.observe(decision_ms / 1_000.0)
    REQUEST_TOTAL.labels(
        cloud=decision.cloud, model=decision.model, bucket=features.bucket.value
    ).inc()

    # --- Inference (live mode only) ---
    result: InferenceResult | None = None
    if not dry_run and (decision.cloud, decision.model) in clients:
        t_e2e = time.monotonic()
        result = clients[(decision.cloud, decision.model)].infer(
            req.query, max_new_tokens=req.max_new_tokens
        )
        E2E_LATENCY.observe(time.monotonic() - t_e2e)
        if not result.success:
            INFERENCE_ERRORS.labels(cloud=decision.cloud, model=decision.model).inc()

    return RouteResponse(
        cloud=decision.cloud,
        model=decision.model,
        est_cost_usd=decision.est_cost_usd,
        est_latency_ms=decision.est_latency_ms,
        reason=decision.reason,
        actual_latency_ms=result.latency_ms if result else None,
        actual_cost_usd=result.cost_usd if result else None,
        output_tokens=result.output_tokens if result else None,
        success=result.success if result else True,
        error=result.error if result else None,
    )


if __name__ == "__main__":
    import uvicorn
    cfg: dict = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
    svc = cfg.get("router_service", {})
    uvicorn.run(
        "router.router_service:app",
        host=svc.get("host", "0.0.0.0"),
        port=svc.get("port", 8080),
        reload=False,
    )
