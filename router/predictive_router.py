"""
Predictive Router
-----------------
The heart of the system. Given a query and the live telemetry, it:

  1. Decides which MODEL is good enough (capability prior by bucket)
  2. Estimates COST and LATENCY for each viable (cloud, model) endpoint
  3. Applies COMPLIANCE constraints (region/provider restrictions)
  4. Picks the endpoint minimising a cost/latency objective subject to
     a latency SLO.

This is the "Routing Decision Engine + Cost/Latency/Reliability models"
box from Figure 1 of the proposal. Right now the cost/latency models are
analytic (tokens x telemetry). In the full project this is where the
trained predictive model plugs in -- the interface stays identical.
"""

from dataclasses import dataclass
from router.query_analyzer import QueryFeatures, LengthBucket
from router.telemetry import TelemetryStore, EndpointTelemetry


@dataclass
class RoutingDecision:
    cloud: str
    model: str
    est_cost_usd: float
    est_latency_ms: float
    reason: str


class PredictiveRouter:
    # Latency SLO in ms; endpoints predicted to exceed this are filtered out
    # unless none qualify (then we pick the fastest available, degraded mode).
    LATENCY_SLO_MS = 2500

    # Estimated OUTPUT tokens per bucket (output dominates cost for short
    # prompts). Crude but enough for the skeleton; replaced by the predictor.
    OUTPUT_TOKENS = {
        LengthBucket.SHORT: 80,
        LengthBucket.MEDIUM: 350,
        LengthBucket.LONG: 800,
    }

    # Capability prior: which models can acceptably serve each bucket.
    # Short/medium -> cheap Mistral is fine. Long/complex -> allow both,
    # let cost/latency decide (the "interesting choice" from the proposal).
    VIABLE_MODELS = {
        LengthBucket.SHORT: {"mistral-7b"},
        LengthBucket.MEDIUM: {"mistral-7b", "llama-13b"},
        LengthBucket.LONG: {"mistral-7b", "llama-13b"},
    }

    def __init__(self, telemetry: TelemetryStore, latency_weight: float = 0.4):
        self.telemetry = telemetry
        # objective = cost + latency_weight * (latency_ms / 1000)
        self.latency_weight = latency_weight

    def _total_tokens(self, f: QueryFeatures) -> int:
        return f.input_tokens + self.OUTPUT_TOKENS[f.bucket]

    def _score(self, t: EndpointTelemetry, total_tokens: int) -> tuple[float, float, float]:
        cost = (total_tokens / 1000.0) * t.cost_per_1k_tokens
        latency = t.p95_latency_ms
        objective = cost + self.latency_weight * (latency / 1000.0)
        return objective, cost, latency

    def route(
        self,
        f: QueryFeatures,
        allowed_clouds: set[str] | None = None,   # compliance constraint
    ) -> RoutingDecision:
        total_tokens = self._total_tokens(f)
        viable_models = self.VIABLE_MODELS[f.bucket]

        candidates: list[tuple[float, float, float, EndpointTelemetry]] = []
        for t in self.telemetry.all_healthy():
            if t.model not in viable_models:
                continue
            if allowed_clouds is not None and t.cloud not in allowed_clouds:
                continue
            obj, cost, lat = self._score(t, total_tokens)
            candidates.append((obj, cost, lat, t))

        if not candidates:
            raise RuntimeError("No healthy endpoint satisfies constraints")

        # Prefer endpoints within SLO; fall back to fastest if none qualify
        within_slo = [c for c in candidates if c[2] <= self.LATENCY_SLO_MS]
        pool = within_slo if within_slo else candidates
        pool.sort(key=lambda c: c[0])
        obj, cost, lat, best = pool[0]

        reason = (
            f"bucket={f.bucket.value}, viable={sorted(viable_models)}, "
            f"chose {best.cloud}:{best.model} "
            f"(obj={obj:.4f}, cost=${cost:.4f}, p95={lat:.0f}ms"
            f"{'' if within_slo else ', DEGRADED: no endpoint within SLO'})"
        )
        return RoutingDecision(best.cloud, best.model, round(cost, 5), lat, reason)
