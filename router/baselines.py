"""
Baseline Routers (experimental conditions)
-------------------------------------------
Implements the four conditions from Section 3.3 of the proposal so the
predictive router can be compared head-to-head:

  (a) static AWS EKS baseline   -> always AWS, capability-appropriate model
  (b) static Azure AKS baseline -> always Azure
  (c) naive round-robin         -> alternate clouds, cost-unaware
  (d) predictive router         -> PredictiveRouter (the proposed system)

All share the same interface so the harness can swap them freely.
"""

import itertools
from router.query_analyzer import QueryFeatures
from router.predictive_router import PredictiveRouter, RoutingDecision
from router.telemetry import TelemetryStore


class _FixedCloudRouter:
    def __init__(self, telemetry: TelemetryStore, cloud: str):
        self.telemetry = telemetry
        self.cloud = cloud
        self._pr = PredictiveRouter(telemetry)

    def route(self, f: QueryFeatures, allowed_clouds=None) -> RoutingDecision:
        # reuse predictive scoring but lock to one cloud
        return self._pr.route(f, allowed_clouds={self.cloud})


class StaticAWS(_FixedCloudRouter):
    def __init__(self, telemetry): super().__init__(telemetry, "aws")


class StaticAzure(_FixedCloudRouter):
    def __init__(self, telemetry): super().__init__(telemetry, "azure")


class RoundRobin:
    """Cost-unaware multi-cloud control. Alternates clouds regardless of price."""
    def __init__(self, telemetry: TelemetryStore):
        self.telemetry = telemetry
        self._pr = PredictiveRouter(telemetry)
        self._cycle = itertools.cycle(["aws", "azure"])

    def route(self, f: QueryFeatures, allowed_clouds=None) -> RoutingDecision:
        cloud = next(self._cycle)
        return self._pr.route(f, allowed_clouds={cloud})


def make_router(name: str, telemetry: TelemetryStore):
    return {
        "static_aws": StaticAWS,
        "static_azure": StaticAzure,
        "round_robin": RoundRobin,
        "predictive": PredictiveRouter,
    }[name](telemetry)
