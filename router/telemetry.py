"""
Telemetry Store
---------------
Holds the live cost-per-1k-tokens and P95 latency for every
(cloud, model) endpoint. In the real system this is Redis with a
30s TTL, populated by the Kubernetes controller scraping Prometheus
and the cloud billing APIs (FOCUS-normalised).

For the offline skeleton this is an in-memory dict that you can
perturb to simulate price/latency drift between AWS and Azure,
so the router has something non-trivial to decide on.
"""

import time
import random
from dataclasses import dataclass, field


@dataclass
class EndpointTelemetry:
    cloud: str               # "aws" | "azure"
    model: str               # "mistral-7b" | "llama-13b"
    cost_per_1k_tokens: float
    p95_latency_ms: float
    healthy: bool = True
    updated_at: float = field(default_factory=time.time)


class TelemetryStore:
    TTL_SECONDS = 30

    def __init__(self):
        self._store: dict[str, EndpointTelemetry] = {}

    @staticmethod
    def key(cloud: str, model: str) -> str:
        return f"{cloud}:{model}"

    def set(self, t: EndpointTelemetry) -> None:
        t.updated_at = time.time()
        self._store[self.key(t.cloud, t.model)] = t

    def get(self, cloud: str, model: str) -> EndpointTelemetry | None:
        t = self._store.get(self.key(cloud, model))
        if t is None:
            return None
        # TTL guard: stale telemetry is treated as unavailable, matching the
        # proposal's protection against persisting weights during API outages
        if time.time() - t.updated_at > self.TTL_SECONDS:
            return None
        return t

    def all_healthy(self) -> list[EndpointTelemetry]:
        return [
            t for t in self._store.values()
            if t.healthy and (time.time() - t.updated_at) <= self.TTL_SECONDS
        ]


def seed_realistic(store: TelemetryStore, jitter: bool = True) -> None:
    """
    Seed with plausible numbers. Mistral-7B is cheaper & faster than
    Llama-13B; AWS and Azure differ slightly so cross-cloud arbitrage
    is meaningful. Jitter simulates time-of-day price/latency drift.
    """
    base = [ # cloud, model, cost_per_1k_tokens, p95_latency_ms
        ("aws",   "mistral-7b", 0.20,  900),
        ("aws",   "llama-13b",  0.40, 1600),
        ("azure", "mistral-7b", 0.23,  820),
        ("azure", "llama-13b",  0.36, 1750),
    ]
    for cloud, model, cost, lat in base:
        if jitter:
            cost *= random.uniform(0.85, 1.20)
            lat *= random.uniform(0.90, 1.15)
        store.set(EndpointTelemetry(cloud, model, round(cost, 4), round(lat, 1)))
