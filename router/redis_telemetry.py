"""
Redis-backed Telemetry Store
-----------------------------
Drop-in replacement for the in-memory TelemetryStore when running
against real cloud endpoints. Stores EndpointTelemetry in Redis with
the same 30-second TTL so the router always sees fresh cost/latency.

Falls back to the in-memory parent class if Redis is unreachable,
allowing dry-run mode to work without a running Redis instance.
"""

from __future__ import annotations
import json
import logging
import time

from router.telemetry import EndpointTelemetry, TelemetryStore

log = logging.getLogger(__name__)

_KEY_PREFIX = "llm_router:telemetry:"


class RedisTelemetryStore(TelemetryStore):
    """
    Extends TelemetryStore with Redis persistence.
    The in-memory dict in the parent is kept as a local fallback cache.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0) -> None:
        super().__init__()
        self._redis_ok = False
        try:
            import redis as _redis
            self._r = _redis.Redis(
                host=host, port=port, db=db,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            self._r.ping()
            self._redis_ok = True
            log.info("Redis connected at %s:%d", host, port)
        except Exception as exc:
            log.warning("Redis unavailable (%s) — using in-memory fallback", exc)

    # ------------------------------------------------------------------
    def set(self, t: EndpointTelemetry) -> None:
        t.updated_at = time.time()
        super().set(t)   # always update in-memory

        if not self._redis_ok:
            return
        key = _KEY_PREFIX + self.key(t.cloud, t.model)
        payload = json.dumps({
            "cloud":              t.cloud,
            "model":              t.model,
            "cost_per_1k_tokens": t.cost_per_1k_tokens,
            "p95_latency_ms":     t.p95_latency_ms,
            "healthy":            t.healthy,
            "updated_at":         t.updated_at,
        })
        try:
            self._r.setex(key, self.TTL_SECONDS, payload)
        except Exception as exc:
            log.warning("Redis set failed: %s", exc)

    def get(self, cloud: str, model: str) -> EndpointTelemetry | None:
        if not self._redis_ok:
            return super().get(cloud, model)
        key = _KEY_PREFIX + self.key(cloud, model)
        try:
            raw = self._r.get(key)
        except Exception:
            return super().get(cloud, model)
        if not raw:
            return None
        d = json.loads(raw)
        return EndpointTelemetry(
            cloud=d["cloud"],
            model=d["model"],
            cost_per_1k_tokens=d["cost_per_1k_tokens"],
            p95_latency_ms=d["p95_latency_ms"],
            healthy=d["healthy"],
            updated_at=d["updated_at"],
        )

    def all_healthy(self) -> list[EndpointTelemetry]:
        if not self._redis_ok:
            return super().all_healthy()
        try:
            keys = self._r.keys(_KEY_PREFIX + "*")
        except Exception:
            return super().all_healthy()
        results: list[EndpointTelemetry] = []
        for key in keys:
            try:
                raw = self._r.get(key)
            except Exception:
                continue
            if not raw:
                continue
            d = json.loads(raw)
            t = EndpointTelemetry(
                cloud=d["cloud"],
                model=d["model"],
                cost_per_1k_tokens=d["cost_per_1k_tokens"],
                p95_latency_ms=d["p95_latency_ms"],
                healthy=d["healthy"],
                updated_at=d["updated_at"],
            )
            if t.healthy:
                results.append(t)
        return results
