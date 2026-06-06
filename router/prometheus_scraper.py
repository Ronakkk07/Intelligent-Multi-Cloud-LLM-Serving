"""
Prometheus Scraper
------------------
Scrapes vLLM /metrics endpoints to read real P95 end-to-end latency.

vLLM exposes a Prometheus histogram:
  vllm:e2e_request_latency_seconds_bucket{...le="X"} <cumulative_count>

We parse the histogram and interpolate to find the 95th percentile.
The result feeds directly into TelemetryStore so the router sees
real measured latencies rather than seeded approximations.
"""

from __future__ import annotations
import logging
import re

import httpx

log = logging.getLogger(__name__)

_BUCKET_RE = re.compile(
    r'vllm:e2e_request_latency_seconds_bucket\{[^}]*le="([^"]+)"[^}]*\}\s+([\d.eE+\-]+)'
)


def scrape_p95_latency_ms(metrics_url: str, timeout_s: float = 5.0) -> float | None:
    """
    Fetch Prometheus text from *metrics_url* and return P95 latency in ms.
    Returns None if the endpoint is unreachable or the metric is absent
    (e.g. no requests have been served yet).
    """
    try:
        resp = httpx.get(metrics_url, timeout=timeout_s)
        resp.raise_for_status()
        return _parse_p95_ms(resp.text)
    except httpx.HTTPStatusError as exc:
        log.debug("HTTP error scraping %s: %s", metrics_url, exc)
    except httpx.RequestError as exc:
        log.debug("Request error scraping %s: %s", metrics_url, exc)
    except Exception as exc:
        log.warning("Unexpected error scraping %s: %s", metrics_url, exc)
    return None


def _parse_p95_ms(text: str) -> float | None:
    """Parse a Prometheus text payload and return P95 latency in ms."""
    buckets: list[tuple[float, float]] = []

    for m in _BUCKET_RE.finditer(text):
        le_str = m.group(1)
        le = float("inf") if le_str == "+Inf" else float(le_str)
        count = float(m.group(2))
        buckets.append((le, count))

    if not buckets:
        return None

    buckets.sort(key=lambda x: x[0])
    total = buckets[-1][1]   # +Inf bucket = total request count
    if total == 0:
        return None

    target = 0.95 * total
    prev_le, prev_count = 0.0, 0.0

    for le, count in buckets:
        if count >= target:
            if count == prev_count:
                p95_s = le
            else:
                # Linear interpolation within the bucket
                frac = (target - prev_count) / (count - prev_count)
                p95_s = prev_le + frac * (le - prev_le)
            return round(p95_s * 1_000.0, 1)
        prev_le, prev_count = le, count

    return None


def scrape_all(
    prometheus_urls: dict[tuple[str, str], str],
) -> dict[tuple[str, str], float | None]:
    """
    Scrape P95 latency for every (cloud, model_key) → url mapping.
    Returns the same keys with float ms values (or None on failure).
    """
    return {
        key: scrape_p95_latency_ms(url)
        for key, url in prometheus_urls.items()
    }
