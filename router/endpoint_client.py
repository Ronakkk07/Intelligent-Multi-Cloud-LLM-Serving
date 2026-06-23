"""
vLLM Endpoint Client
--------------------
Makes real HTTP inference calls to vLLM-served models running on
AWS EKS and Azure AKS. Uses vLLM's OpenAI-compatible
POST /v1/chat/completions API.

Cost is estimated from published on-demand GPU instance prices and
the token counts returned by vLLM in the usage field. The billing
APIs (AWS Cost Explorer, Azure Cost Management) have a 24-48h delay
so per-request cost cannot come from them during the experiment;
we use the proxy formula documented in Section 3.2.
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# On-demand cost estimates (USD / 1k tokens)
# GPU baseline — used for final experiments with NC4as_T4_v3 / g5.xlarge:
#   AWS g5.xlarge $1.006/hr, Azure NV6ads_A10_v5 $0.454/hr
#   Mistral-7B A10G throughput ~1500 tok/s → 5.4M tok/hr
#   Llama-2-13B A10G throughput ~650 tok/s  → 2.34M tok/hr
# CPU baseline — used for Azure during CPU-only development phase:
#   Azure Standard_D2ds_v7 $0.096/hr
#   TinyLlama-1.1B CPU throughput ~15 tok/s → 54k tok/hr
# cost_per_1k = (instance_$/hr) / (tok/hr) * 1000
# ---------------------------------------------------------------------------
ON_DEMAND_COST_PER_1K: dict[tuple[str, str], float] = {
    # GPU (final experiments)
    ("aws",   "mistral-7b"): round(1.006 / 5_400_000 * 1_000, 6),   # ≈ $0.000186
    ("aws",   "llama-13b"):  round(1.006 / 2_340_000 * 1_000, 6),   # ≈ $0.000430
    ("azure", "mistral-7b"): round(0.454 / 5_400_000 * 1_000, 6),   # ≈ $0.000084
    ("azure", "llama-13b"):  round(0.454 / 2_340_000 * 1_000, 6),   # ≈ $0.000194
    # CPU dev phase — Azure D2ds_v7 running TinyLlama via Ollama
    ("azure", "tinyllama"):  round(0.096 / 54_000 * 1_000, 6),      # ≈ $0.001778
}

CONFIG_PATH = Path(__file__).parent.parent / "config" / "endpoints.yaml"


@dataclass
class InferenceResult:
    cloud: str
    model_key: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    success: bool
    error: str | None = None


class EndpointClient:
    """
    One client per (cloud, model_key) endpoint.
    Wraps httpx with retry on transient errors and records real latency.
    """

    def __init__(
        self,
        cloud: str,
        model_key: str,
        base_url: str,
        model_id: str,
        timeout_s: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.cloud = cloud
        self.model_key = model_key
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s))

    # ------------------------------------------------------------------
    def infer(self, text: str, max_new_tokens: int = 512) -> InferenceResult:
        """Send a chat completion request and return a measured result."""
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.monotonic()
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat/completions", json=payload
                )
                latency_ms = (time.monotonic() - t0) * 1_000.0
                resp.raise_for_status()

                body = resp.json()
                usage = body.get("usage", {})
                in_tok  = int(usage.get("prompt_tokens",     0))
                out_tok = int(usage.get("completion_tokens", 0))
                total   = in_tok + out_tok

                rate = ON_DEMAND_COST_PER_1K.get((self.cloud, self.model_key), 0.0002)
                cost = (total / 1_000.0) * rate

                return InferenceResult(
                    cloud=self.cloud,
                    model_key=self.model_key,
                    latency_ms=round(latency_ms, 1),
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    total_tokens=total,
                    cost_usd=round(cost, 7),
                    success=True,
                )

            except Exception as exc:
                last_exc = exc
                latency_ms = (time.monotonic() - t0) * 1_000.0
                if attempt < self.max_retries:
                    log.debug(
                        "Retry %d/%d for %s:%s — %s",
                        attempt + 1, self.max_retries, self.cloud, self.model_key, exc,
                    )

        log.warning(
            "Inference failed [%s:%s] after %d attempts: %s",
            self.cloud, self.model_key, self.max_retries + 1, last_exc,
        )
        return InferenceResult(
            cloud=self.cloud,
            model_key=self.model_key,
            latency_ms=round((time.monotonic() - t0) * 1_000.0, 1),
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            success=False,
            error=str(last_exc),
        )

    def health_check(self) -> bool:
        """Return True if the /health endpoint responds 200."""
        try:
            r = self._client.get(f"{self.base_url}/health", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_clients(
    config_path: Path = CONFIG_PATH,
) -> dict[tuple[str, str], EndpointClient]:
    """
    Load all (cloud, model_key) endpoint clients from config/endpoints.yaml.
    Skips any URL that still contains the placeholder string 'REPLACE_'.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    clients: dict[tuple[str, str], EndpointClient] = {}
    for cloud, models in cfg.get("endpoints", {}).items():
        for model_key, spec in models.items():
            url = spec.get("url", "")
            if "REPLACE_" in url:
                log.warning(
                    "Skipping %s:%s — URL not configured (%s)", cloud, model_key, url
                )
                continue
            clients[(cloud, model_key)] = EndpointClient(
                cloud=cloud,
                model_key=model_key,
                base_url=url,
                model_id=spec["model_id"],
            )
            log.info("Registered endpoint %s:%s → %s", cloud, model_key, url)

    return clients
