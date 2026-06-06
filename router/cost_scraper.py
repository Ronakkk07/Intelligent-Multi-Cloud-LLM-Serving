"""
Cloud Cost Scraper
------------------
Provides cost-per-1k-token signals for each endpoint.

Two modes (Section 3.2 of the paper):

  proxy  — published on-demand GPU prices ÷ measured token throughput.
            Works immediately; used during the live experiment because
            billing APIs have a 24-48h delay (Junior & Marcon, 2025).

  live   — AWS Cost Explorer / Azure Cost Management APIs.
            Used for post-experiment cost reconciliation, not real-time routing.
"""

from __future__ import annotations
import logging
import os
from dataclasses import dataclass

from router.endpoint_client import ON_DEMAND_COST_PER_1K

log = logging.getLogger(__name__)


@dataclass
class CostSignal:
    cloud: str
    model_key: str
    cost_per_1k_tokens: float
    source: str   # "published" | "cost_explorer" | "azure_cost_mgmt"


# ---------------------------------------------------------------------------
# Proxy mode (default during experiment)
# ---------------------------------------------------------------------------

def get_proxy_costs() -> list[CostSignal]:
    """
    Return cost signals derived from published on-demand instance prices.
    These are the same constants used by EndpointClient; keeping them
    in one place (endpoint_client.ON_DEMAND_COST_PER_1K) avoids drift.
    """
    return [
        CostSignal(
            cloud=cloud,
            model_key=model,
            cost_per_1k_tokens=cost,
            source="published",
        )
        for (cloud, model), cost in ON_DEMAND_COST_PER_1K.items()
    ]


# ---------------------------------------------------------------------------
# Live mode — AWS Cost Explorer (24h delay)
# ---------------------------------------------------------------------------

def get_aws_cost_explorer_costs(
    tag_key: str = "Project",
    tag_value: str = "llm-router",
    region: str = "us-east-1",
) -> list[CostSignal] | None:
    """
    Pull yesterday's AWS spend tagged Project=llm-router from Cost Explorer.
    Returns None when boto3 or credentials are unavailable.

    Note: granularity is daily, not per-request. Use for post-experiment
    reconciliation only — not for the real-time routing loop.
    """
    try:
        import boto3
        from datetime import date, timedelta

        ce = boto3.client("ce", region_name=region)
        today     = date.today().isoformat()
        yesterday = (date.today() - timedelta(days=1)).isoformat()

        resp = ce.get_cost_and_usage(
            TimePeriod={"Start": yesterday, "End": today},
            Granularity="DAILY",
            Filter={"Tags": {"Key": tag_key, "Values": [tag_value]}},
            Metrics=["UnblendedCost"],
            GroupBy=[{"Type": "DIMENSION", "Key": "USAGE_TYPE"}],
        )

        total_usd = sum(
            float(group["Metrics"]["UnblendedCost"]["Amount"])
            for result in resp["ResultsByTime"]
            for group in result["Groups"]
        )
        log.info("AWS Cost Explorer: $%.4f yesterday (tag %s=%s)", total_usd, tag_key, tag_value)
        return None   # TODO: break down by model when per-node tagging is set up
    except ImportError:
        log.debug("boto3 not installed — skipping AWS Cost Explorer")
        return None
    except Exception as exc:
        log.warning("AWS Cost Explorer error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Live mode — Azure Cost Management (48h delay)
# ---------------------------------------------------------------------------

def get_azure_cost_management_costs(
    resource_group: str = "llm-router-rg",
    subscription_id: str | None = None,
) -> list[CostSignal] | None:
    """
    Pull Azure spend from Cost Management for the llm-router resource group.
    Returns None when Azure SDK or credentials are unavailable.
    """
    try:
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.costmanagement import CostManagementClient

        sub_id = subscription_id or os.environ.get("AZURE_SUBSCRIPTION_ID")
        if not sub_id:
            log.debug("AZURE_SUBSCRIPTION_ID not set — skipping Azure Cost Management")
            return None

        cred   = DefaultAzureCredential()
        client = CostManagementClient(cred)
        scope  = f"/subscriptions/{sub_id}/resourceGroups/{resource_group}"

        log.info("Azure Cost Management scope: %s (stub — implement full query)", scope)
        # TODO: call client.query.usage() with a DateRange filter and
        #       group by resource to get per-model GPU cost breakdown
        return None
    except ImportError:
        log.debug("azure-mgmt-costmanagement not installed")
        return None
    except Exception as exc:
        log.warning("Azure Cost Management error: %s", exc)
        return None
