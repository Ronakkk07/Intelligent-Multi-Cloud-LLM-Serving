# Intelligent Multi-Cloud LLM Serving — Implementation Skeleton

MSc Research Project · Ronak Rajput (x24195707) · Session 3 progress

## What this is

A runnable, cloud-free skeleton of the Predictive Router from the proposal.
It exercises the full routing pipeline and all four experimental conditions
on a simulated query stream, so the logic can be validated before spending
GPU credits. When real EKS/AKS endpoints come online, **only `telemetry.py`
changes** — everything else stays identical.

## Layout

```
router/
  query_analyzer.py     # tokens -> short/medium/long bucket
  telemetry.py          # mock Redis cost/latency cache w/ 30s TTL
  predictive_router.py  # cost+latency scoring, SLO + compliance filtering
  baselines.py          # static_aws, static_azure, round_robin conditions
experiments/
  run_comparison.py     # runs all 4 conditions, reports cost + P95 latency
k8s/  terraform/  data/ # placeholders for the cloud phase
docs/
  Model_Paper_Metrics_Mapping.docx   # Session 2 feedback deliverable
```

## Run it

```bash
python -m experiments.run_comparison   # head-to-head of all 4 conditions
python router/query_analyzer.py        # bucketing demo
```

## Mapping to Figure 1 (proposal architecture)

| Figure 1 component        | File                      |
|---------------------------|---------------------------|
| Query Analyzer            | router/query_analyzer.py  |
| Cost/Latency/Reliability  | predictive_router._score  |
| Routing Decision Engine   | predictive_router.route   |
| Redis Cache               | router/telemetry.py        |
| Experimental conditions   | router/baselines.py        |

## Honest status note for the demo

The predictive router currently shows only a few % cost saving against the
static-AWS baseline, **not** the 25% H1 target. This is expected with the
placeholder ±15-20% price jitter: the real saving depends on the actual
AWS↔Azure A10 price gap (anchors: AWS g5.xlarge ~$1.00/hr on-demand vs
Azure NV6ads_A10_v5 ~$0.45/hr) and the Mistral↔Llama cost ratio. Validating
that the real gap is large enough to support H1 is the first week-1 task.

## Week-1 plan

1. Request Azure A10 GPU quota today (1-2 day approval).
2. Plug real on-demand/spot prices into `seed_realistic()` and re-run the
   harness to get a realistic projected saving before deploying anything.
3. Stand up ONE cloud (single EKS cluster) with vLLM + Mistral-7B, swap the
   mock telemetry for a real Prometheus scrape, prove the loop on one cloud.
4. Only then add the second cloud. Budget: ~$400 total credit → run short
   measured windows (2-4h), not the full 72h.
