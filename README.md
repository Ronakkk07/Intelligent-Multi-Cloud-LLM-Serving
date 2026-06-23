# Intelligent Multi-Cloud LLM Serving — Implementation Skeleton

MSc Research Project · Ronak Rajput (x24195707) · Session 3 progress

## What this is

A runnable, cloud-free skeleton of the Predictive Router from the proposal.
It exercises the full routing pipeline and all four experimental conditions
on a simulated query stream, so the logic can be validated before spending
GPU credits. When real EKS/AKS endpoints come online, **only `telemetry.py`
changes** — everything else stays identical.
