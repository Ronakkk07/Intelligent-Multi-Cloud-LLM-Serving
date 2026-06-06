"""
Locust Load Test
----------------
Simulates 2x / 5x / 10x burst request rates against the router microservice
to validate H2 (P95 latency <20% degradation at 5x load) and H3 (≥99.5%
completion rate).

The router service must be running before starting Locust:
  uvicorn router.router_service:app --host 0.0.0.0 --port 8080

Usage:
  # Interactive UI (open http://localhost:8089)
  locust -f load_tests/locustfile.py --host http://localhost:8080

  # Headless burst profiles (from Section 3.2)
  locust -f load_tests/locustfile.py --host http://localhost:8080 \
         --headless --users 10 --spawn-rate 2 --run-time 5m   # 1x baseline
  locust -f load_tests/locustfile.py --host http://localhost:8080 \
         --headless --users 50 --spawn-rate 10 --run-time 5m  # 5x burst
  locust -f load_tests/locustfile.py --host http://localhost:8080 \
         --headless --users 100 --spawn-rate 20 --run-time 5m # 10x burst
"""

from __future__ import annotations
import random
import time

from locust import HttpUser, TaskSet, between, task, events

# ---------------------------------------------------------------------------
# Sample queries — realistic short / medium / long distribution
# Weights in the @task decorators: 5 short : 3 medium : 2 long = 50/30/20%
# ---------------------------------------------------------------------------
SHORT_QUERIES = [
    "What is Kubernetes?",
    "Define LLM inference.",
    "What is a REST API?",
    "Explain microservices briefly.",
    "What is cloud computing?",
    "What does vLLM stand for?",
    "What is a GPU?",
    "Define token in NLP.",
]

MEDIUM_QUERIES = [
    "Explain how transformer attention mechanisms work and why they're important for modern NLP models.",
    "Compare SQL and NoSQL databases. When should you choose one over the other?",
    "Describe the CAP theorem and its real-world implications for distributed systems.",
    "What is the difference between supervised and unsupervised machine learning? Provide examples.",
    "Explain Docker containerization and how it compares to virtual machines in terms of isolation and overhead.",
    "How does Kubernetes handle pod scheduling across multiple nodes? What factors does the scheduler consider?",
    "What is the role of a service mesh like Istio in a microservices architecture?",
]

LONG_QUERIES = [
    (
        "Write a comprehensive Python implementation of a distributed rate limiter using Redis "
        "with a sliding window algorithm. The implementation should support multiple rate limit "
        "tiers (free, pro, enterprise), handle race conditions using Lua scripts, and provide "
        "graceful degradation when Redis is unavailable. Include unit tests covering edge cases "
        "such as boundary conditions, concurrent requests, and Redis failure. Explain the time "
        "complexity of each operation and how this could be extended to work with Redis Cluster "
        "for horizontal scalability."
    ),
    (
        "Design a fault-tolerant event-driven microservices architecture for an e-commerce platform "
        "that processes 100,000 orders per day. Include: (1) service decomposition with clear bounded "
        "contexts, (2) message broker selection and justification between Apache Kafka and RabbitMQ, "
        "(3) saga pattern implementation for distributed transactions across order, payment, and "
        "inventory services, (4) circuit breaker pattern using Resilience4j, (5) observability "
        "strategy with distributed tracing using OpenTelemetry, and (6) how to handle the dual-write "
        "problem between your database and message broker without losing consistency."
    ),
    (
        "Implement a complete B-tree data structure in Python that supports insert, delete, search, "
        "and range queries. The implementation should handle all B-tree invariants including node "
        "splits during insertion and node merges/redistributions during deletion. Include a "
        "visualization function that prints the tree structure. Write comprehensive unit tests "
        "covering: insertion of duplicate keys, deletion of leaf and internal nodes, range queries "
        "spanning multiple levels, and stress tests with 10,000+ random operations. Analyse the "
        "time and space complexity of each operation with Big-O notation."
    ),
]


# ---------------------------------------------------------------------------
# Locust user
# ---------------------------------------------------------------------------

class RouterUser(HttpUser):
    """
    Simulates a single client sending queries to the router service.
    Task weights produce the 50/35/15 bucket distribution from Section 3.2.
    """
    wait_time = between(0.05, 0.5)   # ~2-20 req/s per user at baseline

    def on_start(self):
        self._rng = random.Random()

    # ---- tasks (weights: 5 short, 3 medium, 2 long) ----------------------

    @task(5)
    def short_query(self):
        self._route(self._rng.choice(SHORT_QUERIES), "short")

    @task(3)
    def medium_query(self):
        self._route(self._rng.choice(MEDIUM_QUERIES), "medium")

    @task(2)
    def long_query(self):
        self._route(self._rng.choice(LONG_QUERIES), "long")

    # ----------------------------------------------------------------------

    def _route(self, text: str, bucket: str) -> None:
        with self.client.post(
            "/v1/route",
            json={"query": text},
            name=f"/v1/route [{bucket}]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 503:
                resp.failure(f"No healthy endpoint: {resp.text[:120]}")
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:120]}")


# ---------------------------------------------------------------------------
# Custom Locust events — print H2/H3 summary at the end of each run
# ---------------------------------------------------------------------------

@events.quitting.add_listener
def on_quit(environment, **kwargs):
    stats = environment.runner.stats
    total = stats.total
    if total.num_requests == 0:
        return

    p95_ms    = total.get_response_time_percentile(0.95)
    fail_rate = total.num_failures / total.num_requests * 100
    rps       = total.current_rps

    print("\n" + "=" * 50)
    print("  Burst Test Summary")
    print(f"  Requests : {total.num_requests}")
    print(f"  RPS      : {rps:.1f}")
    print(f"  P95 (ms) : {p95_ms:.0f}")
    print(f"  Fail rate: {fail_rate:.2f}%")
    print(f"  H3 (≥99.5% completion): {'PASSED ✓' if fail_rate <= 0.5 else 'NOT MET ✗'}")
    print("=" * 50)
