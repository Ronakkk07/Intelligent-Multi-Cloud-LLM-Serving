"""
Dataset Loader
--------------
Loads real queries from two HuggingFace datasets:

  1. mlabonne/lmsys-arena-human-preference-55k-sharegpt
     Short and medium conversational prompts from LMSYS Chatbot Arena.
     Each row contains two model responses; we extract the first human turn.

  2. m-a-p/CodeFeedback-Filtered-Instruction
     Long, token-heavy coding instructions (naturally >500 tokens).

Replaces synthetic_query() from run_comparison.py.
Both datasets are Apache 2.0 licensed and publicly available.
"""

from __future__ import annotations
import logging
import random
from typing import Iterator

from router.query_analyzer import QueryAnalyzer, QueryFeatures, LengthBucket

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _first_human_turn(row: dict) -> str | None:
    """
    Extract the first human/user turn from a ShareGPT-format row.
    Handles both conversation_a/conversation_b (LMSYS arena) and
    the generic conversations field.
    """
    for field in ("conversation_a", "conversation_b", "conversations"):
        conv = row.get(field)
        if not (conv and isinstance(conv, list)):
            continue
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            role = (turn.get("from") or turn.get("role") or "").lower()
            if role in ("human", "user"):
                text = (turn.get("value") or turn.get("content") or "").strip()
                if text:
                    return text
    return None


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_sharegpt_queries(
    n: int = 400,
    seed: int = 42,
    buckets: set[LengthBucket] | None = None,
    min_tokens: int = 10,
    max_tokens: int = 600,
) -> list[QueryFeatures]:
    """
    Stream the LMSYS ShareGPT dataset and return up to *n* QueryFeatures
    whose bucket is in *buckets* (default: SHORT + MEDIUM).

    Uses streaming so the full 55k dataset is never loaded into memory.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the datasets package: pip install datasets")

    if buckets is None:
        buckets = {LengthBucket.SHORT, LengthBucket.MEDIUM}

    qa = QueryAnalyzer(tokenizer="cl100k_base")
    rng = random.Random(seed)
    results: list[QueryFeatures] = []

    log.info("Streaming mlabonne/lmsys-arena-human-preference-55k-sharegpt …")
    ds = load_dataset(
        "mlabonne/lmsys-arena-human-preference-55k-sharegpt",
        split="train",
        streaming=True,
        trust_remote_code=True,
    ).shuffle(seed=seed, buffer_size=5_000)

    for row in ds:
        if len(results) >= n:
            break
        text = _first_human_turn(row)
        if not text:
            continue
        f = qa.analyze(text)
        if f.bucket in buckets and min_tokens <= f.input_tokens <= max_tokens:
            results.append(f)

    rng.shuffle(results)
    log.info("Loaded %d ShareGPT queries (buckets=%s)", len(results),
             {b.value for b in buckets})
    return results[:n]


def load_codefeedback_queries(
    n: int = 100,
    seed: int = 42,
    min_tokens: int = 80,
) -> list[QueryFeatures]:
    """
    Stream CodeFeedback-Filtered-Instruction and return up to *n* QueryFeatures.
    Code prompts are naturally long (many exceed 500 tokens), giving the
    router's LONG-bucket path real data to work with.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the datasets package: pip install datasets")

    qa = QueryAnalyzer(tokenizer="cl100k_base")
    rng = random.Random(seed)
    results: list[QueryFeatures] = []

    log.info("Streaming m-a-p/CodeFeedback-Filtered-Instruction …")
    ds = load_dataset(
        "m-a-p/CodeFeedback-Filtered-Instruction",
        split="train",
        streaming=True,
        trust_remote_code=True,
    ).shuffle(seed=seed, buffer_size=2_000)

    for row in ds:
        if len(results) >= n:
            break
        text = (row.get("query") or row.get("instruction") or "").strip()
        if not text:
            continue
        f = qa.analyze(text)
        if f.input_tokens >= min_tokens:
            results.append(f)

    rng.shuffle(results)
    log.info("Loaded %d CodeFeedback queries", len(results))
    return results[:n]


def load_experiment_queries(
    n_total: int = 500,
    seed: int = 42,
    # Distribution across buckets: 50% short, 35% medium, 15% long.
    short_frac: float = 0.50,
    medium_frac: float = 0.35,
    long_frac: float = 0.15,
) -> list[QueryFeatures]:
    n_short  = int(n_total * short_frac)
    n_medium = int(n_total * medium_frac)
    n_long   = n_total - n_short - n_medium

    log.info("Loading experiment queries: %d short, %d medium, %d long",
             n_short, n_medium, n_long)

    short_q  = load_sharegpt_queries(n=n_short,  seed=seed,
                                      buckets={LengthBucket.SHORT})
    medium_q = load_sharegpt_queries(n=n_medium, seed=seed + 1,
                                      buckets={LengthBucket.MEDIUM})
    long_q   = load_codefeedback_queries(n=n_long, seed=seed)

    all_q = short_q + medium_q + long_q
    random.Random(seed).shuffle(all_q)

    log.info(
        "Dataset ready: %d short | %d medium | %d long | %d total",
        len(short_q), len(medium_q), len(long_q), len(all_q),
    )
    return all_q
