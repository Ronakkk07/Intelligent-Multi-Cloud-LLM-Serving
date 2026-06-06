"""
Query Analyzer
--------------
Takes an incoming inference request and produces the features the
Predictive Router needs: estimated input tokens and a length bucket
(short / medium / long). The bucket drives both the cost estimate
(cost scales with tokens) and the model-selection prior.

This mirrors the bucketing described in the proposal:
  short  : < 100 tokens
  medium : 100-500 tokens
  long   : > 500 tokens
"""

from dataclasses import dataclass
from enum import Enum


class LengthBucket(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


@dataclass
class QueryFeatures:
    text: str
    input_tokens: int
    bucket: LengthBucket


class QueryAnalyzer:
    """
    Token counting. Uses tiktoken (cl100k_base, same BPE as Mistral/Llama)
    when available; falls back to a word-count approximation so the code
    runs without the optional tiktoken install.

    tokenizer: tiktoken encoding name, e.g. "cl100k_base", or None for approx.
    """

    WORDS_TO_TOKENS = 1.3   # fallback approximation

    def __init__(self, tokenizer: str | None = None) -> None:
        self._enc = None
        if tokenizer is not None:
            try:
                import tiktoken
                self._enc = tiktoken.get_encoding(tokenizer)
            except ImportError:
                pass   # tiktoken not installed — fall back to approximation
            except Exception:
                pass

    def _count_tokens(self, text: str) -> int:
        if self._enc is not None:
            return max(1, len(self._enc.encode(text)))
        words = len(text.split())
        return max(1, int(words * self.WORDS_TO_TOKENS))

    def _bucket(self, tokens: int) -> LengthBucket:
        if tokens < 100:
            return LengthBucket.SHORT
        if tokens <= 500:
            return LengthBucket.MEDIUM
        return LengthBucket.LONG

    def analyze(self, text: str) -> QueryFeatures:
        tokens = self._count_tokens(text)
        return QueryFeatures(text=text, input_tokens=tokens, bucket=self._bucket(tokens))


if __name__ == "__main__":
    qa = QueryAnalyzer()
    samples = [
        "What's the capital of France?",
        "Explain how a transformer attention mechanism works " * 8,
        "Write a complete Python implementation of a B-tree with insert, "
        "delete, and range query, then explain the complexity of each. " * 12,
    ]
    for s in samples:
        f = qa.analyze(s)
        print(f"{f.bucket.value:7s} | {f.input_tokens:5d} tokens | {s[:50]}...")
