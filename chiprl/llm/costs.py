"""Token accounting and a hard per-run spend cap.

Prices are USD per million tokens (first-party API, Nov 2026). Update from the pricing page when
they change; the budget is a safety net, the Console workspace spend limit is the real backstop.
"""

from __future__ import annotations

import dataclasses
import threading
from typing import Any

PRICES: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"in": 4.00, "out": 20.00, "cache_read": 0.20},
    "claude-sonnet-5-5": {"in": 2.00, "out": 10.00, "cache_read": 0.10},
    "claude-haiku-5-5": {"in": 0.10, "out": 0.50, "cache_read": 0.01},
    "claude-fable-5-2": {"in": 8.00, "out": 40.00, "cache_read": 0.20},
}
CACHE_WRITE_MULT = 1.25
BATCH_DISCOUNT = 0.5


class BudgetExceeded(RuntimeError):
    pass


class UnknownModelPrice(KeyError):
    """A model id with no price would cost $0 and silently disable the spend cap."""


def price(model: str) -> dict[str, float]:
    p = PRICES.get(model)
    if p is None:
        raise UnknownModelPrice(f"no price for model {model!r}; add it to chiprl/llm/costs.py PRICES")
    return p


def worst_case_usd(
    model: str, requests: int, input_tokens: int, max_tokens: int, batch: bool = False
) -> float:
    """Upper bound on spend before sending: every request uses all its output tokens, no cache hits."""
    p = price(model)
    usd = requests * (input_tokens * p["in"] * CACHE_WRITE_MULT + max_tokens * p["out"]) / 1e6
    return usd * (BATCH_DISCOUNT if batch else 1.0)


def usage_cost(model: str, usage: Any, batch: bool = False) -> float:
    p = price(model)
    if usage is None:
        return 0.0
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    cr = getattr(usage, "cache_read_input_tokens", 0) or 0
    cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
    usd = (inp * p["in"] + out * p["out"] + cr * p["cache_read"] + cw * p["in"] * CACHE_WRITE_MULT) / 1e6
    return usd * (BATCH_DISCOUNT if batch else 1.0)


@dataclasses.dataclass
class Budget:
    """Thread-safe running total. charge() raises once the cap is crossed, so a runaway loop stops."""

    max_usd: float
    spent_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    calls: int = 0
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock, repr=False)

    def charge(self, model: str, usage: Any, batch: bool = False) -> float:
        usd = usage_cost(model, usage, batch)
        with self._lock:
            self.spent_usd += usd
            self.calls += 1
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0
            self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
            if self.spent_usd > self.max_usd:
                raise BudgetExceeded(f"spent ${self.spent_usd:.2f} > cap ${self.max_usd:.2f}")
        return usd

    def summary(self) -> dict:
        return {
            "spent_usd": round(self.spent_usd, 4),
            "max_usd": self.max_usd,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
        }
