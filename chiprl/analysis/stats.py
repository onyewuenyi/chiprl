"""Intervals and tests for eval claims. Standard library only.

Which one to use (docs/EXPERIMENTS.md R2, Reward Atlas Stage 6):
  smoke run                     no interval: it proves plumbing, not a rate
  one pass rate over n tasks    wilson()            (tasks are the unit, not samples)
  pass@k with k > 1             bootstrap_ci()      over per-task pass@k values
  two methods, same tasks       paired_bootstrap()  for the mean difference,
                                mcnemar_exact()     for pass/fail, using only discordant tasks
  before running a comparison   tasks_needed()      the n that can detect the gain you care about
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials. Behaves at k = 0 and k = n, unlike p ± z·SE."""
    if n <= 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo = 0.0 if k == 0 else max(0.0, center - half)  # exact at the edges, no float residue
    hi = 1.0 if k == n else min(1.0, center + half)
    return (lo, hi)


def bootstrap_ci(
    xs: Sequence[float], iters: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap of the mean over the given units (tasks)."""
    if not xs:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    lo = int((1 - level) / 2 * iters)
    return (means[lo], means[min(iters - 1, int((1 + level) / 2 * iters))])


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], iters: int = 5000, seed: int = 0, level: float = 0.95
) -> dict[str, float]:
    """Mean of b - a over paired units, with a percentile interval and the share of resamples <= 0.

    Resampling the per-task differences keeps the pairing: task difficulty cancels, which is why a
    paired interval is much narrower than two separate ones on the same tasks.
    """
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    d = [y - x for x, y in zip(a, b, strict=True)]
    if not d:
        return {"mean": math.nan, "lo": math.nan, "hi": math.nan, "p_le_0": math.nan, "n": 0}
    rng = random.Random(seed)
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    lo = means[int((1 - level) / 2 * iters)]
    hi = means[min(iters - 1, int((1 + level) / 2 * iters))]
    return {"mean": sum(d) / n, "lo": lo, "hi": hi, "p_le_0": sum(m <= 0 for m in means) / iters, "n": n}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b: tasks only A passed, c: tasks only B passed.

    Tasks both pass or both fail carry no information about which method is better.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def _z(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, |error| < 1.2e-9)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00]  # fmt: skip
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01]  # fmt: skip
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00]  # fmt: skip
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > 1 - 0.02425:
        return -_z(1 - p)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def tasks_needed(delta: float, discordant: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Tasks for a paired pass/fail comparison (McNemar) to detect a gain of `delta`.

    discordant: expected fraction of tasks where the two methods disagree (must exceed delta).
    Example from R2: delta 0.10, discordant 0.20 -> about 150 tasks.
    """
    if not 0 < delta < discordant <= 1:
        raise ValueError("need 0 < delta < discordant <= 1")
    za, zb = _z(1 - alpha / 2), _z(power)
    n = (za * math.sqrt(discordant) + zb * math.sqrt(discordant - delta**2)) ** 2 / delta**2
    return math.ceil(n)


def half_width(p: float, n: int, z: float = Z95) -> float:
    """Normal-approximation half-width of a pass rate: the quick planning number, z·sqrt(p(1-p)/n)."""
    return z * math.sqrt(p * (1 - p) / n)
