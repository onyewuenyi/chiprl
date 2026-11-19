"""Coverage of a task set over chosen axes: what a dataset, a split or a task generator contains.

    uv run python -m chiprl.tasks.coverage                        # family x split x size, VerilogEval
    uv run python -m chiprl.tasks.coverage --axes family size

A claim about "unseen designs" is only as wide as the cells the test set fills. For each axis this
reports the count per value and a balance score (normalized entropy: 1 = even, 0 = one value); over
the cross product it reports how many cells are filled, which are empty, and which are thin
(fewer than `thin` tasks, too few to report a rate with a useful interval). A generator for the
power task family (rung E5) should report the same thing over its own axes (transform, width,
workload), as rows of {axis: value}.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from .splits import load_manifest


def size_bucket(features: dict[str, Any]) -> str:
    bits = features.get("ff_bits", 0)
    if bits == 0:
        return "comb"
    return "small" if bits <= 16 else ("medium" if bits <= 128 else "large")


def balance(counts: Counter) -> float:
    n, k = sum(counts.values()), len(counts)
    if k <= 1 or n == 0:
        return 0.0 if k <= 1 else 1.0
    h = -sum(c / n * math.log(c / n) for c in counts.values() if c)
    return h / math.log(k)


def coverage(
    rows: Iterable[dict[str, Any]],
    axes: Sequence[str],
    values: dict[str, Sequence[str]] | None = None,
    thin: int = 5,
) -> dict[str, Any]:
    """values: the full value set per axis (default: values seen). Empty cells are only visible
    against the full set, so pass it when you know it (e.g. every family, every transform)."""
    rows = list(rows)
    per_axis = {a: Counter(str(r[a]) for r in rows) for a in axes}
    universe = {a: list((values or {}).get(a) or sorted(per_axis[a])) for a in axes}
    cells = Counter(tuple(str(r[a]) for a in axes) for r in rows)
    all_cells = list(itertools.product(*(universe[a] for a in axes)))
    empty = [c for c in all_cells if cells[c] == 0]
    thin_cells = [c for c in all_cells if 0 < cells[c] < thin]
    return {
        "n": len(rows),
        "axes": {a: {"counts": dict(per_axis[a]), "balance": round(balance(per_axis[a]), 3)} for a in axes},
        "cells": len(all_cells),
        "filled": len(all_cells) - len(empty),
        "fill_fraction": round((len(all_cells) - len(empty)) / len(all_cells), 3) if all_cells else 0.0,
        "empty": ["/".join(c) for c in empty],
        "thin": {"/".join(c): cells[c] for c in thin_cells},
        "thin_threshold": thin,
    }


def manifest_rows() -> list[dict[str, Any]]:
    return [
        {"task": tid, "family": r["family"], "split": r["split"], "size": size_bucket(r["features"])}
        for tid, r in load_manifest()["tasks"].items()
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", nargs="+", default=["family", "split", "size"])
    ap.add_argument("--thin", type=int, default=5)
    a = ap.parse_args(argv)
    print(json.dumps(coverage(manifest_rows(), a.axes, thin=a.thin), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
