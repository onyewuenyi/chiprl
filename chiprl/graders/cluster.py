"""Group candidates by behavior, AlphaCode-style: sample many, filter, cluster survivors by what they do.

A behavior signature is a hash of a candidate's outputs on a fixed random stimulus (differential
simulation of the candidate against itself, `tools/diffsim.py`). It uses only the task's interface,
never the hidden testbench or reference (AGENTS rule 7), so a deployed selector can use it.

What clustering is for, and what it is not for:
  * selection: among candidates that pass visible checks, take one from the largest cluster
    (`pick`). Many samples agreeing on a behavior is evidence for that behavior.
  * diversity: the number of distinct behaviors in a GRPO group. A group with one behavior has
    identical rewards and zero advantage, so this predicts wasted rollouts before grading.
  * NOT proof sharing. Equal signatures mean equal outputs on the sampled stimulus only. One
    member's `proven` says nothing about another's, so every candidate keeps its own equivalence
    check. Power is also per candidate: equivalent RTL can synthesize to different power.
"""

from __future__ import annotations

import tempfile
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from ..runtime.executor import pmap
from ..tasks.task import Task
from ..tools.diffsim import diffsim

SIG_SEEDS = (101, 102)
SIG_CYCLES = 500


def signature(code: str, top: str) -> str | None:
    """Behavior signature, or None if the candidate cannot be simulated."""
    with tempfile.TemporaryDirectory(prefix="chiprl_sig_") as d:
        p = Path(d) / "dut.sv"
        p.write_text(code)
        r = diffsim(p, code, top, Path(d), cycles=SIG_CYCLES, seeds=SIG_SEEDS, tag="sig")
    return r.metrics.get("signature") if r.ok else None


def cluster(codes: Sequence[str], top: str, workers: int | None = None) -> list[list[int]]:
    """Indices grouped by behavior, largest group first (ties: earliest index). Unsimulatable
    candidates are singletons."""
    sigs = pmap(lambda c: signature(c, top), codes, backend="local", workers=workers)
    groups: dict[str, list[int]] = defaultdict(list)
    singles: list[list[int]] = []
    for i, s in enumerate(sigs):
        if s is None:
            singles.append([i])
        else:
            groups[s].append(i)
    return sorted([*groups.values(), *singles], key=lambda g: (-len(g), g[0]))


def pick(task: Task, codes: Sequence[str], visible_ok: Callable[[str], bool] | None = None) -> int | None:
    """Index of the candidate to submit: first member of the largest cluster among those passing
    the visible checks. None if nothing passes."""
    keep = [i for i, c in enumerate(codes) if visible_ok is None or visible_ok(c)]
    if not keep:
        return None
    clusters = cluster([codes[i] for i in keep], task.top)
    return keep[clusters[0][0]]


def behaviors(codes: Sequence[str], top: str) -> int:
    """Distinct behaviors in a group (unsimulatable candidates count one each)."""
    return len(cluster(codes, top))
