"""Grader throughput benchmark: graded rollouts/s, latency percentiles, per-stage time, cache hits.

    uv run python -m chiprl.runtime.bench                                  # fixtures, quick
    uv run python -m chiprl.runtime.bench --tasks data/verilogeval_v2/dataset_spec-to-rtl \
        --mutants 3 --workers 1 4 8 10 --out runs/bench.json             # the DESIGN.md numbers
    uv run --extra scale python -m chiprl.runtime.bench --backend ray

Workload: per task, the reference plus up to `--mutants` single-site mutants (a mix of passing and
failing submissions, like a policy's rollouts). Each worker count runs twice against a fresh cache:
`cold` (every job runs the tools) and `warm` (the same jobs again, all cache hits).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

from .. import cache
from ..graders.mutate import mutants
from ..graders.service import grade
from ..llm.evals import load_tasks
from .executor import imap

FIX = Path(__file__).resolve().parents[2] / "tests/fixtures/verilogeval"
STAGES = ("lint", "sim", "equiv")


def _job(args):
    task, code, equiv_bonus = args
    t0 = time.time()
    r = grade(code, task, equiv_bonus=equiv_bonus)
    stage_s = {k: (r.detail.get(k) or {}).get("elapsed_s") for k in STAGES}
    return {
        "task": task.task_id,
        "reward": r.reward,
        "stage": r.stage,
        "cache_hit": r.detail.get("cache_hit", False),
        "latency_s": time.time() - t0,
        "stage_s": stage_s,
    }


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))], 3)


def run_config(jobs, backend: str, workers: int) -> dict:
    t0 = time.time()
    out: list[dict] = [
        r.value
        for r in imap(
            _job,
            jobs,
            backend=backend,
            workers=workers,
            retries=1,
            is_infra=lambda o: o["reward"] is None,
        )
        if r.value is not None
    ]
    wall = time.time() - t0
    lat = [o["latency_s"] for o in out]
    computed = [o for o in out if not o["cache_hit"]]
    stage = {}
    for k in STAGES:
        xs = [o["stage_s"][k] for o in computed if o["stage_s"][k] is not None]
        stage[k] = {"n": len(xs), "p50": _pct(xs, 0.5), "p95": _pct(xs, 0.95), "total_s": round(sum(xs), 1)}
    return {
        "backend": backend,
        "workers": workers,
        "jobs": len(out),
        "wall_s": round(wall, 2),
        "rollouts_per_s": round(len(out) / wall, 2),
        "latency_p50_s": _pct(lat, 0.5),
        "latency_p95_s": _pct(lat, 0.95),
        "latency_p99_s": _pct(lat, 0.99),
        "latency_max_s": round(max(lat), 2) if lat else None,
        "cache_hit_rate": round(sum(o["cache_hit"] for o in out) / len(out), 3) if out else None,
        "infra_rate": round(sum(o["reward"] is None for o in out) / len(out), 4) if out else None,
        "stages": dict(Counter(o["stage"] for o in out)),
        "stage_time": stage,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, default=FIX)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--mutants", type=int, default=7, help="mutants per task besides the reference")
    ap.add_argument("--reps", type=int, default=1, help="repeat the job list (fixtures are small)")
    ap.add_argument("--backend", default="local", choices=["local", "ray"])
    ap.add_argument("--workers", type=int, nargs="+", default=[os.cpu_count() or 4])
    ap.add_argument("--no-equiv", action="store_true", help="skip the +0.1 equivalence bonus check")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    jobs = []
    for t in load_tasks(a.tasks, a.limit):
        ref = t.ref_rtl.replace("RefModule", t.top)
        jobs += [(t, ref, not a.no_equiv)] + [
            (t, m, not a.no_equiv) for _, m in list(mutants(ref))[: a.mutants]
        ]
    # Distinct submissions only within one pass; reps repeat whole passes (warm-cache traffic).
    jobs *= a.reps
    results = []
    for w in a.workers:
        with tempfile.TemporaryDirectory(prefix="chiprl_bench_") as d:
            old = cache.DB
            cache.DB = Path(d) / "results.sqlite"
            try:
                for phase in ("cold", "warm"):
                    res = {
                        "phase": phase,
                        "tasks": len({j[0].task_id for j in jobs}),
                        **run_config(jobs, a.backend, w),
                    }
                    results.append(res)
                    print(json.dumps({k: v for k, v in res.items() if k != "stage_time"}))
            finally:
                cache.DB = old
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
