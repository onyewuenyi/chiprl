"""Grader throughput benchmark: graded rollouts/sec on the fixtures (reference + mutants).

uv run python -m chiprl.runtime.bench --backend local --reps 4
uv run --extra scale python -m chiprl.runtime.bench --backend ray --reps 4
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from ..graders.mutate import mutants
from ..graders.rtl_gen import grade_rtl
from ..tasks.task import load_verilogeval_dir
from .executor import pmap

FIX = Path(__file__).resolve().parents[2] / "tests/fixtures/verilogeval"


def _job(args):
    task, code = args
    t0 = time.time()
    r = grade_rtl(code, task, equiv_bonus=False)
    return {"task": task.task_id, "reward": r.reward, "stage": r.stage, "latency_s": time.time() - t0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="local", choices=["local", "ray"])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--workers", type=int)
    a = ap.parse_args(argv)
    jobs = []
    for d in sorted(p for p in FIX.iterdir() if p.is_dir()):
        t = load_verilogeval_dir(d)
        ref = t.ref_rtl.replace("RefModule", t.top)
        jobs += [(t, ref)] + [(t, m) for _, m in list(mutants(ref))[:7]]
    jobs *= a.reps
    t0 = time.time()
    out = pmap(_job, jobs, backend=a.backend, workers=a.workers)
    wall = time.time() - t0
    lat = sorted(o["latency_s"] for o in out)
    res = {
        "backend": a.backend,
        "jobs": len(out),
        "wall_s": round(wall, 2),
        "rollouts_per_s": round(len(out) / wall, 2),
        "latency_p50_s": round(statistics.median(lat), 3),
        "latency_p95_s": round(lat[int(0.95 * (len(lat) - 1))], 3),
        "infra_errors": sum(o["reward"] is None for o in out),
    }
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
