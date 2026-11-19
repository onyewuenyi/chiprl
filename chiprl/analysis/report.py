"""Eval reports from run directories: pass rate with the right interval, failures by cause, and a
paired comparison of two runs.

    uv run python -m chiprl.analysis.report runs/<run>              # one run
    uv run python -m chiprl.analysis.report runs/<A> runs/<B>       # B vs A, paired on shared tasks

A task's score is its mean pass rate over valid samples (infra rows excluded). With one sample per
task the pass rate gets a Wilson interval; with more, a bootstrap over tasks. Two runs are compared
only on tasks both completed, and only if they ran on the same grader code, toolchain and task
content (the manifests say so); the policy config may differ, that is the point of the comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..runtime import manifest
from .stats import bootstrap_ci, mcnemar_exact, paired_bootstrap, wilson

# Grader stage -> the layer a failure belongs to (Atlas Project 6(c), Project 8 attribution table).
LAYER = {
    "ok": "pass",
    "lint": "lint",
    "compile": "lint",
    "sim": "sim",
    "samples": "sim",
    "timeout": "sim",
    "resource": "sim",
    "infra": "infra",
    "no_submit": "agent",
}


def layer(stage: str) -> str:
    return "agent" if stage.startswith("no_code") else LAYER.get(stage, "other")


def per_task(rows: dict[str, dict[str, Any]]) -> dict[str, list[float | None]]:
    out: dict[str, list[float | None]] = defaultdict(list)
    for r in sorted(rows.values(), key=lambda r: (r["task_id"], r.get("sample", 0))):
        out[r["task_id"]].append(r.get("reward"))
    return out


def task_pass_rate(rewards: list[float | None]) -> float | None:
    valid = [r for r in rewards if r is not None]
    return sum(r >= 1.0 for r in valid) / len(valid) if valid else None


def summarize_run(run_dir: Path) -> dict[str, Any]:
    rows = manifest.load_rows(run_dir / manifest.TRAJ)
    tasks = per_task(rows)
    rates = {t: task_pass_rate(rs) for t, rs in tasks.items()}
    scored = {t: p for t, p in rates.items() if p is not None}
    n_valid = {t: len([r for r in rs if r is not None]) for t, rs in tasks.items()}
    single = all(n_valid[t] == 1 for t in scored)
    mean = sum(scored.values()) / len(scored) if scored else None
    if single:
        k = sum(round(p) for p in scored.values())
        interval = {"method": "wilson95", "ci": wilson(k, len(scored))}
    else:
        interval = {"method": "bootstrap95_over_tasks", "ci": bootstrap_ci(list(scored.values()))}
    stages = Counter(r.get("stage", "?") for r in rows.values())
    layers = Counter(layer(s) for s in stages.elements())
    return {
        "run": run_dir.name,
        "tasks": len(tasks),
        "tasks_scored": len(scored),
        "rows": len(rows),
        "pass_rate": mean,
        "interval": interval,
        "infra_rows": layers.get("infra", 0),
        "failures_by_layer": {k: v for k, v in layers.most_common() if k != "pass"},
        "failures_by_stage": {k: v for k, v in stages.most_common() if k != "ok"},
    }


def compare_runs(a_dir: Path, b_dir: Path) -> dict[str, Any]:
    ma = json.loads((a_dir / manifest.MANIFEST).read_text())
    mb = json.loads((b_dir / manifest.MANIFEST).read_text())
    for field in ("code", "tools", "env", "yosys_slang"):
        if ma["environment"].get(field) != mb["environment"].get(field):
            raise SystemExit(f"refusing to compare: environment.{field} differs between the runs")
    shared_hash = {t for t in set(ma["tasks"]) & set(mb["tasks"]) if ma["tasks"][t] == mb["tasks"][t]}
    ra = {t: task_pass_rate(r) for t, r in per_task(manifest.load_rows(a_dir / manifest.TRAJ)).items()}
    rb = {t: task_pass_rate(r) for t, r in per_task(manifest.load_rows(b_dir / manifest.TRAJ)).items()}
    common = sorted(t for t in shared_hash if ra.get(t) is not None and rb.get(t) is not None)
    a = [ra[t] or 0.0 for t in common]
    b = [rb[t] or 0.0 for t in common]
    pb = paired_bootstrap(a, b)
    out: dict[str, Any] = {"a": a_dir.name, "b": b_dir.name, "paired_tasks": len(common), "b_minus_a": pb}
    if all(x in (0.0, 1.0) for x in a + b):
        only_a = sum(1 for x, y in zip(a, b, strict=True) if x == 1.0 and y == 0.0)
        only_b = sum(1 for x, y in zip(a, b, strict=True) if x == 0.0 and y == 1.0)
        out["mcnemar"] = {"only_a": only_a, "only_b": only_b, "p": mcnemar_exact(only_a, only_b)}
    out["regressed_tasks"] = [t for t, x, y in zip(common, a, b, strict=True) if y < x]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    a = ap.parse_args(argv)
    if len(a.runs) == 1:
        print(json.dumps(summarize_run(a.runs[0]), indent=2))
    elif len(a.runs) == 2:
        print(json.dumps({"a": summarize_run(a.runs[0]), "b": summarize_run(a.runs[1]),
                          "comparison": compare_runs(*a.runs)}, indent=2))  # fmt: skip
    else:
        ap.error("one run, or two to compare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
