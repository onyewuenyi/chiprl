"""Eval harness: pass@k with confidence intervals, cost, and JSONL trajectories.

    uv run --extra llm python -m chiprl.llm.evals --tasks tests/fixtures/verilogeval \
        --mode agent --n 4 --max-usd 5 --effort high

Infra errors (reward None) are excluded from the denominator and reported separately.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

from ..runtime import manifest
from ..runtime.executor import imap
from ..tasks.task import Task, load_verilogeval_dataset, load_verilogeval_dir
from .agent import AgentConfig, run_agent, run_zero_shot
from .auth import make_client, redact
from .costs import Budget, BudgetExceeded, price


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator (Chen et al. 2021): 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def bootstrap_ci(per_task: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not per_task:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(per_task) for _ in per_task) / len(per_task) for _ in range(iters))
    return (means[int(0.025 * iters)], means[int(0.975 * iters)])


def summarize(results: dict[str, list[float | None]], ks: list[int]) -> dict:
    """pass@k over tasks. Infra rows (None) are excluded from `mean`.

    `mean_infra_as_fail` counts them as failures instead. It is the floor a method cannot fall
    below by getting rows excluded; a large gap between the two means the infra rate, not the
    model, is driving the headline number.
    """
    out: dict = {"tasks": len(results), "infra_excluded": 0}
    for k in ks:
        per_task, floor = [], []
        for rewards in results.values():
            valid = [r for r in rewards if r is not None]
            out["infra_excluded"] += 0 if k != ks[0] else len(rewards) - len(valid)
            passes = sum(r >= 1.0 for r in valid)
            if len(valid) >= k:
                per_task.append(pass_at_k(len(valid), passes, k))
            if len(rewards) >= k:
                floor.append(pass_at_k(len(rewards), passes, k))
        lo, hi = bootstrap_ci(per_task)
        out[f"pass@{k}"] = {
            "mean": sum(per_task) / len(per_task) if per_task else None,
            "ci95": [lo, hi],
            "n_tasks": len(per_task),
            "mean_infra_as_fail": sum(floor) / len(floor) if floor else None,
        }
    return out


def log_wandb(run_dir: Path, summary: dict, results: dict[str, list[float | None]]) -> None:
    """Optional experiment tracking. Same run name as the local dir, so the two stay joinable."""
    import wandb  # ty: ignore[import-not-found]  # optional `scale` extra

    run = wandb.init(
        project="chiprl",
        name=run_dir.name,
        job_type="eval",
        config={k: summary[k] for k in ("mode", "model", "effort", "n")},
    )
    flat = {f"{k}/mean": v["mean"] for k, v in summary["metrics"].items() if isinstance(v, dict)}
    run.log({**flat, "cost/usd": summary["cost"]["spent_usd"]})
    table = wandb.Table(columns=["task_id", "sample", "reward"])
    for tid, rewards in results.items():
        for i, r in enumerate(rewards):
            table.add_data(tid, i, r)
    run.log({"per_task": table})
    art = wandb.Artifact(f"traj-{run_dir.name}", type="trajectories")
    art.add_file(str(run_dir / "trajectories.jsonl"))
    run.log_artifact(art)
    run.finish()


def load_tasks(path: Path, limit: int | None) -> list[Task]:
    if any(path.glob("*_prompt.txt")):
        return load_verilogeval_dataset(path, limit)
    dirs = sorted(p for p in path.iterdir() if (p / "prompt.txt").exists())
    return [load_verilogeval_dir(d) for d in dirs[:limit]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--mode", choices=["zero_shot", "agent"], default="zero_shot")
    ap.add_argument("--n", type=int, default=1, help="samples per task")
    ap.add_argument("--k", type=int, nargs="+", default=[1])
    ap.add_argument("--model", default=AgentConfig.model)
    ap.add_argument("--effort", default="high")
    ap.add_argument(
        "--seed", type=int, default=0, help="part of each job's identity; recorded in the manifest"
    )
    ap.add_argument("--max-usd", type=float, required=True, help="hard spend cap for this session")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument(
        "--resume", type=Path, help="run dir to continue: finished jobs are skipped, infra rows rerun"
    )
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases (needs `wandb login`)")
    a = ap.parse_args(argv)

    tasks = load_tasks(a.tasks, a.limit)
    cfg = AgentConfig(model=a.model, effort=a.effort)
    config = {"mode": a.mode, **dataclasses.asdict(cfg)}
    price(cfg.model)  # an unknown model would disable the cap: fail before any work
    if a.resume:
        run_dir = a.resume
        old = json.loads((run_dir / manifest.MANIFEST).read_text())
        new = manifest.build(old["run_id"], config, tasks, a.seed)
        if reasons := manifest.compatible(old, new):
            print(f"cannot resume {run_dir}: " + "; ".join(reasons))
            return 5
    else:
        run_dir = a.out / manifest.new_run_id(a.mode)
        run_dir.mkdir(parents=True, exist_ok=False)
        m = manifest.build(run_dir.name, config, tasks, a.seed)
        (run_dir / manifest.MANIFEST).write_text(json.dumps(m, indent=2))

    traj_path = run_dir / manifest.TRAJ
    manifest.repair_tail(traj_path)
    done = {jid for jid, row in manifest.load_rows(traj_path).items() if row.get("reward") is not None}
    jobs = [
        (t, i, jid)
        for t in tasks
        for i in range(a.n)
        if (jid := manifest.job_id(t, i, config, a.seed)) not in done
    ]
    client = make_client()
    budget = Budget(max_usd=a.max_usd)
    fn = run_agent if a.mode == "agent" else run_zero_shot
    stopped = None
    errors = 0

    def one(job):
        t, _i, _jid = job
        return fn(client, t, budget, cfg)

    # API calls stay on local threads (AGENTS rule 12). A job that raises becomes an infra row and
    # the run continues; only the spend cap stops it, and closing the stream cancels unstarted jobs.
    with traj_path.open("a") as fh:
        stream = imap(one, jobs, backend="local", workers=a.workers, max_in_flight=a.workers)
        for r in stream:
            t, i, jid = r.item
            if isinstance(r.error, BudgetExceeded):
                stopped = str(r.error)
                stream.close()
                break
            if r.error is not None:
                errors += 1
                traj: dict[str, Any] = {
                    "task_id": t.task_id,
                    "mode": a.mode,
                    "reward": None,
                    "stage": "infra",
                    "error": f"{type(r.error).__name__}: {r.error}"[:500],
                }
            else:
                assert r.value is not None
                traj = r.value
            traj.update(sample=i, job_id=jid)
            fh.write(redact(json.dumps(traj, default=str)) + "\n")
            fh.flush()

    # The summary is a pure function of the trajectory file, so a resumed run and an uninterrupted
    # one produce the same numbers.
    rows = manifest.load_rows(traj_path)
    results: dict[str, list[float | None]] = {t.task_id: [] for t in tasks}
    for row in sorted(rows.values(), key=lambda r: (r["task_id"], r["sample"])):
        if row["task_id"] in results:
            results[row["task_id"]].append(row["reward"])
    prev = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    session: dict[str, Any] = {"cost": budget.summary(), "stopped": stopped, "job_errors": errors}
    sessions: list[dict[str, Any]] = [*prev.get("sessions", []), session]
    summary = {
        "run_id": run_dir.name,
        "mode": a.mode,
        "model": a.model,
        "effort": a.effort,
        "n": a.n,
        "jobs_total": len(tasks) * a.n,
        "jobs_done": sum(r.get("reward") is not None for r in rows.values()),
        "metrics": summarize(results, a.k),
        "cost": {
            "spent_usd": round(sum(s["cost"]["spent_usd"] for s in sessions), 4),
            "calls": sum(s["cost"]["calls"] for s in sessions),
        },
        "stopped": stopped,
        "job_errors": errors,
        "sessions": sessions,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if a.wandb:
        log_wandb(run_dir, summary, results)
    print(json.dumps(summary, indent=2))
    return 0 if stopped is None else 3


if __name__ == "__main__":
    sys.exit(main())
