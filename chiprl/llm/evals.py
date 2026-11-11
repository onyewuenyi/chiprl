"""Eval harness: pass@k with confidence intervals, cost, and JSONL trajectories.

    uv run --extra llm python -m chiprl.llm.evals --tasks tests/fixtures/verilogeval \
        --mode agent --n 4 --max-usd 5 --effort high

Infra errors (reward None) are excluded from the denominator and reported separately.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..tasks.task import Task, load_verilogeval_dataset, load_verilogeval_dir
from .agent import AgentConfig, run_agent, run_zero_shot
from .auth import make_client, redact
from .costs import Budget, BudgetExceeded


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
    out: dict = {"tasks": len(results), "infra_excluded": 0}
    for k in ks:
        per_task = []
        for rewards in results.values():
            valid = [r for r in rewards if r is not None]
            out["infra_excluded"] += 0 if k != ks[0] else len(rewards) - len(valid)
            if len(valid) >= k:
                per_task.append(pass_at_k(len(valid), sum(r >= 1.0 for r in valid), k))
        lo, hi = bootstrap_ci(per_task)
        out[f"pass@{k}"] = {
            "mean": sum(per_task) / len(per_task) if per_task else None,
            "ci95": [lo, hi],
            "n_tasks": len(per_task),
        }
    return out


def log_wandb(run_dir: Path, summary: dict, results: dict[str, list[float | None]]) -> None:
    """Optional experiment tracking. Same run name as the local dir, so the two stay joinable."""
    import wandb

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
    ap.add_argument("--max-usd", type=float, required=True, help="hard spend cap for this run")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases (needs `wandb login`)")
    a = ap.parse_args(argv)

    tasks = load_tasks(a.tasks, a.limit)
    cfg = AgentConfig(model=a.model, effort=a.effort)
    client = make_client()
    budget = Budget(max_usd=a.max_usd)
    run_dir = a.out / time.strftime(f"%Y%m%d-%H%M%S-{a.mode}")
    run_dir.mkdir(parents=True, exist_ok=True)
    fn = run_agent if a.mode == "agent" else run_zero_shot
    results: dict[str, list[float | None]] = {t.task_id: [] for t in tasks}
    jobs = [(t, i) for t in tasks for i in range(a.n)]
    stopped = None
    with (run_dir / "trajectories.jsonl").open("w") as fh, ThreadPoolExecutor(a.workers) as pool:
        futs = {pool.submit(fn, client, t, budget, cfg): (t, i) for t, i in jobs}
        for f in as_completed(futs):
            t, i = futs[f]
            try:
                traj = f.result()
            except BudgetExceeded as e:
                stopped = str(e)
                for g in futs:
                    g.cancel()
                continue
            traj["sample"] = i
            results[t.task_id].append(traj["reward"])
            fh.write(redact(json.dumps(traj, default=str)) + "\n")
            fh.flush()
    summary = {
        "mode": a.mode,
        "model": a.model,
        "effort": a.effort,
        "n": a.n,
        "metrics": summarize(results, a.k),
        "cost": budget.summary(),
        "stopped": stopped,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if a.wandb:
        log_wandb(run_dir, summary, results)
    print(json.dumps(summary, indent=2))
    return 0 if stopped is None else 3


if __name__ == "__main__":
    sys.exit(main())
