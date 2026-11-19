"""Experiment tracking with MLflow (optional `scale` extra; the local path works without it).

Where runs go: $MLFLOW_TRACKING_URI if set, else a SQLite store at runs/mlflow.db (gitignored).
Browse with:  uv run --extra scale mlflow ui --backend-store-uri sqlite:///runs/mlflow.db

What a run records, so a number can be traced to exactly what produced it:
  params    identity (run id, git SHA, dirty flag, seed, split) and setup (mode, model, effort,
            samples per task, grader code fingerprint, tool versions, simulator backend)
  metrics   pass@k mean with its 95% interval, mean with infra counted as failure, infra rows,
            job errors, spend and API calls
  artifacts manifest.json, summary.json, trajectories.jsonl, and a per-task table
The run directory under runs/ stays the source of truth; MLflow is an index over it.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from .tools._run import ROOT

EXPERIMENT = "chiprl"


def setup(experiment: str = EXPERIMENT) -> Any:
    """Import and configure mlflow. Returns the module."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow  # ty: ignore[import-not-found]  # optional `scale` extra

    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        (ROOT / "runs").mkdir(exist_ok=True)
        uri = f"sqlite:///{ROOT / 'runs' / 'mlflow.db'}"
        os.environ["MLFLOW_TRACKING_URI"] = uri  # so transformers' MLflowCallback uses the same store
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", experiment)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return mlflow


def manifest_params(m: dict[str, Any]) -> dict[str, Any]:
    """Flatten a run manifest (chiprl/runtime/manifest.py) into MLflow params (strings, <= 6000 chars)."""
    env = m.get("environment", {})
    p: dict[str, Any] = {
        "run_id": m.get("run_id"),
        "git_sha": (m.get("git") or {}).get("sha"),
        "git_dirty": (m.get("git") or {}).get("dirty"),
        "seed": m.get("seed"),
        "grader_code": env.get("code"),
        "yosys_slang": env.get("yosys_slang"),
        "n_tasks": len(m.get("tasks", {})),
    }
    p.update({f"cfg.{k}": v for k, v in (m.get("config") or {}).items()})
    p.update({f"tool.{k}": v for k, v in (env.get("tools") or {}).items()})
    p.update({f"env.{k}": v for k, v in (env.get("env") or {}).items()})
    return {k: str(v)[:6000] for k, v in p.items()}


def log_eval_run(run_dir: Path, summary: dict[str, Any], results: dict[str, list[float | None]]) -> str:
    """One MLflow run per eval run directory, named after it so the two stay joinable. Returns the run id."""
    mlflow = setup()
    manifest = (
        json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
    )
    with mlflow.start_run(run_name=run_dir.name) as run:
        mlflow.log_params(manifest_params(manifest))
        mlflow.set_tags(
            {"kind": "eval", "mode": summary.get("mode", ""), "stopped": str(summary.get("stopped"))}
        )
        metrics: dict[str, float] = {}
        for k_raw, v in (summary.get("metrics") or {}).items():
            key = k_raw.replace("@", "_at_")  # MLflow metric names allow no '@': pass@1 -> pass_at_1
            if isinstance(v, dict) and v.get("mean") is not None:
                metrics[f"{key}/mean"] = v["mean"]
                lo, hi = v.get("ci95") or (None, None)
                if lo is not None and not math.isnan(lo):
                    metrics[f"{key}/ci95_lo"], metrics[f"{key}/ci95_hi"] = float(lo), float(hi or lo)
                if v.get("mean_infra_as_fail") is not None:
                    metrics[f"{key}/mean_infra_as_fail"] = v["mean_infra_as_fail"]
        metrics["infra_excluded"] = (summary.get("metrics") or {}).get("infra_excluded", 0)
        metrics["job_errors"] = summary.get("job_errors", 0)
        cost = summary.get("cost") or {}
        metrics["cost_usd"] = cost.get("spent_usd", 0.0)
        metrics["api_calls"] = cost.get("calls", 0)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.log_dict(
            {
                "rows": [
                    {"task_id": t, "sample": i, "reward": r}
                    for t, rs in results.items()
                    for i, r in enumerate(rs)
                ]
            },
            "per_task.json",
        )
        for name in ("manifest.json", "summary.json", "trajectories.jsonl"):
            if (run_dir / name).exists():
                mlflow.log_artifact(str(run_dir / name))
        return run.info.run_id
