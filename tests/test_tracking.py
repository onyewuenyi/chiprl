"""MLflow tracking: an eval run is indexed with its provenance, metrics and files."""

from __future__ import annotations

import json

import pytest

mlflow = pytest.importorskip("mlflow")

from chiprl import tracking  # noqa: E402


def test_eval_run_is_logged_with_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'm.db'}")
    run = tmp_path / "20261118-000000-zero_shot-abc123"
    run.mkdir()
    m = {"run_id": run.name, "git": {"sha": "deadbeef", "dirty": False}, "seed": 3,
         "environment": {"code": "fp", "tools": {"yosys": "0.70"}, "env": {"CHIPRL_SIM_BACKEND": ""}, "yosys_slang": True},
         "config": {"mode": "zero_shot", "model": "claude-opus-5-5"}, "tasks": {"t1": "h"}}  # fmt: skip
    summary = {"mode": "zero_shot", "metrics": {"pass@1": {"mean": 0.5, "ci95": [0.2, 0.8], "n_tasks": 2,
               "mean_infra_as_fail": 0.4}, "infra_excluded": 1}, "cost": {"spent_usd": 0.1, "calls": 3},
               "job_errors": 0, "stopped": None}  # fmt: skip
    (run / "manifest.json").write_text(json.dumps(m))
    (run / "summary.json").write_text(json.dumps(summary))
    (run / "trajectories.jsonl").write_text("{}\n")
    rid = tracking.log_eval_run(run, summary, {"t1": [1.0, None]})
    r = mlflow.get_run(rid)
    assert r.info.run_name == run.name
    assert r.data.params["git_sha"] == "deadbeef" and r.data.params["cfg.model"] == "claude-opus-5-5"
    assert r.data.params["tool.yosys"] == "0.70" and r.data.params["seed"] == "3"
    assert r.data.metrics["pass_at_1/mean"] == 0.5 and r.data.metrics["pass_at_1/ci95_hi"] == 0.8
    assert r.data.metrics["pass_at_1/mean_infra_as_fail"] == 0.4 and r.data.metrics["cost_usd"] == 0.1
    files = {a.path for a in mlflow.artifacts.list_artifacts(run_id=rid)}
    assert {"manifest.json", "summary.json", "trajectories.jsonl", "per_task.json"} <= files
