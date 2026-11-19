"""Statistics and eval reports: known values, and the report refuses comparisons that mean nothing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chiprl.analysis import report, stats


def test_wilson_known_values_and_edges():
    lo, hi = stats.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    lo, hi = stats.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4), "p ± z·SE would give [0, 0] here"
    assert stats.wilson(10, 10)[1] == 1.0


def test_planning_numbers_from_the_docs():
    assert stats.half_width(0.5, 100) == pytest.approx(0.098, abs=5e-4)  # Atlas 8.4
    assert stats._z(0.975) == pytest.approx(1.959964, abs=1e-6)
    assert 145 <= stats.tasks_needed(0.10, 0.20) <= 160  # EXPERIMENTS.md R2: "about 150"
    with pytest.raises(ValueError):
        stats.tasks_needed(0.3, 0.2)


def test_mcnemar_exact():
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(0, 5) == pytest.approx(0.0625)
    assert stats.mcnemar_exact(2, 8) == pytest.approx(2 * 56 / 1024)
    assert stats.mcnemar_exact(8, 2) == stats.mcnemar_exact(2, 8)


def test_paired_bootstrap():
    a = [0.0, 1.0, 0.5, 1.0, 0.0] * 8
    assert stats.paired_bootstrap(a, a)["mean"] == 0.0
    shifted = stats.paired_bootstrap(a, [x + 1 for x in a])
    assert (
        shifted["mean"] == shifted["lo"] == shifted["hi"] == pytest.approx(1.0) and shifted["p_le_0"] == 0.0
    )
    with pytest.raises(ValueError):
        stats.paired_bootstrap([1.0], [1.0, 0.0])


def _run(tmp: Path, name: str, rewards: dict[str, list[float | None]], code: str = "abc") -> Path:
    d = tmp / name
    d.mkdir()
    m = {
        "environment": {"code": code, "tools": {"yosys": "0.70"}, "env": {}, "yosys_slang": True},
        "tasks": {t: f"h-{t}" for t in rewards},
    }
    (d / "manifest.json").write_text(json.dumps(m))
    rows = []
    for t, rs in rewards.items():
        for i, r in enumerate(rs):
            stage = "infra" if r is None else ("ok" if r >= 1 else "sim")
            rows.append({"job_id": f"{t}-{i}", "task_id": t, "sample": i, "reward": r, "stage": stage})
    (d / "trajectories.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return d


def test_single_sample_run_gets_wilson_and_failure_layers(tmp_path):
    d = _run(tmp_path, "a", {"t1": [1.1], "t2": [0.0], "t3": [None], "t4": [1.0]})
    s = report.summarize_run(d)
    assert s["tasks_scored"] == 3 and s["pass_rate"] == pytest.approx(2 / 3)
    assert s["interval"]["method"] == "wilson95" and s["interval"]["ci"] == pytest.approx(stats.wilson(2, 3))
    assert s["failures_by_layer"] == {"sim": 1, "infra": 1} and s["infra_rows"] == 1


def test_multi_sample_run_bootstraps_over_tasks(tmp_path):
    d = _run(tmp_path, "a", {"t1": [1.0, 0.0], "t2": [0.0, 0.0]})
    s = report.summarize_run(d)
    assert s["pass_rate"] == pytest.approx(0.25) and s["interval"]["method"] == "bootstrap95_over_tasks"


def test_compare_is_paired_and_refuses_a_different_grader(tmp_path):
    a = _run(tmp_path, "a", {"t1": [0.0], "t2": [1.0], "t3": [0.0], "t4": [None]})
    b = _run(tmp_path, "b", {"t1": [1.0], "t2": [1.0], "t3": [1.0], "t4": [1.0]})
    c = report.compare_runs(a, b)
    assert c["paired_tasks"] == 3, "t4 has no valid sample in A, so it is not paired"
    assert c["b_minus_a"]["mean"] == pytest.approx(2 / 3)
    assert c["mcnemar"] == {"only_a": 0, "only_b": 2, "p": 0.5} and c["regressed_tasks"] == []
    other = _run(tmp_path, "c", {"t1": [1.0]}, code="different")
    with pytest.raises(SystemExit, match=r"environment\.code"):
        report.compare_runs(a, other)


def test_layers_cover_every_grader_stage():
    for s in ("lint", "compile", "samples", "sim", "timeout", "resource", "ok", "infra", "no_submit"):
        assert report.layer(s) != "other", s
    assert report.layer("no_code:max_tokens") == "agent"
