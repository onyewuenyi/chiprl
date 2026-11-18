"""Durable runs: a stopped or killed run resumes to the same result an uninterrupted run gives."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("anthropic")

from chiprl.llm import evals
from chiprl.runtime import manifest

FIX = Path(__file__).parent / "fixtures/verilogeval"
TASKS = evals.load_tasks(FIX, None)
REFS = {t.spec: t.ref_rtl.replace("RefModule", t.top) for t in TASKS}


class Block(SimpleNamespace):
    def model_dump(self):
        return dict(self.__dict__)


class RefClient:
    """Answers every task with its reference; costs 1000 in / 500 out tokens per call ($0.014 on Opus 5.5)."""

    def __init__(self, fail_on: str | None = None):
        self.calls = 0
        self.fail_on = fail_on
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        spec = kw["messages"][0]["content"]
        if self.fail_on and self.fail_on in spec:
            raise ConnectionError("transient")
        return SimpleNamespace(
            content=[Block(type="text", text=json.dumps({"code": REFS[spec]}))],
            stop_reason="end_turn",
            model=kw["model"],
            usage=SimpleNamespace(
                input_tokens=1000, output_tokens=500, cache_read_input_tokens=0, cache_creation_input_tokens=0
            ),
        )


def _run(monkeypatch, client, *args):
    monkeypatch.setattr(evals, "make_client", lambda: client)
    return evals.main(["--tasks", str(FIX), "--n", "2", "--workers", "1", *args])


def _summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text())


def test_budget_stop_then_resume_matches_uninterrupted(monkeypatch, tmp_path):
    full_dir = tmp_path / "full"
    assert _run(monkeypatch, RefClient(), "--max-usd", "5", "--out", str(full_dir)) == 0
    [full] = list(full_dir.iterdir())

    part_dir = tmp_path / "part"
    assert _run(monkeypatch, RefClient(), "--max-usd", "0.03", "--out", str(part_dir)) == 3  # cap hit
    [part] = list(part_dir.iterdir())
    first = _summary(part)
    assert first["stopped"] and first["jobs_done"] < first["jobs_total"]

    client = RefClient()
    assert _run(monkeypatch, client, "--max-usd", "5", "--resume", str(part)) == 0
    assert client.calls == first["jobs_total"] - first["jobs_done"], "finished jobs must not be redone"
    resumed = _summary(part)
    assert resumed["jobs_done"] == resumed["jobs_total"] == 8
    assert resumed["metrics"] == _summary(full)["metrics"]
    assert len(resumed["sessions"]) == 2


def test_infra_rows_are_retried_on_resume(monkeypatch, tmp_path):
    bad = TASKS[0]
    assert _run(monkeypatch, RefClient(fail_on=bad.spec), "--max-usd", "5", "--out", str(tmp_path)) == 0
    [run] = list(tmp_path.iterdir())
    assert _summary(run)["metrics"]["infra_excluded"] == 2
    client = RefClient()
    assert _run(monkeypatch, client, "--max-usd", "5", "--resume", str(run)) == 0
    assert client.calls == 2
    s = _summary(run)
    assert s["metrics"]["infra_excluded"] == 0 and s["metrics"]["pass@1"]["mean"] == pytest.approx(1.0)


def test_torn_last_line_is_ignored(monkeypatch, tmp_path):
    assert _run(monkeypatch, RefClient(), "--max-usd", "5", "--out", str(tmp_path)) == 0
    [run] = list(tmp_path.iterdir())
    traj = run / manifest.TRAJ
    lines = traj.read_text().splitlines()
    traj.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][: len(lines[-1]) // 2])  # killed mid-write
    client = RefClient()
    assert _run(monkeypatch, client, "--max-usd", "5", "--resume", str(run)) == 0
    assert client.calls == 1 and _summary(run)["jobs_done"] == 8


def test_resume_refuses_a_different_grader_or_config(monkeypatch, tmp_path):
    assert _run(monkeypatch, RefClient(), "--max-usd", "5", "--out", str(tmp_path)) == 0
    [run] = list(tmp_path.iterdir())
    assert _run(monkeypatch, RefClient(), "--max-usd", "5", "--resume", str(run), "--effort", "low") == 5
    m = json.loads((run / manifest.MANIFEST).read_text())
    m["environment"]["code"] = "older-grader"
    (run / manifest.MANIFEST).write_text(json.dumps(m))
    assert _run(monkeypatch, RefClient(), "--max-usd", "5", "--resume", str(run)) == 5


def test_manifest_records_provenance(monkeypatch, tmp_path):
    assert _run(monkeypatch, RefClient(), "--max-usd", "5", "--out", str(tmp_path), "--seed", "7") == 0
    [run] = list(tmp_path.iterdir())
    m = json.loads((run / manifest.MANIFEST).read_text())
    assert m["seed"] == 7 and m["config"]["model"] == "claude-opus-5-5" and m["config"]["mode"] == "zero_shot"
    assert set(m["tasks"]) == {t.task_id for t in TASKS}
    assert m["git"]["sha"] and {"tools", "code", "env", "yosys_slang"} <= set(m["environment"])
    rows = [json.loads(x) for x in (run / manifest.TRAJ).read_text().splitlines()]
    assert len({r["job_id"] for r in rows}) == 8


def test_run_ids_are_unique():
    assert len({manifest.new_run_id("zero_shot") for _ in range(50)}) == 50


def test_unknown_model_fails_before_any_work(monkeypatch, tmp_path):
    from chiprl.llm.costs import UnknownModelPrice

    client = RefClient()
    with pytest.raises(UnknownModelPrice):
        _run(monkeypatch, client, "--max-usd", "5", "--out", str(tmp_path), "--model", "claude-opus-typo")
    assert client.calls == 0, "no API call before the price check"
    assert not tmp_path.exists() or not any(tmp_path.iterdir()), "no run dir before the price check"
