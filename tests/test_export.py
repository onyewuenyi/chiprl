"""Exported tasks: the agent-visible part never contains the hidden testbench or reference, and the
verifier scores a submission the same way the grader does (AGENTS rules 3 and 7)."""

from __future__ import annotations

import json
from pathlib import Path

from chiprl import verify
from chiprl.tasks import export
from chiprl.tasks.task import load_verilogeval_dir

FIX = Path(__file__).parent / "fixtures/verilogeval"


def _export(tmp_path):
    tasks = [load_verilogeval_dir(d) for d in sorted(FIX.iterdir()) if d.is_dir()]
    return tasks, export.export(tasks, tmp_path / "out")


def test_layout_and_metadata(tmp_path):
    tasks, s = _export(tmp_path)
    out = tmp_path / "out"
    assert {"Dockerfile", "FORMAT.md", "index.json"} <= {p.name for p in out.iterdir()}
    assert len(s["tasks"]) == len(tasks)
    for t in tasks:
        meta = json.loads((out / t.task_id / "task.json").read_text())
        assert meta["id"] == t.task_id and meta["content_hash"] == t.content_hash()
        assert meta["split"] in ("train", "test_iid", "test_ood") and meta["family"] == t.family
        assert meta["score"]["null_means"].startswith("infra")
        assert (out / t.task_id / "verifier" / "verify.sh").stat().st_mode & 0o111
    assert (Path(__file__).parents[1] / ".oss-cad-suite-version").read_text().strip() in (
        out / "Dockerfile"
    ).read_text()


def test_agent_visible_files_hold_no_hidden_material(tmp_path):
    tasks, _ = _export(tmp_path)
    out = tmp_path / "out"
    for t in tasks:
        visible = [out / t.task_id / "instruction.md", out / t.task_id / "task.json", out / "Dockerfile"]
        text = "\n".join(p.read_text() for p in visible)
        body = t.ref_rtl.split(");", 1)[1].strip()
        assert body[:80] not in text, "reference body leaked"
        assert "RefModule" not in text.replace("`RefModule`", "") and "stimulus_gen (" not in text
        assert "Mismatches:" not in text and "module tb" not in text
        assert t.spec.strip()[:200] in (out / t.task_id / "instruction.md").read_text()


def test_verifier_matches_the_grader(tmp_path):
    _tasks, _ = _export(tmp_path)
    tdir = tmp_path / "out" / "Prob075_counter_2bc" / "verifier"
    task = load_verilogeval_dir(tdir)
    ref = task.ref_rtl.replace("RefModule", task.top)
    good, bad = tmp_path / "good.sv", tmp_path / "bad.sv"
    good.write_text(ref)
    bad.write_text(ref.replace("state <= 1;", "state <= 2;"))
    g = verify.verify(tdir, good)
    assert g["score"] == 1.1 and g["passed"] and g["equivalence"] == "proven" and not g["infra"]
    assert g["task_id"] == "Prob075_counter_2bc", "the id comes from TASK_ID, not the directory name"
    b = verify.verify(tdir, bad)
    assert b["score"] == 0.0 and not b["passed"] and b["stage"] == "sim"
    missing = verify.verify(tdir, tmp_path / "nothing.sv")
    assert missing["score"] == 0.0 and missing["stage"] == "no_submission"


def test_verify_cli_writes_score_and_signals_infra(tmp_path, monkeypatch, capsys):
    _tasks, _ = _export(tmp_path)
    tdir = tmp_path / "out" / "Prob009_popcount3" / "verifier"
    sub = tmp_path / "sub.sv"
    sub.write_text(load_verilogeval_dir(tdir).ref_rtl.replace("RefModule", "TopModule"))
    out = tmp_path / "score.json"
    assert verify.main(["--task", str(tdir), "--submission", str(sub), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["passed"] is True
    from chiprl.graders.rtl_gen import R

    monkeypatch.setattr(verify, "grade", lambda *a, **k: R(None, "infra", {}))
    assert verify.main(["--task", str(tdir), "--submission", str(sub)]) == 2
    assert verify.main(["--task", str(tmp_path), "--submission", str(sub)]) == 1
