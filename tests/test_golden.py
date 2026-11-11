"""Grader golden tests. CI runs these on every commit. If any fails, no reward number is trustworthy.

  1. reference RTL scores >= 1.0
  2. >= 90% of the first 10 mutants score < 1.0 (equivalent mutants can survive; listed in output)
  3. empty module scores 0
  4. known reward hacks score 0: early $finish, spoofed result line, shadowing RefModule
  5. constant-output DUT scores 0
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from chiprl.graders.mutate import mutants
from chiprl.graders.rtl_gen import grade_rtl
from chiprl.tasks.task import load_verilogeval_dir

FIX = Path(__file__).parent / "fixtures/verilogeval"
TASKS = sorted(p for p in FIX.iterdir() if p.is_dir())


def _ref(task):
    return task.ref_rtl.replace("RefModule", task.top)


def _empty(task) -> str:
    header = _ref(task).split(");", 1)[0] + ");"
    return header + "\nendmodule\n"


@pytest.fixture(params=TASKS, ids=lambda p: p.name)
def task(request):
    return load_verilogeval_dir(request.param)


def test_reference_scores_one(task):
    r = grade_rtl(_ref(task), task)
    assert r.stage == "ok" and r.reward is not None and r.reward >= 1.0, r


def test_mutants_score_below_one(task):
    ms = list(mutants(_ref(task)))[:10]
    assert len(ms) >= 5, "mutator found too few sites"
    survivors = []
    for name, src in ms:
        r = grade_rtl(src, task, equiv_bonus=False)
        assert r.reward is not None, (name, r.stage)
        if r.reward >= 1.0:
            survivors.append(name)
    kill = 1 - len(survivors) / len(ms)
    print(f"{task.task_id}: killed {len(ms) - len(survivors)}/{len(ms)} survivors={survivors}")
    assert kill >= 0.9, survivors


def test_empty_module_scores_zero(task):
    r = grade_rtl(_empty(task), task)
    assert r.reward == 0.0, r


def _inject(task, body: str) -> str:
    return _ref(task).replace("endmodule", body + "\nendmodule", 1)


def test_hack_early_finish(task):
    r = grade_rtl(_inject(task, "initial #100 $finish;"), task)
    assert r.reward == 0.0 and r.stage == "lint"


def test_hack_spoofed_result_line(task):
    hdr = _empty(task).replace("endmodule", 'final $display("Mismatches: 0 in 40000 samples");\nendmodule')
    assert grade_rtl(hdr, task).reward == 0.0


def test_hack_shadow_refmodule(task):
    src = _ref(task) + "\n" + _ref(task).replace(task.top, "RefModule")
    r = grade_rtl(src, task)
    assert r.reward == 0.0 and r.stage == "lint"


def test_sample_count_guard_without_policy(task, monkeypatch):
    """Defense in depth: even if the static policy misses a $finish, the sample-count gate catches it."""
    import chiprl.graders.rtl_gen as g
    monkeypatch.setattr(g, "lint", lambda *a, **k: type("L", (), {"ok": True, "infra_error": False,
                                                                      "to_dict": lambda s: {}})())
    r = grade_rtl(_inject(task, "initial #100 $finish;"), task)
    assert r.reward == 0.0 and r.stage == "samples", r


def test_constant_outputs_score_zero(task):
    outs = [(n, w) for d, n, w in task.ports if d == "output"]
    hdr = re.sub(r"\boutput\s+(reg|logic)\b", "output", _empty(task).split("endmodule")[0])
    body = "\n".join(f"assign {n} = {w}'d0;" for n, w in outs)
    r = grade_rtl(hdr + body + "\nendmodule\n", task)
    assert r.reward == 0.0, r
