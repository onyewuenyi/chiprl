"""Behavior clustering: reference-free, groups identical behavior across different text, and picks
the majority behavior. It must never be used to share proofs (see graders/cluster.py)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from chiprl.graders import cluster as cl
from chiprl.tasks.task import load_verilogeval_dir

FIX = Path(__file__).parent / "fixtures/verilogeval"
TASK = load_verilogeval_dir(FIX / "Prob035_count1to10")
REF = TASK.ref_rtl.replace("RefModule", TASK.top)
SAME = "// rewritten\n" + REF.replace("q+1", "q + 4'd1")
WRONG = REF.replace("q == 10", "q == 9")
WRONG_TOO = "/* same wrong behavior */\n" + WRONG.replace("q+1", "1 + q")
BROKEN = "module TopModule(input clk; endmodule"


def test_groups_by_behavior_not_text():
    groups = cl.cluster([REF, WRONG, SAME, WRONG_TOO, BROKEN], TASK.top)
    assert sorted(map(sorted, groups)) == [[0, 2], [1, 3], [4]]


def test_signature_uses_no_hidden_information():
    hidden_free = dataclasses.replace(TASK, ref_rtl="", testbench="")
    assert cl.signature(REF, hidden_free.top) == cl.signature(SAME, TASK.top) is not None


def test_pick_takes_the_largest_cluster_among_visible_passes():
    codes = [WRONG, REF, SAME, BROKEN]
    assert cl.pick(TASK, codes) in (1, 2)
    assert cl.pick(TASK, codes, visible_ok=lambda c: c is WRONG) == 0
    assert cl.pick(TASK, codes, visible_ok=lambda c: False) is None


def test_behaviors_counts_distinct_ones():
    assert cl.behaviors([REF, SAME, SAME], TASK.top) == 1
    assert cl.behaviors([REF, WRONG, WRONG_TOO], TASK.top) == 2
