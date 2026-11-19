"""Splits mean something (docs/EXPERIMENTS.md R5): no leakage between train and held-out, families
come from structure, and the committed manifest is what the code and the pinned dataset produce."""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

from chiprl.tasks import splits
from chiprl.tasks.families import FAMILIES, classify, structural_features
from chiprl.tasks.task import load_verilogeval_dataset, load_verilogeval_dir

M = splits.load_manifest()
ROWS = M["tasks"]
FIX = Path(__file__).parent / "fixtures/verilogeval"
HAVE_DATA = splits.DATASET.exists()


def test_manifest_covers_every_task_once():
    assert len(ROWS) == 155
    assert Counter(r["split"] for r in ROWS.values()).keys() == set(splits.SPLITS)
    assert all(r["family"] in FAMILIES for r in ROWS.values())


def test_held_out_family_never_trains():
    ood = set(M["params"]["ood_families"])
    for tid, r in ROWS.items():
        assert (r["family"] in ood) == (r["split"] == "test_ood"), tid


def test_no_group_straddles_train_and_iid_test():
    by_group: dict[str, set[str]] = {}
    for r in ROWS.values():
        by_group.setdefault(r["group"], set()).add(r["split"])
    for g, sides in by_group.items():
        assert not {"train", "test_iid"} <= sides, g
    straddles = sorted(g for g, sides in by_group.items() if len(sides) > 1)
    assert straddles == M["straddling_groups"], "every straddle (all via test_ood) is listed"


def test_split_sizes_support_the_planned_claims():
    c = Counter(r["split"] for r in ROWS.values())
    # Atlas Project 6(c): >= 50 held-out designs; Project 8: >= 30 held-out tasks
    assert c["test_iid"] + c["test_ood"] >= 50 and c["test_ood"] >= 30 and c["test_iid"] >= 30


def test_assignment_is_deterministic_and_reproduces_the_manifest():
    fams = {t: r["family"] for t, r in ROWS.items()}
    grp = {t: r["group"] for t, r in ROWS.items()}
    p = M["params"]
    got = splits.assign(fams, grp, p["ood_families"], p["iid_frac"], p["seed"])
    assert got == {t: r["split"] for t, r in ROWS.items()}
    assert splits.assign(fams, grp, p["ood_families"], p["iid_frac"], p["seed"] + 1) != got


def test_name_groups():
    assert splits.name_group("Prob109_fsm1") == splits.name_group("Prob111_fsm2s") == "fsm"
    assert splits.name_group("Prob034_dff8") == splits.name_group("Prob047_dff8ar") == "dff"
    assert splits.name_group("Prob138_2012_q2fsm") == splits.name_group("Prob091_2012_q2b") == "y2012"
    assert splits.name_group("Prob002_m2014_q4i") == "m"


def test_fixture_families_match_structure():
    for d in sorted(FIX.iterdir()):
        t = load_verilogeval_dir(d)
        f = structural_features(t.ref_rtl, t.top)
        assert f is not None
        assert classify(f, t.spec) == t.family == ROWS[t.task_id]["family"], t.task_id
        assert t.content_hash() == ROWS[t.task_id]["content_hash"]


def test_fsm_needs_an_explicit_state_machine_spec():
    seq = {"ff_bits": 8, "arith": 0, "mux": 1, "cmp": 0, "cells": 3}
    assert classify(seq, "Each cell's next state is the XOR of its neighbours.") == "seq_register"
    assert classify(seq, "Implement this Moore state machine.") == "fsm"
    assert classify({**seq, "ff_bits": 0}, "Build the next-state logic of this FSM.") == "fsm"


@pytest.mark.skipif(not HAVE_DATA, reason="run scripts/fetch_verilogeval.py")
def test_manifest_matches_the_pinned_dataset():
    spec = importlib.util.spec_from_file_location(
        "fetch", Path(__file__).parents[1] / "scripts/fetch_verilogeval.py"
    )
    assert spec and spec.loader
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)
    assert M["dataset_digest"] == fetch.DIGEST
    tasks = load_verilogeval_dataset(splits.DATASET)
    assert {t.task_id: t.content_hash() for t in tasks} == {t: r["content_hash"] for t, r in ROWS.items()}
    assert splits.groups(tasks) == {t: r["group"] for t, r in ROWS.items()}


@pytest.mark.skipif(not HAVE_DATA, reason="run scripts/fetch_verilogeval.py")
def test_no_near_duplicate_reference_across_train_and_held_out():
    tasks = {t.task_id: t for t in load_verilogeval_dataset(splits.DATASET)}
    sh = {tid: splits._shingles(t.ref_rtl) for tid, t in tasks.items()}
    big = [t for t in sh if len(sh[t]) >= splits.MIN_SHINGLES]
    train = [t for t in big if ROWS[t]["split"] == "train"]
    held = [t for t in big if ROWS[t]["split"] == "test_iid"]
    worst = max((splits.jaccard(sh[a], sh[b]), a, b) for a in train for b in held)
    assert worst[0] < splits.NEAR_DUP, worst


def test_eval_loader_filters_by_split():
    pytest.importorskip("anthropic")
    from chiprl.llm.evals import load_tasks

    ids = {t.task_id for t in load_tasks(FIX, None, "test_iid")}
    assert ids == {"Prob009_popcount3", "Prob035_count1to10"}
    assert {t.task_id for t in load_tasks(FIX, None, "train")} == {"Prob068_countbcd", "Prob075_counter_2bc"}
