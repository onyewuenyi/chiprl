"""Task-set coverage: counts, balance, empty and thin cells, against a known value set."""

from __future__ import annotations

import pytest

from chiprl.tasks import coverage as cov
from chiprl.tasks.families import FAMILIES


def test_cells_against_a_full_value_set():
    rows = [{"t": "a", "w": 8}] * 6 + [{"t": "a", "w": 16}] * 2 + [{"t": "b", "w": 8}] * 6
    c = cov.coverage(rows, ["t", "w"], values={"t": ["a", "b", "c"], "w": ["8", "16"]}, thin=3)
    assert c["n"] == 14 and c["cells"] == 6 and c["filled"] == 3 and c["fill_fraction"] == 0.5
    assert set(c["empty"]) == {"b/16", "c/8", "c/16"} and c["thin"] == {"a/16": 2}


def test_balance():
    from collections import Counter

    assert cov.balance(Counter({"x": 5, "y": 5})) == pytest.approx(1.0)
    assert cov.balance(Counter({"x": 10})) == 0.0
    assert 0 < cov.balance(Counter({"x": 9, "y": 1})) < 0.5


def test_size_buckets():
    assert [cov.size_bucket({"ff_bits": b}) for b in (0, 4, 64, 512)] == ["comb", "small", "medium", "large"]


def test_verilogeval_split_has_the_expected_holes():
    rows = cov.manifest_rows()
    c = cov.coverage(rows, ["family", "split"], values={"family": FAMILIES})
    assert c["n"] == 154
    # The held-out family never trains, and nothing else is out-of-distribution: by design.
    assert "fsm/train" in c["empty"] and "fsm/test_iid" in c["empty"]
    assert all(f"{f}/test_ood" in c["empty"] for f in FAMILIES if f != "fsm")
