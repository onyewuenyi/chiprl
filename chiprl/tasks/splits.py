"""Train / held-out splits that mean something (docs/EXPERIMENTS.md R5).

    uv run python -m chiprl.tasks.splits build      # writes chiprl/tasks/manifests/verilogeval_v2.json
    uv run python -m chiprl.tasks.splits show

Three splits:
  test_ood   every task in the held-out families (default: fsm). Out of distribution by construction.
  test_iid   a stratified fraction of the remaining *groups*.
  train      the rest.

The unit of splitting is a group, not a task. Tasks in one exam series or with near-identical
reference RTL (fsm1 / fsm1s, dff8 / dff8r / dff8p, lemmings1..4, mux2to1 / mux2to1v) share a group,
so a held-out "new design" is never a renamed training design. Groups come from the leading name
token (an exam series such as m2014 or review2015 is one group) joined with any pair of references
whose token 5-gram Jaccard similarity is at least NEAR_DUP (references with fewer than MIN_SHINGLES
5-grams are skipped: one-line designs look alike but carry nothing to leak).

Family hold-out wins over grouping: a group can have members in train and test_ood when only some
of its members are FSMs. Those straddles are listed in the manifest so a reader can judge them.

The manifest is committed. It is a pure function of the dataset digest and the parameters below,
and tests/test_splits.py rebuilds the cheap parts to check that.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..runtime.executor import imap
from .families import FAMILIES, classify, structural_features
from .task import Task, load_verilogeval_dataset

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/verilogeval_v2/dataset_spec-to-rtl"
MANIFEST = Path(__file__).with_name("manifests") / "verilogeval_v2.json"
SPLITS = ("train", "test_iid", "test_ood")
DEFAULTS: dict[str, Any] = {"ood_families": ["fsm"], "iid_frac": 0.25, "seed": 0}
NEAR_DUP = 0.6
MIN_SHINGLES = 40  # one-line designs (assign out = in) look alike but carry nothing to leak


def name_group(task_id: str) -> str:
    """Leading token of the name after ProbNNN_: an exam year, or the first word."""
    base = task_id.split("_", 1)[1] if re.match(r"Prob\d+_", task_id) else task_id
    if m := re.match(r"(\d{4})", base):
        return "y" + m.group(1)
    m = re.match(r"[a-z]+", base.lower())
    return m.group(0) if m else base


def _shingles(rtl: str, n: int = 5) -> set[tuple[str, ...]]:
    body = re.sub(r"//[^\n]*|/\*.*?\*/", " ", rtl, flags=re.S)
    toks = re.findall(r"\w+|[^\s\w]", body.split(");", 1)[-1])
    return {tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def groups(tasks: list[Task]) -> dict[str, str]:
    """task_id -> group id (union of name groups and near-duplicate references)."""
    parent = {t.task_id: t.task_id for t in tasks}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_name: dict[str, list[str]] = defaultdict(list)
    for t in tasks:
        by_name[name_group(t.task_id)].append(t.task_id)
    for members in by_name.values():
        for m in members[1:]:
            union(members[0], m)
    sh = {t.task_id: _shingles(t.ref_rtl) for t in tasks}
    ids = sorted(sh)
    ids = [i for i in ids if len(sh[i]) >= MIN_SHINGLES]
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if jaccard(sh[a], sh[b]) >= NEAR_DUP:
                union(a, b)
    return {t.task_id: find(t.task_id) for t in tasks}


def assign(
    families: dict[str, str], group_of: dict[str, str], ood_families: list[str], iid_frac: float, seed: int
) -> dict[str, str]:
    """task_id -> split. Deterministic in its arguments."""
    split = {tid: "test_ood" for tid, f in families.items() if f in ood_families}
    rest = sorted(tid for tid in families if tid not in split)
    # Stratify groups by the family most of their (non-OOD) members have.
    members: dict[str, list[str]] = defaultdict(list)
    for tid in rest:
        members[group_of[tid]].append(tid)
    by_family: dict[str, list[str]] = defaultdict(list)
    for g, tids in sorted(members.items()):
        by_family[Counter(families[t] for t in tids).most_common(1)[0][0]].append(g)
    rng = random.Random(seed)
    for fam in sorted(by_family):
        gs = by_family[fam]
        rng.shuffle(gs)
        target = round(iid_frac * sum(len(members[g]) for g in gs))
        taken = 0
        for g in gs:
            side = "test_iid" if taken < target else "train"
            taken += len(members[g]) if side == "test_iid" else 0
            for tid in members[g]:
                split[tid] = side
    return split


def build(dataset: Path = DATASET, **params: Any) -> dict[str, Any]:
    p = {**DEFAULTS, **params}
    tasks = load_verilogeval_dataset(dataset)
    feats = {r.item.task_id: r.value for r in imap(lambda t: structural_features(t.ref_rtl, t.top), tasks)}
    missing = [tid for tid, f in feats.items() if f is None]
    if missing:
        raise RuntimeError(f"structural features failed for {missing[:5]} (is yosys on PATH?)")
    families = {t.task_id: classify(feats[t.task_id] or {}, t.spec) for t in tasks}
    group_of = groups(tasks)
    split = assign(families, group_of, p["ood_families"], p["iid_frac"], p["seed"])
    straddles = sorted(
        g for g in set(group_of.values()) if len({split[t] for t in group_of if group_of[t] == g}) > 1
    )
    return {
        "dataset": "verilogeval_v2/dataset_spec-to-rtl",
        "dataset_digest": _dataset_digest(dataset),
        "params": {**p, "near_dup_jaccard": NEAR_DUP, "min_shingles": MIN_SHINGLES},
        "straddling_groups": straddles,
        "tasks": {
            t.task_id: {
                "family": families[t.task_id],
                "group": group_of[t.task_id],
                "split": split[t.task_id],
                "content_hash": t.content_hash(),
                "features": feats[t.task_id],
            }
            for t in tasks
        },
    }


def _dataset_digest(dataset: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(dataset.iterdir()):
        h.update(f"{f.name}\0{hashlib.sha256(f.read_bytes()).hexdigest()}\n".encode())
    return h.hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    return json.loads(path.read_text())


def task_ids(split: str, path: Path = MANIFEST) -> list[str]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; one of {SPLITS}")
    return sorted(tid for tid, row in load_manifest(path)["tasks"].items() if row["split"] == split)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "show"])
    ap.add_argument("--dataset", type=Path, default=DATASET)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        m = build(a.dataset)
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
        print(f"wrote {MANIFEST}")
    m = load_manifest()
    rows = m["tasks"].values()
    print(json.dumps(m["params"]))
    print("split x family:")
    for s in SPLITS:
        c = Counter(r["family"] for r in rows if r["split"] == s)
        print(f"  {s:9s} {sum(c.values()):4d}  " + "  ".join(f"{f}={c.get(f, 0)}" for f in FAMILIES))
    print(f"groups: {len({r['group'] for r in rows})}, straddling splits: {m['straddling_groups']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
