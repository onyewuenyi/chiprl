"""Score one submission against one task. The verifier entry point of an exported task.

    python -m chiprl.verify --task <dir with prompt.txt ref.sv test.sv> --submission TopModule.sv
    python -m chiprl.verify --task ... --submission ... --out score.json

Prints one JSON object: {"score": float|null, "stage", "passed", "equivalence", "audit", "infra"}.
`score` is the correctness score of docs/REWARD.md §1; null means the machine or toolchain failed
and the attempt must be excluded, not counted as a failure (AGENTS rule 3). Exit code: 0 when a
score was produced, 2 on infra, 1 on bad arguments.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from .graders.service import grade
from .tasks.task import load_verilogeval_dir


def verify(task_dir: Path, submission: Path) -> dict:
    task = load_verilogeval_dir(task_dir)
    if (task_dir / "TASK_ID").is_file():  # exported verifier dirs are not named after the task
        task = dataclasses.replace(task, task_id=(task_dir / "TASK_ID").read_text().strip())
    if not submission.is_file():
        return {"score": 0.0, "stage": "no_submission", "passed": False, "equivalence": None,
                "audit": None, "infra": False}  # fmt: skip
    r = grade(submission.read_text(errors="replace"), task)
    eq = (r.detail.get("equiv") or {}).get("metrics", {})
    return {
        "task_id": task.task_id,
        "score": r.reward,
        "stage": r.stage,
        "passed": r.reward is not None and r.reward >= 1.0,
        "equivalence": eq.get("verdict"),
        "audit": r.detail.get("audit"),
        "infra": r.reward is None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=Path, required=True, help="directory with prompt.txt, ref.sv, test.sv")
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if not all((a.task / f).is_file() for f in ("prompt.txt", "ref.sv", "test.sv")):
        print(json.dumps({"error": f"{a.task} is not a task directory"}))
        return 1
    res = verify(a.task, a.submission)
    js = json.dumps(res)
    if a.out:
        a.out.write_text(js + "\n")
    print(js)
    return 2 if res["infra"] else 0


if __name__ == "__main__":
    sys.exit(main())
