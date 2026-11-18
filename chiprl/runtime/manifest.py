"""Run provenance and job identity.

A run directory holds:
  manifest.json      written once at creation: what was run, on what code and toolchain
  trajectories.jsonl one row per finished job attempt, appended; the last row per job_id wins
  summary.json       recomputed from trajectories.jsonl at the end of every session

job_id is a hash of everything that defines a job (task content, policy config, sample index, seed),
so a resumed run skips exactly the jobs that already have a durable row, and rerunning a job always
writes to the same id.
"""

from __future__ import annotations

import hashlib
import json
import platform
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .. import cache
from ..tasks.task import Task
from ..tools._run import ROOT

MANIFEST = "manifest.json"
TRAJ = "trajectories.jsonl"


def _git(*args: str) -> str:
    try:
        p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return p.stdout.strip() if p.returncode == 0 else ""


def git_info() -> dict[str, Any]:
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return {"sha": _git("rev-parse", "HEAD") or None, "dirty": bool(dirty)}


def new_run_id(mode: str) -> str:
    """Unique even for two runs started in the same second."""
    return time.strftime("%Y%m%d-%H%M%S") + f"-{mode}-{secrets.token_hex(3)}"


def job_id(task: Task, sample: int, config: dict[str, Any], seed: int) -> str:
    blob = json.dumps(
        {
            "task": task.task_id,
            "task_hash": task.content_hash(),
            "sample": sample,
            "config": config,
            "seed": seed,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def build(run_id: str, config: dict[str, Any], tasks: list[Task], seed: int) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created": time.time(),
        "git": git_info(),
        "environment": cache.environment(),
        "config": config,
        "seed": seed,
        "tasks": {t.task_id: t.content_hash() for t in tasks},
        "python": sys.version.split()[0],
        "platform": f"{platform.system()}-{platform.machine()}",
    }


def compatible(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Reasons a run cannot be resumed with the new settings; empty if it can.

    Mixing grader code, toolchains, task content or policy config inside one run would make its
    numbers meaningless, so all of them must match. Git SHA alone may differ (docs-only commits).
    """
    reasons = []
    for field in ("config", "seed", "tasks"):
        if old.get(field) != new.get(field):
            reasons.append(f"{field} differs")
    oe, ne = old.get("environment", {}), new.get("environment", {})
    for field in ("code", "tools", "env", "yosys_slang"):
        if oe.get(field) != ne.get(field):
            reasons.append(f"environment.{field} differs")
    return reasons


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Last row per job_id. A torn final line (process killed mid-write) is ignored."""
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "job_id" in row:
            rows[row["job_id"]] = row
    return rows


def repair_tail(path: Path) -> None:
    """Make sure the next append starts on a fresh line after a torn write."""
    if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("a") as fh:
            fh.write("\n")
