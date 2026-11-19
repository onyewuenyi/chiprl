"""Export tasks as self-contained agent tasks: container + instruction + verifier + score.

    uv run python -m chiprl.tasks.export --split test_ood --out runs/export/test_ood
    uv run python -m chiprl.tasks.export --tasks tests/fixtures/verilogeval --out runs/export/smoke

This is the shape agentic coding benchmarks use (a task is an environment, a prompt, a hidden
verifier and a score), so a chip task can run in the same kind of harness. Layout:

    <out>/
      Dockerfile            shared environment image: pinned OSS CAD Suite + chiprl at this commit
      FORMAT.md             what the files mean and how a harness runs a task
      index.json            every task, with its split and family
      <task_id>/
        task.json           metadata, limits, how to verify, how to score
        instruction.md      what the agent sees: the spec, the interface, where to write the answer
        verifier/           GRADER-ONLY: prompt.txt, ref.sv, test.sv, TASK_ID, verify.sh
                            mount or copy it in only after the agent has finished (AGENTS rule 7)

The format is chiprl's own. A specific harness (Terminal-Bench, Harbor, Inspect, ...) needs a thin
adapter from task.json; check that harness's current task spec rather than assuming this matches.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from .. import cache
from ..runtime.manifest import git_info
from ..tools._run import ROOT
from .splits import SPLITS, load_manifest, task_ids
from .task import Task, load_verilogeval_dataset, load_verilogeval_dir

FORMAT_VERSION = 1
SUBMISSION = "/workspace/TopModule.sv"
DATASET = ROOT / "data/verilogeval_v2/dataset_spec-to-rtl"

INSTRUCTION = """# {task_id}

{spec}

## What to deliver

Write your design to `{submission}`: one SystemVerilog file defining `module {top}` with exactly
the interface above.

- Synthesizable constructs only. No system tasks (`$display`, `$finish`, ...), no `initial` or
  `final` blocks, no `` `include `` or `` `timescale ``, and do not define modules named
  `RefModule`, `tb` or `stimulus_gen`.
- `iverilog`, `verilator` and `yosys` are installed. Write your own testbench if you want one;
  the grading testbench is not in this environment.
- When you are done, leave the file in place. It is graded once, after you finish.
"""

VERIFY_SH = """#!/bin/sh
# Grader-only. Run after the agent finishes, with this directory mounted at /verifier.
# Writes /logs/score.json; exit 2 means infra failure (exclude the attempt, do not score it 0).
set -u
mkdir -p /logs
exec python3 -m chiprl.verify --task /verifier --submission "{submission}" --out /logs/score.json
"""

DOCKERFILE = """# Environment for exported chiprl tasks. Build from the chiprl repo root:
#   docker build -t chiprl-task-env:{tag} -f <export dir>/Dockerfile .
# The agent runs in this image with no network; the grader mounts <task>/verifier at /verifier.
FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
ARG OSS_CAD_TAG={oss_tag}
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl python3 \\
 && rm -rf /var/lib/apt/lists/*
RUN arch=$(uname -m | sed 's/x86_64/x64/; s/aarch64/arm64/') \\
 && curl -fL "https://github.com/YosysHQ/oss-cad-suite-build/releases/download/$OSS_CAD_TAG/oss-cad-suite-linux-$arch-$(echo $OSS_CAD_TAG | tr -d -).tgz" \\
    | tar -xz -C /opt
ENV PATH=/opt/oss-cad-suite/bin:$PATH
COPY chiprl /opt/chiprl/chiprl
COPY third_party /opt/chiprl/third_party
ENV PYTHONPATH=/opt/chiprl CHIPRL_CACHE=/tmp/chiprl-cache.sqlite
RUN useradd -m agent && mkdir -p /workspace /logs && chown agent /workspace /logs
USER agent
WORKDIR /workspace
"""

FORMAT_MD = """# chiprl task format v{version}

A task is four things: an environment (`Dockerfile`, shared by all tasks), an instruction
(`<task>/instruction.md`), a verifier (`<task>/verifier/`, grader-only) and a score.

Run a task:

1. Start a container from the environment image with no network. Put `instruction.md` in front of
   the agent. The agent writes `{submission}`.
2. When the agent stops, mount `<task>/verifier` at `/verifier` and run `sh /verifier/verify.sh`.
3. Read `/logs/score.json`:
   `{{"score": 1.1|1.0|0.0|null, "stage": ..., "passed": bool, "equivalence": ..., "audit": ..., "infra": bool}}`

Score: the correctness score in chiprl's `docs/REWARD.md` §1. `passed` is `score >= 1.0`; 1.1 means
the design was also formally proven equivalent. `null` (exit code 2) means the machine or toolchain
failed: exclude that attempt, never count it as a failure. A submission that hangs or crashes the
simulator scores 0, not null.

Never put `verifier/` in the agent's container before it finishes: it holds the hidden testbench
and the reference design. `task.json` records the split (train / test_iid / test_ood) so a harness
can keep held-out tasks out of training.
"""


def _instruction(t: Task) -> str:
    return INSTRUCTION.format(task_id=t.task_id, spec=t.spec.strip(), submission=SUBMISSION, top=t.top)


def export_task(t: Task, out: Path, row: dict[str, Any] | None, provenance: dict[str, Any]) -> dict[str, Any]:
    d = out / t.task_id
    v = d / "verifier"
    v.mkdir(parents=True, exist_ok=True)
    (d / "instruction.md").write_text(_instruction(t))
    (v / "prompt.txt").write_text(t.spec)
    (v / "ref.sv").write_text(t.ref_rtl)
    (v / "test.sv").write_text(t.testbench)
    (v / "TASK_ID").write_text(t.task_id + "\n")
    (v / "verify.sh").write_text(VERIFY_SH.format(submission=SUBMISSION))
    (v / "verify.sh").chmod(0o755)
    meta = {
        "format_version": FORMAT_VERSION,
        "id": t.task_id,
        "source": t.source,
        "family": t.family,
        "split": (row or {}).get("split"),
        "group": (row or {}).get("group"),
        "content_hash": t.content_hash(),
        "instruction": "instruction.md",
        "submission": SUBMISSION,
        "environment": {"image": "../Dockerfile", "network": False},
        "verifier": {
            "dir": "verifier",
            "mount": "/verifier",
            "command": "sh /verifier/verify.sh",
            "result": "/logs/score.json",
            "infra_exit_code": 2,
        },
        "score": {"field": "score", "pass_threshold": 1.0, "max": 1.1, "null_means": "infra: exclude"},
        "limits": {"agent_timeout_s": 1800, "verifier_timeout_s": 600},
        "provenance": provenance,
    }
    (d / "task.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"id": t.task_id, "split": meta["split"], "family": t.family, "dir": t.task_id}


def export(tasks: list[Task], out: Path) -> dict[str, Any]:
    try:
        rows = load_manifest()["tasks"]
    except OSError:
        rows = {}
    env = cache.environment()
    prov = {"git": git_info(), "grader_code": env["code"], "tools": env["tools"]}
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    index = [export_task(t, out, rows.get(t.task_id), prov) for t in tasks]
    oss_tag = (ROOT / ".oss-cad-suite-version").read_text().strip()
    tag = (prov["git"]["sha"] or "dev")[:12]
    (out / "Dockerfile").write_text(DOCKERFILE.format(oss_tag=oss_tag, tag=tag))
    (out / "FORMAT.md").write_text(FORMAT_MD.format(version=FORMAT_VERSION, submission=SUBMISSION))
    summary = {"format_version": FORMAT_VERSION, "tasks": index, "provenance": prov}
    (out / "index.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--split", choices=SPLITS)
    src.add_argument("--tasks", type=Path, help="a fixture directory of task dirs")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.split:
        keep = set(task_ids(a.split))
        tasks = [t for t in load_verilogeval_dataset(DATASET) if t.task_id in keep]
    else:
        tasks = [
            load_verilogeval_dir(d) for d in sorted(p for p in a.tasks.iterdir() if (p / "ref.sv").exists())
        ]
    s = export(tasks, a.out)
    print(f"exported {len(s['tasks'])} tasks to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
