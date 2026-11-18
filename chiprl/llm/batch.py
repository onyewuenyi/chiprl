"""Best-of-N sampling through the Message Batches API (50% price, async). This is the bar RL has to
beat at matched sample count.

    uv run --extra llm python -m chiprl.llm.batch submit --tasks tests/fixtures/verilogeval --n 16
    uv run --extra llm python -m chiprl.llm.batch collect runs/batch-<id>.json --max-usd 10

Two selection rules are reported, because they answer different questions:
  * oracle  (pass@N): any sample passes the hidden grader. An upper bound, not deployable.
  * visible (BoN@N): pick the first sample that passes lint + synthesizes, using no hidden tests.
    This is what a deployed best-of-N could actually do.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from ..graders.service import grade
from ..runtime.executor import pmap
from ..tools.lint import lint
from ..tools.synth import synth
from .agent import SYSTEM, ZERO_SHOT_SCHEMA, AgentConfig
from .auth import make_client
from .costs import Budget, worst_case_usd
from .evals import load_tasks, summarize


def build_requests(tasks, n: int, cfg: AgentConfig) -> list[dict]:
    reqs = []
    for t in tasks:
        for i in range(n):
            reqs.append(
                {
                    "custom_id": f"{t.task_id}__{i}",
                    "params": {
                        "model": cfg.model,
                        "max_tokens": cfg.max_tokens,
                        "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                        "output_config": {
                            "effort": cfg.effort,
                            "format": {"type": "json_schema", "schema": ZERO_SHOT_SCHEMA},
                        },
                        "messages": [{"role": "user", "content": t.spec}],
                    },
                }
            )
    return reqs


def visible_ok(code: str, top: str) -> bool:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "dut.sv"
        p.write_text(code)
        return lint(p, top, Path(d)).ok and synth(p, top, Path(d)).ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--tasks", type=Path, required=True)
    s.add_argument("--limit", type=int)
    s.add_argument("--n", type=int, default=16)
    s.add_argument("--effort", default="high")
    s.add_argument(
        "--max-usd", type=float, required=True, help="refuse to submit if the worst-case cost exceeds this"
    )
    c = sub.add_parser("collect")
    c.add_argument("manifest", type=Path)
    c.add_argument("--max-usd", type=float, required=True)
    a = ap.parse_args(argv)
    client = make_client()

    if a.cmd == "submit":
        tasks = load_tasks(a.tasks, a.limit)
        cfg = AgentConfig(effort=a.effort)
        reqs = build_requests(tasks, a.n, cfg)
        # A batch is paid for once submitted, so the cap is checked here, not at collect time.
        # Input size is estimated at 4 characters per token, with the system prompt counted per request.
        in_tok = max((len(SYSTEM) + len(t.spec)) // 4 for t in tasks) if tasks else 0
        worst = worst_case_usd(cfg.model, len(reqs), in_tok, cfg.max_tokens, batch=True)
        if worst > a.max_usd:
            print(
                f"refusing: worst case ${worst:.2f} for {len(reqs)} requests exceeds --max-usd {a.max_usd:.2f}; "
                "lower --n or --limit, or raise the cap"
            )
            return 4
        batch = client.messages.batches.create(requests=reqs)
        manifest = Path("runs") / f"batch-{batch.id}.json"
        manifest.parent.mkdir(exist_ok=True)
        manifest.write_text(
            json.dumps(
                {
                    "batch_id": batch.id,
                    "tasks": str(a.tasks),
                    "n": a.n,
                    "model": cfg.model,
                    "max_usd": a.max_usd,
                    "worst_case_usd": round(worst, 4),
                    "created": time.time(),
                }
            )
        )
        print(f"submitted {batch.id}; manifest {manifest}")
        return 0

    m = json.loads(a.manifest.read_text())
    b = client.messages.batches.retrieve(m["batch_id"])
    if b.processing_status != "ended":
        print(f"status={b.processing_status}; try again later")
        return 1
    tasks = {t.task_id: t for t in load_tasks(Path(m["tasks"]), None)}
    budget = Budget(max_usd=a.max_usd)
    oracle: dict[str, list[float | None]] = {k: [] for k in tasks}
    visible_pick: dict[str, float | None] = {}
    samples: dict[str, list[tuple[int, str]]] = {k: [] for k in tasks}
    for r in client.messages.batches.results(m["batch_id"]):  # results arrive in any order
        tid, idx = r.custom_id.rsplit("__", 1)
        if r.result.type != "succeeded":
            oracle[tid].append(None)
            continue
        msg = r.result.message
        budget.charge(msg.model, msg.usage, batch=True)
        text = next((x.text for x in msg.content if x.type == "text"), "")
        try:
            code = json.loads(text)["code"]
        except (json.JSONDecodeError, KeyError, TypeError):
            oracle[tid].append(0.0)
            continue
        samples[tid].append((int(idx), code))
    # Grade every sample in parallel (EDA only, no API calls). Infra results get one retry.
    jobs = [(tid, idx, code) for tid, lst in samples.items() for idx, code in sorted(lst)]

    def grade_one(job):
        tid, _idx, code = job
        return grade(code, tasks[tid]).reward, visible_ok(code, tasks[tid].top)

    graded = pmap(grade_one, jobs, backend="local", retries=1, is_infra=lambda v: v[0] is None)
    for (tid, _idx, _code), (reward, visible) in zip(jobs, graded, strict=True):
        oracle[tid].append(reward)
        if tid not in visible_pick and visible:  # jobs are in sample order per task
            visible_pick[tid] = reward
    n = m["n"]
    bon = [1.0 if (visible_pick.get(t) or 0) >= 1.0 else 0.0 for t in tasks]
    out = {
        "batch_id": m["batch_id"],
        "n": n,
        "oracle": summarize(oracle, sorted({1, min(4, n), n})),
        "visible_bon": {"mean": sum(bon) / len(bon) if bon else None},
        "cost": budget.summary(),
    }
    print(json.dumps(out, indent=2))
    a.manifest.with_suffix(".results.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
