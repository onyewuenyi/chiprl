"""Claude as a tool-using RTL engineer: the frontier baseline (and the scaffold RL later imitates).

Modes:
  zero_shot  one request, structured output {"code": ...}, no tools
  agent      plan / edit / lint / simulate / synth_report / submit loop, up to max_turns

API choices (Nov 2026):
  * claude-opus-5-5, adaptive thinking (always on) with summarized display so trajectories are
    auditable; effort set explicitly (Opus 5.5 defaults to medium).
  * server-side refusal fallbacks ("default") so a classifier false positive does not zero a task.
  * strict tool schemas; tool_choice stays auto (forced choice is rejected on Opus 5.5).
  * top-level cache_control: system prompt + tools + growing history are reused every turn.
  * append-only history: response.content is appended unchanged (thinking blocks included).
"""

from __future__ import annotations

import dataclasses
import json
import time
from typing import Any

from ..envs.rtl_env import TOOL_SPECS, RTLGenEnv
from ..graders.service import grade
from ..tasks.task import Task
from .auth import redact
from .costs import Budget

MODEL = "claude-opus-5-5"
BETAS = ["server-side-fallback-2026-07-01"]

SYSTEM = """You are a senior RTL design engineer. Implement the requested hardware as synthesizable \
SystemVerilog in a single module named TopModule with exactly the specified ports.

Rules:
- Synthesizable constructs only. No system tasks ($display, $finish, ...), no initial or final blocks, \
no `include or `timescale, and do not define modules named RefModule, tb or stimulus_gen.
- Assume all sequential logic triggers on the positive clock edge unless the spec says otherwise.
- Prefer clear, minimal logic; area and power are measured after functional correctness.

When tools are available: lint first, simulate to check behavior (calls are limited), optionally \
check synth_report, then call submit exactly once with the final complete source."""

ZERO_SHOT_SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string"}},
    "required": ["code"],
    "additionalProperties": False,
}


@dataclasses.dataclass
class AgentConfig:
    model: str = MODEL
    effort: str = "high"  # low | medium | high | xhigh | max
    max_tokens: int = 16000
    max_turns: int = 12
    max_sim_calls: int = 5
    fallbacks: bool = True


def _block_to_log(b: Any) -> dict:
    d = b.model_dump() if hasattr(b, "model_dump") else dict(b)
    if d.get("type") == "thinking":
        d.pop("signature", None)
    return json.loads(redact(json.dumps(d, default=str)))


def _create(client, cfg: AgentConfig, **kw):
    extra: dict[str, Any] = {}
    if cfg.fallbacks:
        extra = {"betas": BETAS, "fallbacks": "default"}
    return client.beta.messages.create(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        system=SYSTEM,
        thinking={"type": "adaptive", "display": "summarized"},
        cache_control={"type": "ephemeral"},
        **extra,
        **kw,
    )


def _output_config(cfg: AgentConfig, fmt: dict | None = None) -> dict:
    oc: dict[str, Any] = {"effort": cfg.effort}
    if fmt:
        oc["format"] = fmt
    return oc


def run_zero_shot(client, task: Task, budget: Budget, cfg: AgentConfig | None = None) -> dict:
    cfg = cfg or AgentConfig()
    t0 = time.time()
    resp = _create(
        client,
        cfg,
        output_config=_output_config(cfg, {"type": "json_schema", "schema": ZERO_SHOT_SCHEMA}),
        messages=[{"role": "user", "content": task.spec}],
    )
    budget.charge(resp.model, resp.usage)
    traj = {
        "task_id": task.task_id,
        "mode": "zero_shot",
        "model": resp.model,
        "stop_reason": resp.stop_reason,
        "events": [_block_to_log(b) for b in resp.content],
    }
    code = None
    if resp.stop_reason == "end_turn":
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            code = json.loads(text)["code"]
        except (json.JSONDecodeError, KeyError, TypeError):
            code = None
    g = grade(code, task) if code else None
    traj.update(
        reward=g.reward if g else 0.0,
        stage=g.stage if g else f"no_code:{resp.stop_reason}",
        code=code,
        elapsed_s=round(time.time() - t0, 2),
    )
    return traj


def run_agent(client, task: Task, budget: Budget, cfg: AgentConfig | None = None) -> dict:
    cfg = cfg or AgentConfig()
    env = RTLGenEnv(task, max_sim_calls=cfg.max_sim_calls)
    tools = [{**t, "strict": True} for t in TOOL_SPECS]
    messages: list[dict[str, Any]] = [{"role": "user", "content": task.spec}]
    events: list[dict] = []
    t0 = time.time()
    stop = "max_turns"
    nudged = False
    models_used: set[str] = set()
    for turn in range(cfg.max_turns):
        resp = _create(client, cfg, output_config=_output_config(cfg), tools=tools, messages=messages)
        budget.charge(resp.model, resp.usage)
        models_used.add(resp.model)
        events.append(
            {
                "turn": turn,
                "stop_reason": resp.stop_reason,
                "content": [_block_to_log(b) for b in resp.content],
            }
        )
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "refusal":
            stop = "refusal"
            break
        if resp.stop_reason == "max_tokens":
            stop = "max_tokens"
            break
        if resp.stop_reason == "pause_turn":
            continue
        calls = [b for b in resp.content if b.type == "tool_use"]
        if not calls:
            if env.submitted is None and not nudged:
                nudged = True
                messages.append({"role": "user", "content": "Call submit with your final TopModule source."})
                continue
            stop = "end_turn"
            break
        results = []
        for c in calls:  # all results go back in ONE user message (keeps parallel tool use healthy)
            out = env.call(c.name, c.input if isinstance(c.input, dict) else {})
            events.append({"turn": turn, "tool": c.name, "result": out})
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": c.id,
                    "content": json.dumps(out),
                    **({"is_error": True} if "error" in out else {}),
                }
            )
        messages.append({"role": "user", "content": results})
        if env.submitted is not None:
            stop = "submitted"
            break
    sub = env.submitted or {}
    return {
        "task_id": task.task_id,
        "mode": "agent",
        "models": sorted(models_used),
        "stop": stop,
        "reward": sub.get("reward", 0.0),
        "stage": sub.get("stage", "no_submit"),
        "code": sub.get("code"),
        "turns": len([e for e in events if "stop_reason" in e]),
        "sim_calls": env.sim_calls,
        "elapsed_s": round(time.time() - t0, 2),
        "events": events,
    }
