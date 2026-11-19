"""GRPO on RTL generation: TRL GRPOTrainer + a LoRA adapter, rewarded by chiprl's grader.

    # sanity run on a Mac (CPU, tiny model, a few steps; downloads ~10 MB)
    uv run --extra train python -m chiprl.train.grpo --model trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 \
        --tasks tests/fixtures/verilogeval --max-steps 2 --num-generations 2 --max-completion-length 64 --cpu

    # a real run on a GPU box (rung E10), HF generation first; add --use-vllm once rollouts dominate
    uv run --extra train --extra scale python -m chiprl.train.grpo --split train --max-steps 300 --mlflow

Design (docs/PLAN.md Phase 4, docs/REWARD.md):
  * Reward = correctness score from `graders.service.grade` (cached, single-flight), graded in
    parallel with one retry for infra results. Default `binary`: 1.0 if the hidden testbench passes,
    else 0.0. `bonus` keeps the +0.1 for proven equivalence.
  * Infra failures return None. TRL >= 1.18 marks a completion whose rewards are all None as
    unscorable: it is left out of the group baseline and its advantage is forced to 0, so it moves
    nothing (checked in tests/test_train.py). Never 0, which would teach the policy that infra is
    its fault (AGENTS rule 3).
  * Train on the `train` split only. Held-out splits are refused, so no held-out task ever trains.
  * beta = 0 (no reference model), DAPO loss and per-group scaling: TRL 1.18 defaults, recorded.
  * Generation: Hugging Face `generate` by default; `--use-vllm` (colocate or server) when rollout
    time dominates. The reward function does not care which.
  * Provenance: a run manifest (git SHA, grader fingerprint, tool versions, split, seed, config)
    is written next to the checkpoints and logged to MLflow with TRL's metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..graders.service import grade
from ..runtime import manifest
from ..runtime.executor import pmap
from ..tasks.task import Task

SYSTEM = """You are a senior RTL design engineer. Implement the requested hardware as synthesizable \
SystemVerilog in a single module named TopModule with exactly the specified ports. No system tasks, \
no initial or final blocks, no `include or `timescale, and do not define modules named RefModule, tb \
or stimulus_gen. Reply with the complete module in one ```verilog code block."""

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # same base family as VeriReason / VERIRL: comparable
_FENCE = re.compile(r"```(?:verilog|systemverilog|sv|v)?\s*\n(.*?)```", re.S | re.I)
_MODULE = re.compile(r"\bmodule\s+TopModule\b.*\bendmodule\b", re.S)


def extract_code(text: str) -> str | None:
    """The last fenced code block that defines TopModule, else a bare `module TopModule ... endmodule`."""
    blocks = [b for b in _FENCE.findall(text) if re.search(r"\bmodule\s+TopModule\b", b)]
    if blocks:
        return blocks[-1].strip() + "\n"
    m = _MODULE.search(text)
    return m.group(0) + "\n" if m else None


def completion_text(c: Any) -> str:
    """TRL passes plain strings (standard format) or message lists (conversational format)."""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(
            m.get("content", "")
            for m in c
            if isinstance(m, dict) and m.get("role", "assistant") == "assistant"
        )
    return str(c)


class ChipRLReward:
    """TRL reward function. Called with completions plus the dataset's `task_id` column."""

    __name__ = "chiprl_correctness"  # TRL names metrics after the reward function

    def __init__(self, tasks: dict[str, Task], mode: str = "binary", workers: int | None = None,
                 grade_fn: Callable[..., Any] = grade):  # fmt: skip
        if mode not in ("binary", "bonus"):
            raise ValueError("mode must be 'binary' or 'bonus'")
        self.tasks, self.mode, self.workers, self.grade_fn = tasks, mode, workers, grade_fn

    def _score(self, job: tuple[str, str | None]) -> float | None:
        task_id, code = job
        if code is None:
            return 0.0  # the model produced no module: its fault, a real 0
        r = self.grade_fn(code, self.tasks[task_id])
        if r.reward is None:
            return None
        return float(r.reward >= 1.0) if self.mode == "binary" else float(r.reward)

    def __call__(
        self, completions: Sequence[Any], task_id: Sequence[str], **kwargs: Any
    ) -> list[float | None]:
        codes = [extract_code(completion_text(c)) for c in completions]
        rewards = pmap(
            self._score,
            list(zip(task_id, codes, strict=True)),
            backend="local",
            workers=self.workers,
            retries=1,
            is_infra=lambda v: v is None,
        )
        log_metric = kwargs.get("log_metric")
        if callable(log_metric) and rewards:
            n = len(rewards)
            log_metric("chiprl/infra_frac", sum(r is None for r in rewards) / n)
            log_metric("chiprl/no_code_frac", sum(c is None for c in codes) / n)
            log_metric("chiprl/pass_rate", sum(1 for r in rewards if r is not None and r >= 1.0) / n)
        return rewards


def build_dataset(tasks: Sequence[Task]) -> Any:
    from datasets import Dataset  # ty: ignore[import-not-found]  # optional `train` extra

    return Dataset.from_list(
        [
            {
                "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": t.spec}],
                "task_id": t.task_id,
            }
            for t in tasks
        ]
    )


def load_train_tasks(split: str | None, tasks_dir: Path | None, limit: int | None) -> list[Task]:
    from ..llm.evals import load_tasks
    from ..tasks.splits import DATASET

    if split and split != "train":
        raise SystemExit(
            f"refusing to train on '{split}': held-out tasks must never train (docs/EXPERIMENTS.md R5)"
        )
    return load_tasks(tasks_dir or DATASET, limit, split)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--split", default="train", help="only 'train' is accepted")
    src.add_argument("--tasks", type=Path, help="a fixture directory instead of the split (smoke runs)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--reward", choices=["binary", "bonus"], default="binary")
    ap.add_argument("--num-generations", type=int, default=8, help="group size G")
    ap.add_argument("--per-device-batch", type=int, default=8, help="completions per device per step")
    ap.add_argument("--max-completion-length", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-vllm", action="store_true", help="vLLM rollouts (needs vllm on a CUDA box)")
    ap.add_argument("--vllm-mode", choices=["colocate", "server"], default="colocate")
    ap.add_argument("--cpu", action="store_true", help="force CPU (Mac sanity runs)")
    ap.add_argument("--mlflow", action="store_true", help="log to MLflow (needs the `scale` extra)")
    ap.add_argument("--out", type=Path, default=Path("runs/train"))
    a = ap.parse_args(argv)

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    from peft import LoraConfig  # ty: ignore[import-not-found]  # optional `train` extra
    from trl import GRPOConfig, GRPOTrainer  # ty: ignore[import-not-found]

    tasks = load_train_tasks(None if a.tasks else a.split, a.tasks, a.limit)
    if not tasks:
        raise SystemExit("no tasks")
    run_id = manifest.new_run_id("grpo")
    out = a.out / run_id
    out.mkdir(parents=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    m = manifest.build(run_id, {"mode": "grpo", **config}, tasks, a.seed)
    (out / manifest.MANIFEST).write_text(json.dumps(m, indent=2))

    args = GRPOConfig(
        output_dir=str(out),
        run_name=run_id,
        num_generations=a.num_generations,
        per_device_train_batch_size=a.per_device_batch,
        max_completion_length=a.max_completion_length,
        max_steps=a.max_steps,
        learning_rate=a.lr,
        temperature=a.temperature,
        beta=0.0,
        loss_type="dapo",
        scale_rewards="group",
        remove_unused_columns=False,  # the reward function needs task_id
        use_vllm=a.use_vllm,
        vllm_mode=a.vllm_mode,
        use_cpu=a.cpu,
        seed=a.seed,
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=0,
        save_steps=max(50, a.max_steps // 4),
        report_to=["mlflow"] if a.mlflow else "none",
    )
    trainer = GRPOTrainer(
        model=a.model,
        reward_funcs=ChipRLReward({t.task_id: t for t in tasks}, a.reward),
        args=args,
        train_dataset=build_dataset(tasks),
        peft_config=LoraConfig(
            r=a.lora_r, lora_alpha=a.lora_alpha, target_modules="all-linear", task_type="CAUSAL_LM"
        ),
    )
    if a.mlflow:
        from ..tracking import manifest_params, setup

        mlflow = setup()
        with mlflow.start_run(run_name=run_id):  # TRL's MLflow callback logs into this run
            mlflow.log_params(manifest_params(m))
            mlflow.set_tags({"kind": "train", "algo": "grpo"})
            mlflow.log_artifact(str(out / manifest.MANIFEST))
            trainer.train()
    else:
        trainer.train()
    trainer.save_model(str(out / "adapter"))
    print(json.dumps({"run": run_id, "out": str(out), "steps": trainer.state.global_step}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
