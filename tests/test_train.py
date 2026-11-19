"""GRPO reward plumbing: code extraction, grading, infra masking, and the held-out guard.
No model is loaded unless CHIPRL_TRAIN_SMOKE=1 (downloads a ~10 MB test model)."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

from chiprl.graders.rtl_gen import R
from chiprl.tasks.task import load_verilogeval_dir
from chiprl.train import grpo

FIX = Path(__file__).parent / "fixtures/verilogeval"
TASK = load_verilogeval_dir(FIX / "Prob075_counter_2bc")
REF = TASK.ref_rtl.replace("RefModule", TASK.top)


def test_extract_code():
    assert grpo.extract_code(f"Here:\n```verilog\n{REF}```\nDone.") == REF.strip() + "\n"
    two = f"```verilog\nmodule helper; endmodule\n```\n```systemverilog\n{REF}```"
    assert "TopModule" in (grpo.extract_code(two) or "")
    assert grpo.extract_code("blah " + REF + " trailing") is not None
    assert grpo.extract_code("I cannot do that.") is None


def test_completion_text_handles_both_formats():
    assert grpo.completion_text("x") == "x"
    assert (
        grpo.completion_text([{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}])
        == "ab"
    )


def test_reward_grades_with_the_real_grader():
    rf = grpo.ChipRLReward({TASK.task_id: TASK})
    good = f"```verilog\n{REF}```"
    bad = f"```verilog\n{REF.replace('state <= 1;', 'state <= 2;')}```"
    logged: dict[str, float] = {}
    out = rf([good, bad, "no code"], [TASK.task_id] * 3, log_metric=logged.__setitem__)
    assert out == [1.0, 0.0, 0.0]
    assert logged == {
        "chiprl/infra_frac": 0.0,
        "chiprl/no_code_frac": pytest.approx(1 / 3),
        "chiprl/pass_rate": pytest.approx(1 / 3),
    }
    assert grpo.ChipRLReward({TASK.task_id: TASK}, "bonus")([good], [TASK.task_id]) == [1.1]


def test_infra_is_none_after_one_retry_never_zero():
    calls = []

    def infra(code, task):
        calls.append(1)
        return R(None, "infra", {})

    rf = grpo.ChipRLReward({TASK.task_id: TASK}, grade_fn=infra)
    assert rf([f"```verilog\n{REF}```"], [TASK.task_id]) == [None]
    assert len(calls) == 2, "infra results are retried once"


def test_held_out_splits_never_train():
    for split in ("test_iid", "test_ood"):
        with pytest.raises(SystemExit, match="refusing"):
            grpo.load_train_tasks(split, None, None)


def test_installed_trl_masks_unscorable_completions():
    """The infra mask relies on TRL treating all-None rewards as unscorable (advantage 0), not as 0.
    If a TRL upgrade changes that, this fails and grpo.py's masking must be revisited."""
    trl = pytest.importorskip("trl")
    from trl import GRPOTrainer

    src = inspect.getsource(GRPOTrainer)
    assert "unscorable_mask" in src and "nan_to_num(advantages, nan=0.0)" in src, trl.__version__


@pytest.mark.skipif(
    not os.environ.get("CHIPRL_TRAIN_SMOKE"), reason="set CHIPRL_TRAIN_SMOKE=1 (downloads a test model)"
)
def test_two_step_smoke(tmp_path):
    pytest.importorskip("trl")
    rc = grpo.main(["--model", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", "--tasks", str(FIX), "--max-steps", "2",
                    "--num-generations", "2", "--per-device-batch", "2", "--max-completion-length", "32", "--cpu",
                    "--out", str(tmp_path)])  # fmt: skip
    assert rc == 0
    [run] = list(tmp_path.iterdir())
    assert (run / "manifest.json").exists() and (run / "adapter").exists()
