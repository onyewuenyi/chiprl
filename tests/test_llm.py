"""Claude-in-the-loop tests with a scripted fake client: no network, no spend."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("anthropic")

from chiprl.envs.rtl_env import RTLGenEnv
from chiprl.llm import auth
from chiprl.llm.agent import AgentConfig, run_agent, run_zero_shot
from chiprl.llm.costs import Budget, BudgetExceeded, usage_cost
from chiprl.llm.evals import pass_at_k, summarize
from chiprl.tasks.task import load_verilogeval_dir

FIX = Path(__file__).parent / "fixtures/verilogeval"
TASK = load_verilogeval_dir(FIX / "Prob075_counter_2bc")
REF = TASK.ref_rtl.replace("RefModule", TASK.top)


class Block(SimpleNamespace):
    def model_dump(self):
        return dict(self.__dict__)


def usage(i=1000, o=500):
    return SimpleNamespace(
        input_tokens=i, output_tokens=o, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )


def tool(name, code, id_):
    return Block(type="tool_use", name=name, input={"code": code}, id=id_)


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        content, stop = self.script.pop(0)
        return SimpleNamespace(content=content, stop_reason=stop, model=kw["model"], usage=usage())


def test_agent_loop_lint_sim_submit():
    client = FakeClient(
        [
            ([Block(type="thinking", thinking="plan", signature="sig"), tool("lint", REF, "t1")], "tool_use"),
            ([tool("simulate", REF, "t2")], "tool_use"),
            ([tool("submit", REF, "t3")], "tool_use"),
        ]
    )
    b = Budget(max_usd=1.0)
    out = run_agent(client, TASK, b, AgentConfig())
    assert out["stop"] == "submitted" and out["reward"] >= 1.0 and out["sim_calls"] == 1
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5" and req["fallbacks"] == "default"
    assert req["output_config"]["effort"] == "high" and all(t["strict"] for t in req["tools"])
    assert "tool_choice" not in req  # forced tool choice is rejected on Opus 5.5
    # append-only history: the assistant turn is the exact content list we got back
    assert client.requests[1]["messages"][1]["content"][0].signature == "sig"
    # tool results for one turn travel in one user message
    assert all(r["type"] == "tool_result" for r in client.requests[1]["messages"][2]["content"])
    # thinking signatures are not written to trajectories
    assert "signature" not in out["events"][0]["content"][0]
    assert b.calls == 3


def test_agent_nudges_once_then_stops_without_submit():
    client = FakeClient([([Block(type="text", text="done")], "end_turn")] * 2)
    out = run_agent(client, TASK, Budget(max_usd=1.0))
    assert out["stage"] == "no_submit" and out["reward"] == 0.0 and len(client.requests) == 2


def test_agent_refusal_recorded():
    out = run_agent(FakeClient([([], "refusal")]), TASK, Budget(max_usd=1.0))
    assert out["stop"] == "refusal" and out["reward"] == 0.0


def test_zero_shot_structured_output():
    client = FakeClient([([Block(type="text", text=json.dumps({"code": REF}))], "end_turn")])
    out = run_zero_shot(client, TASK, Budget(max_usd=1.0))
    assert out["reward"] >= 1.0
    fmt = client.requests[0]["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["additionalProperties"] is False


def test_env_hides_reference_and_rate_limits_sim():
    env = RTLGenEnv(TASK, max_sim_calls=2)
    r1 = env.call("simulate", {"code": REF})
    blob = json.dumps(r1)
    assert r1["ok"] and "RefModule" not in blob and "stimulus_gen" not in blob
    env.call("simulate", {"code": REF})
    assert "budget exhausted" in env.call("simulate", {"code": REF})["error"]
    assert env.call("submit", {"code": REF}) == {"accepted": True}  # reward not revealed
    assert env.call("submit", {"code": REF})["error"] == "already submitted"


def test_budget_cap_raises():
    b = Budget(max_usd=0.01)
    with pytest.raises(BudgetExceeded):
        b.charge("claude-opus-5-5", usage(i=10_000, o=10_000))
    assert usage_cost("claude-opus-5-5", usage(1_000_000, 0)) == pytest.approx(4.0)
    assert usage_cost("claude-opus-5-5", usage(1_000_000, 0), batch=True) == pytest.approx(2.0)


def test_redact_and_credentials(monkeypatch, tmp_path):
    fake = "sk-ant-api03-" + "x" * 40
    assert fake not in auth.redact(f"key={fake}")
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))
    for v in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_PROFILE",
        "ANTHROPIC_FEDERATION_RULE_ID",
    ):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(auth.CredentialError):
        auth.credential_source()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    with pytest.raises(auth.CredentialError):
        auth.credential_source()
    monkeypatch.setenv("ANTHROPIC_API_KEY", fake)
    assert auth.credential_source() == "api_key_env"


def test_pass_at_k_and_summary():
    assert pass_at_k(10, 0, 1) == 0.0 and pass_at_k(10, 10, 1) == 1.0
    assert pass_at_k(4, 1, 1) == pytest.approx(0.25) and pass_at_k(4, 1, 4) == 1.0
    s = summarize({"a": [1.0, 0.0, None], "b": [0.0, 0.0, 0.0]}, [1])
    assert s["pass@1"]["mean"] == pytest.approx(0.25) and s["infra_excluded"] == 1


def test_duckdb_views_over_runs(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    from chiprl.analysis.db import connect

    run = tmp_path / "20261111-zero_shot"
    run.mkdir()
    rows = [
        {"task_id": "a", "mode": "zero_shot", "reward": 1.1, "events": []},
        {"task_id": "b", "mode": "zero_shot", "reward": 0.0, "events": []},
    ]
    (run / "trajectories.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (run / "summary.json").write_text(
        json.dumps(
            {
                "mode": "zero_shot",
                "model": "m",
                "effort": "high",
                "n": 1,
                "cost": {"spent_usd": 0.5, "calls": 2},
            }
        )
    )
    con = connect(tmp_path)
    assert con.execute(
        "select count(*), sum(CASE WHEN reward >= 1 THEN 1 ELSE 0 END) from traj"
    ).fetchone() == (2, 1)
    assert con.execute("select usd from costs").fetchone()[0] == 0.5
    assert duckdb is not None


def test_pmap_local_preserves_order():
    from chiprl.runtime.executor import pmap

    assert pmap(lambda x: x * x, range(20), backend="local") == [x * x for x in range(20)]


def test_unknown_model_cannot_disable_the_cap():
    from chiprl.llm.costs import UnknownModelPrice

    with pytest.raises(UnknownModelPrice):
        usage_cost("claude-opus-typo", usage(1_000_000, 0))
    with pytest.raises(UnknownModelPrice):
        Budget(max_usd=1.0).charge("claude-opus-typo", usage(10, 10))


def test_batch_submit_refuses_over_cap(monkeypatch, capsys):
    from chiprl.llm import batch

    class NoSubmit:
        class messages:
            class batches:
                @staticmethod
                def create(**kw):
                    raise AssertionError("must not submit a batch over the cap")

    monkeypatch.setattr(batch, "make_client", NoSubmit)
    fix = str(Path(__file__).parent / "fixtures/verilogeval")
    rc = batch.main(["submit", "--tasks", fix, "--n", "64", "--max-usd", "0.5"])
    assert rc == 4 and "refusing" in capsys.readouterr().out
    with pytest.raises(SystemExit):  # --max-usd is required (AGENTS rule 10)
        batch.main(["submit", "--tasks", fix, "--n", "1"])


class SpecClient:
    """Zero-shot fake keyed on the task spec: returns the reference for every task except one, which raises."""

    def __init__(self, refs: dict[str, str], explode_on: str, usd_cap_hit_after: int | None = None):
        self.refs, self.explode_on = refs, explode_on
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        spec = kw["messages"][0]["content"]
        if self.explode_on in spec:
            raise ConnectionError("simulated network failure")
        code = self.refs[spec]
        return SimpleNamespace(
            content=[Block(type="text", text=json.dumps({"code": code}))],
            stop_reason="end_turn",
            model=kw["model"],
            usage=usage(),
        )


def test_eval_run_survives_a_failing_job(monkeypatch, tmp_path):
    from chiprl.llm import evals

    tasks = evals.load_tasks(FIX, None)
    refs = {t.spec: t.ref_rtl.replace("RefModule", t.top) for t in tasks}
    bad = tasks[0]
    monkeypatch.setattr(evals, "make_client", lambda: SpecClient(refs, explode_on=bad.spec))
    rc = evals.main(["--tasks", str(FIX), "--n", "2", "--max-usd", "5", "--out", str(tmp_path)])
    assert rc == 0
    [run] = list(tmp_path.iterdir())
    rows = [json.loads(x) for x in (run / "trajectories.jsonl").read_text().splitlines()]
    summary = json.loads((run / "summary.json").read_text())
    assert len(rows) == 2 * len(tasks), "every job has a row, including the failed ones"
    failed = [r for r in rows if r["task_id"] == bad.task_id]
    assert all(
        r["reward"] is None and r["stage"] == "infra" and "ConnectionError" in r["error"] for r in failed
    )
    assert summary["job_errors"] == 2 and summary["metrics"]["infra_excluded"] == 2, [
        (r["task_id"], r["stage"], r.get("error")) for r in rows if r["reward"] is None
    ]
    assert summary["metrics"]["pass@1"]["mean"] == pytest.approx(1.0)
    assert summary["metrics"]["pass@1"]["mean_infra_as_fail"] == pytest.approx((len(tasks) - 1) / len(tasks))
