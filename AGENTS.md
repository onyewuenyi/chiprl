# Agent guide (Claude Code, Codex, any coding agent)

chiprl builds RL environments and graders for RTL generation, verification, and power-aware PPA.
The reward is only as good as the grader, so grader correctness beats everything else here.

## Commands
- `source .envrc` puts OSS CAD Suite and `scripts/bin` (OpenSTA from the LibreLane container) on PATH
- `make check`: ruff lint + format check, ty typecheck, pytest (parallel)
- `make smoke`: one VerilogEval task end to end, prints JSON
- `uv run python -m chiprl.pipeline <task_dir> [--rtl sub.sv] [--power-backend proxy|opensta] [--pnr]`
- `CHIPRL_SIM_BACKEND=verilator` switches RTL sim to Verilator --timing (gate-level stays on Icarus)

## Rules
1. Never weaken a golden test to make it pass. If a grader change breaks `tests/test_golden.py`,
   the grader is wrong until proven otherwise.
2. Every new reward hack gets a row in `docs/hack_log.md` and a regression test.
3. Infra failures return `reward=None`, never 0. Do not collapse them.
4. Tools return `ToolResult` JSON. The policy never sees raw tool logs.
5. Anything that changes tool flags or parsing must keep the cache key honest
   (`chiprl/cache.py` keys on tool versions; bump the `pipeline/vN` tag if semantics change).
6. Parsers must work on both Yosys 0.33 (apt) and OSS CAD Suite nightly; CI runs both. New tools
   (yosys-slang, eqy) are feature-detected with a fallback, never assumed.
7. Hidden testbenches and reference RTL are grader-only. Nothing in `envs/` may expose them.
8. Python 3.12+, typed, ruff-clean. No new runtime dependencies on the Mac path without a reason;
   GPU/training deps belong in the GPU service image.
9. LibreLane (`--pnr`) is the slow ground-truth path. Never call it from a reward function.
10. Claude API code lives in `chiprl/llm`. Credentials only via `auth.make_client()` (ambient chain,
    never a key argument). Every entry point that calls the API takes a required `--max-usd`.
    Never run a paid eval without the user's go-ahead; tests use the fake client in tests/test_llm.py.
11. The env (`chiprl/envs/rtl_env.py`) is the only surface a policy sees. Do not add tool outputs that
    reveal the testbench, reference, or reward.

## Layout
`chiprl/tools` (tool wrappers), `chiprl/graders` (gated graders, mutants), `chiprl/tasks` (loaders),
`chiprl/envs` / `rewards` / `baselines` / `train` / `analysis` (later phases), `docs/PLAN.md` (roadmap),
`docs/STACK.md` (tool choices).
