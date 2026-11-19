# Possible features

Candidates not yet built, with the model, cost and reason for each. Collected Nov 18 2026 from the
stack review, the Claude API review and the Modal review. Nothing here is committed work: pick an
item, move it into `docs/PLAN.md`, and build it with a test. Every paid item needs `--max-usd` and an
explicit go-ahead (AGENTS rule 10).

Prices used (first-party API, per million tokens, input / output): Claude Opus 5.5 $4 / $20,
Claude Sonnet 5.5 $2 / $10, Claude Haiku 5.5 $0.10 / $0.50. Message Batches halve that; cached input
costs about 5%. Thinking tokens bill as output, and Opus 5.5 always thinks, so output dominates.
Estimates are assumption-driven and good to a factor of about 2: measure with a 10-request run first.

## 1. Claude API in the workflow

Ranked by value to the research question. Already built and not repeated here: zero-shot and agent
baselines, best-of-N through Message Batches (with cluster selection), the hack auditor, Claude Code
on GitHub.

| # | Feature | What it does | Model | Assumed volume | Estimate |
|---|---|---|---|---|---|
| 1 | Generate the power task family (E5) | Claude writes spec + reference + testbench for each transform (operand isolation, clock enable, FSM encodings). A task is accepted only if the reference passes, the testbench kills mutants, differential simulation agrees, and the timing gate binds. `tasks/coverage.py` says which cells to ask for next. Fixes contamination (R5) | Opus 5.5, effort `high`, Batches, structured outputs (`output_config.format`) | 300 candidates for ~150 accepted, ~3K in / ~7K out, run twice | ~$45 once, ~$10/month |
| 2 | Strengthen testbenches, flag ambiguous specs | For grades flagged `testbench_weaker_than_spec`, Claude proposes extra stimulus; accepted only if it now fails the refuted design and the reference still passes. Separately, read every spec against its reference and flag specs that allow more than one behavior (quarantine candidates, like Prob053) | Opus 5.5 for testbenches (iterative); Sonnet 5.5 + Batches for the spec scan | ~30 tasks × 3 rounds × 8K / 8K; 154 specs × 4K / 2K | ~$20 once, ~$5/month |
| 3 | Classify failed rollouts | Every failure into the four failure-log classes with evidence; makes "failures by layer" (Atlas P8) automatic. Validate against hand labels before trusting it (R7) | Haiku 5.5, effort `low`, Batches, enum-constrained structured output | ~10,000 failures/month, ~3K in / ~0.8K out | ~$4/month |
| 4 | Advisor mode (E9a) | Claude reads RTL + synthesis, power and timing reports and returns ranked recommendations with evidence; you decide; every recommendation, decision and measured ΔP is logged | Opus 5.5, effort `high` (default is `medium`), Files API for large reports, prompt caching | 30 tasks × 3 calls, ~30K in (mostly cached) / ~10K out | ~$30 in the E9 month |
| 5 | chiprl tools as an MCP server; agent harness (E9b, Atlas P8) | lint, simulate, synth_report, power_report, submit behind one server, usable from Claude Code, the MCP connector (`mcp_servers` + `mcp_toolset`, beta `mcp-client-2025-11-20`), Managed Agents and the GRPO env. Hidden testbench and reference stay behind the tools (AGENTS rule 7). Task budgets (beta) for pacing; compaction for long sessions | Opus 5.5 (Sonnet 5.5 about halves cost) | 150 tasks × 3 seeds, ~12 turns, ~$0.65/task | ~$300 in the E9 month (~$150 on Sonnet). Managed Agents adds a session runtime fee (not estimated) and is not ZDR-eligible; the hosted code-execution sandbox has no EDA tools, so tools must be yours |
| 5b | Best-of-N baseline runs (built) | The bar RL must beat at matched samples | Opus 5.5, Batches | N = 16 × 150 tasks, ~1K in / ~4K out | ~$100 per run |
| 6 | SFT data from verified successes | Rejection-sampling SFT set for the open model before GRPO, from 5b's grader-verified samples | none extra | — | $0 extra. **Blocked:** check Anthropic's commercial terms on using outputs to train models first |
| 7 | Paper critiques with citations | Upload papers (VeriReason, CVDP, AlphaChip and critiques); claims for contribution / weakest point / next experiment, each tied to a page | Opus 5.5, Files API, citations (incompatible with structured outputs) | ~15 papers × ~10 questions, cached | ~$12 once |
| 8 | Nightly triage | Scheduled run: grader-eval, classify new failures (item 3), draft failure-log rows for review | Sonnet 5.5 or Haiku 5.5 | ~20 new failures/night | ~$5/month in tokens; run from GitHub Actions on the Messages API to avoid the Managed Agents runtime fee |

Monthly shape: about $80 once for items 1, 2 and 7, then $20 to $25/month through E8; add about
$430 in the E9 month ($280 with Sonnet for the agent); little Claude spend in E10.

Housekeeping on the existing Claude code:
- Batch requests omit the `thinking` setting and fallbacks that zero-shot sends. Align the sampling
  configuration; Batches reject the server-side `fallbacks` parameter, so refusals there stay refusals.
- Message threads (beta) for long agent loops: smaller requests, lower time to first token. Not
  combinable with compaction or Batches.
- Keep `fallbacks: "default"` on Opus 5.5 calls (`llm/agent.py` does).

## 2. Modal

Rates from third-party write-ups (Nov 2026), per second, preemptible GPU functions: H100 ≈ $3.95/h,
A100 80GB ≈ $2.50/h, L40S ≈ $1.95/h; CPU ≈ $0.047 per core-hour plus memory; Sandboxes about 3× CPU.
Starter plan: $30 of compute per month free. Verify on modal.com/pricing before budgeting.

| Use | Why | Estimate |
|---|---|---|
| GRPO runs (E10) | Per-second billing, no idle GPU, Python-defined image | 4 to 8 GPU-hours per 1.5B LoRA run ≈ $10 to $30; 3 seeds + ablations (~8 runs) ≈ $130 to $260, inside the plan's $200 to $400. For long runs, RunPod or Lambda via `infra/sky/grpo.yaml` may be cheaper per hour |
| Grading fan-out | Hundreds of CPU containers grade while the GPU generates, so the reward never stalls training; every grade runs in the one Linux image (fixes the Mac/Linux Yosys mismatch) | ≈ $8 per million graded rollouts at ~0.5 core-seconds each |
| Sandboxes for model-written RTL and exported tasks | Isolation for untrusted code (the open container-sandbox item in `docs/DESIGN.md`); hosts the agent side of exported tasks | ≈ $25 per million grades; use for agent episodes, not every reward call |

Implementation sketch: `infra/modal/` app built from the `chiprl-task-env` image, a `grade` function
exposed as `CHIPRL_BACKEND=modal` in `runtime/executor.py`, and a GPU function running
`chiprl.train.grpo`. The local path stays unchanged (AGENTS rule 12).

## 3. Stack upgrades

Do before E1 to E4:
- **Harbor task format.** Terminal-Bench's format, accepted by `verifiers` v1 and the HF environments
  hub. `tasks/export.py` is close (instruction.md, grader-only verifier, score); matching the current
  Harbor spec exactly lets chiprl tasks run in those harnesses unchanged. Read the spec first.
- **Report numbers from the Linux container.** One OSS CAD Suite tag is two Yosys versions
  (failure_log #7). Run CI and reported results in `chiprl-task-env` pinned by digest; pin the
  LibreLane image by digest too.
- **`abc pdr` as the eqy engine.** May prove some of the 8 references Yosys `equiv_*` leaves
  unknown. Measure proof rate and time with the canary self-test kept; if it wins, eqy returns to the
  default path.
- **CVDP as the external held-out benchmark.** VerilogEval-Human is near saturation (an agentic
  system reports 95.9%). CVDP: 783 expert problems, 13 categories, a containerized agent harness.
  Check its license before vendoring.

At E10:
- **Pin the trainer stack** from `uv.lock` (TRL 1.18 today); vLLM within TRL's range (>= 0.23, < 0.34).
  verl only if multi-turn rollouts at scale need it (its mainline dropped the V0 trainer; async
  trainers moved to verl-recipe).
- **Model choice:** Qwen2.5-Coder 1.5B to 7B for comparability with published Verilog RL (VeriReason,
  VERIRL), or a Qwen3-family 4B to 8B with open SFT→RL recipes. Measure base pass@k on the train
  split first: GRPO needs tasks the base model passes 10 to 90% of the time.
- **Write the GRPO loss once in plain PyTorch** (Atlas P1/P4) and check it against TRL on one batch.

Leave alone: EDA tools (already newer nightlies than the latest stable releases), LibreLane 3.0.x
(3.1 is pre-release), Icarus for VerilogEval, uv / ruff / ty. Not on the research path: UPF, EM/IR,
DFT gates, a second PDK.
