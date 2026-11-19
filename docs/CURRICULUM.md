# Reward Atlas ↔ chiprl

Three artifacts, one system, no duplicate implementations:

| | Role | Where |
|---|---|---|
| **chiprl** | The executable system: tools, graders, environments, experiments, training code | this repo |
| **Reward Atlas** | The curriculum: first-principles explanations, missions, quizzes, projects, interview reps | [claude.ai/artifact/UMsEkhB55hmqojd6rj432f](https://claude.ai/artifact/UMsEkhB55hmqojd6rj432f) |
| **Notebooks** | The evidence: prediction, command, result with interval, failures, threats to validity | `docs/notebook/E<n>.md` ([template](notebook/TEMPLATE.md)) |

**Research question** (what everything below serves): *can a trained model learn RTL
transformations that give statistically defensible power improvements, keep functional
correctness, and generalize to unseen designs better than search and best-of-N at matched compute?*

## Order of work

Follow the experiment ladder (`docs/EXPERIMENTS.md`), not the Atlas stage numbers. The Atlas stages
are prepared just in time for the rung that uses them. Trust the measurement and the reward before
anything learns from them.

| Part | Rungs | Atlas stages to study first | Atlas project | What you can say after |
|---|---|---|---|---|
| A. Trust the measurement | E0–E2 | 7 Infra at scale; 6 Rewards you can trust (noise) | P7 rollout service | "I know where every number in my reward comes from and how noisy it is." |
| B. Trust the reward | E3–E4 | 6 Rewards you can trust | **P6 power reward** | "I broke my own reward before a model could." |
| C. Is there anything to learn? | E5–E7 | 5 Many-step RL (episodes, state, actions) | — | "I know when a problem needs learning: here is the oracle and the search curve." |
| D. Learn from reward | E8 | 1 One group of rewards; 4 The policy update | **P1** advantages, **P4** bandit + derivation | "I derived GRPO and measured the variance cut." |
| E. Give the model the problem | E9–E10 | 2 The sampler; 8 Agents and data; 3 Inside the transformer, just before E10 | **P2** sampler, **P8** agent harness, **P3** attention | "RL vs best-of-N at matched compute, held-out, with intervals." |
| F. Research results | E11–E13 | 6 (proxy validity); 5 (placement) | **P5** GridPlace, as E13 prep | "How much the proxy reward lies, and where an agent beats BO." |

Stage 3 (transformer internals) moved late on purpose: none of power measurement, reward design,
environment construction or REINFORCE needs it. P4 and P5 are preparation for E8 and E13, not
separate RL projects.

## Stage by stage

"Exists" is code you build on. "You write" is the Atlas project: from a blank file, tests included,
as the Atlas asks. Nothing here pre-writes those.

| Atlas stage | chiprl rungs | Exists in chiprl | You write (destination) | Check it with |
|---|---|---|---|---|
| 1 One group of rewards | E8, E10 | `analysis/stats.py` (intervals), `graders/service.grade` (the verifier that makes the rewards) | P1 `grpo_advantages` → `chiprl/rl/advantages.py` + its tests | Zero-signal fraction 0.248 ± 0.004 at p = 0.3, G = 4 |
| 2 The sampler | E6, E9, E10 | `llm/evals.pass_at_k` (unbiased), `summarize` with `mean_infra_as_fail` | P2 `sample`, `useful_fraction` → `chiprl/rl/sampling.py`; compare your `pass_at_k` with the existing one | pass@5 at n = 20, c = 3 is 0.601 |
| 3 Inside the transformer | E10 (just in time) | — (training code is not built) | P3 attention + KV cache, outside chiprl (an ML-coding rep) | KV-cache decode equals full forward |
| 4 The policy update | E8 | — | P4 bandit, REINFORCE, baseline, PPO clip → `chiprl/rl/bandit.py`; the derivation in `docs/notebook/E8.md` | Offset test: +5 to every reward changes nothing with a baseline |
| 5 Many-step RL | E5–E7, E13 | `envs/rtl_env.py` (tool surface; no reset/step yet), `tasks/splits.py` | E5 `PowerOptEnv` (reset/step, oracle); P5 GridPlace as E13 prep | Oracle replays to the same reward; determinism test |
| 6 Rewards you can trust | E0–E4, E11 | `graders/rtl_gen.py` gates, `tools/diffsim.py`, `tools/equiv.py`, `tools/timing.py` (ss/tt/ff), `tests/test_golden.py`, `docs/hack_log.md`, `analysis/report.py`, `docs/REWARD.md` | P6 `power_reward` → `chiprl/rewards/power.py` + hack tests; E2 noise band | `docs/REWARD.md` §3 obligations, each a passing test |
| 7 Infra at scale | E0, E5, E9, E10 | `runtime/executor.py`, `cache.py`, `tools/_run.py`, `runtime/manifest.py`, `runtime/bench.py`, `docs/DESIGN.md` | P7 asyncio rollout service, standalone; then diff it against chiprl's runtime | Same tests as P7, then read `tests/test_executor.py` and `tests/test_cache.py` |
| 8 Agents and data | E9–E10 | `llm/agent.py` (tool loop, budget), `llm/batch.py` (best-of-N incl. cluster selection), `graders/cluster.py`, `tasks/export.py` (tasks as container + verifier), `tasks/coverage.py`, `llm/auditor.py`, `evals --split`, `analysis/report.py` | P8 harness: context policy (last 3 verbatim, older summarized), ablation table, deep dive | Held-out pass@1 with a Wilson interval, failures by layer |

## What the Atlas projects need from chiprl, and where it is

| Project needs | chiprl provides |
|---|---|
| P6(b) repeatability: grade the same candidates 5 times | `CHIPRL_NO_CACHE=1` (or each call is a cache hit and σ is 0 by construction) |
| P6(c) ≥ 50 held-out designs, failures by cause | `evals --split test_iid` / `test_ood` (32 + 36 tasks); `python -m chiprl.analysis.report runs/<run>` |
| P6(d) new vs old on the same designs, paired interval | `python -m chiprl.analysis.report runs/<old> runs/<new>` (paired bootstrap, McNemar, tasks regressed) |
| P8 ≥ 30 held-out tasks, Wilson interval, cost per success | `--split test_ood` (36 FSMs) or `test_iid`; `summary.json` cost; `report` |
| P8 budget enforced in code | `llm/costs.Budget`, required `--max-usd`, unknown models refused |
| Any claim: n chosen before the run (R2) | `analysis.stats.tasks_needed(delta, discordant)` |

## From the CS329A and Ricursive reviews (Nov 18)

| Idea | Where it landed |
|---|---|
| Sample, filter, cluster survivors by behavior (AlphaCode) | `graders/cluster.py`; `cluster_bon` in `llm/batch.py`. Selection and diversity only, never proof sharing |
| Differential testing against simulators | `tools/diffsim.py`, the first refuter in `equiv()` |
| Search the harness as a config (Archon, ADAS) | Headline row E9d in EXPERIMENTS.md |
| Stack GRPO with best-of-N; report curves over k | Headline row E10b and the R6 note |
| Small trusted checker, fail closed (AlphaProof, Lean) | `docs/DESIGN.md`, "A small trusted core" |
| Tasks packaged like agentic coding benchmarks | `tasks/export.py`, `chiprl/verify.py` |
| Synthetic tasks with coverage metrics | `tasks/coverage.py` (the E5 generator reports the same) |
| Multi-corner timing | `tools/timing.py`, `pipeline --corners` |
| Content-addressed action cache (Bazel) | `docs/DESIGN.md`, cache decision |
| Not taken: UPF, EM/IR, DFT gates; Ripple's RL recipe | No open flow, or not verified; revisit for E10 |

## Status, Nov 18 2026

| Rung | State |
|---|---|
| E0 | Done (Nov 13); toolchain now pinned so results reproduce across machines |
| E1–E3 | Not started (yours: predictions first) |
| E4 | Partly: eqy false proof root-caused and fixed (failure_log #1), two new hacks blocked (hack_log #5, #6). Still open: the grader-eval set with per-gate error rates, the gate × hack matrix, timing and area gates, the noise floor |
| E5–E13 | Not started. Splits, statistics and reports they need now exist |

Paid runs (`make eval-smoke`, P6(a), P8) need `--max-usd` and your go-ahead (AGENTS rule 10).
