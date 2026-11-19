# Experiment ladder: 0 to 1 in AI for chip design

`PLAN.md` says what gets built. This file says what I learn, and in what order. Each rung is one
experiment that answers one question with a number I did not have before. Each rung depends on the
one below it. The ladder ends where the OpenAI AI for Chip Design job description begins: RL
environments and evals for RTL generation, verification, and PPA optimization, with proxy rewards
and generalization to new designs.

Sources: the Nov 2026 ChatGPT conversation that set five milestones (an 8-bit accumulator baseline,
equivalent RTL actions, a reward pipeline, search and bandit baselines, then PPO), and the current
state of this repo.

**Companion curriculum:** Reward Atlas teaches the concepts each rung needs; [CURRICULUM.md](CURRICULUM.md)
maps every Atlas stage and project to a rung, the code that exists, and the file you write. What is
scored is defined once in [REWARD.md](REWARD.md).

## Why a ladder, and what changed

**Compared with the five-milestone plan.** Its core is right: start tiny, trust the measurement
before optimizing, and remember that search is not RL. The changes:

| Milestone plan | This ladder | Why |
|---|---|---|
| Accumulator in a separate toy flow | Accumulator becomes a chiprl task (`tests/fixtures/power/acc8`) | One grader, one cache, one set of golden tests. The toy reuses the real infrastructure. |
| Reward `1 - P_c/P_b`, `-1` on failure or on a timing/area miss | Gated: `1[sim ∧ eqy ∧ gate-sim ∧ timing ∧ area] · clip(ΔP/P_b)`, `None` on infra error, 0 below the noise floor. Timing = OpenSTA WNS ≥ 0 at 10 ns (logic depth ≤ baseline on the proxy path); area ≤ 1.1 × baseline | Keeps the constraints. Adds an equivalence check, because a testbench alone can be gamed (see the hack log). A 0.3% "gain" inside measurement noise rewards noise. |
| Power from Verilator + Yosys | Gate-level VCD + proxy and OpenSTA | An RTL VCD annotates only ports and registers. The proxy cannot see internal power (E3 shows why that matters). |
| Random → bandit/BO → PPO | Enumerate → random/greedy/BO → contextual bandit → REINFORCE/GRPO → LLM GRPO | Keep the space small enough to enumerate so every agent is scored against an exact optimum. LLM RL dropped PPO's critic, so GRPO is the algorithm worth deriving. |
| (missing) | Red-teaming the reward, proxy vs ground truth, held-out designs, throughput | These are the job, not extras. |

**Compared with `PLAN.md`.** The plan is organized by features, and the infrastructure got ahead of
my hands-on understanding. The README numbers came from a cloud box, and E0 (Nov 13) found that a
local run disagrees with them and that one grader gate is unsound. The ladder makes every component
produce a number I reproduced myself and can explain in an interview.

## Job description coverage

Each line of the posting maps to the rungs that produce evidence for it. If a line has no artifact,
it is not covered.

| The role says | Rungs | Evidence I can show |
|---|---|---|
| Build RL environments and evaluations for RTL generation, design verification, physical design optimization | E5, E9–E10 (RTL/PPA), E12 (DV), E13 (PD) | Three envs behind one `ToolResult` API, each with a grader-eval set (R1) |
| Help models use chip-design tools and improve PPA while preserving correctness | E3, E4, E9 | Gated reward with measured per-gate error rates; agent tool-use transcripts |
| Design experiments, establish baselines, measure whether improvements hold up on new tasks and designs | E5–E7, E9–E10 | Oracle, search curves, family-held-out splits, power analysis before each run (R2, R5) |
| Investigate failures across model behavior, rewards, evaluation tools, and experiment infrastructure | every rung | `docs/failure_log.md`, triaged into exactly those four classes, plus `hack_log.md` |
| Improve iteration speed through better tooling, faster evaluations, and proxy rewards | E0, E2, E11 | Rollouts/s and p95 latency over time, cache hit rate, proxy-vs-signoff rank agreement |
| Turn successful experiments into reusable research code and training workflows | every rung | Each rung leaves a module and one command (see "Leaves behind" in each card) |
| Form clear hypotheses; distinguish meaningful results from noise or evaluation errors | every rung | Pre-registered predictions, noise floor (E2), CIs, grader error rates (R1, R2) |
| Explain what you built, what failed, and what you learned | Results index, report | Technical report with threats to validity and negative results (R8) |
| Care about developing safe, beneficial AI | E4, E9, E12 | Reward-hack red-teaming, auditor validated against hand labels, sandboxed tool execution |

## Rules

1. **One question per experiment.** Write the prediction in the notebook *before* running. Wrong
   predictions are where the learning is.
2. **Work in this order:** ground truth, then reward, then search, then learning. Tabular before LLM.
   Never skip a rung because the code for a later one already exists.
3. **Keep the spaces enumerable** until E8, so there is always an oracle.
4. **Every experiment ends with an artifact:** a table or plot built from `runs/` by one command.
   Record it in the notebook and add a row to the results index at the bottom of this file.
5. **Negative results ship.** "GRPO did not beat best-of-16" is a result, and it needs CIs.
6. **AGENTS.md still applies:** golden tests stay green, every hack gets a `hack_log.md` row and a
   test, infra failures stay `None`, and every paid run gets `--max-usd` and my explicit go-ahead.

## Rigor standards

These apply to every rung. Each one exists because a specific way of fooling myself is cheap here.

- **R1. The grader has measured error rates.** Keep a labeled grader-eval set in
  `tests/grader_eval/`. *Known-correct* designs: re-encoded FSMs, retimed logic, different adder
  styles, the E3 rewrites. *Known-wrong* designs: mutants, reset-value changes, E4 hacks. Report the
  false-accept and false-reject rate of each gate (sim, eqy, yosys equiv, gate-sim, timing, area) in
  CI. Where two formal backends disagree, fail closed. First entry: eqy proves a reset-value
  mutant equivalent (E0, `failure_log.md` #1).
- **R2. Choose the sample size before the run.** The unit of analysis is the task (or design
  family), not the sample. Compare methods paired on the same tasks (McNemar for pass/fail, paired
  bootstrap for ΔP). Before any comparison, compute the number of tasks needed. With 20 pass/fail
  tasks, a 95% CI is about ±22 points. Detecting a 10-point paired gain needs about 150 tasks if
  methods disagree on about 20% of them.
- **R3. Constraints must bind.** A gate that never fires tests nothing. The first P&R run had WNS
  +3.13 ns at a 10 ns clock. Find each design's Fmax and set the task's clock to 90–95% of it. Do the
  same for the area cap. Report how often each gate fires.
- **R4. Judge rewards by the decision they drive.** GRPO uses only the ranking within a group, so a
  proxy reward is judged by within-group rank agreement with ground truth (Kendall τ among the
  candidates for one task), plus how often it flips the group's best choice. Global Spearman is
  secondary.
- **R5. Splits mean something.** Hold out entire design families, and hold out workloads
  separately. Report in-distribution and out-of-distribution for every method. Use generated tasks
  (contamination-free) for the main claims, and VerilogEval only as a public reference point, since
  it has been public since 2023. Include at least one design that is not a toy (Ibex or PicoRV32
  ALU/decoder, or an OpenTitan IP block, thousands of cells) before claiming anything about "real
  designs".
- **R6. Baselines are curves at matched cost.** Every method is a curve of best correct ΔP vs EDA
  calls (and tokens or GPU-hours), on the same axes. An RL claim counts only if its curve is above
  best-of-N and search at equal cost. Use AutoTuner or BO wherever the actions are tool knobs.
- **R7. Validate every judge.** The hack auditor is an LLM classifier. Hand-label about 50
  trajectories and report precision, recall, and Cohen's κ against my labels before quoting its
  hack rates.
- **R8. Provenance, determinism, ablations, write-up.**
  - Every JSONL row carries git SHA, tool versions, config hash, seed and stimulus seed.
  - A test asserts that the same input gives a bit-identical reward.
  - The infra-error rate is reported next to results.
  - Each claimed effect gets an ablation: drop a gate, dense vs binary reward, proxy vs OpenSTA
    reward.
  - Results go in a short technical report with methods, CIs, threats to validity, and negative
    results.

### Failure triage (`docs/failure_log.md`)

Every failed run, surprising number, or broken tool goes in the failure log under exactly one class,
the four the role names: **model behavior**, **reward**, **evaluation tool**, **infrastructure**.
Each entry records symptom, root cause, how it was found, the fix, and the test that keeps it fixed.
Reward hacks also get a `hack_log.md` row.

### Notebook template (`docs/notebook/E<n>.md`)
```
# E<n>: <question>
Prediction (written <date>, before running):
Sample size: <n tasks/seeds, and the effect it can detect (R2)>
Setup: commit <sha>, command, tool versions (yosys -V, iverilog -V, sta -version)
Result: <table or plot path, with CIs>
Failures: <failure_log.md entries opened, by class>
Threats to validity: <what could make this result wrong: grader error, contamination, noise, split leakage>
Surprise: <what I got wrong and why>
Leaves behind: <module, test, or command another experiment can reuse>
Next: <what this changes about the next rung>
```

## The ladder at a glance

| # | Question | Artifact | Core concept | JD signal | Days |
|---|---|---|---|---|---|
| **A. Can I trust the measurement?** |
| E0 | Does the pipeline reproduce on my machine? | Diff of README numbers, Mac vs cloud | Determinism, tool pinning, cache keys | Sandboxed EDA infra | 1 |
| E1 | Where does an 8-bit accumulator's power go? | Power vs enable duty cycle | Dynamic power = activity × capacitance | PPA fundamentals | 2 |
| E2 | How noisy is the power number? | CV% and minimum detectable effect | Reward noise floor | Reliable evals | 1 |
| **B. Can I trust the reward?** |
| E3 | Which "low-power" rewrites actually save power, and on which workload? | Variants × workloads table, proxy vs OpenSTA | Workload dependence, tool already optimizes, proxy blind spots | Proxy rewards | 3 |
| E4 | Can I cheat my own reward? | New hack-log rows plus regression tests | Goodhart, gating, equivalence | Reward design | 2 |
| **C. Is there anything to learn?** |
| E5 | What does the full design space look like? | Enumerated optimum per workload | Env API, oracle, sequential vs independent actions | RL env design | 2 |
| E6 | How many EDA calls does search need? | Best-so-far vs evaluations curve | Sample efficiency, search ≠ RL | Baselines | 2 |
| E7 | Can a policy pick the right edit with *zero* EDA calls on an unseen workload? | Held-out regret vs search at budget k | Amortization: the real reason to learn | Generalization | 2 |
| **D. Learning from reward** |
| E8 | Derive policy gradient: does a group baseline cut variance? | Learning curves, 3 seeds | REINFORCE → GRPO | RL fundamentals | 2 |
| E9 | LLM as advisor (my Intel workflow) vs autonomous agent vs best-of-N: which wins, and how do they cheat? | ΔP per EDA run per mode, hack rate | Autonomy levels, open action space, auditing | Tool-using models | 4 |
| E10 | Does RL beat best-of-N at matched compute on held-out designs? | Matched-compute table with CIs | GRPO on an LLM | Training + eval | 5–7 |
| **E. Breadth: the three env types the JD names** |
| E11 | How well does the cheap proxy track signoff, and does training on it drift? | Spearman, top-k, drift plot | Proxy-reward Goodhart | Proxy rewards | 4 |
| E12 | Can a model write tests that kill mutants? | Kill rate per model | DV reward design | Verification env | 4 |
| E13 | Does an agent beat AutoTuner on flow knobs? | PPA vs runs, both methods | Physical-design env, BO vs RL | Physical design | 4 |

Rough calendar from Nov 13: A by Nov 17, B by Nov 23, C by Nov 30 (resume link goes live, as in
PLAN), D in December, E from late December into January.

---

## A. Can I trust the measurement?

### E0: Reproduce the pipeline on my machine
- **Question:** Does `make smoke` give the README's numbers on my Mac?
- **Predict:** Area and cell counts match exactly, since Yosys/ABC are deterministic for the same
  version. Proxy power matches to better than 1%. Wall time differs.
- **Do:** `./scripts/setup_mac.sh`, `source .envrc`, `make check`, `make smoke`, then run the pipeline
  on all 4 fixtures. Write down `yosys -V` and the other tool versions. Run `make bench` and record
  rollouts/s.
- **Artifact:** README table vs my table, with a column for any delta and its cause.
- **Done when:** golden tests pass locally and any delta is explained (most likely a different
  Yosys nightly; check that `cache.py` keys on that).
- **Learn:** Why the cache key must include tool versions (AGENTS rule 5). What "reproducible
  reward" costs in practice.

### E1: The accumulator, by hand
- **Question:** For `acc <= rst ? 0 : en ? acc + d : acc`, how do flops, adder and clock split the
  power, and how does that change with the enable duty cycle?
- **Predict:** Sketch it before running: share of power in flops vs logic at 10/50/100% enable
  duty, and whether power scales linearly with duty.
- **Do:** Write `ref.sv` (`module RefModule`), `prompt.txt`, and a self-checking `test.sv` *by hand*,
  in the VerilogEval layout under `tests/fixtures/power/acc8/`, so `load_verilogeval_dir` and
  `pipeline.py` work unchanged. Make workload a parameter of `test.sv` (enable duty, data toggle
  rate) and generate one tb per workload from a small Python template. That avoids touching
  `sim.py`. Then run the pipeline and read the netlist (`yosys stat`, open the gate-level VCD in
  Surfer).
- **Artifact:** Stacked bar of switching vs leakage at 4 duty cycles. Cell list with the role of
  each cell.
- **Done when:** I can explain every cell in the netlist, and the duty-cycle curve's shape matches
  (or explainably differs from) my prediction.
- **Learn:** Dynamic power is ½CV²fα. The stimulus is part of the measurement. Reading a netlist.

### E2: The noise floor
- **Question:** If I rerun the same design with a different random stimulus of the same workload,
  how much does power move? What is the smallest real improvement?
- **Predict:** CV under 2% for 40k cycles, and larger for short tbs.
- **Do:** 10 seeds × 3 workloads × 2 tb lengths on acc8 and on `Prob068_countbcd`. Also sweep the
  clock period with OpenSTA to find each design's Fmax, so tasks can be set at 90–95% of it (R3).
  Check determinism: the same seed twice must give a bit-identical reward (R8).
- **Artifact:** CV% table, a minimum detectable effect (MDE) at 95% for paired comparisons, and
  Fmax per design.
- **Done when:** The reward config has a `noise_floor` (ΔP below MDE → 0) and a per-task clock
  period at which timing binds, each with a test.
- **Learn:** A reward is a measurement with error bars. This is the first input to every later
  comparison.

## B. Can I trust the reward?

### E3: Low-power rewrites, measured
- **Question:** Which classic power fixes help acc8 after synthesis, and under which workload?
- **Variants:** (a) baseline; (b) operand isolation (gate `d` to 0 when `!en`); (c) explicit
  clock-enable flop style; (d) an integrated clock gate (`sky130_fd_sc_hd__dlclkp`, present in the
  vendored lib); (e) one functionally *wrong* variant that looks lower-power.
- **Predict, then check:** (1) Does Yosys already map `en` to `edfxtp` (enable flop), or to mux +
  `dfxtp`? If it maps to `edfxtp`, variant (c) is a no-op. (2) Clock gating saves mostly
  *internal* flop clock power, which the proxy omits. So the proxy should under-credit (d) relative
  to OpenSTA. (3) Operand isolation only pays when `en` is low *and* `d` toggles.
- **Do:** 5 variants × 3 workloads × {proxy, OpenSTA}. Equivalence (eqy) on each.
- **Artifact:** One table: power (both backends), area, depth, WNS, equiv, gate-sim pass. Highlight
  every cell where proxy and OpenSTA disagree on the sign of ΔP.
- **Done when:** Every row is explained by netlist and activity, not by intuition.
- **Learn:** Optimizations are workload-dependent. Synthesis already does some of them. A proxy can
  be confidently wrong on exactly the change you care about. This is the seed of E11.

### E4: Red-team my own reward
- **Question:** Without a correct design, how many ways can I lower the reported power, and which
  ones get through the gates?
- **Try at least:** behavior that differs only on input patterns the tb never drives; logic that idles
  on the tb's stimulus; `initial` values that sim honors but gate-sim does not; deleting logic the
  tb never observes; constants on rarely sampled outputs; buying power with area or timing (for
  example, duplicating logic to cut fanout, or a long ripple chain that misses the clock).
- **Predict:** For each attempt, which gate catches it (lint, sim, sample count, eqy, gate-sim,
  timing, area, or none).
- **Also:** Build the R1 grader-eval set, starting from the E3 rewrites (known-correct), the
  `mutate.py` mutants, and these hacks (known-wrong). Fix the eqy false proof first: find out why
  the `sat` strategy accepts the reset-value mutant (x-tolerant miter, `-set-init-undef`,
  `async2sync`), then add a bounded check from reset, or require backend agreement.
  *Done Nov 18 (failure_log #1): it was not the miter. The `sat` strategy's `-set-init-undef` and
  `-set-def-formal` contradict each other on `$anyinit` flops, so the proof was vacuous. Equivalence now
  needs a bounded check from power-up and Yosys `equiv_*` to agree; eqy is opt-in behind a self-test.*
- **Artifact:** Rows in `docs/hack_log.md` and regression tests in `tests/test_golden.py`. A
  gate × hack matrix showing which gate is load-bearing. A per-gate false-accept / false-reject
  table on the grader-eval set.
- **Done when:** Every successful hack is blocked and has a test (AGENTS rule 2). The timing and
  area gates and the E2 noise floor are in the reward config, each with a test. False-accept rate on
  the grader-eval set is 0, and the false-reject rate is known.
- **Leaves behind:** `tests/grader_eval/` and a `make grader-eval` target that CI runs.
- **Learn:** The grader is the product. Equivalence checking, not testbenches, is what makes a PPA
  reward safe. This experiment is the interview story.

## C. Is there anything to learn?

### E5: The environment and its oracle
- **Question:** If an episode is a short sequence of transforms from E3, is the problem actually
  sequential, or is it independent picks (a bandit)?
- **Do:** A Gymnasium-style `PowerOptEnv` in `chiprl/envs/` with state = (current RTL id, workload
  features), action = one of the transforms or `stop`, and reward = gated ΔP from E2/E4, at 2–3
  steps per episode. Add 2–3 parameters (width 8/16/32, adder style) to `chiprl/tasks/power_family.py`
  so there are dozens of designs. **Enumerate every episode** with the cache warm. Build the
  generator so it can reach the at least 150 tasks across at least 5 families that R2 and R5 need for
  E9 and E10. Each generated task gets a reference, a hidden tb, held-out workloads, and a family tag.
- **Predict:** Whether transform order matters, and whether the best transform differs across
  workloads.
- **Artifact:** Optimal sequence per (design, workload), and the fraction of states where the best
  first action depends on what comes after.
- **Done when:** The env passes a determinism test and an "oracle replays to the same reward" test.
- **Learn:** State, action and reward design. Why an oracle is the most valuable thing a small
  environment gives you. If order never matters, say so: the problem is a bandit, and that is a
  finding.

### E6: Search baselines
- **Question:** How many EDA evaluations do random search, greedy, and Bayesian optimization
  (Optuna TPE) need to reach 95% of the oracle?
- **Artifact:** Best-so-far reward vs number of evaluations (mean and CI over 20 seeds), one line per
  method, with the oracle as a horizontal line.
- **Learn:** Sample efficiency is the currency, because EDA calls are expensive. An agent that tries
  10 variants and keeps the best is search, not learning. This curve is the bar every learned method
  must clear.

### E7: Amortization, the actual reason to learn
- **Question:** Trained on some workloads and designs, can a policy pick a near-optimal transform for
  an *unseen* one with zero EDA calls? How does that compare with search given k calls?
- **Do:** A contextual bandit (linear or small tree over workload and design features), trained on
  70% of (design, workload) pairs and tested on the held-out 30%.
- **Artifact:** Held-out regret: the learned policy at 0 calls vs E6 search at k = 1, 2, 4, 8.
- **Learn:** Search pays per problem. A learned policy pays once and generalizes. This is the
  concrete version of "why RL instead of an LLM plus a human loop" from the ChatGPT conversation,
  and it is what "generalize to new designs" means in the JD.

## D. Learning from reward

### E8: Policy gradient from scratch
- **Question:** On the E5 env, does REINFORCE learn? Does a baseline (mean, then group-relative à la
  GRPO) reduce gradient variance as the math says?
- **Do:** Derive ∇J = E[∇log π · A] on paper. Write about 50 lines of numpy: a softmax policy over
  actions, with no baseline, a mean baseline, and a group-relative baseline (G samples per state,
  normalized). Log gradient variance directly.
- **Artifact:** Learning curves (3 seeds) and gradient variance per variant, with the E5 oracle and
  E6 search on the same axes.
- **Learn:** Advantage, baseline, why GRPO dropped the critic, and what clip-higher and dynamic
  sampling fix (read DeepSeekMath §4, DAPO, Dr. GRPO alongside).

### E9: Frontier LLMs at three levels of autonomy
- **Question:** On the same power tasks, which mode gets the most correct power reduction per EDA
  run? (a) **Advisor**, my current Intel workflow in miniature: the model reads the power and
  synthesis reports plus the RTL and returns ranked recommendations with evidence, and I decide
  what to implement. (b) **Agent:** the model drives `lint`, `simulate`, `synth_report`,
  `power_report`, `submit` itself. (c) **Best-of-N** with grader selection. Does any mode find gains
  outside my E3 transforms? How often does each one try to cheat?
- **Predict:** Advisor has the highest hit rate per EDA run but costs my time. Agent finds more
  gains but makes more invalid attempts. Best-of-N is the strongest per dollar at small N.
- **Do:** The R2 sample size from the E5 generator (about 150 tasks, held-out families and
  workloads). Advisor mode costs my time, so run it on a stratified subset of about 30 tasks, and
  report it with its wider CI. Validate the auditor against my hand labels first (R7). Advisor mode is a new prompt in
  `chiprl/llm` that returns recommendations only, with no tool access. Log every recommendation,
  whether I accepted it, and its measured ΔP. Agent and best-of-N (N = 1/4/16, Message Batches)
  run through the existing `chiprl/llm` code. Run the auditor on every pass. Budget:
  `--max-usd 10`, only after my go-ahead.
- **Artifact:** Per mode: correct %, median ΔP with CIs, EDA runs used, ΔP per EDA run, cost per
  task, and my minutes per task in advisor mode. Hack attempts found become hack-log rows.
- **Learn:** What the human contributes, measured rather than asserted. Open action spaces, real
  reward hacking, and the best-of-N bar that RL must beat at matched samples. The advisor's logged
  (recommendation, accepted, ΔP) rows are exactly the experience data that the combined design in
  the ChatGPT conversation would train a policy on.

### E10: GRPO on a small open model
- **Question:** Does GRPO on a 0.5B to 1.5B coder improve held-out pass rate or ΔP over its own
  best-of-N at matched sample count?
- **Do:** First rejection-sampling SFT on E9 successes, then TRL `GRPOTrainer` (verl only if
  multi-turn needs it). Split train and test by *design*, not just workload. Mask tool output
  tokens, and watch entropy. Do sanity runs on MLX locally, then Modal or SkyPilot (budget from
  PLAN Phase 4).
- **Artifact:** Matched-compute curves (base, SFT, GRPO, base best-of-N) of quality vs samples and
  vs EDA calls, with paired bootstrap CIs over tasks, in-distribution and held-out-family separately
  (R5, R6). The % of tasks that regressed. Reward and entropy curves. Ablation: same run trained on
  the proxy reward vs on OpenSTA.
- **Done when:** Either result is reported with CIs, and the hand audit of 20 passes is clean.
- **Leaves behind:** A one-command training recipe (config + SkyPilot/Modal launcher) that reruns
  from a pinned image and dataset version.

## E. Breadth: the three environment types

### E11: Proxy vs ground truth, and Goodhart drift
- **Question:** On about 200 variants, how well does the Python proxy rank designs against OpenSTA,
  and against post-route LibreLane on a subset? If E10 trains on the proxy, does OpenSTA-measured
  gain shrink as proxy reward rises?
- **Do:** First improve the proxy. It is 3.6–10.7× below OpenSTA on the fixtures because it omits
  internal power, which the vendored liberty file has tables for. Then measure in three tiers on the
  same variants: proxy (every rollout), OpenSTA (sampled slice), and post-route (smaller slice). Use
  at least one non-toy design (R5).
- **Artifact:** Within-group Kendall τ and best-choice flip rate (R4), then global Spearman ρ and
  top-k, then a drift plot (proxy reward vs signoff ΔP over training steps). Frame it with the PPRTL
  vs PrimePower experience from Intel.

### E12: Verification environment
- **Question:** Can a model write cocotb tests that kill `graders/mutate.py` mutants while passing
  the clean DUT? Which vacuous-test hacks appear?
- **Artifact:** Kill rate per model, plus hack-log rows (asserts that never fire, tests that only
  check reset).

### E13: Physical-design environment
- **Question:** On a few designs, does an LLM agent choosing flow knobs (utilization, clock period,
  synthesis strategy) beat OpenROAD AutoTuner per LibreLane run?
- **Artifact:** Best PPA vs number of runs for both methods. Expect BO to win on a pure parameter
  sweep. Show where sequential decisions would change that.

---

## The headline table

Every rung from E5 on adds a row. One table, on the same held-out designs and workloads, answers
the question from the ChatGPT conversation: does learned decision-making beat search and human-guided
LLM reasoning, and at what cost in EDA runs?

| System | Rung | Correct % | Median ΔP | Constraint misses | EDA runs per task | ΔP per EDA run | Held-out regret |
|---|---|---|---|---|---|---|---|
| Baseline RTL | E1 | 100 | 0 | 0 | 0 | n/a | n/a |
| Oracle (enumeration) | E5 | | | | all | | 0 |
| Random / greedy / BO search | E6 | | | | | | |
| Contextual bandit | E7 | | | | 0 at test | | |
| LLM advisor + me | E9a | | | | | | |
| LLM agent | E9b | | | | | | |
| LLM best-of-N | E9c | | | | N | | |
| GRPO-trained model | E10 | | | | | | |

## What each rung buys in an interview

- **A:** "I know where every number in my reward comes from and how noisy it is."
- **B:** "I broke my own reward before a model could, and here is the gate that stopped each hack.
  My grader has a measured false-accept rate of zero on a labeled set, and I found and fixed a case
  where a formal tool proved a wrong design equivalent."
- **C:** "I know when a problem needs learning at all. Here is search, here is the oracle, here is
  amortization."
- **D:** "I built the workflow I run at Intel (LLM advises, engineer decides) and the RL version on
  the same problem, and measured what each one buys per EDA run. I derived GRPO, ran it, and
  compared it honestly against best-of-N at matched compute."
- **E:** "I've built envs for all three task types in the JD, and measured how much the proxy reward
  lies."

## Results index

| # | Date | Headline number | Notebook |
|---|---|---|---|
| E0 | 2026-11-13 | Reproduces natively at 36.7 rollouts/s. Area/depth differ from README (newer Yosys). Proxy is 3.6–10.7× below OpenSTA. eqy false-proves a mutant. | STACK.md "Where things run" |
| E4 (part) | 2026-11-18 | eqy proved every sequential design vacuously; fixed. Over 155 references: 0 refuted, 147 proven, grading 193 s → 23 s. Two new hacks blocked (crash-to-exclude, RTL initializer). | failure_log #1, #11–#13; hack_log #5–#6 |
| infra | 2026-11-18 | 155-task sweep: 3.7 rollouts/s at 8–10 workers, wall bound by one 120 s simulator timeout | DESIGN.md "Measured" |
