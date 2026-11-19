# The reward contract

chiprl produces three different numbers. Earlier docs used "reward" for all of them, and the README's
"reference scores 1.1" sat next to the ladder's continuous power reward as if they were the same
thing. They are not. This page is the single definition; code and docs refer to it.

| Number | Where | What it answers | Status |
|---|---|---|---|
| **Correctness score** | `grade.reward` from `graders/rtl_gen.py` (via `graders/service.grade`) | Did the RTL meet the spec? | Live. Used by every eval and baseline |
| **PPA measurement** | `ppa` from `pipeline.run_task` | What does a correct design cost? | Live. A measurement, not a reward |
| **Optimization reward** | `power_reward(candidate, original)` | Did a rewrite save power without breaking anything? | Not built. Atlas Project 6 / rung E4, written by you |

Common to all three (AGENTS rule 3): a failure of the machine or toolchain is `None` and is excluded;
a failure of the submission is `0`, never `None`. The grader tells them apart with a reference
control run (`docs/DESIGN.md`, "Blame by control run").

## 1. Correctness score (RTL generation)

Gates run in order; the first failure sets the stage and the score.

| Stage | Meaning | Score |
|---|---|---|
| `lint` | Policy violation (system task, `final`, reserved module name) or Verilator error | 0 |
| `compile` | The simulator rejects the source | 0 |
| `samples` | Wrong sample count: the run ended early | 0 |
| `sim` | Functional mismatch against the hidden testbench | 0 (`pass_frac` if `dense=True`) |
| `timeout` / `resource` | The submission hangs or kills a tool, and the reference control run is healthy | 0 |
| `ok` | Passed the hidden testbench | 1.0, or **1.1** if equivalence is `proven` |
| `infra` | The machine or toolchain failed (the reference fails too) | `None` |

- **Binary is the default.** `dense=True` gives partial credit for nearly-right sequential logic,
  a gradient toward plausible-but-wrong RTL (`docs/hack_log.md` #4). It stays behind a flag.
- **The +0.1 bonus** needs three things (`tools/equiv.py`): differential simulation on random
  stimulus finds no mismatch, a bounded check from power-up (RTL initializers stripped) finds no
  counterexample, and Yosys `equiv_*` proves it. It is a bonus because induction is incomplete for
  correct designs with a different state encoding. `refuted` on a design that passes the testbench
  means the testbench is weaker than the spec; the grade then carries
  `audit: testbench_weaker_than_spec`. That is how `Prob053` was found and quarantined.
- **pass@k** counts `score >= 1.0` as a pass. Report `mean` (infra excluded) next to
  `mean_infra_as_fail` (`llm/evals.summarize`); a gap between them means infra, not the model, is
  moving the number.
- A reference scoring 1.1 is a smoke test that the gates are wired, not a result.

## 2. PPA measurement

Reported only for a design with correctness score >= 1.0, and only when each later gate passes:

```
grade ok  →  synth ok  →  gate-level sim passes with the reference sample count  →  power ok
```

`ppa.valid` is true only if all four passed. A netlist that fails gate-level simulation gets no
power number (`docs/hack_log.md` #6: an RTL initializer passes RTL sim and fails on silicon). Power
backends: `proxy` (Python, switching + leakage, 3.6 to 10.7x below OpenSTA because it omits internal
power) or `opensta` (gate-level VCD). One backend per run: `auto` falls back silently and must not
be used to produce rewards.

## 3. Optimization reward (to build: Atlas Project 6, rung E4)

The target for the power task family, where the policy rewrites a working design to use less power.

```
r = 1[gates] · clip((P_orig − P_cand) / P_orig, −1, 1)      if |ΔP / P_orig| ≥ noise_band
r = 0                                                        if |ΔP / P_orig| <  noise_band
r = None                                                     on infra
gates = lint ∧ hidden sim ∧ samples ∧ equivalence(cand, orig) is `proven`
        ∧ gate-level sim ∧ timing met at ss, tt and ff ∧ area ≤ 1.1 × area_orig
```

What the implementation must satisfy. Each is a test you write (Project 6 asks you to write them
yourself):

1. Gate order as above, `None` only for infra, never `0` for infra and never `None` for a hang.
2. Equivalence is against the *original* design, and it is a hard gate here, unlike the bonus in §1:
   `refuted` and `unknown` both score 0 (fail closed).
3. The noise band comes from a measurement (E2: same design, different stimulus seeds), not a guess.
   State it before any result.
4. Power is scored on the hidden stimulus and audited on held-out stimulus; report the gap
   (stimulus overfitting is the expected first hack).
5. Hack tests that must score 0: a testbench memorizer, a mode deleter, an output tie-off, and a
   length-padding attack on any length penalty.
6. The timing and area gates must bind on the tasks (R3): set each task's clock at 90 to 95% of its
   Fmax, or the gate tests nothing. `tools/timing.multi_corner` gives setup and hold slack per
   corner and `min_period_ns` at the worst corner (`pipeline --corners`).
7. Equivalence against the original uses `tools/diffsim.py` first: a mismatch on random stimulus
   is a counterexample, and its held-out seeds double as the stimulus audit in obligation 4.

`configs/reward/gated.yaml` sketches this formula but no code reads it. When `power_reward` lands, it
replaces that file or is configured by it; do not keep both.
