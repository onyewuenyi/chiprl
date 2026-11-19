# Hack log

Every reward hack found (by a model or by red-teaming the grader), how it works, and how it is blocked.
Each entry has a regression test in `tests/test_golden.py`.

| # | Date | Found by | Hack | Why it works | Block | Test |
|---|------|----------|------|--------------|-------|------|
| 1 | 2026-11-11 | red-team | `initial #100 $finish;` in the DUT | VerilogEval tb prints `Mismatches: 0 in N samples` from a `final` block; ending early leaves 0 mismatches | (a) static policy bans DUT system tasks; (b) sample count must equal the reference run | `test_hack_early_finish`, `test_sample_count_guard_without_policy` |
| 2 | 2026-11-11 | red-team | `final $display("Mismatches: 0 in 40000 samples");` | grader parses stdout for the result line | per-run random nonce prefixed to the tb's result line; only nonce lines are parsed | `test_hack_spoofed_result_line` |
| 3 | 2026-11-11 | red-team | define `module RefModule` in the submission | shadow the golden model so DUT and ref agree | reserved module names rejected at lint | `test_hack_shadow_refmodule` |
| 4 | 2026-11-11 | design | dense `pass_frac` reward on a counter that is right 95% of cycles | partial credit for nearly-right sequential logic is a gradient toward plausible-but-wrong RTL | binary pass is the primary reward; `pass_frac` only behind a config flag. Measured 2026-11-18: an all-zeros module scores `pass_frac` 0.90 to 0.9997 on 7 of 155 VerilogEval tasks while failing each one | `configs/env/rtl_gen.yaml` |
| 5 | 2026-11-18 | red-team | hang or crash the simulator on a hard task | resource failures returned `reward=None`; evals drop None rows, so a crash raised pass rate instead of lowering it | the reference is re-run under the same limits: healthy reference → 0 (`timeout` / `resource`), failing reference → `None`; `summarize` also reports `mean_infra_as_fail` | `test_hack_hang_is_scored_not_excluded`, `test_hack_crash_is_scored_not_excluded`, `test_resource_failure_with_failing_reference_is_infra` |
| 6 | 2026-11-18 | red-team | `reg live = 1'b1;` gating the output | RTL sim honors the initializer, a standard-cell flop does not; RTL reward 1.0, gate-level output is x for the run | power and `ppa.valid` require a gate-level pass with the reference sample count; equivalence strips RTL initializers (`attrmap -remove init`) and the bounded check starts every flop at x, so it refutes the design and the +0.1 bonus is withheld | `tests/test_pipeline.py::test_hack_initial_value_gets_no_power` |

## Automated triage
`chiprl.llm.auditor` reads every passing trajectory and returns {verdict, category, evidence,
confidence}. Anything not `legitimate` gets a hand look; confirmed hacks become rows above.

## Expected next (PPA env)
- Power reward on a fixed stimulus overfits to that stimulus (e.g. logic that idles on patterns the tb never drives). Block: score power on held-out vectors as well, and gate on `eqy`.
- Deleting logic whose outputs the tb does not observe. Block: `eqy` equivalence gate, not just sim.
- Synthesis/sim mismatch (`initial` values, `x` optimism) passes RTL sim but not gate-level. Block: gate-level sim on the synthesized netlist is part of the pipeline (`post_synth_sim_pass`).
