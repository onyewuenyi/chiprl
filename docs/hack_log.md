# Hack log

Every reward hack found (by a model or by red-teaming the grader), how it works, and how it is blocked.
Each entry has a regression test in `tests/test_golden.py`.

| # | Date | Found by | Hack | Why it works | Block | Test |
|---|------|----------|------|--------------|-------|------|
| 1 | 2026-11-11 | red-team | `initial #100 $finish;` in the DUT | VerilogEval tb prints `Mismatches: 0 in N samples` from a `final` block; ending early leaves 0 mismatches | (a) static policy bans DUT system tasks; (b) sample count must equal the reference run | `test_hack_early_finish`, `test_sample_count_guard_without_policy` |
| 2 | 2026-11-11 | red-team | `final $display("Mismatches: 0 in 40000 samples");` | grader parses stdout for the result line | per-run random nonce prefixed to the tb's result line; only nonce lines are parsed | `test_hack_spoofed_result_line` |
| 3 | 2026-11-11 | red-team | define `module RefModule` in the submission | shadow the golden model so DUT and ref agree | reserved module names rejected at lint | `test_hack_shadow_refmodule` |
| 4 | 2026-11-11 | design | dense `pass_frac` reward on a counter that is right 95% of cycles | partial credit for nearly-right sequential logic is a gradient toward plausible-but-wrong RTL | binary pass is the primary reward; `pass_frac` only behind a config flag | `configs/env/rtl_gen.yaml` |

## Expected next (PPA env)
- Power reward on a fixed stimulus overfits to that stimulus (e.g. logic that idles on patterns the tb never drives). Block: score power on held-out vectors as well, and gate on `eqy`.
- Deleting logic whose outputs the tb does not observe. Block: `eqy` equivalence gate, not just sim.
- Synthesis/sim mismatch (`initial` values, `x` optimism) passes RTL sim but not gate-level. Block: gate-level sim on the synthesized netlist is part of the pipeline (`post_synth_sim_pass`).
