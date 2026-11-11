# chiprl plan (personalized)

Goal: a public repo that proves I can build the infrastructure the OpenAI AI for Chip Design team
runs every day: trustworthy graders, sandboxed EDA tool environments, baselines, an RL run, and an
honest analysis. Primary target is the Software Engineer role, so infra quality, throughput and
reproducibility are first-class results, not afterthoughts. Research Engineer is the stretch.

Deadline logic: it is Nov 11. Applications should not wait for RL. The repo is resume-linkable
once Phase 2 lands (target Nov 30). RL results are interview material in December.

## Why this angle is mine

Ten years of CPU power analytics at Intel: PPRTL RTL power estimation, PrimePower/PrimeTime,
StarRC/SPEF, FE/BE hierarchy mapping, Python automation on a batch-compute farm. Prior art
(ChipSeek, VeriReason) optimizes area and delay. Switching-activity power is underexplored, and
"how well does a cheap RTL power estimate track signoff" is a question I have answered
professionally (PPRTL vs PrimePower). The open-source mapping:

| Intel flow | chiprl |
|---|---|
| Fusion Compiler / DC | Yosys + ABC, Sky130 HD |
| PPRTL (RTL-stage estimate) | `tools/power.py` proxy (toggle-weighted fanout cap) |
| PrimePower with FSDB/VCD | OpenSTA `read_vcd` + `report_power` on gate-level VCD |
| StarRC SPEF | OpenROAD post-route parasitics (Phase 5 only) |
| Netbatch job farm | process pool / Ray + sqlite result cache |

Gaps to close on purpose: RL and post-training (biggest), formal (eqy/SBY), RTL generation as an
ML task. The learning path below is weighted toward those.

## Changes from the original plan, and why

1. **Binary pass is the primary RTL reward.** `pass_frac` gives credit to sequential logic that is
   right most cycles, which is a gradient toward plausible-but-wrong RTL. Kept behind a flag.
2. **Gate-level sim is part of the pipeline, not optional.** It (a) catches synth/sim mismatch
   (`initial`, x-optimism), and (b) gives 100% net annotation for power. An RTL VCD only annotates
   ports and registers; OpenSTA propagates default activity through everything else, which is
   noise you would then train on.
3. **Normalize the clock.** VerilogEval testbenches use 1 ps timescale (a 100 GHz clock).
   The power tb is rescaled to 1 ns so power is reported at 100 MHz.
4. **Stimulus overfitting is a power-specific hack.** Score power on the hidden tb (train) and on
   held-out random vectors (audit). Report the gap.
5. **Build a power-specific task family I own** (also the contamination fix): operand isolation on
   multipliers, clock-enable insertion, one-hot vs binary vs Gray FSM/counter encodings, bus
   inversion, glitch-heavy adder chains. Generated programmatically with reference + hidden tb, so
   it is guaranteed not in pretraining. These are the tasks where a power reward can move.
6. **Report throughput as a result.** Graded rollouts/sec/core, cache hit rate, p50/p95 grader
   latency, infra-error rate. This is the SWE-role headline.
7. **OpenSTA is ground truth, the Python proxy is the cheap reward.** That makes the Phase 5 proxy
   study fall out of the stack for free: Spearman and top-k agreement between proxy and OpenSTA,
   then Goodhart drift when training on the proxy.

## Phases and dates

### Phase 1: eval + RTL-gen env (Nov 11 to Nov 18)
- [x] Repo scaffold, tool wrappers to JSON, sqlite cache keyed on tool versions
- [x] One VerilogEval task through lint, sim, equiv, synth, gate-sim, power
- [x] Golden tests: ref 1.1, 35/36 mutants killed (1 equivalent), empty 0, constant 0, 3 hacks blocked
- [ ] OpenSTA backend verified in Docker on my machine; record proxy vs OpenSTA on the 4 fixtures
- [x] LibreLane slow path wired (`--pnr`); first real run is the nightly CI job / my Mac
- [x] yosys-slang frontend, eqy equivalence, Verilator --timing backend (all feature-detected)
- [ ] Load 100 VerilogEval v2 tasks; family tags; train/val split; decontam manifest (n-gram overlap vs CVDP)
- [ ] pass@1 / pass@5 table for 2 to 3 models (one small open model via vLLM or API, one frontier)

### Phase 2: DV + PPA envs (Nov 18 to Nov 30) -> resume link goes live
- [ ] DV env: model writes cocotb tests; reward = kill rate x 1[clean DUT passes]; reject vacuous asserts
- [ ] PPA env: gates = hidden tests + eqy; reward = clip((P_base - P_sub)/P_base) on power, area, delay
- [ ] 20 power-family tasks (point 5 above)
- [ ] Multi-turn tool API: lint, sim, synth_report, power_report, submit (JSON only)
- [ ] Docker sandbox: `--network none`, read-only grader mounts, non-root, rlimits
- [ ] Throughput benchmark + CI badge

### Phase 3: baselines (Dec 1 to Dec 7)
- zero-shot; scaffold (plan, edit, lint, sim, fix; K turns); best-of-N with grader selection at
  N = 4/16/64 (the bar RL must beat at matched samples); frontier model ceiling; OpenROAD AutoTuner
  (Ray Tune) over Yosys/OpenROAD knobs as the non-LLM PPA baseline.

### Phase 4: training (Dec 8 to Dec 21), 1 to 2 rented GPUs, budget about $200 to $400
- Rejection-sampling SFT on best-of-N successes (LoRA, Qwen-class 1.5B to 7B coder).
- GRPO from the SFT checkpoint: dynamic sampling (drop all-pass/all-fail groups), clip-higher,
  token-level loss, tool-output tokens masked, KL ~0, watch entropy.
- Curriculum: tasks where base pass rate is 10 to 90%. DPO on (pass, fail) pairs as a cheap comparison.

### Phase 5: experiments and proxy study (Dec 22 to Jan 4)
- 3 seeds, paired bootstrap 95% CIs, % of tasks regressed; held-out families and CVDP.
- Failure taxonomy: infra / grader / reward / behavior; hand-audit 20 passes per run; hack log.
- Proxy study on ~200 variants: proxy vs OpenSTA (and OpenROAD post-route for a subset).

### Phase 6: packaging (continuous)
- One command per run, pinned image digest, versioned datasets, CI on golden tests, README with
  results table, hack log, proxy correlation plot.

## Learning path, weighted to my gaps

| Week | Read | Do |
|---|---|---|
| Nov 11 | Spinning Up: VPG, PPO intuition | Phase 1; derive the REINFORCE gradient by hand |
| Nov 18 | DeepSeekMath (GRPO) section 4; DAPO; Dr. GRPO | write a 50-line GRPO loss on a toy bandit |
| Nov 25 | ChipSeek + VeriReason code end to end | map their reward code onto chiprl graders |
| Dec 2 | VerlTool paper, verl multi-turn docs; TRL GRPOTrainer | run TRL GRPO on a 0.5B model on 10 tasks |
| Dec 9 | SBY / eqy docs; CVDP paper | add SVA gate to the DV env |

## Interview artifacts this produces
1. Hack log with regression tests (already 3 entries).
2. Proxy vs signoff power correlation plot, framed by my PPRTL vs PrimePower experience.
3. Matched-compute RL vs best-of-N table with CIs.
4. Throughput and reliability numbers for the grader service.
