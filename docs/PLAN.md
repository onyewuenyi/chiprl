# chiprl plan (personalized)

Goal: a public repo that is the evidence for a Research Engineer, AI for Chip Design role, with the
Software Engineer evidence kept as a by-product. The research question:

> Can a trained model learn RTL transformations that give statistically defensible power
> improvements, keep functional correctness, and generalize to unseen designs better than search
> and best-of-N at matched compute?

Reward Atlas is the curriculum for it ([docs/CURRICULUM.md](CURRICULUM.md)); the experiment ladder
([docs/EXPERIMENTS.md](EXPERIMENTS.md)) is the order of work; [docs/REWARD.md](REWARD.md) defines
what is scored.

| Keep | Increase emphasis |
|---|---|
| Reliable graders and EDA infrastructure (`docs/DESIGN.md`) | Reward validity and measurement uncertainty (E2, E4, E11) |
| Caching and throughput | Experimental design and statistics (R2, `analysis/stats.py`) |
| Tool-using agents | Policy learning and model behavior (E8, E10) |
| Reproducible pipelines | Ablations and held-out generalization (`tasks/splits.py`) |
| PPA optimization | Evidence that RL beats search and best-of-N (E6, E9, E10) |

History: until Nov 18 this plan targeted the Software Engineer role first, with Research Engineer as
the stretch, so infrastructure got ahead of the science (EXPERIMENTS.md says the same). The
infrastructure is now in good shape (`docs/DESIGN.md`); the next work is E1 to E4.

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
- [x] Golden tests: ref 1.1, 35/36 mutants killed (1 equivalent), empty 0, constant 0, every logged hack blocked
- [x] eqy false proof root-caused and fixed; equivalence fails closed (failure_log #1, #11, #12)
- [x] OpenSTA backend verified in Docker on my machine; proxy vs OpenSTA on the 4 fixtures (STACK.md)
- [x] LibreLane slow path wired (`--pnr`); first real run is the nightly CI job / my Mac
- [x] yosys-slang frontend, eqy equivalence, Verilator --timing backend (all feature-detected)
- [x] All 156 VerilogEval v2 tasks at a pinned commit (155 usable); structural family labels; train / test_iid / test_ood split by group with a leakage test
- [ ] Decontamination manifest (n-gram overlap vs CVDP)
- [ ] pass@1 / pass@5 table for 2 to 3 models (one small open model via vLLM or API, one frontier)

### Phase 2: DV + PPA envs (Nov 18 to Nov 30) -> resume link goes live
- [ ] DV env: model writes cocotb tests; reward = kill rate x 1[clean DUT passes]; reject vacuous asserts
- [ ] PPA env: gates = hidden tests + eqy; reward = clip((P_base - P_sub)/P_base) on power, area, delay
- [ ] 20 power-family tasks (point 5 above)
- [ ] Multi-turn tool API: lint, sim, synth_report, power_report, submit (JSON only)
- [ ] Docker sandbox: `--network none`, read-only grader mounts, non-root, rlimits (rlimits + process-group kill done)
- [x] Throughput benchmark (`make bench`, local vs Ray); CI badge pending
- [x] Scale-up layer: Ray executor, DuckDB over runs, MLflow tracking (replaced W&B), SkyPilot GPU recipe

### Phase 3: baselines (Dec 1 to Dec 7), code landed Nov 11
- [x] Claude zero-shot + tool-using agent baselines, pass@k + CIs, cost caps, JSONL trajectories
- [x] Best-of-N through Message Batches; reward-hack auditor
- [ ] First paid run on the 4 fixtures (~$1 to $3), then 100 VerilogEval tasks
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

The learning path is now the experiment ladder in [EXPERIMENTS.md](EXPERIMENTS.md): 14
first-principles experiments (E0 to E13), each answering one question with a reproducible artifact.
The phases above are the build work those experiments pull in. Reading list, by rung:

| Rung | Read |
|---|---|
| E1–E3 | Sky130 HD liberty for `edfxtp` / `dlclkp`; Yosys `dfflibmap` and `abc` docs; OpenSTA power docs |
| E5–E7 | Spinning Up (VPG, PPO intuition); a contextual-bandit chapter (Lattimore & Szepesvári, ch. 18–19) |
| E8 | DeepSeekMath section 4 (GRPO); DAPO; Dr. GRPO |
| E9–E10 | ChipSeek + VeriReason code end to end; VerlTool; verl multi-turn docs; TRL GRPOTrainer |
| E12 | SBY / eqy docs; CVDP paper |

## Interview artifacts this produces
1. Hack log with regression tests (already 3 entries).
2. Proxy vs signoff power correlation plot, framed by my PPRTL vs PrimePower experience.
3. Matched-compute RL vs best-of-N table with CIs.
4. Throughput and reliability numbers for the grader service.
