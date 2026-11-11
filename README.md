# chiprl

RL environments and graders for LLM-driven chip design: RTL generation, verification, and
**power-aware** PPA optimization (switching activity from the hidden testbench, not just area and delay).

Status: Phase 1 pipeline working. One command takes a VerilogEval v2 task through
lint, simulation, equivalence, Yosys/ABC synthesis to Sky130 HD, gate-level simulation and
switching-activity power, and emits structured JSON.

```bash
sudo apt-get install -y yosys iverilog verilator     # OpenSTA optional, see docker/
python3 -m pytest -q                                  # grader golden tests (32 pass)
python3 -m chiprl.pipeline tests/fixtures/verilogeval/Prob068_countbcd --out run.json
python3 -m chiprl.pipeline <task_dir> --rtl my_submission.sv
```

## Pipeline

```
submission.sv
  -> lint     policy (no $systasks, no final, no RefModule shadowing) + verilator --lint-only
  -> sim      iverilog vs hidden tb; nonce-tagged result line; sample count must match reference
  -> equiv    yosys equiv_make / equiv_simple -seq / equiv_induct   (+0.1 bonus if proven)
  -> synth    yosys synth + dfflibmap + abc -liberty sky130_fd_sc_hd tt_025C_1v80
  -> glsim    iverilog on the netlist + sky130 cell models, same hidden stimulus, 100 MHz, VCD
  -> power    OpenSTA read_vcd + report_power   (if `sta` on PATH)
              python proxy: sum 0.5*C_fanout*V^2*toggles/T + leakage   (always)
  -> JSON     reward, stage, area_um2, cells, flops, logic_depth, power_w, post_synth_sim_pass
```

Infra failures (missing tool, simulator killed, reference run failed) return `reward=None` and are
excluded from training and eval, never scored as 0.

## First results (reference RTL as submission, cloud CPU, proxy power backend)

| Task | Reward | Equiv | Area (um2) | Flops | Depth | P_sw+leak @100 MHz | Gate-sim | Wall (s) |
|---|---|---|---|---|---|---|---|---|
| Prob009_popcount3 | 1.1 | proven | 33.8 | 0 | 3 | 3.06 uW | pass | 1.1 |
| Prob035_count1to10 | 1.1 | proven | 143.9 | 4 | 3 | 7.16 uW | pass | 1.1 |
| Prob068_countbcd | 1.1 | proven | 646.9 | 16 | 6 | 14.80 uW | pass | 2.3 |
| Prob075_counter_2bc | 1.1 | proven | 96.3 | 2 | 4 | 4.49 uW | pass | 1.1 |

Proxy power excludes internal (short-circuit + flop clock-pin) power and wire cap, so it
undercounts versus OpenSTA. Quantifying that gap is the Phase 5 proxy study.

Grader golden tests: reference scores 1.1, 35/36 mutants killed across 4 tasks (the survivor,
`i<4 -> i<=4` in the BCD loop, is an equivalent mutant), empty module 0, constant outputs 0,
three reward hacks blocked. See [docs/hack_log.md](docs/hack_log.md).

## Layout

```
chiprl/
  tasks/      Task dataclass, VerilogEval loader (splits, decontam manifests next)
  tools/      lint, sim, equiv, synth, power -> ToolResult JSON
  graders/    rtl_gen.py (gated grader), mutate.py (mutants for golden tests + DV env)
  envs/       ChipEnv (Phase 2)
  rewards/    gated_ppa, proxies (Phase 2/5)
  baselines/  zero_shot, scaffold, best_of_n, optuna_flow (Phase 3)
  train/      rejection-sampling SFT, GRPO (Phase 4)
  analysis/   bootstrap CIs, failure taxonomy, proxy correlation (Phase 5)
  pipeline.py one task end to end
  cache.py    sqlite cache keyed on sha256(rtl, task, flags, tool versions)
configs/      env / reward / eval yaml
tests/        golden tests; fixtures are 4 VerilogEval v2 tasks (MIT, NVlabs)
third_party/  Sky130 HD liberty + cell models (Apache-2.0, via OpenSTA examples)
docker/       pinned EDA image incl. OpenSTA
```

See [docs/PLAN.md](docs/PLAN.md) for the roadmap.
