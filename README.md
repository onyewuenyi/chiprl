# chiprl

RL environments and graders for LLM-driven chip design: RTL generation, verification, and
**power-aware** PPA optimization (switching activity from the hidden testbench, not just area and delay).

One command takes a VerilogEval v2 task through lint, simulation, equivalence, Yosys/ABC synthesis
to Sky130 HD, gate-level simulation and switching-activity power, and emits structured JSON.

| Area | State |
|---|---|
| Grader (lint, sim, equivalence, synth, gate-level sim, power) | Works. 155 of 156 VerilogEval references pass, 0 refuted; 2 upstream tasks quarantined (one reference does not compile, one testbench accepts any design). Zero-false-accept rule tested on mutants and 6 logged reward hacks |
| Grading service (cache, executor, runner) | Works. Cached single entry point, streaming executor with backpressure and infra-only retry, process-group kill on timeout, fault-injection tests. Design and numbers in [docs/DESIGN.md](docs/DESIGN.md) |
| Eval harness (Claude zero-shot, agent, best-of-N, auditor) | Works with a fake client in tests. Resumable runs with a provenance manifest. No paid run yet |
| Differential simulation, behavior clustering, multi-corner timing | Works. Random-stimulus refuter (found a broken upstream testbench), AlphaCode-style cluster selection, OpenSTA at ss/tt/ff (`pipeline --corners`, needs Docker) |
| Task export | Works. Each task as container + instruction + grader-only verifier + score (`python -m chiprl.tasks.export`), run end to end in Docker |
| Task splits, statistics, eval reports | Works. 154 tasks in train / test_iid / test_ood by structural family, leakage-tested; Wilson, paired bootstrap, McNemar; `python -m chiprl.analysis.report` |
| Power reward, PPA env, baselines | Planned: E4 to E7. `chiprl/rewards`, `envs` (beyond `RTLGenEnv`), `baselines` are stubs |
| Open-model training (SFT, GRPO) | Planned. `chiprl/train` is a stub; `infra/sky/verl_grpo.yaml` has not been run |

**Research question:** can a trained model learn RTL transformations that give statistically
defensible power improvements, keep functional correctness, and generalize to unseen designs better
than search and best-of-N at matched compute? The order of work is the experiment ladder
([docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)); what is scored is [docs/REWARD.md](docs/REWARD.md); the
curriculum that goes with it is [docs/CURRICULUM.md](docs/CURRICULUM.md).

Runs locally on an Apple Silicon Mac. GPU phases use a rented service (see [docs/STACK.md](docs/STACK.md)).

```bash
brew install uv                # plus Docker Desktop if you want OpenSTA
./scripts/setup_mac.sh         # OSS CAD Suite (arm64), uv venv, pre-commit, OpenSTA image
source .envrc
make check                     # ruff + ty + grader golden tests
make smoke                     # one VerilogEval task end to end -> JSON
make bench-full                # fetch 156 VerilogEval tasks (pinned) + throughput sweep -> runs/bench.json
uv run python -m chiprl.pipeline tests/fixtures/verilogeval/Prob068_countbcd --pnr   # + post-route
uv run python -m chiprl.pipeline <task_dir> --rtl my_submission.sv
```

## Claude in the loop

| Feature | Command | What it gives you |
|---|---|---|
| Frontier baseline, zero-shot | `make eval-smoke` | pass@k with bootstrap CIs, cost, JSONL trajectories |
| Tool-using agent baseline | `make agent-smoke` | Claude drives lint / simulate / synth_report / submit through the same env RL will use |
| Best-of-N via Message Batches | `python -m chiprl.llm.batch submit --n 16` | 50%-price sampling; oracle pass@N vs deployable visible-signal BoN |
| Reward-hack auditor | `python -m chiprl.llm.auditor runs/<run>/trajectories.jsonl --max-usd 2` | second model flags passes that look like hacks; feeds docs/hack_log.md |
| Claude Code on GitHub | `@claude` on issues/PRs; auto review of reward-path PRs | reviews focused on grader integrity, auth via OIDC federation |

Credentials: `ant auth login` locally, Workload Identity Federation in CI, no static keys in the
repo or GitHub. Every run takes a hard `--max-usd` cap. Details in [docs/SECURITY.md](docs/SECURITY.md).

## Pipeline

```
submission.sv
  -> lint     policy (no $systasks, no final, no RefModule shadowing) + verilator --lint-only
  -> sim      iverilog vs hidden tb; nonce-tagged result line; sample count must match reference
  -> equiv    bounded check from power-up + yosys equiv_*, both required (+0.1 bonus if proven)
  -> synth    yosys-slang frontend, synth + dfflibmap + abc, sky130_fd_sc_hd tt (dont_use filtered)
  -> glsim    iverilog on the netlist + sky130 cell models, same hidden stimulus, 100 MHz, VCD
              (must pass with the reference sample count, or no power number is reported)
  -> power    OpenSTA read_vcd + report_power   (if `sta` on PATH)
              python proxy: sum 0.5*C_fanout*V^2*toggles/T + leakage   (always)
  -> [--pnr] LibreLane RTL-to-GDS, post-route gate sim, OpenSTA with SPEF + VCD (slow path)
  -> JSON     reward, stage, area_um2, cells, flops, logic_depth, power_w, post_synth_sim_pass
```

Infra failures (missing tool, simulator killed, reference run failed) return `reward=None` and are
excluded from training and eval, never scored as 0.

## Fixture results (reference RTL as submission, M5 Mac, OSS CAD Suite 2026-11-12, proxy power)

| Task | Reward | Equivalence | Area (um2) | Flops | Depth | P_sw+leak @100 MHz | Gate-sim | Wall (s) |
|---|---|---|---|---|---|---|---|---|
| Prob009_popcount3 | 1.1 | proven | 33.8 | 0 | 3 | 3.00 uW | pass | 0.8 |
| Prob035_count1to10 | 1.1 | proven | 148.9 | 4 | 3 | 8.09 uW | pass | 0.8 |
| Prob068_countbcd | 1.1 | proven | 611.8 | 16 | 8 | 14.65 uW | pass | 1.5 |
| Prob075_counter_2bc | 1.1 | proven | 102.6 | 2 | 3 | 4.73 uW | pass | 0.9 |

Wall times were measured while a benchmark ran on the same machine. Proxy power excludes internal
(short-circuit + flop clock-pin) power and wire cap, so it reads 3.6 to 10.7x below OpenSTA on these
fixtures (docs/STACK.md). Quantifying that gap is the proxy study.

The 1.1 is the correctness score with the equivalence bonus ([docs/REWARD.md](docs/REWARD.md) §1), a smoke test of the gates, not a result. Grader golden tests: reference scores 1.1, mutants killed (equivalent mutants listed), empty module
0, constant outputs 0, and every reward hack in [docs/hack_log.md](docs/hack_log.md) blocked by a
regression test. Grader and infrastructure failures found along the way are in
[docs/failure_log.md](docs/failure_log.md).

## Layout

```
chiprl/
  tasks/      Task, VerilogEval loader, structural families, splits + committed manifest
  tools/      lint, sim, equiv, synth, power -> ToolResult JSON
  graders/    rtl_gen.py (gated grader), service.py (cached entry point), mutate.py (mutants)
  envs/       ChipEnv (Phase 2)
  rewards/    gated_ppa, proxies (Phase 2/5)
  baselines/  zero_shot, scaffold, best_of_n, optuna_flow (Phase 3)
  train/      rejection-sampling SFT, GRPO (Phase 4)
  analysis/   stats (Wilson, paired bootstrap, McNemar), run reports, DuckDB views
  pipeline.py one task end to end
  cache.py    sqlite cache keyed on task, submission, options, tool versions, grader source
configs/      env / reward / eval yaml
tests/        golden tests; fixtures are 4 VerilogEval v2 tasks (MIT, NVlabs)
third_party/  Sky130 HD liberty + cell models (Apache-2.0, via OpenSTA examples)
docker/       arm64-native OpenSTA image
scripts/      setup_mac.sh, bin/sta (Docker wrapper)
```

See [docs/PLAN.md](docs/PLAN.md) for the roadmap and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
for the experiment ladder (E0 to E13) that drives it.
