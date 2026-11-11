# Stack (as of Nov 2026)

How it runs: everything is local on an Apple Silicon Mac up to Phase 3. GPU work (rollouts at
scale, SFT, GRPO) runs on a rented service. Choices favor what frontier-lab infra teams actually
use, and what is open and available to me.

## EDA
| Layer | Pick | Notes |
|---|---|---|
| Front-end bundle | **OSS CAD Suite** (YosysHQ nightly) | Native darwin-arm64. Yosys, eqy, SBY, Icarus, Verilator, cocotb in one pinned dated tarball. Same tag in CI. |
| SV frontend | **yosys-slang** | Full SystemVerilog for Yosys; `read_verilog -sv` is partial. Fallback kept for apt Yosys |
| Simulation | Icarus (VerilogEval tbs today), **Verilator 5 `--timing`** | Verilator runs event-driven tbs and is the throughput path |
| Testbenches | **cocotb 2.0** | 2.0 released Nov 2025; breaking API, so start on 2.x |
| Formal / equivalence | **eqy** + **SBY** (replacing the hand-written `equiv_*` script) | eqy partitions; SBY for SVA properties in the DV env |
| STA + power | **OpenSTA** (`read_vcd`, `report_power`) | Built natively for arm64 in `docker/Dockerfile`; called through `scripts/bin/sta` |
| Place and route | **OpenROAD** via **LibreLane** (OpenLane 2 successor, FOSSi, Aug 2025), used as a Python library of Steps | Adopted now as the slow ground-truth path on a ~10% audit slice; minutes per design, so never in the inner loop |
| Flow tuning baseline | **OpenROAD AutoTuner** (Ray Tune) | Non-LLM PPA baseline instead of hand-rolled Optuna |
| Waveforms | **Surfer** | Modern GTKWave replacement for debugging |
| PDK | Sky130 HD (vendored liberty + cell models); **ciel** (volare successor) to manage full PDKs | GF180 / IHP SG13G2 as a second PDK for generalization |

## Python and engineering
| Concern | Pick |
|---|---|
| Env + lockfile | **uv** (`uv.lock`, `uv sync --locked` in CI) |
| Lint + format | **ruff** |
| Type check | **ty** (Astral) |
| Tests | pytest + pytest-xdist; golden tests are the CI gate |
| Hooks | pre-commit (ruff, uv-lock, size guard) |
| CI | GitHub Actions: lint/type, golden on OSS CAD Suite, golden on apt Yosys 0.33 |
| Deps hygiene | Dependabot for actions + uv |
| Containers | Docker, `--network none`, non-root, read-only repo mount |
| Agent docs | `AGENTS.md` (Codex and others) + `CLAUDE.md` importing it (Claude Code) |

## LLM and RL (Phase 3+)
| Concern | Pick |
|---|---|
| Frontier baselines | Claude (Anthropic API, Agent SDK for the tool-using scaffold) and OpenAI API, same tool interface |
| Local small-model iteration | MLX-LM on the Mac for 0.5B to 3B sanity runs |
| Rollout inference | vLLM (SGLang as alternative) on the GPU service |
| RL | **verl** (multi-turn tool rollouts, GRPO/DAPO built in) or TRL `GRPOTrainer` for the first run |
| GPU service | **Modal** for bursty jobs (pay per second, Python-defined images); RunPod / Lambda for multi-hour runs |
| Tracking | Weights & Biases |
| Config | Hydra-style YAML in `configs/` (pydantic models once the env API settles) |

## Deliberately not used
- Hydra CLI magic before there are enough configs to justify it.
- Ray before a single-node process pool is the bottleneck.
- Commercial EDA. Everything must be reproducible by a reviewer.

## Two-speed evaluation
- Fast path, every rollout (~1-2 s): lint, sim, Yosys/ABC, gate-level sim, OpenSTA or proxy power.
- Slow path, sampled (~minutes): LibreLane post-route area, WNS, power with parasitics. Feeds the proxy study and Goodhart-drift audits from Phase 1 on.
