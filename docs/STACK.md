# Stack (audited Nov 11, 2026)

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
| STA + power | **OpenSTA** (`read_vcd`, `report_power`, `read_spef`) | From the pinned LibreLane container via `scripts/bin/sta`; `docker/Dockerfile` is a from-source fallback |
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
| Frontier baselines | Claude Opus 5.5 via `anthropic` SDK 2.8: adaptive thinking, explicit effort, strict tools, prompt caching, server-side refusal fallbacks, structured outputs, Message Batches (50%) |
| Agent-in-CI | `anthropics/claude-code-action@v1` with Workload Identity Federation |
| Secrets | `ant auth login` (OAuth) locally, Keychain fallback, WIF in CI, gitleaks pre-commit |
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

## Audit, Nov 11 2026

| Item | Was | Now | Status |
|---|---|---|---|
| Yosys | apt 0.33 | OSS CAD Suite nightly (Yosys 0.70) | CI job; apt 0.33 job kept for drift |
| SV frontend | `read_verilog -sv` | yosys-slang (`read_slang`), auto-detected, fallback kept | wired; verified in CI |
| Equivalence | hand-written `equiv_*` | eqy 0.71 `sat` strategy, fallback kept | wired; test skips without eqy |
| Simulation | Icarus only | Icarus default, Verilator 5 `--binary --timing` backend | tested here: same verdicts. Verilator build 3.5 s vs Icarus 0.3 s total, so Icarus stays default for VerilogEval |
| P&R / signoff | none | LibreLane 3.0.17 (`--dockerized`, `USE_SLANG`) + post-route gate sim + OpenSTA with SPEF and VCD | wired; nightly CI job is the first real run |
| OpenSTA | custom Docker build | LibreLane image (same OpenROAD build the flow uses) | wrapper updated |
| PDK mgmt | vendored lib | vendored lib for fast path; ciel 3.2 via LibreLane for slow path | done |
| cocotb | 2.0 | 2.1 | optional `dv` extra |
| Python tools | ruff 0.8, uv-pre-commit 0.5 | ruff 0.17.4, ty 0.0.101, pytest 9.2, xdist 3.8, pre-commit 4.7, uv 0.13 | locked |
| Actions | checkout v4, setup-uv v6, upload v4, oss-cad v3 | checkout v7, setup-uv v10, upload v7, oss-cad v4 | done |
| Python | 3.12 | 3.12 kept | GPU stack (torch, vLLM, verl) is most reliable on 3.12 |

Not adopted, on purpose: synlig (superseded by yosys-slang), OpenLane 2 (superseded by LibreLane), volare (superseded by ciel).
