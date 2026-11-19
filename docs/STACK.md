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
| RL | **TRL 1.18 `GRPOTrainer` + PEFT LoRA** (`chiprl/train/grpo.py`); verl only if multi-turn rollouts at scale need it |
| GPU service | **Modal** for bursty jobs (pay per second, Python-defined images); RunPod / Lambda for multi-hour runs |
| Tracking | MLflow (`chiprl/tracking.py`; local SQLite store in `runs/`, `make mlflow-ui`) |
| Config | Hydra-style YAML in `configs/` (pydantic models once the env API settles) |

## Deliberately not used
- Hydra CLI magic before there are enough configs to justify it.
- Ray before a single-node process pool is the bottleneck.
- Commercial EDA. Everything must be reproducible by a reviewer.

## Two-speed evaluation
- Fast path, every rollout (~1-2 s): lint, sim, Yosys/ABC, gate-level sim, OpenSTA or proxy power.
- Slow path, sampled (~minutes): LibreLane post-route area, WNS, power with parasitics. Feeds the proxy study and Goodhart-drift audits from Phase 1 on.

## Where things run (measured Nov 13 2026, Apple M5 10-core, 24 GB)

Decision: EDA fast path **native** (OSS CAD Suite darwin-arm64). Docker only for OpenSTA, OpenROAD
and LibreLane, which have no macOS builds. Docker becomes the default on Linux cloud boxes, where
it also sandboxes model-written RTL.

| Component | Where | Measured |
|---|---|---|
| Yosys 0.70, Icarus 14, Verilator 5.055, eqy 0.71, SBY | native | `make test`: 46 tests in 10 s (1 real failure, see below) |
| Fast path per task, proxy power | native | 1.0–1.9 s |
| Fast path per task, OpenSTA power | native + Docker `sta` | 1.3–2.3 s (image is arm64, 7.5 GB) |
| Throughput (`make bench`, 128 jobs) | native | 36.7 graded rollouts/s, p50 0.20 s, p95 0.60 s, 0 infra errors (cloud box: 21.6/s) |
| Full sweep (`make bench-full`, 155 tasks, 516 submissions, Nov 18) | native | 3.7 graded rollouts/s cold at 8 to 10 workers, p50 0.15 s, p95 0.66 s; wall bound by one 120 s simulator timeout (see DESIGN.md); warm cache 11k to 32k/s |
| Verilator `--timing` vs Icarus | native | 2.37 s vs 0.17 s sim on countbcd; Icarus stays default |
| LibreLane RTL-to-GDS (countbcd) | Docker | 63 s, WNS +3.13 ns, 2814 instances incl. fill/tap. Needs a pty (see below) |
| Small-model generation (MLX, Qwen2.5-Coder-1.5B 4-bit) | native | 116 tok/s single stream, 1.0 GB |
| Claude API (E9) | API | spend, not hardware; `--max-usd` |
| vLLM / verl / TRL+vLLM GRPO, 7B-class models (E10 at scale) | **cloud GPU** | CUDA only. 1–2× 80 GB GPUs via SkyPilot/Modal |

OpenSTA vs proxy on the 4 fixtures (reference RTL, 100 MHz): 10.8 vs 3.0, 48.5 vs 8.1, 156.3 vs
14.7, 24.8 vs 4.7 µW. Internal power (which the proxy omits) is 78% of countbcd's OpenSTA total.
Area and depth also differ from the README's cloud numbers (countbcd 611.8 vs 628.1 µm², depth 8
vs 6): a newer Yosys nightly, which is why the cache keys on tool versions.

Problems found during this run (eqy false proof, LibreLane TTY, `ty` extras, exec bits) are tracked in
[failure_log.md](failure_log.md).

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

## Frontier-lab alignment and scale-up (Nov 11 2026)

Rule: adopt a layer when a measured bottleneck asks for it, not before. Claims about internal lab
stacks are hedged to what is public.

| Layer | Pick | Lab alignment (public evidence) | Adopt when |
|---|---|---|---|
| Fan-out | thread pool -> **Ray** (`chiprl/runtime/executor.py`, `CHIPRL_BACKEND=ray`) | OpenAI has said it used Ray to train ChatGPT-era models; verl, OpenRLHF and SkyRL all run on Ray | Now as a flag. Default when grading is the RL bottleneck or goes multi-node |
| Rollout + RL | **vLLM** + **verl** (TRL fallback) | Most open RL-for-LLM reproductions (DAPO, VerlTool, ChipSeek-style work) | Phase 4 |
| GPU provisioning | **SkyPilot** (`infra/sky/grpo.yaml`) over RunPod/Lambda; **Modal** for short bursty jobs | SkyPilot is the common open multi-cloud launcher and ships a verl example | First GPU run |
| Tracking | **MLflow** (`evals --mlflow`, `train.grpo --mlflow`) | Open source, self-hosted, no account; switched from W&B on Nov 18 to keep the whole stack reproducible offline | Now |
| Run data | JSONL per run -> **DuckDB** views + Parquet export (`chiprl/analysis/db.py`) | Columnar files + SQL is the standard pattern; scales to many GB on one laptop | Now |
| Warehouse | BigQuery or ClickHouse | Only if data outgrows one machine | Probably never for this project |
| Cluster | KubeRay on k8s | How Ray runs in production at large orgs | Only with a persistent multi-node cluster |
| LLM | Anthropic SDK (Claude Opus 5.5), OpenAI SDK for a second frontier baseline | The labs' own APIs | Now / Phase 3 |

Not adopted: **Metaflow** (Netflix/Outerbounds ML workflow orchestration; good tool, but not what RL
infra at frontier labs is known to run, and Ray + SkyPilot already cover scheduling here).
Airflow/Kubeflow for the same reason.

Measured (cloud box, 4 fixtures x 8 variants, `make bench`): local threads 21.6 graded rollouts/s,
p50 0.08 s; Ray 10.8/s. At this scale Ray loses to worker startup and serialization. It pays off at
thousands of rollouts per step or more than one node, which is exactly the Phase 4 regime.
