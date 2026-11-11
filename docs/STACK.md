# Stack (as of Nov 2026)

How it runs: everything is local on an Apple Silicon Mac up to Phase 3. GPU work (rollouts at
scale, SFT, GRPO) runs on a rented service. Choices favor what frontier-lab infra teams actually
use, and what is open and available to me.

## EDA
| Layer | Pick | Notes |
|---|---|---|
| Front-end bundle | **OSS CAD Suite** (YosysHQ nightly) | Native darwin-arm64. Yosys, eqy, SBY, Icarus, Verilator, cocotb in one pinned dated tarball. Same tag in CI. |
| Simulation | Icarus (VerilogEval tbs), Verilator 5 (speed) | Verilator for throughput once tbs are cocotb |
| Testbenches | **cocotb 2.0** | 2.0 released Nov 2025; breaking API, so start on 2.x |
| Formal / equivalence | Yosys `equiv_*` now, **eqy** + **SBY** in Phase 2 | eqy partitions; SBY for SVA properties in the DV env |
| STA + power | **OpenSTA** (`read_vcd`, `report_power`) | Built natively for arm64 in `docker/Dockerfile`; called through `scripts/bin/sta` |
| Place and route | **OpenROAD** via **LibreLane** (OpenLane 2 successor, FOSSi, Aug 2025) or ORFS | Phase 5 ground truth only; too slow for the RL loop |
| PDK | Sky130 HD (vendored liberty + cell models); **ciel** to manage full PDKs | GF180 / IHP SG13G2 as a second PDK for generalization |

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
