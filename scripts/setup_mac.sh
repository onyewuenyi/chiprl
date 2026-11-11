#!/usr/bin/env bash
# One-time setup on Apple Silicon macOS (also works on Linux x64).
#   - OSS CAD Suite (YosysHQ nightly, native arm64): yosys, eqy, sby, iverilog, verilator, cocotb ...
#   - uv-managed Python 3.12 venv with dev tools + pre-commit hooks
#   - Optional: OpenSTA in Docker (native arm64 build), exposed on PATH as scripts/bin/sta
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${CHIPRL_EDA_HOME:-$HOME/.local/share/chiprl}"
mkdir -p "$PREFIX"

command -v uv >/dev/null || { echo "install uv first: brew install uv"; exit 1; }

if [ ! -x "$PREFIX/oss-cad-suite/bin/yosys" ]; then
  TAG="${OSS_CAD_TAG:-$(git ls-remote --tags --refs https://github.com/YosysHQ/oss-cad-suite-build.git \
        | awk -F/ '{print $3}' | grep -E '^20[0-9]{2}-[0-9]{2}-[0-9]{2}$' | sort | tail -1)}"
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) ARCH=darwin-arm64 ;;
    Darwin-x86_64) ARCH=darwin-x64 ;;
    Linux-x86_64) ARCH=linux-x64 ;;
    Linux-aarch64) ARCH=linux-arm64 ;;
    *) echo "unsupported platform"; exit 1 ;;
  esac
  URL="https://github.com/YosysHQ/oss-cad-suite-build/releases/download/$TAG/oss-cad-suite-$ARCH-${TAG//-/}.tgz"
  echo "Downloading OSS CAD Suite $TAG ($ARCH)"
  curl -fL "$URL" | tar -xz -C "$PREFIX"
  echo "$TAG" > "$PREFIX/oss-cad-suite/CHIPRL_TAG"
fi
# macOS Gatekeeper: the suite ships an `activate` script that clears the quarantine attribute.
if [ "$(uname -s)" = Darwin ] && [ -x "$PREFIX/oss-cad-suite/activate" ]; then
  "$PREFIX/oss-cad-suite/activate" >/dev/null 2>&1 || true
fi

cat > "$ROOT/.envrc" <<ENV
# Loaded by direnv (brew install direnv), or run: source .envrc
export PATH="$PREFIX/oss-cad-suite/bin:$ROOT/scripts/bin:\$PATH"
ENV

cd "$ROOT"
uv sync
uv run pre-commit install

if command -v docker >/dev/null && [ "${SKIP_STA:-0}" != 1 ]; then
  docker build -t chiprl-eda -f docker/Dockerfile . && echo "OpenSTA ready via scripts/bin/sta"
else
  echo "Docker not found or SKIP_STA=1: power uses the python proxy backend only"
fi
echo "Done. Next: source .envrc && make check && make smoke"
