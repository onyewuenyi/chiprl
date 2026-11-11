.PHONY: setup test lint fmt typecheck smoke check clean auth eval-smoke agent-smoke
setup:      ; ./scripts/setup_mac.sh
test:       ; uv run --extra llm pytest -n auto
lint:       ; uv run ruff check . && uv run ruff format --check .
fmt:        ; uv run ruff format . && uv run ruff check --fix .
typecheck:  ; uv run ty check
smoke:      ; uv run python -m chiprl.pipeline tests/fixtures/verilogeval/Prob068_countbcd --no-cache
check: lint typecheck test
clean:      ; rm -rf .pytest_cache .ruff_cache third_party/sky130/*.chiprl.* ~/.cache/chiprl
auth:       ; uv run --extra llm python -c "from chiprl.llm.auth import credential_source as c; print(c())"
eval-smoke: ; uv run --extra llm python -m chiprl.llm.evals --tasks tests/fixtures/verilogeval --mode zero_shot --n 1 --max-usd 1
agent-smoke:; uv run --extra llm python -m chiprl.llm.evals --tasks tests/fixtures/verilogeval --mode agent --n 1 --max-usd 3
