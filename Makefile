.PHONY: setup test lint fmt typecheck smoke check clean
setup:      ; ./scripts/setup_mac.sh
test:       ; uv run pytest -n auto
lint:       ; uv run ruff check . && uv run ruff format --check .
fmt:        ; uv run ruff format . && uv run ruff check --fix .
typecheck:  ; uv run ty check
smoke:      ; uv run python -m chiprl.pipeline tests/fixtures/verilogeval/Prob068_countbcd --no-cache
check: lint typecheck test
clean:      ; rm -rf .pytest_cache .ruff_cache third_party/sky130/*.chiprl.* ~/.cache/chiprl
