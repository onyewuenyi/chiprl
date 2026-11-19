.PHONY: train-smoke mlflow-ui setup test lint fmt typecheck smoke check clean auth eval-smoke agent-smoke bench bench-full bench-ray db
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
bench:      ; uv run python -m chiprl.runtime.bench --reps 4
bench-full: ; uv run python -I scripts/fetch_verilogeval.py && uv run python -m chiprl.runtime.bench --tasks data/verilogeval_v2/dataset_spec-to-rtl --mutants 3 --workers 1 4 8 10 --out runs/bench.json
bench-ray:  ; uv run --extra scale python -m chiprl.runtime.bench --backend ray --reps 4
db:         ; uv run --extra scale python -m chiprl.analysis.db --export runs/traj.parquet
train-smoke:; CHIPRL_TRAIN_SMOKE=1 uv run --extra train pytest -q -p no:xdist tests/test_train.py
mlflow-ui:  ; uv run --extra scale mlflow ui --backend-store-uri sqlite:///runs/mlflow.db
