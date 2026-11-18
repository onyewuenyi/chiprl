"""SQL over every run with zero ETL: DuckDB reads the JSONL trajectories and audits in place.

    uv run --extra scale python -m chiprl.analysis.db "select mode, avg(reward) from traj group by 1"
    uv run --extra scale python -m chiprl.analysis.db --export runs/warehouse.parquet

Views: traj (one row per attempt), audit (auditor verdicts), costs (per-run spend).
Scale path: the same SQL runs on Parquet in object storage, then on a warehouse
(BigQuery / ClickHouse) if the data ever outgrows one machine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VIEWS = {
    "traj": """create or replace view traj as
        select regexp_extract(filename, 'runs/([^/]+)/', 1) as run, *
        from read_json_auto('{root}/*/trajectories.jsonl', filename=true, union_by_name=true,
                            ignore_errors=true)""",
    "audit": """create or replace view audit as
        select regexp_extract(filename, 'runs/([^/]+)/', 1) as run, *
        from read_json_auto('{root}/*/audit.jsonl', filename=true, union_by_name=true)""",
    "costs": """create or replace view costs as
        select regexp_extract(filename, 'runs/([^/]+)/', 1) as run, mode, model, effort, n,
               cost.spent_usd as usd, cost.calls as calls
        from read_json_auto('{root}/*/summary.json', filename=true, union_by_name=true)""",
}


def connect(root: Path = Path("runs")):
    import duckdb  # ty: ignore[import-not-found]  # optional `scale` extra

    con = duckdb.connect()
    root = root.resolve()
    for name, sql in VIEWS.items():
        pattern = {"traj": "trajectories.jsonl", "audit": "audit.jsonl", "costs": "summary.json"}[name]
        if any(root.glob(f"*/{pattern}")):
            con.execute(sql.format(root=root.as_posix()))
    return con


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sql", nargs="?")
    ap.add_argument("--root", type=Path, default=Path("runs"))
    ap.add_argument("--export", type=Path, help="write the traj view to Parquet")
    a = ap.parse_args(argv)
    con = connect(a.root)
    if a.export:
        con.execute(f"copy (select * exclude (events) from traj) to '{a.export}' (format parquet)")
        print(f"wrote {a.export}")
    if a.sql:
        print(con.sql(a.sql))
    return 0


if __name__ == "__main__":
    sys.exit(main())
