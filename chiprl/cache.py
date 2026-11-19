"""SQLite result cache. A key covers everything that can change a result (AGENTS rule 5).

Key = sha256 of:
  * tool versions (yosys, iverilog, verilator, sta, eqy, sby)
  * code fingerprint: the source of every module on the grading path (tools, graders, pipeline,
    task loader) and the vendored PDK files. Editing grader code invalidates the cache by itself;
    no tag needs a hand bump for that. Bump a caller's tag only for a change outside these files.
  * runtime knobs read from the environment: CHIPRL_SIM_BACKEND, CHIPRL_NO_SLANG, and whether the
    yosys-slang frontend is in use
  * the caller's parts (task hash, submission, options)

Concurrency: one connection per thread, WAL journal, busy timeout, so many grading threads (and
several processes) can share one file. `get_or_compute` is single-flight within a process: N
threads asking for the same key run the computation once. Infra results are never stored; the
caller passes `store_if` to say which results are durable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from functools import lru_cache
from pathlib import Path
from typing import Any

from .tools._run import ROOT, THIRD_PARTY, has_yosys_slang, tool_versions

DB = Path(os.environ.get("CHIPRL_CACHE", Path.home() / ".cache/chiprl/results.sqlite"))

_PKG = ROOT / "chiprl"
_CODE_PATHS = [_PKG / "tools", _PKG / "graders", _PKG / "pipeline.py", _PKG / "tasks" / "task.py"]
_PDK_FILES = ["sky130/sky130hd_tt.lib.gz", "sky130/sky130_hd.v", "sky130/sky130_hd_primitives.v"]
_ENV_KNOBS = ("CHIPRL_SIM_BACKEND", "CHIPRL_NO_SLANG")


@lru_cache(maxsize=1)
def code_fingerprint() -> str:
    h = hashlib.sha256()
    files: list[Path] = []
    for p in _CODE_PATHS:
        files += sorted(p.rglob("*.py")) if p.is_dir() else [p]
    files += [THIRD_PARTY / f for f in _PDK_FILES]
    for f in files:
        h.update(str(f.relative_to(ROOT)).encode() + b"\0")
        h.update(f.read_bytes() if f.exists() else b"<missing>")
    return h.hexdigest()[:16]


def environment() -> dict[str, Any]:
    """Everything outside the call arguments that can change a result. Also goes into run manifests."""
    return {
        "tools": tool_versions(),
        "code": code_fingerprint(),
        "env": {k: os.environ.get(k, "") for k in _ENV_KNOBS},
        "yosys_slang": has_yosys_slang(),
    }


def key(*parts: str) -> str:
    h = hashlib.sha256(json.dumps(environment(), sort_keys=True).encode())
    for p in parts:
        h.update(b"\0" + p.encode())
    return h.hexdigest()


_local = threading.local()
_init_lock = threading.Lock()
_initialized: set[Path] = set()
log = logging.getLogger(__name__)


BUSY_TIMEOUT_S = 30.0  # how long a write waits for another writer before giving up


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path, timeout=BUSY_TIMEOUT_S, isolation_level=None, check_same_thread=True)


def _init_db(path: Path) -> None:
    """Create the file, table and WAL mode once per process.

    Switching journal mode needs an exclusive lock and fails with "database is locked" instead of
    waiting if several connections race to do it on a fresh file, so it is serialized here and
    retried across processes.
    """
    with _init_lock:
        if path in _initialized:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(20):
            try:
                c = _connect(path)
                try:
                    c.execute("pragma journal_mode=wal")
                    c.execute("create table if not exists kv (k text primary key, v text)")
                finally:
                    c.close()
                break
            except sqlite3.OperationalError:
                if attempt == 19:
                    raise
                time.sleep(0.05 * (attempt + 1))
        _initialized.add(path)


def _conn() -> sqlite3.Connection:
    """Per-thread connection to the current DB path (tests repoint DB)."""
    c: sqlite3.Connection | None = getattr(_local, "conn", None)
    if c is None or getattr(_local, "path", None) != DB:
        _init_db(DB)
        c = _connect(DB)
        c.execute("pragma synchronous=normal")
        _local.conn, _local.path = c, DB
    return c


def get(k: str) -> Any:
    row = _conn().execute("select v from kv where k=?", (k,)).fetchone()
    return json.loads(row[0]) if row else None


def put(k: str, v: Any) -> None:
    _conn().execute("insert or replace into kv values (?,?)", (k, json.dumps(v, default=str)))


def _safe_get(k: str) -> Any:
    try:
        return get(k)
    except sqlite3.Error as e:  # the cache is an optimization: a broken cache must not cost a result
        log.warning("cache read failed, computing instead: %s", e)
        return None


def _safe_put(k: str, v: Any) -> None:
    try:
        put(k, v)
    except sqlite3.Error as e:
        log.warning("cache write failed, result not stored: %s", e)


_inflight: dict[str, Future] = {}
_inflight_lock = threading.Lock()


def get_or_compute(
    k: str, compute: Callable[[], Any], store_if: Callable[[Any], bool] = lambda v: True
) -> tuple[Any, bool]:
    """(value, hit). Single-flight: concurrent callers with the same key share one computation."""
    if (v := _safe_get(k)) is not None:
        return v, True
    with _inflight_lock:
        fut = _inflight.get(k)
        owner = fut is None
        if owner:
            fut = _inflight[k] = Future()
    assert fut is not None
    if not owner:
        return fut.result(), True
    try:
        v = compute()
        if store_if(v):
            _safe_put(k, v)
        fut.set_result(v)
        return v, False
    except BaseException as e:
        fut.set_exception(e)
        raise
    finally:
        with _inflight_lock:
            _inflight.pop(k, None)
