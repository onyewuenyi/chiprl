"""SQLite result cache keyed on sha256(rtl + constraints + tool versions + flags)."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from functools import lru_cache
from pathlib import Path

from .tools._run import tool_versions

DB = Path(os.environ.get("CHIPRL_CACHE", Path.home() / ".cache/chiprl/results.sqlite"))


@lru_cache(maxsize=1)
def _versions() -> str:
    return json.dumps(tool_versions(), sort_keys=True)


def key(*parts: str) -> str:
    h = hashlib.sha256(_versions().encode())
    for p in parts:
        h.update(b"\0" + p.encode())
    return h.hexdigest()


def _conn() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("create table if not exists kv (k text primary key, v text)")
    return c


def get(k: str):
    with _conn() as c:
        row = c.execute("select v from kv where k=?", (k,)).fetchone()
    return json.loads(row[0]) if row else None


def put(k: str, v) -> None:
    with _conn() as c:
        c.execute("insert or replace into kv values (?,?)", (k, json.dumps(v)))
