"""The one grading entry point for RTL generation: every caller that scores a submission uses it.

`grade()` is `grade_rtl` behind the result cache. Identical (task, submission, options) under the
same toolchain and grader code are graded once, across threads and runs. A result is stored only
if it is durable: infra results (reward None) and results whose equivalence check timed out (the
+0.1 bonus then depends on machine load) are recomputed next time.
"""

from __future__ import annotations

import json
import os

from .. import cache
from ..tasks.task import Task
from .rtl_gen import R, grade_rtl

CACHE_TAG = "grade/v1"  # bump only for a semantic change outside the fingerprinted files


def _durable(v: dict) -> bool:
    if v["reward"] is None:
        return False
    eq = v["detail"].get("equiv") or {}
    return not eq.get("timeout")


def cache_enabled() -> bool:
    return os.environ.get("CHIPRL_NO_CACHE", "") in ("", "0")


def grade(sub_rtl: str, task: Task, *, equiv_bonus: bool = True, dense: bool = False) -> R:
    """Cached grade_rtl. result.detail['cache_hit'] says whether this call ran any tool."""
    if not cache_enabled():
        r = grade_rtl(sub_rtl, task, equiv_bonus=equiv_bonus, dense=dense)
        r.detail["cache_hit"] = False
        return r
    opts = json.dumps({"equiv_bonus": equiv_bonus, "dense": dense}, sort_keys=True)
    k = cache.key(CACHE_TAG, task.content_hash(), sub_rtl, opts)

    def compute() -> dict:
        r = grade_rtl(sub_rtl, task, equiv_bonus=equiv_bonus, dense=dense)
        return {"reward": r.reward, "stage": r.stage, "detail": r.detail}

    v, hit = cache.get_or_compute(k, compute, store_if=_durable)
    return R(v["reward"], v["stage"], {**v["detail"], "cache_hit": hit})
