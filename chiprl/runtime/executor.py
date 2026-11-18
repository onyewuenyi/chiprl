"""Fan out grading jobs. Scale-up path, switch with one flag:

  local  thread pool on this machine (each job is a set of EDA subprocesses, so threads suffice)
  ray    Ray tasks: local multi-core with `ray.init()`, or a cluster via RAY_ADDRESS. The same
         Ray runtime later hosts verl's actors, so there is one scheduler for rollouts and grading.

LLM API calls never go through Ray: they are IO-bound and the client is not picklable.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def _ray_map[T](fn: Callable[[Any], T], items: list[Any], cpus_per_job: float) -> Iterator[T]:
    # Under `uv run`, Ray tries to re-create the uv env in each worker; workers should reuse ours.
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    import ray  # ty: ignore[import-not-found]  # optional `scale` extra

    if not ray.is_initialized():
        ray.init(address=os.environ.get("RAY_ADDRESS"), ignore_reinit_error=True, log_to_driver=False)
    remote = ray.remote(num_cpus=cpus_per_job)(fn)
    refs = [remote.remote(x) for x in items]
    yield from ray.get(refs)


def pmap[T](
    fn: Callable[[Any], T],
    items: Iterable[Any],
    backend: str | None = None,
    workers: int | None = None,
    cpus_per_job: float = 1.0,
) -> list[T]:
    """Order-preserving parallel map. backend: local | ray (default from CHIPRL_BACKEND)."""
    backend = backend or os.environ.get("CHIPRL_BACKEND", "local")
    items = list(items)
    if backend == "ray":
        return list(_ray_map(fn, items, cpus_per_job))
    with ThreadPoolExecutor(workers or os.cpu_count() or 4) as pool:
        return list(pool.map(fn, items))
