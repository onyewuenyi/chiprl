"""Fan out jobs. Scale-up path, switch with one flag:

  local  thread pool on this machine (each grading job is a set of EDA subprocesses, so threads suffice)
  ray    Ray tasks: local multi-core with `ray.init()`, or a cluster via RAY_ADDRESS. The same
         Ray runtime later hosts verl's actors, so there is one scheduler for rollouts and grading.

`imap` is the primitive. It streams results as jobs finish, keeps at most `max_in_flight` jobs
submitted (backpressure: memory and scheduler load stay flat however many items there are),
isolates failures (a job that raises becomes a JobResult with `error`, the rest keep running), and
retries only results the caller marks as infra, a bounded number of times. Retrying a wrong answer
would just resample the grader; retrying infra is how a transient failure stops costing a row.

Closing the iterator early (break, or an exception in the consumer) cancels every job not yet
started. That is how a spend cap stops a run.

LLM API calls never go through Ray (AGENTS rule 12): they are IO-bound and the client is not
picklable. Callers that make API calls pass backend="local".
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any


@dataclasses.dataclass
class JobResult[T]:
    index: int  # position in the input
    item: Any
    value: T | None = None
    error: BaseException | None = None  # set if fn raised on the last attempt
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.error is None


def _default_workers() -> int:
    return os.cpu_count() or 4


def imap[T](
    fn: Callable[[Any], T],
    items: Iterable[Any],
    *,
    backend: str | None = None,
    workers: int | None = None,
    max_in_flight: int | None = None,
    retries: int = 0,
    is_infra: Callable[[T], bool] | None = None,
    retry_errors: bool = False,
    cpus_per_job: float = 1.0,
) -> Generator[JobResult[T]]:
    """Yield a JobResult per item, in completion order.

    retries:      extra attempts for a result where is_infra(value) is True (or that raised, if
                  retry_errors). The last attempt's result is yielded either way.
    max_in_flight defaults to 2 x workers.
    """
    backend = backend or os.environ.get("CHIPRL_BACKEND", "local")
    workers = workers or _default_workers()
    window = max(1, max_in_flight or 2 * workers)
    if backend == "ray":
        yield from _ray_imap(fn, items, window, retries, is_infra, retry_errors, cpus_per_job)
        return
    if backend != "local":
        raise ValueError(f"unknown backend {backend!r}")

    def retryable(r: JobResult[T]) -> bool:
        if r.attempts > retries:
            return False
        if r.error is not None:
            return retry_errors
        return bool(is_infra and is_infra(r.value))  # ty: ignore[invalid-argument-type]

    it = enumerate(items)
    pool = ThreadPoolExecutor(workers)
    pending: dict[Future, JobResult[T]] = {}

    def submit(r: JobResult[T]) -> None:
        pending[pool.submit(fn, r.item)] = r

    def refill() -> None:
        while len(pending) < window:
            nxt = next(it, None)
            if nxt is None:
                return
            submit(JobResult(index=nxt[0], item=nxt[1]))

    try:
        refill()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                r = pending.pop(f)
                try:
                    r.value, r.error = f.result(), None
                except Exception as e:
                    r.value, r.error = None, e
                if retryable(r):
                    submit(dataclasses.replace(r, value=None, error=None, attempts=r.attempts + 1))
                else:
                    yield r
            refill()
    finally:
        for f in pending:
            f.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def _ray_imap[T](
    fn: Callable[[Any], T],
    items: Iterable[Any],
    window: int,
    retries: int,
    is_infra: Callable[[T], bool] | None,
    retry_errors: bool,
    cpus_per_job: float,
) -> Generator[JobResult[T]]:
    # Under `uv run`, Ray tries to re-create the uv env in each worker; workers should reuse ours.
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    import ray  # ty: ignore[import-not-found]  # optional `scale` extra

    if not ray.is_initialized():
        ray.init(address=os.environ.get("RAY_ADDRESS"), ignore_reinit_error=True, log_to_driver=False)
    remote = ray.remote(num_cpus=cpus_per_job)(fn)
    it = enumerate(items)
    pending: dict[Any, JobResult[T]] = {}

    def refill() -> None:
        while len(pending) < window:
            nxt = next(it, None)
            if nxt is None:
                return
            pending[remote.remote(nxt[1])] = JobResult(index=nxt[0], item=nxt[1])

    try:
        refill()
        while pending:
            [ref], _ = ray.wait(list(pending), num_returns=1)
            r = pending.pop(ref)
            try:
                r.value, r.error = ray.get(ref), None
            except Exception as e:
                r.value, r.error = None, e
            again = r.attempts <= retries and (
                retry_errors if r.error is not None else bool(is_infra and is_infra(r.value))  # ty: ignore[invalid-argument-type]
            )
            if again:
                pending[remote.remote(r.item)] = dataclasses.replace(
                    r, value=None, error=None, attempts=r.attempts + 1
                )
            else:
                yield r
            refill()
    finally:
        for ref in pending:
            ray.cancel(ref)


def pmap[T](
    fn: Callable[[Any], T],
    items: Iterable[Any],
    backend: str | None = None,
    workers: int | None = None,
    cpus_per_job: float = 1.0,
    **kw: Any,
) -> list[T]:
    """Order-preserving parallel map over imap. Raises the first job error after all jobs finish."""
    results = sorted(
        imap(fn, items, backend=backend, workers=workers, cpus_per_job=cpus_per_job, **kw),
        key=lambda r: r.index,
    )
    for r in results:
        if r.error is not None:
            raise r.error
    return [r.value for r in results]  # ty: ignore[invalid-return-type]
