"""Executor contract: ordering, bounded in-flight, error isolation, infra-only retry, early stop."""

from __future__ import annotations

import threading
import time

import pytest

from chiprl.runtime.executor import imap, pmap


def test_pmap_preserves_order():
    assert pmap(lambda x: x * x, range(50), backend="local", workers=8) == [x * x for x in range(50)]


def test_imap_streams_in_completion_order():
    out = [r.value for r in imap(lambda s: (time.sleep(s), s)[1], [0.3, 0.0], backend="local", workers=2)]
    assert out == [0.0, 0.3]


def test_in_flight_is_bounded_and_input_is_consumed_lazily():
    lock = threading.Lock()
    running = peak = 0
    pulled = 0

    def source():
        nonlocal pulled
        for i in range(200):
            pulled += 1
            yield i

    def job(x):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.002)
        with lock:
            running -= 1
        return x

    gen = imap(job, source(), backend="local", workers=8, max_in_flight=5)
    first = next(gen)
    assert pulled <= 6, f"pulled {pulled} items before the first result: no backpressure"
    rest = list(gen)
    assert len(rest) + 1 == 200 and first.ok
    assert peak <= 5


def test_one_failure_does_not_lose_the_batch():
    def job(x):
        if x == 3:
            raise RuntimeError("boom")
        return x

    rs = sorted(imap(job, range(10), backend="local", workers=4), key=lambda r: r.index)
    assert [r.ok for r in rs].count(False) == 1
    assert isinstance(rs[3].error, RuntimeError) and rs[3].value is None
    assert [r.value for r in rs if r.ok] == [0, 1, 2, 4, 5, 6, 7, 8, 9]
    with pytest.raises(RuntimeError, match="boom"):
        pmap(job, range(10), backend="local")


def test_retries_infra_only_and_is_bounded():
    calls: dict[str, int] = {}
    lock = threading.Lock()

    def job(name):
        with lock:
            calls[name] = calls.get(name, 0) + 1
            n = calls[name]
        if name == "flaky":
            return None if n == 1 else 1.0  # infra once, then fine
        if name == "dead":
            return None  # infra every time
        return 0.0  # a wrong answer: never retried

    rs = {
        r.item: r
        for r in imap(
            job, ["flaky", "dead", "wrong"], backend="local", retries=2, is_infra=lambda v: v is None
        )
    }
    assert rs["flaky"].value == 1.0 and rs["flaky"].attempts == 2
    assert rs["dead"].value is None and rs["dead"].attempts == 3 and calls["dead"] == 3
    assert rs["wrong"].value == 0.0 and calls["wrong"] == 1


def test_errors_retry_only_when_asked():
    n = {"c": 0}

    def job(_):
        n["c"] += 1
        raise OSError("transient")

    [r] = list(imap(job, [0], backend="local", retries=2))
    assert n["c"] == 1 and not r.ok
    [r] = list(imap(job, [0], backend="local", retries=2, retry_errors=True))
    assert n["c"] == 4 and r.attempts == 3


def test_closing_early_cancels_unstarted_jobs():
    started = []
    lock = threading.Lock()

    def job(x):
        with lock:
            started.append(x)
        time.sleep(0.02)
        return x

    gen = imap(job, range(1000), backend="local", workers=2, max_in_flight=4)
    next(gen)
    gen.close()
    time.sleep(0.1)
    assert len(started) <= 8, f"{len(started)} jobs started after the consumer stopped"
