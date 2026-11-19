# Grading service design

How chiprl grades thousands of model-written designs reliably, what it measured, and what breaks
first at 100x. Every claim here has a test or a measurement behind it; file references are to the
code that implements it.

## The job

A job is (task, submission, options) → a reward and a stage. A policy produces many rollouts per
task, most of them wrong, some identical. The grader runs untrusted RTL through external tools
(Verilator, Icarus, Yosys, optionally OpenSTA in a container), any of which can hang, crash or be
made to hang by the submission.

Three properties matter more than speed:

1. **The reward is right.** A wrong design never scores as correct (`tests/test_golden.py`,
   `tests/test_equiv.py`, `docs/hack_log.md`).
2. **Infra and behavior are never confused.** A broken machine is `reward=None` (excluded); a broken
   submission is `0`. Getting this wrong in either direction is a reward hack or a silent bias.
3. **The same input gives the same answer**, on another day and another machine, so cached and
   resumed results are valid.

## Flow

```
policy rollouts ──► imap (bounded window) ──► grade() ──► cache hit? ──yes──► result
                       │  per-job isolation        │            no
                       │  infra-only retry         ▼
                       │                       grade_rtl: lint → sim → samples → equiv
                       │                           │  each tool via run(): own process group,
                       │                           │  rlimits, TERM then KILL on timeout
                       ▼                           ▼
                 trajectories.jsonl ◄────── result (stored only if durable)
                 (append, job_id)
```

## Decisions

**Single entry point with a content-addressed cache** (`chiprl/graders/service.py`, `chiprl/cache.py`).
The key is a hash of task content, submission, options, every tool version, a fingerprint of the
grading-path source and vendored PDK, and the runtime knobs that change results. Editing grader code
invalidates the cache without anyone remembering to bump a tag. Infra results and anything
load-dependent (an equivalence check that hit a solver timeout) are never stored, so a transient
failure is retried instead of remembered. Within a process the cache is single-flight: identical
in-flight submissions are graded once. SQLite in WAL mode with one connection per thread; a cache
error falls back to grading, because the cache is an optimization and must never cost a result.

**Blame by control run** (`chiprl/graders/rtl_gen.py`). A timeout or a killed tool is ambiguous: a
hostile submission and an overloaded machine look the same. Scoring it `None` lets a policy raise
its pass rate by crashing on hard tasks (evals drop `None` rows); scoring it `0` turns real infra
failures into wrong answers. The grader re-runs the reference through the same tools under the same
limits: reference healthy → the submission scores 0 (`timeout` / `resource`); reference also failing
→ `None`. Cost: one extra reference run, only on the failure path.

**Streaming executor with backpressure** (`chiprl/runtime/executor.py`). `imap` keeps at most
`max_in_flight` jobs submitted and pulls input lazily, so memory and scheduler load stay flat at any
job count. One job raising does not lose the batch. Retries apply only to results the caller marks
as infra, a bounded number of times: retrying a wrong answer would resample the grader, which is a
way to launder noise into reward. Closing the stream cancels unstarted jobs; the spend cap uses this.

**Process-group kill** (`chiprl/tools/_run.py`). Each tool runs in its own session. On timeout the
whole group gets SIGTERM, then SIGKILL after 2 s. Before this, a timeout killed only the direct child
and left grandchildren running (`tests/test_runner.py` reproduced it). Rlimits are applied by a
`sh -c 'ulimit …; exec "$@"'` prefix instead of `preexec_fn`, which Python documents as unsafe in a
multithreaded process. The OpenSTA container runs with `--init`, so the TERM the docker client
proxies stops it (not yet verified on this machine: Docker was not running).

**Deterministic equivalence budget** (`chiprl/tools/equiv.py`). The bounded check's depth comes from
the reference's flop bits, not from a clock, so the verdict does not depend on machine load and can
be cached. A solver timeout is flagged and never cached.

**Durable runs** (`chiprl/runtime/manifest.py`, `chiprl/llm/evals.py`). `job_id` hashes task
content, policy config, sample index and seed. Rows are appended; the last row per `job_id` wins;
a torn final line is ignored. `--resume` skips jobs with a durable row and reruns infra rows.
`summary.json` is a pure function of the trajectory file, so a resumed run reports the same numbers
as an uninterrupted one (`tests/test_resume.py`). Resuming refuses if config, seed, tasks, toolchain
or grader code changed.

## Measured

Apple M5, 10 cores, 24 GB, OSS CAD Suite 2026-11-12, proxy power, 2026-11-18.

| What | Number |
|---|---|
| Grade all 155 VerilogEval references with the equivalence bonus, 10 threads, cache off | 23.4 s wall; per-job p50 0.39 s, p95 2.1 s, max 22.6 s |
| Same sweep before the equivalence changes (fixed depth 12 + eqy) | 192.6 s wall; p95 24 s, max 142 s; 5 solver timeouts |
| Fixtures, reference + 7 mutants x 4 reps, no equivalence, cache off (`make bench`, before the cache was wired in) | 39.9 graded rollouts/s, p50 0.19 s, p95 0.42 s, 0 infra |
| Fixtures, same jobs, warm cache | about 14,000 rollouts/s (a lookup) |

Equivalence method cost over the 155 references:

| Method | Proves | p50 | max | Total |
|---|---|---|---|---|
| Yosys `equiv_*` | 147 | 0.04 s | 9.8 s | 22 s |
| eqy (SBY strategy) | 139, all also proven by Yosys | 1.1 s | 88 s | 503 s |
| Bounded check, depth 4 | 154 clean | 0.06 s | 43 s | 131 s |

The full sweep (`make bench-full`: 155 tasks x (reference + 3 mutants), 1/4/8/10 workers, cold and
warm cache, per-stage time) is pending; its table goes here.

## What breaks first at 100x

From the numbers above, ahead of the full sweep:

1. **Equivalence on wide designs.** It is the long tail: a few designs account for most grading
   time (one conwaylife-sized design costs as much as dozens of small ones). At 100x the p95 is set
   by these. Levers: the size-based depth already applied, skipping the bonus check for rollouts
   that do not pass simulation (already the case), and moving the bonus off the hot path entirely
   for RL, where only the ranking within a group matters.
2. **One machine's cores.** Grading is CPU-bound subprocess work, so throughput scales with cores
   until then; the warm-cache rate shows the Python layer is not the limit. Past one machine, the
   same `imap` contract runs on Ray (`CHIPRL_BACKEND=ray`), and the cache needs to move from SQLite
   to a shared store.
3. **The shared cache file.** WAL SQLite serializes writers. At thousands of writes per second
   from many processes, writes queue behind the busy timeout; reads keep working. The fix is a
   network key-value store with the same key; the key design does not change.

## Not done

- Container sandbox for model-written RTL (`--network none`, read-only grader mounts, non-root).
  Today's isolation is a separate work directory plus rlimits, which is enough on a dev machine and
  not enough for untrusted code on shared hardware.
- Cross-process single-flight. Two processes grading the same new submission both compute it; the
  second write wins and both results are identical, so this costs time, not correctness.
- The LibreLane image is pinned by tag, not digest.
