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

Full sweep (`make bench-full`): 155 tasks, reference + up to 3 mutants each = 516 distinct
submissions, equivalence bonus on. Outcomes: 171 pass, 340 sim mismatch, 3 compile, 1 lint,
1 timeout, 0 infra. Other work (test runs) shared the machine during parts of the sweep.

| Workers | Cold wall | Cold rollouts/s | p50 | p95 | p99 | max | Warm rollouts/s |
|---|---|---|---|---|---|---|---|
| 1 | 255.4 s | 2.0 | 0.10 s | 0.51 s | 2.8 s | 120.2 s | 31,700 |
| 4 | 150.3 s | 3.4 | 0.11 s | 0.49 s | 3.1 s | 120.2 s | 24,300 |
| 8 | 138.0 s | 3.7 | 0.15 s | 0.66 s | 4.1 s | 120.3 s | 11,100 |
| 10 | 139.8 s | 3.7 | 0.22 s | 0.92 s | 5.6 s | 120.3 s | 16,000 |

Time by stage at 1 worker (sum over jobs that ran tools): lint 22 s, simulation 159 s, equivalence
58 s. Per-call p50 is about 0.05 s for each stage.

**Reading it.** Throughput stops scaling at 4 workers, and the cause is one job: a mutant that hangs
the simulator runs into the 120 s wall-clock timeout, then the reference control run. That single
job is about 120 s of the 138 to 150 s wall, and about 120 of the 159 s of simulation time. Every
other simulation takes 0.05 to 0.1 s, so the timeout is over 1000x the p95. Without it, 10 workers
would finish in roughly 15 s.

## What breaks first at 100x

1. **The fixed simulation timeout, already at 1x.** One hanging submission holds a worker for
   120 s while normal jobs take 0.1 s, so wall time is set by the slowest job, not by the work. At
   100x with a policy that learns to hang the simulator, this dominates everything. The fix is a
   per-task timeout derived from the reference's own run time (for example, max(5 s, 20x the
   reference's simulation time), measured once per task like the sample count), so a hang costs
   seconds. It changes how `timeout` is classified, so it needs its own golden test before it lands.
2. **Equivalence on wide designs.** The long tail of real work: a few large designs (conwaylife,
   gshare, popcount255) cost as much as dozens of small ones. Size-based depth already cut the
   full-set sweep from 193 s to 23 s. For RL, the bonus can move off the hot path entirely, since
   GRPO uses only the ranking within a group.
3. **One machine's cores, then the shared cache.** Grading is CPU-bound subprocess work; the warm
   rate (11,000 to 31,000 lookups/s) shows the Python layer is not the limit. Past one machine the
   same `imap` contract runs on Ray (`CHIPRL_BACKEND=ray`), and the cache moves from WAL SQLite,
   which serializes writers, to a shared key-value store with the same key.

## Not done

- Container sandbox for model-written RTL (`--network none`, read-only grader mounts, non-root).
  Today's isolation is a separate work directory plus rlimits, which is enough on a dev machine and
  not enough for untrusted code on shared hardware.
- Cross-process single-flight. Two processes grading the same new submission both compute it; the
  second write wins and both results are identical, so this costs time, not correctness.
- The LibreLane image is pinned by tag, not digest.
