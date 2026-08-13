# fix: GPU use-after-free — device memory released while kernels still read it (scan read-lock lifetime + error-path teardown)

## Problem

Two independent instances of one defect class — **device memory released for
reuse while kernels still read it** — corrupted or wedged roughly **50% of
concurrent multi-stream workloads** (staged-refresh TPC-H throughput runs at
SF1000). Because the corruption lands in whatever pool memory gets rebound
next, one defect showed four faces:

1. `cudaErrorIllegalAddress` surfacing at the victim query's next stream sync
   (observed at `gpu_decode_strings.cu:240` — an innocent bystander).
2. Negative-size (2^64−N) allocations in HASH_JOIN/CONCAT — the signature of
   scribbled, non-monotonic string offsets reaching size subtraction.
3. Silent never-terminating kernels at full power — a cuco open-addressing
   insert livelocks when its row comparator reads memory mutating under it.
4. Garbage VARCHAR bytes with valid geometry — client-side
   `Invalid unicode (byte sequence mismatch)` errors.

Sequential runs were almost always clean: the freed region must be rebound
quickly by concurrent work for the stale read to become garbage, which is why
this presented as a flaky, concurrency-only corruption.

## Root cause

### Instance 1 (primary): the scan drops the cached-batch read-lock while its kernels still read the batch

The resident-scan path wraps the cached batch's read-only accessor (the batch
READ LOCK) in an `owning_table_view`, then enqueues the kernels that read the
batch — the filter gather and the view→table `release()` copy in
`post_filter_and_project`, plus normalization work in the scan's `execute()` —
and destroyed the owner (dropping the lock) before any of that work completed:
inside `post_filter_and_project` on some paths (the duckdb-native no-filter
path moves the owner into a local; parquet re-assigns `input` right after
enqueueing the gather), or at `execute()` exit — both ahead of
`run_one_operator`'s post-execute sync.

The moment the lock drops, the downgrade executor's `try_to_mutable` succeeds
and, under memory pressure, evicts the batch GPU→HOST and **frees its device
buffers while the scan's kernels still read them**; the async pool rebinds the
region to concurrent work. The keep-mask branches of
`gpu_ingestible::materialize_table` already synchronized for exactly this
hazard — the non-mask paths lacked the same discipline.

**Fix:** synchronize the stream before every owner-death point — the end of
both `post_filter_and_project` implementations and the end of the scan
`execute()` (the latter is nearly free: `run_one_operator` syncs the same
stream immediately after).

### Instance 2: query-abort teardown races the aborted query's still-running kernels

A pipeline task that throws out of `gpu_pipeline_task::execute()` (e.g.
`rmm::out_of_memory` inside a cudf call) unwinds with its kernels still
enqueued on `exc_stream`. In `gpu_pipeline_executor`'s worker lambda:

- the `task_reschedule_exception` handler synced `exc_stream` — but only
  *after* an early `return` taken when the completion handler was already in
  an error state (the exact path taken when a sibling task's fail-fast had
  already errored the query), and
- the generic `catch (std::exception&)` / `catch (...)` paths never synced.

On those unsynced exits the task (and the reschedule exception, which owns the
task's input batches) is destroyed. Its buffers are freed *stream-ordered on
the streams that allocated them* — long-idle upstream streams — so the async
pool considers the memory immediately reusable while `exc_stream`'s orphaned
kernels still read/write it. `drain_after_error()` only joins host lambdas;
`QueryEnd` then frees the whole query's device memory the same way. One abort
poisons many concurrent victims.

**Fix:** quiesce `exc_stream` on every abnormal exit of the worker lambda
before any owner of device memory is destroyed — the sync is hoisted to the
top of the reschedule handler (now covering the has-error early return and the
failed-cast return) and added to both generic catches (sync failures from
sticky CUDA errors are logged, never thrown). Belt-and-suspenders: a
device-scoped `cudaDeviceSynchronize` per GPU executor in
`task_scheduler::drain_after_error()` (error path only) so QueryEnd can never
free device memory with pending kernels even through a throw path the handlers
miss.

## Verification

Forensics and verification were run on the SF1000 staged-refresh concurrent
throughput stack where the corruption reproduced (~50% incidence):

- **Deterministic repro via poison-on-free.** An env-gated diagnostic
  (`SIRIUS_POISON_FREES=1`) enqueues a stream-ordered 0xEE fill over every
  freed device region before the free, converting use-after-free reads from
  timing-dependent stale-but-intact data into deterministic garbage.
  Pre-fix: **3/3 runs wedged**, always within the *sequential* phases —
  proving a freed-while-still-read buffer in the normal warm serve path, no
  query abort required. Post-fix: **3/3 poison-armed sequential trials
  clean** (full power coverage, zero corruption events) plus **1/1
  poison-armed full concurrent run clean**, including seven genuine OOM→CPU
  fallbacks — i.e. the error-path teardown executed seven times at maximum
  use-after-free sensitivity and corrupted nothing.
- **GPU core dump naming the livelocked kernel.** A user-triggered core dump
  (`CUDA_ENABLE_USER_TRIGGERED_COREDUMP`) at a pre-fix wedge showed exactly
  one active kernel: `cuco::detail::open_addressing_ns::insert_if_n<...,
  cudf::detail::primitive_keys_fn<rhs_index_type>...>` — the cudf
  `distinct_hash_join` build insert over 200 M rows, livelocked because its
  row comparator read memory mutating under it (dangling build-input views
  into an evicted batch). This ties face 3 to the same defect.
- **Regression test** (`test/cpp/pipeline/test_gpu_pipeline_executor.cpp`,
  tag `[gpu_pipeline_executor]`): a task enqueues slow stream-ordered work
  and throws (a) `oom_reschedule_exception` with the completion handler
  pre-errored — the exact regressed path — and (b) a generic exception; the
  task destructor records whether it ran while stream work was still pending.
  Fails on the pre-fix code, passes with the fix.
- **Clean scored production runs:** with the fixes (poison off, decode/ASCII
  guards armed as canaries), 3/3 full SF1000 power+throughput runs completed
  with zero corruption events (QphH 7,211,669 / 7,365,117 / 7,044,510).

**Known residual (honest status):** one post-fix production run wedged
silently with a different signature (low-occupancy ~254 W spin vs the 480 W
distinct-build wedges), zero errors. The poison-deterministic instance and the
teardown instance are gone, but the bug *class* is reduced, not proven extinct
at production timing — poison slows the engine ~3.5×, which can shift a narrow
race out of its window. A follow-up hunt with the coredump pipe armed is the
next step; it does not block this fix.

## Note for reviewers

The poison-on-free diagnostic used for the deterministic repro is intentionally
kept out of this PR (it taxes every free). It lives on the cucascade branch
`diag/poison-on-free` if you want to reproduce the verification.

## Performance

The added syncs are on paths that were already followed by a sync
(`run_one_operator` syncs the operator stream right after `execute()`), on
error-only paths, or replace an unsynchronized owner-death that was simply
wrong. No measurable cost in the scored SF1000 runs above.
