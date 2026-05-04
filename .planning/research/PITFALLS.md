# Pitfalls Research — v1.4 Rebase After DataBatch Changes

**Domain:** RAII data-batch migration + multi-GPU IO framework adoption in an existing GPU SQL engine
**Researched:** 2026-05-04
**Confidence:** HIGH — derived from v1.1–v1.3 post-mortems, live git inspection of PRs #117/#675/#731/#739, and cucascade source

---

## Critical Pitfalls

### Pitfall 1: Holding `read_only_data_batch` / `mutable_data_batch` Across a Blocking Tier Conversion

**What goes wrong:**
A site acquires `auto ro = batch->to_read_only()` and then calls `mutable_data_batch::convert_to()` or any function that internally tries `to_mutable()` on the same batch. The shared lock prevents the exclusive acquisition — the thread blocks forever waiting for its own shared lock to be released. This is a single-thread self-deadlock: the caller is both the lock holder and the waiter.

The same shape appears with two threads if thread A holds `read_only` while thread B is in `convert_to` (waiting for exclusive): both threads can be legitimately required to complete for the pipeline to make progress, but `convert_to` needs exclusive and `to_read_only` callers are stacked up — `shared_mutex` writer starvation under heavy reader load can make this look like a live-lock rather than a hard deadlock.

**Why it happens:**
Pre-#117, `data_batch` had no locking — callers accessed `get_data()` directly with no compiler enforcement. When migrating ~12 operators mechanically, developers wrap every access in the appropriate accessor but forget to scope the lifetime tightly. A common pattern: store `auto ro = batch->to_read_only()` in a local, then pass the batch (not the accessor) to a helper that also calls `to_read_only()` — two accessors on the same thread, fine. But passing the batch to a helper that calls `convert_to()` or `to_mutable()` will deadlock.

**How to avoid:**
- Keep every `read_only_data_batch` and `mutable_data_batch` scoped to the narrowest possible block — prefer `{ auto ro = batch->to_read_only(); use(ro); }` not `auto ro = batch->to_read_only(); ... 20 lines ... use(ro)`.
- Never call `batch->to_mutable()` or `data_batch::readonly_to_mutable()` while a `read_only_data_batch` from the same `batch` is live in the same call stack.
- For upgrade paths: use `data_batch::readonly_to_mutable(std::move(ro))` which consumes the reader and atomically upgrades — do not call `to_mutable()` while still holding `ro`.
- Add a code-review checklist: every `to_read_only()` / `to_mutable()` call site must have its destructor visibly before any call that acquires the same batch under a different lock type.

**Warning signs:**
- Hang during multi-GPU pipeline tests with GPU activity dropping to near-zero while CPU threads spin (futex wait visible in `perf top`).
- `compute-sanitizer racecheck` does not catch mutex deadlocks — use `helgrind` (`valgrind --tool=helgrind`) or compile with `-fsanitize=thread` (TSan). TSan reports lock-order violations during pre-deadlock scenarios.
- `[mgpu_stress]` 500-iter test will hang rather than fail — timeout after 3× expected runtime is the signal.

**Phase to address:** Cucascade rebase phase (Phase 16) — every RAII migration site must be reviewed for scope tightness before integration tests.

---

### Pitfall 2: `writer_stream` / `writer_event` Contract Broken by the RAII Layer

**What goes wrong:**
Phase 13 fixed Q11's illegal-address by requiring a `writer_stream` argument in the `gpu_table_representation` constructor (`62e0517`). Under PR #117's RAII model, `convert_to()` is now a method of `mutable_data_batch` rather than `data_batch`. If the post-#117 `representation_converter.cpp` internal convert functions are rebased naively, the `writer_stream` argument may be passed at the wrong moment — after the lock is released — or may be wired to a default-constructed `cuda_stream_view` because the migration author does not realize that the `target_stream` must come from the *mutable accessor's* call site, not from any stream available in scope.

More concretely: our `7ed84f2` fix (`representation_converter`: use target-bound stream) already patches `convert_host_to_gpu` / `convert_gpu_to_gpu`. But `73d00c4` (PR #117) also modifies `representation_converter.cpp`. The three-way merge of our `7ed84f2` + `62e0517` against `73d00c4` on the `representation_converter.cpp` conflict file is the highest-risk single file in the entire rebase. If the merge result passes `cuda_stream_view{}` (default) to the new `gpu_table_representation` constructor, the `cudaStreamWaitEvent` path in `convert_gpu_to_gpu` falls back to `cudaDeviceSynchronize` on the source device — hiding the race under a heavy sync instead of fixing it. Tests will pass but SF100 multi-GPU will exhibit the same illegal-address that Q11 had.

**Why it happens:**
`representation_converter.cpp` is flagged as a conflict file in the cucascade rebase. The developer resolving the conflict will be tempted to take "ours" for the stream-lineage lines and "theirs" for the RAII accessor restructuring. The two changes operate on adjacent functions — the manual merge result can silently drop the `writer_stream` argument.

**How to avoid:**
- Treat `representation_converter.cpp` as a **single atomic re-implementation** rather than a merge: start from `73d00c4`'s version, then re-apply the stream-lineage changes from `7ed84f2` + `62e0517` by hand.
- After the rebase, grep-verify: `grep -n "gpu_table_representation" cucascade/src/data/representation_converter.cpp` must show the `writer_stream` / `target_stream` argument passed at every construction site.
- Add a post-merge static assertion: any `gpu_table_representation` constructed with a default `cuda_stream_view` should emit a compile warning (`[[deprecated]]` tag on that constructor overload) — not blocking in v1.4, but a strong signal during dev.
- Confirm with SF100 Q11 num_gpus=2 specifically (not just `[mgpu]` suite) since Q11 is the only query with the cross-GPU converter path exercised at that data volume.

**Warning signs:**
- `compute-sanitizer racecheck` on `[mgpu]` suite reports a race in `cudaMemcpyPeerAsync`.
- Q11 2-GPU produces wrong results or CUDA_ERROR_ILLEGAL_ADDRESS — but only at SF100, not SF1 (the race is data-size-dependent).
- `grep -n "cuda_stream_view{}" cucascade/src/data/representation_converter.cpp` returns any line inside the `convert_*` functions (any default-constructed stream view at a construction site is wrong).

**Phase to address:** Cucascade rebase phase (Phase 16) — verify immediately after `representation_converter.cpp` conflict resolution, before any other integration work.

---

### Pitfall 3: `pop_next_data_batch()` Non-Blocking Semantics Break Callers That Assumed Blocking

**What goes wrong:**
Pre-#117, `pop_data_batch(target_state)` blocked on a condition variable until a batch reached the requested state. PR #117 replaces this with `pop_next_data_batch()` which is an immediate non-blocking FIFO pop — returns `nullptr` if the partition is empty. Any Sirius caller that spins on `pop_next_data_batch()` in a tight loop (replacing the old blocking call) will burn CPU. Any caller that treats a `nullptr` return as "no more batches" (rather than "not yet") will silently drop data, producing wrong query results.

**Why it happens:**
The old API name hinted at blocking (`wait_to_create_task`, `try_to_lock_for_processing`). The new names do not. When migrating the ~12 operator sites, developers may copy-paste the old blocking-loop pattern from before the API existed.

**How to avoid:**
- Audit every call site of the old `pop_data_batch()` in Sirius source after the rebase. Each site needs one of: (a) a proper condition-variable driven wait loop, (b) a `try_to_read_only()` retry loop with backoff, or (c) restructuring to use the new subscriber / connector model.
- The Scan Manager's `split_connector` is the right model for producer/consumer coordination post-#117: use it as a template for any new batch-pop sites.
- Run `[TPC-H][parquet] 22/22` against a small SF1 dataset with SIRIUS_LOG_LEVEL=debug and verify row counts match DuckDB baseline — silent data loss from premature `nullptr` is invisible to performance tests but visible to correctness tests.

**Warning signs:**
- `[TPC-H][parquet]` correctness failures (wrong row counts or values) without any CUDA errors.
- High CPU usage (`top` shows sirius_unittest at 100% CPU per thread) during a query that should be I/O-bound.
- `grep "pop_data_batch\|pop_next_data_batch" src/` showing any site that unconditionally discards `nullptr` without a retry path.

**Phase to address:** DataBatch API migration phase — after #117 rebase is applied to Sirius operators.

---

### Pitfall 4: `uring_reactor` Thread Inherits CUDA Context From Construction Site, Not Per-Request Device

**What goes wrong:**
In `uring_reactor.cpp`, the reactor thread calls `cudaSetDevice(req.device_id)` per-request inside the `O_DIRECT` I/O loop before issuing `cudaMemcpyAsync`. This is correct for simple cases, but `cudaSetDevice` is a thread-local call — if the reactor thread was initialized (first CUDA API call) under a different device's context, the CUDA runtime may not establish the correct device context for the `cudaMemcpyAsync` target. This is exactly the pattern that v1.1 fixed for kvikio: kvikio bound to a single CUDA context during library init, making it unsafe for multi-GPU dispatch.

Under the single-GPU (PR #675) design, one shared `sirius_ioctx` is constructed and the reactor threads are spawned from whatever thread creates it. When Sirius v1.4 adapts this to multi-GPU (per-GPU reactor pools), if the reactor factory is called from a thread that already has GPU 0's context active (the SiriusContext init thread), all reactor threads inherit GPU 0's context — `cudaSetDevice(1)` before `cudaMemcpyAsync` switches the active device but does not create a fresh primary context on GPU 1 if the thread has never been associated with GPU 1 before.

**Why it happens:**
The PR #675 design doesn't expose a per-GPU reactor factory — `sirius_datasource` takes a shared `sirius_ioctx`. Multi-GPU adaptation requires creating *N* ioctx instances, one per GPU, each with their reactor threads spawned under `rmm::cuda_set_device_raii` for that GPU. If instead a single shared ioctx is used for all GPUs (the single-GPU shape applied naively to multi-GPU), the bounce slots are allocated under one device's context (`cudaHostAllocPortable` helps for the memory visibility, but the CUDA device context for the `cudaMemcpyAsync` call is still per-reactor-thread).

The v1.1 decision log explicitly calls this out: "kvikio/cuFile bind to a single CUDA context → unsafe for multi-GPU task dispatch." `sirius_datasource` risks reintroducing this exact anti-pattern if the multi-GPU adaptation doesn't create per-GPU ioctx instances.

**How to avoid:**
- Create one `uring_ioctx` per GPU. Each ioctx is constructed under `rmm::cuda_set_device_raii` for its target GPU so reactor threads inherit the correct primary context.
- Store per-GPU ioctx on `SiriusContext` using the same `std::unordered_map<int, ...>` pattern as `gpu_io_backends` established in v1.1.
- The `device_read_req_type::device_id` field in `uring_reactor` already propagates device_id per-chunk — use this as the discriminator at the ioctx selection site (pick the ioctx by device_id), NOT as a `cudaSetDevice` override in the reactor thread loop.
- After v1.4: add a HYG-class grep gate: `grep -rn "cudaSetDevice" src/io/` must return zero results (all device selection must happen through `rmm::cuda_set_device_raii` or the ioctx dispatch path, never raw `cudaSetDevice` inside reactor threads).

**Warning signs:**
- `compute-sanitizer memcheck` reports `cudaErrorInvalidDevice` on device_read calls on GPU 1 while GPU 0 calls succeed.
- `[TPC-H][parquet]` passes on num_gpus=1 but fails or produces wrong results on num_gpus=2.
- `nvidia-smi` shows all I/O activity on GPU 0 even when tasks are dispatched to GPU 1 (zero PCIe bandwidth on GPU 1 side).

**Phase to address:** IO Framework multi-GPU adaptation phase — create per-GPU ioctx instances at SiriusContext::initialize() time, mirroring the v1.1 `gpu_io_backends` pattern.

---

### Pitfall 5: Admission Control Budget Is Global, Not Per-GPU

**What goes wrong:**
`admission_control` takes a single `size_t budget` parameter — the 2 GiB default applies to the entire system, not per-GPU. When N=2 GPUs are dispatching I/O concurrently, each GPU's I/O path competes for slots from the same global budget. At SF100 with two GPUs both scanning, the budget is consumed twice as fast. The admission control then serializes GPU 0 and GPU 1 reads against each other, halving effective I/O parallelism vs. what the hardware supports.

The correct model for multi-GPU: either (a) per-GPU admission_control instance (each GPU's ioctx gets its own budget), or (b) a global budget that is set to N × per-GPU budget. Using the default single-budget with a per-GPU ioctx pool means each ioctx has its own admission_control — this is actually the correct design if ioctx instances are per-GPU. But if a single shared ioctx is used (the wrong multi-GPU adaptation), a single admission_control serializes all GPUs.

**Why it happens:**
The PR #675 admission_control is designed for single-GPU. There is no `num_gpus` multiplier in the constructor. A developer adapting it to multi-GPU might create per-GPU ioctx correctly but set each admission_control budget to 2 GiB — fine. Or might create a single ioctx with a single budget and not realize the contention. The failure mode is perf-only (not correctness), so it passes all functional tests.

**How to avoid:**
- When constructing per-GPU ioctx instances, each gets its own `admission_control` with the configured per-GPU budget.
- The SF100 Q1 num_gpus=2 perf gate (≤ 5.7s) is the direct test — if the per-GPU budget is misconfigured, I/O throughput drops and the gate fails.
- Log admission_control saturation events at DEBUG level (count of slots blocked > 0 at any point) so SF100 runs produce a detectable signal.

**Warning signs:**
- SF100 Q1 2-GPU wall-clock ≥ 1× the 1-GPU baseline (should be ≤ 1.05× given I/O-bound workload).
- `nvidia-smi dmon` shows GPU 1 PCIe activity is temporally offset from GPU 0 (reads serialized, not concurrent).
- `perf stat` on the process shows high `futex` syscall count on the `admission_control::acquire()` path.

**Phase to address:** IO Framework multi-GPU adaptation phase — budget configuration check in acceptance criteria.

---

### Pitfall 6: SCHED-RR Counter and `_batch_gpu_affinity` Become Stale Under Split-Provider

**What goes wrong:**
v1.3's SCHED-RR distribution ran in `task_scheduler::management_eventloop` and stamped `_batch_gpu_affinity` in `duckdb_scan_executor` as batches were dispatched. Under PR #731's Scan Manager, split allocation happens in `parquet_split_provider` at `prepare_for_query` time, driven by a scan manager driver thread — not in `management_eventloop`. If the SCHED-RR affinity logic is left in `management_eventloop` without being ported to the split_provider, the round-robin counter is never incremented and all splits get dispatched to GPU 0 (the default when no affinity is set).

Separately, the `_batch_gpu_affinity` map was written by `duckdb_scan_executor` as batches completed. The scan manager / split_connector architecture separates split production from batch consumption — if `_batch_gpu_affinity` recording is wired at the wrong layer (the split_connector rather than the batch processing site), affinity records appear before the batch is actually consumed, breaking the disjointedness REQUIRE assertion.

**Why it happens:**
The old scan path (`sirius_parquet_metadata_scan_operator.hpp` — deleted by PR #731) held both metadata scan logic and the SCHED-RR affinity assignment in one file. PR #731 deletes this file entirely. Phase 13's stream-lineage fix in the same operator is also in deleted-line territory. The developer porting v1.3 work into the new Scan Manager must identify the correct attachment points in `parquet_split_provider` and `split_connector` for two distinct concerns: (1) GPU affinity assignment for splits, (2) stream-lineage tracking for cross-GPU reads.

**How to avoid:**
- Map Phase 14's `_no_pref_rr_counter` increment logic to `parquet_split_provider::start()` or its split-emission loop. The split_provider is the production site — this is where GPU affinity should be assigned (not post-hoc in management_eventloop).
- Map Phase 13's `writer_stream` plumbing to wherever `gpu_table_representation` is constructed in the new scan path — likely in `sirius_gpu_parquet_scan_operator.cpp` or the scan task, not in `split_provider`.
- Preserve the Phase 9 disjointedness REQUIRE assertion at the AUDIT TEST_CASE level — it is the regression gate for SCHED-RR correctness. If the assertion doesn't fire green post-porting, the affinity wiring is wrong.
- The `[mgpu_stress]` 500-iter test specifically exercises the SCHED-RR counter across varied offset seeds — it will fail if the counter is not incremented per-query.

**Warning signs:**
- `[mgpu_stress]` 500-iter test fails with assertion that batches are NOT disjoint across GPUs (intersection > 0).
- `[mgpu]` 16/16 passes but AUDIT log shows scan_batch_ids distributed only to GPU 0 (grep `GPU_DEVICE=1` in sirius.log returns 0 matches).
- `task_scheduler::_no_pref_rr_counter` remains 0 throughout a multi-GPU query run (add a TRACE-level log at increment site before porting to verify).

**Phase to address:** Scan Manager integration phase — v1.3 SCHED-RR + affinity re-attachment to split_provider.

---

### Pitfall 7: PR #739 Pins Cucascade to `0cd4a6a` (Between #112 and #117); Cherry-Picking Then Rebasing on #117 Creates Combinatorial Mismatch

**What goes wrong:**
PR #739 (`468f6e1`) bumps cucascade to `0cd4a6a` — which includes PR #112 (memory-space bandwidth profiler) and PR #116 (gpu_data_representation from cudf::table_view) but NOT PR #117 (RAII data_batch). The API changes in #117 are breaking: `batch_state` enum is renamed, `pop_data_batch(state)` is removed, `data_batch_processing_handle` is removed. If #739 is cherry-picked onto the working branch first and then cucascade is rebased to include #117, the Sirius source files from #739 are now compiled against the pre-#117 API shape. The build fails with 50+ errors about removed symbols.

Conversely, if #739 is applied after the cucascade rebase to #117, the files it modifies (operator sources like `sirius_physical_hash_join.cpp`, `sirius_physical_table_scan.cpp`) need to be re-adapted to the #117 API on top of #739's changes — creating a three-way adaptation rather than a simpler linear application.

**Why it happens:**
The Sirius dev merge brings in 11 conflict files + 33 auto-merges, including #739 which touches the cucascade submodule and 14 operator files. The natural instinct is to apply all dev PRs first (including #739), then deal with cucascade. But #739 was authored against `0cd4a6a`, not `73d00c4` (#117).

**How to avoid:**
- Apply PRs in the correct dependency order: cucascade rebase to `73d00c4` + our 11 fixes first, then port #739's Sirius-side operator changes (not the submodule bump — submodule is already handled by the cucascade rebase).
- During the Sirius dev merge conflict resolution for #739's files, resolve each conflict against the post-#117 API shape, not the `0cd4a6a` shape.
- Verify after merge: `grep -rn "task_created\|in_transit\|batch_state::idle" src/` must return zero results (old `batch_state` enum values that no longer exist in #117).
- Keep a short checklist of #117 API removals: `pop_data_batch(state)`, `data_batch_processing_handle`, `idata_batch_probe`, `lock_for_processing_result/status`, static `to_read_only(PtrType&&)` / `to_mutable(PtrType&&)` — any occurrence of these in post-merge source is a merge error.

**Warning signs:**
- Build fails with "use of undeclared identifier 'task_created'" or similar `batch_state` enum value.
- Build fails with "no member named 'pop_data_batch'" on `data_repository` type.
- Any file touching both cucascade API and the `sirius_physical_*` operator pattern fails to compile.

**Phase to address:** Cucascade rebase phase (Phase 16) — resolve ordering before applying any Sirius dev PRs that touch cucascade.

---

### Pitfall 8: `io_worker` Member-Order Fix (`eda349a`) Lost in `pipeline_io_backend.cpp` Conflict Resolution

**What goes wrong:**
`pipeline_io_backend.cpp` is one of the 6 conflict files in the cucascade rebase. Our fix `eda349a` ("reorder io_worker members so _thread is last") is a destruction-order fix: `_thread` (a `std::thread`) must be declared after `_queue` and other members so that when `io_worker` is destroyed, the thread joins only after the queue is fully drained. If the conflict resolution takes "theirs" for the io_worker class layout and drops our member reordering, the destructor will join a thread that is still writing to the queue — UB, often manifesting as a crash or hang on query teardown.

This is the same class of bug as Phase 10-03's `translated_expression::owned_stream` fix (RETROSPECTIVE: "C++ destruction-order discipline for CUDA stream-allocated objects"). It is easy to miss in a conflict resolution because the compiler does not warn about destruction-order issues.

**Why it happens:**
Conflict resolution tools (`git mergetool`, manual edit) default to showing structural diffs — they don't highlight that member declaration order controls destruction sequence. The fix in `eda349a` changes only the order of member declarations, not any logic. During conflict resolution, this looks like a cosmetic change and may be dropped when accepting PR #117's structural refactor of `pipeline_io_backend.cpp`.

**How to avoid:**
- After resolving `pipeline_io_backend.cpp`, explicitly verify that in `io_worker` (or its equivalent class in the post-#117 shape), the thread member (`_thread` or `std::jthread`) is declared last among the members it depends on.
- Run `[mgpu_stress]` 500-iter test immediately after the cucascade rebase — destructor-order bugs manifest as test-ordering-dependent crashes (Phase 10 pattern: bisect returns NONE, SIGSEGV is ordering-dependent).
- Add a comment at the thread member declaration: `// MUST be last: thread joins on destruction, must outlive _queue`.

**Warning signs:**
- SIGSEGV on test teardown that bisect returns NONE for (test-ordering dependent — see Phase 10 retrospective).
- Crash only under `[mgpu_stress]` or `[integration][TPC-H]` (which run many queries), not under single-query tests.
- Stack trace shows `io_worker` destructor calling `~std::queue` after `~std::thread` join has started.

**Phase to address:** Cucascade rebase phase (Phase 16) — explicitly check member ordering in any conflict-resolved class with a `std::thread` or `std::jthread` member.

---

### Pitfall 9: Pinned Host Memory Portable/Mapped Flag Dropped in Memory Conflict Files

**What goes wrong:**
Fixes `3743621` (Portable flag for `cudaMallocHost` sites) and `2dcab24` (Mapped flag for all pinned allocation sites) are in `cucascade/src/memory/common.cpp` and `cucascade/src/memory/memory_space.cpp` — both are conflict files in the cucascade rebase. If these flags are dropped during conflict resolution, pinned host memory is no longer accessible from both GPU 0 and GPU 1's CUDA contexts. The `cudaMemcpyPeerAsync` P2P copy path will silently fall back to a slower path or fail with `cudaErrorInvalidValue` on the GPU 1 side.

The failure mode is insidious: single-GPU tests pass (GPU 0 can always access its own pinned memory). Multi-GPU tests fail intermittently because whether the fallback path is taken depends on the driver's peer access state at the time of the copy.

**Why it happens:**
The Portable/Mapped flags are one-line additions inside `cudaMallocHost` / `cudaHostAlloc` call sites. When PR #117 restructures the memory layer (its `memory/common.cpp` and `memory/memory_space.cpp` conflict files), conflict resolution may drop these flag additions if the resolver doesn't know why they exist.

**How to avoid:**
- After resolving the memory conflict files, verify with: `grep -n "cudaHostAllocPortable\|cudaHostAllocMapped\|cudaMallocHost" cucascade/src/memory/common.cpp cucascade/src/memory/memory_space.cpp` — every `cudaMallocHost` / `cudaHostAlloc` call should have `Portable` (and where relevant `Mapped`) flags.
- Run the P2P transfer tests (`[multi_gpu_transfer]`, `[mem_04_p2p_transfer]`) immediately after the cucascade rebase to verify GPU↔GPU converter still works.
- The `[mgpu]` 16/16 test suite at SF1 is insufficient to catch this — P2P only exercises the fast path under load. Run SF100 Q1 num_gpus=2 to exercise the DMA path at scale.

**Warning signs:**
- `compute-sanitizer memcheck` on `[mgpu]` reports `cudaErrorInvalidValue` in `cudaMemcpyPeerAsync`.
- Peer DMA probe at `SiriusContext::initialize()` logs "disabling GPU↔GPU peer DMA" on the server hardware (where it was previously enabled).
- GPU 1 scan throughput drops to 0 in SF100 Q1 (data is copied via host staging instead of P2P).

**Phase to address:** Cucascade rebase phase (Phase 16) — verify P2P probe passes after conflict resolution.

---

### Pitfall 10: `sirius_parquet_metadata_scan_operator.hpp` Delete/Modify Conflict Drops Phase 13 Stream-Lineage Work

**What goes wrong:**
This file is explicitly called out in the milestone context as a "modify/delete special case": PR #731 deletes `sirius_parquet_metadata_scan_operator.hpp` (replaced by Scan Manager), but Phase 13 applied stream-lineage changes to this file. Git will present this as a conflict and force the developer to choose between "deleted by them" (keeping our modifications to the file) and "modified by us" (accepting the deletion). The correct answer is "accept the deletion" — but not before extracting the Phase 13 stream-lineage logic and re-attaching it to the correct location in the new Scan Manager architecture.

If the developer accepts the deletion without re-attaching the stream-lineage logic, the Q11 2-GPU fix regresses silently (no compile error, but the `writer_stream` is no longer recorded at parquet scan task construction time).

**Why it happens:**
The modify/delete conflict is handled at the git level by choosing one side — there is no three-way merge to guide "take parts of the deleted content and move them elsewhere." Developers under time pressure accept the deletion and move on, assuming the Scan Manager already handles the concern. It does not — the Scan Manager was authored against the pre-Phase-13 codebase.

**How to avoid:**
- Before accepting the deletion of `sirius_parquet_metadata_scan_operator.hpp`, extract the Phase 13 stream-lineage changes (the `writer_stream` plumbing from `record_writer_event` → constructor argument) and identify the corresponding attachment point in `sirius_gpu_parquet_scan_operator.cpp` or `parquet_scan_task.cpp`.
- Gate the phase with an explicit acceptance criterion: after the Scan Manager integration, `grep -rn "writer_stream\|record_writer_event" src/op/scan/` must return a non-zero count at the construction site of `gpu_table_representation`.
- The Q11 SF100 2-GPU run is the regression check — the stream-lineage fix only manifests at SF100 (SF1 runs fast enough that the race is rare).

**Warning signs:**
- `grep -rn "writer_stream\|record_writer_event" src/op/scan/` returns zero after the Scan Manager integration.
- SF100 Q11 num_gpus=2 produces `CUDA_ERROR_ILLEGAL_ADDRESS` or wrong results.
- `[mgpu]` 16/16 passes (Q11 SF1 is in the suite) but SF100 Q11 fails — the scale gap is the tell.

**Phase to address:** Scan Manager integration phase — explicit re-attachment of Phase 13 stream-lineage to the new scan path.

---

### Pitfall 11: HYG-02 Regression From `uring_reactor.cpp` Raw `cudaSetDevice`

**What goes wrong:**
HYG-02 tracks `rmm::cuda_stream_default` references (baseline: 40). This is a project constraint: all GPU operations use explicit streams. PR #675's `uring_reactor.cpp` uses raw `cudaSetDevice` (not the project-standard `rmm::cuda_set_device_raii`) and `cudaMemcpyAsync` with a bare `stream.value()`. These are not `rmm::cuda_stream_default` violations per se, but the bare `cudaSetDevice` is effectively a "switch device without tracking" — if the reactor thread was using a non-default stream for something, the device switch leaves that stream's device association stale.

Additionally, `uring_reactor.cpp` uses `cudaFreeHost(p)` directly in the pinned_deleter (not `rmm::mr::get_current_device_resource()->deallocate`), which bypasses the RMM tracking layer. This does not cause HYG-02 to fire but does mean pinned memory allocations in the IO framework are invisible to cucascade's reservation manager — it cannot account for them in memory pressure decisions.

**Why it happens:**
PR #675 was authored independently of the Sirius stream conventions. It is a self-contained IO subsystem that predates the HYG-02 grep gate. No pre-commit hook checks for bare `cudaSetDevice` calls.

**How to avoid:**
- After integrating PR #675, run `grep -rn "cudaSetDevice\b" src/io/` — any raw call is a HYG-class violation. Replace with `rmm::cuda_set_device_raii` RAII guards.
- The `rmm::cuda_stream_default` count must remain ≤ 40 after the IO framework is integrated. Run the HYG-02 grep gate before the first integration test.
- For the pinned memory lifecycle: align `uring_reactor`'s pinned allocations with the project convention (use `cudaHostAlloc(..., cudaHostAllocPortable)` with matching `cudaFreeHost`, acceptable as a named exception since these are not RMM-tracked allocations, but document the exception).

**Warning signs:**
- `grep -c "rmm::cuda_stream_default" src/` returns > 40 after IO framework integration.
- `compute-sanitizer memcheck` reports context mismatches on `cudaMemcpyAsync` calls from the uring_reactor thread.
- HYG-02 grep gate (run as part of phase acceptance) fails.

**Phase to address:** IO Framework integration phase — HYG-02 gate must be run before merge, not just at ship time.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Single shared `sirius_ioctx` for all GPUs | Simpler construction, no per-GPU bookkeeping | Re-introduces the kvikio anti-pattern (single CUDA context for multi-GPU I/O) | Never — per-GPU ioctx is the v1.4 design decision |
| `default` `cuda_stream_view` at `gpu_table_representation` ctor sites that "don't have a stream yet" | Compiles without forcing stream plumbing | Falls back to `cudaDeviceSynchronize` in `convert_gpu_to_gpu`, hiding cross-GPU races at SF1 but not SF100 | Only for CPU-tier representations that truly never have stream-produced data |
| Accepting the `pop_next_data_batch` migration without a blocking wrapper | Faster migration, fewer lines changed | CPU spin loops burn power and cause test flakes under load | Never — always wrap with a proper wait mechanism |
| Carrying `_batch_gpu_affinity` in `duckdb_scan_executor` unchanged after Scan Manager integration | Avoids restructuring the affinity map | Affinity records are populated from the wrong code path (scan_executor no longer drives split allocation) | Acceptable in Phase 16 as a temporary shim if Scan Manager integration is phased; must be resolved before ship |
| Global (not per-GPU) admission_control budget | Single configuration knob, no per-GPU tuning needed | Serializes multi-GPU I/O under load; fails SF100 Q1 perf gate at scale | Never if per-GPU ioctx instances are being created |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| PR #117 × Phase 13 stream-lineage | Accept the `representation_converter.cpp` conflict from one side, losing writer_stream argument | Treat as a re-implementation: start from #117's shape, add writer_stream from `62e0517` manually |
| PR #739 × #117 | Apply #739's operator changes against `0cd4a6a` API, then rebase cucascade | Apply cucascade rebase to `73d00c4` first, then port #739's Sirius-side changes against the new API |
| PR #731 delete × Phase 13 modify | Accept file deletion, re-implement stream-lineage from scratch | Extract stream-lineage attachment points before accepting deletion; re-attach to new scan path explicitly |
| `uring_reactor` × multi-GPU | Create one shared `uring_ioctx`, pass `device_id` per-request, rely on `cudaSetDevice` in reactor thread | Create per-GPU `uring_ioctx` instances, each constructed under `rmm::cuda_set_device_raii` for that GPU |
| `split_connector` × SCHED-RR | Leave round-robin counter in `management_eventloop` (no longer called for source-pipeline splits) | Move counter to `parquet_split_provider`'s split-emission loop so affinity is stamped at production time |
| Cucascade rebase × Portable flag | Resolve memory conflict files by taking #117's side, dropping Portable/Mapped flag additions | Verify `cudaHostAllocPortable` survives conflict resolution; run P2P tests immediately |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Global admission_control for N-GPU I/O | SF100 Q1 2-GPU wall-clock ≥ 1-GPU | Per-GPU ioctx with per-GPU budget | At SF10+ with 2+ GPUs concurrent |
| Single `uring_ioctx` shared across GPUs | GPU 1 shows zero PCIe I/O activity in `nvidia-smi dmon`; all reads go through GPU 0's context | Per-GPU ioctx instances | Any multi-GPU query with parquet scans |
| SCHED-RR counter not ported to split_provider | All splits assigned to GPU 0; GPU 1 idle during scan phase | Port `_no_pref_rr_counter` to `parquet_split_provider::start()` | Every multi-GPU query post-Scan Manager integration |
| `mutable_data_batch` held across tier conversion calls | Pipeline stalls: convert_to() waits forever for exclusive lock | Scope RAII accessors tightly; `readonly_to_mutable` consumes rather than stacks | Any convert_to() call site reached while holding read_only |
| Overly broad `to_mutable()` at sites that only need read | Exclusive lock blocks all concurrent readers, eliminating multi-GPU pipeline parallelism | Audit every `to_mutable()` — use `to_read_only()` unless set_data() is actually called | At 16+ concurrent GPU pipeline tasks |

---

## "Looks Done But Isn't" Checklist

- [ ] **Cucascade rebase conflict resolution:** `grep -n "writer_stream\|writer_event" cucascade/src/data/representation_converter.cpp` returns non-zero at every `gpu_table_representation` construction site
- [ ] **PR #739 API migration:** `grep -rn "task_created\|in_transit\|pop_data_batch(" src/` returns zero results (old batch_state values)
- [ ] **HYG-02 gate:** `grep -c "rmm::cuda_stream_default" src/` ≤ 40 (baseline from v1.3)
- [ ] **Phase 13 stream-lineage:** `grep -rn "writer_stream\|record_writer_event" src/op/scan/` returns non-zero after Scan Manager integration
- [ ] **Portable/Mapped flags:** `grep -n "cudaHostAllocPortable" cucascade/src/memory/` returns results at every pinned allocation site
- [ ] **SCHED-RR counter ported:** `[mgpu_stress]` 500-iter test exits 0 on the rebased branch
- [ ] **Per-GPU ioctx:** `grep -rn "cudaSetDevice\b" src/io/` returns zero raw calls
- [ ] **Disjointedness REQUIRE:** AUDIT TEST_CASE `std::set_intersection(scan_ids) == ∅` fires green on num_gpus=2 after Scan Manager integration
- [ ] **SF100 Q11:** Explicitly run SF100 Q11 num_gpus=2 (not just `[mgpu]` suite) — stream-lineage race only manifests at this data volume
- [ ] **io_worker member order:** Confirm `_thread` is last-declared member in post-rebase `pipeline_io_backend.cpp`'s io_worker class
- [ ] **Submodule pin:** After cucascade rebase, `cat cucascade/.git/HEAD` shows descendant of `73d00c4` with our 11 fixes applied on top

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| writer_stream dropped in representation_converter conflict | MEDIUM — requires targeted re-edit + SF100 re-run | Re-implement the four `convert_*` sites in `representation_converter.cpp` from Phase 13 commit message; re-run SF100 Q11 |
| pop_next_data_batch semantics misunderstood | LOW — compile-time errors surface most sites | grep all old `pop_data_batch(state)` call sites, replace with wait loop; run `[TPC-H][parquet]` correctness check |
| Portable flag lost in conflict | MEDIUM — multi-GPU P2P fails silently at scale | Re-add `cudaHostAllocPortable` to all `cucascade/src/memory/` pinned sites; re-run P2P transfer tests |
| SCHED-RR counter not ported | LOW — functional correctness unaffected, only distribution | Add counter to `parquet_split_provider`'s split loop; re-run `[mgpu_stress]` |
| io_worker member order wrong | HIGH — test-ordering-dependent SIGSEGV; bisect returns NONE; may take 2+ plans | Reorder members, rebuild; run full `[integration][TPC-H]` suite; use Phase 10 bisect→gdb→fix pattern |
| #739 × #117 combinatorial mismatch | MEDIUM — build breaks with 50+ errors, but clearly diagnosed | Accept cucascade rebase to `73d00c4` fully; re-port #739's operator files against new API |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| RAII lock scope / self-deadlock (P1) | Phase 16 (Cucascade rebase + DataBatch API migration) | Helgrind / TSan on `[mgpu]` suite; `[mgpu_stress]` no hang |
| writer_stream × RAII collision (P2) | Phase 16 (cucascade rebase conflict resolution) | `grep` gate on `representation_converter.cpp`; SF100 Q11 2-GPU |
| pop_next_data_batch semantics (P3) | DataBatch API migration phase | `[TPC-H][parquet]` 22/22 correctness on num_gpus=2 |
| uring_reactor single CUDA context (P4) | IO Framework multi-GPU adaptation phase | `compute-sanitizer` on num_gpus=2 parquet tests; GPU 1 PCIe activity |
| Global admission_control budget (P5) | IO Framework multi-GPU adaptation phase | SF100 Q1 2-GPU perf gate ≤ 5.7s |
| SCHED-RR counter stale (P6) | Scan Manager integration phase | `[mgpu_stress]` 500-iter; AUDIT disjointedness REQUIRE |
| PR #739 × #117 ordering (P7) | Phase 16 (pre-apply ordering check) | Build succeeds; zero `task_created` / `pop_data_batch(state)` occurrences |
| io_worker member-order (P8) | Phase 16 (conflict resolution review) | `[integration][TPC-H]` 48/48 with varied test ordering |
| Portable/Mapped flags dropped (P9) | Phase 16 (conflict resolution review) | P2P transfer tests; SF100 GPU 1 PCIe activity |
| Phase 13 work in deleted file (P10) | Scan Manager integration phase | `grep` gate on `src/op/scan/`; SF100 Q11 2-GPU |
| HYG-02 raw `cudaSetDevice` (P11) | IO Framework integration phase | HYG-02 grep gate ≤ 40 before merge |

---

## Sources

- `git -C cucascade show 73d00c4` — PR #117 full diff and breaking-changes list (direct inspection)
- `git show 62e0517` — Phase 13 writer_stream constructor fix (direct inspection)
- `git show 4c0f1ac` — PR #675 IO Framework source: `uring_reactor.cpp`, `admission_control.hpp`, `sirius_datasource.cpp`
- `git show aa0f29a` — PR #731 Scan Manager: `parquet_split_provider.hpp`, `sirius_scan_manager.hpp`, deleted `sirius_parquet_metadata_scan_operator.hpp`
- `git show 468f6e1` — PR #739: cucascade pin bump from `c6bcf34` to `0cd4a6a` (pre-#117)
- `.planning/RETROSPECTIVE.md` — v1.2 Phase 10 post-mortem: member destruction order, SIGSEGV ordering dependency, bisect-returns-NONE pattern
- `.planning/MILESTONES.md` — v1.3 Phase 13 record: writer_event Path-1 vs Path-2, 22 un-migrated producers, conflict file list
- `.planning/PROJECT.md` — Key Decisions table: kvikio anti-pattern rationale, `sirius_p2p_converter` rationale, per-GPU filter translation
- `cucascade/include/cucascade/data/gpu_data_representation.hpp` — current `writer_stream` constructor contract (STREAM-LINEAGE comment)
- Project memory: `feedback_no_stream_default.md` — no `rmm::cuda_stream_default`, always explicit streams

---
*Pitfalls research for: v1.4 Rebase After DataBatch Changes (Sirius multi-GPU GPU SQL engine)*
*Researched: 2026-05-04*
