# Compile-time enforcement: operator `execute()` inputs are immutable

> **Goal.** Make it a **compile error** for an operator's `execute()` to mutate its input
> `data_batch`es. This guarantees the two invariants:
> 1. **Retry-safety** — on OOM/CUDA-launch reschedule the pipeline replays operators from
>    `resume_operator_index` over the *same* input batches; if an operator had mutated an input
>    in place before the failure, the retry runs over corrupted data. (See
>    `gpu_pipeline_task.cpp` `oom_reschedule_exception` / `cuda_launch_reschedule_exception`, and
>    `gpu_pipeline_executor.cpp:376-381` `release_intermediate_data` + `remove_read_only_lock`.)
> 2. **Fan-out / concurrent-read safety** — the same batch is shared (fan-out via `shared_ptr`
>    copies into multiple repos; multi-GPU join build read by many probes). No consumer may mutate
>    a batch a sibling consumer also holds.
>
> **Key fact (from the audit).** *No operator mutates its input today.* All `to_mutable()` /
> `release_table()` in operators are on **self-owned** batches (fresh clone/concat/filtered/
> materialized outputs): `table_scan.cpp:224`, `grouped_aggregate_merge.cpp:244`,
> `scan/sirius_gpu_scan_operator_data.cpp:51`. So this is **type enforcement of already-true
> behavior**, not a behavioral change.
>
> **Mechanism.** Narrow the *return types* of the read path so a read handle yields a
> `shared_ptr<const data_batch>` (pointee-const), which makes `->to_mutable()` fail to compile.
> Keep the batch mutable *outside* `execute()` (prepare-time placement, downgrade, forwarding
> re-park) by localizing the single `const_pointer_cast` to two named infra boundaries.
>
> **Scoping insight.** Immutability is a *phase* (during `execute()`), not permanent — batches are
> mutated legitimately **before** (`lock_or_prepare_batch`) and **after** (`convertible_data_batch::
> convert`, forwarding). Do **not** try to make `shared_ptr<const data_batch>` the universal
> currency: `data_batch`'s `to_read_only()`/`to_mutable()` are non-const, so a const `shared_ptr`
> would block reads too.

Line numbers are against branch `fix/1063-projection-edge`; they will drift — navigate by symbol.

---

## Phase 0 — Baseline
- [ ] Build green first: `pixi run make`. Run the operator suite you'll use as the regression
  signal: `pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[physical_table_scan]"`
  (plus `[physical_filter]`, join/aggregate tags). *Why:* this change is compiler-driven; you want
  a clean baseline so every new error is attributable.

## Phase 1 — cuCascade: close the "read handle → mutable owner" escapes

### 1a. Confirm the representation level is already safe (no change)
- [ ] Verify `read_only_data_batch::get_data()` returns `const idata_representation*`
  (`cucascade/include/cucascade/data/data_batch.hpp:298`). *Why:* this already makes
  `get_data()->cast<gpu_table_representation>()` select the **const** `cast<>` overload
  (`common.hpp:176`), so `release_table()`/`rebind_stream()` are already un-callable via a read
  handle. Nothing to do — just confirm.

### 1b. Make `to_idle(read_only&&)` and the read handle's owner pointer const
This is the primary leak: `to_idle` hands back a non-const `shared_ptr<data_batch>` on which you
can call `to_mutable()`.
- [ ] `read_only_data_batch::_batch` (`data_batch.hpp:407`): change to
  `std::shared_ptr<const data_batch> _batch;`
- [ ] `to_idle(read_only_data_batch&&)` (`data_batch.hpp:160`, impl `data_batch.cpp:89`): return
  `std::shared_ptr<const data_batch>`.
  ```cpp
  // data_batch.hpp
  [[nodiscard]] static std::shared_ptr<const data_batch> to_idle(read_only_data_batch&& accessor);
  // data_batch.cpp
  std::shared_ptr<const data_batch> data_batch::to_idle(read_only_data_batch&& accessor) {
    std::shared_ptr<const data_batch> ptr = accessor._batch;   // _batch is now const-owning
    { auto _ = std::move(accessor); }                          // release the shared lock
    return ptr;
  }
  ```
- [ ] `const`-qualify the read transitions so a `const data_batch` can still be *read*:
  `to_read_only()` (`:182`) and `try_to_read_only()` (`:200`) become `const` members. They use
  `shared_from_this()`; on a const `this` that yields `shared_ptr<const data_batch>`, which the
  `read_only_data_batch` ctor now accepts.
- [ ] Mark the counters `mutable` so those const methods can update them:
  `mutable std::atomic<batch_state> _state;` and `mutable std::atomic<size_t> _read_only_count;`
  (`:269-270`). `_rw_mutex` is already `mutable` (`:267`).

*Why:* now `data_batch::to_idle(std::move(ro))->to_mutable()` **won't compile** — `to_mutable()`
is a non-const member and the pointer is `shared_ptr<const>`. This is the core of the guarantee.

### 1c. Make `readonly_to_mutable` the single, named escalation
`readonly_to_mutable(read_only_data_batch&&)` (`:222`) exists to upgrade a reader to a writer. It's
used **only** by prepare-time placement (`batch_lock_utils.hpp`), never by operators.
- [ ] Inside `readonly_to_mutable` (`data_batch.cpp:141`), the `accessor._batch` is now
  `shared_ptr<const>`, so add the one deliberate escalation:
  ```cpp
  auto mutable_ptr = std::const_pointer_cast<data_batch>(accessor._batch);  // ESCALATION #1
  ```
- [ ] (Recommended) restrict its callers: make `readonly_to_mutable` private + `friend` the
  prepare path, or leave it public with a loud comment. *Why:* it's the only sanctioned
  read→write upgrade; keeping it greppable/auditable is the whole point.

### 1d. (Optional, defense-in-depth) const repo read accessors
The repository returns non-const `shared_ptr<data_batch>` from `get_data_batch_by_id`/`pop_*`
(`data_repository.hpp:99/127/161`), so code could *re-fetch by id* to launder a mutable pointer.
- [ ] Add `get_data_batch_by_id(...) const -> std::shared_ptr<const data_batch>` (keep the mutable
  `pop_*` for the consuming/downgrade path; storage stays `shared_ptr<data_batch>`). *Why:* closes
  the re-fetch bypass without touching the mutable ownership the downgrade engine needs. Skip if
  the repository is considered trusted infra.

## Phase 2 — Sirius: make the input's mutable batch accessor unreachable from `execute()`

The second leak: `pipelineable_operator_data::get_data_batches()` (`sirius_physical_operator.hpp:260`,
impl `sirius_physical_operator.cpp:40`) is a `const` method returning
`const vector<shared_ptr<data_batch>>&` — the *pointees* are non-const, so
`input.get_data_batches()[i]->to_mutable()` compiles.

- [ ] Narrow the pointee to const. Change the member and the accessor:
  ```cpp
  // sirius_physical_operator.hpp
  mutable std::optional<std::vector<std::shared_ptr<const ::cucascade::data_batch>>> _data_batches;
  [[nodiscard]] const std::vector<std::shared_ptr<const ::cucascade::data_batch>>&
    get_data_batches() const;
  ```
  The impl (`:40`) already builds `_data_batches` from `to_idle(...)`, which now returns
  `shared_ptr<const data_batch>` — so it type-checks unchanged.
- [ ] Leave `get_read_only_batches()` (`:266`) as-is — it already returns `read_only_data_batch`
  (immutable; `get_data()` const).

*Why:* now `execute()` (which gets `const operator_data&`) can only obtain **immutable** handles —
`read_only_data_batch` (read) or `shared_ptr<const data_batch>` (forward). `->to_mutable()` fails to
compile from any operator body. The compiler now becomes your worklist for Phase 3.

## Phase 3 — Fix the re-park boundary + migrate the flagged forwarders

Narrowing Phase 2 breaks exactly the sites that **forward** an input batch downstream, because the
downstream currency (`add_data_batch`) is `shared_ptr<data_batch>` (mutable — the repo must stay
mutable so the **downgrade engine** can spill parked batches).

### 3a. The re-park boundary — ESCALATION #2 (infra, localized)
- [ ] `sink()` (`sirius_physical_operator.cpp:248`) iterates `get_data_batches()` (now
  `shared_ptr<const>`) and calls `push_data_batch(...)`. Do the single const→mutable conversion
  where a batch re-enters the owning/mutable world:
  ```cpp
  // push_data_batch / the sink→add_data_batch hop:
  p->repo->add_data_batch(std::const_pointer_cast<::cucascade::data_batch>(batch));  // ESCALATION #2
  ```
  *Why:* re-parking hands the batch back to the layer that legitimately mutates it (downgrade,
  next-task prepare). This is the same principled escalation as `readonly_to_mutable` — localized
  to infra, greppable, never reachable from operator logic. Comment it as such.

### 3b. Migrate the two `execute()` bodies that forward via `get_data_batches()`
These break in Phase 2; switch them to forward via **read handles** (no mutable pointer needed):
- [ ] `sirius_physical_streaming_source.cpp:109` — replace
  `pipelineable_operator_data>(pod.get_data_batches())` with
  `pipelineable_operator_data>(pod.get_read_only_batches())`.
- [ ] `scan/sirius_physical_dynamic_filter.cpp:61,64,84` — same substitution for the no-filter
  fast path and the zero-copy passthrough of undropped batches.
  *Why:* `pipelineable_operator_data(vector<read_only_data_batch>)` forwards the same batches via
  the read-lock path — immutable, and the re-park const_cast (3a) happens later in `sink`.

### 3c. The three `to_idle` re-emitters
`concat.cpp:195`, `merge_sort.cpp:117`, `ungrouped_aggregate_merge.cpp:474` do
`output.push_back(to_idle(std::move(ro)))`. `to_idle` now returns `shared_ptr<const>`, which won't
go into a `vector<shared_ptr<data_batch>>`.
- [ ] Prefer re-emitting via read handles instead of `to_idle`:
  ```cpp
  // was: outputs.push_back(cucascade::data_batch::to_idle(std::move(ro)));
  // now: construct the output operator_data straight from the read handles
  return std::make_unique<pipelineable_operator_data>(
    std::vector<cucascade::read_only_data_batch>{ std::move(ro) });
  ```
  *Why:* keeps the whole forward on the immutable read-handle path; the mutable re-park is done once
  in `sink` (3a). (If a given site genuinely needs the owning pointer, use the ESCALATION #2 cast
  there and comment it.)

### 3d. The two `sink()` sites that read input via `get_data_batches()`
- [ ] `partition.cpp:251` and `result_collector.cpp:127` — adjust to the const pointee; apply
  ESCALATION #2 at the actual `add_data_batch`/repo hop, or read via `get_read_only_batches()`.
  *Why:* same re-park boundary; make the const→mutable step explicit and local.

> **Compiler-driven completeness:** after Phases 1–2, `pixi run make` will list *every* remaining
> site that tries to obtain a mutable input batch. Per the audit the set is exactly 3b+3c+3d; if the
> compiler flags anything else, it is a previously-hidden input mutation — investigate it, don't
> reflexively `const_cast`.

## Phase 4 — Verify the exempt writers are untouched
These must still compile and keep mutable access; none go through `execute()`'s const input:
- [ ] Downgrade/spill engine: `convertible_data_batch::convert` (`convertible_data_batch.hpp:88-149`).
- [ ] Prepare-time placement: `lock_or_prepare_batch` (`batch_lock_utils.hpp:67-186`) — uses
  ESCALATION #1 (`readonly_to_mutable`) legitimately, before `execute()`.
- [ ] Operator self-owned output mutation: `table_scan.cpp:224/228`,
  `grouped_aggregate_merge.cpp:244/248`, `scan_operator_data.cpp:51` — these hold their **own**
  mutable `shared_ptr<data_batch>` (a fresh clone/concat/filtered batch), unaffected by input
  narrowing. Confirm they still build.

## Phase 5 — Prove the guarantee + regress
- [ ] Add a **compile-fail** guard (documents the invariant): in a test TU, assert that mutating an
  input does not compile — e.g. a `static_assert` on
  `!std::is_invocable_v<decltype(&cucascade::data_batch::to_mutable), const cucascade::data_batch&>`
  or a commented `// must not compile:` snippet doing
  `input.get_data_batches()[0]->to_mutable();`. *Why:* locks the invariant against regressions.
- [ ] Run `pixi run make test`; run the operator + integration suites (TPC-H validation) — results
  must be unchanged (this is pure type enforcement).
- [ ] Sanity-check the retry path still reschedules: a query that OOMs and retries must still
  succeed (the reschedule now provably feeds unmodified inputs).

---

## The escalation surface (what remains mutable, on purpose)
After this change, exactly **two** named, greppable `const_pointer_cast` boundaries exist, both in
infra, neither reachable from operator `execute()`:
1. **`readonly_to_mutable`** — prepare-time host→GPU readback / cross-GPU handling
   (`batch_lock_utils.hpp`).
2. **Re-park boundary** (`sink`/`push_data_batch` → `add_data_batch`) — a forwarded batch re-enters
   the mutable repo currency so the downgrade engine can spill it.

Everything else — every `execute()` body — can only touch its inputs through immutable handles.

## Decision you may revisit
This checklist uses **return-type narrowing + two localized `const_pointer_cast`s** (minimal churn,
keeps `execute(const operator_data&)`). The airtight-but-heavier alternative is a **dedicated
read-only input facade type** passed to `execute()` that exposes only `get_read_only_batches()` /
`get_input_table_view()` and a pipeline-owned `forward_input(i)` — eliminating both escalations from
the operator-visible surface at the cost of changing the `execute()` signature across the virtual
base + ~29 operators + `run_one_operator`. Start with the narrowing approach; escalate to the facade
only if the two `const_pointer_cast`s prove insufficiently disciplined.
