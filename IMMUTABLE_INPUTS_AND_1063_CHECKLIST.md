# Immutable `execute()` inputs + Issue #1063 — Implementation Checklist & Context

> **Audience:** an engineer (or LLM) picking up this work cold.
> **Two intertwined goals, one end-state.** Read the "How it all stacks up" section first,
> then work the checklist items. Each item is written to be implemented and built
> **independently** and in the order given within its part. File:line references are from
> branch `enfore-task-input-invariants-at-compile-time` as of 2026-08-19; re-grep before editing.

---

## How it all stacks up (the thesis)

We want two guarantees about a **data batch** (`cucascade::data_batch`, the unit of GPU data
that flows between operators and gets parked in repositories between pipeline stages):

1. **Issue #1063 — every batch parked in a repository is `idle`** (holds no read-only pin on
   another batch, and is not itself pinned). Only `idle` batches are spillable by the downgrade
   executor and poppable by the next operator. A parked batch that secretly holds a
   `read_only_data_batch` lock on its source keeps that source pinned `read_only` **forever**,
   which (a) makes the source unspillable and invisible to the downgrade scan → OOM/deadlock,
   and (b) violates the "single physical owner, no batch references another batch" invariant.

2. **Immutable `execute()` inputs — an operator's `execute()` physically cannot mutate its
   inputs.** This buys **retry-safety** (an OOM reschedule re-runs `execute()` on the *same,
   unmodified* input batches) and **fan-out safety** (the same input batch shared by N
   downstream consumers is never mutated by one of them under another's feet).

These compose into a single clean rule for the pipeline:

> A batch handed to `execute()` is **read-only** (Goal 2), and every batch an operator hands
> back to be **parked** is **`idle` and owns only its own storage** (Goal 1). So inputs are
> never mutated and outputs are always independently spillable.

**Goal 2 makes Goal 1 easy to hold:** if `execute()` can't reach a mutable/owning handle to its
input, it *can't* build an output batch that secretly pins that input (which is exactly how the
#1063 producers in `projection.cpp` arise). Conversely, **Goal 1's `idle`-at-park assert
(item A1) is the runtime tripwire** that proves Goal 2's static guarantee actually held.

---

## Current state (what already landed)

- **cuCascade `data_batch`** already reshaped for immutability: `to_idle(read_only&&) →
  std::shared_ptr<const data_batch>` (`data_batch.hpp:162`), `to_read_only()/try_to_read_only()`
  are `const` (`:184`,`:206`), and `_state`/`_read_only_count`/`_subscriber_count` are `mutable`
  (`:330-332`) so a `const` batch can still be locked/observed.
- **`batch_lock_utils.hpp`** reworked: `lock_and_prepare_batch(...)` returns
  `lock_and_prepare_batch_result { std::optional<std::shared_ptr<data_batch>> new_batch;
  read_only_data_batch ro_lock; }` (`:37-43`); the upgrade path uses
  `data_batch::to_idle(std::move(read_accessor)); batch->to_mutable();` instead of a
  `readonly_to_mutable` shortcut.
- **`pipelineable_operator_data`**: the `vector<read_only_data_batch>` constructor was removed;
  `_data_batches` is a plain `std::vector<std::shared_ptr<data_batch>>` (`sirius_physical_operator.hpp:309`);
  `prepare_for_processing` pins each input via `lock_and_prepare_batch` and updates
  `_data_batches[i]` in place when a clone was produced (`operator.cpp:63-113`).
- **`table_scan::execute`** (#1063 Goal-1, Phase 1) already move-reassembles an **owned** batch
  (deep-clone of the input at `table_scan.cpp:151`) instead of a view that pins the source.
- 12 forwarders + 3 `to_idle` re-emitters were migrated to `get_data_batches()`, and the dead
  `leave_locked` parameter was removed.

> ⚠️ **The tree does not compile right now.** `get_data_batches()` was left **non-`const`**
> (`sirius_physical_operator.hpp:254`) but ~20 call sites invoke it through a
> `const pipelineable_operator_data&` / `dynamic_cast<const ...>` (inside `execute()` bodies).
> **Item B0 restores the build.** Do it first.

---

# PART A — Issue #1063: every parked batch is `idle` & self-owning

### A1. Assert batches are `idle` when added to a repository *(tripwire; do first)*

**Where:** `cucascade/include/cucascade/data/data_repository.hpp:75`
`add_data_batch(std::shared_ptr<data_batch> batch, size_t partition_idx = 0)` — and the manager
wrapper `data_repository_manager.hpp:132`.

**Do:** assert the batch is `idle` before parking it.
```cpp
void add_data_batch(std::shared_ptr<data_batch> batch, size_t partition_idx = 0) {
  assert(batch && batch->get_state() == batch_state::idle &&
         "repositories may only hold idle batches (issue #1063)");
  // ... existing insert ...
}
```
**Achieves:** turns every #1063 violation into an immediate, localized failure at the moment of
parking, instead of a deadlock much later in the downgrade scan. This is the executable
statement of the invariant and the regression guard for items A2–A4 and all of Part B.

**Independent?** Yes. Build + run the existing tests; any pre-existing producer (A2) will trip
it, which is the point — gate A1 behind `#ifdef DEBUG` if you want to land it before A2.

---

### A2. Stop `projection::execute` from parking a view that pins its input *(primary #1063 fix)*

**Where:** `src/op/sirius_physical_projection.cpp` — two producers:
- **Path 2, pure passthrough** (`:122-140`): builds `out_view` over the input's columns and calls
  `make_data_batch_from_view(out_view, std::move(input_ro), ...)` where the owner **is the input's
  `read_only_data_batch` lock** (`:137-138`). That lock pins the source `read_only` for the output
  batch's entire parked lifetime → **#1063**.
- **Path 3, mixed** (`:142-179`): owner is `projection_owner { shared_ptr<cudf::table> evaluated;
  read_only_data_batch input_lock; }` (`:173-179`) — same pin via `input_lock`.

**Do:** the output batch must own only its **own** storage and be `idle`. Two options:

- **A2a (minimal, correctness-first):** materialize the passthrough columns into an owned table
  (gather/`std::make_unique<cudf::table>` copy of the referenced columns) so no input lock is
  retained. Mirrors what `table_scan.cpp:147-155` already does via `clone`. Costs one copy of the
  passthrough columns.
  ```cpp
  // instead of make_data_batch_from_view(out_view, std::move(input_ro), ...):
  auto owned = std::make_unique<cudf::table>(out_view, stream, mem.get_default_allocator());
  output_batches.push_back(make_data_batch(std::move(owned), mem, stream, batch_telemetry()));
  // input_ro drops here → source returns to idle → spillable.
  ```
- **A2b (optimal, follow-up):** keep zero-copy but move per-consumer projection **off the batch
  and onto the consumer edge/port** (column-select at one choke-point). The parked batch stays a
  full owned batch; each downstream edge applies its own column permutation at read time. Larger
  refactor; do after A2a proves correctness. (This is the "projection on the edge, not the token"
  decision — needed anyway for fan-out with differing projections; see context dump §5.)

**Achieves:** removes the two real production #1063 producers; parked projection outputs become
`idle`/self-owned and A1 stops tripping on them.

**Independent?** Yes (A2a). Requires A1 only if you want the assert to prove it.

---

### A3. Make `owning_table_view` structurally unable to hold a batch/lock as owner

**Where:** `cucascade/include/cucascade/cudf/gpu_data_representation.hpp:216-220`
```cpp
struct owning_table_view {
  std::any owner;          // ← today: can be ANYTHING, incl. a read_only_data_batch
  std::size_t alloc_size{0};
  cudf::table_view view;
};
```

**Do:** replace the `std::any owner` with a type that can only hold **column storage**, e.g.
`std::vector<std::shared_ptr<cudf::column>>` (or a small owned-columns struct). Update the
templated ctor at `:231-244` and `make_data_batch_from_view` (`data_batch_utils.hpp:193-209`)
accordingly. This makes "a batch owns another batch / a read-only lock" **un-representable**.

**Achieves:** converts the #1063 class of bugs from "reviewer must notice" to "won't compile."
The structural counterpart to A1's runtime assert.

**Independent?** Depends on A2 (A2 is the only remaining caller that stashes a lock as owner;
once A2a lands, no caller passes a lock, so tightening the type is safe). Do **after** A2.

---

### A4. Gate `table_scan`'s unconditional deep-clone to the column-drop path *(perf only)*

**Where:** `src/op/sirius_physical_table_scan.cpp:147-155`. Correctness already landed (Phase 1);
today it deep-clones the first input **unconditionally** even when no filter/projection will drop
columns, which is a needless full copy on the common path.

**Do:** only clone when the scan will actually mutate/drop columns (filter present, or
`num_batch_cols > expected_output_columns`); otherwise forward the owned input batch directly
(it is already `idle` at park). Keep the clone strictly for the column-dropping/reassembly path.

**Achieves:** removes a per-scan-task full-table copy on the no-op path. No behavior change.

**Independent?** Yes.

---

# PART B — Immutable `execute()` inputs (retry-safety + fan-out safety)

**Enforce at the `execute()` API boundary, NOT inside `operator_data`.** The pipeline still needs
`operator_data` to hand out mutable/owning `shared_ptr<data_batch>` for sinking, token
resolution, repository parking, and the downgrade executor. So we keep `operator_data` mutable and
instead give `execute()` a **read-only view** that never exposes a mutable/owning handle.

**The 14 `execute()` overrides** (`grep '::execute(const operator_data'`):
`concat:172`, `cte:75`, `filter:59`, `gpu_values:270`, `hash_join:1583`, `merge_sort:80`,
`operator(base):240`, `order:43`, `partition:169`, `projection:70`, `sort_sample:162`,
`table_scan:106`, `top_n:160`, `top_n_merge:246`.

**The 4 `sink()` overrides:** `concat:203`, `operator(base):230`, `partition:247`,
`materialized_collector:122` (result_collector).

---

### B0. Restore the build: make `get_data_batches()` `const` again *(do first, trivial)*

**Where:** decl `sirius_physical_operator.hpp:254`, def `operator.cpp:40-44`.
```cpp
// header
[[nodiscard]] const std::vector<std::shared_ptr<cucascade::data_batch>>& get_data_batches() const;
// cpp
const std::vector<std::shared_ptr<cucascade::data_batch>>&
pipelineable_operator_data::get_data_batches() const { return _data_batches; }
```
It only `return`s a member now, so `const` is free and correct.

**Achieves:** unblocks compilation — the ~20 `const`-context callers (all the `execute()`-body
forwarders + `sink()` bodies) compile again. **Does not yet enforce anything** (the `const` is
shallow: `input.get_data_batches()[i]->to_mutable()` still compiles). B1–B4 close that.

**Independent?** Yes. This is the prerequisite that makes every other item buildable in isolation.

---

### B1. Let the sink/park path hold `operator_data` **non-`const`** (no `const_cast`)

**Where:** the 4 `sink()` overrides + base virtual (`sirius_physical_operator.hpp:532`
`virtual void sink(const operator_data&, ...)`). Note `publish_output` already receives it
non-`const` (`gpu_pipeline_task.cpp:462` `op::operator_data& output_data`).

**Do:** change `sink(const operator_data&, ...)` → `sink(operator_data&, ...)` on the base and
all overrides. Sinking legitimately needs the **owning** `shared_ptr`s to push into downstream
repositories (`concat.cpp:208`, `result_collector.cpp:127`, `partition.cpp:251`) and those parked
batches must stay mutable-capable and `idle` — so this is genuinely a mutable, pipeline-side
operation, not part of `execute()`'s read-only contract.

**Achieves:** the mutable owner (pipeline/sink) and the read-only view (`execute()`) are held by
**different parties**, so no `const_cast` is ever needed to cross the boundary. Cleanly separates
"produce" (`execute`, read-only in) from "distribute" (`sink`, mutable out).

**Independent?** Yes, after B0.

---

### B2. Add the read-only input view type + the `forward_token` result type *(scaffolding)*

**Where:** new declarations near `pipelineable_operator_data` in `sirius_physical_operator.hpp`.

```cpp
// A read-only facade handed to execute(). Exposes ONLY immutable accessors — there is no
// get_data_batches() here, so execute() cannot obtain a mutable/owning shared_ptr<data_batch>,
// and `input...->to_mutable()` is unspellable.
class execute_input_view {
 public:
  explicit execute_input_view(const pipelineable_operator_data& d) : _d(&d) {}
  [[nodiscard]] std::vector<cucascade::read_only_data_batch> get_read_only_batches() const {
    return _d->get_read_only_batches();
  }
  [[nodiscard]] std::size_t size() const { return get_read_only_batches().size(); }
  // partitioned operators additionally need the partition index:
  [[nodiscard]] std::optional<std::size_t> partition_idx() const;  // null unless partitioned
 private:
  const pipelineable_operator_data* _d;
};

// execute() result: per output, either "re-emit input i unchanged" or "here is a new owned batch".
struct forward_token { std::size_t input_index; };
using execute_output = std::variant<forward_token, std::shared_ptr<cucascade::data_batch>>;
```

**Achieves:** provides the types B3/B4 migrate onto, without touching any operator yet. Compiles
on its own (unused types).

**Independent?** Yes, after B0.

---

### B3. Migrate `execute()` to take the view and return forward-tokens/new batches

**Where:** the 14 overrides above + base decl (`sirius_physical_operator.hpp`, look for the
`virtual std::unique_ptr<operator_data> execute(const operator_data&, ...)` decl).

**Do, per operator:**
1. Change the signature to accept `execute_input_view` (and return the token-carrying result — see
   B4 for the exact return shape) instead of `const operator_data&`.
2. Replace every in-body `input.get_data_batches()` **forward** with a `forward_token`:
   - `table_scan.cpp:116` (passthrough), `merge_sort.cpp:117`, `ungrouped_aggregate.cpp:474`,
     `concat.cpp:195`, `partition.cpp:191`, `sort_partition.cpp:66`, `sort_sample.cpp:172/191/355`,
     `grouped_aggregate_merge.cpp:215`, `cte.cpp:80`, `delim_join.cpp:202/210`,
     `column_data_scan.cpp:123`, `result_collector.cpp:72`, `streaming_source.cpp:115`,
     `dynamic_filter.cpp:61/64` → `return { forward_token{i} };`.
3. Compute operators (`filter`, `order`, `hash_join`, `projection` non-passthrough,
   `top_n`, real `table_scan` path) already build **new** batches from
   `get_read_only_batches()` — they just return the new `shared_ptr<data_batch>` in the variant.
4. After migration, **no `execute()` body references `get_data_batches()`** — verify with grep.

**Achieves:** `->to_mutable()` (or any mutation) of an input is now a **compile error** inside any
`execute()`. Retry-safety and fan-out-safety become structural.

**Independent?** Do operator-by-operator; each override compiles once its signature + body are
updated and B4's resolver is in place. Pairs with B4.

---

### B4. Resolve tokens in `run_one_operator` (the mutable owner side)

**Where:** `run_one_operator` (`gpu_pipeline_task.cpp:169-186`) calls `op.execute(...)`.

**Do:**
- `run_one_operator`'s input is currently `const op::operator_data& operator_input_data`
  (`:171`); it must hold the **mutable** `pipelineable_operator_data` (the pipeline owns it —
  trace back from `local_state._input_data`) so it can resolve tokens.
- Wrap the input in an `execute_input_view` and pass that to `op.execute(view, stream)`.
- Map the returned `std::vector<execute_output>` into the output `pipelineable_operator_data`:
  ```cpp
  std::vector<std::shared_ptr<cucascade::data_batch>> out;
  for (auto& o : results)
    out.push_back(std::holds_alternative<forward_token>(o)
        ? input._data_batches[std::get<forward_token>(o).input_index]  // pipeline's OWN mutable owner
        : std::get<std::shared_ptr<cucascade::data_batch>>(o));
  ```
  The forwarded batch is the pipeline's own owning `shared_ptr` (idle at park) — **no `const_cast`,
  no lock retained**.

**Achieves:** completes the boundary: `execute()` sees only read-only inputs; the pipeline
resolves "forward input i" against the mutable owner it never gave up. Forwarded batches stay
`idle` (satisfying A1) and mutable-capable for the next stage.

**Independent?** Pairs with B3 (implement together; the resolver must exist before a token-returning
`execute()` runs).

---

### B5. Final lockdown + verify

**Do:**
- Confirm `grep -rn 'get_data_batches' src/op` shows **zero** hits inside `execute()` bodies (only
  sink/pipeline internals remain).
- Keep `get_data_batches()` `const` and pipeline-internal; it is no longer reachable from
  `execute()` because `execute()` receives `execute_input_view`, which does not expose it.
- Run the suite; A1's assert now also guards that every forwarded/parked batch is `idle`.

**Achieves:** the two guarantees now hold **structurally** (won't compile otherwise) and are
**checked** at runtime (A1). Done.

---

## Suggested order

`B0` (build) → `A1` (tripwire, DEBUG-gated) → `A2a` (kill real #1063 producers) →
`B1` (sink non-const) → `B2` (types) → `B3`+`B4` together (the boundary, operator-by-operator) →
`A2b`/`A3`/`A4` (optimizations & structural lockdown) → `B5` (verify) → flip `A1` on for all builds.

---
---

# CONTEXT DUMP (for future Q&A)

Everything below is background so an LLM can answer questions about this work without the original
thread. Facts are from branch `enfore-task-input-invariants-at-compile-time`, 2026-08-19.

## §1. What Sirius is (relevant slice)

Sirius is a GPU-native SQL engine running as a DuckDB extension ("Super Sirius" is the live
engine; `src/op/` operators, `src/pipeline/`, `src/planner/`, `src/cuda/`). **cuCascade**
(`cucascade/` submodule) is the generic, cudf-free-core data/memory layer: three memory tiers
(GPU/HOST/DISK), reservation manager, converter registry, and the `data_batch` locking system.
`src/legacy/` is dead — never touch.

## §2. The `data_batch` locking model (cuCascade)

`cucascade/include/cucascade/data/data_batch.hpp`:
- `enum class batch_state { idle, read_only, mutable_locked };` (`:53`).
- `data_batch` owns a `unique_ptr<idata_representation>` (the physical storage — a
  `gpu_table_representation` wrapping either a `unique_ptr<cudf::table>` or an
  `owning_table_view`).
- Three-accessor system:
  - `read_only_data_batch to_read_only() const` (`:184`) — **shared** lock; multiple concurrent
    readers allowed (`_read_only_count`).
  - `mutable_data_batch to_mutable()` (`:198`) — **exclusive** lock; blocks until all readers drop.
  - `static std::shared_ptr<const data_batch> to_idle(read_only_data_batch&&)` (`:162`) and
    `static std::shared_ptr<data_batch> to_idle(mutable_data_batch&&)` (`:170`) — release a lock,
    returning the batch to `idle`.
- `get_state()` (`:144`), `get_read_only_count()` (`:154`), `get_subscriber_count()` (`:134`).
- `_state`, `_read_only_count`, `_subscriber_count` are **`mutable std::atomic<…>`** (`:330-332`)
  so a `const data_batch` can still be locked/observed (needed after `to_idle` returns `const`).
- `subscribe()` = a lighter reservation (interest count) that does **not** take a lock — used to
  reserve a batch against eviction without pinning its state.

## §3. Why #1063 happens (the deadlock chain)

- The **downgrade executor** (spills GPU→HOST/DISK under memory pressure) and the next operator's
  pop **only consider `idle` batches**: `convertible_data_batch.hpp:331`
  `if (batch->get_state() != cucascade::batch_state::idle) return nullptr;` (the idle-only
  `try_get_batch` used by the two-tier downgrade scan at `:218-267`).
- If a parked batch's `idata_representation` is a `gpu_table_representation` holding an
  `owning_table_view` whose `std::any owner` **is a `read_only_data_batch` on the source**, then
  the source is pinned `read_only` for as long as the parked batch lives. The source is therefore
  never `idle` → never spillable, never poppable → under pressure the pipeline deadlocks / OOMs.
- Producers today: `projection.cpp:137` (passthrough owner = `input_ro`) and `projection.cpp:178`
  (mixed owner = `projection_owner{evaluated, input_lock}`). `table_scan` used to do this too but
  was fixed to deep-clone an owned batch (`table_scan.cpp:151`).
- Invariants #1063 restores: (1) parked ⇒ `idle`; (2) single physical owner per storage;
  (3) no `data_batch` references another; (4) `idata_representation` = physical storage only;
  (5) cuCascade stays generic; (6) preserve `read_only` multi-reader concurrency for legit
  prolonged holds (see §6); (7) per-consumer column-select at one choke-point; (8) `add_data_batch`
  asserts `idle`.

## §4. Why immutable `execute()` inputs (the two payoffs)

- **Retry-safety:** the pipeline reschedules a task on OOM (`oom_reschedule_exception`,
  `gpu_pipeline_task.cpp:427/565`; `create_rescheduled_task:747`). The reschedule replays
  `execute()` on the *same* input batches. If `execute()` had mutated an input on the first
  (failed) attempt, the replay would run on corrupted data. Read-only inputs ⇒ replay is sound.
- **Fan-out safety:** `data_repository_manager::add_data_batch(batch, ops)` copies the *same*
  `shared_ptr<data_batch>` into multiple downstream repositories (one physical batch, N consumers).
  If consumer A's `execute()` could mutate it, consumer B would see corruption. Read-only inputs
  ⇒ safe sharing. Per-consumer **projection lives on the consumer edge/port**, not on the batch or
  a token (a single forwarded token can't carry N different projections — hence A2b/edge-projection).

## §5. The chosen enforcement design (and the rejected alternative)

**Chosen (facade / Option A):** enforce at the `execute()` boundary. `execute()` receives a
read-only **`execute_input_view`** (exposes `get_read_only_batches()` + table views + partition
idx; **no** `get_data_batches()`), and returns a `std::vector<execute_output>` where
`execute_output = variant<forward_token, shared_ptr<data_batch>>`. The pipeline
(`run_one_operator`) holds the mutable owner and resolves `forward_token{i}` by copying its own
`_data_batches[i]`. `operator_data` stays fully mutable for sink/park/downgrade.

**Rejected (Option B):** narrow `get_data_batches()` to return `vector<shared_ptr<const
data_batch>>`. Rejected because it (a) forces `const_cast` at the sink/re-park boundary (the
pipeline genuinely needs mutable owners) and (b) cripples `operator_data` for its legitimate
pipeline-side uses. Enforcement belongs at the *execute boundary only*.

Key subtlety — **no `const_cast` in the chosen design** because the mutable owner (pipeline) and
the read-only view (`execute`) are held by *different parties*; the batch never crosses the const
boundary. This is why the view + token shape was preferred over a single "always-const
`get_data_batches`".

## §6. Legit prolonged `read_only` holds (must be preserved by both goals)

- `prepare_for_processing` (`operator.cpp:63-113`) pins each input `read_only` for the whole
  `execute()` (via `lock_and_prepare_batch`, stored in `_read_only_data_batches`). This is a
  *self* pin on the input being processed — released after execute; not a #1063 park.
- Hash-join `build_table` (`per_partition_build_state.build_table` = `optional<read_only_data_batch>`)
  is held `read_only` across all probes of that partition — a legit long-lived shared lock,
  concurrently read by multiple probe tasks (possibly on other GPUs). #1063's fix must **not**
  break multi-reader `read_only` concurrency; that's why the cross-GPU path in
  `lock_and_prepare_batch` (`batch_lock_utils.hpp:123-140`) *clones under the shared lock* rather
  than taking an exclusive one.

## §7. `batch_lock_utils.hpp` — `lock_and_prepare_batch` shape

Returns `lock_and_prepare_batch_result { optional<shared_ptr<data_batch>> new_batch;
read_only_data_batch ro_lock; }`. Same-space ⇒ `{nullopt, ro_lock on batch}`. Cross-GPU ⇒ clone
under the shared lock, `{clone, clone->to_read_only()}` (source left idle → spillable).
HOST/DISK→GPU ⇒ `to_idle(read_accessor)` then `batch->to_mutable()` + `convert_to` in place
(move), `{nullopt, batch->to_read_only()}`, with a non-atomic-upgrade re-dispatch guard for the
gap between releasing shared and acquiring exclusive.

## §8. `operator_data` / `pipelineable_operator_data` API (post-refactor)

`src/include/op/sirius_physical_operator.hpp`:
- `get_data_batches()` — decl `:254` (currently **non-`const`**, returns
  `const vector<shared_ptr<data_batch>>&`; **make `const` in B0**).
- `get_read_only_batches() const` — `:260` (returns cached `_read_only_data_batches` if present,
  else lazily builds from `_data_batches`; `operator.cpp:46-61`).
- `prepare_for_processing(const memory_space*, stream)` — `operator.cpp:63-113`.
- `remove_read_only_lock()` — resets `_read_only_data_batches = nullopt` (`:263-268`).
- members: `_data_batches` (`:309`, plain vector), `_read_only_data_batches`
  (`:310`, `optional<vector<read_only_data_batch>>`).
- `partitioned_operator_data : public pipelineable_operator_data` (`:319`) adds a partition index.

## §9. Pipeline execute→sink flow

`run_one_operator` (`gpu_pipeline_task.cpp:169`): `op.execute(operator_input_data, stream)`
(`:186`), sticky-CUDA-error checks, returns output `operator_data`. `publish_output(op::operator_data&
output_data, …)` (`:462`) calls `sink_operators->sink(output_data, stream)` (`:473`) — **already
non-`const`**, so B1 (sink non-const overrides) needs no caller change. OOM →
`oom_reschedule_exception` (`:427`), `prepare_for_processing` OOM → reschedule (`:565`).

## §10. `get_data_batches()` call-site census (for B0/B3)

`const`-context (inside `execute()` bodies — these are the forward sites B3 converts to
`forward_token`): `table_scan.cpp:116`, `merge_sort.cpp:117`, `ungrouped_aggregate.cpp:474`,
`concat.cpp:195`, `partition.cpp:191`, `sort_partition.cpp:66`, `sort_sample.cpp:172/191/355`,
`grouped_aggregate_merge.cpp:215`, `cte.cpp:80`, `delim_join.cpp:202/210`,
`column_data_scan.cpp:123`, `result_collector.cpp:72`, `streaming_source.cpp:115`,
`dynamic_filter.cpp:61/64`.
Pipeline/sink-internal (stay, hold non-`const` owner): `task_creator.cpp:398/440/443`,
`convertible_gpu_pipeline_task.hpp:117/309`, `gpu_pipeline_task.cpp:68/282/690`,
`concat.cpp:208` (sink), `partition.cpp:251` (sink), `result_collector.cpp:127` (sink),
`operator.cpp:233` (base sink).

## §11. Reference

Issue: https://github.com/sirius-db/sirius/issues/1063 — "Ensure all data batches stored in
repositories are idle (no read-only pins via owning_table_view owners)."
Related PR discussion referenced during design: NVIDIA/cuCascade#120 (mbrobbel comments on the
libguarded-style lock API limitations).

Prior notes files in repo root: `ISSUE_1063_NOTES.md`, `IMMUTABLE_EXECUTE_INPUTS_CHECKLIST.md`
(earlier iterations; this file supersedes them).

## §12. Build / verify

No GPU/CUDA on the current dev box — cannot compile here. Real verification:
`pixi run make` (release) / `pixi run make test` on a GPU box. Pre-commit:
`pixi run pre-commit run -a`. One known pre-commit nit: `table_scan.cpp:206` redundant
`// Apply the projection on output_batch` comment.
