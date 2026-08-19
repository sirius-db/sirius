# Issue #1063 — design notes & implementation checklist

> **Issue:** Ensure all data batches stored in repositories are idle (no read-only pins via `owning_table_view` owners).
> <https://github.com/sirius-db/sirius/issues/1063>
>
> **Status (as of writing):** design settled; **no code applied to `dev`.** A partial first-pass
> implementation + a deterministic reproducer test live (uncommitted) in the worktree
> `/home/dvats/repos/sirius-repro-1063`. This tree (`/home/dvats/repos/sirius`) is clean on `dev`
> (~`4d88a0fe`). Line numbers below are against that tree and will drift — navigate by symbol.

---

## CONTINUATION PROMPT (paste to resume)

> We are fixing sirius issue #1063 (view-backed data_batches pin their source `read_only` →
> unspillable + invisible to the downgrader → deadlock). Read `ISSUE_1063_NOTES.md` at the repo
> root for the full context. We have SETTLED the design (see "Settled design" + "Case handling" +
> the phased "Implementation checklist"). The invariant is: *no batch parked in a repository may
> hold a `read_only` owner on another batch.* Pure-passthrough projections become a per-edge
> column permutation applied through a single input-view accessor (zero-copy, shared-safe); mixed
> projections and the column-dropping scan become owned batches; enforcement is an `add_data_batch`
> idle-assert plus changing `owning_table_view`'s owner from `std::any` to
> `std::vector<std::shared_ptr<cudf::column>>`. The pinned-GPU scan is the one legitimate view
> producer and is KEPT. Continue from the checklist. Do NOT commit unless asked. There is no
> GPU/CUDA toolchain on this box, so I cannot build/run — flag anything unverified.

---

## The bug (root cause)

View-backed batches from `sirius::make_data_batch_from_view` (`src/include/data/data_batch_utils.hpp:175`)
carry a type-erased owner that is a `read_only_data_batch` **lock on a source `data_batch`**. While
the view-backed batch is alive — including while it just sits parked in a `shared_data_repository` —
the source stays in `batch_state::read_only` and can never return to `idle`. Because:

- the downgrade candidate scanner only considers `idle` batches
  (`src/include/data/convertible_data_batch.hpp`, `try_get_batch` skips non-idle/subscribed), and
- the source is often **popped out of its repo** (`sirius_physical_operator.cpp` `get_next_task_input_data`),
  so it is reachable only through the view's owner and is **invisible** to the two-tier downgrade
  scan (repos + task queue; there is no global resident-batch registry),

the source's GPU memory becomes unspillable → under memory pressure the downgrade executor finds no
viable candidate and the pipeline deadlocks (`src/downgrade/downgrade_executor.cpp` logs
"no viable downgrade target … backing off").

### The 3 pinning producers (owner = `read_only_data_batch`) — must all go
| Site | What |
|---|---|
| `src/op/sirius_physical_projection.cpp:137` | pure-passthrough projection (Path 2) |
| `src/op/sirius_physical_projection.cpp:178` | mixed evaluated+passthrough projection (Path 3) |
| `src/op/sirius_physical_table_scan.cpp:248` | scan dropping filter-only columns |

### The 1 legitimate view producer (owner = `std::vector<std::shared_ptr<cudf::column>>`) — KEEP
`src/scan_manager/sirius_scan_manager.cpp:195,217` — the **pinned-GPU-table scan**. It shares the
pinned columns' *device buffers* by refcount; it pins **no batch's state**. This is why
`owning_table_view` was created, and it stays. (It's intentionally resident — pinning = "don't
spill" — so it's simply not a spill candidate.)

### Production query shapes that hit it
- A retained all-passthrough projection: `SELECT b, a FROM t` (reorder), `SELECT a, c FROM t`
  (subset), `SELECT a, a FROM t` (dup). The planner only *elides* the identity projection
  (`src/planner/sirius_plan_projection.cpp:67-76`); any reorder/subset/dup is retained → view path.
- A scan carrying filter-only columns it must drop (`table_scan.cpp:248`).
- Severity escalates to a **total deadlock** when the view batch is in-flight in a memory-blocked
  consumer: the view is read-locked by the consumer AND the source is read-locked by the view's
  owner, so `try_to_mutable` fails on both and the spiller frees nothing.

---

## Invariants we settled on (with why)
1. **Only `idle` batches are spillable/movable; every batch parked in a repository must be `idle`/self-owned.** — Downgrade skips non-idle; a pinned source is unspillable *and* invisible.
2. **A batch's bytes have exactly one physical owner.** — Two owners ⇒ spilling one frees nothing (copy-and-keep) and duplicates on downgrade.
3. **No `data_batch` references/owns another `data_batch`** (not via representation, a mode field, or an extension). — Otherwise state/lock/downgrade/accounting invariants become recursive; category error.
4. **`idata_representation` = physical tier storage only.** — A logical projection owns no bytes; it doesn't belong there.
5. **cuCascade stays generic** (no cudf/SQL/projection in `data_batch`/repository/downgrade). — It's the cudf-free engine; interpreters are Sirius-side.
6. **`read_only` multi-reader concurrency is preserved.** — Multi-GPU join build shared across probes, dynamic-filter build, sink fan-out legitimately read one owned batch from many places.
7. **Column-select ops are applied at ONE choke-point, called from many sites.** — Makes correctness local and compile-enforceable, not a 20-operator distributed invariant.
8. **`add_data_batch` asserts the batch is `idle`.** — The runtime guard #1063 asks for.

## Rejected forks (with why)
- **`shared_ptr<data_batch>` owner** — keeps the box alive, not the buffers; downgrade swaps the representation → dangling `table_view` (raw pointers).
- **`read_only_data_batch` owner (status quo)** — buys buffer stability by freezing source state → the bug.
- **`shared_ptr<cudf::column>` sharing as the *general* fix** — two owners ⇒ downgrade doesn't free. (Kept only for the pinned scan, which owns shared columns and is intentionally resident.)
- **Lazy view / projection-as-a-mode / projection-in-an-extension** — all reduce to a batch owning a batch (violates #3).
- **Raising the pipeline currency to a `{batch, indices}` handle** — forces cuCascade's generic core to become projection-aware (violates #5).
- **Producer-materialize-by-copy as the *primary* path** — correct but loses zero-copy for the common passthrough case.
- **"Move the input's columns" as a *general* zero-copy trick** — only valid when the producer solely owns its input; breaks under fan-out (violates #6).

## The trilemma that forces the design
For a passthrough result you can have at most two of: **{flows as its own batch, no copy, no batch
references/co-owns another}**. Since #3 (no reference) is non-negotiable, a distinct passthrough
batch must **copy**; otherwise it is **not a batch** (edge metadata / consumed-and-moved). Every
rejected design violated exactly this.

---

## Settled design

Two mechanisms, unified by *no parked batch holds a `read_only` owner on another batch*:

1. **Pure-passthrough projection → edge permutation + view-at-read.** The projection produces **no
   batch**; the source flows as one (possibly shared) owned batch; each consumer's input edge
   carries a `std::vector<cudf::size_type>` permutation, applied as a **zero-copy `cudf::table_view::select`
   on a view at read**, through the single accessor `get_input_table_view(i)`. Making that accessor
   the *only* way to read input columns (raw path removed → bypass = compile error) is what makes
   it verifiable. Batch-forwarders (`dynamic_filter` probe, `cte`, `concat`) propagate the
   permutation to their output edge. This is inherently **shared/fan-out safe** — different
   consumers view the same shared batch with different permutations; no move, no copy, no pin.
2. **Mixed projection + column-dropping scan → producer-owned batch.** Mixed: move evaluated
   columns + copy passthrough columns into one owned table. Scan: move-reassemble the selected
   columns out of its own uniquely-owned output (zero-copy), freeing the filter-only columns now.

**Enforcement / lock-in:** (a) debug assert `idle` in `add_data_batch`; (b) change
`owning_table_view::owner` from `std::any` → `std::vector<std::shared_ptr<cudf::column>>` (makes a
`read_only_data_batch` owner a **compile error**; the pinned scan already passes this exact type).

### Case handling
| Case | Handling | Copy? | Pin? |
|---|---|---|---|
| Passthrough projection (reorder/subset/dup) | Removed → edge carries permutation; consumer `select`s a view at read | no | no |
| Fan-out (one output → N consumers) | One shared owned source; each consumer edge has its own permutation; views+selects independent | no | no |
| Mixed projection (evaluated + passthrough) | Owned batch: evaluated moved, passthrough copied | passthrough only | no |
| All-evaluated projection | Already owned (`make_data_batch`) — unchanged | (computed) | no |
| Column-dropping scan | Move-reassemble owned batch; filter-only columns freed now | no | no |
| Multi-GPU join build / dynamic-filter build | Owned batches read concurrently under `read_only` — never view producers, unaffected | no | no |
| Pinned-table scan (`scan_manager`) | Unchanged: shares `shared_ptr<cudf::column>`, no lock | no | no |
| Downgrade under memory pressure | Every parked batch is idle + singly-owned → source enumerable & spillable; one downgrade frees; `add_data_batch` assert guards | — | — |

---

## Implementation checklist (phased; keeps `dev` green at each phase)

### Phase 0 — Safety net
- [ ] Port the reproducer test from the worktree: `test/cpp/data/test_view_backed_batch_pins_source_issue_1063.cpp` + register in `CMakeLists.txt` `TEST_SOURCES`.

### Phase 1 — Scan drop → owned (self-contained; kills pin #3) — `table_scan.cpp:248`
*Why:* the scan uniquely owns its fresh `output_batch`; move selected columns into a new owned
table, freeing filter-only columns now. Zero-copy, no pin.
```cpp
auto* space = /* capture before locking */;
std::unique_ptr<cudf::table> owned;
{
  auto mut     = output_batch->to_mutable();               // sole owner, idle
  auto& gpu    = mut.get_data()->cast<cucascade::gpu_table_representation>();
  auto table   = gpu.release_table(stream);
  auto columns = table->release();                         // vector<unique_ptr<column>>
  std::vector<std::unique_ptr<cudf::column>> out;
  for (std::size_t i = 0; i < expected_output_columns; ++i)
    out.push_back(std::move(columns[static_cast<std::size_t>(*batch_column_map[projection_ids[i]])]));
  owned = std::make_unique<cudf::table>(std::move(out));    // unselected cols freed at scope end
}
output_batch = sirius::make_data_batch(std::move(owned), *space, stream, batch_telemetry());
```
- [ ] (No duplicate-column guard needed — see below. `projection_ids` at the scan is a
  prune+reorder without duplicates, because DuckDB expresses column *duplication* via a separate
  PROJECTION operator, not via the scan. Optionally add `assert(no duplicate values in projection_ids)`
  to enforce that assumption loudly if DuckDB ever changes.)

### Phase 2 — Mixed projection → owned (self-contained; kills pin #2) — `projection.cpp:178`
*Why:* one output can't be assembled from two owned sources without a copy; input is only held
`read_only` (and may be shared) so it can't be moved. Copy is correct + shared-safe; mixed is rare.
```cpp
auto eval_cols = evaluator->evaluate(input_view)->release();
std::vector<std::unique_ptr<cudf::column>> out(column_plan.size());
for (std::size_t i = 0; i < column_plan.size(); ++i) {
  const auto& p = column_plan[i];
  out[i] = (p.kind == projection_source::passthrough)
             ? std::make_unique<cudf::column>(input_view.column(p.index), stream, mr) // copy
             : std::move(eval_cols[p.index]);                                          // move
}
output_batches.push_back(sirius::make_data_batch(
  std::make_unique<cudf::table>(std::move(out)), mem, stream, batch_telemetry()));
```

### Phase 3 — Single input-view choke-point (infra)
- [ ] Consolidate `pipelineable_operator_data` (`sirius_physical_operator.hpp:319-320`) onto one
  `input_entry{ shared_ptr<data_batch> batch; optional<read_only_data_batch> lock; optional<vector<cudf::size_type>> projection; }`
  store; rewrite `get_data_batches` (`operator.cpp:41`), `get_read_only_batches` (`:58`),
  `prepare_for_processing` (`:83`) over it. *Why:* fuse batch+lock+permutation so a permutation
  can't be mis-indexed; the lock is relocated (still load-bearing), not removed.
- [ ] Add the sole accessor:
```cpp
cudf::table_view pipelineable_operator_data::get_input_table_view(std::size_t i) const {
  const auto& e = _inputs.at(i);
  auto full = sirius::get_cudf_table_view(*e.lock);        // lock held since prepare
  return e.projection ? full.select(*e.projection) : full; // zero-copy select
}
```

### Phase 4 — Pure passthrough → edge permutation (kills pin #1) — `projection.cpp:137`
- [ ] Add `std::optional<std::vector<cudf::size_type>> projection;` to `port`
  (`sirius_physical_operator.hpp:604`) and `pending_output_projection` near `next_port_after_sink` (`:693`).
- [ ] In `create_plan(LogicalProjection&)` (`sirius_plan_projection.cpp:52`), after the
  identity-omit block (`:76`): if all outputs are `BOUND_REF` but not identity, stash
  `perm[i]=bound_ref.index` on the child and `return plan` (no operator).
- [ ] In `materialize_repository_wiring` (`repository_wiring_materializer.cpp:74`), before
  `add_port`: `if (source_op->pending_output_projection) new_port->projection = std::move(...)`.
- [ ] Zip the port's `projection` into each popped batch in `get_next_task_input_data`
  (`operator.cpp:331`).
- [ ] Delete Path 2 at `projection.cpp:137`.

### Phase 5 — Consumer migration + forwarders (mechanical breadth)
- [ ] Route **every** consumer's input read through `get_input_table_view(i)` (filter, limit,
  top_n, sort_partition, concat, order, partition, merge, aggregates, both joins, result_collector).
- [ ] Make the raw path private/removed → bypass is a **compile error** (this is what makes it verifiable).
- [ ] Propagate the permutation across batch-forwarders (`dynamic_filter` probe, `cte`, `concat`).

### Phase 6 — Enforcement (do the type change LAST)
- [ ] Debug assert `batch_state::idle` in `data_repository::add_data_batch`
  (`cucascade/include/cucascade/data/data_repository.hpp:75`) or Sirius `push_data_batch`
  (`operator.cpp:274`).
- [ ] Change `owning_table_view::owner` `std::any` → `std::vector<std::shared_ptr<cudf::column>>`
  (`cucascade/include/cucascade/cudf/gpu_data_representation.hpp:201-202`); drop `template<typename Owner>`
  on the view ctor and on `make_data_batch_from_view` (`data_batch_utils.hpp:175`). Only after
  Phases 1/2/4, or it won't build. Pinned scan (`scan_manager.cpp:195,217`) already passes this type.
- [ ] Grep-assert `make_data_batch_from_view` has only the pinned-scan caller left.

### Phase 7 — Verify
- [ ] Correctness: reorder / subset / duplicate projection; mixed projection; scan with filter-only
  columns; fan-out (one output → two consumers, different projections).
- [ ] Regression: #1063 repro completes under a tight-GPU config; TPC-H validation passes;
  idle-assert never trips (confirms multi-GPU-join / dynamic-filter shared-reader paths untouched).
- [ ] Update `docs/super-sirius/data-management.md`: passthrough projections produce no batch;
  `owning_table_view` is buffer-sharing only.

**Suggested PR split:** (A) Phases 0–2 + idle-assert; (B) Phases 3–5; (C) Phase 6 type lock-in.

---

## Key code sites (navigate by symbol; line numbers drift)
- `src/op/sirius_physical_projection.cpp:137,178` — pinning producers (passthrough, mixed)
- `src/op/sirius_physical_table_scan.cpp:248` — pinning producer (scan drop)
- `src/scan_manager/sirius_scan_manager.cpp:195,217` — pinned scan (KEEP; shared columns)
- `src/include/data/data_batch_utils.hpp:175` — `make_data_batch_from_view`
- `cucascade/include/cucascade/cudf/gpu_data_representation.hpp:201-207` — `owning_table_view` (`std::any owner`), the `variant`
- `src/include/data/convertible_data_batch.hpp` — downgrade candidate filter (idle/subscribed skip)
- `src/downgrade/downgrade_executor.cpp` — two-tier scan; "no viable downgrade target" stall log
- `src/planner/sirius_plan_projection.cpp:52,67-76,82` — projection elision / create
- `src/include/op/sirius_physical_operator.hpp:266,319-320,604,693` — operator_data + port
- `src/op/sirius_physical_operator.cpp:41,58,83,274,331` — operator_data methods, sink/push, task input
- `src/pipeline/repository_wiring_materializer.cpp:74,76` — edge wiring
- `cucascade/include/cucascade/data/data_repository.hpp:75` — `add_data_batch`

## Reproducer
- `/home/dvats/repos/sirius-repro-1063/test/cpp/data/test_view_backed_batch_pins_source_issue_1063.cpp`
  — deterministic C++ characterization test (invariant violation + blocking-downgrade deadlock).
- `/home/dvats/repos/sirius-repro-1063/test/repro/issue_1063/` — `sirius_1063.yaml` (tight-GPU config),
  `repro_1063.sql`, `README.md` (end-to-end recipe).
- No GPU/CUDA on the dev box → reproducer is unbuilt here; run on a GPU host.

## Maintainer Q (wmalpica) — pinned-scan answer (drafted, not posted)
The pinned-GPU scan is exactly why `owning_table_view` exists and is KEPT — it is not what #1063
removes. Distinguish the two owner kinds: `vector<shared_ptr<cudf::column>>` (pinned scan; shares
buffers; no batch pinned) vs `read_only_data_batch` (projection/filter-scan; locks a source batch's
state; the bug). The pinned table isn't one `shared_ptr<data_batch>` reused as the scan output; it's
persistent shared *columns*, and each scan split is a separate view batch co-owning the needed
columns by refcount. The `size_type` edge-permutation is for a *flowing* batch, not the pinned
keep-alive. Each scan-output view batch is itself idle/spillable (downgrading it copies to host and
drops the refcount; the pinned originals persist).
