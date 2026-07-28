# Spill Compression Plan

Compress GPU operator output on the downgrade path (GPU→HOST and GPU→DISK) using
Simpatico, so spilled batches have a smaller footprint in host/disk memory. Decompression
is automatic on the next `lock_or_prepare_batch` call via the existing converter registry.

## Context

The `compression-subsystem` branch established the representation hierarchy
(`compressed_host_representation`, `compressed_device_representation`) and the
input-table pin path. This work extends it to the **spill path**.

Prior art on `compress-spills` (branched ~2026-06-29, never merged) did the same thing
but with a file-backed host tier (writing to `/dev/shm`). The architecture has since
changed: the host tier is now memory-backed (`pinned_compressed_blob`), so the converters
and disk-cascade differ significantly from that branch.

## Design decisions

### Query-graph location key

The plan register needs a per-edge-in-the-graph key so each operator output gets its own
Simpatico plan. The natural key is the `shared_data_repository*`:

- Each repo is created once at wiring time, keyed by `(operator_id, port_id)`.
- All batches in a repo share schema and data distribution.
- The pointer is stable for the query duration.
- It is already available in `convertible_data_batch_provider` (the `_repo` member).

Thread the repo pointer into `convertible_data_batch` at construction, then pass it to
the plan register in `convert()`.

### Per-column plans and verdicts

Compressibility is a property of a column, not of a batch: a wide operator output
routinely mixes columns that shrink 10x with ones that do not compress at all. The
explorer already works one column at a time, so its per-column results are stored
per column rather than flattened into one `---`-joined plan.

Compression therefore runs column by column (`simpatico::compress_column`, the same
loop `compress_with_plan` performs internally), assembling the `compressed_table`
directly. Each column is measured against its *own* original bytes, so:

- one incompressible column no longer disqualifies its well-compressing neighbours;
- a column that does not pay is stored with a passthrough plan (`input -> identity`)
  on later batches instead of being re-compressed and discarded every time.

`identity` is safe for every dtype — on STRING it decomposes via `str_split` and
round-trips through both the in-memory and the file path.

A cached entry whose column count does not match the batch describes a different
schema, so it is discarded and the edge explored afresh.

### Per-edge plan lifecycle

The schedule stays per edge — batches arrive per edge, so all its columns are
re-explored together. Each entry carries, alongside its per-column plans:

- **`viable`** (per column) — cleared when that column misses
  `max_compressed_fraction`. Later batches store it raw instead of paying to
  compress and discard it every time. When *no* column is viable the edge is
  skipped outright, with no conversion attempted at all.
- **`uses`** — spill attempts since the entry was installed. Once it reaches
  `spill_replan_after_uses` the entry expires and the edge is explored afresh.

Expiry deliberately overrides *both* verdicts. It re-plans data whose distribution
has drifted, and it re-tests an edge written off as unviable — otherwise one
unrepresentative early batch could disable compression for that edge permanently.

`uses` is counted once per spill attempt including skipped ones, so a skipped edge
still ages toward its retry.

### Adopting a re-explored plan

The explorer is a beam search over a large space and readily returns a *differently
spelled* plan that performs identically. Adopting those churns the cache and — worse
— registers as a change, which resets the backoff below and locks the edge into
re-exploring for the rest of the query.

So a candidate is adopted only when its compression ratio or one of its throughputs
differs from the cached plan's by more than `spill_replan_change_threshold`
(default 20%, relative to the larger of the two values). The explorer already reports
all three metrics, so no extra measurement is needed.

Adoption is decided per column. An adopted column resets to viable with a clear error
streak; a column that keeps its cached plan keeps its verdict too — an equivalent plan
will not compress any better than the one already judged, so a written-off column is
not resurrected by a cosmetic re-explore.

### Adaptive replan backoff

Re-exploring costs a beam search per column, so the interval is only worth holding
at its configured value while re-exploring keeps paying off. Each entry carries its
own `replan_interval`, seeded from `spill_replan_after_uses` and adapted after every
re-explore cycle:

| Re-explore outcome | Interval |
| --- | --- |
| Same plan, still compressing | doubled |
| Same plan, still failing the threshold | doubled |
| Different plan, still failing the threshold | doubled |
| Different plan that compresses | reset to configured |
| Same plan, but viability recovered | reset to configured |

The rule is: reset only when the cycle produced a change that *actually compresses*;
otherwise back off. A stable good plan and a stubbornly incompressible edge both stop
paying for explores they learn nothing from, while an edge that is genuinely moving
stays on the frequent schedule. Doubling saturates rather than wrapping.

A hard compression failure (e.g. a plan that no longer fits the table) counts as a
failed attempt for both viability and backoff — otherwise every later batch would
repeat it and throw.

### On-first-spill explore

When `convert()` fires and no spill plan exists for the source repo:

1. Call `simpatico::explore_column_compression()` per column on the batch currently
   held under the mutable lock.
2. Join the per-column DSL strings with `"---"`.
3. Store the assembled plan in `plan_register` keyed by repo pointer.
4. Proceed immediately with `compress_with_plan()` using the new plan.

Explorer config on the spill path should use reduced `beam_width` / `max_explore_bytes`
(fast rough plan preferred over optimal slow plan under memory pressure). Make these
tunable via `sirius_config::compression_config`.

### Disk tier

The `pinned_compressed_blob` IS the `.hpln` file format split across pinned RAM blocks:
- `blob->header` = the binary header produced by `build_compressed_table_header()`
- `blob->payload` blocks (concatenated) = the compressed leaf buffer payload

**HOST→DISK** cascade: flush the blob to a file with standard file I/O (write header
bytes then walk payload blocks). No Simpatico API needed. Result is a valid `.hpln`
readable by `read_compressed_table()`.

**GPU→DISK** direct: use `simpatico::write_compressed_table(ct, path, stream)` since
there is no blob yet.

**DISK→GPU** decompression: `simpatico::read_compressed_table(path, stream, mr)` then
`simpatico::decompress()`.

## Status

**All work items (1–9) are implemented.** The full C++ suite passes (2178 cases);
the compression suite is 32 cases, of which 8 are the new spill tests.

Remaining before this is production-ready: the reservation-oversizing item under
"Future / deferred" below, and an end-to-end run under real memory pressure (the
unit tests drive `convert()` directly rather than going through the downgrade
executor).

One design point resolved during implementation: cuCascade's converter signature
(`source, target_space, stream, reservation`) cannot carry the repo pointer, so the
key is passed via a thread-local `spill_context` installed by
`convertible_data_batch::convert()` for the duration of the `convert_to<>` call —
see `src/compression/spill_context.hpp`. Doing the compression inline with
`set_data()` instead was rejected: it would bypass `convert_to`'s probe/telemetry
events and its sync-before-destroying-the-old-representation barrier.

## Work items

### 1. `plan_register` — spill plan storage (keyed by repo pointer)

Add to `plan_register.hpp / .cpp`:

```cpp
void set_spill_plan(const cucascade::shared_data_repository* repo, std::string plan_dsl);
void clear_spill_plan(const cucascade::shared_data_repository* repo);
[[nodiscard]] std::optional<std::string>
    resolve_spill_plan(const cucascade::shared_data_repository* repo) const;
```

New private map: `std::unordered_map<const cucascade::shared_data_repository*, std::string> _spill_plans`.
Extend `clear_all()` to also clear `_spill_plans`.

### 2. `convertible_data_batch` — carry source repo

Add `const cucascade::shared_data_repository* _source_repo{nullptr}` to
`convertible_data_batch`. Update `convertible_data_batch_provider::try_get_batch()` to
pass `_repo` into the `convertible_data_batch` constructor.

In `convert()`, pass `_source_repo` to the plan register lookup and to the explore call.

### 3. `sirius_config` — explorer knobs

Add to `compression_config`:

```cpp
bool enable_spill_compression{false};
uint32_t spill_explore_beam_width{20};
size_t   spill_explore_max_bytes{256ull << 20};  // 256 MiB cap per column
```

### 4. `compressed_disk_representation` (new file)

DISK-tier `idata_representation` backed by a `.hpln` file path. RAII: unlinks the file
when the last owner drops (shared ownership via `shared_ptr<std::string> _path` +
`shared_ptr<bool> _owns_file`).

Files:
- `src/compression/compressed_disk_representation.hpp`
- `src/compression/compressed_disk_representation.cpp`

### 5. `simpatico_bridge` (new file)

Thin helpers:
- `initialize_simpatico_jit()` — calls `codegen::jit::ensure_cuda_context()`, needed
  before first JIT operation (hook into extension load).
- `make_compressed_temp_path(dir)` — returns a unique `.hpln` temp file path for disk
  spills.

Files:
- `src/compression/simpatico_bridge.hpp`
- `src/compression/simpatico_bridge.cpp`

### 6. New converters in `compression_converters.cpp`

| Converter | Notes |
|---|---|
| `gpu_table_representation → compressed_host_representation` | Run explore if no plan; compress with plan; build `pinned_compressed_blob` via `build_compressed_table_header()` + D→H copies |
| `gpu_table_representation → compressed_disk_representation` | Run explore if no plan; `simpatico::write_compressed_table(ct, path, stream)` |
| `compressed_disk_representation → gpu_table_representation` | `read_compressed_table(path, stream, mr)` → project → `decompress()` |
| `compressed_host_representation → compressed_disk_representation` | Flush `pinned_compressed_blob` to file (write header bytes + walk payload blocks) |

### 7. `convertible_data_batch::convert()` — wire spill path

In the HOST tier branch, before falling through to `convert_to<host_data_representation>`:
- Check `plan_register::global().resolve_spill_plan(_source_repo)`.
- If `enable_spill_compression` is set, try `convert_to<compressed_host_representation>`
  inside a try/catch; on exception log and fall through to uncompressed.

Same pattern for DISK tier with `compressed_disk_representation`.

### 8. `sirius_extension.cpp` — startup + settings

- Call `sirius::compression::initialize_simpatico_jit()` at extension load.
- Register `SET spill_compression` (bool) and `SET spill_compression_plan` (VARCHAR for
  a per-session default DSL, optional override for the explore step).

### 9. Tests

Port and rewrite `test/cpp/compression/test_spill_compression.cpp` from `compress-spills`:
- GPU→compressed_host roundtrip (check blob size, decompress+compare).
- GPU→compressed_disk roundtrip (check file exists, decompress+compare).
- compressed_host→compressed_disk cascade (check flush, decompress+compare).
- No-plan / explore-fallback (first batch: explore fires and plan is stored).
- Column-count-mismatch fallback (explore produces wrong-width plan → uncompressed).

## Future / deferred

### MEASURED: exploration dominates, and spill compression is a large net loss

First benchmark, TPC-H q21 at SF100 on a 12 GB card with `usage_limit_fraction: 0.5`
(configs: `test/tpch_performance/spill_compression_{on,off}.yaml`, driver:
`run_spill_ab.sh`):

| Arm | Wall clock (2 iterations) |
| --- | --- |
| spill compression off | **14.7 s** |
| spill compression on | **68.3 s** (4.6x slower) |

nsys on the same query (1 iteration) attributes it unambiguously:

| NVTX range | Total | Instances | Share of query |
| --- | --- | --- | --- |
| `sirius::query` | 23.45 s | 1 | — |
| `compression::gpu_to_host_compress` | 18.0685 s | 14 | 77% |
| ↳ `compression::explore_spill_plan` | 18.0682 s | 14 | 77% |
| `simpatico::compress_column` | 4.87 s | 59,739 | (inside explore) |

**Exploration is 99.998% of the compress path** — of 18.0685 s spent in the converter,
actual compression is 0.35 ms. Average explore is 1.29 s per invocation, worst 3.34 s.
The 59,739 `compress_column` calls are the beam search's internal trials.

Two distinct problems, both needing a fix before this feature is worth enabling:

**1. The explorer needs GPU memory exactly when the GPU is full.** The run logged 2,430
`simpatico::codegen: cpp encode: exception: std::bad_alloc` — the beam search allocating
for trial encodes during a downgrade, i.e. while the GPU is by definition out of memory.
Of 42 spill attempts, 1 produced a compressed batch and 41 fell back. We paid full
exploration cost for almost no compression. Plausible directions: explore on a small
sampled copy taken before pressure, explore off the critical path (background, or at
first-touch rather than first-spill), reserve a scratch arena for exploration, or drop
the in-query explorer in favour of pre-generated per-schema plans.

**2. A failed exploration is not memoized, so every spill repeats it.**
`resolve_or_explore_spill_plan` throws when the explorer fails, and it throws *before*
`set_spill_plan` has created the register entry. `conclude_spill_attempt` then finds no
entry and returns early, so no error streak accumulates and the edge is never written
off — the `outcome_guard` is not even constructed yet. Every subsequent spill from that
edge re-runs the full beam search and fails again: 41 times in this query alone. The
existing memoization covers a failing *compression* but not a failing *exploration*,
which is the far more expensive case. Fix: record the attempt against the edge before
exploring, or install a placeholder entry so failures have somewhere to accumulate.

Until both are addressed, `spill_compression` should stay off by default (it is).

#### After fixing both (same query, same configs)

| | Wall (on) | vs off | encode OOMs | explores OK | spills compressed |
| --- | --- | --- | --- | --- | --- |
| original | 68.3 s | 4.6x | 2,430 | 0 | 1 |
| + memoize explore failures, `sample_rows` 65536 | 49.1 s | 3.2x | 54 | 0 | 0 |
| + `max_explore_bytes` 8 MiB, `beam_width` 8 | **36.4 s** | **2.3x** | **0** | **3** | **18** |

Compression now actually functions rather than just burning time. Two lessons:

**`sample_rows` alone made exploration *slower* per call** (1.29 s → 2.13 s), despite
cutting OOMs 45x. It only trims the beam *ranking*; finalists are still re-measured on
the full column. The OOMs had been acting as an accidental cost limiter — removing them
let more candidates survive to the expensive rerank. `max_explore_bytes` is the knob
that bounds both phases, and dropping it to 8 MiB is what eliminated the OOMs outright
and let exploration finally succeed.

**It is still a 2.3x regression.** What remains is the irreducible cost of running a
beam search inside a query on the downgrade thread — nsys still attributes ~81% of query
time to `explore_spill_plan`. No amount of tuning removes that; it needs the plan to
come from somewhere other than an in-query search. Hence lineage seeding below.

### Column lineage (in progress)

`src/planner/column_origin.{hpp,cpp}` resolves each operator output column back to
a base table column, so the spill compressor can reuse the plan already explored
offline for that column instead of searching for one mid-query.

DuckDB's `ColumnBinding`s already are the lineage graph. The resolver walks the plan
once, seeding at each `LogicalGet` and propagating through the operators that
introduce their own `table_index` (projections keep an origin for a bare column
reference, aggregates for their group keys). Everything else — filter, join, order,
partition — re-exposes its children's bindings unchanged.

Two ordering constraints, both learned the hard way:

- The walk must run **before** `ColumnBindingResolver`. That pass rewrites
  `BoundColumnRefExpression` into positional `BoundReferenceExpression`, erasing the
  bindings the walk follows.
- Operators inserted by the later rewrites (GPU pipeline wrappers, partitions,
  merges) never pass through the `create_plan` dispatcher, so they start with no
  lineage. `propagate_column_origins()` runs after `insert_gpu_pipeline_operators`
  and lets a pass-through operator inherit its child's origins, gated on matching
  output arity.

**Parquet scans are now named**, via `table_name_from_files()`: `read_parquet` is a
table function with no catalog entry, so the table identity comes from the directory
holding the files (`<root>/lineitem/part.0.parquet` → `lineitem`) — the same layout
`pin_table(name=…)` is given, and the key the pin plans use. Note the *file* stem is
useless and actively dangerous here: every file is `part.N.parquet`, and `part` is
itself a TPC-H table.

**Coverage is still far too low to build on.** Measured at SF100:

| Query | Columns at spilling edges | Resolved |
| --- | --- | --- |
| q21 | 47 | 2 (`lineitem.0`, `lineitem.2` — correct) |
| `lineitem JOIN orders`, 3 projected columns | 3 | 0 |

The two q21 hits confirm `record_get` and the name derivation work end to end. What
does not work is everything downstream: most edges report `cols=0`, meaning their
source operator has an *empty* origins vector, and the ones that do carry a vector
(e.g. the final 3-column projection above) have every entry nullopt.

So there are two separate failures still to diagnose, and both need doing before any
of this is worth consuming:

1. `cols=0` edges — `wiring.source_op` is a pipeline sink inserted by the rewrites.
   `propagate_column_origins()` is supposed to give it the child's origins but
   evidently is not firing. Suspect the arity gate (`child->column_origins.size() ==
   op.types.size()`) or that the sink's child is itself empty.
2. `cols=N resolved=0` edges — the operator went through the dispatcher but none of
   its bindings were in the map. Suspect a mismatch between the bindings recorded
   during `resolve()` (pre-`ColumnBindingResolver`) and those returned by
   `GetColumnBindings()` during `create_plan` (post-resolution).

Only once coverage is real does the rest follow: seed `plan_register` from the origins
(and decouple `input_plan_dir` loading from `enable_pin_table_compression`, which
currently gates it), then stagger exploration so at most one column is explored per
spill.

### Tune the explorer for use during query execution

The explorer's defaults were chosen for offline plan generation, where a long beam
search is amortized over every future query. On the spill path it runs *inside* a
query, under memory pressure, on the downgrade thread — a completely different cost
model. The current spill defaults (`beam_width` 20 vs the offline 100,
`max_explore_bytes` 256 MiB) are a guess, not a measurement.

What needs establishing, from a TPC-H sweep at a scale factor that forces spilling:

- **What does exploration actually cost?** Wall-clock per column, and as a share of
  total query time. `explore_spill_plan` is already an NVTX range, so nsys can
  attribute it directly.
- **How does beam width trade off?** Narrower is faster but finds worse plans; the
  right point is where the compression it gives up costs more spill bandwidth than
  the search saves.
- **Is `max_explore_bytes` doing useful work?** It trims large columns to a prefix.
  Too small and the plan is chosen from unrepresentative data (the explorer's own
  docs warn that prefix sampling misleads badly on sorted columns); too large and
  the search allocates heavily at the worst possible moment.
- **Does `sample_rows` help here?** Unused so far. It is the explorer's cheaper
  approximation, with the same caveat about sorted/monotonic columns.
- **Are the defaults for `spill_replan_after_uses` (128) and
  `spill_replan_change_threshold` (0.20) sane?** Both were reasoned about rather
  than measured. The replan interval only matters if exploration is expensive
  enough to be worth amortizing — which the same profile answers.

The end state is defaults backed by numbers, and a note in this document on which
workload shapes they suit.

### Test the config → converter-global plumbing

The spill tests call `set_spill_compression_settings()` directly, so they exercise
the converters while proving nothing about whether real configuration reaches them.
That gap hid two live bugs (YAML never pushed at init; `SET
compression_max_compressed_fraction` not propagated). Worth a test that sets the
DuckDB setting and asserts observable spill behaviour changes — best folded into the
end-to-end memory-pressure run rather than added as a narrow unit test.

### Reservation sizing on the compressed path

The reservation handed to the compress converter was sized for the *uncompressed*
batch (`convert()` reserves `data_size` before picking a target representation). The
compressed payload is smaller, so the reservation is safe but oversized — the host
budget is over-charged for every compressed spill until the reservation is resized
to the actual compressed footprint. Fixing this needs either a two-phase reserve
(compress, then reserve the real size) or a reservation-shrink API on cuCascade.


### Binary plan storage (avoid DSL roundtrip)

Currently `compress_with_plan()` accepts a DSL text string and re-parses it on every
call. The plan is internally a `PlanTree` (a flat node+edge structure). Storing the
`PlanTree` directly in the register and exposing a
`compress_with_plan_tree(table_view, PlanTree const&, ...)` API variant would eliminate
the parse step on every spill batch after the first.

This is worth doing once the spill path is proven: the DSL parse is cheap but not free,
and on a hot spill path it fires per-batch. The `PlanTree` is already returned by the
explorer as part of the internal beam-search result; surfacing it would require a small
API extension to `exploration_result` (add a `plan_tree` field) and a new
`compress_with_plan_tree` entry point in `simpatico_codegen.hpp`.
