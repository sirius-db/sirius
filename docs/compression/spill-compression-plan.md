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

**Coverage is now good.** Measured after fixing the two failures below:

| Query | Columns at spilling edges | Resolved |
| --- | --- | --- |
| `lineitem JOIN orders`, 3 projected columns | 15 | 15 (100%) |
| q21 | 64 | 59 (92%) |

Two bugs had to be fixed to get there, both worth recording because neither was
where the symptom pointed:

1. **The GPU scan dropped the lineage.** `insert_gpu_pipeline_operators` replaces the
   table scan with a `sirius_gpu_scan_operator` built from scratch, which did not copy
   `column_origins`. Since the scan is a *leaf*, `propagate_column_origins()` had
   nothing to inherit from — so the loss at the leaf silently emptied the entire plan.
   The symptom looked like propagation failing; the cause was one missing assignment
   at the point where every origin enters the tree.

2. **Expressions arrive already binding-resolved.** The plan Sirius receives has been
   through `ColumnBindingResolver` *upstream*, so a pass-through projection column is a
   positional `BOUND_REF` into the child's output, not a named `BOUND_COLUMN_REF`.
   Moving our walk earlier in `create_plan` (as an earlier revision of this document
   advised) does not help and was based on a wrong assumption. `resolve_expression`
   now handles both forms, mapping a position back through the child's bindings.
   `LogicalGet` had masked this: it reads `column_ids` directly and never looks at an
   expression, so scans resolved while everything above them missed.

The remaining ~8% on q21 are genuinely computed columns — aggregate results and
expressions — which have no single base column by definition.

Only once coverage is real does the rest follow: seed `plan_register` from the origins
(and decouple `input_plan_dir` loading from `enable_pin_table_compression`, which
currently gates it), then stagger exploration so at most one column is explored per
spill.

### MEASURED: lineage seeding removes exploration, but compression itself OOMs

With plans seeded from lineage (q21, SF100, same configs as above):

| | Wall (on) | vs off | explores | spills compressed | declines |
| --- | --- | --- | --- | --- | --- |
| in-query explorer, tuned | 36.4 s | 2.3x | 3 | 18 | 12 |
| lineage-seeded | 31.3 s | 2.4x (off fell to 12.9 s) | **0** | 0 | 20 |

Exploration is gone — the thing that was 81% of query time no longer runs at all, and
seeding resolves cleanly where lineage reaches (`seeded 2/2 columns`). Two fixes were
needed to get there beyond the lineage work itself:

- `input_plan_dir` was read **lazily inside `pin_table()`'s bind**, one table at a
  time. A query that never pinned anything never loaded any plan, so the spill path —
  which reaches them through lineage, not through pinning — found nothing. Plans are
  now loaded once at `SiriusContext::initialize()`. Decoupling the load from
  `enable_pin_table_compression` was necessary but not sufficient; the real problem
  was *where* the load lived.

But the wall clock barely moved, because the cost simply relocated: every one of the
20 declines is now `std::bad_alloc: out_of_memory` from `compress_column` itself.

**This is the architectural problem, now isolated.** Compression on the spill path
needs to allocate GPU memory at exactly the moment the GPU has none — that is what
triggered the spill. Exploration made it far worse (hundreds of trial encodes) and
removing it was worth doing, but the residual is inherent to compressing *on the
device being evacuated*. Tuning cannot fix it.

> **Scope narrowed later.** This holds for the `std::bad_alloc: out_of_memory`
> declines measured here, which are genuine allocation failures. It was subsequently
> over-applied to the `launch_encode_fused_tree failed` signature, which is a CUDA
> *handle* error and had an unrelated one-line cause — see "the device tier's
> `launch_encode_fused_tree` failure was a missing `cudaSetDevice`" below.

Plausible directions, roughly in order of appeal:

1. Reserve a small device arena for spill compression up front, outside the pool the
   query allocates from, so a spill always has room to work in.
2. Compress in chunks sized to fit whatever headroom exists, rather than whole columns.
3. Stage to host uncompressed first and compress there (moves the cost off the GPU
   entirely, at the price of host bandwidth and CPU codecs).
4. Accept it: spill compression only pays when the GPU is *near* full rather than
   completely full, which argues for triggering the downgrade earlier.

A profile of the current state (0 explores, 20 failed compressions) is the next
diagnostic step — the ~18 s of remaining overhead is not yet attributed.

### MEASURED: full TPC-H sweep with a properly configured GPU

The earlier numbers were distorted by a bad benchmark config of my own making
(`usage_limit_fraction: 0.5`, no disk tier, and Sirius's default
`downgrade_trigger_fraction: 1.0` — i.e. spill only once the GPU is *completely*
full, which is exactly why compression had no memory to allocate in). Re-run against
the real working config from `~/.sirius/sirius.yaml`: 0.95 device usage, trigger at
0.6 so ~35% stays free, disk tier configured. Both arms identical except the spill
compression block.

All 22 queries, SF100, 1 iteration:

| | off | on | ratio |
| --- | --- | --- | --- |
| **sum of per-query time** | **34.97 s** | **40.15 s** | **1.15x** |
| q21 | 4.65 s | 10.22 s | **2.20x** |
| other 21 queries | — | — | 0.79x – 1.06x |

**The 1.15x is entirely q21.** Every other query lands within noise of parity; several
are marginally faster with compression on, which is measurement scatter, not a win.

Whole-sweep spill activity — the reason for this:

| | count |
| --- | --- |
| seeded from lineage | 1 |
| explored | 2 |
| spills compressed | 3 |
| declined (fallback) | 3 |
| **OOM declines** | **0** |

Two conclusions:

1. **The OOM problem is solved by configuration, not code.** Zero allocation failures
   across the whole sweep, against 20 in one query before. Triggering the downgrade at
   0.6 rather than 1.0 leaves the compressor room to work. The earlier conclusion that
   this was "inherent to compressing on the device being evacuated" was wrong — it was
   inherent to spilling only when the device is *already* full.

2. **At SF100 on a 12 GB card with 0.95 usage, almost nothing spills.** Six spill
   events across 22 queries. Spill compression is therefore close to a no-op for this
   workload, and q21's 2.2x comes from roughly two explorations costing ~1–2 s each on
   a 4.65 s query. The feature needs a workload that genuinely spills before its value
   can be judged — a smaller GPU budget, a larger scale factor, or a
   deliberately memory-starved config.

Lineage seeding fired only **once** in the sweep, so it is not yet carrying the load
it was built for. Worth checking whether the edges that actually spill are the ones
without lineage (aggregate/computed outputs), which would explain both the low seed
count and why q21 still explores.

### MEASURED: at 30% GPU budget, the cost is per-edge and hits short queries

The downgrade thresholds are **relative to the usage limit, not the device**:
`usage_limit_fraction` feeds `_gpu_capacity` → `memory_capacity`, and
`downgrade_trigger/stop_fraction` multiply that. So they scale automatically and need
no adjustment when the budget changes — confirmed empirically: dropping
`usage_limit_fraction` from 0.95 to 0.30 (11.4 GB → 3.6 GB budget on a 12 GB card)
took spill activity from 6 events to 322 with the fractions untouched.

All 22 queries, SF100, `usage_limit_fraction: 0.30`:

| | off | on | ratio |
| --- | --- | --- | --- |
| **sum** | **115.81 s** | **135.67 s** | **1.17x** |
| q21 | 82.12 s | 80.79 s | **0.98x** |
| q22 | 0.48 s | 8.70 s | **18.02x** |
| q10 | 1.81 s | 9.05 s | 4.99x |
| q4 | 0.77 s | 3.44 s | 4.50x |
| q11 | 0.78 s | 3.47 s | 4.47x |
| other 17 | — | — | 0.92x – 1.06x |

Spill activity: 37 seeded from lineage, 13 explored, 204 compressed, 81 declined,
**3 OOM declines**.

Three things this settles:

1. **The overhead is per-edge and roughly constant, not proportional to query time.**
   q21 spills heavily for 80 s and compression is *free* there (0.98x — it pays for
   itself). q22 runs in 0.48 s and pays 8.2 s of setup. The absolute cost of standing
   up compression for an edge — explore or seed, then compress the first batches —
   is what hurts, so it is catastrophic on short queries and neutral on long ones.
   This argues for gating compression on expected spill volume per edge rather than
   enabling it globally: an edge that will spill a handful of batches should never
   pay to set it up.

2. **There is a floor on the headroom compression needs.** OOM declines reappeared (3)
   once the free margin fell from ~4.6 GB to ~1.44 GB. The fractions scale, but the
   *absolute* room left at the trigger point is what compression actually requires, and
   a percentage cannot express that. A minimum-bytes floor alongside the fraction would.

3. **Lineage seeding is now doing real work** — 37 seeds against 13 explores, against
   1 seed in the 0.95 run. It engages once there is enough spilling for it to matter.

The remaining question is whether the 204 compressed spills bought anything: this
measures time, not bytes saved or host-memory pressure relieved. A run that reports
spilled-bytes-before/after is needed before concluding the feature is or is not worth
its cost.

### MEASURED: defer exploration, default to bitpack — spill compression is ~free

The 30% sweep showed the overhead is a fixed *per-edge setup* cost, so an edge's
first spill no longer explores. It installs plans immediately — seeded from the base
table's offline plan where lineage reaches, and a fixed default everywhere else —
and defers exploration to the edge's first expiry on the normal
`spill_replan_after_uses` schedule, by which point it has spilled enough to amortize
a beam search.

Choosing that default mattered more than expected. All 22 queries, SF100,
`usage_limit_fraction: 0.30`:

| default | overall | q21 | short queries | explored |
| --- | --- | --- | --- | --- |
| explore on first spill | 1.17x | 0.98x | q22 **18.0x**, q10 5.0x, q4/q11 4.5x | 13 |
| `bitcomp` | 2.69x | **8.57x** | ~1.0x | 0 |
| **`bitpack`** | **0.95x** | **0.92x** | ~1.0x | 0 |
| `delta -> bitpack` | 1.07x | 0.98x | q10 4.5x, q7 2.7x | 2 |

**A default is applied to every un-seeded column on every spilling edge, so its
speed matters more than its ratio.** `bitcomp` compresses well but is an entropy
coder: on q21 — the one query that spills heavily — it cost 8.6x, because the
explored plans it displaced were cheap bitpack/delta cascades. `bitpack` gets both
halves right, and lands slightly *ahead* of no compression at all.

`delta -> bitpack` was expected to win on the monotonic key columns that dominate
TPC-H spill traffic (`l_orderkey`, `o_orderkey`). It does not: partitioning has
already narrowed those columns by the time they spill, so the extra pass costs more
than the width it recovers. Its q10/q7 regressions are a separate effect — the only
run since deferral where exploration fired at all (2 edges hit their 128-use expiry),
which is the same fixed-cost problem on a ~2 s query.

STRING and nested columns are stored raw. A `str_split -> {bitcomp chars, delta ->
rle -> bitpack offsets}` cascade was written and reverted: its cost profile on real
data is unmeasured, and a blind default that is wrong is expensive on exactly the
heavily-spilling queries this is meant to help.

**These margins are not solid.** Two caveats:

- `run_spill_sweep.sh` runs the off arm first and the on arm second, so the off arm
  reads colder and the bias flatters compression. The off arm swung 43.72 s →
  118.84 s across runs on identical config — larger than the 5% margin separating
  bitpack from parity. Fixing the harness to interleave or pre-warm is the highest-
  value next change before any of these numbers are trusted.
- One iteration, one workload, one GPU budget. bitpack vs `delta -> bitpack` differ
  by 12%, which is not comfortably outside that noise.

Treat "plain bitpack, and delta does not help" as a working conclusion, not a settled
one. What is solid is the shape: deferring exploration removes the short-query
regressions outright (q22 18.0x → 1.0x), and a cheap default keeps the
heavily-spilling case fast.

### MEASURED: 3-way — spill compression is a no-op, output compression is a small win

All 22 queries, SF100, `usage_limit_fraction: 0.30`, 3 iterations x 2 repeats,
three arms identical but for the compression settings
(`test/tpch_performance/threeway_{none,spill,output}.yaml`):

| scope | none | spill | output |
| --- | ---: | ---: | ---: |
| all 22 queries | 42.45 s | 41.46 s (**0.98x**) | 39.77 s (**0.94x**) |
| excl. q21 | 33.12 s | 32.88 s (0.99x) | 30.58 s (**0.92x**) |
| excl. q21 + CPU-fallback queries | 26.81 s | 26.52 s (0.99x) | 24.22 s (**0.90x**) |

Results are byte-identical across all three arms (1359 result rows each).

Signals that clear within-arm noise:

- **spill**: q14 at 0.96x, and nothing else. Neutral — no measurable cost *or*
  benefit at this budget.
- **output**: q3 4.29 -> 1.15 (**0.27x**), q9 3.81 -> 4.38 (1.15x).

Output compression compressed 117 batches, 16.07 GiB -> 5.76 GiB (2.79x),
**saving 10.31 GiB**.

**Two earlier conclusions in this document were artefacts and are withdrawn.**

1. *"Spill compression costs 1.15x-1.17x, concentrated in q21/q22."* Measured
   here at 0.98x. The earlier runs used `run_spill_sweep.sh`, which always ran the
   off arm first — the bias this document already flagged as "the highest-value
   next change before any of these numbers are trusted". Rotating arm order per
   repeat removes it.
2. *"Output compression turns q21's intermittent livelock into a permanent one"*
   (0-for-12 across two sweeps). It does not. With arm order rotated, every arm
   succeeds sometimes and at indistinguishable rates:

   | arm | q21 per-iteration (s) | succeeded |
   | --- | --- | --- |
   | none | 9.3, 84.5, 80.4, 82.0, 86.0, 9.8 | 2/6 |
   | spill | 85.0, 66.8, 79.9, 8.6, 89.8, 89.7 | 1/6 |
   | output | 85.5, 84.7, 86.6, 9.2, 82.4, 88.0 | 1/6 |

   q21 livelocks intermittently in the **baseline** (see the livelock note below);
   compression neither causes nor worsens it.

The methodological lesson is the same one twice: at this budget five of 22 queries
carry no usable signal — q2/q17/q18/q20 fall back to DuckDB CPU in every arm
(a pre-existing `SiriusGeneratePhysicalPlan` failure), and q21 is a coin flip — so
any sweep that reports only a total will mostly be reporting those five.

### MEASURED: the output path's cost is per-batch, not per-byte

The first output-compression sweep regressed (1.16x excl. q21) while compressing
1861 batches averaging 13.4 MiB. The added time was +5.49 s, i.e. **~2.95 ms per
batch** — against ~30 us of expected codec work for 13.4 MiB at the offline plans'
recorded 250-800 GB/s. That is 1-2% of rated throughput: essentially all of the
cost is fixed per-batch overhead (a per-column, per-plan-node
`cudaStreamSynchronize` in simpatico's `compress_column`, needed because
variable-output codecs report their size from device memory, plus blob staging).

Gating on batch size (`output_compression_min_batch_bytes`, 64 MiB) took it from
1861 batches to 117 — **16x fewer batches, still 61% of the bytes saved** — and
turned the 1.16x regression into 0.92x.

This also rules out the obvious optimization: batches carry 2-4 columns, so
per-column parallelism is bounded at a 2-4x dent in a ~100x gap. Worse, it is not
free to attempt — `reservation_aware_resource_adaptor` keys its tracker state in a
`thread_local` map, so any allocation on a `stream_pool` worker is neither charged
to the task's reservation nor subject to its limit/OOM policy. (Note
`pin_table`'s existing use of `compress_columns_parallel` is untracked for this
reason; it is tolerable only because pinning runs outside a query reservation.)

### MEASURED: the device tier's `launch_encode_fused_tree` failure was a missing `cudaSetDevice`, not memory pressure

**A third conclusion in this document is withdrawn: that compressing on the device
being evacuated is architecturally blocked by memory pressure.** For the
`launch_encode_fused_tree failed` signature it is not. That failure was one missing
`cudaSetDevice` on one thread.

`downgrade_executor::start()` installs a `per_thread_init` calling `cudaSetDevice`,
but only on the worker pool — `_processing_thread` never got it, and the in-place
compression pass runs there, in `processing_loop()`. `cudaSetDevice` is what makes
the device's primary context current. Simpatico derives its JIT `CUfunction` lazily
on whichever thread first asks and caches it **by device id, not by context**
(`CompiledKernel::func_for_current_device`, `nvrtc_compiler.cpp:96`). When the
unbound processing thread won that race, `cuKernelGetFunction` returned a handle
`cuLaunchKernel` then rejected with `CUDA_ERROR_INVALID_HANDLE`.

q3/SF100, arm `device`, output compression off, before → after the one-line fix:

| | before | after |
| --- | ---: | ---: |
| batches compressed | 0 / 78 | **43 / 43** (7 passes) |
| declines | 75 | 0 |
| `cuLaunchKernel` failures | 75 | 0 |
| freed per pass | — | 40–241 MB |

The control that identified it: enabling task-output compression *alongside* the
device tier — so a task-executor thread populates the `CUfunction` cache first —
took the identical unfixed run from 0/78 to **76/76**. Same query, same config, same
process. The variable was cache-warm order, not free memory.

Two lessons worth keeping:

1. **The error was invisible by construction.** `launch_encode_fused_tree` discards
   its `CUresult` and reports only `"launch_encode_fused_tree failed"`; the actual
   `cuLaunchKernel failed: invalid resource handle` goes to **stderr via `fprintf`**,
   not the log. Two separate investigations read the generic message as OOM. Reading
   the swallowed status was the cheapest possible diagnostic and would have
   short-circuited both. Propagating it belongs upstream in simpatico.
2. **`CUDA_ERROR_INVALID_HANDLE` is not an allocation failure.** A launch-time
   handle error and `std::bad_alloc` are different problems; only the latter is about
   memory. The "lineage seeding … but compression itself OOMs" section above reports
   genuine `std::bad_alloc: out_of_memory` from `compress_column` — **that** finding
   stands and is untouched by this fix.

The peak-residency work (216 MB → 92 MB on q3/SF100, `U + 2C` → `2C`) was aimed at
the pressure this section now shows was not the blocker. It is still worth having —
it removed an unretryable throw after `release_table()` — but it did not and could
not fix the device tier.

Corroboration, not proof, for the spill path: the archived sweeps carry the same
signature on the spill arm only (`spill_sweep` 26, `threeway_sweep` 28,
`fourway_sweep` 40 occurrences), which runs on this same thread. A post-fix q3 spill
run shows 76 successful compressions and zero launch failures, but there is no
matched pre-fix single-query baseline.

Consequence for the record: **every previously reported `device`-arm timing measured
a feature that never fired**, which is exactly why the 4-way sweep found that arm
"behaviourally identical to `normal`". Those numbers say nothing about the tier and
should not be cited. A re-measurement is now meaningful for the first time.

### Spill state is per-query and is cleared at query end

`_spill_plans` and `_spill_origins` are keyed by `shared_data_repository*`, and
`SiriusContext::QueryEnd` destroys every repository. Without clearing, those maps
grew without bound holding entries keyed by freed pointers, and a repository later
allocated at a recycled address would inherit plans and verdicts belonging to an
unrelated edge. `clear_spill_state()` now runs at query end, just before
`clear_all_repositories()`.

Nothing is carried across queries by design. The offline table plans
(`input_plan_dir`) survive — they come from startup, not from a query — so the next
query re-seeds through column lineage from the same source. Explored plans are
**not** written back to the per-column store: an exploration is evidence about one
spilling edge's data, not about the base column in general, and promoting it would
let one query's intermediate distribution silently redirect every later query's
plans for that column.

The consequence is worth stating plainly: **`spill_replan_after_uses` counts uses
within a single query.** Few edges reach 128, so exploration rarely fires and the
adaptive machinery around it (backoff, the 20% adoption threshold, explore-failure
memoization) is mostly dormant in practice. What actually carries the work is
lineage seeding plus the fixed default. That is a deliberate trade — bounded,
predictable per-query behaviour over cross-query learning — but it means the
replan tuning knobs matter far less than their presence suggests.

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


### Re-Pareto the offline plans against compression throughput

The plans in `plans/tpch_sf1000/` were picked, per their own header, for
*"max ratio with decompress >= 250 GB/s per column"*. Compression throughput was
never a selection criterion, and it shows: of the 27 columns with ratio > 3x, **13
fail a 300 GB/s compress gate while zero fail the decompress gate**. Median compress
throughput is 181 GB/s for STRING columns against 338 GB/s for numerics.

That asymmetry is now load-bearing. Any consumer that has to compress *during* a
query — the spill path, and the task-output path below — pays compression cost on
the critical path and reads back at most once, so compress speed matters at least as
much as decompress. The current plans cannot answer whether a column has a fast
plan, only whether it has a fast-to-*read* one; a column may have been assigned a
slow-to-write cascade when a slightly worse ratio at 3x the write speed existed and
was discarded.

Regenerate with compression throughput as a first-class axis (a Pareto front over
all three of ratio / comp / decomp, rather than a decompress threshold plus a ratio
argmax), and record the front rather than one pick per column, so a consumer can
choose by its own cost model instead of inheriting the offline one.

Two things have changed since the current plans were generated and should land in the
same regeneration:

- **Dictionary encode is believed to be faster now.** Every STRING column that misses
  the gate misses it on the dictionary/`str_split` build. Several are close enough
  (`p_mfgr` 281, `p_type` 265 GB/s) that a rebuild may move them over on its own — and
  the ones that are far off (`l_returnflag` 98, `l_linestatus` 90) are exactly the
  low-cardinality columns dictionary should be best at. Re-measure before concluding
  strings are structurally unsuitable.
- **FSST for STRING columns.** There is an FSST operator on `joost/fsst-operator`
  (unmerged). FSST is designed for exactly this shape — short repetitive strings with
  symbol-table reuse — and would plausibly beat the dictionary cascades on both ratio
  and write speed. Wiring it into the plan space is a prerequisite for the string
  columns being usable by any compress-on-the-critical-path consumer.

Until this is done, treat the recorded `comp` numbers as a floor rather than a
measurement of what the codec can do.

### Compress task output columns with a fast, high-ratio lineage plan

Compress selected columns of a task's output when it finishes, rather than only when
memory pressure forces a spill. The lineage machinery already resolves an output
column back to a base-table column, and the offline plans already carry measured
ratio and throughputs — so for a column whose plan is both *fast* and *high-ratio*,
compressing eagerly buys a smaller resident footprint at a cost the measurements say
is small.

**Gate: ratio > 3x, compress > 250 GB/s, decompress > 250 GB/s.** Against the current
SF1000 plans this admits 13 of 53 columns:

| table | column | ratio | comp | decomp | plan | order-dependent |
| --- | --- | ---: | ---: | ---: | --- | --- |
| orders | o_shippriority | 455x | 2006 | 3537 | bitpack | |
| part | p_partkey | 254x | 301 | 504 | zigzag → delta | **yes** |
| orders | o_orderkey | 106x | 338 | 741 | delta → lz4 | **yes** |
| partsupp | ps_partkey | 101x | 251 | 715 | delta → … | **yes** |
| part | p_mfgr | 46.9x | 281 | 579 | dictionary | |
| part | p_type | 24.4x | 265 | 339 | dictionary | |
| lineitem | l_orderkey | 20.7x | 531 | 627 | delta → ans | **yes** |
| lineitem | l_linenumber | 10.4x | 726 | 2123 | bitpack | |
| lineitem | l_tax | 8.8x | 598 | 691 | ans | |
| lineitem | l_discount | 8.3x | 594 | 686 | ans | |
| customer | c_nationkey | 6.3x | 783 | 1772 | bitpack | |
| supplier | s_nationkey | 6.3x | 307 | 787 | bitpack | |
| part | p_size | 5.3x | 844 | 1756 | bitpack | |

At 300 GB/s the set is 10 columns and contains **no** STRING at all; at 200 it is 18.
250 is the knee — it keeps the two string columns that are within reach of a
dictionary rebuild without admitting the sub-100 GB/s encoders.

**The high-ratio plans are the least trustworthy ones.** Four of the thirteen are
delta cascades, and their ratios come from the base table being *stored sorted* by
that key — `l_orderkey` at 20.7x, `o_orderkey` at 106x, `p_partkey` at 254x. A task
output has been through joins, partitioning and hash shuffles, so that ordering is
gone and delta collapses toward no compression while still costing a full pass. The
gate as specified therefore selects hardest for the plans most likely to disappoint.

So a base-table plan is a *hypothesis about* the output column, not a measurement of
it. Delta plans are admitted, then **verified on first use**: compress the first
output batch, compare the achieved ratio against the gate, and drop the plan for that
edge if it misses. That costs one wasted pass on a bad edge and nothing on a good
one, and it is the only way to tell a still-sorted output from a shuffled one without
inspecting the data first. The per-column `viable` flag and `conclude_spill_attempt`
already implement exactly this measure-and-write-off loop; this reuses it rather than
adding a parallel mechanism.

The order-robust plans need no such check: `ans` is a pure entropy coder over the
value distribution, and `bitpack` on a low-cardinality column (`nationkey`, `size`,
`shippriority`, `linenumber`) keeps a narrow `chunk_min`/`chunk_bits` regardless of
row order.

Open questions before building:

- **Which task outputs are eligible at all?** Lineage only resolves columns tracing
  back to a scan; aggregate results and computed expressions have no origin and would
  never seed. On q21 that was ~8% of columns, but a task-output feature may skew
  toward exactly those computed columns.
- **What is the actual win?** This trades GPU compute for resident footprint. It pays
  only if the compressed output is held long enough, or is large enough, for the
  footprint to matter — which argues for gating on output size as well as on plan
  quality, in the same spirit as the per-edge setup cost measured above.
- **Interaction with the spill path.** An already-compressed output that later spills
  should flush rather than re-compress, the way `flush_host_to_disk` already does.

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
