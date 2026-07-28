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

### Per-edge plan lifecycle

Each edge's register entry carries the plan DSL plus two pieces of state:

- **`viable`** — cleared when a compressed batch misses `max_compressed_fraction`.
  Later batches from that edge then skip the compress attempt entirely instead of
  paying for it and discarding the result every time. Without this, a plan that
  fails the threshold stays cached and every subsequent batch repeats the full
  compress → reject cycle for the rest of the query.
- **`uses`** — spill attempts since the entry was installed. Once it reaches
  `spill_replan_after_uses` the entry expires and the edge is explored afresh.

Expiry deliberately overrides *both* verdicts. It re-plans data whose distribution
has drifted, and it re-tests an edge written off as unviable — otherwise one
unrepresentative early batch could disable compression for that edge permanently.

`uses` is counted once per spill attempt including skipped ones, so a skipped edge
still ages toward its retry.

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
