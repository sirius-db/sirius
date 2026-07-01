# Dynamic Filters — Multi-GPU Cross-Device Hazard

> **Status: KNOWN BUG, fix pending a decision.** Dynamic table-filter pushdown
> (Phase 1) is correct on a single GPU but triggers an illegal memory access on
> multi-GPU (`num_gpus > 1`). This document records the root cause, the failing
> repro, and the fix options so the work can be finished and validated on a
> multi-GPU machine. See [dynamic-filters.md](dynamic-filters.md) for the feature
> itself.

## TL;DR

The runtime membership/zone-map filter structures (`cuco::static_set`,
`cuco::bloom_filter`, zone-map `cudf::scalar` bounds) are built on the **hash-join
build side's GPU** and carry **no device identity**. On one GPU the build and the
probe-side scans that consume the filter always share that device, so it works. On
multiple GPUs, the scan subsystem's load balancer distributes probe scans across
**all** active GPUs; a scan running on GPU 1 then evaluates `set.contains(key)` (or
reads a zone-map device scalar) against storage that lives on GPU 0 → a
cross-device dereference → `cudaErrorIllegalAddress`.

The filter code did **not** change in the dev merge; dev's new load-balancing scan
distribution merely **exposed** a latent single-GPU assumption.

## Symptom / repro

CI `Test` job, self-hosted **2-GPU** runner, `num_gpus := 2`:

```
gpu_execution - TPC-H Query 2 parquet
test/cpp/integration/test_gpu_execution_tpch.cpp:229: FAILED:
  REQUIRE_FALSE( gpu_result->HasError() )
with messages:
  num_gpus := 2
  transparent GPU execution error: INTERNAL Error: Sirius GPU execution failed:
  Invalid Error: copy_if failed on 2nd step: cudaErrorIllegalAddress: an
  illegal memory access was encountered
```

- Fails on Q2 (min-cost-supplier: `part ⋈ partsupp ⋈ supplier ⋈ nation ⋈ region`),
  which builds selective IN-list membership filters that are pushed into probe scans.
- Dynamic filters are **on by default** (`0c7ca1d4`), so the integration suite
  exercises them.
- **Attribution:** the pre-merge feature tip passed this multi-GPU `Test` job
  (CI run `27307838188`, 2026-06-10, success). The post-merge branch fails it
  (run `28545289166`). The filter `.cu`/`.cpp` sources are byte-identical across
  the merge (`git diff 318cf9a6 HEAD -- src/cuda/sirius_dynamic_in_list_filter.cu
  src/op/sirius_dynamic_filter.cpp src/op/scan/dynamic_filter_merge.cpp` is empty),
  so this is a latent bug newly surfaced by the merged scan-distribution changes,
  not a broken merge resolution.
- **Not reproducible on a single-GPU box** (e.g. the GB10 dev machine): with one
  active GPU, build and probe always share a device. A multi-GPU machine is
  required to reproduce and to validate any fix.

### Where the illegal access surfaces

The CUDA error is asynchronous: the illegal read happens inside the cuco `contains`
kernel launched by `compute_mask`, but is not detected until the next synchronizing
call — the stream-compaction that applies the mask:

- `src/op/scan/dynamic_filter_merge.cpp:142` — `e.filter->compute_mask(probe, stream, mr)`
  launches the cross-device `contains` kernel.
- `src/op/scan/dynamic_filter_merge.cpp:80` — `cudf::apply_boolean_mask(current, mask->view(), stream, mr)`
  (internally `cudf::copy_if`) is the sync point that reports `copy_if failed on 2nd step`.

## Root cause

The filter structures are device-local with no device tag:

| Filter kind | Backing structure | Built where | Consumed where |
|---|---|---|---|
| IN-list (`sirius_dynamic_in_list_filter`) | `cuco::static_set` | build stream's GPU (`build_set`, `src/cuda/sirius_dynamic_in_list_filter.cu:65`) | `compute_mask` on the probe scan's stream/GPU (`...cu:146`, `set.ref(cuco::contains)` at `:160/:165`) |
| Bloom (`sirius_dynamic_bloom_filter`) | `cuco::bloom_filter` | build stream's GPU (`src/cuda/sirius_dynamic_bloom_filter.cu`) | `compute_mask` on the probe scan's stream/GPU |
| Zone-map (`sirius_dynamic_zone_map_filter`) | `cudf::scalar` min/max device bounds (`zone_map_entry`, `src/include/op/sirius_dynamic_filter.hpp:121`) | build side | referenced by the reader-side AST evaluated on the probe scan's GPU |

Key facts that make this unfixable without device awareness:

1. `sirius_mask_applicable::compute_mask(probe, stream, mr)`
   (`src/include/op/sirius_dynamic_filter.hpp:202`) has **no device parameter** — the
   device is only implicit in `stream`. Nothing checks that the set's device matches
   `stream`'s device.
2. The IN-list filter **does not retain the build keys** — it "builds its own
   persistent set and does not retain the view" (`sirius_dynamic_filter.hpp:224`). So a
   per-device set cannot be rebuilt from the filter alone as written.
3. The filter is a single device-agnostic `shared_ptr<sirius_dynamic_filter const>`
   in the channel (`sirius_dynamic_filter_set`, `sirius_dynamic_filter.hpp:346`), fanned
   out to consumers on any device.

### Why #996 does not cover it

`e57c6d7a fix(mgpu): enforce partition device pin for cuco-backed operators (#996)`
pins every **partition task** of a join (its build + probe) to the same real GPU via
`preferred_device_id`, indexing the active-executor set (`task_creator::_active_gpu_ids`,
`src/include/creator/task_creator.hpp:213`). That keeps a join's *own* cuco table on one
device. But a dynamic filter is published by the build and consumed by **probe-side scan
tasks**, which are *not* part of the join's partition and are distributed across GPUs by
the scan load balancer. The build set therefore still gets touched cross-device.

## Fix options

### Option A — Disable dynamic filters when `num_gpus > 1` (recommended stopgap)

Gate the feature off whenever more than one GPU is active. Correctness-safe (the join
is authoritative; a dropped filter only forgoes pruning), tiny, and unblocks CI
immediately. Cost: no dynamic-filter benefit on multi-GPU until Option B lands.

Cleanest gate point — the plan-gen enable check, which already has `config` in scope
and disables the whole feature (producer wiring + operator injection) before any set is
built:

- `src/planner/sirius_physical_plan_generator.cpp` — `dynamic_filter_pushdown_enabled(context)`
  returns `state->get_config().get_operator_params().enable_dynamic_filter_pushdown`. Add
  `&& state->get_config().get_hw_topology().num_gpus <= 1`. Do the same for the zone-map
  enable check (`enable_dynamic_zone_map_filter`), since zone-map bounds have the same
  cross-device hazard.

```cpp
bool dynamic_filter_pushdown_enabled(duckdb::ClientContext& context)
{
  auto state = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!state) { return false; }
  auto const& cfg = state->get_config();
  // Phase-1 filters are device-local (cuco set / zone-map scalars built on the build
  // GPU); on multi-GPU a probe scan on another device would dereference them
  // cross-device. Disable until per-device replication (Option B) lands.
  if (cfg.get_hw_topology().num_gpus > 1) { return false; }
  return cfg.get_operator_params().enable_dynamic_filter_pushdown;
}
```

Confirm `get_hw_topology().num_gpus` reflects the **effective/active** GPU count used by
execution (align with what `task_creator::_active_gpu_ids.size()` sees). A belt-and-suspenders
runtime guard can also early-return from `sirius_physical_hash_join::push_build_side_dynamic_filters`
(`src/op/sirius_physical_hash_join.cpp:1321`) once the active-GPU count is plumbed there.

### Option B — Per-device replication (the real multi-GPU fix; Phase 2)

Keep the benefit on multi-GPU by making each filter hold one structure per active GPU
and having `compute_mask` pick the one matching the probe stream's device.

Design sketch:

1. **Retain the source, not just the structure.** IN-list retains the build-key column
   (small — bounded by the build cardinality, ≤ a few 100k INT keys). Bloom retains its
   bit array (or the keys). Zone-map already retains host-derivable bounds; keep host
   copies of min/max.
2. **Per-device structure cache** inside each filter: `unordered_map<int device_id, structure>`
   guarded by a `std::mutex`. Populate either
   - **eagerly at publish** (`push_build_side_dynamic_filters`): loop over
     `task_creator::_active_gpu_ids`, `cudaSetDevice`, copy the source to that device,
     build the structure. The publish path already synchronizes before fan-out, so no
     consumer observes a half-built map. Preferred — avoids lazy-build concurrency.
   - or **lazily** on first `compute_mask` for a device, under the mutex.
3. **`compute_mask` selects by current device**: read the device from `stream` (or
   `cudaGetDevice`), look up (or build) that device's structure, then run the probe
   kernel. Zone-map's `to_ast` similarly emits device scalars resident on the consumer's
   device.
4. Files: `src/cuda/sirius_dynamic_in_list_filter.cu`, `src/cuda/sirius_dynamic_bloom_filter.cu`,
   the zone-map path in `src/op/sirius_dynamic_filter.cpp` + `...hpp` (add device-keyed
   storage), and the publish path in `src/op/sirius_physical_hash_join.cpp`.

Estimated ~200–300 lines plus concurrency care. **Must be developed and validated on a
multi-GPU machine** — the per-device path cannot be exercised on a single GPU.

### Option C — Pin filtered probe scans to the build GPU (not recommended)

Force scan tasks that consume a dynamic filter onto the build set's device. Preserves a
single set, but serializes the probe scan of filtered tables onto one GPU (loses
multi-GPU scan parallelism) and needs new scan-affinity wiring keyed on filter presence.

## Reproduce & validate on a multi-GPU machine

1. Config `num_gpus: 2` (integration suite uses `test/cpp/integration/integration.yaml`;
   the failing case sets `num_gpus := 2`).
2. Build and run the integration TPC-H target; Q2 is the reliable trigger:
   ```bash
   pixi run build/release/extension/sirius/test/cpp/sirius_unittest \
     "gpu_execution - TPC-H Query 2 parquet"
   ```
   Before a fix: `copy_if failed on 2nd step: cudaErrorIllegalAddress`.
   With Option A: filters disabled, query passes (verify via debug log that no
   `Pushed N dynamic filter(s)` line appears when `num_gpus > 1`).
   With Option B: query passes **and** debug logs still show `Pushed …` +
   `apply_dynamic_filters … apply: X -> Y rows` with real pruning across both GPUs.
3. Add a focused regression modeled on
   `test/cpp/operator/test_partition_memspace_mgpu.cpp` (from #996): a small selective
   build + large probe over ≥2 partitions, dynamic filters on, `num_gpus = 2`, asserting
   GPU result == CPU reference. A pre-fix run illegal-accesses (aborts the binary); the
   single-GPU control passes.
4. Single-GPU A/B correctness gate for the feature itself (unaffected by this bug) is in
   the memory note `tpch-ab-correctness-gate`.

## References

- Feature doc: [dynamic-filters.md](dynamic-filters.md); multi-GPU model:
  [multi-gpu-architecture.md](multi-gpu-architecture.md).
- Filters: `src/cuda/sirius_dynamic_in_list_filter.cu`,
  `src/cuda/sirius_dynamic_bloom_filter.cu`,
  `src/include/op/sirius_dynamic_filter.hpp`,
  `src/op/scan/dynamic_filter_merge.cpp` (apply at `:80` / `:142`).
- Publish path: `src/op/sirius_physical_hash_join.cpp:1321`
  (`push_build_side_dynamic_filters`).
- Device pinning precedent: `e57c6d7a` (#996); `task_creator::_active_gpu_ids`.
- Enable gate: `src/planner/sirius_physical_plan_generator.cpp`
  (`dynamic_filter_pushdown_enabled`).
- CI: fail `28545289166` (post-merge, `num_gpus=2`); last green `27307838188` (pre-merge).
