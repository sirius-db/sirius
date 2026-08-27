# PLAN-03 — Derive operator batch budgets from the CONFIGURED pool, and refuse over-subscribed device configs at bring-up

**Status:** plan only. Nothing in this document has been implemented.
**Written to be executed in a fresh session with zero prior context.** Everything you need is here.

---

## 0. Orientation (read this first if you have no context)

**Repo:** `/home/ubuntu/sirius`. Working branch `demo-multi-cn`; **the default/PR-target branch is `dev`**
(not `main`/`master`).

**What Sirius is:** a GPU-native SQL engine shipped as a DuckDB extension. Loading the extension makes
it transparently intercept normal SQL and route supported operators to the GPU (cuDF / RMM / cuCascade).
The live engine is "Super Sirius": `src/op/` (operators), `src/planner/`, `src/pipeline/`, `src/cuda/`.
Read `docs/super-sirius/` before touching engine code. **`src/legacy/` is the dead `gpu_processing`
path — do not modify it.**

**The compute node (CN):** `experimental/starrocks/` is a Rust binary (`sirius-starrocks-cn`) that
impersonates a StarRocks compute node and runs query fragments through Sirius via the C++ FFI in
`src/sirius_ffi.cpp`. One CN per GPU. This is what the SF500 TPC-H benchmark in
`bench/rtxpro6000-2gpu/` exercises.

**Build and test:**

```bash
pixi run make                # full build
pixi run make clean          # wipe build dir after a failed build
pixi run make test           # build + run the Catch2 C++ unit tests (what CI runs)

# one Catch2 tag / test name:
pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[sirius][config]"
# one SQLLogic file:
pixi run build/release/test/unittest --test-dir . test/sql/tpch-sirius.test
```

**Box this plan was written on:** 2× NVIDIA RTX PRO 6000 Blackwell Server Edition, driver 580.126.09,
CUDA 13.0, 48 cores, ~1.1 TiB host RAM. Two CNs, one per GPU.

**Related plans** (siblings in this directory, referenced but not required):
`PLAN-01-copy-out-on-arrival.md`, `PLAN-02-park-ownership-teardown.md`,
`PLAN-04-scheduler-stall.md`, `PLAN-05-bench-harness.md`.
Background analysis: `bench/rtxpro6000-2gpu/SF500-CONFIG-AND-ARCHITECTURE.md`.

**Verification legend used throughout:**
`[V]` = verified against source or measured in this session on this box.
`[V-doc]` = taken from a prior session's measurement recorded in `SF500-CONFIG-AND-ARCHITECTURE.md`
or a results CSV in this directory; the artifact exists and was read, but the measurement was not
re-run here.
`[UNVERIFIED]` = asserted, not confirmed. Treat as a hypothesis.

---

## 1. Problem statement

There are two independent defects. They share a root cause — **nothing in the engine relates the
operator budgets, the RMM pool, and the out-of-pool staging arena to each other or to the device** —
so they are fixed together.

### 1.1 Defect A — the batch budgets are derived from the physical card, not the configured pool

`sirius::config::derived_default_batch_size()` at **`src/sirius_config.cpp:38-58`**: `[V]`

```cpp
uint64_t derived_default_batch_size()
{
  // cudaGetDeviceCount/Properties honor CUDA_VISIBLE_DEVICES and do not create a context.
  static uint64_t const value = [] {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
      return DEFAULT_BATCH_SIZE;
    }
    uint64_t min_total = 0;
    for (int id = 0; id < device_count; ++id) {
      cudaDeviceProp prop{};
      if (cudaGetDeviceProperties(&prop, id) != cudaSuccess) { continue; }
      auto const total = static_cast<uint64_t>(prop.totalGlobalMem);
      min_total        = min_total == 0 ? total : std::min(min_total, total);
    }
    if (min_total == 0) { return DEFAULT_BATCH_SIZE; }
    constexpr uint64_t min_batch = 512ULL * 1024 * 1024;       // 512 MiB floor
    constexpr uint64_t max_batch = 5ULL * 1024 * 1024 * 1024;  // 5 GiB ceiling
    return std::clamp(min_total / 40, min_batch, max_batch);   // 2.5%
  }();
  return value;
}
```

It is the in-class initializer for four `operator_params` members and, doubled, a fifth
(`src/include/sirius_config.hpp:82, 91, 94, 97, 102`): `[V]`

| member | initializer | header line |
|---|---|---|
| `scan_task_batch_size` | `config::derived_default_batch_size()` | `src/include/sirius_config.hpp:82` |
| `hash_partition_bytes` | `config::derived_default_batch_size()` | `src/include/sirius_config.hpp:91` |
| `concat_batch_bytes` | `config::derived_default_batch_size()` | `src/include/sirius_config.hpp:94` |
| `sort_sample_bytes` | `config::derived_default_batch_size()` | `src/include/sirius_config.hpp:97` |
| `max_build_hash_table_bytes` | `2 * config::derived_default_batch_size()` | `src/include/sirius_config.hpp:102` |

**Measured on this box** (probe compiled against the repo's pixi CUDA headers; see §10 to reproduce): `[V]`

```
device_count=2
gpu0 totalGlobalMem=101973950464 (94.9706 GiB)
gpu1 totalGlobalMem=101973950464 (94.9706 GiB)
min_total/40=2549348761 (2.3743 GiB)  clamped=2549348761 (2.3743 GiB)
```

So on this box every default budget is **2,549,348,761 B = 2.3743 GiB**, and
`max_build_hash_table_bytes` is **5,098,697,522 B = 4.7486 GiB**. `[V]`

**Why that is wrong.** The pool the operators actually allocate from is the *configured*
`usage_limit`, not the card. The deployed SF500 config gives each CN a **60 GiB** pool
(`GPU_MEM=60GiB` → `usage_limit_bytes: "60GiB"` in the generated YAML; see
`/opt/dlami/nvme/sirius-build/up-sf500-x.sh`). `[V]` Against that pool:

| budget | absolute | as a fraction of the 60 GiB pool |
|---|---|---|
| `scan_task_batch_size` / `hash_partition_bytes` / `concat_batch_bytes` / `sort_sample_bytes` | 2.3743 GiB | pool/25.3 (3.96%) |
| `max_build_hash_table_bytes` | 4.7486 GiB | pool/12.6 (7.91%) |

The intent stated in the code and in `docs/super-sirius/configuration.md:360-362` is **2.5%**. Against
the pool it is 3.96%, i.e. **2.4× oversized** relative to a pool-relative reading of the same 2.5%
rule under the SF500 config. `[V]`

**It degenerates further with carve-outs.** `--gpu-memory-limit 8GiB` (the shape used by
`experimental/starrocks/pixi.toml:77,83` for the two-CN demo) produces an 8 GiB pool while the budgets
stay card-derived. On *this* box that would be `2.3743 GiB / 8 GiB = 29.7%` of the pool per batch and
`max_build_hash_table_bytes = 4.7486 GiB = 59.4%` of the entire pool. `[V]` (On the 23 GiB L4 the demo
was written for the pathology is mild — `23 GiB/40 ≈ 590 MiB` — which is why it has gone unnoticed.)
**The smaller the carve-out, the larger the budget relative to it.** That is backwards.

**This is not theoretical.** TPC-H q08 at SF500 on this box:

| config | q08 cold | q08 warm | source |
|---|---|---|---|
| defaults (2.3743 GiB budgets) | `refused` (10910 ms, `bad_alloc`) | `refused` (10226 ms) | `results/sf500xcold.csv` `[V]` |
| `HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB` | `pass` 17573 ms | `pass` 14163 ms | `results/sf500e5.csv` `[V]` |

Answer correctness: q08 agrees with the DuckDB oracle to 1 ULP
(`0.04051322057535745` vs `…744`). `[V-doc]` (`SF500-CONFIG-AND-ARCHITECTURE.md`)

Sweep-level effect, from `results/sf500xcold.csv` (cold column, full 22, defaults): 19 `pass`,
`q11 wedge`, `q08/q09/q21 refused`. `[V]` With the reduced budgets q08 passes.
Counting q11 as a correct-empty pass (its oracle `q11.tsv` is header-only, 17 bytes — the query
hardcodes the SF1 threshold `0.0001` where the spec scales it `0.0001/SF`) `[V-doc]`, that is
**20/22 → 21/22**. q21 is intermittent (3 samples: 17.7 s pass / 600 s hang / 20.6 s pass) `[V-doc]`,
so the exact headline count depends on how q21 and q11 are scored; the *unambiguous* claim is
**q08 goes from a hard `bad_alloc` refusal to a correct 14.2 s warm answer, by budget change alone.** `[V]`

**Why smaller budgets help.** `natural_num_partitions()` at
`src/include/op/sirius_physical_partition_consumer_operator.hpp:65-80`: `[V]`

```cpp
int num_partitions = max(1, ceil(total_bytes / hash_partition_bytes));
int const min_parts = partition_min_num_partitions(num_gpus);   // = num_gpus when >1, else 1
if (min_parts > 1 && total_bytes >= partition_small_table_bytes(num_gpus)) {
  num_partitions = std::max(num_partitions, min_parts);
}
```

Consumers: `sirius_physical_hash_join.cpp:733` and
`sirius_physical_grouped_aggregate_merge.cpp:157`. `[V]` Each CN declares `topology.num_gpus: 1`, so
`min_parts == 1` and `natural = ceil(total_bytes / hash_partition_bytes)` fully governs. **Lowering
`hash_partition_bytes` raises the partition count and lowers per-partition bytes** — exactly the lever
that moved q08.

**Config plumbing is confirmed reachable.** `operator_params` is parsed at
`src/sirius_config.cpp:535` (`if (auto n = r.optional_node("operator_params")) { from_yaml(*n, _operator_params); }`,
reader at `src/sirius_config.cpp:216-240`) `[V]` and read at execution time by
`src/sirius_engine.cpp:267`
(`sirius_ctx_ptr->get_config().get_operator_params()`) `[V]` — which is the engine the CN's fragments
run through (`src/sirius_ffi.cpp:163-241`, `Context::Context(config_path)` at `src/sirius_ffi.cpp:251`). `[V]`

**But `operator_params` is reachable ONLY via `--sirius-config`.** `experimental/starrocks/src/main.rs:67-71`
makes `--sirius-config` mutually exclusive with `--gpu-memory-limit` / `--gpu-memory-fraction` /
`--host-memory-limit`: `[V]`

```rust
#[arg(long, conflicts_with_all = ["gpu_memory_limit", "gpu_memory_fraction", "host_memory_limit"])]
sirius_config: Option<PathBuf>,
```

and the flag path synthesises `.cn<i>/derived-sirius-config.yaml`
(`main.rs:258-301`, generator `experimental/starrocks/src/engine_settings.rs:34-110`) containing
**only**: `topology.num_gpus: 1`, `memory.gpu.usage_limit_bytes` (or `usage_limit_fraction`),
`memory.gpu.reservation_limit_fraction: 1.0`, `memory.host.capacity_bytes`,
`executor.scan_manager.use_sirius_datasource`, `cpu_affinity` on three pools, and
`telemetry.output_directory`. **No `operator_params`, no disk tier, no downgrade fractions.** `[V]`
So today the *only* way to get sane budgets on a CN is to hand-write a full YAML — which is precisely
what `/opt/dlami/nvme/sirius-build/up-sf500-x.sh` had to do. `[V]`

### 1.2 Defect B — nothing validates that the config fits on the device

Per-GPU occupancy for a CN is:

```
occupancy = RMM pool (memory.gpu.usage_limit)
          + staging arena (SIRIUS_EXCHANGE_STAGING_BYTES — a bare cudaMalloc OUTSIDE the pool)
          + CUDA context / cuDF / transport overhead
```

The arena is deliberately outside every pool: `src/include/exec/exchange_staging_arena.hpp:30-33`
explains that UCX `cuda_ipc` cannot export `cudaMallocAsync` allocations and silently degrades ~220×.
`[V]` It is created from the env var at `src/exec/exchange_staging_arena.cpp:190-201`
(`exchange_staging_arena::from_env()`), keyed by
`exchange_staging_arena::kCapacityEnvVar = "SIRIUS_EXCHANGE_STAGING_BYTES"`
(`src/include/exec/exchange_staging_arena.hpp:49`), and instantiated on the CN path at
`src/sirius_ffi.cpp:228` — **after** engine bring-up. `[V]`

**Device numbers measured on this box** `[V]`:

```
nvidia-smi -q -d MEMORY -i 0:  Total 97887 MiB   Reserved 638 MiB   Used 0 MiB   Free 97250 MiB
cudaGetDeviceProperties(0).totalGlobalMem = 101,973,950,464 B = 94.9706 GiB
cudaMemGetInfo(0) = { free 101,388,648,448 B (94.4255 GiB), total 101,973,950,464 B }
```

> **CORRECTION to a widely-repeated claim in this directory.** `97887 MiB − 638 MiB = 97249 MiB =
> 94.970 GiB`, which is **exactly** `totalGlobalMem`. `cudaGetDeviceProperties().totalGlobalMem` and
> `cudaMemGetInfo()`'s `total` **already exclude the 638 MiB driver reservation.** A validator must
> *not* subtract it again — doing so would double-count and reject valid configs. The figure that
> still has to be subtracted is the **CUDA context**, measured below. `[V]`

**CUDA context cost, measured** `[V]` (probe in §10): after only
`cudaGetDeviceCount` + `cudaGetDeviceProperties`, `nvidia-smi` reports **no compute app** and
`memory.used = 3 MiB` — confirming the comment at `src/sirius_config.cpp:40` that these two calls do
**not** create a context. After the first `cudaMemGetInfo`, the probe's pid appears holding
**550 MiB** and `memory.used` is 559 MiB. So a *bare* cudart context costs ~550 MiB here; a real CN
additionally carries cuDF, RMM, nixl/UCX registrations and fragmentation.

**The deployed config runs at the very edge.** At `GPU_MEM=60GiB STAGING=32GiB` the live CNs were
measured holding **95850 MiB (GPU0)** and **96976 MiB (GPU1)**. `[V-doc]` Since
`60 GiB + 32 GiB = 94208 MiB`, the implied non-pool, non-arena overhead is **1642 MiB (GPU0)** and
**2768 MiB (GPU1)** — the "~1.6–2.7 GiB" figure quoted elsewhere in this directory. `[V]` (arithmetic
from the `[V-doc]` readings). Against `97250 MiB` allocatable, GPU1 had roughly
`97250 − 96976 = 274 MiB` free. (The figure "252 MiB" appears in the session notes; the exact value
is `[UNVERIFIED]` here — no cluster is running in this session. The order of magnitude, a few hundred
MiB, is what matters.)

**And nothing checks this.** Search results: `[V]`

* `SiriusContext::initialize()` (`src/sirius_context.cpp:617`) validates only that
  `topology.num_gpus != 0` (`:626-635`) and that a HOST memory space exists (`:794-799`). It
  constructs the memory manager at `src/sirius_context.cpp:641` with no device-budget check.
* cuCascade's configurator *fetches* the number it would need and throws it away:
  `cucascade/src/memory/reservation_manager_configurator.cpp:320-324` calls
  `rmm::available_device_memory()` (which returns `{free, total}` — see
  `.pixi/envs/default/include/rmm/cuda_device.hpp:94-99`) and keeps only `total`.
* The CN's one preflight, `EngineConfig::ensure_gpu_unclaimed()`
  (`experimental/starrocks/src/main.rs:304-361`), **explicitly returns early** when a carve-out or a
  config file is supplied (`main.rs:306-311`) — i.e. it never fires for any real deployment.

The consequence: an over-subscribed config does not fail at bring-up. It fails minutes later, mid-sweep,
as either an opaque `std::bad_alloc` from RMM or
`"exchange staging arena exhausted: … (raise SIRIUS_EXCHANGE_STAGING_BYTES)"`
(`src/exec/exchange_staging_arena.cpp:245-256`) — and the arena message actively **misdirects**,
because raising the arena shrinks the pool 1:1.

### 1.3 Retired folklore

`STAGING ≈ 96 GiB × (SF/500) / N` is **retired**. It was wrong in both directions. Measured arena
high-water: **SF100 full sweep 6.51 GiB**; **SF500 healthy max 26.78 GiB** (q18 18.68, q17 16.06); the
*same* q21 measured **47.40 GiB** with a starved pool. `[V-doc]` No `(SF, N)` formula produces a 1.8×
spread on identical work — arena occupancy is a **pressure gauge for the pool**, because
`Fragment::push_packed` deep-copies each staged batch into pool memory *before* the lease is released
(`src/sirius_ffi.cpp:840-859`, the copy at `:849`). `[V]` This plan does **not** try to derive an
arena size; it only makes the engine *check* the one the operator chose.

---

## 2. Every consumer of `derived_default_batch_size()`, and what each would do differently

### 2.1 Direct references

Exhaustive (`grep -rn derived_default_batch_size`, excluding `src/legacy/`): `[V]`

| file:line | role |
|---|---|
| `src/sirius_config.cpp:38` | definition |
| `src/include/sirius_config.hpp:43` | declaration |
| `src/include/sirius_config.hpp:82` | initializer of `operator_params::scan_task_batch_size` |
| `src/include/sirius_config.hpp:91` | initializer of `operator_params::hash_partition_bytes` |
| `src/include/sirius_config.hpp:94` | initializer of `operator_params::concat_batch_bytes` |
| `src/include/sirius_config.hpp:97` | initializer of `operator_params::sort_sample_bytes` |
| `src/include/sirius_config.hpp:102` | initializer of `operator_params::max_build_hash_table_bytes` (`2 *`) |

There are **no other call sites**, and **no test asserts its value** (`grep -rn derived_default_batch_size test/`
returns nothing). `[V]`

### 2.2 Where each derived value is consumed, and the behavioural delta

Read `pool` below as "the per-GPU `usage_limit`". The delta column assumes the recommended
`pool/40` (§4); at a 60 GiB pool that is 1.500 GiB vs today's 2.3743 GiB, i.e. **0.632×**.

#### `scan_task_batch_size`

| consumer | file:line | what changes |
|---|---|---|
| parquet scan sizing | `src/planner/sirius_physical_plan_generator.cpp:215` (`info->approximate_batch_size`) | smaller scan tasks → more, smaller batches; more parallelism, more per-batch overhead |
| duckdb-native scan sizing | `src/planner/sirius_physical_plan_generator.cpp:252` | same |
| `GPU_VALUES` replacement for `COLUMN_DATA_SCAN` / `EMPTY_RESULT` | `sirius_physical_plan_generator.cpp:755-757` | same; note the `== 0` fallback to `config::DEFAULT_SCAN_TASK_BATCH_SIZE` |
| `GPU_VALUES` replacement for `DUMMY_SCAN` | `sirius_physical_plan_generator.cpp:766-768` | same |
| `CALL pin_table` batch size | `src/sirius_extension.cpp:1248-1249` | pinned tables ingest in smaller chunks; lower peak during pinning |
| registered DuckDB setting default | `src/sirius_extension.cpp:2242-2247` | `current_setting('scan_task_batch_size')` reports the pool-derived value |

#### `hash_partition_bytes`

| consumer | file:line | what changes |
|---|---|---|
| hash-join partition strategy | `src/op/sirius_physical_hash_join.cpp:733` via `natural_num_partitions` | **more partitions, smaller per-partition working set** — the q08 lever |
| grouped-aggregate merge | `src/op/sirius_physical_grouped_aggregate_merge.cpp:157` | same |
| plan-time wiring into the partition operator | `sirius_physical_plan_generator.cpp:486`, `:673` | same |
| comparison-join plan path | `src/planner/sirius_plan_comparison_join.cpp:692-694` (params read at `:573`) | same |
| registered setting default | `src/sirius_extension.cpp:2284-2288` | reported default changes |
| zero-guard | `src/sirius_config.cpp:225-227` and `src/sirius_extension.cpp:1892-1895` | unchanged; must keep rejecting 0 |

#### `concat_batch_bytes`

| consumer | file:line | what changes |
|---|---|---|
| CONCAT flush threshold | `src/op/sirius_physical_concat.cpp:106` and `:148` (`total_batch_size > _concat_batch_bytes`) | flushes sooner → smaller output batches, lower transient peak |
| plan-time construction | `sirius_physical_plan_generator.cpp:610` | same |
| registered setting default | `src/sirius_extension.cpp:2290-2294` | reported default changes |

#### `sort_sample_bytes`

| consumer | file:line | what changes |
|---|---|---|
| sample sufficiency test | `src/op/sirius_physical_sort_sample.cpp:61`, `:116`, `:153` | **fewer bytes sampled before boundaries are computed** — this is a *statistical sample*, not a working-set bound, so shrinking it trades boundary quality (partition skew) for a smaller sample buffer |
| plan-time construction | `sirius_physical_plan_generator.cpp:550` | same |
| registered setting default | `src/sirius_extension.cpp:2296-2300` | reported default changes |

> **Call this out in review.** `sort_sample_bytes` does not belong on the same scale as the other
> three: it bounds a sample, not a partition. It shares the default today purely for convenience
> (`src/include/sirius_config.hpp:97`). The SF500 config never touched it (it ran at 2.3743 GiB while
> the other three were cut to 1 GiB) and nothing regressed `[V]` — evidence that it is not the binding
> constraint. This plan keeps it on the shared default (minimal change) but flags it as the most
> likely candidate for a separate rule later.

#### `max_build_hash_table_bytes` (= `2 ×` the shared default)

| consumer | file:line | what changes |
|---|---|---|
| BUILD_PROBE eligibility gate | `src/op/sirius_physical_hash_join.cpp:769` (`per_gpu_build_bytes < max_build_hash_table_bytes`) | **fewer joins take BUILD_PROBE**; more take STANDARD. This is the one place where a smaller value can *cost* performance |
| build-side `concat_all` decision | `src/op/sirius_physical_partition.cpp:496-501` (via `hash_join->max_build_hash_table_bytes()`, accessor at `src/include/op/sirius_physical_hash_join.hpp:350`) | build side is concatenated into one table less often |
| plan-time wiring | `sirius_plan_comparison_join.cpp:692`, `sirius_physical_hash_join.cpp:803-804` | same |
| registered setting default | `src/sirius_extension.cpp:2302-2307` | reported default changes |

Note the deliberate design at `src/op/sirius_physical_hash_join.cpp:762-763`:
`hash_partition_bytes` "targets a streaming batch size and must not veto `max_build_hash_table_bytes`,
which is what sizes the folded hash table." Keeping the `2×` relation preserves that. The SF500 working
config used `MBHT = 2 × HPB` (2 GiB / 1 GiB), consistent with the existing rule. `[V]`

### 2.3 The static fallbacks (unchanged by this plan)

`src/include/sirius_config.hpp:37, 45-50` define compile-time constants used by operator constructors
that are built without a config (unit tests, direct construction): `[V]`

```cpp
constexpr uint64_t DEFAULT_BATCH_SIZE = 800ULL * 1024 * 1024;  // 800 MiB
constexpr uint64_t DEFAULT_SCAN_TASK_BATCH_SIZE       = DEFAULT_BATCH_SIZE;
constexpr uint64_t DEFAULT_HASH_PARTITION_BYTES       = DEFAULT_BATCH_SIZE;
constexpr uint64_t DEFAULT_CONCAT_BATCH_BYTES         = DEFAULT_BATCH_SIZE;
constexpr uint64_t DEFAULT_SORT_SAMPLE_BYTES          = DEFAULT_BATCH_SIZE;
constexpr uint64_t DEFAULT_MAX_BUILD_HASH_TABLE_BYTES = 2 * DEFAULT_BATCH_SIZE;
```

These are the defaults on `sirius_physical_hash_join.hpp:262,264,274,275,414`,
`sirius_physical_sort_sample.hpp:38,46,103`, `sirius_physical_concat.hpp:43`,
`sirius_physical_grouped_aggregate_merge.hpp:51`,
`sirius_physical_partition_consumer_operator.hpp:121`. `[V]`
**Leave them alone.** Operator unit tests that construct operators directly depend on them and are
unaffected by this plan.

---

## 3. The design problem, honestly

### 3.1 Why this is not a one-line change

`derived_default_batch_size()` is a **`static`-memoised free function** whose value is computed the
first time an `operator_params` is default-constructed — which happens **before any config is parsed**,
and in fact before `main()` in some translation units, because it is an *in-class member initializer*.
Three specific hazards:

1. **Memoisation.** `static uint64_t const value = [] {...}();` (`src/sirius_config.cpp:41`) is
   computed once per process, thread-safely, and never recomputed. Any scheme that tries to make this
   function pool-aware by "just changing what it reads" is broken by construction: the first
   `operator_params{}` anywhere in the process freezes the answer. `[V]`
2. **Ordering.** The configured pool does not exist until the memory-space configs are built. Timeline
   for the extension path: `[V]`
   * `SiriusContextExtensionCallback::SiriusContextExtensionCallback()` — `src/sirius_context.cpp:1581-1591`
     → `read_config_file_if_exists()` (`:1590`)
   * `read_config_file_if_exists()` — `src/sirius_context.cpp:1630-1668` → either
     `config_.load_from_file(path)` (`:1640`) **or** `config_.apply_defaults()` (`:1663`)
   * then `context_->initialize(config_)` (`:1667`)
   * later, from `LoadInternal`, `SiriusExtension::InitialGPUConfigs(config, callback_ptr->get_loaded_config())`
     (`src/sirius_extension.cpp:2470`) registers every DuckDB setting using
     `defaults.get_operator_params()` (`src/sirius_extension.cpp:2121`).

   Timeline for the FFI/CN path: `Context::Context()` → `apply_defaults()` (`src/sirius_ffi.cpp:244-248`)
   or `Context::Context(path)` → `load_from_file()` (`src/sirius_ffi.cpp:251-255`), then
   `bring_up()` → `context->initialize(config)` (`src/sirius_ffi.cpp:181`).

   **Both paths end their config phase with `_memory_space_configs` populated.** That is the hook point.
3. **`cudaGetDeviceCount` / `cudaGetDeviceProperties` do not create a CUDA context** — the comment at
   `src/sirius_config.cpp:40` claims this, and it is **measured true** on this box (§1.2). That property
   is why the current call is safe in a static initializer. **Any replacement must preserve it or move
   out of static-initialization entirely.** `cudaMemGetInfo` *does* create a context (measured: 550 MiB),
   so it may only be called at bring-up, never from a static initializer.

Also note: `load_from_file()` does **not** call `apply_defaults()` — it builds `_memory_space_configs`
itself at `src/sirius_config.cpp:559-585` (`_memory_space_configs.clear()` at `:559`, the configurator
branch at `:571-585`). `[V]` Any hook must be installed in **both** functions.

### 3.2 Where the pool size actually lives after parsing

`sirius_config::get_memory_space_configs()` returns
`std::vector<cucascade::memory::memory_space_config>`; the GPU alternative is
`cucascade::memory::gpu_memory_space_config` with
`std::size_t memory_capacity` (`cucascade/include/cucascade/memory/config.hpp:32-59`, field at `:37`). `[V]`
`memory_capacity` is the **resolved absolute pool bytes** — the configurator turns both the fraction
and the absolute form into bytes at
`cucascade/src/memory/reservation_manager_configurator.cpp:222`
(`config.memory_capacity = _gpu_capacity.get_capacity(info.gpu_capacity)`;
`get_capacity` at `reservation_manager_configurator.hpp:223-232`). `[V]`
`info.gpu_capacity` is `cudaMemGetInfo`'s `total` for that device
(`reservation_manager_configurator.cpp:319-325`). `[V]`

So after `apply_defaults()` or `load_from_file()`, the exact per-GPU pool is available with no CUDA
call and no guessing. **This is the single most important fact in the plan.**

### 3.3 There is already a precedent for lazy, memory-manager-keyed resolution

`max_sort_partition_bytes == 0` means "auto", resolved at *runtime* from the live memory space:
`src/op/sirius_physical_sort_sample.cpp:265-270` `[V]`

```cpp
size_t available_memory    = space->get_available_memory(stream);
size_t max_partition_bytes = _max_partition_bytes_override > 0
                               ? _max_partition_bytes_override
                               : static_cast<size_t>(available_memory * _max_partition_memory_fraction);
```

This is the shape a "lazy resolution keyed off the memory manager" would take. It is rejected as the
primary design below, for reasons given.

### 3.4 Three options

#### Option A — resolve after config parse (RECOMMENDED)

Turn the derivation into a **pure function of the pool**, and apply it once, right after the config is
parsed, to every budget the user did not set explicitly.

1. **`src/include/sirius_config.hpp`** — replace the CUDA-calling declaration at `:43` with a pure one,
   and change the five in-class initializers at `:82, 91, 94, 97, 102` to the compile-time
   `config::DEFAULT_*` constants (§2.3). Net effect: `operator_params{}` becomes trivially
   constructible with **no CUDA call in a static initializer at all**.

   ```cpp
   /// Shared operator batch default for a GPU pool of `pool_bytes`: pool/DIVISOR, clamped to
   /// [512 MiB, 5 GiB]. Pure — no CUDA, no memoization, no global state. `pool_bytes == 0`
   /// (no GPU space configured) returns DEFAULT_BATCH_SIZE.
   [[nodiscard]] constexpr uint64_t batch_size_for_pool(uint64_t pool_bytes) noexcept;
   ```

2. **`src/include/sirius_config.hpp`** — record which budgets the YAML set explicitly. Put the flags on
   `sirius_config` (private), **not** on `operator_params`, so the SET/`current_setting`-facing struct
   is untouched:

   ```cpp
   /// Which batch budgets an explicit YAML key set. The rest are filled from the configured
   /// GPU pool by resolve_operator_batch_defaults().
   struct operator_batch_explicit {
     bool scan_task_batch_size{false};
     bool hash_partition_bytes{false};
     bool concat_batch_bytes{false};
     bool sort_sample_bytes{false};
     bool max_build_hash_table_bytes{false};
   };
   ```

   `from_yaml(node, operator_params&)` at `src/sirius_config.cpp:216-240` currently reads straight into
   the struct with `r.optional("scan_task_batch_size", yaml::bytes(opt.scan_task_batch_size))`. The
   reader gives no presence signal (`src/include/yaml_reader.hpp:331-343` — `optional` just returns
   when the key is absent) `[V]`, so parse into `std::optional<uint64_t>` locals and set both the field
   and the flag — **the exact pattern `gpu_mem_config::from_yaml` already uses** for
   `usage_limit_bytes` (`src/sirius_config.cpp:317-322`). `[V]` This requires threading the flag struct
   into `from_yaml`; make it a two-argument overload or a small aggregate parameter.

3. **`src/sirius_config.cpp`** — add a private
   `void sirius_config::resolve_operator_batch_defaults()` that:
   * scans `_memory_space_configs` for `gpu_memory_space_config` alternatives and takes
     `min(memory_capacity)` across them (mirroring today's "smallest visible GPU" semantics);
   * for each of the five budgets not flagged explicit, assigns `batch_size_for_pool(pool)`
     (`max_build_hash_table_bytes` gets `2 *`, saturating);
   * logs one INFO line naming the pool, the divisor, the resulting value, and which fields were
     overridden by YAML — this must be greppable, because today the effective budget is invisible.

   Call it as the **last statement** of `sirius_config::apply_defaults()` (`src/sirius_config.cpp:472-487`)
   and of the `try` block in `sirius_config::load_from_file()` (after
   `enforce_sirius_datasource_for_multi_gpu()` at `src/sirius_config.cpp:587`). `[V]`

4. Keep the old free function only if something outside this plan needs it. **Nothing does** (§2.1) —
   prefer to delete `derived_default_batch_size()` outright and update
   `docs/super-sirius/configuration.md`.

**Why this option:** it eliminates the memoisation problem by eliminating the memoised state; it makes
the derivation a `constexpr`-testable pure function; it runs after parse and before *every* consumer
(`InitialGPUConfigs` at `src/sirius_extension.cpp:2470` runs after the callback ctor, and
`SiriusContext::initialize` at `src/sirius_context.cpp:617` is called from `:1666`) `[V]`; and it keeps
one authoritative value that `current_setting()` reports honestly.

**Cost:** touches the YAML reader for `operator_params`, adds a flag struct, and changes the reported
defaults. Moderate but contained.

#### Option B — lazy resolution keyed off the memory manager (NOT recommended as primary)

Mirror `max_sort_partition_bytes`: treat `0` as "auto" and resolve at operator-construction time from
`space->get_available_memory()`.

Rejected because:
* `0` is already **taken** for two of the five knobs with *different* meanings —
  `hash_partition_bytes == 0` is a hard error (`src/sirius_config.cpp:225-227`,
  `src/sirius_extension.cpp:1892-1895`), and `scan_task_batch_size == 0` already means "use the
  compile-time `DEFAULT_SCAN_TASK_BATCH_SIZE`" (`sirius_physical_plan_generator.cpp:755-757`). `[V]`
* `get_available_memory()` is *free* memory, not *configured* capacity, so the derived value would
  drift query to query — non-reproducible plans and non-reproducible benchmarks.
* `current_setting('hash_partition_bytes')` would report a sentinel rather than the value in force.

#### Option C — leave the function, add a pool-aware post-parse override only for the CN

Rejected: the same 2.4× oversizing hits every non-CN user with a `usage_limit` below 1.0, which is the
default (`gpu_mem_config::usage_limit{0.95}`, `src/sirius_config.cpp:306`). `[V]`

### 3.5 Tests that depend on current behaviour

`grep` results: `[V]`

* **No test asserts `derived_default_batch_size()`'s value.** There is no test file referencing it.
* `test/cpp/config/test_context.cpp:243-352` ("YAML-backed operator and compression settings are DuckDB
  defaults") asserts `current_setting(...)` equals the **explicit** values in
  `test/cpp/config/data/setting_defaults.yaml` (1–7 MiB). Because every budget is set explicitly there,
  the flags mark all five as explicit and the resolver is a no-op. **This test must keep passing
  unchanged** — it is the regression guard for the "explicit YAML wins" half of the change.
* `test/cpp/config/test_context.cpp:215-226` ("rejects zero hash partition bytes") calls
  `load_from_file` on `data/invalid_hash_partition_zero.yaml` and expects a throw. The throw happens
  inside `from_yaml` (`src/sirius_config.cpp:225-227`), before memory spaces are built, so the resolver
  never runs. Unaffected.
* Every other YAML fixture sets `operator_params` explicitly and is therefore unaffected:
  `test/cpp/integration/integration.yaml:21-26`, `integration-2gpu.yaml:22-27`,
  `integration-gb10.yaml:36-41`, `integration_s3cache.yaml:24-29`,
  `test/cpp/config/data/single_node.yaml:16-20`. `[V]`

---

## 4. What ratio to use

### 4.1 The evidence, stated as pool-relative divisors

| box / config | pool | budget | divisor (pool/x) | outcome | verif |
|---|---|---|---|---|---|
| RTX PRO 6000, 2 CN, SF500 — **current default** | 60 GiB | 2.3743 GiB (all four) | **/25.3** | **q08 `bad_alloc`** | `[V]` |
| RTX PRO 6000, 2 CN, SF500 — shipped fix | 60 GiB | 1 GiB (HPB/STB/CBB), 2 GiB MBHT | **/60** | **q08 pass, 17.6 s cold / 14.2 s warm, 1 ULP** | `[V]` |
| RTX PRO 6000, 2 CN, SF500 — q09 probe | 60–70 GiB | 512 MiB | **/120–/140** | q08 fine; q09 still fails (not a budget problem) | `[V-doc]` |
| GB300, 1 GPU, SF1000 (`bench/sf1000-repro/sirius-sf1000.yaml`) | ~243 GB (0.95 × 256 GB) | `scan_task_batch_size: 8GB` | **/30.4** | measured optimum; 5 GB → 8 GB is −1.85 % suite (q4 −25 %, q12 −18 %); 10 GB adds nothing | `[V]` file; `[V-doc]` measurement |
| GB300, same file | ~243 GB | `hash_partition_bytes: 32GB`, `max_build_hash_table_bytes: 32GB` | **/7.6** | file header records 16 GB vs 32 GB as **inert** (−0.06 %) | `[V]` |
| GB300, same file | ~243 GB | `concat_batch_bytes: 5GB` | **/48.6** | — | `[V]` |
| A100×8, 8 CN (`bench/a100x8/engine-a-sirius.yaml`) | 68 GiB (0.85 × 80 GiB) | `scan 3GB`, `hpb 8GB`, `concat 2GB`, `MBHT 8GB` | **/24.3, /9.1, /36.5, /9.1** | prescriptive config with "SWEEP {2,3,4} GB" annotations — **not** a measured optimum | `[V]` file; validation status `[UNVERIFIED]` |

The GB300 pool figure `~243 GB` is `0.95 × 256 GB`; the file header states 8 GB "peaks at 253.9 GB of
256 GB". The card's exact `totalGlobalMem` is `[UNVERIFIED]` from this box.

### 4.2 What the evidence does and does not support

**Supported:** the *basis* must change from card to pool. Every anchor above is naturally expressed
against the pool, and the card-relative rule produces absurdities under carve-out (29.7 % of an 8 GiB
pool per batch; 59.4 % for `max_build_hash_table_bytes` — §1.1). That part is not in doubt.

**Not supported:** a single universal divisor. The anchors span **/7.6 to /120** for
`hash_partition_bytes` alone. The spread is real and explicable — the GB300 ran a *single* CN with the
whole card as pool and no exchange arena, while the RTX PRO 6000 runs *two* CNs each of which also
sustains a 32 GiB out-of-pool arena whose drain is gated on pool availability
(§1.3, `src/sirius_ffi.cpp:849`) and leaks ~11.3 GiB of parked exchange output per q07 run per CN
`[V-doc]` (that leak is PLAN-02's subject). **A pool-relative divisor is strictly more correct than a
card-relative one, but it is not a complete model of memory pressure**, and this plan should not
pretend otherwise.

### 4.3 Recommended formula

```cpp
// src/include/sirius_config.hpp
constexpr uint64_t BATCH_SIZE_POOL_DIVISOR = 40;
constexpr uint64_t BATCH_SIZE_FLOOR        = 512ULL * 1024 * 1024;   // 512 MiB  (unchanged)
constexpr uint64_t BATCH_SIZE_CEILING      = 5ULL * 1024 * 1024 * 1024;  // 5 GiB (unchanged)

[[nodiscard]] constexpr uint64_t batch_size_for_pool(uint64_t pool_bytes) noexcept
{
  if (pool_bytes == 0) { return DEFAULT_BATCH_SIZE; }             // 800 MiB, no GPU space
  return std::clamp(pool_bytes / BATCH_SIZE_POOL_DIVISOR, BATCH_SIZE_FLOOR, BATCH_SIZE_CEILING);
}
```

`max_build_hash_table_bytes` stays at `2 ×` this (saturating), preserving both the documented relation
and the SF500 working config's own 2:1 ratio. `[V]`

**Why /40 and not /60:** `/40` is the *literal translation of today's stated intent* ("2.5 %") from
card to pool. On the un-carved-out default path (`usage_limit_fraction: 0.95`) it changes the value by
only 5 % — `0.95 × card / 40 = 2.256 GiB` vs today's `card/40 = 2.3743 GiB` — so the entire existing
single-node performance history remains valid. All the movement lands exactly where the defect is: on
carve-outs. `/60` (1 GiB at a 60 GiB pool) is the only SF500-verified divisor, but it changes the
default path by 1.58×, in a direction nobody has measured.

**Merge gate — Experiment E1.** `/40` at a 60 GiB pool is **1.5 GiB**, which sits in the untested gap
between the known-good 1 GiB and the known-bad 2.3743 GiB. Before merging, run:

```bash
GPU_MEM=60GiB STAGING=32GiB HOST_MEM=200GiB \
  HPB=1536MiB MBHT=3GiB STB=1536MiB CBB=1536MiB \
  /opt/dlami/nvme/sirius-build/sweep-sf500x-cold.sh q08
```

* **q08 passes** → ship `/40`.
* **q08 fails** → ship `/60` (`BATCH_SIZE_POOL_DIVISOR = 60`), which is already verified end-to-end,
  and record the failure in this file.

Do not merge without running E1. It is a single query and ~2 minutes of cluster time.

### 4.4 Validity range and binding constraints

With `/40`, floor 512 MiB, ceiling 5 GiB:

| pool | derived budget | binding term |
|---|---|---|
| < 20 GiB | 512 MiB | **floor** |
| 20 GiB | 512 MiB | floor (exactly) |
| 60 GiB | 1.500 GiB | divisor — the SF500 CN |
| 68 GiB (A100×8 arm) | 1.700 GiB | divisor |
| 90.2 GiB (this box, no YAML, 0.95 default) | 2.256 GiB | divisor |
| 200 GiB | 5 GiB | **ceiling** |
| ≥ 200 GiB | 5 GiB | ceiling — the GB300 class |

**The divisor is only evidence-anchored in the 60–90 GiB band.** Below 20 GiB the floor governs and
the change is a no-op relative to small cards (the 23 GiB L4 in `pixi run cluster2` gets 512 MiB
either way — today `23 GiB/40 ≈ 590 MiB`, after the change `8 GiB/40 = 205 MiB` → floored to 512 MiB;
a 1.15× change). Above 200 GiB the ceiling governs and the derived value can never reach the
GB300-measured optimum of 8 GB for `scan_task_batch_size` — but that box sets the value explicitly
(`bench/sf1000-repro/sirius-sf1000.yaml`), so the ceiling costs nothing today. **Do not raise the
ceiling in this change**: with divisor 40 it only binds above a 200 GiB pool, and there is no measured
case of a *derived* value being clipped in production.

### 4.5 Evidence that is missing (state this in the PR)

1. **The 1–2.37 GiB gap at a 60 GiB pool.** Only 512 MiB, 1 GiB (good) and 2.3743 GiB (bad) were
   tested. E1 closes this for q08 only.
2. **Whether the four knobs want the same divisor.** They were only ever moved together
   (`HPB=MBHT/2=STB=CBB`). No single-knob ablation exists. `sort_sample_bytes` in particular was left
   at 2.3743 GiB in the passing SF500 config and did not matter. `[V]`
3. **The default (no-YAML) path was never re-measured.** `pixi run cluster` supplies no memory flags,
   so it runs at `0.95 × card`. No SF-scale timing exists for it on this box.
4. **SF100 and SF300 were never re-run with the new budgets.** `results/sf100-*.csv` and
   `results/sf300-float64.csv` are all at defaults. `[V]` (This is also PLAN-08 item 1.)
5. **A100×8 divisors are prescriptive, not measured.** `bench/a100x8/engine-a-sirius.yaml` carries
   "SWEEP {2,3,4} GB" comments; whether its results CSVs were produced with that exact file is
   `[UNVERIFIED]`.

---

## 5. Bring-up validation

### 5.1 Where

**Primary (authoritative): C++, `SiriusContext::initialize()`, `src/sirius_context.cpp`.**
Insert immediately **after** the topology check that ends at `src/sirius_context.cpp:640` and
**before** `memory_manager_ = std::make_unique<...>(config_.get_memory_space_configs());` at
`src/sirius_context.cpp:641`. `[V]`

Rationale: this is the one funnel every entry point passes through —
extension `LOAD` (`src/sirius_context.cpp:1666`), FFI `Context()` and `Context(path)`
(`src/sirius_ffi.cpp:181`, reached from `:244-256`), and therefore every CN. `[V]` It has
`config_.get_memory_space_configs()` with resolved `memory_capacity` per GPU (§3.2), and the pool has
not been created yet, so a refusal costs nothing.

**Secondary (optional, cheap): Rust, `experimental/starrocks/src/main.rs`.**
`EngineConfig::ensure_gpu_unclaimed()` (`main.rs:304-361`) already exists and already shells out to
`nvidia-smi`; today it returns early whenever `--sirius-config` or a GPU carve-out is set
(`main.rs:306-311`), i.e. always in practice. `[V]` A useful minimal change: keep the early return for
`--sirius-config` (Rust cannot see inside the YAML) but **stop returning early for
`--gpu-memory-limit` / `--gpu-memory-fraction`** — instead compare
`carve-out + SIRIUS_EXCHANGE_STAGING_BYTES + headroom` against `nvidia-smi --query-gpu=memory.free`.
There is a direct precedent for CN-side bring-up validation: `Tunables::resolve()` is called first
thing in `Args::run()` (`main.rs:161-162`, implementation `experimental/starrocks/src/tunable.rs:259-284`),
explicitly so a bad knob "fails startup here rather than surfacing as an unexplained timeout
mid-sweep". `[V]`

Treat the Rust half as optional polish. **The C++ half is the deliverable.**

### 5.2 What it should check

```
For each configured GPU memory space g (cucascade::memory::gpu_memory_space_config):
    pool_g       = g.memory_capacity                          # already absolute bytes
    arena_g      = (g is the arena's device) ? arena_bytes : 0
    headroom     = SIRIUS_DEVICE_HEADROOM_BYTES (default 2 GiB)
    allocatable_g = cudaMemGetInfo(g.device_id).free          # measured NOW
    required_g   = pool_g + arena_g + headroom
    if required_g > allocatable_g:  refuse
```

**`allocatable` must be `cudaMemGetInfo`'s `free`, not `total`, and must NOT have the driver
reservation subtracted by hand.** Measured on this box (§1.2): `total` (94.9706 GiB) already excludes
the 638 MiB driver reservation, and `free` additionally accounts for this process's own CUDA context
and for any co-tenant already on the device. `free` is therefore both correct and strictly more
conservative. `[V]` Note that cuCascade already fetches this pair and discards `free`
(`cucascade/src/memory/reservation_manager_configurator.cpp:322`). `[V]`

**Getting `arena_bytes`.** The arena arrives via the env var, not YAML. Read it with the same two
symbols the arena itself uses so there is exactly one parser:
`exchange_staging_arena::kCapacityEnvVar` (`src/include/exec/exchange_staging_arena.hpp:49`) and
`sirius::yaml::parse_bytes` (`src/include/yaml_reader.hpp:86`, used at
`src/exec/exchange_staging_arena.cpp:196`). `[V]` Unset ⇒ `arena_bytes = 0` ⇒ the check degenerates to
`pool + headroom <= free`, which is still worth having.

**Device attribution of the arena.** The arena allocates on device ordinal **0** — the comment at
`src/exec/exchange_staging_arena.cpp:89-92` states the CN pins one GPU via `CUDA_VISIBLE_DEVICES`
so ordinal 0 is the only device it sees. `[V]` Charge the arena to ordinal 0 only, and emit a WARNING
when `num_gpus > 1` **and** the arena env var is set, because that combination is untested and the
attribution is then a guess.

**Fabric arenas round up.** With `SIRIUS_EXCHANGE_STAGING_ARENA=fabric`
(`src/include/exec/exchange_staging_arena.hpp:52`), `cuMemCreate` is given a size rounded **up** to the
allocation granularity (`src/exec/exchange_staging_arena.cpp:108-110`). `[V]` The real footprint can
therefore exceed the env value; the 2 GiB headroom absorbs this, but say so in a comment.

**Headroom default — and its honest calibration problem.** Recommend **2 GiB**, overridable via
`SIRIUS_DEVICE_HEADROOM_BYTES`. The measured CN overhead at `60/32` was 1642 MiB (GPU0) and
**2768 MiB (GPU1)** (§1.2), so 2 GiB is *below* the worst observed value. That is deliberate, and the
reason is a hard constraint:

```
idle allocatable here          94.4255 GiB   (cudaMemGetInfo free, measured)
deployed pool + arena          92.0000 GiB   (60 + 32)
=> the largest headroom that still ACCEPTS the known-good config is 2.42 GiB
```

A 3 GiB headroom would **refuse the config that scores 21/22**. `[V]` So the check cannot be
calibrated to GPU1's observed 2.77 GiB overhead without rejecting a demonstrably working deployment —
which is itself the real finding: **the deployed SF500 config has essentially no margin, and 2 GiB is
the largest round number that does not reject it.** Do not silently raise the default to "be safe";
that converts a working benchmark into a bring-up failure. If a future change reduces per-CN overhead
(PLAN-01/PLAN-02 both would), revisit this number with fresh measurements.

**Known limitation — the concurrent-bring-up race.** Two CNs launched in parallel (which is exactly
what `up-sf500-x.sh` does — the CN loop backgrounds each process) `[V]` will both observe a nearly
empty device and both pass. The check is a **floor, not a guarantee**. Say so in the comment. It still
catches every single-CN misconfiguration and every case where one CN is already up.

### 5.3 Error text

Draft (throw `std::runtime_error` from `initialize`, matching the two existing refusals at
`src/sirius_context.cpp:626-635` and `:794-799`). **The numbers below are illustrative** — they show a
device with a co-tenant already holding ~3.5 GiB. On an *idle* device here the deployed `60/32` config
**passes**: `60 + 32 + 2 = 94 GiB` required vs `94.4255 GiB` free, i.e. 0.43 GiB of slack — tight, but
not refused. `[V]` That is the intended calibration: the check should pass the config that works and
fail the ones that do not.

```
SiriusContext::initialize: GPU 1 is over-subscribed by 3.06 GiB — refusing to start.

  configured RMM pool  (memory.gpu.usage_limit)     64424509440 B  (60.00 GiB)
  exchange staging arena (SIRIUS_EXCHANGE_STAGING_BYTES) 34359738368 B  (32.00 GiB)
  runtime headroom     (SIRIUS_DEVICE_HEADROOM_BYTES)     2147483648 B  ( 2.00 GiB)
  ------------------------------------------------------------------------------
  required                                          100931731456 B  (94.00 GiB)
  allocatable now (cudaMemGetInfo free, GPU 1)       97648115712 B  (90.94 GiB)

The staging arena is a bare cudaMalloc OUTSIDE the RMM pool: it is NOT covered by
memory.gpu.usage_limit, and raising it shrinks the pool 1:1. To fix, either lower
memory.gpu.usage_limit_bytes / usage_limit_fraction, or lower
SIRIUS_EXCHANGE_STAGING_BYTES (env var, not a YAML key), or free the device.
Set SIRIUS_ALLOW_OVERSUBSCRIBED_DEVICE=1 to downgrade this to a warning.
```

Requirements for the message, in priority order:

1. Name the **device ordinal** and the **deficit**, so the operator knows how much to move.
2. Break out all three terms with both bytes and GiB — the whole failure mode is that people do not
   know the arena is outside the pool.
3. Name the **exact knob** for each term, including that `SIRIUS_EXCHANGE_STAGING_BYTES` is an
   environment variable and has no YAML key.
4. State the 1:1 trade-off, so nobody responds to it by raising the arena.
5. Name the escape hatch.

Also emit, on the **success** path, one INFO line per GPU with the same four numbers. Today the
effective split is invisible — `up-sf500-x.sh` prints it, but nothing in the engine confirms it, and
`SF500-CONFIG-AND-ARCHITECTURE.md` records that every earlier SF500 run was "flying blind". `[V-doc]`

### 5.4 Refuse or warn?

**Refuse by default. Warn only under `SIRIUS_ALLOW_OVERSUBSCRIBED_DEVICE=1`.**

For:
* The current failure is an opaque `std::bad_alloc` or a *misleading* `arena exhausted` message,
  arriving minutes into a sweep, after which the CN must be restarted anyway
  (`bench.sh` runs `$RESTART_CMD` on failure — `experimental/starrocks/benchmarks/tpch/bench.sh`). `[V]`
* The codebase already refuses at bring-up for less: zero GPUs
  (`src/sirius_context.cpp:626-635`), a missing HOST memory space (`:794-799`), a bad transport
  tunable (`experimental/starrocks/src/tunable.rs:259`), and a claimed GPU under default config
  (`experimental/starrocks/src/main.rs:353-361`, whose message explicitly says bring-up "would abort
  with the rmm OOM … refusing"). `[V]`

Against (why the escape hatch exists):
* The concurrent-bring-up race (§5.2) can produce a false negative in the other direction if a
  co-tenant is transiently holding memory.
* 2 GiB of headroom is a heuristic, and someone with a genuinely leaner build should be able to
  override it — though `SIRIUS_DEVICE_HEADROOM_BYTES` already covers that case more precisely than the
  blanket override.

---

## 6. Backward compatibility

### 6.1 Who is affected

A config is affected **only** if it omits an `operator_params` batch key **and** its GPU pool differs
from `card/40 × 40 = card` — i.e. any `usage_limit` below 1.0, which is every config.

| consumer | sets `operator_params`? | pool | today | after (`/40`) | risk |
|---|---|---|---|---|---|
| `bench/sf1000-repro/sirius-sf1000.yaml` | **yes** (all five) `[V]` | 243 GB | — | **unchanged** | none |
| `bench/a100x8/engine-a-sirius.yaml` | **yes** (all five) `[V]` | 68 GiB | — | **unchanged** | none |
| `/opt/dlami/nvme/sirius-build/up-sf500-x.sh` with `HPB/MBHT/STB/CBB` set | yes `[V]` | 60 GiB | — | **unchanged** | none |
| `up-sf500-x.sh` with those env vars unset | **no** `[V]` | 60 GiB | 2.3743 GiB | 1.500 GiB | **intended fix** |
| CN `--gpu-memory-limit 8GiB` (`pixi.toml:77,83`) on this box | **no** (derived YAML has none) `[V]` | 8 GiB | 2.3743 GiB (29.7 % of pool) | 512 MiB (floor) | **intended fix**, 4.6× change — re-run the two-CN demo |
| same on the 23 GiB L4 the demo targets | no | 8 GiB | ~590 MiB | 512 MiB | 1.15× — negligible |
| `pixi run cluster` / any `LOAD` with no `sirius.yaml` | **no** — `apply_defaults()` `[V]` | 0.95 × card | `card/40` | `0.95 × card/40` | **−5 %**, the whole point of choosing /40 |
| `test/cpp/**/*.yaml` fixtures (5 files) | **yes**, all `[V]` | — | — | **unchanged** | none |
| operator unit tests constructing operators directly | n/a — use `config::DEFAULT_*` (800 MiB) `[V]` | — | — | **unchanged** | none |

### 6.2 Is this a silent perf regression anywhere?

**The one real exposure is `max_build_hash_table_bytes` gating BUILD_PROBE**
(`src/op/sirius_physical_hash_join.cpp:769`). `[V]` A smaller value pushes joins from BUILD_PROBE onto
the STANDARD path. On the no-YAML default path the change is only −5 % (4.7486 → 4.512 GiB), which
cannot plausibly flip many decisions. On carve-outs the change is large — but on carve-outs the
current value (59.4 % of an 8 GiB pool) is not a real budget, it is an accident.

**Nothing in `bench/` regresses**, because every benchmark config in the tree sets `operator_params`
explicitly (verified above). The regression surface is exactly "users who never configured", and for
them `/40` is a 5 % move.

**Documentation must change** or it becomes wrong:
`docs/super-sirius/configuration.md:356-373` (the "Operator Parameters" section, which states
"2.5 % of the smallest visible GPU's memory") plus the repeat rows in the DuckDB-settings tables at
`docs/super-sirius/configuration.md:539` (Scan) and `:563-565` (the batch/partition rows). `[V]`
Also the doc comment at `src/include/sirius_config.hpp:39-42` and the inline `// 2.5%` at
`src/sirius_config.cpp:55`. `[V]`

---

## 7. Tests

### 7.1 New unit tests — the derivation (pure, no GPU)

Add to `test/cpp/config/test_config.cpp` (already in `TEST_SOURCES`,
`/home/ubuntu/sirius/CMakeLists.txt:631`) `[V]`, tag `[config_opt][batch_size]`:

| case | assertion |
|---|---|
| zero pool | `batch_size_for_pool(0) == config::DEFAULT_BATCH_SIZE` (800 MiB) |
| floor binds | `batch_size_for_pool(8 GiB) == 512 MiB` |
| divisor binds | `batch_size_for_pool(60 GiB) == 60 GiB / 40` (1.5 GiB exactly) |
| ceiling binds | `batch_size_for_pool(400 GiB) == 5 GiB` |
| monotone | `batch_size_for_pool(a) <= batch_size_for_pool(b)` for a sampled ladder `a < b` |
| `constexpr` | `static_assert(batch_size_for_pool(60ULL<<30) == (60ULL<<30)/40);` — proves it is pure |

### 7.2 New unit tests — the resolver (config-level, no GPU needed if you construct configs by hand)

Add to `test/cpp/config/test_context.cpp`, tag `[sirius][config]`, with new fixtures under
`test/cpp/config/data/`:

| case | fixture | assertion |
|---|---|---|
| all budgets derived | new `pool_derived_defaults.yaml` with `memory.gpu.usage_limit_bytes: 60GiB` and **no** `operator_params` | after `load_from_file`, all four == `batch_size_for_pool(60 GiB)` and `max_build_hash_table_bytes == 2 ×` that |
| explicit wins, per key | new `pool_partial_operator_params.yaml` setting **only** `hash_partition_bytes: 3MiB` | `hash_partition_bytes == 3 MiB`; the other four are pool-derived |
| existing behaviour preserved | existing `data/setting_defaults.yaml` | `test_context.cpp:243-352` passes **unmodified** |
| `apply_defaults` path | none (call `sirius_config c; c.apply_defaults();`) | budgets equal `batch_size_for_pool(min gpu memory_capacity)` |
| zero-guard intact | existing `data/invalid_hash_partition_zero.yaml` | still throws (`test_context.cpp:215-226`) |

### 7.3 New unit tests — the validator

Add `test/cpp/config/test_device_budget.cpp`, register it in `TEST_SOURCES`
(`/home/ubuntu/sirius/CMakeLists.txt`, next to `test/cpp/config/test_context.cpp:632`). `[V]`
Factor the arithmetic into a pure helper so the table cases need no GPU:

```cpp
struct device_budget { uint64_t pool; uint64_t arena; uint64_t headroom; uint64_t allocatable; };
[[nodiscard]] std::optional<uint64_t> device_budget_deficit(device_budget const&);  // nullopt = fits
```

| case | assertion |
|---|---|
| exact fit | `deficit({60 GiB, 32 GiB, 2 GiB, 94 GiB}) == nullopt` |
| one byte over | `deficit({60 GiB, 32 GiB, 2 GiB, 94 GiB - 1}) == 1` |
| no arena | `deficit({90 GiB, 0, 2 GiB, 94 GiB}) == nullopt` |
| arena dominates | the SF500 `60/32` shape against this box's measured `free` — expect a deficit |
| message | the thrown text contains `"over-subscribed"`, `"SIRIUS_EXCHANGE_STAGING_BYTES"`, `"usage_limit"`, and the deficit in bytes |
| escape hatch | with `SIRIUS_ALLOW_OVERSUBSCRIBED_DEVICE=1`, `initialize` succeeds and logs a WARNING (use the `finally cleanup_env{...}` + `setenv` idiom already used at `test/cpp/config/test_context.cpp:245-256`) `[V]` |

Env-mutating cases must follow the existing `[isolated_context]` tag convention in
`test_context.cpp`. `[V]`

### 7.4 Full C++ suite

```bash
pixi run make test
```

### 7.5 SF500 regression — the acceptance run

Requires the 2-CN cluster on this box. **`bench.sh` has NO correctness gate** — its own header says
"this script times and counts rows only — it does not check answers"
(`experimental/starrocks/benchmarks/tpch/bench.sh:54-56`). `[V]` **You must diff against the oracle
separately.**

```bash
# 1. Rebuild the CN with the change.
cd /home/ubuntu/sirius && pixi run make
cd /home/ubuntu/sirius/experimental/starrocks && cargo build --release

# 2. Bring the cluster up WITHOUT operator_params overrides, so the new derived
#    defaults are what is under test. (HPB/MBHT/STB/CBB deliberately unset.)
GPU_MEM=60GiB STAGING=32GiB HOST_MEM=200GiB \
  /opt/dlami/nvme/sirius-build/restart-sf500x.sh

# 3. Confirm in the CN log that the derived budgets are what you expect
#    (this line does not exist yet — adding it is part of §3.4 step 3).
grep -i "operator batch defaults" /opt/dlami/nvme/sirius-build/siriuslog/*.log

# 4. Full 22, fresh cluster per query.
GPU_MEM=60GiB STAGING=32GiB HOST_MEM=200GiB \
  OUT=/opt/dlami/nvme/sirius-build/bench/PLAN03/timings.csv \
  /opt/dlami/nvme/sirius-build/sweep-sf500x-cold.sh

# 5. CORRECTNESS — mandatory. The .out files live in dirname(OUT).
python3 /opt/dlami/nvme/sirius-build/compare.py \
  /opt/dlami/nvme/sirius-build/bench/PLAN03 \
  /opt/dlami/nvme/sirius-build/oracle-sf500f64
```

`compare.py` usage is `compare.py <sirius_out_dir> <oracle_dir> [rel_tol]`, default tolerance `1e-6`;
it prints one verdict per query (`MATCH` / `VALUES-DIFFER` / `ROWS-DIFFER` / `ERROR` / `EMPTY` /
`NO-ORACLE`) and a final `N/22 match` line. `[V]` `bench.sh` sets `OUT=$(dirname "$OUT_CSV")` at
`experimental/starrocks/benchmarks/tpch/bench.sh:86` and writes `$OUT/<q>.r<N>.out`. `[V]`
The oracle directory `/opt/dlami/nvme/sirius-build/oracle-sf500f64` exists and holds `q01.tsv`…. `[V]`

**Expected:** the same 21/22 the hand-tuned config achieved
(`results/sf500xcold.csv` + `results/sf500e5.csv`) — **without any `operator_params` in the YAML.**
That is the whole point of the change. q09 will still fail; that is PLAN-01's subject and is out of
scope here. q11 will still record as `wedge` (a correct empty answer mis-filed by `bench.sh:175`'s
`[ -s "$f" ]` test); that is PLAN-05's subject. `[V-doc]`

Known confounders when reading the result: q21 is intermittent (§1.1), and q08/q09 currently rely on
hand-reordered `FROM` clauses (commit `7af763c0`) `[V]`.

### 7.6 Validator smoke test on real hardware

```bash
# Deliberately over-subscribe: 90 GiB pool + 32 GiB arena on a 94.97 GiB device.
GPU_MEM=90GiB STAGING=32GiB HOST_MEM=200GiB \
  /opt/dlami/nvme/sirius-build/restart-sf500x.sh
# EXPECT: the CN exits at bring-up with the §5.3 message naming a ~29 GiB deficit,
# BEFORE any query runs. Today it comes up and dies later with bad_alloc.
```

### 7.7 Rust tests (only if the optional §5.1 secondary check is implemented)

```bash
cd /home/ubuntu/sirius/experimental/starrocks && cargo test
```
Note `experimental/starrocks/src/main.rs:758-790` already contains clap conflict tests
(`gpu_memory_limit_conflicts_with_fraction`, `sirius_config_conflicts_with_each_memory_flag`,
`sirius_config_composes_with_gpu_device`) `[V]` — extend that module, do not create a new one.

### 7.8 Formatting

```bash
pixi run pre-commit run -a
```

---

## 8. Success criteria

1. `derived_default_batch_size()` no longer reads `prop.totalGlobalMem`; the batch default is a pure
   function of the configured pool, computed after config parse. No CUDA call remains in any
   `operator_params` in-class initializer. `[must]`
2. An explicit `operator_params` key in YAML always wins over the derived value, proven by
   `test/cpp/config/test_context.cpp:243-352` passing **unmodified**. `[must]`
3. `pixi run make test` is green. `[must]`
4. Experiment E1 (§4.3) has been run and its outcome recorded in this file, and the shipped divisor
   matches it. `[must]`
5. The SF500 22-query cold sweep with **no `operator_params` in the YAML** reaches the same verdict as
   the hand-tuned `HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB` config, verified with `compare.py` against
   `oracle-sf500f64` — specifically **q08 passes and matches the oracle**. `[must]`
6. A deliberately over-subscribed config (`GPU_MEM=90GiB STAGING=32GiB`) is **refused at bring-up**
   with a message naming the device, the deficit, all three terms and both knobs — before any query
   runs. `[must]`
7. Every successful bring-up logs one INFO line per GPU with pool / arena / headroom / allocatable,
   and one INFO line naming the derived batch budgets and which were YAML-overridden. `[must]`
8. `docs/super-sirius/configuration.md:356-373`, `:539` and `:563-565` describe the new rule. `[must]`
9. No file under `src/legacy/` is touched. `[must]`

---

## 9. Risks

| # | risk | likelihood | mitigation |
|---|---|---|---|
| R1 | `/40` at 60 GiB (1.5 GiB) still OOMs q08 — the untested gap | medium | E1 is a merge gate; fall back to the verified `/60` |
| R2 | Smaller `max_build_hash_table_bytes` demotes joins from BUILD_PROBE and costs throughput on the no-YAML path | low (−5 % on that path) | `test/tpch_performance` before/after; keep the `2×` relation; the knob is still individually settable |
| R3 | The `explicit` flag plumbing is missed for one key, so a YAML value gets silently overwritten | medium — this is the most likely implementation bug | one test per key (§7.2 case 2); keep the flags adjacent to the parse in `from_yaml` so adding a key without a flag is visually obvious |
| R4 | Reported `current_setting()` defaults change, breaking a downstream expectation | low | no test asserts the derived value (§3.5); documented in the PR |
| R5 | The validator refuses a config that actually worked, blocking a benchmark mid-campaign | medium | `SIRIUS_DEVICE_HEADROOM_BYTES` to tune, `SIRIUS_ALLOW_OVERSUBSCRIBED_DEVICE=1` to bypass; both named in the error text |
| R6 | Concurrent CN bring-up races the check, so both pass and one later OOMs (§5.2) | high, by design | documented as a floor not a guarantee; unchanged from today's zero-check baseline, so never worse |
| R7 | `cudaMemGetInfo` in `initialize` creates a CUDA context ~550 MiB earlier than before | certain, benign | the engine creates one microseconds later anyway (`memory_manager_` at `src/sirius_context.cpp:641`); the extra cost is **zero** |
| R8 | Arena charged to the wrong device in a multi-GPU single-process config | low (untested combination) | charge ordinal 0 per `exchange_staging_arena.cpp:89-92`; WARN when `num_gpus > 1` and the arena env var is set |
| R9 | A fabric arena's granularity round-up (`exchange_staging_arena.cpp:108-110`) makes the real footprint exceed the checked value | low | 2 GiB headroom absorbs it; comment it |
| R10 | Someone "fixes" an `arena exhausted` error by raising the arena, shrinking the pool | high — this already happened | the §5.3 message states the 1:1 trade-off explicitly; the diagnostic table in `SF500-CONFIG-AND-ARCHITECTURE.md` covers the triage |

---

## 10. Appendix — how the device measurements in this plan were taken

Reproducible in ~30 s. Compile a bare cudart probe against the repo's pixi environment:

```bash
E=/home/ubuntu/sirius/.pixi/envs/default
g++ -I$E/targets/x86_64-linux/include -o /tmp/probe /tmp/probe.cpp \
    -L$E/targets/x86_64-linux/lib -lcudart -Wl,-rpath,$E/targets/x86_64-linux/lib
/tmp/probe
```

`/tmp/probe.cpp` — reproduces `derived_default_batch_size()`'s arithmetic exactly and then reports
`cudaMemGetInfo`:

```cpp
#include <cuda_runtime_api.h>
#include <cstdio>
#include <algorithm>
int main(){
  int n=0; cudaGetDeviceCount(&n);
  printf("device_count=%d\n", n);
  unsigned long long minTotal=0;
  for(int i=0;i<n;i++){
    cudaDeviceProp p{}; cudaGetDeviceProperties(&p,i);
    unsigned long long t=(unsigned long long)p.totalGlobalMem;
    printf("gpu%d totalGlobalMem=%llu (%.4f GiB)\n", i, t, t/1073741824.0);
    minTotal = minTotal==0? t : std::min(minTotal,t);
  }
  unsigned long long batch = minTotal/40ULL;
  unsigned long long lo=512ULL<<20, hi=5ULL<<30;
  unsigned long long clamped = std::max(lo, std::min(hi, batch));
  printf("min_total/40=%llu (%.4f GiB) clamped=%llu (%.4f GiB)\n",
         batch, batch/1073741824.0, clamped, clamped/1073741824.0);
  for(int i=0;i<n;i++){
    cudaSetDevice(i);
    size_t f=0,t=0; cudaMemGetInfo(&f,&t);
    printf("gpu%d cudaMemGetInfo free=%zu (%.4f GiB) total=%zu (%.4f GiB)\n",
           i, f, f/1073741824.0, t, t/1073741824.0);
  }
  return 0;
}
```

Output on this box, 2026-08-20, idle (no cluster running):

```
device_count=2
gpu0 totalGlobalMem=101973950464 (94.9706 GiB)
gpu1 totalGlobalMem=101973950464 (94.9706 GiB)
min_total/40=2549348761 (2.3743 GiB) clamped=2549348761 (2.3743 GiB)
gpu0 cudaMemGetInfo free=101388648448 (94.4255 GiB) total=101973950464 (94.9706 GiB)
gpu1 cudaMemGetInfo free=101388648448 (94.4255 GiB) total=101973950464 (94.9706 GiB)
```

To reproduce the "no context from `cudaGetDeviceProperties`, 550 MiB context from `cudaMemGetInfo`"
finding, interleave `system("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i 0")`
between the two calls. Observed: **no compute app / 3 MiB used** after
`cudaGetDeviceCount` + `cudaGetDeviceProperties`; **`<pid>, 550 MiB` / 559 MiB used** after the first
`cudaMemGetInfo`.

Device totals for cross-checking:

```
$ nvidia-smi -q -d MEMORY -i 0
    FB Memory Usage
        Total    : 97887 MiB     # = 102,641,795,072 B
        Reserved :   638 MiB     # = 668,991,488 B  (driver)
        Used     :     0 MiB
        Free     : 97250 MiB
```

`102,641,795,072 − 668,991,488 = 101,972,803,584` ≈ `totalGlobalMem = 101,973,950,464`
(the 1.1 MiB residue is MiB rounding in `nvidia-smi`'s report). **`totalGlobalMem` is already the
post-driver-reservation figure.**
