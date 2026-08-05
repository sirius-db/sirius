# WS9 — New telemetry fields & events (parser handoff)

**Status:** implemented 2026-08-04; **built and trace-verified 2026-08-05** on the post-merge
binary (pr-1409 late-mat merge). Demo trace: TPC-H q6/q10 ×2 at SF10, fresh reads, gates dark —
22/22 io_request FSMs closed with non-nil task uuids and sane bytes/rows; 72/72
`Constructed` events carried the new fields (16 `num_rows==0` from the documented unknown
producers); 64/64 `BatchRegistered` with `producer_task_uuid`; 68/68 successful `Finalizing`
with `output_rows>0`; all 37 Init custom_attribute keys present. `python -m hwsim info`
parses the new traces (io_request dir ignored, as designed). Field names come from the model
source `rust/crates/telemetry/model/src/`; serialized shapes below are demo-trace-verified.

**Compatibility contract:** everything is additive. Old traces lack the new fields/entity and
must keep parsing (treat every new field as optional with the stated default). New traces add
fields *inside existing state payload maps* and one new entity directory. Nil uuid =
`"00000000-0000-0000-0000-000000000000"`.

Envelope, seq/ordering, and usage encoding are unchanged — see `quent-extraction.md` §(a).

---

## 1. `data_batch` — `Constructed` state, 3 new fields (G2, G3)

| field | type | default (old traces) | semantics |
|---|---|---|---|
| `producer_task_uuid` | uuid string | absent → treat as nil | The task whose execution constructed this batch. **Nil when:** scan-manager staging/pinned-cache batches, batches created off the executor thread, unit tests. |
| `num_rows` | u64 | absent → 0 | Rows in the batch's table. **0 = unknown** (undecoded host staging batches; a handful of direct-probe producers: limit, top-n, result-collector, ungrouped-aggregate, table-scan pin path, grouped-aggregate-merge clone). |
| `num_columns` | u64 | absent → 0 | Column count; same 0 = unknown rule. |

Example (new trace):

```json
{"id":"019f...","timestamp":1785548587401110237,"data":{"seq":0,"state":{"Constructed":{"instance_name":"batch","data_batch_id":42,"producer_pipeline_uuid":"019f...","producer_task_uuid":"019f...","num_rows":131072,"num_columns":8}}}}
```

**Simulator impact:** `producer_task_uuid` replaces the (pipeline, time-window) producer-task
inference — the v0 heuristic in `quent-extraction.md` §(d) "Task-level dependencies" is only
needed for old traces and for nil cases (scan batches have no producer task by design; their
inputs come from the scan manager).

## 2. `batch_placement` — `BatchRegistered` state, 1 new field (G2)

| field | type | default | semantics |
|---|---|---|---|
| `producer_task_uuid` | uuid string | absent → nil | Task that published the batch to this consumer port. **Nil when** `origin == "reschedule_intermediate"` (lazy registration at claim; producer unknown) or published outside a task. Emitted for both `operator_output` and `partition_output` origins. |

Field sits between `origin` and the `tier` usage:

```json
{"state":{"BatchRegistered":{"instance_name":"batch-42","batch_id":42,"pipeline_uuid":"019f...","port_uuid":"019f...","origin":"operator_output","producer_task_uuid":"019f...","tier":{"resource_id":"019f...","capacity":{"capacity_bytes":498694030}}}}}
```

## 3. `task` — `Computing` + `Finalizing`, 3 new fields (G3)

### `Computing.input_rows` (u64, default 0)

Total rows across the operator's input batches, computed at the same instant as the existing
`input_bytes`. Since operator *i*'s input is operator *i-1*'s output, per-operator output rows =
next `Computing.input_rows` within the same task (exactly the existing `input_bytes`
convention). **0 =** non-pipelineable input — notably the first `Computing` (`GPU_SCAN`) of a
fresh-read scan task, whose row count is unknown before decode (use the task's `io_request`
`Completed.rows` instead, §5).

### `Finalizing.output_rows`, `Finalizing.output_bytes` (u64, default 0)

The task's final output (the last operator's output, measured before publish). Closes the
"last operator's output volume" hole. **Both 0 when:** `success == false` (all failure paths),
the task produced no batch output, or old traces.

```json
{"state":{"Computing":{"instance_name":"HASH_GROUP_BY(3)","current_operator_id":3,"input_bytes":40894464,"peak_allocated_bytes":121634816,"input_rows":1277952,"executor_thread":{...},"reservation":{...}}}}
{"state":{"Finalizing":{"instance_name":"","success":true,"output_rows":4,"output_bytes":224}}}
```

## 4. `engine` — `Init.implementation.custom_attributes` now populated (G6)

Previously empty — **parse defensively**: keys may be absent (unit-test contexts still emit
empty), and the key set may grow. **Serialized ndjson shape (verified against the 2026-08-05
demo trace, session `019fcfff-48d4-...`):** ONE flat array of tagged values, not the model
source's three parallel arrays:

```json
{"data":{"Init":{"implementation":{"name":"siriusDB","version":null,"custom_attributes":[
  {"key":"host.name","value":{"String":"pmgb300ws-0163"}},
  {"key":"hw.num_gpus","value":{"I64":1}},
  {"key":"scan_manager.cache.eviction_threshold_fraction","value":{"F64":0.6}}]}}}}
```

Note the engine `Init` event also has no `seq`/`state` wrapper — `Init` sits directly under
`data` (pre-existing engine-event shape, unchanged by WS9).

Keys emitted (one-time, per session):

| kind | keys |
|---|---|
| string | `host.name`; `gpu.<id>.name`; `scan_manager.io_backend` (`"uring"`\|`"kvikio"`); `late_mat.pin_unique_cols` (raw `SIRIUS_LATE_MAT_PIN_UNIQUE_COLS` value; omitted when unset/empty) |
| i64 | `hw.num_gpus`, `hw.num_numa_nodes`, `hw.host_cores`; `gpu.<id>.numa_node`, `gpu.<id>.sm_count`, `gpu.<id>.sm_clock_khz`, `gpu.<id>.mem_clock_khz`, `gpu.<id>.mem_bus_width_bits` (each `gpu.<id>.*` hw attr omitted if the CUDA attribute query fails); `memory.gpu<dev>.capacity_bytes`, `memory.gpu<dev>.reservation_limit_bytes`, `memory.host<numa>.capacity_bytes`, `memory.host<numa>.reservation_limit_bytes`, `memory.disk<id>.capacity_bytes`; `executor.num_threads`, `task_creator.num_threads`, `downgrade.num_threads`, `downgrade.monitor_period_ms`; `scan_manager.num_threads`, `scan_manager.uring_n_reactors`, `scan_manager.rest_n_reactors`, `scan_manager.prefetch_cache_enabled` (0/1), `scan_manager.memory_prefetcher.enabled` (0/1), `scan_manager.memory_prefetcher.num_threads`, `scan_manager.cache.inflight_io_chunk_budget`; `operator.scan_task_batch_size`, `operator.hash_partition_bytes`; `telemetry.batch_events` (0/1); `late_mat.enabled`, `late_mat.v2`, `late_mat.v3`, `late_mat.defer`, `late_mat.compressed`, `fused_scan_filter.enabled` (all 0/1) |
| f64 | `scan_manager.cache.min_prefetching_budget_fraction`, `scan_manager.cache.eviction_threshold_fraction` |

Notes: `<id>`/`<dev>`/`<numa>` are the numeric device / NUMA ids. `hw.host_cores` is
`std::thread::hardware_concurrency()`. Dataset identity is still not in-trace.

The `late_mat.*` / `fused_scan_filter.enabled` keys snapshot the PR #1409 experimental
env gates (added post-merge; absent on traces from earlier binaries). Values are
EFFECTIVE, not raw env: sub-gates imply their parents (`v3` ⇒ `v2` ⇒ `enabled`),
`late_mat.defer` defaults ON under the main gate, and `late_mat.compressed` defaults
OFF — matching the in-engine readers (`src/include/late_mat/column_origin.hpp`,
`src/scan_manager/late_mat_defer_policy.cpp`). A trace with `late_mat.enabled=1` came
from the compute-bound late-mat lane (see `tools/hwsim/docs/late-mat-lane.md`).

## 5. New entity: `io_request` (G1) — scan-split disk I/O

New session subdirectory `io_request/<stream-uuid>.ndjson` — **old parsers must ignore unknown
entity directories**. FSM: `Issued` → `Completed` → `Exit` (`"state":"Exit"` terminal line,
same as every FSM). One instance per **fresh-read** split materialization
(`gpu_ingestible::materialize_table` inside the owning task's `GPU_SCAN` `Computing` span);
resident/cached splits emit nothing; the table-pinning path emits nothing.

### `Issued` (timestamp = materialize start)

| field | type | semantics |
|---|---|---|
| `instance_name` | string | `"scan_split"` |
| `task_uuid` | uuid | Owning task (join key to the `task` FSM). Nil only outside task execution. |
| `pipeline_uuid` | uuid | Owning pipeline (== quent Operator id). |
| `file_count` | u64 | Sirius datasources this split reads through (parquet: row-group slices, possibly multiple files; duckdb-native: 1). 0 = split reads through plain/non-sirius datasources only. |
| `estimated_compressed_bytes` | u64 | Expected on-disk bytes from scan metadata (parquet: summed `reserved_compressed_bytes`; duckdb-native: 0 — not tracked). |
| `estimated_decoded_bytes` | u64 | The split's decoded-output estimate (`scan_info::estimated_bytes()`). |

### `Completed` (timestamp = materialize end) → then `Exit`

| field | type | semantics |
|---|---|---|
| `bytes_read` | u64 | Measured bytes through the split's `sirius_datasource`s during materialize (snapshot diff). 0 if the split used plain datasources (rare local-path fallback). |
| `read_time_ns` | u64 | Sum of per-read-call spans (sync call time; async issue→settle). **Span sum, not critical path** — concurrent reads can make it exceed `Completed.ts − Issued.ts`. Cache hits count at cache-copy (memcpy) speed. |
| `read_calls` | u64 | Number of read calls measured. |
| `rows` | u64 | Rows in the materialized table (post reader-side filter pushdown). 0 on the materialize-exception path (an exception still closes the FSM with zeroed stats before rethrow). |

### Derivations

- **Split materialize wall** = `Completed.ts − Issued.ts` (a lower-bound sub-span of the task's
  `GPU_SCAN` Computing span — masking/normalization/post-filter work sits outside it).
- **Read vs decode (first order):** read ≈ `read_time_ns`, decode ≈ span − `read_time_ns`
  (clamp at 0: async overlap can make the subtraction negative — treat as fully overlapped).
- **Effective storage bandwidth** = `bytes_read / read_time_ns`.
- **`io_bandwidth` knob:** scale `read_time_ns` and rebuild the split span as
  `max(decode, scaled_read)`..`decode + scaled_read` depending on your overlap assumption.

Example lines (schematic — verify against the demo trace):

```json
{"id":"<io-req-uuid>","timestamp":T0,"data":{"seq":0,"state":{"Issued":{"instance_name":"scan_split","task_uuid":"019f...","pipeline_uuid":"019f...","file_count":2,"estimated_compressed_bytes":221249536,"estimated_decoded_bytes":536870912}}}}
{"id":"<io-req-uuid>","timestamp":T1,"data":{"seq":1,"state":{"Completed":{"instance_name":"","bytes_read":221249536,"read_time_ns":183000000,"read_calls":64,"rows":6291456}}}}
{"id":"<io-req-uuid>","timestamp":T1,"data":{"seq":2,"state":"Exit"}}
```

## 6. Volume impact

- `io_request`: 3 lines per fresh-read split — O(hundreds)/query, ~0.01% of event volume.
- Everything else: fields on existing events; no volume change.

## 7. Gotchas for the parser

1. `model.qmi` records the Sirius model commit — it changes with this schema. Keep the old-trace
   code path keyed on absence of the new fields rather than on the commit if you want one parser
   for both.
2. `Finalizing.output_*` is only meaningful on `success:true`; OOM-rescheduled tasks report 0
   and their real output shows up on the replacement task.
3. `Computing.input_rows` uses uncompressed-bytes semantics for `input_bytes` unchanged; rows
   and bytes are sampled at the same point.
4. A scan task's per-batch `Constructed.producer_task_uuid` IS set (the output batch is built by
   the scan task) even though its `io_request` inputs have no upstream task.
5. The `sirius-telemetry-analyzer` Rust crate accepts and ignores `io_request` events
   (`rust/crates/telemetry/analyzer/src/model.rs`).
