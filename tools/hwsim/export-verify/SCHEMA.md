# Quent session schema contract (WS18 — export verification)

The contract WS17's `--export-quent` output is validated against. Derived from the
**authoritative deserialization types** — the Rust analyzer stack — and cross-checked against
real traces. Sources, in priority order:

1. `rust/crates/telemetry/model/src/*.rs` (the `model!`/`fsm!`/`state!`/`resource!`
   definitions — Sirius-specific entities) and the pinned quent crates at rev `2a5ca834`
   (`~/.cargo/git/checkouts/quent-515d44f958e14372/2a5ca83/`): `domains/query_engine/model/src/`
   (Engine/Worker/QueryGroup/Query/Plan/Operator/Port), `crates/stdlib/src/` (Memory/Channel),
   `crates/events/src/lib.rs` (envelope), `crates/io/ndjson/src/lib.rs` +
   `crates/io/src/filesystem/` + `crates/io/types/src/lib.rs` (import behavior),
   `crates/dynamic-attributes/src/lib.rs` (custom_attributes encoding).
2. `rust/crates/telemetry/analyzer/src/model.rs` (event → model mapping; what the analyzer
   actually requires to build a resource tree) and
   `domains/query_engine/server/src/analyzer_cache.rs` (session discovery).
3. Real traces (calibration):
   `tools/hwsim/experiments/traces/LB/019fd052-29d6-7f30-bc37-614f7afda3a4/` (current model,
   commit `a228b890`, WS9 fields present),
   `telemetry_data/019fd006-2d8f-7430-ada0-1a9c0d87d584/` (worktree; non-empty `io_request`),
   `tools/hwsim/experiments/traces/E2-lo-q9/019fcf46-.../` (pressured; `Downgrading`,
   `InTransit`; **pre-WS9 model** commit `b77aa438`).

Everything below was verified against real trace lines, not just the type source.

---

## 1. Session-directory layout

```
<output_dir>/<session-uuid>/          # dir name MUST parse as a UUID (see discovery rules)
  model.qmi                           # JSON sidecar: quent + model source provenance
  <entity_name>/<uuid>.ndjson         # exactly ONE .ndjson file per entity subdir
```

Entity names (= `EntityEvent::NAME`, = subdir names). 19 entity types in the current model
(`model!` block in `rust/crates/telemetry/model/src/lib.rs`):

| kind | entities |
|---|---|
| plain (event-style) | `engine`, `worker`, `query_group`, `gpu_device`, `thread_group`, `plan`, `operator`, `port` |
| FSM (seq/state-style) | `query`, `task`, `data_batch`, `batch_placement`, `io_request` |
| resource FSM (Init/Operating/Finalizing/Exit) | `task_queue`, `executor_thread`, `task_manager_loop_thread`, `memory`, `memory_tier`, `channel` |

**Hard importer rules** (from `quent-io` at the pinned rev — these are the sharp edges):

- `Sirius::import_events(dir)` iterates the model's entity list; for each name it imports
  `dir/<name>/` **only if that subdir exists**. Missing subdirs are fine; **unknown extra
  subdirs are ignored** entirely.
- `resolve_import_path` picks the **first** `*.ndjson` file it finds in the subdir
  (readdir order, unspecified). More than one file per entity dir ⇒ silent data loss.
- On the first line that fails JSON/serde deserialization the importer logs
  (`tracing::error`, usually invisible) and **returns None — silently truncating the rest of
  that entity's stream**. A malformed line does not fail ingestion; it silently drops data.
  This is why line-level validation must happen outside the analyzer (this rig).
- Serde structs have **no `#[serde(default)]`**: every field of the current model is
  **required**. A missing field (e.g. WS9's `input_rows` absent in old traces) is a
  deserialization error ⇒ stream truncation. Unknown extra JSON keys are tolerated
  (serde default: no `deny_unknown_fields`).
- Format is auto-detected per session from the first recognized file extension in any entity
  subdir (`ndjson` | `msgpack` | `postcard`); don't mix formats in one session.
- An empty `.ndjson` file is legal (observed: `io_request` with 0 lines on a fully-cached run).

**Session discovery** (two consumers, different rules):

- `sirius-telemetry-server` (what `pixi run quent <dir>` runs; see `pixi.toml [tasks.quent]`
  and `rust/crates/telemetry/server/src/main.rs`): scans the *direct children* of
  `--output-dir`; a child is a session iff its **directory name parses as a UUID** and its
  format is detectable. `model.qmi` is NOT required by this server. Engine listing:
  `GET http://localhost:8080/api/engines` (add `?with_metadata=true` for names); other routes
  under `/api/engines/{engine_id}/...`.
- `quent open` (upstream viewer): recursively discovers dirs containing a `model.qmi` sidecar.
  So a spec-complete export ships `model.qmi` anyway.

`model.qmi` (observed shape): `{"quent": {version, commit, branch, remote, built_at},
"model": {"name": "Sirius", "package": "instrumentation-model", "type_path":
"instrumentation_model::Sirius", "source": {version, commit, ...}, "analyzer_package":
"sirius-telemetry-analyzer"}}`. The analyzer itself never reads it, but `quent open` builds
the viewer from it.

## 2. Event envelope (every ndjson line)

```json
{"id":"<uuid of the entity instance>","timestamp":<u64 unix-nanoseconds>,"data":<payload>}
```

- `id`: UUID string; real emitters use UUIDv7 (time-ordered). One FSM instance = one `id`.
- `timestamp`: u64 Unix **nanoseconds**. Monotonic within a session (single process anchor);
  never compare across sessions.
- Two `data` payload shapes:
  - **plain**: `{"EventName": {…fields}}` or `{"EventName": null}` (unit events: `Exit`).
  - **FSM**: `{"seq": N, "state": {"StateName": {…}}}`; terminal line
    `{"seq": N, "state": "Exit"}` (a JSON *string*, not an object).

## 3. FSM mechanics (seq / ordering / transitions)

- Per `id`: `seq` starts at 0, is **contiguous** (0,1,2,…), and the event timestamps are
  non-decreasing in seq order.
- Lines within a file are **not** globally timestamp-sorted (multi-threaded emission);
  consumers must order per-id by `seq`. Interleaving of different ids in one file is normal.
- `seq==0` must be the FSM's **entry state**; the terminal `"state":"Exit"` (when present)
  must be the last seq, immediately after a state in the FSM's `exit_from` set.
- A missing terminal `Exit` (process killed mid-run) is degraded data, not a hard schema
  violation — the analyzer still builds spans. The validator reports it as a warning.

State payload composition (from the `state!` macro):

- A state declared with an `attributes:` block auto-gains a **required**
  `instance_name: String` field (first event's instance_name names the FSM instance, later
  states carry `""`). States without attributes have **no** `instance_name`.
- Each usage in `usages:` serializes as a named field:
  `"<usage_name>": {"resource_id": "<uuid>", "capacity": {"capacity_bytes": B} |
  {"capacity_entries": E} | null}` (`null` for unit resources such as threads).
- A state with neither attributes nor usages serializes as `{}` (e.g. `{"Planning":{}}`,
  `{"Destructed":{}}`) — except macro-generated *resource* states with no payload, which
  serialize as `null` (e.g. `{"ExecutorThreadOperating":null}`, `{"MemoryFinalizing":null}`).

## 4. Per-entity contract

Notation: `→X` = uuid must reference an entity of type X in the same session. `nil` =
`00000000-0000-0000-0000-000000000000`. u64/u32/i64/bool are JSON numbers/bools.

### 4.1 Plain entities

| entity | event | required fields |
|---|---|---|
| `engine` | `Init` | `implementation{name: str\|null, version: str\|null, custom_attributes: [ {key: str, value: {String\|I64\|F64\|U64\|…: v} \| null} ]}`, `instance_name: str\|null` |
| `engine` | `Exit` | `null` payload |
| `worker` | `Init` | `parent_engine_id →engine`, `instance_name: str` |
| `worker` | `Exit` | `null` payload |
| `query_group` | `Declaration` | `instance_name: str`, `engine_id →engine` |
| `gpu_device` | `Declaration` | `instance_name: str` (e.g. `gpu-0`), `parent_group_id →engine`, `ordinal: u32` |
| `thread_group` | `Declaration` | `instance_name: str` (`shared`\|`executor_thread`\|`task_manager_loop_thread`), `parent_group_id →engine or →gpu_device` |
| `plan` | `Declaration` | `parent{query_id: →query\|null, plan_id: →plan\|null}` (exactly one set), `instance_name: str`, `edges: [{source →port, target →port}]`, `worker_id: →worker\|null` |
| `operator` | `Declaration` | `plan_id →plan`, `parent_operator_ids: [→operator]`, `instance_name: str`, `type_name: str`, `custom_attributes: []` |
| `operator`/`port` | `Statistics` | `custom_attributes` (in-schema, never emitted by Sirius) |
| `port` | `Declaration` | `operator_id →operator`, `instance_name: str` (`*_receiver`/`*_sender`) |

`custom_attributes` is ONE flat array of `{"key": k, "value": {"<Type>": v} | null}` —
the `DynamicAttributes`/`DynamicStruct` serde shape (`crates/dynamic-attributes`). Types seen
in real traces: `String`, `I64`, `F64`.

### 4.2 `query` FSM  (entry `Init`, exit from `Executing`)

| state | fields |
|---|---|
| `Init` | `instance_name: str` (query label or `unnamed_query`), `query_group_id →query_group` |
| `Planning` | `{}` |
| `Executing` | `{}` |

Transitions: `Init→Planning→Executing→Exit`.

### 4.3 `task` FSM  (entry `Created`, exit from `Finalizing`)

| state | attributes (all required) | usages |
|---|---|---|
| `Created` | `instance_name` (`task-<n>`), `pipeline_uuid →operator` | — |
| `Queued` | — (no instance_name) | `queue{→task_queue, capacity_entries:1}` |
| `Routing` | `instance_name`, `preferred_device_id: i64` | `manager_thread{→task_manager_loop_thread, null}` |
| `Reserving` | `instance_name`, `requested_bytes, input_basis, peak_estimate, bytes_to_materialize: u64` | `manager_thread` |
| `Downgrading` | `instance_name`, `shortfall_bytes, partial_bytes: u64` | `manager_thread` |
| `Preparing` | `instance_name`, `origin_tier: str`, `target_tier: str`, `input_bytes: u64` | `executor_thread{→executor_thread, null}`, `reservation{→memory_tier, capacity_bytes}` |
| `Computing` | `instance_name` (`OPNAME(op_id)`), `current_operator_id: u32`, `input_bytes: u64`, `peak_allocated_bytes: u64`, `input_rows: u64` (WS9) | `executor_thread`, `reservation` |
| `Finalizing` | `instance_name`, `success: bool`, `output_rows: u64` (WS9), `output_bytes: u64` (WS9) | — |

Transitions (`fsm!` in `model/src/task.rs`): `created→queued`, `queued→routing`,
`routing→queued`, `routing→reserving`, `queued→reserving`, `reserving→downgrading`,
`reserving→preparing`, `downgrading→preparing`, `preparing→computing`, `computing→computing`,
`computing→finalizing`, plus abnormal `{created,queued,routing,reserving,downgrading,
preparing}→finalizing`. Tier strings: `GPU`/`GPU-<n>`, `HOST`, `DISK`.

### 4.4 `data_batch` FSM  (entry `Constructed`, exit from `Destructed`)

| state | attributes | usages |
|---|---|---|
| `Constructed` | `instance_name` (`batch`), `data_batch_id: u64` (process-unique), `producer_pipeline_uuid →operator`, `producer_task_uuid →task or nil` (WS9), `num_rows: u64` (WS9, 0=unknown), `num_columns: u64` (WS9) | — |
| `Stationary` | — | `memory{→memory, capacity_bytes}` |
| `InTransit` | — | `source_memory{→memory}`, `dest_memory{→memory}`, `channel{→channel}` — all with `capacity_bytes` = batch size |
| `Destructed` | `{}` | — |

Transitions: `constructed→stationary`, `stationary→in_transit`, `in_transit→stationary`,
`stationary→stationary`, `stationary→destructed`.

### 4.5 `batch_placement` FSM  (entry `BatchRegistered`, exit from `BatchConsumed`)

| state | attributes | usages |
|---|---|---|
| `BatchRegistered` | `instance_name` (`batch-<id>`), `batch_id: u64` (joins `data_batch.Constructed.data_batch_id`), `pipeline_uuid →operator` (consumer), `port_uuid →port`, `origin ∈ {operator_output, partition_output, reschedule_intermediate}`, `producer_task_uuid →task or nil` (WS9) | `tier{→memory_tier, capacity_bytes}` |
| `BatchQueued` | — | `tier` |
| `BatchPackaged` | `instance_name`, `task_uuid →task` | `tier` |
| `BatchProcessing` | `instance_name`, `task_uuid →task` | `tier` |
| `BatchConsumed` | `instance_name`, `reason ∈ {processed, task_failed, query_end}` | — |

Transitions: `batch_registered→{batch_queued, batch_packaged}`,
`batch_queued→{batch_queued, batch_packaged, batch_consumed}`,
`batch_packaged→{batch_packaged, batch_processing, batch_consumed}`,
`batch_processing→{batch_processing, batch_consumed}`. Self-transitions = tier changes.

### 4.6 `io_request` FSM  (entry `Issued`, exit from `Completed`; WS9, one per fresh-read split)

| state | attributes |
|---|---|
| `Issued` | `instance_name` (`scan_split`), `task_uuid →task or nil`, `pipeline_uuid →operator`, `file_count: u64`, `estimated_compressed_bytes: u64`, `estimated_decoded_bytes: u64` |
| `Completed` | `instance_name` (`""`), `bytes_read: u64`, `read_time_ns: u64`, `read_calls: u64`, `rows: u64` |

Transitions: `issued→completed→Exit`.

### 4.7 Resource FSMs

All six follow `XInitializing → XOperating → XFinalizing → Exit`
(self-loops on `Operating` allowed for capacity updates; not observed in Sirius traces).

`XInitializing` payload: `instance_name: str`, `parent_group_id: uuid` (→ resource-tree
group), `resource_type_name: str` — which **must** equal the analyzer's expected constant
(`rust/crates/telemetry/analyzer/src/model.rs` `validate_resource_type`, a **hard ingest
error** if wrong):

| entity | `resource_type_name` | `XOperating` payload | typical parent |
|---|---|---|---|
| `task_queue` | `task_queue` | `{"capacity_entries": u64}` (u64::MAX placeholder) | `thread_group` (shared) or `gpu_device` |
| `executor_thread` | `executor_thread` | `null` | `thread_group` (executor bucket) |
| `task_manager_loop_thread` | `task_manager_loop_thread` | `null` | `thread_group` |
| `memory` | `memory` | `{"capacity_bytes": u64}` | `engine` |
| `memory_tier` | `memory_tier` | `{"capacity_bytes": u64}` (DISK=0) | `engine` |
| `channel` | `channel` | `{"capacity_bytes": u64}` (u64::MAX placeholder) | `engine` |

`channel.ChannelInitializing` additionally requires `source_id →memory`, `target_id →memory`.
`memory_tier` instance names: `GPU-<n>` | `HOST` | `DISK` (the analyzer's data-flow view
keys on these; `Preparing.origin_tier/target_tier` strings must be consistent with them).
`XFinalizing` payload: `null`.

## 5. Id-reference graph (what "resolves" means)

```
engine.id ←─ worker.parent_engine_id, query_group.engine_id, gpu_device.parent_group_id,
             thread_group.parent_group_id (shared), memory/channel/memory_tier.parent_group_id
gpu_device.id ←─ thread_group.parent_group_id, task_queue.parent_group_id
thread_group.id ←─ executor_thread/task_manager_loop_thread/task_queue.parent_group_id
memory.id ←─ channel.source_id/target_id, data_batch usages (memory/source_memory/dest_memory)
query_group.id ←─ query.Init.query_group_id
query.id ←─ plan.parent.query_id
plan.id ←─ operator.plan_id
operator.id ←─ port.operator_id, task.Created.pipeline_uuid, data_batch.producer_pipeline_uuid,
               batch_placement.pipeline_uuid, io_request.pipeline_uuid
port.id ←─ plan.edges[].source/target, batch_placement.port_uuid
task.id ←─ batch_placement.{BatchPackaged,BatchProcessing}.task_uuid,
           *.producer_task_uuid (nil allowed), io_request.task_uuid (nil allowed)
resource ids ←─ every usage's resource_id (typed: queue→task_queue, reservation→memory_tier, …)
data_batch.Constructed.data_batch_id (u64) ←─ batch_placement.BatchRegistered.batch_id (u64)
```

**Resource-tree integrity** (what `print_resource_tree` / the viewer needs): every
`parent_group_id` chain terminates at the engine id; the engine is the root
(`Engine: ResourceGroup<Root = true>`).

## 6. Analyzer-imposed constraints beyond serde

From `SiriusUiAnalyzer::try_new` + `SiriusModelBuilder` (`analyzer/src/model.rs`):

- The engine id is discovered from the **first line of the engine stream** (the
  `print_resource_tree` example and the server index both do this); the `Init` event must be
  first in that file for the engine name to be picked up (`extract_engine` scans for `Init`).
- `try_build()` resolves every task/data_batch/batch_placement **usage's resource_id** via
  `resources.resource(...)` — an unresolvable usage id is a **hard error** (fails the whole
  session). Data batches are more forgiving: an invalid data_batch is dropped with a `warn!`.
- Per-query views require: query has `query_group_id`, plan tree resolves, and each task's
  `pipeline_uuid` points at an operator of some plan (tasks with dangling pipelines won't
  crash ingest but disappear from query views).
- The data-flow view requires `memory_tier` resources to exist (else HTTP 501 Unsupported).

## 7. Reality-vs-types divergences (found while calibrating)

1. **Old traces are incompatible with the current analyzer.** Pre-WS9 sessions (e.g. every
   `E1-*/E2-*/E3-*` trace, model commit `b77aa438`) lack `Computing.input_rows`,
   `Finalizing.output_rows/bytes`, `Constructed.producer_task_uuid/num_rows/num_columns`,
   `BatchRegistered.producer_task_uuid`. The current serde structs have no defaults ⇒ the
   importer silently truncates those streams at the first task/data_batch/batch_placement
   line. The ws9-new-fields.md "treat as optional" contract applies to the *Python* hwsim
   parser only, **not** to the Rust analyzer. The validator's `--allow-legacy` flag downgrades
   exactly these missing-field violations to warnings.
2. **`custom_attributes` serializes as ONE flat array** of `{key, value:{Tag:v}}` — not the
   three parallel typed arrays a casual reading of the C++ emitter suggests (already noted in
   `ws9-new-fields.md`; confirmed in the LB trace).
3. **Engine `Init`/`Exit` have no `seq`/`state` wrapper** (plain-entity shape) even though the
   engine feels lifecycle-ish. Same for `worker`.
4. **States without attributes have no `instance_name`** (`Queued`, `BatchQueued`,
   `Stationary`, `InTransit`); states with attributes always carry it, `""` after the first
   event. `Destructed`/`Planning`/`Executing` serialize `{}` while resource unit states
   serialize `null` — two different empty encodings.
5. **Unit/placeholder capacities**: task_queue and channel declare `u64::MAX` capacity;
   `memory_tier` `DISK` declares 0. Occupancy is derived from usages, not declared capacity.
6. **`task_manager_loop_thread` instances are ephemeral** — re-declared per query
   (`gpu-N-exec-manager` is a local of `manager_loop()`); N task-manager FSMs per session is
   normal, and thread identity is (name, span), not uuid.
7. **`io_request/` may contain a single empty ndjson file** (fully-cached run) — legal, and
   the analyzer accepts-and-ignores io_request events entirely.
8. **Sirius never emits** `operator.Statistics`, `port.Statistics`, or `engine`
   `instance_name` ≠ null… (observed `"instance_name":null` on engine Init). All legal per
   the types.
9. In the 654 MB main-repo sample (`telemetry_data/019fbafc-…`, pre-WS9), `Downgrading`
   occurs zero times; the pressured `E2-*` traces have it — transition coverage of the
   validator's legality table came from both.

## 8. Additional conventions for SIMULATED sessions (`--simulated` mode)

Agreed contract for hwsim's exporter (WS17):

- `engine.Init.implementation.name == "hwsim-sim"` (so the UI engine picker distinguishes
  simulated from `siriusDB` sessions).
- Engine `Init.implementation.custom_attributes` must additionally contain:
  `hwsim.simulated` = I64 `1`; `hwsim.source_session` (String: source session uuid);
  `hwsim.source_query` (String: traced query label); one `hwsim.knob.<name>` per non-default
  knob (numeric or string value).
- Every `query.Init.instance_name` is suffixed `@<knob>=<value>[,...]` (e.g.
  `LB_tpch_q9_iter1@gpu_compute=0.5`), or `@baseline` when all knobs are at
  their defaults (WS17's documented marker).
- Fresh, time-ordered (v7) uuids for all entities — never reuse source-trace uuids.
- Sim t0 anchored to the traced query start (timestamps stay in real epoch-ns range).
