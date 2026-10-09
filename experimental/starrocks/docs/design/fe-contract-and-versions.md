# Plan: FE contract and versions

## Why

Which parts of the StarRocks FE → backend interface are contracts and which are implementation details (StarRocks 4.1.3, `8a8e186`):

- **Two dispatch paths:**
  - `exec_plan_fragment`: one instance per call, the thrift plan in the attachment;
  - `exec_batch_plan_fragments`: a shared parameter block plus per-instance parameters. Used with `enable_single_node_schedule` (off by default), for single-worker queries over internal tables.
- **Versions:** a backend is upgraded before the FE, and nothing in the protocol checks versions. The thrift plan structures change every release. A third-party executor built against one release should stay on the FE's release.
- **Column order:** aggregate i ↔ slot `group_by_count + i` of the output tuple (the intermediate tuple for a non-finalizing phase), counted in descriptor order. Window nodes follow the same pattern. Order holds only within a tuple; tuples sit in a hash map.

## Today

Line numbers refer to `f4780e74`. `CNS` is `experimental/starrocks/src/compute_node_service.rs`.

- **`exec_plan_fragment`:** works, and only `binary` encoding is accepted (CNS:2268 test).
- **`exec_batch_plan_fragments`: broken for the FE's real request shape.**
  - The FE sets each instance's `desc_tbl` to an empty table marked cached, and puts the real one in `common_param` (`Deployer.java:145,461-466`).
  - The CN copies the shared table only when an instance's own is unset (CNS:1273). So it treats the empty table as a cache reference, misses, and fails with "descriptor table cache miss for query".
  - The unit test (CNS:2287) leaves `desc_tbl` unset, which hides the bug.
- **Version:** the heartbeat reports `sirius-starrocks-cn/<crate version>` (`lib.rs:199-201`). The StarRocks release the CN is built for (4.1.3, `8a8e186`) appears nowhere.
- **Column order:** matches the BE.
  - `slot_global_index` (`descriptor_table.rs`) reads slots in descriptor order and tuples in `row_tuples` order.
  - The intermediate tuple of a non-finalizing aggregation isn't handled (`node_translator.rs:594` refuses separate intermediate and output tuples).
  - Window nodes aren't supported.

## Steps

### Step 1: fix batch dispatch

- In `translate_batch_attachment`, also use `common_param.desc_tbl` when the instance's table is cached and empty.
- Put the shared table in the per-query cache before processing instances, the way the BE prepares the shared parameter block first.
- **Tests:**
  - change the unit test to the FE's real shape (an empty cached `desc_tbl` per instance, the real one in `common_param`);
  - end to end: `SET enable_single_node_schedule = true` on a one-CN cluster with an internal table, if internal tables are in scope by then.

### Step 2: report and pin the StarRocks release

- Build the heartbeat version string as `sirius-starrocks-cn/<crate version> (starrocks 4.1.3, 8a8e186)`. Take it from the submodule pin at build time (`build.rs`), so `SHOW COMPUTE NODES` shows it.
- Write the rule in the CN README:
  - build the CN for each StarRocks release;
  - run it on the same release as the FE;
  - upgrade both together.
- **Test:** a unit test on the heartbeat payload.

### Step 3: column-order contract tests

- Translator tests that pin aggregate i → slot `group_by_count + i`:
  - for a finalizing aggregation (output tuple);
  - for a non-finalizing one (intermediate tuple). The intermediate case is a prerequisite for two-phase AVG ([multi-phase-aggregation.md](multi-phase-aggregation.md)).
- Write down the window-node rule (function i → slot i of the result tuple) where window support will go, so it's followed when windows are added.

### Step 4: keep up with StarRocks releases

- The interfaces we depend on, and how stable they are (our reading of 4.1.3):

  | interface | stability |
  |---|---|
  | `ALTER SYSTEM ADD COMPUTE NODE` | public, stable |
  | `SHOW PROC '/compute_nodes'` | works; output columns not promised |
  | `PInternalService` methods | internal; methods get added |
  | thrift plan structures (`TPlanNode`, `TExprNode`, `TDescriptorTable`) | internal; change every minor release |
  | heartbeat fields | internal; `cluster_id` already deprecated |

- When the submodule moves to a new release:
  - re-record the FE fragments used as translator fixtures (`tests/fixtures/`);
  - re-run the translator tests and the 22-query run.

  This turns a planner change into a failing test instead of a wrong result.

## Out of scope

- Supporting more than one StarRocks release in one CN build.
