# Sirius Task Analyzer Semantic Testing

These checks compare Sirius task telemetry with the Quent simulator at the UI
contract level. Exact event names and resource names do not need to match; the
important behavior is that the Quent UI receives equivalent model shapes.

## Simulator Reference

1. Generate Quent simulator telemetry with the simulator example from the Quent
   checkout used by this workspace.
2. Run the simulator analyzer/server over the generated data.
3. Capture a query bundle and task resource timelines.
4. Record these semantic expectations:
   - The engine is the root resource group.
   - Task-like FSM metadata is present in `entities.fsm_types`.
   - Resource timelines return `Binned` data.
   - Per-task-state requests return `BinnedByState` data.
   - Long entity thresholds return FSM payloads for long tasks.

## Sirius Runtime

1. Run a Sirius workload with `telemetry.enable_quent=true`.
2. Start the Sirius telemetry server:

   ```sh
   pixi run cargo run --manifest-path rust/Cargo.toml -p sirius-telemetry-server -- --output-dir <telemetry-dir>
   ```

3. Open the Quent UI against the analyzer endpoint and select the captured
   Sirius engine/query.
4. Compare the same semantic expectations:
   - Engine-rooted task resources appear in the resource tree.
   - The `task` FSM type is available.
   - `task_queue` timelines show nonzero `capacity_entries` occupancy while
     tasks are queued.
   - `executor_thread` and `task_manager_loop_thread` timelines show `unit`
     occupancy while tasks are preparing/computing/routing/reserving.
   - Per-state task timelines include states such as `queued`, `routing`,
     `reserving`, `preparing`, and `computing`.
   - Long task FSM payloads render on the timeline.

If simulator generation, Sirius runtime generation, GPU availability, or
external permissions are unavailable, keep the Rust analyzer tests as the
blocking signal and record which runtime comparison step was skipped.
