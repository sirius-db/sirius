# Sirius StarRocks compute node (CN)

A Rust StarRocks **compute node** shim (`sirius-starrocks-cn`). It registers with a StarRocks
**frontend** (FE) over MySQL (`ALTER SYSTEM ADD COMPUTE NODE`), serves the Thrift
`HeartbeatService` and `BackendService`, and reports inventory. Plan execution is not implemented
yet; the backend RPCs are skeletons.

Workspace layout:

- `src/` — the `sirius-starrocks-cn` binary + library (heartbeat/registration/report logic, and
  the reusable `ComputeNode` runner).
- `crates/starrocks-thrift` — Rust bindings generated from the StarRocks Thrift IDL.
- `crates/starrocks-plan-translator` — StarRocks plan → Substrait translation.
- `crates/starrocks-test-harness` — integration-test harness (`TestCluster`) that boots a real FE
  and runs the CN in-process.
- `starrocks/` — the StarRocks source submodule (FE built from here).

## Build & run

Run everything through `pixi run <task>` so commands execute in the right activated environment.

```bash
pixi run cn-build        # build the CN release binary
pixi run fe-build        # build + package the StarRocks FE into starrocks/output/fe (slow)
pixi run cluster         # run the FE and CN together in the foreground
```

## Testing

```bash
# Rust unit + plan-translator tests only (no FE, no Java needed):
pixi run -e cn cargo test -p sirius-starrocks-cn --lib
pixi run -e cn cargo test -p starrocks-plan-translator

# Full suite including the FE↔CN integration tests (needs Java + Rust):
pixi run -e default cargo test
```

The integration tests in `tests/integration.rs` boot a real StarRocks FE and run the compute node
in-process via `starrocks_test_harness::TestCluster`, exercising the full handshake: SQL
registration, FE→CN heartbeats/liveness, FE-restart resilience, and graceful shutdown.

Notes:

- Run them in the **`default`** pixi environment (`pixi run -e default cargo test`) — it provides
  both the Java/Maven toolchain (for the FE) and Rust.
- The FE is **built on demand**: if `starrocks/output/fe` is missing, the harness runs
  `pixi run fe-build` for you. The first run takes several minutes (Maven build); later runs reuse
  the packaged FE and are fast.
- Each test runs in an isolated, throwaway `STARROCKS_HOME` under `target/fe-it/` (its own meta
  dir, log dir, config, and freshly allocated ports), so runs are reproducible and clusters don't
  collide. On success the home is removed.
- On failure (startup error or a panicking assertion) the home is **preserved** and its path is
  printed to stderr; FE logs are under its `log/` directory and the launcher output is in
  `fe.bootstrap.log`. In CI, collect `experimental/starrocks/target/fe-it/` as an artifact to
  debug failures.
