# Sirius as an Apache Doris backend

A Rust process that joins an **unmodified, official Apache Doris FE** as a backend node and
runs the plan fragments it receives on the embedded Sirius engine. The Doris side is
configuration only (`conf/fe.conf`, `sql/session.sql`, views over `local()`); everything
else lives here, structured like [`../starrocks`](../starrocks): a protocol shell whose every
piece builds and tests without an engine or a GPU (`--no-default-features`, what CI and a Mac
run), plus the engine-linked build for a GPU box.

```
Doris FE (official 4.1.4 binary)          sirius-doris-be (this crate)
  ALTER SYSTEM ADD BACKEND  <──MySQL───   node.rs        self-registration
  HeartbeatService.heartbeat ──thrift──>  node.rs        Alive in SHOW BACKENDS
  BackendService.*           ──thrift──>  node.rs        periodic probes (stubs)
  PBackendService (gRPC h2c) ──brpc_port> backend_service.rs
      glob / fetch_table_schema           file_schema.rs analysis-time local() support
      exec_plan_fragment(_prepare/_start) params.rs      TCompact TPipelineFragmentParamsList
                                           doris-plan-translator → Substrait → engine.rs
      fetch_data / cancel_plan_fragment    result_store.rs / result_encoder.rs
```

## Layout

| path | what |
|---|---|
| `doris/` | `apache/doris` submodule, shallow, pinned to the FE release tag (only `gensrc/thrift`, `gensrc/proto` are used) |
| `crates/doris-thrift`, `crates/doris-proto` | codegen: thrift 0.22 `--gen rs` and prost/tonic over the submodule IDL |
| `crates/doris-plan-translator` | `TPipelineFragmentParams` → Substrait: descriptor table, type gate, expressions, plan nodes, the single-plan stitcher (`translate_batch`), and `explain` (readable plan text) |
| `src/` | the backend: `node.rs`, `backend_service.rs`, `params.rs`, `file_schema.rs`, `result_*.rs`, `fragment_executor.rs`, `engine.rs` |
| `src/bin/dump-fragments.rs` | pretty-print a captured dispatch payload (`--summary`, `--translate` per fragment, `--stitch` as one plan) |
| `conf/fe.conf`, `sql/` | FE config, global session defaults, TPC-H views over `local()`, the 22 queries (`sql/tpch`), probes for plan-level semantic gaps (`sql/gaps`) |
| `scripts/` | `fetch-fe.sh` (official tarball → `.doris-fe/fe`), `fe.sh`, `be.sh`, `run-tpch.sh` |
| `tests/fixtures/tpch/`, `tests/fixtures/gaps/` | captured FE→BE dispatches for all 22 TPC-H queries and for the `sql/gaps` probes (`INDEX.md` has the shapes and coverage) |
| `tests/snapshots/` | the reviewed translation of every captured query (stitched node tree + `substrait-explain` text, or the refusal); `tests/corpus.rs` diffs against these |

## Running on a laptop (no GPU)

```bash
cd experimental/doris
pixi run -e fe fe-fetch                 # once: download the pinned Doris release, keep fe/
pixi run -e fe fe-start                 # FE on 127.0.0.1 (9030 mysql, 8030 http)
pixi run -e be bash scripts/be.sh start # engine-less backend; registers itself, translate-only
pixi run -e fe mysql -h127.0.0.1 -P9030 -uroot -e 'SHOW BACKENDS\G'   # Alive: true

# TPC-H SF1 parquet (see the repo's dataset-manager skill / test/tpch_performance), then:
pixi run -e fe bash scripts/run-tpch.sh --data /tmp/tpch-sf1 --translate-only   # refresh the corpus
pixi run -e fe bash scripts/run-tpch.sh --data /tmp/tpch-sf1 --translate-only \
    --sql-dir sql/gaps --out tests/fixtures/gaps                                # refresh the gap probes
pixi run -e be be-test-no-engine        # cargo test --workspace --no-default-features
```

In translate-only mode every query fails at `fetch_data` on purpose; the message says whether
the dispatch translated into one plan (`all N fragments translated into one plan with output
[...]`) or why not, and `run-tpch.sh` prints that verdict per query. Inspect a capture with
`cargo run --no-default-features --bin dump-fragments -- --stitch tests/fixtures/tpch/q01/batch-00-request.tcompact`.

After a translator change, regenerate the snapshots and review the diff before committing:

```bash
UPDATE_SNAPSHOTS=1 pixi run -e be cargo test --test corpus -p sirius-doris-be --no-default-features
git diff tests/snapshots/
```

Environment variables read by the backend:

| variable | effect |
|---|---|
| `SIRIUS_BE_TRANSLATE_ONLY` | on (default): accept every dispatch, dump it, translate it as one plan, and fail the query at `fetch_data` with the translation verdict; `0` executes |
| `SIRIUS_BE_DUMP_FRAGMENTS` | directory for per-query dispatch dumps (`be.sh` sets `log/dump`) |
| `RUST_LOG` | tracing filter (`sirius_doris_be=debug` shows every heartbeat) |

## With the engine (Linux + NVIDIA)

`pixi run be-build` builds libsirius via the repo-root build and links it (`sirius-engine`
feature, on by default); `pixi run be-run -- --fe-host <fe>` then runs queries on the GPU.
See `plan-doc/doris-pseudo-be-plan.md` for the milestones (P0 scaffolding → P1 translator →
MVP-A0 single plan → MVP-A fragments → MVP-B multi-node).
