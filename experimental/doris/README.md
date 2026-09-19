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
| `scripts/` | `fetch-fe.sh` (official tarball → `.doris-fe/fe`), `fe.sh`, `be.sh`, `run-tpch.sh`; `build-duckdb-substrait.sh`, `validate_tpch_results.py`, `cpu-diff.sh` (the DuckDB baseline and the CPU differential, below); the benchmark harness `fetch-be.sh`, `be-native.sh`, `bench.sh`, `bench-all.sh`, `bench-report.py`, `run-tpch-duckdb.sh`, `fe-audit.py`, `evict-cache.py`, `olap-load.sh` (below) |
| `tests/fixtures/tpch/`, `tests/fixtures/gaps/` | captured FE→BE dispatches for all 22 TPC-H queries and for the `sql/gaps` probes (`INDEX.md` has the shapes and coverage) |
| `tests/snapshots/` | the reviewed translation of every captured query (stitched node tree + `substrait-explain` text, or the refusal); `tests/corpus.rs` diffs against these |
| `tests/expected/tpch-sf1/`, `tests/expected/tpch-sf10/`, `tests/expected/gaps-sf1/` | the queries' results on the SF1 / SF10 datasets, computed by DuckDB from the same parquet files (`INDEX.md` has the row counts); what `run-tpch.sh` and the CPU differential validate against |

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

With the engine (Linux + NVIDIA, `SIRIUS_BE_TRANSLATE_ONLY=0`), `run-tpch.sh --data DIR` runs
the queries for real and validates every result against `tests/expected/tpch-sf1` (see below).

In translate-only mode every query fails at `fetch_data` on purpose; the message says whether
the dispatch translated into one plan (`all N fragments translated into one plan with output
[...]`) or why not, and `run-tpch.sh` prints that verdict per query. Inspect a capture with
`cargo run --no-default-features --bin dump-fragments -- --stitch tests/fixtures/tpch/q01/batch-00-request.tcompact`.

After a translator change, regenerate the snapshots and review the diff before committing:

```bash
UPDATE_SNAPSHOTS=1 pixi run -e be cargo test --test corpus -p sirius-doris-be --no-default-features
git diff tests/snapshots/
```

## CPU differential of the translator (no GPU)

The plans the translator produces can be executed without the engine: DuckDB's substrait
consumer (`SubstraitToDuckDB` in the repo's `substrait/` submodule, the reader
`src/sirius_ffi.cpp` compiles into libsirius) turns the bytes into a DuckDB plan, and DuckDB
runs it on the CPU. `scripts/cpu-diff.sh` stitches every captured dispatch into one plan,
runs it that way with the optimizer rules the FFI disables, and validates the rows against
`tests/expected/tpch-sf1` (22/22 pass; MVP-A0 preparation, `plan-doc/tasklist.md` A0.1):

```bash
pixi run -e check duckdb-substrait-build      # once, ~3 min: DuckDB v1.5.5 + substrait extension
                                              # under .duckdb-substrait/ (upstream tag = the
                                              # commit substrait/.gitmodules pins; python-duckdb
                                              # in the `check` env is the same version)
pixi run -e check tpch-cpu-diff               # or: bash scripts/cpu-diff.sh [--data DIR] [--queries 1,6]
ls log/cpu-diff/q01/                          # result.tsv, duckdb-plan.txt (the optimized logical plan)
```

The same script checks the gap probes that translate (`--corpus tests/fixtures/gaps --sql-dir
sql/gaps --expected tests/expected/gaps-sf1 --queries g13-distinct,g13-distinct-topn`). To
re-root the corpus plans, `cpu-diff.sh` rewrites their `/tmp/tpch-sf1` scan paths to `--data`
(`dump-fragments --stitch --write-plan FILE --rewrite-path OLD=NEW` does it for one dispatch).

`scripts/validate_tpch_results.py` is the shared piece: `expected` (re)generates the baseline
from a parquet dataset (`pixi run -e check tpch-expected -- --data DIR`, for SF10 give it
another `--out`), `consume` runs plans through the consumer and compares, `validate` compares
a `run-tpch.sh` output tree — which `run-tpch.sh` does itself after an execution-mode run
(`--expected DIR`, `--no-validate`). Numbers match within a relative tolerance (1e-9) or half
a unit of the coarser decimal scale, whichever is larger (Doris types `avg(DECIMAL)` as
DECIMAL(38,4); DuckDB computes a DOUBLE); a query with a top-level ORDER BY is compared in
order, allowing ties in either order; the rest are compared as sets.

Environment variables read by the backend:

| variable | effect |
|---|---|
| `SIRIUS_BE_TRANSLATE_ONLY` | on (default): accept every dispatch, dump it, translate it as one plan, and fail the query at `fetch_data` with the translation verdict; `0` executes |
| `SIRIUS_BE_DUMP_FRAGMENTS` | directory for per-query dispatch dumps (`be.sh` sets `log/dump`) |
| `RUST_LOG` | tracing filter (`sirius_doris_be=debug` shows every heartbeat) |

## With the engine (Linux + NVIDIA)

The engine-linked backend (`sirius-engine` feature, on by default) links `libsirius.so` from
the repo-root build tree and runs every dispatched query on the GPU:

```bash
# once: the engine (repo root; CUDA 13 + RAPIDS come from the root pixi env)
git submodule update --init --depth=1 --jobs 3 duckdb substrait cucascade
pixi run make TEST_BUILD_TARGET=                  # → build/release/extension/sirius/libsirius.so

cd experimental/doris
pixi run cargo build --release -p sirius-doris-be # default env carries the engine's toolchain/libs
pixi run bash scripts/be.sh start --engine        # SIRIUS_BE_TRANSLATE_ONLY=0, --sirius-config conf/sirius.yaml
pixi run -e fe bash scripts/run-tpch.sh --data /tmp/tpch-sf1   # execute + validate the 22 queries
```

The run validates every result against `tests/expected/tpch-sf1` (`--ulps 1` by default: the GPU's
DOUBLE→DECIMAL cast truncates the last digit where DuckDB rounds, see `plan-doc/reference/
semantics-gaps.md` G-19) and writes `log/tpch/timings.csv` with each query's engine time, read
back from the backend's `query executed on the engine` log line.

`conf/sirius.yaml` sizes the engine for a small box shared with the FE (GPU 90 %, pinned host
tier 12 GiB, spill and telemetry under `log/`); without it Sirius pins 90 % of host RAM. `be.sh
--engine` puts the build tree and the env's `lib/` on `LD_LIBRARY_PATH` (`SIRIUS_BUILD_DIR`
overrides the tree). The `be-build`/`be-run` tasks do the same but first re-run the root build
including its C++ unit tests. See `plan-doc/doris-pseudo-be-plan.md` for the milestones (P0
scaffolding → P1 translator → MVP-A0 single plan → MVP-A fragments → MVP-B multi-node).

## Benchmark: Doris vs Doris + Sirius (plan-doc `experiments/sf10-bench/plan.md`)

The same FE, the same parquet dataset (views over `local()`), the same 22 queries, with the
backend swapped: the official Doris BE on the CPU versus this backend on the GPU, plus the
single-process references (DuckDB, Sirius's transparent path) and Doris on internal tables.
`bench.sh` runs one system for N rounds (round 1 cold: freshly started process + evicted page
cache) and validates every round; `bench-all.sh` chains the systems and writes the report.

```bash
cd experimental/doris
pixi run bash scripts/fetch-be.sh                 # once: the official BE (be/ of the same tarball) → .doris-be/be
# dataset: tpchgen-cli -s 10 --format=parquet --parts=1 → <dir>/<table>/part.0.parquet (on the NVMe)
pixi run -e check python scripts/validate_tpch_results.py expected --data /mnt/nvme/tpch_parquet_sf10 \
    --out tests/expected/tpch-sf10                # the DuckDB baseline the runs are validated against
pixi run bash scripts/bench-all.sh --data /mnt/nvme/tpch_parquet_sf10 --rounds 4 \
    --price native-split=0.752 --price sirius-buffered=0.752   # → log/bench/sf10-<system>/, log/bench/sf10-results.md
pixi run bash scripts/bench-all.sh --data /mnt/nvme/tpch_parquet_sf10 --systems native-olap --load-olap
                                                  # optional: Doris on internal tables (scripts/olap-load.sh)
pixi run bash scripts/bench-all.sh --data /mnt/nvme/tpch_parquet_sf100 --expected /mnt/nvme/expected-sf100 \
    --host-capacity 160Gi --baseline native --price native=4.529 --price sirius-buffered=4.529 --load-olap
                                                  # SF100: the baseline is generated onto the NVMe (not into the repo), the
                                                  # pinned host tier leaves room for the dataset's page cache, and stock
                                                  # Doris beats the split variant at this scale
```

Systems (`bench.sh --system`): `native` (official BE, stock 4.1.4 session defaults,
`sql/session-native.sql`), `native-split` (same plus `file_split_size_on_be = 0`: FE-side file
splits, because the stock BE-side split reads a single-file `local()` table with one scanner),
`native-olap` (internal tables), `sirius` (this backend, `conf/sirius-bench.yaml`, O_DIRECT
reads), `sirius-buffered` (same through the page cache — the hot-run counterpart of Doris's IO
path), `duckdb` / `duckdb-gpu` / `duckdb-gpu-pinned` (`run-tpch-duckdb.sh`: the build tree's
DuckDB shell with `SIRIUS_DISABLE=1`, with Sirius, with every table pinned in GPU memory).

Each run directory holds `env.txt` (machine, commit, config hashes, backends), `variables.txt`
(the GLOBAL session variables in effect), the effective `sirius.yaml`, `round<k>/` (per-query
`result.tsv`, `explain.txt`, `timings.csv` with the client wall time, the engine time and the
FE audit numbers joined by `fe-audit.py`, `summary.csv` from the validator, `samples.csv` from
the 0.5 s sampler: RSS, bytes read from disk, CPU ticks of the backend process, and for the
GPU systems nvidia-smi's memory in use, `utilization.gpu` and `utilization.memory`) and
`rounds.csv` (everything flattened by `bench-report.py rounds`: per query the peaks, deltas
and mean GPU utilization over its window). `bench-report.py report` turns several run directories into the Markdown tables.

Two things the harness does to keep the comparison fair: it drops this backend from the FE
(`ALTER SYSTEM DROPP BACKEND`) while a native system runs — the backend never reports its CPU
count, and the FE's automatic parallelism is the minimum over every registered backend, so the
native BE would otherwise run one instance per fragment; and it never restarts the FE, only
the backends (both stay on their own ports: this backend on 9050/9060/8040/8060, the native BE
on 9150/9160/8140/8160 per `conf/be.conf`).
