# Onboarding: Sirius as a StarRocks compute node

This is the code-grounded companion to the project's meeting notes and design proposal
(see [Further reading](#9-further-reading)). Everything below is verified against the actual
source in `experimental/starrocks/`, `rust/`, and `src/include/sirius_ffi.hpp` as of this
writing, plus the real GitHub history under the `starrocks` label — not a transcript
summary. Where the design doc describes something *proposed*, this doc says so explicitly.
It also folds in the decisions from the July 2026 onboarding/pairing session with Matthijs
Brobbel (project lead): task ownership (§7), the division of ownership between StarRocks
and Sirius (§2.5), the streaming-operator semantics (§2.3), goals (§8), and the known
environment blocker + workaround (§5.1).

**Quick start:** run `experimental/starrocks/scripts/setup-dev-env.sh` (see
[Build & run locally](#5-build--run-locally)) to bootstrap a working dev environment, then
come back here for the map.

## 1. What this is, and why

Sirius normally runs as a DuckDB extension. This project embeds it instead as a
**compute node (CN)** inside [StarRocks](https://github.com/starrocks/starrocks)' shared-data
architecture — StarRocks keeps its existing Java frontend (FE: SQL parsing, planning,
coordination, metadata), and query fragments that would normally run on a StarRocks C++
backend (BE) are instead executed by a Rust process that embeds the Sirius GPU engine.

Why StarRocks specifically: Alibaba is the primary customer driving this, with Trend Micro
and RM Labs (blockchain analytics) also interested — StarRocks is currently the fastest
open-source distributed OLAP database, so its users are exactly the performance-sensitive
audience GPU acceleration targets. An earlier, architecturally similar effort targeted
[Apache Doris](https://doris.apache.org/) (which StarRocks forked); that work is **paused,
not continued** — the team decided to build StarRocks-specific support first (Alibaba only
cares about StarRocks) and extract reusable pieces (Substrait translation, a public Sirius
embedding API, exchange components) afterward. Prior-art from the Doris branch
(`origin/doris`) is referenced throughout the design proposal for exactly this reason: the
shape of a future generic "embed Sirius in any distributed query engine" layer is expected
to fall out of doing StarRocks properly first. The long-term direction points the same way:
Sirius is being decoupled from DuckDB-internal APIs so that eventually the DuckDB extension
itself becomes a thin shim around a standalone Sirius library — the same embedding surface
this CN uses.

## 2. Architecture

### 2.1 Topology — what's implemented today

```
                          MySQL wire protocol (SQL clients)
                                      │
                    ┌─────────────────▼─────────────────┐
                    │   StarRocks FE (Java, unmodified)  │
                    └───┬───────────┬───────────┬────────┘
             MySQL proto│    thrift │      BRPC │ (Baidu PRPC framing,
        ALTER SYSTEM ADD│ heartbeat │ exec_plan_│  raw TCP, not gRPC)
          COMPUTE NODE   │  (:9050)  │ fragment /│
                         │           │ fetch_data│
                         │           │ (:8060)   │
                    ┌────▼───────────▼───────────▼────────┐
                    │     Rust "sirius-starrocks-cn"        │
                    │  process (one per GPU, in principle)  │
                    │                                        │
                    │  heartbeat/backend thrift servers  ────┼─ src/lib.rs
                    │  BRPC PInternalService  ────────────────┼─ src/{brpc,prpc,
                    │                                        │   compute_node_service}.rs
                    │  StarRocks-thrift → Substrait  ─────────┼─ crates/starrocks-plan-translator
                    │  Substrait execution (today: stub;      │
                    │   landing: SiriusEngine via rust/crates/ │
                    │   sirius, over src/include/sirius_ffi.hpp)│
                    └────────────────────────────────────────┘
```

Two independent transport planes, both real StarRocks wire protocols (nothing custom):

- **Thrift** — CN→FE registration is a plain MySQL `ALTER SYSTEM ADD COMPUTE NODE` (so the
  CN looks like any other node from FE's SQL surface); FE→CN heartbeat/control uses
  StarRocks' generated `HeartbeatService`/`BackendService` thrift interfaces.
- **BRPC** — FE dispatches actual query fragments to the CN over StarRocks'
  `PInternalService`, which in production StarRocks rides Baidu's BRPC framework. This
  project doesn't link BRPC's C++ runtime; it re-implements just enough of BRPC's wire
  format (the "PRPC" TCP frame: 4-byte magic, 12-byte header, protobuf meta + body +
  attachment — see `src/prpc.rs`) to be a compatible peer, and generates the
  `PInternalService` async Tower-service facade from StarRocks' own `.proto` files via a
  custom `prost_build::ServiceGenerator` (`build.rs`) — playing the role `tonic-build` plays
  for gRPC.

### 2.2 The translation pipeline (implemented)

A dispatched fragment carries a StarRocks-thrift `TExecPlanFragmentParams`, whose plan and
expression trees are encoded as **flat pre-order node lists** (each node states its
`num_children`; children are the nodes immediately following it). The
`starrocks-plan-translator` crate rebuilds the tree with a cursor and emits a Substrait
`Plan` — v1 covers scan/filter/project relations and comparison/boolean/cast/is-null
expressions (see that crate's module doc for the exact table and the "adding a node"
checklist). From there, Substrait → DuckDB logical plan → Sirius physical plan is the same
path Sirius already uses elsewhere (Substrait extension), which is why this integration
needed no new query-planning machinery in Sirius core.

### 2.3 Streaming / cross-CN exchange — proposed, not implemented

Draft PR **#914** (`docs(starrocks): compute-node exchange & public-API design proposal`,
not merged) proposes how *multi-fragment, multi-CN* queries will work: a public **stream
session** API on Sirius (push batches in, pull batches out, non-blocking), streaming
source/sink operators, GPU-direct cross-CN transport via `nixl` (wrapping UCXX, with a
bRPC/CPU fallback), and a recommended shared-memory-budget model so exchange buffers
participate in Sirius's existing downgrade/spill executor. **None of this is built yet** —
today's CN only executes single, self-contained fragments (see §2.4) with no cross-CN data
movement at all. Don't re-derive this design from scratch; read the PR (or
`doc-plan.md` Part 2, a copy of the same content) and the [status board](#3-status-board)
below for exactly which pieces (issues #836–#840) are still open.

The concrete operator semantics, as pinned down in the onboarding session:

- **The streaming source exists only for exchange input** — a fragment whose input arrives
  from *another* CN. A leaf fragment reading parquet uses Sirius's normal, already-existing
  scan source; the streaming source appears only downstream of an exchange.
- **Fragment boundaries → streaming source (bottom) + streaming sink (top), for
  intermediate fragments.** A leaf fragment may still get a streaming sink (to stream
  results out) but never a streaming source; a trivial single-node `SELECT *` has neither.
- **Who feeds the source:** the Rust CN wrapper instantiates the Sirius context, creates a
  streaming session carrying that fragment's source/sink operators, handles the nixl
  exchange, and pushes each received batch into the matching streaming source; Sirius then
  executes the fragment as a normal pipeline.
- **Sink output stays on the GPU**, held in a cuCascade data repository with its own
  channel — so exchange-queued batches stay spillable (downgradable) under memory pressure,
  and nixl can pick its fastest path (`cudaMemcpy` within a GPU, NVLink within a box,
  UCX-class backends across nodes) precisely because the data is GPU-resident. nixl is a
  high-level, agent-based point-to-point GPU transfer API from the Dynamo project; the CN
  just tells it where partitions go, since the FE-planned fragment already encodes the
  routing — no custom transport logic to build.
- **Backpressure survives, but via a different mechanism than engine-side checks:** nixl's
  side channel between agents (announce/request transfers) plus a receive-completion
  notification callback, with host-memory staging as a pressure valve.
- **Configuration stays additive.** Streaming-session/operator settings (e.g. min/max
  exchange batch size) get added to the one generic Sirius context in a way that does not
  affect the non-distributed path — no StarRocks-specific context variant, so any engine
  can embed the same context unchanged. Scope this config surface early; it feeds
  #839/#840.

### 2.4 What "executes" a fragment today

Nothing GPU-backed yet, in `dev`. `fragment_executor.rs` defines a `FragmentExecutor` trait
and today's only implementation, `StubExecutor`, fabricates one placeholder row per output
column so the dispatch → translate → encode → fetch plumbing is fully exercisable without a
GPU or a Sirius build tree. Open PR **#1024** replaces it with `SiriusEngine`: an actor that
owns a real `sirius::SiriusContext` (from the `rust/crates/sirius` crate added by PR #908)
on a dedicated thread — needed because the underlying context is `!Send`/`!Sync` and the
engine serializes queries through one process-global context — and executes the translated
Substrait plan on the GPU. That PR is stacked on #1008 (the CN's execute path), #1022 (a
`SiriusContext::execute_substrait` FFI entry point), and #1021 (translator support for
`local_files` parquet reads, without which real file scans can't resolve). Reading #1024's
diff is the fastest way to see how every layer below connects end to end.

For scale: the *entire* public FFI surface today is `make_context` /
`make_context_from_config` returning a context `unique_ptr` — that's all. #1022 adds the
one-shot "execute these Substrait bytes, return Arrow" call plus a result collector; the
streaming-session surface (create session, push to a source, pull from a sink, destroy) is
the extension after that — the seam where the §2.3 operators eventually get exposed.

### 2.5 Division of ownership — what this project is and isn't

The most load-bearing framing from the onboarding session: this is **not** a distributed
Sirius, and explicitly not a Theseus rebuild — it is Sirius as the execution engine
*inside* StarRocks, replacing CPU stage execution with GPU stage execution. In Matthijs's
words: "it's very dumb — we just translate, execute, exchange, move on."

| Layer | Owns |
|-------|------|
| StarRocks FE | Everything distributed: SQL parsing, planning, coordination, metadata — and, critically, plan-fragment creation *including partition routing* (a fragment already encodes which partition goes to which CN). |
| Sirius | Per-fragment execution only: a single pipeline, front to back, with no sideways information passing during a streaming session. Deliberately kept unaware it's part of distributed execution, so the same context can be embedded in any engine. |
| Rust CN wrapper | The exchange: nixl transfers between CNs, and routing received batches into the right streaming source. |

The consequence, stated plainly in the session: bespoke distributed algorithms with custom
cross-worker synchronization (e.g. a Theseus-style distributed sort where a worker checks
the target's free GPU memory before dispatching the next task) are out of scope **by
design** at the Sirius layer — "this is completely out of our control… StarRocks drives
the planning and creation of plan fragments." Distributed-systems depth applies one layer
down instead: the exchange, backpressure, and GPU-memory machinery inside a fragment's
streaming session (cuCascade + nixl) — which is exactly where #838/#839/#840 sit.

## 3. Status board

Ground truth as of this writing (`gh`/GitHub REST, `starrocks` label, 30 issues+PRs; also
`git log --oneline --all -- experimental/starrocks`, 24 commits). ✅ = merged, 🚧 = open PR,
⬜ = open issue / not started.

**Tracking issue #826** ("Sirius compute node implementation for Starrocks") and its
sub-issues:

| # | Title | Status | Notes |
|---|-------|--------|-------|
| #835 | Add Rust bindings for Sirius | ✅ | Landed via #908 (`sirius`/`sirius-sys` crates) |
| #841 | Plan fragment → Substrait conversion | ✅ | Landed via #852 (`starrocks-plan-translator`) |
| #836 | Streaming source operator | ⬜ | **Alexander — start here** (design: PR #914 §2–§4) |
| #837 | Streaming sink operator | ⬜ | **Alexander — second**, after the source (PR #914 §3–§4) |
| #838 | Streaming sink partitioning support | ⬜ | **Alexander** — design-sensitive (PR #914 §4) |
| #839 | Stream session support | ⬜ | **Alexander** — blocked on #836/#837 (PR #914 §3) |
| #840 | Resource management API | ⬜ | **Alexander** — ties deep into cuCascade; wants a design pass (PR #914 §6, "shared cuCascade manager") |
| #833 | StarRocks CI workflow | ✅ | Landed via #916 (`.github/workflows/experimental.yml`) |

**Everything else that's actually landed** (the tracking issue's 6 sub-issues undersell how
far this has come — most of the CN's current behavior isn't tracked by any of them):

| PR | What it added |
|----|---------------|
| #816, #832 | Initial CN process skeleton, made CN-only (dropped an initial BE-shaped shim) |
| #856 | Thrift RPC skeleton (heartbeat + `BackendService` stub) |
| #908 | `rust/crates/{sirius,sirius-sys}` — the Sirius FFI bindings |
| #941 | BRPC `PInternalService` fragment dispatch (translate-only at this point) |
| #960 | Bring up a real `sirius::SiriusContext` inside the CN process |
| #961 | MySQL client pixi task + local single-node cluster config |
| #962 | `FILES()` parquet schema inference (`get_file_schema`) |
| #1004 | Register the CN as a shared-data worker |
| #1006, #1025 | Descriptor-slot keying correctness fixes |
| #1007 | BRPC handlers can return a response attachment |
| #1008 | **First real execution**: a single `RESULT_SINK` fragment runs (via `StubExecutor`) and rows come back through `fetch_data` |
| #1009 | Compile the Substrait→DuckDB reader into `libsirius` |
| #1023 | Identify fragment instances by UUID |

**Open PRs — the current frontier:**

| PR | What it does | State / depends on |
|----|---------------|--------------------|
| #914 | `exchange-design.md` — streaming/nixl design proposal (docs only, draft) | Read it before touching #836–#840 |
| #959 | FE↔CN integration test harness (`cargo test` boots a real FE) | Draft; Matthijs finishing |
| #1021 | Translator emits `local_files` parquet reads (unblocks real file scans) | **Ready for review** |
| #1022 | FFI: one-shot "execute these Substrait bytes, return Arrow" + result collector | Stacked on #1021 |
| #1024 | Real `SiriusEngine` GPU executor replacing `StubExecutor` | Stacked on #1021 + #1022 — **check this branch out to test end-to-end** |

Three session notes on reading this table: Matthijs's in-flight plumbing is intentionally
*not* tracked as issues (it's structure, not design work), so the PR list — not the issue
list — is where his current work shows up. The `experimental` CI check is currently red on
a formatting nit in the stacked PRs — known and deliberately ignored until the stack lands,
since stacked branches get rebased anyway. And a nixl agent for the CN is in progress on
his machine with no PR yet (the nixl build is non-trivial).

## 4. Repository map

| Path | Role |
|------|------|
| `experimental/starrocks/starrocks/` | Git submodule → upstream `starrocks/starrocks`. Supplies the Thrift/Proto IDL under `starrocks/gensrc/{thrift,proto}` that get code-generated below, and the Java FE source the `fe` pixi env builds. |
| `experimental/starrocks/brpc/` | Git submodule → upstream `apache/brpc`. Only its `.proto` files are used, to generate PRPC frame-metadata types. |
| `experimental/starrocks/build.rs` | Runs `prost_build` over the BRPC + StarRocks protos; custom `BrpcServiceGenerator` emits the `PInternalService` Tower-service facade + router (the `tonic-build`-equivalent for BRPC). |
| `experimental/starrocks/src/main.rs` | Binary entrypoint — CLI args, engine bring-up, starts all three servers, FE registration with retry, graceful shutdown. |
| `experimental/starrocks/src/lib.rs` | `ComputeNodeConfig`/`FeConfig`; the heartbeat thrift service (`ComputeNodeHeartbeatHandler`); the `BackendService` thrift skeleton (`ComputeNodeBackendHandler` — mostly `NOT_IMPLEMENTED` stubs); MySQL-protocol FE registration. |
| `experimental/starrocks/src/{brpc,prpc}.rs` | The BRPC transport itself: raw Baidu PRPC TCP framing, no gRPC/HTTP2. |
| `experimental/starrocks/src/compute_node_service.rs` | `SiriusComputeNodeService` — implements `exec_plan_fragment`, `exec_batch_plan_fragments`, `fetch_data`, `get_file_schema`. This is where the real logic lives. |
| `experimental/starrocks/src/fragment_executor.rs` | `FragmentExecutor` trait + `StubExecutor` (see §2.4). |
| `experimental/starrocks/src/result_encoder.rs` | Arrow `RecordBatch` → StarRocks MySQL text-protocol rows. |
| `experimental/starrocks/src/result_store.rs` | Buffers results between `exec_plan_fragment` (producer) and FE's `fetch_data` polling (consumer), keyed by `FragmentInstanceId`. |
| `experimental/starrocks/src/file_schema.rs` | Parquet footer → StarRocks slot descriptors for `FILES()`. |
| `experimental/starrocks/crates/starrocks-thrift/` | `build.rs` shells out to the `thrift` compiler over the submodule's IDL; generated bindings re-exported from `src/lib.rs`. |
| `experimental/starrocks/crates/starrocks-plan-translator/` | `PlanTranslator` — StarRocks thrift plan/expr trees → Substrait. Read its `src/lib.rs` module doc before adding a node type. |
| `experimental/starrocks/conf/fe.conf`, `pixi.toml` | FE runtime config; pixi environments/tasks (§5). |
| `rust/crates/sirius-sys/` | Low-level `cxx` bindings to `src/include/sirius_ffi.hpp`. |
| `rust/crates/sirius/` | Safe wrapper (`SiriusContext::new()`/`from_config_file()`, RAII teardown) — what `SiriusEngine` (#1024) holds. |
| `src/include/sirius_ffi.hpp` + `src/sirius_ffi.cpp` | The actual public C++ surface being grown into a standalone `libsirius`; today a minimal RAII `Context` around `duckdb::SiriusContext`. |

## 5. Build & run locally

Two submodules are the minimum for any Rust CN work:

```bash
git submodule update --init --recursive experimental/starrocks/starrocks experimental/starrocks/brpc
```

Pixi environments (`experimental/starrocks/pixi.toml`): `fe` (JDK 17 + Maven — builds the
Java frontend), `cn` (Rust + `thrift-compiler` + `libprotobuf` — builds the Rust CN),
`client` (the `mysql` CLI), `engine` (CUDA 13 / cudf / rmm / duckdb — links the real GPU
engine; CUDA-pinned platform, so it needs a GPU box or `CONDA_OVERRIDE_CUDA` set).

**Fast loop — no GPU, no build tree, what CI runs** (`.github/workflows/experimental.yml`):

```bash
cd experimental/starrocks
pixi run -e cn cargo fmt --package sirius-starrocks-cn --package starrocks-plan-translator --package starrocks-thrift -- --check
pixi run -e cn cargo clippy --all-targets --no-default-features -- -D warnings
pixi run -e cn cargo test --workspace --no-default-features   # = the `cn-test-no-engine` task
```

**Full GPU loop** (needs a GPU box + CUDA toolchain):

```bash
pixi run -e cn engine-build   # builds libsirius via the repo-root `pixi run make`
pixi run -e cn cn-build       # builds the engine-linked CN binary
pixi run -e cn cn-test        # runs tests including the GPU-requiring ones
```

**Run a real local FE+CN cluster:**

```bash
pixi run -e fe fe-build       # packages the StarRocks FE (slow, JDK/Maven)
pixi run -e cn cluster        # starts FE + the CN together in the foreground (depends on fe-build, cn-build)
pixi run -e client client     # in another terminal: mysql CLI against the FE (port 9030)
```

The setup script (`experimental/starrocks/scripts/setup-dev-env.sh`) wraps the fast loop by
default and can pull in the FE/engine steps with flags — see its `--help`.

### 5.1 Recommended day-to-day loop: three terminals

`cluster` is convenient for a first end-to-end check, but it runs FE and CN as one
foregrounded process — both logs interleaved, and stopping it kills both. Day to day it's
more useful to run FE, CN, and the client in three separate terminals, so you can restart
the CN alone while iterating and keep FE's logs separate from the CN's:

```bash
cd experimental/starrocks
git submodule update --init --recursive   # once; initializes every submodule this repo needs
```

```bash
# Terminal 1 — build everything once and confirm FE+CN come up together...
pixi run -e cn cluster
# ...then Ctrl-C and use this to just restart the already-built FE on its own:
pixi run -e fe fe-run
```

```bash
# Terminal 2 — MySQL client against the FE (port 9030)
pixi run -e client client
```

```bash
# Terminal 3 — run the CN standalone, pointed at the built Sirius engine's shared library
LD_LIBRARY_PATH="$PWD/../../build/release/extension/sirius:${LD_LIBRARY_PATH:-}" pixi run -e cn cn-run
```

**Known blocker (temporary) — why terminal 3 needs `LD_LIBRARY_PATH`:** in the onboarding
pairing session, `pixi run cluster` / `cn-run` failed to load the Sirius engine — the build
looked for a **static** library while the build tree only produces the shared
`sirius.duckdb_extension` (`.so`). Until Matthijs fixes the `sirius-sys` cxx-crate build to
link a local build correctly, the workaround is exactly the env var above: point it (as an
**absolute** path — `$PWD/../..` expands to one) at
`<sirius-repo-root>/build/release/extension/sirius`, set only on the `cn-run` invocation.
Don't persist it in shell profiles or pixi config; the fix removes the need.
(`rust/README.md` documents the same requirement for `cargo test`.) One more sharp edge:
run every pixi task **from `experimental/starrocks/`** — from anywhere else you silently
pick up the repo-root pixi manifest instead of this workspace's.

**Verify real end-to-end execution:** check out PR #1024's branch (it carries #1021 and
#1022), rebuild, and in the client (terminal 2):

```sql
SHOW COMPUTE NODES;   -- the Sirius CN should be listed and alive
SELECT * FROM FILES('path' = 'file:///absolute/path/to.parquet', 'format' = 'parquet');
```

The path must use the `file://` scheme with an **absolute** path (three slashes total).
Two expected-failure modes worth recognizing: on `dev` or #1021 alone, the select fails
with "plan node type 9 not supported" — expected, it needs #1022/#1024; and rows that come
back as the literal string `stub` mean the fragment ran through the `StubExecutor` (§2.4),
not the GPU engine.

## 6. Contribution workflow

- Branch and open PRs against **`dev`** (repo-wide convention).
- Observed commit/PR title convention: `feat(starrocks): ...`, `fix(starrocks): ...`,
  `refactor(starrocks): ...`, `docs(starrocks): ...`, `test(starrocks): ...`, `build: ...`.
  Dependency bumps read `build(deps): bump X ... in /experimental/starrocks` (Dependabot).
- The `starrocks` GitHub label tags every issue/PR for this project — filter with
  `is:pr label:starrocks` / `is:issue label:starrocks` on `sirius-db/sirius`.
- CI gate is `.github/workflows/experimental.yml`, triggered on changes under
  `experimental/**`. It only runs the `--no-default-features` (pure-Rust, no engine/GPU)
  path on a CPU GitHub runner — the engine-linked path (`cn-build`/`cn-test`) has to be
  validated manually on a GPU box before merging anything that touches the `sirius-engine`
  feature.
- PRs stack on each other while a feature is mid-flight (e.g. #1024 explicitly says
  "Stacked on #1008, #1022, #1021 ... this PR's diff includes their commits") — check a PR's
  description for a "Stacked on" note before reviewing it in isolation. The new streaming
  operators (#836/#837) are the explicit exception: keep them standalone and unwired
  (see §7).
- Several PRs in this project so far were authored with Claude Code assistance (bodies
  signed "🤖 Generated with Claude Code" / "🤖 Drafted by an AI agent (Claude Code) at the
  direction of @mbrobbel") — worth knowing since PR descriptions here tend to be unusually
  thorough "what/how/validation" write-ups; match that style.

## 7. Where to start

Ownership was settled in the onboarding session, replacing the earlier "coordinate before
claiming" guidance: **the streaming/exchange work (#836–#840) is Alexander's**. Matthijs
keeps the plumbing — the #1021→#1022→#1024 stack, the nixl agent for the CN, the #959 test
harness — and will build the follow-ups that wire the operators in (streaming-session FFI
plus the `sirius`-crate wrappers). He's handing these five off deliberately: they're the
pieces that need real C++/distributed-systems design judgment, not more plumbing.

In order:

1. **Read PR #914 carefully.** It is the reference for everything below — the streaming
   operators, the cuCascade data repository + channel, and the nixl backpressure story.
   For background, the StarRocks docs (good, per Matthijs) cover FE/plan-fragment
   mechanics, and the Dynamo/nixl docs cover the transport.
2. **Finish local setup and prove the loop** (§5.1): `SHOW COMPUTE NODES;` lists the CN,
   and on a #1024 checkout a `FILES('file:///…')` select returns real rows.
3. **Implement the streaming source operator (#836), then the sink (#837).** Each is an
   isolated `sirius_physical_operator` subclass in Sirius core with its own unit tests, in
   a **standalone PR** — do *not* wire them into the CN yet, and do *not* stack them on
   Matthijs's PRs; wiring is a later follow-up item. Since these are Super Sirius operators
   (`src/op/`), read `docs/super-sirius/` first (`operators.md`, `task-creator.md`,
   `data-management.md` are the load-bearing three — see
   [discoveries.md](discoveries.md) for a code-level map gathered while scoping this). The
   agreed sequencing afterwards: operator → expose via FFI → add to the Rust `sirius`
   crate → hook into plan translation → hook into execution. #839 (stream session) sits on
   top and stays blocked until both operators exist.
4. **Then #838 (partitioning) and #840 (resource management).** #840 ties deeply into
   cuCascade memory management and explicitly wants a design pass (PR #914 §6); it is also
   where to scope the streaming config surface early (min/max exchange batch size etc. —
   additive and engine-agnostic, per §2.3).
5. **Reorganize the project board.** Everything sits at P2 and the board has been
   maintained ad hoc (Bobbi/Rodrigo/William); Alexander volunteered to own it and Matthijs
   agreed ("just go for it — make sure we're tracking things better"). Tracking issue #826
   is the anchor; its sub-issues exist for visibility — including for Alibaba to watch
   progress. Draft the reorg, then review it with Matthijs.

## 8. Goals & timeline

- **North star: Alibaba buy-in.** The earlier Doris experiment came out only marginally
  faster than CPU, for two understood reasons: no compute/exchange overlap, and the I/O
  overhaul hadn't landed yet. Alibaba was still convinced enough to redirect the effort to
  StarRocks rather than keep tuning Doris — and that history is why this work happens on
  `dev`, so every Sirius improvement automatically reaches the StarRocks path.
- **First internal goal: run all of TPC-H across multiple compute nodes, correctly** —
  potentially single-node multi-GPU, but the real target is multi-node on an NVL72, with
  StarRocks as the driver. Only after correctness comes measuring the speedup and going
  back to Alibaba.
- **"Make it work first, then make it fast."** No early optimization and no benchmark
  baseline yet — conceptually nothing here should be much slower than direct Sirius
  execution except the exchange itself, and nixl owns path selection there. Performance
  comparison (implicitly against CPU StarRocks) is explicitly deferred until it works.
- **Always keep it demoable.** If Alibaba wants to try it earlier than planned, it should
  just work and be simple to test.

## 9. Further reading

- `experimental/starrocks/docs/doc-plan.md` — local notes combining (a) the engineering
  sync that set StarRocks-over-Doris priority and staffing, and (b) a copy of PR #914's
  design proposal. (Originally saved inside the `starrocks/` submodule's checkout path,
  which blocked that submodule from ever being initialized — moved here for that reason.)
  It's local scratch context, not a tracked/citable doc — not committed anywhere, and not
  covered by CI or review; treat it as background reading, and prefer this doc or the live
  PR/issue numbers above when the two disagree.
- [PR #914](https://github.com/sirius-db/sirius/pull/914) — the canonical, up-to-date
  version of the streaming/nixl/resource-management design (draft; will move to
  `experimental/starrocks/docs/exchange-design.md` on merge).
- [Issue #826](https://github.com/sirius-db/sirius/issues/826) — the tracking issue; check
  it for the current sub-issue list before relying on the [status board](#3-status-board)
  above, which is a point-in-time snapshot.
- [sirius-engine-onboarding.md](sirius-engine-onboarding.md) — the Sirius engine-internals
  companion to this doc: architecture mental model, design-decision history (with PRs), the
  module map, and the invariants checklist for new operators. Read it before starting
  #836/#837 if the engine still feels opaque.
- [discoveries.md](discoveries.md) — code-level field notes on the Sirius engine internals
  (operator base class, task-creation hints, cuCascade data/memory APIs, test patterns)
  gathered while scoping #836/#837; the fastest way from this doc into the actual code.
- [StarRocks documentation](https://docs.starrocks.io/) — good (Matthijs's assessment) for
  FE architecture and plan-fragment background.
- [nixl](https://github.com/ai-dynamo/nixl) — the exchange transport (from the Dynamo
  project); read alongside PR #914 §5.
