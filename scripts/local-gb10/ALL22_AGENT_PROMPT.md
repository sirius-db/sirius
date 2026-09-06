# Task: build, profile, and benchmark Sirius all22/integration

Work autonomously through setup, build, validation, profiling, and the benchmark runs below. Do the work, not just a plan. Do not ask for routine confirmations. Adapt reversible setup steps to this machine, preserve existing work, and record your decisions. Do not push, open PRs, merge, or publish anything unless I ask.

## 1. Context and reference results

Repository: https://github.com/aocsa/sirius.git
Branch: `all22/integration`.

This branch integrates the Sirius GPU engine with an experimental StarRocks FE and Rust compute nodes. A standalone Sirius process is different from a StarRocks deployment with one CN. Multi-CN exchange uses native GPU batches and NIXL/UCX; a healthy FE alone does not prove the engine or transport works.

Reference measured on **2026-09-05**, commit **95bec853b684e1510c7ddb3d9becc9b73374e983**:

- Ubuntu 24.04 aarch64, one NVIDIA GB10 (SM121), driver 580.126.09, 119 GiB shared CPU/GPU memory.
- Engine: CUDA 13.2, cuDF 26.06. Native transport: CUDA-enabled UCX 1.21.0 and NIXL 1.3.2. Python DuckDB oracle: 1.5.5.
- SF500: **47 Parquet files, 140,988,717,545 bytes / 131.306 GiB**, **4,330,028,272 total rows**, including **3,000,028,242 lineitems**. Monetary columns were DECIMAL128(15,2), dates DATE32. Generator commit: `cdcf74def0072f94bf1886667e8d2ac51feb8721` from `sirius-db/tpchgen-rs`.
- Standalone Sirius: **22/22 passed**, **244.319 seconds** summed execute-plus-fetch time, **567,682 result rows matched exactly** after Decimal normalization. GPU execution was verified, with whole-query DuckDB fallback disabled. Budget: 48 GiB GPU + 16 GiB host, disk spill enabled.
- StarRocks, **two CNs sharing that one GB10**: **15/22 passed**. **Q5, Q7, Q8, Q9, Q17, Q18, Q21** failed from GPU memory exhaustion. Each CN had 24 GiB GPU + 8 GiB host + 2 GiB exchange staging and a separate 512 GiB disk spill limit. Every query was attempted; failures were followed by clean restarts.
- Those failures involved retained sender/inbound exchange buffers that were not convertible by the downgrade executor. Configuring engine disk spill did not make exchange buffers spillable. Q5's raw projected lineitem output alone was approximately 39 GiB per CN before temporary allocations. A "sender output was discarded" message can be cleanup after the original OOM, not a transport timeout.
- The reference build passed 434 CPU CN/translator tests and seven selected GPU rounding cases with 64 assertions. These are historical checks, not a substitute for validating your build.

These reference timings were **one measured attempt per query**, not cold/warm medians or a formal TPC-H benchmark. No 92 GiB single-CN arm was measured in that run. Treat the numbers as diagnostic context, not guaranteed performance on another machine or commit.

The reference SQL came from `test/tpch_performance/queries.py`, with identical Q8/Q9 join-order adaptations, Q11's SF500 fraction, and Q22's substring syntax on both engines. The StarRocks TPC-H kit can use different fixed parameters and tie ordering. Do not expect identical answers or row totals unless SQL, parameters, schema, generator, and data match. Generate an oracle for the exact SQL you run.

If available, inspect `build/tpch-sf500-summary.md`, the archived query SQL/results, and `scripts/local-gb10/README.md`. These machine-local artifacts/helpers may not exist in a fresh clone; do not depend on them being tracked upstream.

## 2. Read first

Read applicable `AGENTS.md` and `CLAUDE.md` instructions, especially `.claude/CLAUDE.md`, then:

1. `experimental/starrocks/DEMO.md`
2. `experimental/starrocks/docs/TUNABLES.md`
3. `experimental/starrocks/benchmarks/tpch/README.md`, `bench.sh`, `QUERY-DEVIATIONS.md`, and the actual `queries/qNN.sql`
4. `experimental/starrocks/benchmarks/cluster8.sh`, `scripts/cn-env.sh`, and `tools/oracle.py` / `tools/compare.py`
5. `docs/super-sirius/README.md`, `memory-management.md`, `pipeline-execution.md`, `configuration.md`, `streaming-sessions.md`, and `streaming-fragments.md`
6. Root `pixi.toml`, `experimental/starrocks/pixi.toml`, and the tasks they invoke.

Resolve stale doc statements against this checkout's code. Search for moved files rather than inventing their contents. Do not modify `src/legacy/`. Read the architecture/API context before any engine change.

## 3. Inventory, ownership, and build

Create `STATUS.md` immediately and update it throughout. Record host/OS/architecture, CPU count, RAM and available memory, GPU model/UUID/index/VRAM, shared versus discrete memory, driver/CUDA versions, free disk, installed toolchains, repository state, and existing GPU processes/listeners.

Before GPU tests or cluster startup, acquire an advisory GPU lock keyed by physical GPU UUID for every GPU used, in a stable order. Keep locks for the whole experiment. A lock file alone is insufficient: inspect `nvidia-smi`, process command lines, listening ports, FE registrations, and foreign CNs. Do not benchmark alongside another workload or kill unrelated processes. You may replace our prior benchmark stacks only when PID/run manifests and current command lines establish ownership. Never use broad `pkill` patterns from example docs. Stop with a clear resource-blocker description if foreign work prevents isolation.

Clone into a fresh directory, or preserve an existing checkout and use an isolated worktree if needed:

```bash
git clone --branch all22/integration https://github.com/aocsa/sirius.git
cd sirius
git submodule update --init --recursive
git rev-parse HEAD
git submodule status --recursive
```

Record the resolved branch SHA; do not silently equate a newer branch head with the reference commit. Preserve lockfiles and use frozen Pixi resolution when possible. Build in this order:

```bash
pixi run --frozen make
cd experimental/starrocks
pixi run --frozen apply-starrocks-patches
pixi run --frozen fe-check
# If the matching FE package is absent/incompatible:
pixi run --frozen fe-build
pixi run --frozen cn-build
pixi run --frozen cn-test-no-engine
pixi run --frozen cn-test
```

Adapt task syntax to the checked-out manifests. Reapply the idempotent StarRocks patch after clean submodule updates. Build the engine-linked CN with real NIXL, not a no-engine binary or stub transport. Run appropriate engine tests/smokes and verify that selected tests actually ran.

Build gotchas:

- Set CUDA architecture for the actual GPU; GB10 used `CUDAARCHS=121-real`. Bound build parallelism by available CPU/RAM; this box used six engine build jobs.
- On aarch64, build/use native UCX/NIXL and native compiler/linker paths. Inspect any architecture shims or transferred wrappers before using them; keep necessary wrappers local and documented. Do not reuse x86 binaries or fake successful linking with placeholder libraries.
- Source `scripts/cn-env.sh` and resolve `TOOLS_DIR`, `NIXL_PREFIX`, `UCX_PREFIX`, plugin paths, driver libraries and libclang headers. Missing external transport installs must be built before `cn-build`; its task does not provision them automatically.
- Keep `NIXL_NO_STUBS_FALLBACK=1`. Check `readelf`/`ldd`, native plugin loading, and CUDA-capable UCX. Same-host transport needs `cuda_copy` and `cuda_ipc` alongside the appropriate control/network transports.
- Pixi/login shells and `cn-env.sh` can reorder PATH. Check the actual Python, Cargo, compiler and linker used; use explicit environment executable paths when necessary. Avoid mixing a conda sysroot with system libc/linker flags. Do not casually rewrite lockfiles or invalidate Cargo caches by changing global `RUSTFLAGS`.
- **Unset inherited `CUDA_VISIBLE_DEVICES`; do not use it to place CNs.** Use `--gpu-device` and verify physical UUID/PID placement. Inspect/adapt launchers that set a mask; otherwise multiple CNs can silently land on the same GPU.
- Preserve build logs, tool versions, test counts, source/dirty state, and hashes of the actual extension/CN binaries. Confirm runtime processes load those artifacts.

## 4. SF500 data and CPU oracle

Reuse the verified SF500 dataset if present and unchanged. Otherwise generate into a new directory, using the pinned generator above when reproducing the reference. Preserve existing data. Record generation options, schemas, partition layout, exact table counts and per-file hashes; decode all Parquet columns/pages before calling the dataset fully verified.

Important: `bench/common/gen-tpch.sh` defaults to floating-point monetary columns. Use `--decimal` to reproduce the reference decimal schema. Its normal count/schema checks are weaker than full decoding/checksums, and its shard layout can differ from the 47-file reference. Record differences instead of claiming identical data. Inspect its `TPCHGEN`, `TPCH_PYTHON`, output-path and overwrite behavior before running.

Use the kit's documented SQL for the new arms, or a clearly named reference-reproduction query set. Freeze one chosen query set across all arms. Export `TPCH_SF=500` for both benchmark and oracle; substitute `__TPCH_DATA__` and **`__TPCH_SF__`**, and inspect Q11 to confirm `0.0001 / 500`. Preserve `QUERY-DEVIATIONS.md` and explicit tie ordering. Avoid inline SQL `--` comments in command-line query strings.

Generate the oracle through an isolated Python environment containing the **pip `duckdb` package**, without loading Sirius. Verify interpreter/package version and module path; do not accidentally use a Sirius-linked CLI as the CPU oracle. Run commands through Pixi with that interpreter explicitly if necessary. Size CPU threads/memory/temp storage; do not inherit `oracle.py`'s large-machine defaults of 48 threads / 380 GB.

From `experimental/starrocks/`, the command shape is:

```bash
TPCH_SF=500 ORACLE_THREADS=4 ORACLE_MEM=8GB ORACLE_TMP="$ORACLE_TMP" \
  "$ORACLE_PYTHON" tools/oracle.py benchmarks/tpch/queries "$TPCH_DATA" "$ORACLE_DIR"
```

Those resource values worked on the reference box; adapt and record them. `oracle.py` can catch errors into `qNN.err` and still exit zero: require all 22 successful oracle files, no error files, and complete SQL/input fingerprints. Preserve the oracle and its manifest before another run can overwrite metadata.

Run `tools/compare.py <arm-results> <oracle-dir> 1e-6` against **every result**, including cold/recovery runs. Also verify exact counts/integers/text/NULLs with a typed comparison when needed: the kit comparator converts numeric cells to float and strips surrounding text whitespace. Preserve duplicate rows. Diagnose ORDER BY ties separately from value errors; any canonicalization must preserve multiplicity and be applied symmetrically. Never widen tolerance or change SQL silently to obtain a pass.

## 5. Cluster and sweep protocol

Plan these three StarRocks arms:

| Arm | Topology | FE CTE policy |
| --- | --- | --- |
| `1cn-92g-default` | One CN, target GPU pool **92 GiB** | Fresh FE defaults |
| `ncn-default` | N CNs with documented launcher defaults | Fresh FE defaults |
| `ncn-cte-reuse` | Identical to the preceding N-CN arm | Force `cbo_cte_reuse_rate=0` |

Select and record N and the physical mapping. Normally use one CN per available GPU. Reproducing today's two-CN GB10 setup means two CNs on **one** GPU, explicitly labeled; stock `cluster8.sh` does not support that mapping without adaptation. Distinguish these arms from standalone Sirius.

Do memory preflight before launching. Current `cluster8.sh` defaults are 64 GiB GPU + 128 GiB host + 8 GiB staging **per CN**; staging is outside the GPU pool. Add context, FE, other native allocations and safety headroom. On unified-memory systems, all GPU/host/staging allocations consume the same physical RAM. A 92 GiB pool is not automatically feasible on a 119 GiB GB10, and the stock N-CN defaults are unsuitable for it. If a requested arm cannot fit, mark it `RESOURCE_PREFLIGHT_BLOCKED`, explain the arithmetic, and run a clearly named feasible adapted variant. Do not label reduced limits as a measured 92 GiB/default arm. Keep N-CN default/reuse variants identical except for the CTE policy.

Give each CN unique ports, explicit device assignment, separate engine/log/telemetry directories, and separate spill paths. Precreate spill directories. `--engine-dir` alone does not change process cwd; relative spill paths need separate cwd values or absolute paths.

Implement an owned-process launcher/restart hook with a PID ledger and fresh cluster-lifetime directories. Readiness after **every** start/restart requires:

- Exactly N intended alive CNs and zero unintended BEs; parse the actual `Alive` column and check twice. `MIN_BACKENDS` means expected count, not a floor. Do not enable `ALLOW_EXTRA_BACKENDS`.
- Correct physical GPU/PID mapping, listeners and binary/config identities.
- Successful peer warmup and bandwidth canary where remote exchange is required. Do not disable the canary to hide a broken transport.
- A small SQL smoke result and verified FE settings, without running the target benchmark query before its cold measurement.

Set a nonzero engine stall watchdog **below** both FE/client deadlines, leaving time for error propagation. For example, a 300-second no-progress watchdog, 900-second peer RPC budget, 1100-second FE query timeout and 1200-second client timeout are starting values, not universal prescriptions. Inspect current semantics and legitimate operator durations. Record resolved `SIRIUS_QUERY_WATCHDOG_SECS`, `SIRIUS_CN_RPC_TIMEOUT_SECS`, transfer timeout and all effective settings. A watchdog measures stalled progress, not necessarily total runtime.

FE globals persist and each `mysql -e` opens a new session. Reset/verify fresh defaults for default arms (documented CTE rate 1.15; verify the checked-out default), and apply `SET GLOBAL cbo_cte_reuse_rate=0` for the forced-reuse arm. Reapply and verify through a new client after **every FE restart**. A one-time session `SET` is insufficient. `FE_SETUP_SQL` is mentioned in the kit README but is not implemented by `cluster8.sh`; add an actual startup hook. Capture other relevant defaults, including pipeline DOP and aggregation policy.

Run Q1–Q22, **one recorded cold + two warm executions per successful query**, one query/workload at a time. Restart before each query's cold run. Define cold as a fresh application cluster after mandatory readiness/transport warmup; do not imply OS page-cache eviction or cold transport. Do not drop shared machine caches indiscriminately.

The kit invocation is shaped like:

```bash
TPCH_SF=500 TPCH_DATA="$TPCH_DATA" MIN_BACKENDS="$N" \
  COLD_TIMEOUT=1200 QUERY_TIMEOUT=1200 RESTART_CMD="$RESTART_CMD" \
  ORACLE_DIR="$ORACLE_DIR" \
  bash benchmarks/tpch/bench.sh --cold-restart "$ARM_RESULTS/timings.csv" 2
```

**Do not use that script unmodified for the required failure protocol.** At the reference commit it deliberately continues after a failed cold run without restarting. Its restart topology check is also weaker than its initial check. Make a small local wrapper/fix that:

1. Preserves every failed output/status before recovery and validates correctness before advancing.
2. Restarts owned FE/CNs after **every failure**, including cold failure, timeout, wrong result or lost node, and reruns the full readiness/settings gate.
3. Never calls the first execution after recovery "warm." Label recovery attempts separately. If a query cannot complete cold, mark its warm measurements explicitly skipped/unavailable; do not fabricate two warm results or retry indefinitely.
4. Continues with all remaining queries after deterministic failures. Never overwrite an original failure with a successful retry.
5. Produces an explicit **22 × 3 planned-run matrix per arm**, including failed/skipped slots. The comparator only discovers existing outputs, so its exit status alone cannot prove coverage.

Report cold time separately from the two individual warm times and their median. Never mix profiling/recovery timing into headline medians or publish a full-suite speedup when queries failed. Preserve exit codes, stdout/stderr, rows, comparison, SQL, settings, topology and cluster lifetime for every attempt.

Classify failures from the earliest causal evidence: setup/link/ABI, placement/topology, transport timeout, explicit peer refusal, GPU/host/staging/retained-exchange OOM, scheduler watchdog/stall, translation/unsupported operator, crash, wrong result, or incomplete instrumentation. A peer "failed with status" response is different from "failed to read reply frame." Raise RPC timeout only for an actual timeout diagnostic, preserving the baseline failure. Do not misclassify OOM cleanup as a timeout or mask deterministic engine limitations by changing memory/SQL midway through an arm.

## 6. Profiling and per-query analysis

Keep baseline timings separate from detailed profiling. Produce a profiling note for **every query in every arm**, even when execution failed or profiling is incomplete.

- Save FE `EXPLAIN COSTS` for the exact SQL and effective arm settings. Identify estimated versus actual cardinalities, join/broadcast/shuffle decisions, repeated scans, CTE multicast, aggregation stages and fragment count. FILES estimates can be misleading.
- Capture CN INFO logs: fragment start/finish/failure, query and fragment IDs, sender/receiver roles, native relay/NIXL transfers, bytes/batches, fusion decisions and reasons. Record resolved `fusion_mode` (`leaf` is the documented default). Fused senders legitimately lack separate fragment-start lines; missing log lines alone are not evidence of idle work.
- Capture engine plans and `[gpu_pool]` / `[host_pool]` allocated/peak/reserved data, OOM/reservation messages, downgrade bytes, reschedule/retry/futile-abort counters, and wait/stall evidence. Distinguish live allocation from a process-lifetime high-water mark and query-local peaks. Do not sum shared/overlapping allocations blindly.
- Set `SIRIUS_CN_DUMP_FRAGMENTS` to a **precreated, unique directory per CN and cluster lifetime**. It writes `fragment-NNNN.txt` and translated `plan-NNNN.substrait`; counters restart per process, so a shared directory overwrites evidence.
- For the distribution analyzer, configure NDJSON telemetry explicitly. Archive `<engine-dir>/telemetry/<run-uuid>/<record-type>/*.ndjson` after a graceful shutdown of owned processes; telemetry is buffered until shutdown. Mark crashes/forced-kill captures incomplete. Do not treat empty unflushed telemetry as zero GPU work.
- Analyze each lifetime separately using the correct directory prefix:

```bash
python scripts/cn-distribution.py --dir "$CAPTURE" --prefix .cn --metric entities
python scripts/cn-distribution.py --dir "$CAPTURE" --prefix .cn --metric entities --json \
  > "$CAPTURE/distribution.json"
```

The analyzer selects the newest UUID per CN by default. Do not pool lifetimes with `--all-runs`. Ensure all expected CN directories are represented, including missing/zero-output CNs. Distinct task/operator/batch entities measure work; raw event counts also include state transitions and retries.

- Correlate FE query IDs → fragment instance IDs → CN engine/telemetry query IDs. Report per-CN scan rows/bytes, task counts, exchange bytes, runtime/idle intervals and imbalance. "All CNs alive" or "each CN finished a fragment" does not prove balanced execution. Use structural IDs/log correlations, not timestamps alone.
- Optionally use `nsys` for selected bottlenecks or failures, with CUDA/NVTX and appropriate OS tracing. First inspect installed support and the actual launch command. Keep traces bounded, avoid requiring unavailable privileges, and never mix instrumented runtimes with baseline timing. Report unavailable profiling honestly.

Each query's `PROFILE.md` must state: correctness and cold/warm outcomes; plan/CTE/fusion shape; dominant operators; memory/reservation/retained-buffer evidence; per-CN participation and balance; exchange/wait/reschedule behavior; failure category and confidence; concrete artifact paths/line references; and the next useful diagnostic or optimization. Separate measurements from inference. Do not call generic warnings fatal without a causal error/result mismatch.

## 7. Deliverables and completion

Deliver:

- `STATUS.md`: current phase, machine inventory, versions/SHAs, ownership/locks, build/test results, exact commands, data/oracle readiness, requested versus adapted arms, blockers and recovery history. Update it during work and at completion.
- `results/oracle/`: frozen SQL/data manifests, CPU version/settings, all oracle answers and verification.
- `results/<arm>/`: config/settings snapshots, build identities, cold/warm matrix, per-attempt outputs/comparisons, failure ledger, EXPLAINs, per-CN logs, fragment dumps, archived telemetry/distribution, optional traces, and each query's `PROFILE.md`. Keep cluster lifetimes distinct.
- `RESULTS.md`: all 22 outcomes for each arm, cold and warm timings, valid warm medians, correctness/tolerance, CN balance, memory findings, default versus forced-CTE behavior, comparison limits relative to today's reference, reproducible commands, and prioritized findings. Clearly identify resource-blocked/skipped arms and incomplete suites. Include final service/lock state and owned stop/restart commands.

Resolve routine setup problems, run the complete feasible matrix, and investigate failures enough to classify them. Preserve baseline failures; put any engine fix or diagnostic tuning in a separately named experiment with appropriate tests. No pushes or PRs. Start by reading the repository instructions, inventorying the machine, and writing `STATUS.md`, then execute the workflow.
