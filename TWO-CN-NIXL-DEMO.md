# Two compute nodes, one GPU, the exchange hop over nixl — what was built and how to reproduce it

Branch **`demo-multi-cn`** (worktree `sirius-worktrees/integration`), 2026-08-05.
Hardware: one NVIDIA L4 (23 GiB), driver/runtime 13020, Ubuntu 24.04.

## 1. What was achieved

One SQL query's plan fragments execute across **two Sirius compute-node processes on one host**,
each holding a configured slice of the GPU, and the intermediate data crosses the process
boundary **GPU-to-GPU through nixl over UCX `cuda_ipc`** — no Arrow on the hop, no host
serialization of data. Only control frames (agent metadata, lease grants, per-batch signaling,
EOS) travel over brpc.

The finish-line run, verified live:

```
revenue: 61567694.95020001          -- exact decimal: 61567694.9502
```

with the four log lines that prove how the data moved:

```
nixl bandwidth canary  peer=127.0.0.1:8060  gbps="67.3"  bytes=16777216
transmitted batches via nixl  stream_id=2 sender_id=1 dest=127.0.0.1:8060 batches=1 bytes=457856
relayed native batches across a fragment boundary  stream_id=2 sender_id=0 batches=1
received remote batches  stream_id=2 sender_id=1 batches=1
```

The receiver aggregated a **cross-node fan-in**: sender 0 relayed in-process (native
`data_batch` pointer move), sender 1 arrived over nixl (457,856 packed bytes). `nvidia-smi`
showed both CNs at ~8.9 GiB each (8 GiB engine pool + 512 MiB exchange arena + CUDA context).
The last-ulp difference vs the single-node answer (`61567694.95019999`) is double-precision
summation order across the distributed scan; both round to the exact decimal.

### The commit stack (each independently tested before commit)

| Commit | What it delivers |
|---|---|
| `d6cce3ae` | `EngineConfig` GPU carve-outs (`--gpu-memory-limit/fraction/device`, `--host-memory-limit`, `--engine-dir` → derived YAML), the pre-priming guardrail, the `cluster2` pixi task |
| `34931de5` | Multi-file `FILES()` schema inference (first-file schema, fail-closed cross-file agreement) |
| `ca774e84` | Destination routing by `brpc_server` (hostname AND port — two CNs on one host are remote to each other), receiver execution moved off the RPC thread to a dispatch worker, `SenderSource` rendezvous enum |
| `3f7a9756` | The `cudaMalloc` exchange staging arena (`SIRIUS_EXCHANGE_STAGING_BYTES`), bump leases, 256-B alignment |
| `83d68072` | `Fragment::export_packed` / `push_packed` / `close_input` + `Context::staging_*` FFI — the device-resident fragment boundary (`cudf::chunked_pack` into a lease; zero-alloc `unpack` + copy-out-on-arrival), GPU-equivalence-tested against `relay_from` |
| `dead333b` | The nixl tier: one agent per CN (`{host}:{brpc_port}`), arena registered once, WRITE-based transfers with **receiver-granted leases** (every lease's lifetime stays process-local), per-batch signaling on brpc (`transmit_packed` with column names + pack metadata in the attachment), EOS on the existing rendezvous, and the **mandatory bandwidth canary**. Proto rpcs live in the vendored submodule commit `04cd3136` (`demo-multi-cn-proto`) |
| `afb8fbb3` | `fetch_data` long-poll — found by the first live run: async receiver dispatch made the "not-ready" reply common, and every reply consumes a packet-sequence slot in the FE's ResultReceiver (`expect=1, receive=0` cancel). Stock-BE semantics restored; timeout is a loud failure |
| `3473a686` | `DEMO.md` two-CN section |

### Why the design looks the way it does (the two load-bearing measurements)

- **Pool memory is a silent trap.** `cudaMallocAsync`/rmm-pool memory over UCX `cuda_ipc` does
  **not error — it silently degrades ~220×** (0.38 GB/s vs 85–90 GB/s, correct bytes, endpoint
  still advertising a cuda_ipc lane). Hence: a plain-`cudaMalloc` arena by contract, and a
  first-contact **bandwidth canary** that refuses the tier below 2 GB/s — nothing in nixl/UCX
  will ever flag this itself.
- **`cuda_copy` is mandatory in `UCX_TLS`** — it provides VRAM memory-type detection; without it
  `register_memory` fails with `NIXL_ERR_BACKEND`.

## 2. Prerequisites

### 2.1 The nixl/UCX install (already present at `sirius-worktrees/tools/`)

Built from source, no sudo: UCX **1.21.0** (`tools/ucx-install`, `ucx_info -d` lists `cuda_copy`
+ `cuda_ipc`) and nixl **v1.3.2** (`tools/nvda_nixl`; header `include/nixl.h`, libs under
`lib/x86_64-linux-gnu/`, UCX plugin under `.../plugins/`). `tools/nvda_nixl/ENV.sh` exports
everything. To rebuild from scratch:

```bash
# UCX 1.21.0 (release tarball ships ./configure — no autotools needed)
curl -LO https://github.com/openucx/ucx/releases/download/v1.21.0/ucx-1.21.0.tar.gz
tar xf ucx-1.21.0.tar.gz && cd ucx-1.21.0
./contrib/configure-release --prefix=$TOOLS/ucx-install \
  --with-cuda=<repo>/.pixi/envs/default/targets/x86_64-linux --enable-mt
make -j$(nproc) && make install

# nixl v1.3.2 (meson/ninja from a pip venv; pybind11 needed — python bindings are unconditional)
python3 -m venv $TOOLS/buildtools-venv && $TOOLS/buildtools-venv/bin/pip install meson ninja pybind11
git clone --depth 1 --branch v1.3.2 https://github.com/ai-dynamo/nixl.git $TOOLS/nixl-src
cd $TOOLS/nixl-src && PATH=$TOOLS/buildtools-venv/bin:/usr/local/cuda/bin:$PATH \
  meson setup build --prefix=$TOOLS/nvda_nixl -Ducx_path=$TOOLS/ucx-install \
  -Denable_plugins=UCX -Ddisable_gds_backend=true -Dbuild_tests=false \
  -Dbuild_examples=false -Dbuild_docs=false -Drust=false -Dnixl_cuda_arch_list=89
ninja -C build && ninja -C build install
```

### 2.2 The two-file scan data (already present at `/home/ubuntu/git/sirius/scratch/tpch_sf1/lineitem_multi/`)

Two **byte-identical** parquet files (74,139,347 B each, ZSTD; 6,001,215 rows total == sf1
lineitem). Byte-identical is load-bearing: the FE byte-splits any file that overflows
`totalBytes/numInstances`, and the CN rejects split ranges that don't reassemble within one
instance — equal sizes make splitting arithmetically impossible, so each CN gets one whole file
(deterministic cross-node scan) and the single-CN control still works. To regenerate:

```bash
# split by key, ZSTD (snappy totals >192MiB → 3 instances → splits; ZSTD keeps it at 2)
duckdb -c "COPY (SELECT * FROM read_parquet('.../lineitem/part.0.parquet') WHERE l_orderkey <  3000000)
           TO '.../lineitem_multi/part.0.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);"
duckdb -c "COPY (SELECT * FROM read_parquet('.../lineitem/part.0.parquet') WHERE l_orderkey >= 3000000)
           TO '.../lineitem_multi/part.1.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);"
# byte-equalize: insert a zero pad into the smaller file BETWEEN the last data byte and the
# footer (spec-legal: parquet offsets are absolute; the footer is found from the trailing 8B).
python3 - <<'PY'
import pathlib
small, large = sorted(pathlib.Path('.../lineitem_multi').glob('*.parquet'), key=lambda p: p.stat().st_size)
pad = large.stat().st_size - small.stat().st_size
b = small.read_bytes()
footer_len = int.from_bytes(b[-8:-4], 'little')
cut = len(b) - 8 - footer_len          # start of the footer metadata
small.write_bytes(b[:cut] + b'\0'*pad + b[cut:])
PY
# verify: sizes identical; glob count == original count; Q6 revenue == 61567694.9502 (CPU)
```

Validated on the GPU reader: the padded ZSTD file scans via `GPU_SCAN` with exact counts.

### 2.3 The packaged FE

`starrocks/output/fe` must exist (`pixi run fe-check` tells you; else
`git submodule update --init --recursive experimental/starrocks/starrocks && pixi run fe-build`).
Note the submodule must be on `demo-multi-cn-proto` (commit `04cd3136`) — it carries the three
CN-to-CN rpcs (`exchange_nixl_md`, `request_staging_lease`, `transmit_packed`).

## 3. Build

```bash
cd <worktree>                    # branch demo-multi-cn
git submodule update --init --recursive
pixi run make                    # the Sirius engine + FFI (~15 min cold)

cd experimental/starrocks
pixi run cn-build                # release CN, linked against real libnixl
```

The pixi tasks carry the required build env; if you build ad-hoc instead, you MUST export
**inside** the activated shell (pixi activation overwrites outer exports):

```bash
pixi run -e cn bash -c 'export \
  NIXL_PREFIX=/home/ubuntu/git/sirius-db/sirius-worktrees/tools/nvda_nixl \
  NIXL_NO_STUBS_FALLBACK=1 \
  BINDGEN_EXTRA_CLANG_ARGS="-isystem <worktree>/.pixi/envs/default/lib/clang/21/include" \
  CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/gcc && \
  cargo build --release -p sirius-starrocks-cn'
```

Why each: `NIXL_NO_STUBS_FALLBACK=1` — nixl-sys otherwise **silently builds a dlopen stub**;
bindgen needs clang builtin headers the system libclang lacks; the conda cross-gcc cannot
resolve libnixl's glibc-2.34/2.38 symbols, so cargo's linker is pinned to system gcc. Verify the
real link: `readelf -d target/release/sirius-starrocks-cn | grep nixl` → `libnixl.so`.

## 4. Run

```bash
cd experimental/starrocks
pixi run cluster2        # FE + CN1 (ports 9050/9060/8060) + CN2 (9052/9062/8062),
                         # each: --gpu-memory-limit 8GiB --host-memory-limit 12GiB,
                         # SIRIUS_EXCHANGE_STAGING_BYTES=512MiB (arena is OUTSIDE the 8GiB),
                         # UCX_TLS=cuda_copy,cuda_ipc,tcp,self, NIXL_PLUGIN_DIR set
```

Wait for both `nixl transport ready; staging arena registered agent=127.0.0.1:806{0,2}` lines
and `SHOW COMPUTE NODES` listing two alive nodes. Then, in another terminal:

```bash
pixi run client          # mysql to the FE
```

```sql
-- No session variable needed since the two-phase stack (64977ebb..11625add, 2026-08-05):
-- the FE's default plan (partial agg per CN -> gather -> merge finalize) runs natively, and
-- only one partial-state row per CN crosses the exchange (nixl log: bytes=64 vs the 457 KB
-- of filtered rows that `SET new_planner_agg_stage = 1` ships for the same query).
WITH lineitem AS (SELECT * FROM FILES(
  "path"="file:///home/ubuntu/git/sirius/scratch/tpch_sf1/lineitem_multi/*.parquet",
  "format"="parquet"))
SELECT sum(l_extendedprice * l_discount) AS revenue
FROM lineitem
WHERE l_shipdate >= date '1997-01-01' AND l_shipdate < date '1998-01-01'
  AND l_discount BETWEEN 0.03 - 0.01 AND 0.03 + 0.01 AND l_quantity < 24;
-- expect 61567694.95020001 (or ...95019999 depending on partial-sum order; exact: ...9502)
```

## 5. Verify (read the numbers, not just the answer)

| Check | Expect |
|---|---|
| Canary (first peer contact only) | `nixl bandwidth canary ... gbps=` **tens of GB/s** (healthy L4: 67–90). Below 2 GB/s the tier refuses loudly — that means the arena is not `cudaMalloc`-backed or IPC is broken |
| Sender CN | `transmitted batches via nixl stream_id=... batches=N bytes=M` with **N>0** — `batches=0` means the boundary carried nothing |
| Receiver CN | `received remote batches ...` AND `relayed native batches ...` (the cross-node fan-in: one local, one remote sender) |
| GPU | `nvidia-smi`: two CNs ≈ 8.9 GiB each; **0 compute apps after teardown** |
| Negative controls | Single-CN `pixi run cluster` + same query → same exact-decimal answer. Unset `SIRIUS_EXCHANGE_STAGING_BYTES` → a cross-node placement fails loudly naming the remedy (never a hang) |

Placement note: the gather receiver round-robins between the CNs per query, but with two
byte-identical files one scan instance is **always** remote from it — cross-node is guaranteed
every run; only the direction varies.

## 6. Known boundaries (by design, all loud)

- **Scalar `sum`/`count`/`min`/`max` aggregation only** on two CNs — now at the FE's default
  settings. `GROUP BY` fails loudly at any stage: the default grouped two-phase plan is refused
  in the translator (`grouped two-phase aggregation needs partitioned streaming output (#838)`),
  and at `agg_stage=1` the per-CN aggregate instances mean a hash shuffle → `a data stream sink
  with 2 destinations needs partitioned streaming output (#838)`. `avg` in a two-phase plan is
  refused too (opaque VARBINARY partial state). Partitioned output is the recorded next step;
  joins are behind the same guard.
- No merge-grade hardening (Path B scope): no cancellation/GC (restart CNs after failures), the
  arena sits outside the cuCascade budget (size it into your headroom), transmit stalls >60s
  trip the client timeout loudly.
- Same-host `cuda_ipc` is the transport; cross-host needs GPUDirect-capable fabric (the code
  path is the same — nixl/UCX chooses the wire).

## 7. Troubleshooting the traps we actually hit

| Symptom | Cause / fix |
|---|---|
| Query cancels with FE `ResultReceiver ... expect=1, receive=0` | You are running a CN binary older than `afb8fbb3` (`fetch_data` must long-poll) |
| Transfers "work" but canary reports <1 GB/s | Staging memory is pool-backed, not `cudaMalloc` — the silent 220× cliff; the canary exists precisely for this |
| `register_memory` fails `NIXL_ERR_BACKEND` | `cuda_copy` missing from `UCX_TLS` |
| CN links but nixl does nothing | nixl-sys built its silent stub — rebuild with `NIXL_NO_STUBS_FALLBACK=1` |
| Link error `GLIBC_2.38 not found` | conda cross-gcc linking — pin `CARGO_TARGET_..._LINKER=/usr/bin/gcc` (inside the pixi shell) |
| Second CN aborts at bring-up with rmm OOM | No carve-out configured — pass `--gpu-memory-limit` (the guardrail catches this pre-priming when nvidia-smi is available) |
| ≥128 MiB single-file scans fail on 2 CNs | FE byte-splitting vs the CN's whole-file rule — use the byte-identical multi-file layout (§2.2) |
| A "finished" run that never finished | Bound every run (`timeout --signal=KILL`); check `nvidia-smi --query-compute-apps` — `pgrep -f` misses filtered invocations and matches your own command line |
