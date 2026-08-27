# Building Sirius + StarRocks from scratch on a fresh box

Self-contained procedure to take a bare GPU machine to a working **Sirius-as-StarRocks-compute-node**
stack: the Sirius GPU engine (`libsirius`), the StarRocks front end (FE), and the Rust compute
node (CN) that links both plus the nixl/UCX exchange transport.

This is a **build** document. Operating the cluster and benchmarking it is the separate
`tpch-bench` skill; §9 here is only the smoke test that proves the build works.

Verified end to end on 2026-08-19 (see §12 for the exact box and results).

---

## 0. What you are building

Five artifacts, with a hard dependency order:

| # | Artifact | Produced by | Depends on |
|---|---|---|---|
| 1 | **UCX** → `$TOOLS_DIR/ucx-install` | autotools build from release tarball | CUDA 13, hwloc |
| 2 | **nixl** → `$TOOLS_DIR/nvda_nixl` | meson build from git | UCX (1) |
| 3 | **libsirius** → `build/release/extension/sirius/` | `pixi run make` at repo root | root pixi env (cudf/rmm/CUDA 13) |
| 4 | **StarRocks FE** → `experimental/starrocks/starrocks/output/fe/` | `pixi run -e fe fe-build` (Maven) | patched StarRocks submodule |
| 5 | **CN binary** → `experimental/starrocks/target/release/sirius-starrocks-cn` | `pixi run cn-build` | 1, 2, 3, and the patched proto |

**(1), (3) and (4) are mutually independent — build them concurrently.** They are the entire
wall clock. (2) follows (1); (5) is last and takes a couple of minutes.

The StarRocks submodule needs a **patch applied to its proto** before *either* the FE or the CN
builds — see §4.

**Everything is pinned to CUDA 13.** The engine's pixi env resolves `__cuda=13.0`, and UCX and
nixl must be pointed at the CUDA 13 toolkit explicitly (§5c, §6) because `/usr/local/cuda` is
frequently a symlink to an older one. §11 has the three commands that verify this.

---

## 1. Survey the box first

Everything below branches on what this prints. Record it — these numbers size the CN carve-outs
and decide whether §2 is needed.

```bash
nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv
nvidia-smi topo -m          # GPU-to-GPU link type
nvidia-smi topo -p2p r      # must be OK for every pair, or cuda_ipc cannot go GPU-to-GPU
nproc; free -g; df -h
lscpu | grep -iE 'numa|model name'
ls -d /usr/local/cuda*      # which CUDA toolkits exist; note where /usr/local/cuda points
ls /usr/lib/jvm             # a JDK for launching the FE later
sudo -n true && echo "have root" || echo "NO ROOT -- see §3"
```

Four findings change the procedure:

- **Driver older than r580.** The engine is built against **CUDA 13**, which needs r580+. On an
  r570 box every CN dies at its first device call with `cudaErrorInsufficientDriver`. Fix: put
  `/usr/local/cuda/compat` **first** on `LD_LIBRARY_PATH` (CUDA forward compatibility; supported
  on data-center GPUs).
- **GPU-to-GPU link reads `PIX`/`PHB`/`SYS` rather than `NV#`.** No NVLink. `cuda_ipc` still works
  over PCIe P2P provided `topo -p2p r` says `OK`, so the stack is fully functional — just never
  quote NVLink bandwidth from such a box.
- **No passwordless root.** The UCX/nixl build deps are normally `apt-get install`ed. Without
  root you take them from conda instead — §3. This works identically and touches nothing system-wide.
- **Small root filesystem.** Budget **~13 GB** of build output plus a **~7 GB** pixi package
  cache. If `/` cannot hold ~25 GB but a large scratch volume exists, do §2.

---

## 2. Relocate the heavy directories (only if `/` is small)

Symlink **before** anything is created — pixi, cargo and Maven all follow symlinks correctly.

```bash
BIG=/opt/dlami/nvme/sirius-build          # <-- your large volume
REPO=/home/ubuntu/sirius
mkdir -p $BIG/{pixi-cache,root-pixi,sr-pixi,cargo,m2,tools,build}

ln -sfn $BIG/root-pixi $REPO/.pixi
ln -sfn $BIG/build     $REPO/build
ln -sfn $BIG/sr-pixi   $REPO/experimental/starrocks/.pixi
ln -sfn $BIG/m2        ~/.m2
ln -sfn $BIG/tools     $(dirname $REPO)/tools      # TOOLS_DIR default: sibling of the repo root
```

Write one env file and **source it in every shell you use from here on**. The builds are long
and a forgotten `PIXI_CACHE_DIR` silently refills `/`:

```bash
cat > $BIG/env.sh <<'EOF'
export PATH=$HOME/.pixi/bin:$PATH
export PIXI_CACHE_DIR=/opt/dlami/nvme/sirius-build/pixi-cache
export CARGO_HOME=/opt/dlami/nvme/sirius-build/cargo
export PATH=$CARGO_HOME/bin:$PATH
export TOOLS_DIR=/home/ubuntu/tools
export JAVA_HOME=/usr/lib/jvm/java-21-amazon-corretto   # <-- whatever `ls /usr/lib/jvm` showed
EOF
source $BIG/env.sh
```

`TOOLS_DIR` is the one input `scripts/cn-env.sh` cannot derive. It defaults to a `tools/`
directory **next to the repo root**, which is why the symlink above uses that path.

---

## 3. Clone, submodules, pixi, build deps

```bash
git clone <repo-url> /home/ubuntu/sirius && cd /home/ubuntu/sirius
git submodule update --init --recursive        # ~2 min
curl -fsSL https://pixi.sh/install.sh | bash   # installs to ~/.pixi/bin
source $BIG/env.sh && pixi --version           # need >= 0.71
```

The **root** `git submodule update --init --recursive` is mandatory even if you only care about
the CN: `substrait`, `vcpkg`, `duckdb` and `cucascade` are unpopulated in a fresh clone, and the
engine cmake otherwise fails with `No SOURCES given to target: sirius_extension`. **Worktrees do
not auto-initialize submodules** — run it explicitly from inside one.

### Build dependencies without root

UCX and nixl need `meson ninja pkg-config hwloc pybind11`. With root, `apt-get install -y meson
ninja-build pkg-config libhwloc-dev libnuma-dev pybind11-dev` is the fast path. **Without root**,
stage them in a throwaway conda env:

```bash
mkdir -p $TOOLS_DIR/toolenv && cd $TOOLS_DIR/toolenv
pixi init .
pixi add meson ninja pkg-config libhwloc numactl pybind11 python
```

Two package-name traps, each of which costs a full round trip:

- The conda package is **`libhwloc`**, not `hwloc` — `pixi add hwloc` fails the solve with
  `No candidates were found for hwloc`.
- **`pybind11` and `python` are required**, even though nothing here uses nixl's Python bindings.
  nixl's `src/bindings/python/meson.build` is unconditional, so `meson setup` aborts with
  `ERROR: Dependency "pybind11" not found` without them. There is no meson option to skip it.

The env prefix — `$TOOLS_DIR/toolenv/.pixi/envs/default` — is referred to below as `$TENV`.

---

## 4. Patch the StarRocks submodule

The three Sirius-only exchange RPCs (`exchange_nixl_md`, `request_staging_lease`,
`transmit_packed`) are **not** in the submodule's upstream commit. They arrive as a working-tree
patch. **Both** downstream builds consume the patched proto: the FE's Maven build generates Java
stubs from it, and the CN's `build.rs` runs prost over the same
`gensrc/proto/internal_service.proto`.

```bash
cd /home/ubuntu/sirius/experimental/starrocks
pixi run -e fe apply-starrocks-patches      # prints "applied nixl-exchange-proto.patch"
```

Use `-e fe` so you patch without solving the heavy default env (which pulls libcudf). The script
is idempotent — it detects an applied patch via `git apply --reverse --check` and prints
`already applied`.

The submodule now shows permanently as `-dirty`. **That is by design.** Never `git add` the
submodule after patching — that records an unpushable local commit as the gitlink and breaks
`git submodule update` for everyone. `.gitmodules` sets `ignore = dirty`;
`git config diff.ignoreSubmodules dirty` silences it locally.

A `git submodule update` reverts the patch. Every task that needs it (`cn-build`, `cn-test`,
`cn-run`, `fe-check`) depends on `apply-starrocks-patches`, so re-running the build heals it.

---

## 5. The three concurrent builds

Launch all three at once — each in its own shell or background task.

### 5a. Sirius engine (`libsirius`)

```bash
source $BIG/env.sh
cd /home/ubuntu/sirius
pixi run make
```

Solves a ~6 GB pixi env first, then compiles CUDA + cudf (1308 ninja targets). The `default`
environment resolves the `linux-64-cuda13` platform automatically from the driver's `__cuda`
virtual package — nothing to select by hand, but verify it (§11).

### 5b. StarRocks FE (Maven)

```bash
source $BIG/env.sh
cd /home/ubuntu/sirius/experimental/starrocks
pixi run -e fe fe-build
```

The `fe` pixi feature brings its own `openjdk 17`, `maven` and `thrift-compiler 0.20` — you do
**not** need a system JDK for this step (`JAVA_HOME` from §2 is for *launching* the FE later).
The task patches `build-support/gen_notice.py`'s shebang, fakes a
`.pixi/starrocks-fe-thirdparty/installed/llvm/lib/libLLVMInstCombine.a`, calls
`./build.sh --fe`, and copies `conf/fe.conf` over the packaged one — all inside the task.

Output lands at `experimental/starrocks/starrocks/output/fe/`. Older notes call this a
"multi-hour" build; measured here it is **~4 minutes**. If you have a box that already built it,
copying that directory over is valid — and a **stock FE of the same StarRocks version also
works**, since the Sirius patch only adds CN↔CN RPCs (the submodule tracks `branch-4.1.1`).

### 5c. UCX

Build UCX with the **system** gcc, taking only meson/ninja/hwloc from `$TENV`. Point
`--with-cuda` at the **CUDA 13** tree explicitly:

```bash
source $BIG/env.sh
TENV=$TOOLS_DIR/toolenv/.pixi/envs/default
export CUDA_HOME=/usr/local/cuda-13.0        # NOT /usr/local/cuda -- often an older symlink
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export PATH=/usr/bin:$TENV/bin:$PATH

cd $TOOLS_DIR
wget -nc https://github.com/openucx/ucx/releases/download/v1.21.0/ucx-1.21.0.tar.gz
tar xf ucx-1.21.0.tar.gz && cd ucx-1.21.0
./configure --prefix=$TOOLS_DIR/ucx-install \
            --with-cuda=$CUDA_HOME --with-hwloc=$TENV --enable-mt
make -j$(nproc) install
ls $TOOLS_DIR/ucx-install/lib/ucx/libuct_cuda.so   # must exist, or cuda_ipc is unavailable
```

`--enable-mt` is **required** — the nixl agent is touched from a dedicated thread.

---

## 6. nixl

Follows UCX. Its meson build rejects include paths containing `..`, so feed `realpath`'d
absolutes. Build **only the UCX plugin**: it is the one the exchange path uses, and every other
backend drags in a dependency you would otherwise have to satisfy (GDS needs `libcufile-dev`,
which needs root).

```bash
source $BIG/env.sh
TENV=$TOOLS_DIR/toolenv/.pixi/envs/default
export CUDA_HOME=/usr/local/cuda-13.0
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export PATH=/usr/bin:$TENV/bin:$PATH
export PKG_CONFIG_PATH="$TOOLS_DIR/ucx-install/lib/pkgconfig:$TENV/lib/pkgconfig:$PKG_CONFIG_PATH"

cd $TOOLS_DIR
git clone --depth 1 https://github.com/ai-dynamo/nixl nixl-src && cd nixl-src
rm -rf build
meson setup build \
  --prefix="$(realpath -m $TOOLS_DIR/nvda_nixl)" \
  -Ducx_path="$(realpath $TOOLS_DIR/ucx-install)" \
  -Dcudapath_inc=$CUDA_HOME/include \
  -Dcudapath_lib=$CUDA_HOME/lib64 \
  -Dcudapath_stub=$CUDA_HOME/lib64/stubs \
  -Denable_plugins=UCX \
  -Ddisable_gds_backend=true -Ddisable_mooncake_backend=true -Ddisable_infinia_backend=true \
  -Dbuild_tests=false -Dbuild_examples=false -Dbuild_docs=false
meson compile -C build && meson install -C build
ls $TOOLS_DIR/nvda_nixl/lib/*-linux-gnu/plugins/    # MUST contain libplugin_UCX.so
```

That last `ls` is the most important check in this section — a missing UCX plugin is the most
common bring-up failure, and it surfaces much later as an obscure agent-creation error.

The three `cudapath_*` options are how you pin nixl to **CUDA 13**. They default to empty, in
which case meson finds whatever `/usr/local/cuda` points at.

Notes:
- The build dir **must be on local disk**, not NFS — meson's clock-skew check fails on NFS.
- nixl vendors and builds `prometheus-cpp` and `asio` subprojects; that is expected, not an error.
- If `pkg-config` reports a system `libfabric` older than 1.21, `-Denable_plugins=UCX` sidesteps it.
- If you do build GDS, never enable `GDS` and `GDS_MT` in one agent — rejected at runtime.

---

## 7. The derived environment

`experimental/starrocks/scripts/cn-env.sh` computes every machine-specific path from its own
location plus `TOOLS_DIR`. **Source it, never execute it** — run as a child process it configures
a shell that immediately exits. It also *unsets* `CXXFLAGS`/`CFLAGS`/`CPATH`, so use a subshell
if the rest of your shell needs those.

```bash
cd /home/ubuntu/sirius/experimental/starrocks
( source scripts/cn-env.sh && echo "$NIXL_PREFIX" && ls "$NIXL_PLUGIN_DIR" )
```

It fails loudly rather than half-configuring: no nixl under `$TOOLS_DIR` tells you to fix
`TOOLS_DIR`; no clang headers in the repo pixi env tells you to run `pixi install` at the root.

What it handles, and why each matters if you ever build by hand:

| It sets | Because |
|---|---|
| `NIXL_NO_STUBS_FALLBACK=1` | otherwise a broken nixl link **silently** compiles a dlopen stub — you find out at runtime, or never |
| `UCX_TLS=cuda_copy,cuda_ipc,tcp,self` | without `cuda_copy` UCX cannot detect VRAM pointers (registration fails); without `cuda_ipc` transfers take a ~200× host bounce |
| `CC`/`CXX` = `/usr/bin/*`, conda `CXXFLAGS` cleared | pixi's conda flags mix the conda sysroot with `/usr/include` and die on `bits/timesize.h` |
| `PATH=/usr/bin:$PATH` (system `ld`) | conda's `ld` links the conda sysroot libpthread against the system libc → 39 `GLIBC_PRIVATE` undefined refs. Do **not** fix this with `RUSTFLAGS` — setting `RUSTFLAGS` invalidates cargo's fingerprint cache, re-runs the `nixl-sys` build script, and lands you in a *different* failure |
| `LIBRARY_PATH` incl. `/usr/lib/$(uname -m)-linux-gnu` | `libsirius.so` has `libcuda.so.1`/`libnvidia-ml.so.1` in `DT_NEEDED`; without the driver dir the CN link fails on `cuLaunchKernel` / `nvmlDeviceGetMemoryAffinity`. This is *link* time — `LD_LIBRARY_PATH` does not cover it |
| `LIBCLANG_PATH` + `BINDGEN_EXTRA_CLANG_ARGS` | `nixl-sys` bindgen needs libclang **and its builtin headers**; the system libclang ships none, the repo pixi env's clang does |

### aarch64 boxes need three extra shims

On aarch64 the vendored `nixl-sys` `build.rs` hardcodes `cc::Build::compiler("g++")`, overriding
`cn-env.sh`'s `CXX`, and the driver ships no `libnvidia-ml.so` symlink. Create a shim dir on
**persistent** storage (not `/tmp` — it must survive a reboot) and put it on `PATH` *inside* pixi:

```bash
SHIMS=$TOOLS_DIR/toolchain-shims && mkdir -p $SHIMS
ln -sf /usr/bin/g++ $SHIMS/g++
ln -sf /usr/bin/ld  $SHIMS/ld
ln -sf /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1 $SHIMS/libnvidia-ml.so
pixi run --manifest-path $PWD/pixi.toml bash -c "
  export PATH=$SHIMS:\$PATH
  export RUSTFLAGS=\"-C link-arg=-L$SHIMS -C link-arg=-lnvidia-ml\"
  source scripts/cn-env.sh && cargo build --release -p sirius-starrocks-cn"
```

An **outer** `PATH=$SHIMS:$PATH pixi run ...` is defeated — pixi prepends its own env bin to the
inherited PATH. That is why the export goes inside the `bash -c`, and why `pixi run cn-build`
cannot carry it. **On x86_64 none of this is needed**; use §8.

---

## 8. Build the compute node

Last, because it links artifacts 1–4.

```bash
source $BIG/env.sh
cd /home/ubuntu/sirius/experimental/starrocks
pixi run cn-build
```

`cn-build` depends on `engine-build` and `apply-starrocks-patches`, so it heals a reverted patch
and builds `libsirius` if you skipped §5a. With a warm tree it is ~40 s of cargo.

**Verify it is really nixl-linked, not stubbed:**

```bash
readelf -d target/release/sirius-starrocks-cn | grep -Ei 'nixl|sirius'
# expect NEEDED: libnixl.so, libnixl_build.so, sirius.duckdb_extension
```

If `nixl` is absent from that list, the stub path was taken: recheck `NIXL_NO_STUBS_FALLBACK=1`
and rebuild. (`ldd` shows the same plus resolved paths, but needs `cn-env.sh`'s `LD_LIBRARY_PATH`
or it prints "not found".)

---

## 9. Smoke test: one CN, one query, one oracle

Proves all five artifacts link and talk. Full cluster operation is the `tpch-bench` skill.

```bash
# terminal 1 -- BLOCKS. Give it its own shell; never chain it behind `&` in another command
# (its EXIT/INT trap tears the cluster down with that shell).
source $BIG/env.sh
cd /home/ubuntu/sirius/experimental/starrocks
unset CUDA_VISIBLE_DEVICES        # see §10
pixi run cluster 2>&1 | tee /tmp/cluster.log
```

```bash
# terminal 2 -- wait for the CN to register. Count the Alive column (9); `grep -c true` overcounts.
cd /home/ubuntu/sirius/experimental/starrocks
until [ "$(pixi run -e client bash -c "mysql -h127.0.0.1 -P9030 -uroot -N -e 'SHOW COMPUTE NODES;'" \
        2>/dev/null | awk -F'\t' '$9=="true"' | wc -l)" -ge 1 ]; do sleep 5; done; echo alive
```

Run a query against any TPC-H `lineitem` parquet you have:

```sql
SELECT sum(l_extendedprice * l_discount) AS revenue
FROM FILES("path"="file:///path/to/lineitem/part.0.parquet","format"="parquet")
WHERE l_shipdate >= date '1997-01-01' AND l_shipdate < date '1998-01-01'
  AND l_discount BETWEEN 0.02 AND 0.04 AND l_quantity < 24;
```

and a `GROUP BY`, which forces a two-phase aggregate and therefore a **fragment boundary**:

```sql
SELECT l_returnflag, count(*) AS n, sum(l_quantity) AS qty
FROM FILES("path"="file:///path/to/lineitem/part.0.parquet","format"="parquet")
GROUP BY l_returnflag ORDER BY l_returnflag;
```

Three things must all hold:

1. **The values match a CPU oracle.** There is no correctness gate anywhere in this stack — a
   query returning the wrong number looks exactly like one returning the right one. Check it:

   ```bash
   python3 -c "
   import duckdb
   print(duckdb.sql(\"SELECT sum(l_extendedprice*l_discount) FROM read_parquet('/path/to/lineitem/part.0.parquet')
     WHERE l_shipdate >= date '1997-01-01' AND l_shipdate < date '1998-01-01'
       AND l_discount BETWEEN 0.02 AND 0.04 AND l_quantity < 24\").fetchall())"
   ```

   Use a **plain** `duckdb` (pip), not `build/release/duckdb` — the repo's binary auto-loads the
   Sirius extension and will try to claim GPU memory the running CN already holds, failing with
   `cudaErrorMemoryAllocation`.

2. **The engine really initialized on the GPU** — `grep 'Sirius engine context created' /tmp/cluster.log`.

3. **Fragments crossed natively** — `grep 'relayed native batches' /tmp/cluster.log` (one line per
   boundary; the `GROUP BY` produces several).

Teardown, and the check that actually proves it:

```bash
pkill -f '[s]irius-starrocks-cn'; pkill -f '[S]tarRocksFE'
nvidia-smi --query-compute-apps=pid --format=csv,noheader   # must be EMPTY
```

Check **compute-apps, not `memory.used`** — the latter's idle floor is tens of MiB and never
reaches 0. Use the bracket pattern `'[s]irius...'` in `pkill` or you kill your own shell.

### Sizing a multi-CN launch

`benchmarks/cluster8.sh` runs one CN per visible GPU; its defaults (`GPU_MEM=64GiB
HOST_MEM=128GiB STAGING=8GiB`) target an 80 GiB A100.

The staging arena sits **outside** `--gpu-memory-limit`, so a CN really occupies
`GPU_MEM + STAGING + ~2–3 GiB CUDA context`. Size against what the GPU *reports*, and remember
that raising `STAGING` requires lowering `GPU_MEM` by the same amount. Scale `STAGING` with the
scale factor, not the GPU: 512 MiB starved TPC-H q09 at SF1 (one 648 MB packed-table lease) and
8 GiB starved q05 at SF100. Fewer CNs makes it worse — each carries more of the fan-out.

---

## 10. Traps that silently produce a working-but-wrong stack

- **`CUDA_VISIBLE_DEVICES` must be unset before launching.** An already-exported value **wins over
  `--gpu-device` and is only `warn!`ed**, collapsing every CN onto one GPU — a cluster that still
  answers queries. `cluster8.sh` does not clear it. After launch,
  `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv` must list **distinct** GPUs.
- **`grep -c true` does not count alive nodes.** `SHOW COMPUTE NODES` also emits
  `SystemDecommissioned`, `ClusterDecommissioned` and `HasStoragePath`, so a *booting* node
  matches. `Alive` is column 9: `awk -F'\t' '$9=="true"' | wc -l`.
- **Never run two engines at once.** Sirius CNs and stock StarRocks BEs share port 9030 and the
  host CPUs.
- **`cn-env.sh` unsets `CXXFLAGS`/`CFLAGS`/`CPATH`.** Source it in a subshell if you need them.
- **Disk fills under `.cn*/telemetry/`.** The CN writes one ndjson dir per bring-up *and per query*
  and never GCs. `benchmarks/nixl-nvlink/clean-telemetry.sh` clears them.

---

## 11. Verifying the CUDA 13 pin

Three independent checks — run all three, since each covers a different artifact:

```bash
# 1. Engine pixi env resolved the cuda13 platform
cd /home/ubuntu/sirius && pixi info | grep -E 'Resolved platform|__cuda'
#    expect: __cuda=13.0 ... Resolved platform: linux-64 (cuda=13, ...)
.pixi/envs/default/bin/nvcc --version | tail -2      # expect release 13.x
ls .pixi/envs/default/lib/libcudart.so.13            # must exist

# 2. UCX configured against the CUDA 13 tree
grep -E '^CUDA_CPPFLAGS' $TOOLS_DIR/ucx-1.21.0/config.log
#    expect: CUDA_CPPFLAGS='-I/usr/local/cuda-13.0/include'

# 3. nixl's UCX plugin loads the driver
ldd $TOOLS_DIR/nvda_nixl/lib/*-linux-gnu/plugins/libplugin_UCX.so | grep -i cuda
```

The engine env resolving `cuda12` instead is the failure to watch for: the root `pixi.toml`
declares **both** `linux-64-cuda12` and `linux-64-cuda13` platforms, and pixi picks by the
driver's `__cuda` virtual package. An r570 driver silently selects the CUDA 12 environment.

---

## 12. Verified reference run

Recorded on the box this document was written against, so you can distinguish a *new* failure
from a known-good deviation.

**Box**

| | |
|---|---|
| OS / CPU | Ubuntu 24.04.4, Intel Xeon Platinum 8559C, 48 vCPU, 499 GB RAM, 1 NUMA node |
| GPU | 2× NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB each, cc 12.0 |
| Driver | 580.126.09 — r580, so **no** CUDA forward-compat shim needed |
| GPU link | `PIX` (PCIe switch), **not** NVLink; `topo -p2p r` = `OK`, so `cuda_ipc` works |
| CUDA trees | 12.6 / 12.8 / 12.9 / 13.0; `/usr/local/cuda` → **12.9**, so `CUDA_HOME=/usr/local/cuda-13.0` was set explicitly |
| Root | **none** — build deps came from conda (§3) |
| Disk | `/` 72 GB (21 GB free) → everything relocated to `/opt/dlami/nvme` (§2) |
| System gcc | 13.3.0 |
| JDK | `/usr/lib/jvm/java-21-amazon-corretto` (the FE *build* used the pixi `openjdk 17`) |

**Build times** — engine, UCX and FE run concurrently, so wall clock ≈ the slowest:

| Step | Time | Notes |
|---|---|---|
| submodules | ~2 min | |
| engine (`pixi run make`) | ~6 min | includes solving/downloading the pixi env; 1308 ninja targets |
| UCX | ~4 min | concurrent with the engine |
| FE (Maven) | **252 s** | concurrent; far from the "multi-hour" older notes claim |
| nixl | ~2 min | after UCX |
| CN (`cn-build`) | ~40 s | warm engine tree |
| **total wall clock** | **~15 min** | on 48 cores |

**Disk consumed**: 12.3 GB total — pixi cache 6.7 GB, engine `build/` 2.4 GB, `~/.m2` 959 MB,
`$TOOLS_DIR` 882 MB, starrocks pixi env 785 MB, cargo 290 MB, root pixi env 248 MB.

**Toolchain resolved**: engine env `__cuda=13.0`, `nvcc 13.2.78`, `libcudart.so.13.2.75`,
`libcudf` 26.06; UCX `CUDA_CPPFLAGS='-I/usr/local/cuda-13.0/include'`.

**Link check**: `readelf -d sirius-starrocks-cn` → `libnixl.so`, `libnixl_build.so`,
`sirius.duckdb_extension`.

**Smoke test results** — 1 CN, TPC-H SF100 `lineitem/part.0.parquet` (2.86 GB, 100M rows):

| Query | Sirius (GPU) | DuckDB (CPU) | |
|---|---|---|---|
| Q6-shape `sum(l_extendedprice*l_discount)` | `1028501805.6483` | `1028501805.6483` | exact match, 0.74 s |
| `GROUP BY l_returnflag` | A/24674179/629237003, N/50635887/1291118412, R/24678771/629496751 | identical | exact match |

Log evidence: `Sirius engine context created`, and 3× `relayed native batches across a fragment
boundary` — fragment output crossed as native GPU batch handles, not via Arrow or temp parquet.

---

## 13. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No SOURCES given to target: sirius_extension` | root submodules not initialized — `git submodule update --init --recursive` at the repo root |
| CN build: missing `transmit_packed` / `PTransmitPackedParams` | submodule reverted to stock; `pixi run apply-starrocks-patches` (or just re-run `cn-build`) |
| `Cannot solve ... No candidates were found for hwloc` | the conda package is `libhwloc`, not `hwloc` |
| nixl `meson setup`: `Dependency "pybind11" not found` | add `pybind11` + `python` to the conda tool env (§3); nixl's python bindings are unconditional |
| nixl-sys: `bits/timesize.h: No such file or directory` | conda `CXXFLAGS` leaked in — you did not source `cn-env.sh`, or you set `RUSTFLAGS` and busted cargo's fingerprint cache |
| Link: 39 `GLIBC_PRIVATE` undefined refs | conda `ld` was used; `cn-env.sh` prepends `/usr/bin` — on aarch64 add the `ld` shim (§7) |
| Link: undefined `cuLaunchKernel` / `nvmlDeviceGetMemoryAffinity` | driver libdir missing from `LIBRARY_PATH` (link-time, not `LD_LIBRARY_PATH`) |
| Every CN dies `cudaErrorInsufficientDriver` | CUDA-13 engine on an r570 driver — put `/usr/local/cuda/compat` first on `LD_LIBRARY_PATH` (§1) |
| CN dies `libstdc++.so.6: GLIBCXX_3.4.31 not found` | repo pixi env `lib/` missing from `LD_LIBRARY_PATH`; only bites outside `pixi run` |
| Agent creation fails at CN startup | libnixl not found or the stub was linked — check `TOOLS_DIR`, confirm `NIXL_PLUGIN_DIR` really holds `libplugin_UCX.so`, rebuild with `NIXL_NO_STUBS_FALLBACK=1` |
| nixl registration fails, `NIXL_ERR_BACKEND` | `UCX_TLS` missing `cuda_copy` |
| Works but ~200× slow | `UCX_TLS` missing `cuda_ipc`, or the arena is not `cudaMalloc`-backed |
| Oracle `duckdb` dies with `cudaErrorMemoryAllocation` | you used `build/release/duckdb`, which auto-loads Sirius and fights the running CN for the GPU — use a pip `duckdb` |
| `no packaged FE at .../output/fe/bin/start_fe.sh` | `pixi run -e fe fe-build`, or copy `starrocks/output/fe` from a box that has it (a stock FE of the same version also works) |
| `SHOW COMPUTE NODES` lists dead nodes | FE metadata persists — `ALTER SYSTEM DROP COMPUTE NODE "host:port"` |
| CN exits immediately, no log | port collision; check `ss -ltnp` against the plan (CN base 9100, stride 10; FE uses 8030/9010/9020/9030) |
| `rdma_create_event_channel failed` in the log | benign — no InfiniBand on the box |
| Submodule permanently `-dirty` | the applied proto patch; by design (§4) |
| `pkill` killed your shell | use the bracket pattern `pkill -f '[s]irius-starrocks-cn'` |
