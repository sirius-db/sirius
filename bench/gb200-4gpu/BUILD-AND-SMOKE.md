# GB200 aarch64: build the CN stack and smoke-test it

Verified 2026-08-27 on `presto-gb200-gcn-18` (4× GB200, driver 580.105.08, CUDA 13.0,
aarch64 Neoverse-V2, 144 cores). Same hardware class as [`HARDWARE.md`](HARDWARE.md).

The generic procedure is [`../rtxpro6000-2gpu/BUILD-SIRIUS-STARROCKS.md`](../rtxpro6000-2gpu/BUILD-SIRIUS-STARROCKS.md).
This note is the **aarch64 + NFS-home** delta: where artifacts landed, which commands actually
ran, and the smoke numbers.

Do **not** `git add` the StarRocks submodule after the proto patch. It is dirty by design.

---

## 0. Paths to set on the new box

`$HOME` is NFS. Compiles and pixi caches go on `/scratch`. `/raid` is local NVMe RAID0 but
root-owned on this box (`presto-gb200-gcn-09`), so it is not writable. On gcn-18, `$BIG` was
`/raid/$USER/sirius-build`.

```bash
export USER=prestouser                          # change
export REPO=/home/$USER/aocsa/sirius            # clone location
export BIG=/scratch/$USER/aocsa                 # /raid is not writable here
export DATASETS=/scratch/sirius/datasets        # TPC-H parquet (all scales, now and later)
export TOOLS_DIR=/home/$USER/aocsa/tools        # default of cn-env.sh: sibling named tools/
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
export CUDA_HOME=/usr/local/cuda-13.0           # on gcn-18, /usr/local/cuda already → 13.0
export PATH=$HOME/.pixi/bin:$PATH
export PIXI_CACHE_DIR=$BIG/pixi-cache
export CARGO_HOME=$BIG/cargo
export PATH=$CARGO_HOME/bin:$PATH
```

`cn-env.sh` derives `TOOLS_DIR` as `<parent-of-repo>/tools`. If you clone to a different parent,
export `TOOLS_DIR` yourself.

TPC-H parquet lives under `$DATASETS` as `tpch_sf<N>/`. Do not copy datasets onto the box.
Smoke uses SF100 `lineitem` **part.0 only** (~100M rows, not the full SF100 set):

```text
/scratch/sirius/datasets/tpch_sf100/lineitem/part.0.parquet
```

---

## 1. What you are building, and where it ends up

| # | Artifact | Command | Output on this box |
|---|---|---|---|
| 1 | UCX | autotools, CUDA 13 | `$TOOLS_DIR/ucx-install` |
| 2 | nixl (UCX plugin only) | meson after (1) | `$TOOLS_DIR/nvda_nixl` |
| 3 | libsirius | `pixi run make` at `$REPO` | `$BIG/build/release/extension/sirius/sirius.duckdb_extension` (`$REPO/build` is a symlink) |
| 4 | StarRocks FE | `pixi run -e fe fe-build` | `$REPO/experimental/starrocks/starrocks/output/fe/` |
| 5 | CN | cargo + aarch64 shims (not plain `pixi run cn-build`) | `$REPO/experimental/starrocks/target/release/sirius-starrocks-cn` |

(1), (3), (4) can run concurrently. (2) follows (1). (5) is last.

On gcn-18, (1) and (2) were already present under `/home/prestouser/aocsa/tools`. A new box
must build them (runbook §5c and §6). Check:

```bash
ls $TOOLS_DIR/ucx-install/lib/ucx/libuct_cuda.so
ls $TOOLS_DIR/nvda_nixl/lib/aarch64-linux-gnu/plugins/libplugin_UCX.so
```

CN link check (must list all three):

```bash
readelf -d $REPO/experimental/starrocks/target/release/sirius-starrocks-cn | grep -Ei 'nixl|sirius'
# NEEDED: libnixl.so, libnixl_build.so, sirius.duckdb_extension
```

---

## 2. Relocate caches onto local disk (before first configure)

```bash
mkdir -p $BIG/{pixi-cache,root-pixi,sr-pixi,cargo,build}
ln -sfn $BIG/root-pixi $REPO/.pixi
ln -sfn $BIG/build     $REPO/build
ln -sfn $BIG/sr-pixi   $REPO/experimental/starrocks/.pixi
```

Write `$BIG/env.sh` with the exports from §0 and `source` it in every shell.

---

## 3. Submodules, patch, aarch64 shims

```bash
cd $REPO
git submodule update --init --recursive

cd $REPO/experimental/starrocks
pixi run --manifest-path pixi.toml -e fe apply-starrocks-patches
# expect: applied nixl-exchange-proto.patch   (or "already applied")

mkdir -p $TOOLS_DIR/toolchain-shims
ln -sf /usr/bin/g++ $TOOLS_DIR/toolchain-shims/g++
ln -sf /usr/bin/ld  $TOOLS_DIR/toolchain-shims/ld
ln -sf /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1 $TOOLS_DIR/toolchain-shims/libnvidia-ml.so
```

`pixi run cn-build` **cannot** carry those shims: pixi prepends conda `g++`/`ld`. PATH must be
set *inside* `pixi run bash -c`. An outer `PATH=$SHIMS:$PATH pixi run …` loses.

`~/.local/bin/uv` on this fleet is often an **x86-64** binary. Do not use it on aarch64 nixl
builds. Take meson/ninja/hwloc/pybind11 from a conda toolenv (runbook §3), not `apt-get` (no root).

---

## 4. Build

```bash
source $BIG/env.sh

# engine (long). Confirm CUDA 13: pixi info | grep -E 'Resolved platform|__cuda'
# expect __cuda=13.0 and Resolved platform linux-aarch64 (cuda=13, …)
cd $REPO && pixi run make

# FE (~6 min here). Maven can print BUILD SUCCESS and then pixi exits 1 on
# `pop_var_context`. That is a wrapper glitch. Check start_fe.sh exists:
cd $REPO/experimental/starrocks
pixi run -e fe fe-build
test -x starrocks/output/fe/bin/start_fe.sh

# CN. Do not use `pixi run cn-build` on aarch64.
cd $REPO/experimental/starrocks
SHIMS=$TOOLS_DIR/toolchain-shims
pixi run --manifest-path "$PWD/pixi.toml" bash -c "
  set -euo pipefail
  export PATH=$SHIMS:\$PATH
  export RUSTFLAGS=\"-C link-arg=-L$SHIMS -C link-arg=-lnvidia-ml\"
  source scripts/cn-env.sh
  cargo build --release -p sirius-starrocks-cn
"
```

---

## 5. Smoke test (1 CN, SF100 `lineitem/part.0`)

Do **not** `pixi run cluster`: it depends on `cn-build`, which would cargo-rebuild without shims.

CPU oracle must be a **plain** duckdb (pixi/pip). `$REPO/build/release/duckdb` auto-loads Sirius
and fights the CN for GPU memory.

```bash
source $BIG/env.sh
unset CUDA_VISIBLE_DEVICES
export SIRIUS_QUERY_WATCHDOG_SECS=60
cd $REPO/experimental/starrocks
PARQ="file://$DATASETS/tpch_sf100/lineitem/part.0.parquet"

# terminal 1 — BLOCKS. Its EXIT trap tears the cluster down.
pixi run --manifest-path "$PWD/pixi.toml" bash -lc '
set -euo pipefail
source scripts/cn-env.sh
cleanup() {
    status=$?
    trap - EXIT INT TERM
    kill "${fe_pid:-}" "${cn_pid:-}" 2>/dev/null || true
    wait "${fe_pid:-}" "${cn_pid:-}" 2>/dev/null || true
    exit "${status}"
}
trap cleanup EXIT INT TERM
starrocks/output/fe/bin/start_fe.sh --logconsole &
fe_pid="$!"
target/release/sirius-starrocks-cn &
cn_pid="$!"
wait -n "${fe_pid}" "${cn_pid}"
' 2>&1 | tee /tmp/cluster.log
```

Terminal 2: wait until column 9 of `SHOW COMPUTE NODES` is `true` (`Alive`), then run:

```sql
SELECT sum(l_extendedprice * l_discount) AS revenue
FROM FILES("path"="file:///scratch/sirius/datasets/tpch_sf100/lineitem/part.0.parquet","format"="parquet")
WHERE l_shipdate >= date '1997-01-01' AND l_shipdate < date '1998-01-01'
  AND l_discount BETWEEN 0.02 AND 0.04 AND l_quantity < 24;

SELECT l_returnflag, count(*) AS n, sum(l_quantity) AS qty
FROM FILES("path"="file:///scratch/sirius/datasets/tpch_sf100/lineitem/part.0.parquet","format"="parquet")
GROUP BY l_returnflag ORDER BY l_returnflag;
```

Expected (gcn-18, measured on the old `tpch_parquet_sf100_f64v1` part.0). Re-check against the
oracle on `$DATASETS/tpch_sf100` if the numbers differ.

| Query | Result |
|---|---|
| Q6-shape | `1028501805.6483` |
| GROUP BY | A / 24674179 / 629237003 ; N / 50635887 / 1291118412 ; R / 24678771 / 629496751 |

Also required:

```bash
grep 'Sirius engine context created' /tmp/cluster.log
grep 'relayed native batches across a fragment boundary' /tmp/cluster.log
# GROUP BY produces 3 of those lines
```

Teardown: kill the CN and FE **by PID** (`pgrep -a target/release/sirius-starrocks-cn` and
`pgrep -a com.starrocks.StarRocksFE`). Do not `pkill -f` a pattern that also appears in the
launcher script. Then:

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader   # must be empty
```

Check compute-apps, not `memory.used`.

Oracle (once):

```bash
mkdir -p $BIG/oracle && cd $BIG/oracle
pixi init . && pixi add python duckdb
pixi run python -c "import duckdb; print(duckdb.sql(\"\"\"
SELECT sum(l_extendedprice*l_discount) FROM read_parquet('/scratch/sirius/datasets/tpch_sf100/lineitem/part.0.parquet')
WHERE l_shipdate >= date '1997-01-01' AND l_shipdate < date '1998-01-01'
  AND l_discount BETWEEN 0.02 AND 0.04 AND l_quantity < 24
\"\"\").fetchall())"
```

---

## 6. Layout on gcn-09 (this session)

```text
/scratch/sirius/datasets/          ← $DATASETS, TPC-H parquet (tpch_sf1 … tpch_sf1000, …)
  tpch_sf100/lineitem/part.0.parquet

/scratch/prestouser/aocsa/
  env.sh
  build/          ← $REPO/build                    libsirius + duckdb
  root-pixi/      ← $REPO/.pixi
  sr-pixi/        ← $REPO/experimental/starrocks/.pixi
  pixi-cache/
  cargo/          ← CARGO_HOME
  oracle/         ← pixi duckdb 1.5.5

/home/prestouser/aocsa/sirius/experimental/starrocks/
  target/release/sirius-starrocks-cn
  starrocks/output/fe/

/home/prestouser/aocsa/tools/
  ucx-install/
  nvda_nixl/
  toolchain-shims/{g++, ld, libnvidia-ml.so}
```
