# Study 1 — Sirius GPU Scale-Out · TPC-H SF1000 · pinned tables

**Internal:** `scale-out` · **Chart:** `Sirius GPU Scale-Out · TPC-H SF1000`
**Subtitle:** `2 → 4 → 8 × GB200 · pinned GPU-tier (compressed) · gcn-18 + gcn-09`

Self-contained runbook for a **new session**. Do not start from the A100 SF500 study, from
`notes/2026-08-09-gb200-sf100/`, or from `run-abc.sh`. Those launchers and boxes are wrong here.

Date of this plan: 2026-08-28. Authoring host: `presto-gb200-gcn-18`.

---

## 0. Question

Does Sirius get faster as GPU count doubles at SF1000 when **I/O is held constant** by
`pin_table` (GPU tier, Simpatico-compressed lineitem + orders)?

Y is the **warm median of iterations 1–3** after a per-query pin. Iteration 0 is unpinned
(page-cache lukewarm) and is **not** the scale-out number. Suite total is the sum of 22 warm
medians; NA if any query is not pass on every timed (pinned) run.

Pinning is legitimate here because every arm is Sirius vs Sirius. Do not mix these numbers with
cudf-polars or with the unpinned 4-GPU/8-GPU canvas.

---

## 1. Fleet (do not guess)

| | gcn-18 (this box) | gcn-09 |
|---|---|---|
| Hostname | `presto-gb200-gcn-18` | `presto-gb200-gcn-09` |
| Control LAN | **10.87.140.53** | **10.87.140.44** |
| GPUs | 4× GB200, ~184 GiB usable HBM each | same |
| Cores | 144 (2 sockets × 72) | same |
| Role | **FE + CNs**. All harness commands run here. | **CNs only** (8-GPU arm) |
| SSH name from 18 | (local) | `presto-gb200-gcn-09` (`REMOTE_HOST`) |

MNNVL ClusterUUID **must match** on both hosts (`nvidia-smi -q | grep -A3 Fabric`). Different
UUID = no cross-host `cuda_ipc`. Do not put `rc_mlx5` in `UCX_TLS` (`nvidia_peermem` is not
loaded). Nightly CI owns the GPUs **~02:00–03:50 UTC**.

Repo (NFS home, **same inode on both hosts**):

```
/home/prestouser/aocsa/sirius
branch: feat/pin-table-cn
HEAD at plan time: f05fd97b
```

Dataset (GPFS, both hosts): `/scratch/sirius/datasets/tpch_sf1000` (264.5 GiB, 60 lineitem files).
Do **not** use `/opt/sirius-ci/datasets` or `/raid/...` (18-only).

Env in every shell:

```bash
source /scratch/prestouser/aocsa/env.sh
# TOOLS_DIR=/home/prestouser/aocsa/tools
# PIXI_CACHE_DIR=/scratch/prestouser/aocsa/pixi-cache
# JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
cd /home/prestouser/aocsa/sirius
```

---

## 2. What is already built (gcn-18, 2026-08-28 ~21:17 UTC)

| Artifact | Path | Notes |
|---|---|---|
| libsirius | `build/release/extension/sirius/sirius.duckdb_extension` (`build` → `/scratch/prestouser/aocsa/build`) | rebuilt this afternoon |
| FE | `experimental/starrocks/starrocks/output/fe/` | **must** contain `files_query_whole_file_ranges` (patch `experimental/starrocks/patches/files-query-whole-file-ranges.patch`). Pins cannot serve byte-range `FILES()` splits. |
| CN | `experimental/starrocks/target/release/sirius-starrocks-cn` | aarch64 shims; do **not** use bare `pixi run cn-build` |
| Compression plans | `src/compression/simpatico_codegen/plans/tpch_sf1000/` | lineitem + orders active |

Confirm the FE config exists before any pinned arm:

```bash
python3 - <<'PY'
import zipfile, glob
jars = glob.glob("experimental/starrocks/starrocks/output/fe/lib/fe-core-*.jar")
assert jars, "no packaged FE"
with zipfile.ZipFile(jars[0]) as z:
    data = z.read("com/starrocks/common/Config.class")
print("files_query_whole_file_ranges" in data.decode("latin1"))
PY
# expect True
```

Never `git add` the StarRocks submodule. It is dirty by design (proto + FE config patches).

---

## 3. Node 9: you do **not** rebuild Sirius / FE / CN

`$HOME` is NFS (`master:/home`). `/scratch` is GPFS. The repo, the CN binary, libsirius, UCX,
nixl, and pixi prefixes are **the same files** on 09 and 18. A second `pixi run make` on 09
does not give you a second engine; it races the 18 build and can replace inodes under a live CN.

**Do this on 09 instead** (shell on 09, or `ssh presto-gb200-gcn-09`):

1. **Kill stale CNs** mapping the pre-rebuild `.so`. ninja writes a new inode; a CN started
   before today's rebuild keeps the old mapping forever.

   ```bash
   source /scratch/prestouser/aocsa/env.sh
   pgrep -af 'sirius-starrocks-cn|StarRocksFE'
   nvidia-smi --query-compute-apps=pid,process_name --format=csv
   cd /home/prestouser/aocsa/sirius/experimental/starrocks
   ./benchmarks/stop-cn-2host.sh
   nvidia-smi --query-compute-apps=pid --format=csv,noheader   # must be empty
   ```

2. **Same-inode check** (run on **both** hosts; the three numbers must match):

   ```bash
   stat -c '%i %n' /home/prestouser/aocsa/sirius/experimental/starrocks/target/release/sirius-starrocks-cn
   stat -c '%i %n' /scratch/prestouser/aocsa/build/release/extension/sirius/sirius.duckdb_extension
   readlink -f /home/prestouser/aocsa/sirius/build
   # expect /scratch/prestouser/aocsa/build
   ```

   If 09's `build` symlink still points at `/raid/...` (old gcn-18 convention), **fix the
   symlink**. That is the one case that looks like "09 needs a rebuild" and is actually a
   path bug.

3. **Runtime libs on 09** (no compile). The CN `NEEDED` list includes `libnixl.so`,
   `libnixl_build.so`, `sirius.duckdb_extension`. The nvidia-ml **shim** is an NFS symlink to a
   **node-local** path:

   ```bash
   ls /home/prestouser/aocsa/tools/toolchain-shims/libnvidia-ml.so
   ls /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1
   ls /home/prestouser/aocsa/tools/nvda_nixl/lib/aarch64-linux-gnu/plugins/libplugin_UCX.so
   source /scratch/prestouser/aocsa/env.sh
   source /home/prestouser/aocsa/sirius/experimental/starrocks/scripts/cn-env.sh
   ldd /home/prestouser/aocsa/sirius/experimental/starrocks/target/release/sirius-starrocks-cn \
     | grep -Ei 'nixl|sirius|nvidia-ml|not found'
   ```

   `not found` → install/fix the missing **local** `.so` (driver / CUDA 13), do not rebuild
   libsirius. Recreate the shim if 09's `libnvidia-ml.so.1` path differs:

   ```bash
   ln -sfn /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1 \
     /home/prestouser/aocsa/tools/toolchain-shims/libnvidia-ml.so
   ```

4. **Fabric + identity**

   ```bash
   nvidia-smi -q | grep -A3 Fabric          # State: Completed, UUID == 18
   python3 -c "import socket; print(socket.gethostbyname(socket.gethostname()))"
   # expect 10.87.140.44
   ```

5. **Do not** on 09:
   - `pixi run make` / `fe-build` / `cargo build -p sirius-starrocks-cn`
   - start an FE (`launch.sh` without `--no-fe`)
   - `git submodule update` (resets the FE/proto patches on the **shared** tree)
   - write TPC-H or FE meta under `$HOME`

6. **Rebuild on 09 only if** `readlink -f $REPO/build` is a different directory than 18, or
   `ldd` cannot resolve `sirius.duckdb_extension` after the symlink is fixed. Then follow
   `bench/gb200-4gpu/BUILD-AND-SMOKE.md` on 09 with `BIG=/scratch/prestouser/aocsa` and the
   aarch64 shims **inside** `pixi run bash -c`. Still do **not** `fe-build` on 09; the FE
   runs only on 18.

From 18, passwordless SSH must already work (`ssh -o BatchMode=yes presto-gb200-gcn-09 true`).
`relaunch.sh` SSHs from 18; Cursor agents often cannot SSH, a gcn-18 shell can.

---

## 4. Harness and launchers

Python harness (repo root, after `source env.sh`):

`test/tpch_performance/starrocks_performance_test.py`

| Arm | How the harness brings the cluster up |
|---|---|
| 2-GPU, 4-GPU | `benchmarks/pinned/gen-config.sh` + `benchmarks/pinned/up.sh` (1 host). YAML is required for pin-compression keys. |
| 8-GPU | `gen-config.sh` + `configs/gb200-8gpu/relaunch.sh` → `cn-2host.sh`. Must run on **gcn-18**. |

`--gpus` is **CNs per host**. `--hosts` is the FE first.

Do **not** use `cluster4-numa.sh`, `benchmarks/cluster8.sh`, `run-abc.sh`, or
`benchmarks/tpch/bench.sh` for this study. Those either omit `--sirius-config` (no pin
compression) or skip `files_query_whole_file_ranges`.

**FE SET after every start** (harness does this):

```
SET GLOBAL query_timeout = 1800;
SET GLOBAL enable_pipeline_engine = true;
SET GLOBAL pipeline_dop = 18;
ADMIN SET FRONTEND CONFIG ("files_query_whole_file_ranges" = "true");
```

**Pin protocol:** per query, iter 0 unpinned, then `ADMIN EXECUTE` pin on every alive CN
(parallel), iters 1–3, then unpin. `ADMIN EXECUTE` has a hard **600 s** FE ceiling; do **not**
retry a pin after a timeout (watch `/tmp/cn-*.log` / up.sh CN logs for `pin_table finished`).
`Unable to validate object` is retried; a 600 s timeout is not.

`q11` fraction is `0.0001/SF` = `0.000000100000`. The harness substitutes this from
`--scale-factor 1000`.

### 2-GPU GPU pairing (do this before the 2-GPU arm)

`up.sh` today binds `--gpu-device $i` (GPU0+GPU1, both socket 0). The fair 2-GPU arm on this
box is **GPU0 + GPU2** (one CN per socket, NV18 is identical for every pair). Patch `up.sh`
to honor `CN_GPU` (3 lines). If you skip this, say so on the chart; you measured the
host-NUMA-confined variant, not scale-out.

```bash
# experimental/starrocks/benchmarks/pinned/up.sh  — inside the CN launch loop, replace
#   --gpu-device "$i"
# with:
#   --gpu-device "${GPUS[$i]}"
# and before the loop:
#   read -r -a GPUS <<< "${CN_GPU:-$(seq -s ' ' 0 $((NUM_CNS - 1)))}"
```

2-GPU command then exports `CN_GPU="0 2"`. Harness `_launch_1host` already forwards the
environment into `up.sh`; add `env["CN_GPU"] = os.environ.get("CN_GPU", "")` in
`starrocks_performance_test.py` `Cluster._env` **or** export it in the same shell as `pixi run`
(child inherits it). If `_env` overwrites via `os.environ.copy()`, an exported `CN_GPU` already
survives. Prefer:

```bash
export CN_GPU="0 2"
```

before the 2-GPU `pixi run`.

### 8-GPU FE conf (once, on 18, before the first 8-GPU launch)

Packaged `starrocks/output/fe/conf/fe.conf` is single-host (`priority_networks` unset, meta on
NFS `output/fe/meta`). Two-host needs both fixed. Exact steps:
`experimental/starrocks/benchmarks/2NODE-REPLICATE.md` §3.

```bash
FE_CONF=/home/prestouser/aocsa/sirius/experimental/starrocks/starrocks/output/fe/conf/fe.conf
grep -q 'priority_networks = 10.87.140.32/27' "$FE_CONF" || \
  printf '\npriority_networks = 10.87.140.32/27\n' >> "$FE_CONF"
META=/scratch/prestouser/aocsa/fe/meta
mkdir -p "$META"
grep -q '^meta_dir' "$FE_CONF" && sed -i "s|^meta_dir.*|meta_dir = $META|" "$FE_CONF" \
  || printf '\nmeta_dir = %s\n' "$META" >> "$FE_CONF"
```

`relaunch.sh` already wipes `$META` on each 8-GPU start.

---

## 5. Per-arm configuration

Occupancy rule (GB200, usable 184.00 GiB):

```
GPU_MEM + STAGING + 0.76 GiB  ≤  ~184 GiB     (leave ≥12 GiB headroom)
```

The staging arena is a bare `cudaMalloc` **outside** the RMM pool. Fewer CNs need a **larger**
arena (per-node exchange ~`D/N`). Fitted A100 SF500 thumb: `staging(N) ≈ 96 GiB / N` at SF500;
at SF1000 the 4-CN/8-CN measured split is **32 GiB**, not 24. Do not copy 16 GiB from the SF500
pin README.

Shared across all arms:

| Knob | Value |
|---|---|
| Scale | 1000 |
| Data | `/scratch/sirius/datasets/tpch_sf1000` |
| `--pin` | `gpu` |
| `--pin-after-iteration` | `1` |
| `--pin-compression` | on |
| `--iterations` | `4` |
| `--mode` | `grouped` |
| `--pipeline-dop` | **18** (always `SET GLOBAL`; never 36) |
| `--query-timeout` | `1800` |
| Scan | uring (`SIRIUS_CN_USE_SIRIUS_DATASOURCE=true` or unset) |
| Plans | default `plans/tpch_sf1000` |
| `ADMIN EXECUTE` pin timeout | 600 s, no retry |

### Arm table

| | 2-GPU | 4-GPU | 8-GPU |
|---|---|---|---|
| Hosts | 18 only | 18 only | 18 + 09 |
| `--gpus` / `--hosts` | `2` / `127.0.0.1` | `4` / `127.0.0.1` | `4` / `10.87.140.53,10.87.140.44` |
| GPUs | **0 and 2** | 0–3 | 0–3 on each host |
| `NUM_CNS` total | 2 | 4 | 8 |
| `GPU_MEM` | **120GiB** | **128GiB** | **128GiB** |
| `STAGING` | **48GiB** | **32GiB** | **32GiB** |
| `HOST_MEM` | **200GiB** | **112GiB** | **112GiB** |
| Occupancy | 168.8 / 184 (92%) | 160.8 / 184 (87%) | 160.8 / 184 (87%) per GPU |
| Host commit / box | 2×200 = 400 GiB | 4×112 = 448 GiB | 4×112 = 448 GiB **per box** |
| Arena | same-host `cuda_ipc` (default) | same-host `cuda_ipc` | **`SIRIUS_EXCHANGE_STAGING_ARENA=fabric`** |
| `UCX_TLS` | `cuda_copy,cuda_ipc,tcp,self` | same | same. **No `rc_mlx5`.** |
| CPU / membind | `up.sh` today: none | none | `cn-2host.sh`: disjoint 36-core, membind 0/0/1/1 |
| Watchdog / RPC | 300 / 300 | 300 / 300 | 300 / 300 (`sf1000/env.sh`) |
| Expected CNs alive | 2 | 4 | **8** |

Why 2-GPU is 120/48 not 128/32: at N=2 each CN holds ~2× the 4-CN shuffle. Today's 4-GPU pin
sweep at 128/32 still lost **q8, q9, q21** to arena fragmentation (32 GiB total, hundreds of
leases, largest free block < request). Doubling staging is the first 2-GPU lever. GPU_MEM
drops 8 GiB so occupancy stays under 92%. Compressed pin of a query's file-subset at 2 CN is
larger per GPU than at 4; if pin OOMs, see fallbacks below. Do **not** raise 2-GPU `HOST_MEM`
above 200 with GPU0+GPU1 (both socket 0); GPU0+GPU2 is one CN per socket and 200 is fine.

Why 4-GPU stays 128/32/112: that is the unpinned 4-GPU GPFS arm
(`abc-sf1000-gpfs-20260828T013342Z`) and today's pinned rerun. Changing staging here mixes
"pin vs unpin" with "different split".

Why 8-GPU stays 128/32/112 + dop=18 + uring + fabric: that is the **retuned** unpinned 8-GPU
arm (`sf1000-8gpu-2host-20260828T035635Z`) that closed 22/22. 16 GiB staging died at q05 on
8 CN. dop=36 + kvikio was 1.21× **slower** than 4-GPU; do not revive it.

### Fallbacks (one change at a time)

| Symptom | Arm | Change |
|---|---|---|
| `exchange staging arena exhausted` | 2-GPU | `STAGING=56GiB` `GPU_MEM=112GiB` (occupancy 168.8) |
| pin OOM / `bad_alloc` during pin | 2-GPU | `GPU_MEM=128GiB` `STAGING=32GiB` first; if still OOM, `--pin host` for **that arm only** and label the chart |
| q8/q9/q21 refuse at 4-GPU 32 GiB | 4-GPU | **leave them empty** for the scale-out curve. Optional extra arm `STAGING=40GiB` `GPU_MEM=120GiB` is a sensitivity run, not the study point |
| 8-GPU CNs = 4 (09 never joined) | 8-GPU | logs `/tmp/gb200-8gpu-launch-09.log`, fabric UUID, `priority_networks`, SSH |
| 8-GPU CN fail at start `ibv_reg_mr` | 8-GPU | `rc_mlx5` leaked into `UCX_TLS`. Force `UCX_TLS=cuda_copy,cuda_ipc,tcp,self` |
| `files_query_whole_file_ranges` missing | any pin | FE jar is unpatched; re-apply patch + `pixi run -e fe fe-build` **on 18 only** |
| q22 `Unable to validate object` after a refuse restart | any | harness race; rerun that query on a clean cluster |

---

## 6. Commands (copy as-is)

Preflight on 18 (and 09 before the 8-GPU arm):

```bash
source /scratch/prestouser/aocsa/env.sh
pgrep -af 'sirius-starrocks-cn|StarRocksFE'
nvidia-smi --query-compute-apps=pid --format=csv,noheader   # empty
```

### 6.1 2-GPU (gcn-18)

```bash
source /scratch/prestouser/aocsa/env.sh
cd /home/prestouser/aocsa/sirius
export CN_GPU="0 2"
pixi run python3 test/tpch_performance/starrocks_performance_test.py \
  --input /scratch/sirius/datasets/tpch_sf1000 \
  --scale-factor 1000 \
  --mode grouped --iterations 4 \
  --pin gpu --pin-after-iteration 1 --pin-compression \
  --gpus 2 --hosts 127.0.0.1 \
  --gpu-mem 120GiB --staging 48GiB --host-mem 200GiB \
  --pipeline-dop 18 \
  --query-timeout 1800 \
  --name tpch_sf1000_2gpu_pinned
```

### 6.2 4-GPU (gcn-18)

Already collected 2026-08-28T21:49Z (reuse unless you changed the engine):

`test/tpch_performance/output/tpch_20260828_214959_tpch_sf1000_4gpu_pinned/`

18/22 timed. **q8, q9, q21** arena-exhausted at 32 GiB (iter 0, so pinning never ran). **q22**
died on `Unable to validate object` immediately after the q21 restart; rerun q22 only if you
need it:

```bash
source /scratch/prestouser/aocsa/env.sh
cd /home/prestouser/aocsa/sirius
pixi run python3 test/tpch_performance/starrocks_performance_test.py \
  --input /scratch/sirius/datasets/tpch_sf1000 \
  --scale-factor 1000 \
  --mode grouped --iterations 4 \
  --pin gpu --pin-after-iteration 1 --pin-compression \
  --gpus 4 --hosts 127.0.0.1 \
  --gpu-mem 128GiB --staging 32GiB --host-mem 112GiB \
  --pipeline-dop 18 \
  --query-timeout 1800 \
  --queries 22 \
  --name tpch_sf1000_4gpu_pinned_q22
```

Full 4-GPU rerun (only if binaries moved):

```bash
# same as above without --queries 22, --name tpch_sf1000_4gpu_pinned
```

### 6.3 8-GPU (must be on gcn-18)

```bash
source /scratch/prestouser/aocsa/env.sh
cd /home/prestouser/aocsa/sirius
# FE conf already patched (§4). 09: stale CNs dead, inode check done (§3).
pixi run python3 test/tpch_performance/starrocks_performance_test.py \
  --input /scratch/sirius/datasets/tpch_sf1000 \
  --scale-factor 1000 \
  --mode grouped --iterations 4 \
  --pin gpu --pin-after-iteration 1 --pin-compression \
  --gpus 4 --hosts 10.87.140.53,10.87.140.44 \
  --gpu-mem 128GiB --staging 32GiB --host-mem 112GiB \
  --pipeline-dop 18 \
  --query-timeout 1800 \
  --name tpch_sf1000_8gpu_pinned
```

`relaunch.sh` sources `bench/gb200-8gpu/sf1000/env.sh` (fabric + uring + dop 18). Harness
`GPU_MEM`/`STAGING`/`HOST_MEM` override the file. Confirm 8 `Alive=true` before the first
query. 09 log: `/tmp/gb200-8gpu-launch-09.log`. 18 log: `/tmp/gb200-8gpu-launch-18.log`.

Teardown if the harness dies before `finally`:

```bash
# [18]
cd /home/prestouser/aocsa/sirius/experimental/starrocks
./benchmarks/stop-cn-2host.sh
ssh -o BatchMode=yes presto-gb200-gcn-09 \
  'cd /home/prestouser/aocsa/sirius/experimental/starrocks && ./benchmarks/stop-cn-2host.sh'
```

1-host teardown is `pkill -f '[s]irius-starrocks-cn'; pkill -f '[S]tarRocksFE'` (harness
already does this).

---

## 7. How to score a run

Per query, from `csv/runtimes.csv`:

- **unpinned lukewarm** = iteration 0
- **pinned warm median** = median of iterations 1, 2, 3 (need all three; else NA)

Compare:

1. **Scale-out (this study):** 2-GPU vs 4-GPU vs 8-GPU pinned warm medians.
2. **Pin vs unpin at 4-GPU (optional overlay):** today's pinned 4-GPU vs the unpinned 4-GPU
   GPFS warm medians below (ms). Same files, same 128/32/112, dop 18, uring. Launcher
   differs (`cluster4-numa.sh` vs `up.sh`); say that if you publish.

Unpinned 4-GPU warm median (ms), `abc-sf1000-gpfs-20260828T013342Z`:

```
q01 7271  q02 1098  q03 1843  q04  982  q05 2366  q06 1222  q07 2007
q08 refused  q09 refused  q10 3050  q11  989  q12 1663  q13 1138
q14 1300  q15 2170  q16  627  q17 4373  q18 3061  q19 1422  q20 1988
q21 refused  q22  806
```

Unpinned 8-GPU retune warm median (ms), `sf1000-8gpu-2host-20260828T035635Z` (22/22):

```
q01 4185  q02 1245  q03 1347  q04  996  q05 2145  q06  958  q07 1642
q08 2376  q09 2813  q10 1455  q11  955  q12  854  q13  699  q14  777
q15 1328  q16  421  q17 2692  q18 1819  q19  932  q20 1301  q21 3070
q22  505
```

Today's **pinned** 4-GPU hot times (s, iters 1–3; convert ×1000 for the chart). Median:

| q | iter1 | iter2 | iter3 | hot median s | unpinned 4-GPU s | pin / unpin |
|---|---|---|---|---|---|---|
| 1 | 5.853 | 5.759 | 5.748 | 5.759 | 7.271 | 0.79× |
| 2 | 1.247 | 1.223 | 1.273 | 1.247 | 1.098 | 1.14× |
| 3 | 1.100 | 1.030 | 1.034 | 1.034 | 1.843 | 0.56× |
| 4 | 0.732 | 0.691 | 0.764 | 0.732 | 0.982 | 0.75× |
| 5 | 1.851 | 1.775 | 1.755 | 1.775 | 2.366 | 0.75× |
| 6 | 0.472 | 0.413 | 0.411 | 0.413 | 1.222 | 0.34× |
| 7 | 1.347 | 1.281 | 1.286 | 1.286 | 2.007 | 0.64× |
| 8 | — | — | — | NA | refused | |
| 9 | — | — | — | NA | refused | |
| 10 | 1.760 | 1.621 | 1.585 | 1.621 | 3.050 | 0.53× |
| 11 | 1.121 | 1.034 | 0.979 | 1.034 | 0.989 | 1.05× |
| 12 | 1.112 | 1.001 | 1.012 | 1.012 | 1.663 | 0.61× |
| 13 | 0.850 | 0.858 | 0.815 | 0.850 | 1.138 | 0.75× |
| 14 | 0.613 | 0.558 | 0.559 | 0.559 | 1.300 | 0.43× |
| 15 | 0.657 | 0.593 | 0.602 | 0.602 | 2.170 | 0.28× |
| 16 | 0.745 | 0.735 | 0.717 | 0.735 | 0.627 | 1.17× |
| 17 | 3.956 | 3.895 | 3.833 | 3.895 | 4.373 | 0.89× |
| 18 | 2.968 | 2.894 | 2.642 | 2.894 | 3.061 | 0.95× |
| 19 | 0.929 | 0.876 | 0.881 | 0.881 | 1.422 | 0.62× |
| 20 | 1.172 | 1.130 | 1.101 | 1.130 | 1.988 | 0.57× |
| 21 | — | — | — | NA | refused | |
| 22 | — | — | — | NA | 0.806 | restart race |

q2 and q16 slower pinned than unpinned 4-GPU is real in this sample (small queries, pin
overhead / NUMA-less `up.sh`). Do not drop them.

Confirm pins actually served: CN log `serves operator ... as a file subset` / `using
cached_split_provider`. Fallback line is `not all the columns are pinned`.

---

## 8. Sequence for the new session

1. Preflight 18 idle. Confirm FE jar has `files_query_whole_file_ranges`.
2. **2-GPU** first (shortest cluster, hardest arena). Patch `up.sh` `CN_GPU` if not done.
3. **4-GPU:** reuse `tpch_20260828_214959_tpch_sf1000_4gpu_pinned` + optional q22 rerun.
4. **Node 9 checks** (§3). Patch `fe.conf` (§4).
5. **8-GPU** from gcn-18. Stop 09 leftover CNs first.
6. Chart: X = {2,4,8}, Y = pinned warm-median suite time (log), per-query speedup panel.
   Annotate q8/q9/q21 empty at 2/4 if they stay empty.

Do not start 8-GPU while 18 still has a 4-GPU `up.sh` cluster. Do not run during CI.

---

## 9. Companion docs (read if something breaks)

| Need | Doc |
|---|---|
| Two-host bring-up, FE meta, fabric | `experimental/starrocks/benchmarks/2NODE-REPLICATE.md` |
| 8-GPU SF1000 unpinned knobs | `bench/gb200-8gpu/sf1000/env.sh`, `bench/gb200-8gpu/SIRIUS-TUNING-RUNBOOK.md` |
| 4-GPU memory arithmetic | `experimental/starrocks/configs/gb200-4gpu/engine-a.env` |
| Pin path, FE patch, 600 s ceiling | `experimental/starrocks/benchmarks/pinned/README.md` |
| aarch64 build (only if 09 tree is actually separate) | `bench/gb200-4gpu/BUILD-AND-SMOKE.md` |
| Harness flags | `test/tpch_performance/CLAUDE.md` (StarRocks section) |

---

## 10. What not to do

- Do not copy 8-GPU `pipeline_dop=36` or kvikio (`SIRIUS_CN_USE_SIRIUS_DATASOURCE=false`).
- Do not copy SF500 pin README `GPU_MEM=110GiB STAGING=16GiB` to SF1000.
- Do not retry `ADMIN EXECUTE` after a 600 s timeout.
- Do not treat a cluster as valid after a refuse without a restart (harness restarts; a
  manual mysql session does not).
- Do not publish "22/22" if q8/q9/q21 are empty.
- Do not `numactl --interleave=all` or `--membind` an HBM node (2, 10, 18, 26).
- Do not rebuild on 09 "to be safe" while 18 has CNs running against the same NFS binary.
