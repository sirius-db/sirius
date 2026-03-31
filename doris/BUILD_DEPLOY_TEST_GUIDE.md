# Sirius-Doris: Build, Deploy & Test Guide

## What is Sirius-Doris?

Sirius-Doris is a **GPU-accelerated query execution backend** for Apache Doris. It replaces the standard C++ Doris Backend (BE) with a Rust service that wraps a **C++ GPU engine** (a DuckDB extension using CUDA and RAPIDS/cuDF). The architecture is:

```
MySQL Client → Doris FE (standard Java, unchanged) → Sirius GPU BE (Rust + C++ GPU engine)
```

The FE handles SQL parsing, planning, and fragment distribution — exactly like stock Doris. The Sirius BE receives query fragments via gRPC, translates them to Substrait plans, and executes them on the GPU. Results are returned via Arrow Flight. From the FE's perspective, Sirius BE looks like a normal Doris BE.

---

## 1. Build Environment Requirements

### Hardware
- **NVIDIA GPU** with compute capability >= 7.5 (Turing or newer: T4, V100, A100, H100, B200, etc.)
- **CUDA driver** >= 13.0 installed on the host (for CUDA 13 env) or >= 12.0 (for CUDA 12 env)
- Minimum **8 GB GPU VRAM** recommended (default config uses 2 GB cache + 2 GB processing per BE)
- Minimum **32 GB system RAM** recommended for building

### Software
- **Linux x86_64** (tested on Ubuntu 22.04 and NixOS)
- **[Pixi](https://pixi.sh/)** package manager (handles all build dependencies — CUDA toolkit, Rust, Java, CMake, UCX, meson, etc.)
- **Git** (with submodules)

### Install Pixi (if not already installed)
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Clone and initialize
```bash
git clone --recurse-submodules --single-branch --branch doris https://github.com/mbrobbel/sirius.git sirius-doris
cd sirius-doris
```

If you already cloned without `--recurse-submodules`:
```bash
git submodule update --init --recursive
```

---

## 2. Build Steps

There are **four components** to build. Pixi manages all dependencies automatically — the first `pixi run` downloads and configures the environment (CUDA toolkit, Rust toolchain, JDK 17, Maven, CMake, UCX, meson, etc.).

### Step 1: Build the C++ GPU Engine (Sirius DuckDB extension)
```bash
pixi run make release
```
Produces: `build/release/extension/sirius/sirius.duckdb_extension`

### Step 2: Build the DuckDB Substrait extension
```bash
pixi run -e doris substrait-build
```
Produces: `substrait/build/release/extension/substrait/substrait.duckdb_extension`

Both extensions are **required** — the BE will refuse to start if either fails to load.

### Step 3: Build the Rust GPU Backend binary
```bash
pixi run -e doris doris-build
```
This automatically builds the [NIXL](https://github.com/ai-dynamo/nixl) C++ library from the `doris/thirdparty/nixl` submodule (for GPU-direct inter-BE transfers via UCX/RDMA), then builds the Rust workspace.

Produces: `doris/target/release/sirius-doris-be`

### Step 4: Build the Doris Frontend (Java)
```bash
pixi run -e doris-fe doris-fe-build
```
Produces: `doris/thirdparty/apache-doris/output/fe/`

### Build time expectations
| Component | First build | Rebuild |
|-----------|-------------|---------|
| C++ GPU engine | ~10-30 min | ~1-2 min (sccache) |
| Substrait extension | ~5-10 min | ~1 min |
| NIXL + Rust BE | ~10-15 min | ~30 sec |
| Doris FE | ~10-15 min | ~2-3 min |

---

## 3. Running the Cluster

Run each component in a separate terminal:

```bash
# Terminal 1: Start FE
pixi run -e doris-fe doris-fe-start    # builds if needed, then starts
# or: pixi run -e doris-fe doris-fe    # starts without rebuild

# Terminal 2: Start GPU BE 1
pixi run -e doris sirius-be

# Terminal 3 (optional): Start GPU BE 2
pixi run -e doris sirius-be-2
```

The `sirius-be-2` task automatically uses a separate home directory (`/tmp/sirius-be-2`) to avoid lock file conflicts with BE 1 (the sirius extension locks `~/.sirius/sirius.lock` to prevent concurrent access to the same config).

The BE auto-registers with the FE via the `--fe` flag. If registration fails (e.g., FE not ready yet), add it manually:
```bash
pixi run -e doris-fe doris-fe-add-sirius
```

### Verify the cluster
```bash
# The mysql client is available in the doris-fe pixi environment:
pixi run -e doris-fe -- mysql -h 127.0.0.1 -P 9030 -u root

# Inside mysql:
SHOW BACKENDS\G    -- Should show Alive: true
SELECT 1;          -- Basic connectivity test
```

### Expected BE startup log
A healthy BE startup shows:
```
Extension loaded: sirius
DuckDB engine initialized
GPU buffers initialized cache=2GB processing=2GB
CUDA context initialized for UCX GPU support
nixl agent initialized with UCX backend
nixl GPU-direct exchange enabled
starting PBackendService multi-protocol server (gRPC + bRPC)
received heartbeat from FE
```

If you see `FATAL: engine init failed`, the substrait or sirius extension could not be loaded. Verify Steps 1-2 completed successfully.

### Clean restart
To reset FE state (metadata) for a completely fresh start:
```bash
pixi run -e doris-fe doris-fe-clean
```

---

## 4. CPU vs GPU Mode Comparison

You do **NOT** need separate clusters. The Sirius BE supports runtime flags to toggle execution mode:

| Flag | Behavior |
|------|----------|
| *(default)* | GPU execution with automatic CPU fallback for unsupported queries |
| `--force-cpu` | All queries run on CPU (DuckDB), GPU is not used at all |
| `--no-cpu-fallback` | GPU-only: queries that can't run on GPU will error instead of falling back |

**To compare GPU vs CPU performance**, stop the BE and restart with `--force-cpu`:

```bash
pixi run -e doris -- doris/target/release/sirius-doris-be \
  --heartbeat-port 19050 --be-port 19060 --brpc-port 18060 \
  --http-port 18040 --arrow-flight-port 18071 \
  --force-cpu --fe 127.0.0.1:9030
```

The FE doesn't know or care whether the BE is using GPU or CPU — the protocol is identical. Benchmark the same queries against the same FE, just toggling the BE's execution mode.

---

## 5. Running Performance Tests (TPC-H)

### TPC-H data setup

You need TPC-H parquet data. Generate it with DuckDB:

```bash
pixi run duckdb <<'SQL'
INSTALL tpch; LOAD tpch; CALL dbgen(sf=1);
COPY lineitem TO '/data/tpch/sf1/snappy/lineitem.parquet' (FORMAT PARQUET);
COPY orders TO '/data/tpch/sf1/snappy/orders.parquet' (FORMAT PARQUET);
COPY customer TO '/data/tpch/sf1/snappy/customer.parquet' (FORMAT PARQUET);
COPY supplier TO '/data/tpch/sf1/snappy/supplier.parquet' (FORMAT PARQUET);
COPY nation TO '/data/tpch/sf1/snappy/nation.parquet' (FORMAT PARQUET);
COPY region TO '/data/tpch/sf1/snappy/region.parquet' (FORMAT PARQUET);
COPY part TO '/data/tpch/sf1/snappy/part.parquet' (FORMAT PARQUET);
COPY partsupp TO '/data/tpch/sf1/snappy/partsupp.parquet' (FORMAT PARQUET);
SQL
```

For partitioned data (used by `run-tpch.sh`), export to directories:
```bash
mkdir -p /data/tpch/sf1/p16/snappy
pixi run duckdb <<'SQL'
INSTALL tpch; LOAD tpch; CALL dbgen(sf=1);
COPY (SELECT *, (row_number() OVER ()) % 16 AS part FROM lineitem) TO '/data/tpch/sf1/p16/snappy/lineitem' (FORMAT PARQUET, PARTITION_BY part);
COPY (SELECT *, (row_number() OVER ()) % 16 AS part FROM orders) TO '/data/tpch/sf1/p16/snappy/orders' (FORMAT PARQUET, PARTITION_BY part);
COPY (SELECT *, (row_number() OVER ()) % 16 AS part FROM customer) TO '/data/tpch/sf1/p16/snappy/customer' (FORMAT PARQUET, PARTITION_BY part);
COPY (SELECT *, (row_number() OVER ()) % 16 AS part FROM supplier) TO '/data/tpch/sf1/p16/snappy/supplier' (FORMAT PARQUET, PARTITION_BY part);
COPY (SELECT *, (row_number() OVER ()) % 16 AS part FROM nation) TO '/data/tpch/sf1/p16/snappy/nation' (FORMAT PARQUET, PARTITION_BY part);
COPY (SELECT *, (row_number() OVER ()) % 16 AS part FROM region) TO '/data/tpch/sf1/p16/snappy/region' (FORMAT PARQUET, PARTITION_BY part);
COPY (SELECT *, (row_number() OVER ()) % 16 AS part FROM part) TO '/data/tpch/sf1/p16/snappy/part' (FORMAT PARQUET, PARTITION_BY part);
COPY (SELECT *, (row_number() OVER ()) % 16 AS part FROM partsupp) TO '/data/tpch/sf1/p16/snappy/partsupp' (FORMAT PARQUET, PARTITION_BY part);
SQL
```

### Manual query testing

```bash
pixi run -e doris-fe -- mysql -h 127.0.0.1 -P 9030 -u root
```

```sql
-- TPC-H Q1 (GPU-accelerated: group by + aggregation)
SELECT l_returnflag, l_linestatus,
       SUM(l_quantity) as sum_qty,
       SUM(l_extendedprice) as sum_base_price,
       COUNT(*) as count_order
FROM local("file_path"="/data/tpch/sf1/snappy/lineitem.parquet",
           "format"="parquet", "shared_storage"="true")
WHERE l_shipdate <= DATE '1998-09-02'
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;

-- TPC-H Q6 (GPU-accelerated: simple aggregation)
SELECT SUM(l_extendedprice * l_discount) as revenue
FROM local("file_path"="/data/tpch/sf1/snappy/lineitem.parquet",
           "format"="parquet", "shared_storage"="true")
WHERE l_shipdate >= DATE '1994-01-01'
  AND l_shipdate < DATE '1995-01-01'
  AND l_discount BETWEEN 0.05 AND 0.07
  AND l_quantity < 24;
```

**Note on `local()` TVF:** Use `"shared_storage"="true"` so the FE picks which BE to send the query to. Alternatively, specify `"backend_id"="<id>"` (get IDs from `SHOW BACKENDS`).

### Automated test script

The repo includes `doris/scripts/run-tpch.sh` which runs all 22 TPC-H queries, rewrites table names to `local()` TVFs, and reports pass/fail status. It expects the FE and BE(s) to already be running (see Section 3):

```bash
# Run specific queries:
./doris/scripts/run-tpch.sh --skip-build --queries 1,6,14

# All 22 queries with 1 BE:
./doris/scripts/run-tpch.sh --skip-build --bes 1

# Custom data directory:
./doris/scripts/run-tpch.sh --skip-build --data-dir /path/to/tpch/sf10/p16/snappy
```

### GPU-verified TPC-H queries
These queries have been verified to run on the GPU pipeline: **Q1, Q2, Q4, Q5, Q6, Q12, Q14**. Other queries may fall back to CPU execution (which is still correct, just not GPU-accelerated).

---

## 6. Configuration Tuning

### GPU Memory (`~/.sirius/sirius.cfg`)

Controls GPU memory allocation per BE. The default config allocates 20% of GPU VRAM per BE (designed for 2 BEs sharing 1 GPU):

```libconfig
sirius = {
    topology = { num_gpus = 1; };
    memory = {
        gpu = {
            usage_limit_fraction = 0.2;       // Max 20% of VRAM
            reservation_limit_fraction = 0.2;
        };
        host = {
            capacity_bytes = 4294967296;       // 4 GB host memory spill
        };
    };
    executor = {
        pipeline = { num_threads = 4; };
    };
};
```

**For a dedicated single-BE setup**, increase the GPU fraction:
```libconfig
gpu = {
    usage_limit_fraction = 0.8;
    reservation_limit_fraction = 0.8;
    downgrade_trigger_fraction = 0.7;
    downgrade_stop_fraction = 0.6;
};
```

### BE CLI flags reference

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu-cache-size` | `2GB` | GPU memory for caching data |
| `--gpu-processing-size` | `2GB` | GPU memory for query processing |
| `--force-cpu` | `false` | Disable GPU, use CPU-only DuckDB |
| `--no-cpu-fallback` | `false` | Error on GPU failures instead of falling back |
| `--nixl-only` | `false` | Require GPU-direct exchange (no bRPC fallback) |
| `--gpu-ids` | `0` | Comma-separated GPU device IDs |
| `--fe` | *(none)* | FE address for auto-registration (e.g. `127.0.0.1:9030`) |

---

## 7. Troubleshooting

### BE logs
Logs go to stdout in the terminal where you started the BE. Log level is controlled by `RUST_LOG`:
```bash
RUST_LOG=doris_rpc=debug,plan_translator=debug,sirius_ffi=debug,info pixi run -e doris sirius-be
```

### Common issues
- **BE crashes on start ("FATAL: engine init failed")**: The substrait or sirius DuckDB extension failed to load. Verify `build/release/extension/sirius/sirius.duckdb_extension` and `substrait/build/release/extension/substrait/substrait.duckdb_extension` exist and are built for the correct DuckDB version (v1.4.4).
- **"Extension already loaded in another process"**: Another BE holds `~/.sirius/sirius.lock`. The `sirius-be-2` pixi task handles this automatically by using a separate home directory. If running BEs manually, use different `HOME` dirs or remove the stale lock: `rm -f ~/.sirius/sirius.lock`
- **"CUDA error" on startup**: Verify `nvidia-smi` works and CUDA driver version is >= 13.0 (or >= 12.0 for cuda12 env).
- **BE not appearing in `SHOW BACKENDS`**: Check that `--fe` points to the correct FE address. The BE self-registers via the FE HTTP API on port 8030.
- **Query hangs/timeout**: `COUNT(*)` without `GROUP BY` causes a GPU hang (known DUMMY_SCAN issue). Use `COUNT(*) ... GROUP BY ...` or add `WHERE` clauses. The GPU engine falls back to CPU for unsupported patterns in default mode.

### Known limitations
- **Hash-partitioned exchange not implemented**: Multi-BE `GROUP BY` queries may produce incorrect results (duplicate rows). Use single-BE for accurate benchmarks.
- **Nixl GPU-direct exchange**: Inter-BE data exchange currently falls back to bRPC. Nixl GPU-direct transfers are not yet working because cuDF/RMM uses CUDA virtual memory allocations that UCX cannot register for direct transfer. The bRPC fallback is correct but slower.
- **`COUNT(*)` without `GROUP BY`**: Produces a degenerate plan that hangs the GPU engine. Add a `GROUP BY` or `WHERE` clause, or use `--force-cpu`.
- **Some TPC-H queries (Q3, Q10, Q13)**: Have column ordering issues in the Substrait translation and fall back to CPU.

---

## Summary: Quick Start Checklist

```bash
# 1. Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Clone and init
git clone --recurse-submodules --single-branch --branch doris https://github.com/mbrobbel/sirius.git sirius-doris
cd sirius-doris

# 3. Build (all four components)
pixi run make release              # C++ GPU engine
pixi run -e doris substrait-build  # Substrait extension
pixi run -e doris doris-build      # NIXL + Rust BE
pixi run -e doris-fe doris-fe-build  # Doris FE

# 4. Start (3 terminals)
pixi run -e doris-fe doris-fe      # Terminal 1: FE
pixi run -e doris sirius-be        # Terminal 2: GPU BE

# 5. Test
pixi run -e doris-fe -- mysql -h 127.0.0.1 -P 9030 -u root -e "SELECT 1"

# 6. Run TPC-H
pixi run -e doris-fe -- mysql -h 127.0.0.1 -P 9030 -u root -e "
SELECT l_returnflag, SUM(l_quantity), COUNT(*)
FROM local(\"file_path\"=\"/data/tpch/sf1/snappy/lineitem.parquet\",
           \"format\"=\"parquet\", \"shared_storage\"=\"true\")
WHERE l_shipdate <= DATE '1998-09-02'
GROUP BY l_returnflag ORDER BY l_returnflag"

# 7. Compare GPU vs CPU: restart BE with --force-cpu and re-run
```
