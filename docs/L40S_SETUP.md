# Sirius cudf-25.12 — L40S Build & Benchmark Setup

## Hardware
- **GPU:** NVIDIA L40S (48GB VRAM, compute capability 8.9, Ada Lovelace)
- **CPU:** 2x Intel Xeon Gold 6526Y (64 threads)
- **RAM:** 54GB (no swap)
- **CUDA Driver:** 535.274.02, CUDA 12.2

## Build Changes from GH200 Baseline

The `cudf-25.12` branch targets GH200 (ARM, sm_90a, 96GB HBM3). These changes adapt it for L40S:

1. **pixi.toml**: `CUDAARCHS` from `"90a-real"` → `"89-real"`
2. **build.sh**: `-DDUCKDB_EXPLICIT_PLATFORM` from `linux_arm64` → `linux_amd64`

## Build Instructions

```bash
cd ~/sirius-cudf25

# Install dependencies (one-time)
pixi install

# Set up substrait extension (one-time)
cd duckdb && mkdir -p extension_external && cd extension_external
git clone https://github.com/duckdb/substrait.git
cd substrait && git reset --hard ec9f8725df7aa22bae7217ece2f221ac37563da4
cd ~/sirius-cudf25

# Build
LIBCUDF_ENV_PREFIX=$HOME/sirius-cudf25/.pixi/envs/default bash build.sh
```

## Memory Configuration

**Constraints:**
- 46GB GPU VRAM available (L40S advertises 48GB, ~46GB usable)
- 54GB system RAM, no swap — pinned memory must stay well under this

**Safe defaults for L40S:**
- GPU caching: 20 GB
- GPU processing: 20 GB
- CPU pinned memory: 40 GB

**Warning:** The GH200 config uses 80GB caching + 40GB processing + 100GB pinned. Using those values on L40S will OOM-kill the process (and may disconnect your SSH session).

## Running ClickBench

```bash
# Create hits.duckdb (one-time, ~3 min)
./build/release/duckdb hits.duckdb -c "CREATE TABLE hits AS SELECT * FROM read_parquet('hits.parquet');"

# Run benchmark
bash run_benchmark.sh

# With debug timing
SIRIUS_LOG_LEVEL=debug bash run_benchmark.sh
```

## Key Differences from GH200

| Aspect | GH200 | L40S |
|--------|-------|------|
| VRAM | 96 GB HBM3 | 46 GB GDDR6X |
| Memory BW | ~4 TB/s | ~864 GB/s |
| Arch | sm_90a (Hopper) | sm_89 (Ada) |
| CPU | ARM Neoverse V2 | x86 Xeon Gold |
| System RAM | 480+ GB | 54 GB |
| sudo | Yes | No (can't drop page caches) |
