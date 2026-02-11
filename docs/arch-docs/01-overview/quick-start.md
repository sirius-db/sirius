# Quick Start Guide

This guide will help you build Sirius and run your first GPU-accelerated query in under 30 minutes.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with compute capability 7.0+ (Volta, Turing, Ampere, or newer)
  - Check your GPU: `nvidia-smi`
- **GPU Memory**: 8GB minimum (16GB+ recommended)
- **System RAM**: 32GB+ recommended

### Software Requirements
- **OS**: Linux (Ubuntu 20.04+, RHEL 8+, or similar)
- **CUDA**: 11.5 or higher
  - Verify: `nvcc --version`
- **GCC**: 9.x or 10.x (for CUDA compatibility)
  - Verify: `gcc --version`
- **CMake**: 3.20 or higher
  - Verify: `cmake --version`
- **Git**: For cloning the repository
- **Pixi**: Package manager (recommended)

> **For Beginners**: If you don't have CUDA installed, follow [NVIDIA's CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

---

## Installation Methods

### Method 1: Using Pixi (Recommended)

Pixi is a package manager that handles all dependencies automatically.

```bash
# Install Pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Navigate to Sirius directory
cd /home/roaramburu/coding/sirius

# Install dependencies and build
pixi run build

# Run tests to verify installation
pixi run test
```

### Method 2: Manual Build

If you prefer manual dependency management:

```bash
# Navigate to Sirius directory
cd /home/roaramburu/coding/sirius

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80  # Adjust for your GPU (70, 75, 80, 86, 89, 90)

# Build (use -j to parallelize)
make -j$(nproc)

# Install (optional)
sudo make install
```

#### Determining CUDA Architecture

Your GPU's compute capability determines the CUDA architecture:

| GPU Family | Compute Capability | CMake Flag |
|------------|-------------------|------------|
| Volta (V100) | 7.0 | 70 |
| Turing (RTX 20xx) | 7.5 | 75 |
| Ampere (A100, RTX 30xx) | 8.0 | 80 |
| Ada Lovelace (RTX 40xx) | 8.9 | 89 |
| Hopper (H100) | 9.0 | 90 |

Check your GPU's compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

---

## Running Your First Query

### Step 1: Start DuckDB with Sirius

```bash
# Launch DuckDB (assuming Pixi environment)
pixi run duckdb

# Or if built manually
./build/duckdb
```

### Step 2: Load Sirius Extension

```sql
-- Load the Sirius extension
LOAD 'sirius';

-- Verify it loaded successfully
SELECT * FROM duckdb_extensions() WHERE extension_name = 'sirius';
```

### Step 3: Create Sample Data

```sql
-- Create a sample table with 10 million rows
CREATE TABLE test_data AS
SELECT
    i AS id,
    i * 2 AS value,
    i % 100 AS category,
    DATE '2024-01-01' + INTERVAL (i % 365) DAY AS date
FROM range(10000000) t(i);
```

### Step 4: Run GPU-Accelerated Query

```sql
-- Run query on GPU using new mode (recommended)
SELECT * FROM gpu_execution('
    SELECT category, COUNT(*) as count, SUM(value) as total
    FROM test_data
    WHERE value > 1000000
    GROUP BY category
    ORDER BY total DESC
') LIMIT 10;
```

### Step 5: Compare with CPU Execution

```sql
-- Run same query on CPU for comparison
.timer on

-- CPU execution
SELECT category, COUNT(*) as count, SUM(value) as total
FROM test_data
WHERE value > 1000000
GROUP BY category
ORDER BY total DESC
LIMIT 10;

-- GPU execution
SELECT * FROM gpu_execution('
    SELECT category, COUNT(*) as count, SUM(value) as total
    FROM test_data
    WHERE value > 1000000
    GROUP BY category
    ORDER BY total DESC
') LIMIT 10;
```

Expected output:
```
CPU time: ~500ms
GPU time: ~50ms (10x speedup)
```

---

## Configuration

### Loading Configuration File

Sirius can be configured via a configuration file:

```sql
-- Set config file path
SET sirius_config_path = '/path/to/sirius.cfg';
```

### Sample Configuration File

Create `sirius.cfg`:

```ini
# Thread pool sizes
pipeline_executor_threads=4
task_creator_threads=2
downgrade_executor_threads=2
duckdb_scan_executor_threads=4

# Memory limits (in MB)
gpu_memory_limit=8192
host_memory_limit=32768

# Logging
log_level=INFO
log_file=/tmp/sirius.log

# Debugging
enable_fallback=true  # Fallback to DuckDB on errors
```

Load the config:

```sql
-- In DuckDB
LOAD 'sirius';
SET sirius_config_path = '/path/to/sirius.cfg';
```

---

## Common Issues and Solutions

### Issue 1: "CUDA driver version is insufficient"

**Cause**: CUDA runtime version > driver version

**Solution**:
```bash
# Check driver version
nvidia-smi

# Check CUDA version
nvcc --version

# Update NVIDIA drivers
sudo ubuntu-drivers autoinstall
sudo reboot
```

### Issue 2: "Cannot allocate GPU memory"

**Cause**: GPU memory exhausted

**Solution**:
1. Check GPU memory usage: `nvidia-smi`
2. Kill other GPU processes
3. Reduce batch size in config
4. Enable spilling to host memory

### Issue 3: "Library not found: libcudf.so"

**Cause**: cuDF/RAPIDS libraries not in library path

**Solution**:
```bash
# Add RAPIDS libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/rapids/lib:$LD_LIBRARY_PATH

# Or use Pixi environment which handles this automatically
pixi shell
```

### Issue 4: Build fails with "incompatible CUDA architecture"

**Cause**: Wrong CUDA architecture specified

**Solution**:
```bash
# Rebuild with correct architecture
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80  # Change to match your GPU
make clean
make -j$(nproc)
```

### Issue 5: "gpu_execution not found"

**Cause**: Sirius extension not loaded

**Solution**:
```sql
-- Explicitly load the extension
LOAD 'sirius';

-- Check extension is loaded
SELECT * FROM duckdb_extensions() WHERE loaded = true;
```

---

## Benchmarking: TPC-H

Sirius includes TPC-H benchmark queries for performance testing:

### Generate TPC-H Data

```bash
# Using DuckDB's built-in TPC-H extension
cd /home/roaramburu/coding/sirius
pixi run duckdb
```

```sql
-- Install and load TPC-H extension
INSTALL tpch;
LOAD tpch;

-- Generate scale factor 1 (1GB) data
CALL dbgen(sf=1);

-- Verify tables created
SHOW TABLES;
```

### Run TPC-H Query

```sql
-- TPC-H Q1 on CPU
.timer on
SELECT
    l_returnflag,
    l_linestatus,
    SUM(l_quantity) as sum_qty,
    SUM(l_extendedprice) as sum_base_price,
    COUNT(*) as count_order
FROM lineitem
WHERE l_shipdate <= DATE '1998-09-01'
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;

-- TPC-H Q1 on GPU
SELECT * FROM gpu_execution('
    SELECT
        l_returnflag,
        l_linestatus,
        SUM(l_quantity) as sum_qty,
        SUM(l_extendedprice) as sum_base_price,
        COUNT(*) as count_order
    FROM lineitem
    WHERE l_shipdate <= DATE ''1998-09-01''
    GROUP BY l_returnflag, l_linestatus
    ORDER BY l_returnflag, l_linestatus
');
```

---

## Development Workflow

### Building After Code Changes

```bash
# Quick incremental build
cd /home/roaramburu/coding/sirius/build
make -j$(nproc)

# Or with Pixi
pixi run build
```

### Running Tests

```bash
# Run all tests
cd /home/roaramburu/coding/sirius
pixi run test

# Run specific test category
pixi run test_unit        # C++ unit tests
pixi run test_sql         # SQL integration tests

# Run specific test file
cd build
./test/cpp/test_operators
```

### Debugging

Enable debug logging:

```sql
-- In DuckDB
SET sirius_log_level = 'DEBUG';

-- Run query
SELECT * FROM gpu_execution('SELECT * FROM test_data LIMIT 10');
```

Check logs:
```bash
tail -f /tmp/sirius.log
```

---

## Next Steps

Now that you have Sirius running:

1. **Explore the Architecture**: [System Overview](../02-architecture/system-overview.md)
2. **Understand Execution Modes**: [Execution Modes](../02-architecture/execution-modes.md)
3. **Deep Dive into New Mode**: [New Mode Overview](../04-new-mode/overview.md)
4. **Development Guide**: [Building and Testing](../07-development/building-and-testing.md)

---

## Additional Resources

### Documentation
- [DuckDB Extension API](https://duckdb.org/docs/extensions/overview)
- [cuDF Documentation](https://docs.rapids.ai/api/cudf/stable/)
- [RMM Documentation](https://docs.rapids.ai/api/rmm/stable/)

### Tools
- **nvidia-smi**: Monitor GPU utilization
- **nvprof/nsys**: CUDA profiling
- **gdb**: Debug C++ code
- **valgrind**: Memory leak detection (CPU only)

### Getting Help
- Check [Debugging Guide](../07-development/debugging.md)
- Review [API Reference](../08-reference/api-reference.md)
- Search [File Index](../08-reference/file-index.md) for specific components

---

## Summary

You've successfully:
- ✅ Built Sirius from source
- ✅ Run your first GPU-accelerated query
- ✅ Compared GPU vs CPU performance
- ✅ Configured Sirius for your environment

**Ready to dive deeper?** Continue to [Key Concepts](key-concepts.md) to understand Sirius internals.
