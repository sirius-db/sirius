# Configuration Options Reference

Complete reference of all Sirius configuration options with descriptions, types, defaults, and valid ranges.

## Configuration Methods

Options can be set via:
1. **Configuration File**: `sirius.cfg` (INI/YAML format)
2. **SQL Commands**: `SET sirius_<option> = value;`
3. **Environment Variables**: `SIRIUS_<OPTION>=value`
4. **Programmatic API**: C++/Python

---

## Threading Configuration

### pipeline_executor_threads

**Description**: Number of threads for GPU pipeline execution

| Property | Value |
|----------|-------|
| **Type** | Integer |
| **Default** | 4 |
| **Range** | 1-64 |
| **Restart Required** | No |

**Guidelines**:
- 4-8 for most workloads
- More threads = more parallel GPU execution
- Don't exceed 2x CPU cores

**Example**:
```ini
[threading]
pipeline_executor_threads = 6
```

```sql
SET sirius_pipeline_executor_threads = 6;
```

---

### task_creator_threads

**Description**: Number of threads for dynamic task generation

| Property | Value |
|----------|-------|
| **Type** | Integer |
| **Default** | 2 |
| **Range** | 1-16 |
| **Restart Required** | No |

**Guidelines**:
- 2-4 is sufficient for most cases
- Increase if task creation becomes bottleneck

**Example**:
```ini
[threading]
task_creator_threads = 2
```

---

### downgrade_executor_threads

**Description**: Number of threads for memory tier management (spilling)

| Property | Value |
|----------|-------|
| **Type** | Integer |
| **Default** | 2 |
| **Range** | 1-16 |
| **Restart Required** | No |

**Guidelines**:
- 2-4 threads
- Increase if frequent spilling occurs

**Example**:
```ini
[threading]
downgrade_executor_threads = 2
```

---

### duckdb_scan_executor_threads

**Description**: Number of threads for CPU-based table scans

| Property | Value |
|----------|-------|
| **Type** | Integer |
| **Default** | 4 |
| **Range** | 1-128 |
| **Restart Required** | No |

**Guidelines**:
- Match I/O parallelism needs
- Higher for I/O-bound workloads
- Consider storage backend parallelism

**Example**:
```ini
[threading]
duckdb_scan_executor_threads = 8
```

---

## Memory Configuration

### gpu_memory_limit

**Description**: Maximum GPU memory usage in MB

| Property | Value |
|----------|-------|
| **Type** | Integer (MB) |
| **Default** | Auto-detect (75% of available) |
| **Range** | 512 - GPU memory size |
| **Restart Required** | No |

**Guidelines**:
- Set to 75-80% of available GPU memory
- Leave headroom for OS and other processes
- Lower if sharing GPU with other applications

**Example**:
```ini
[memory]
gpu_memory_limit = 12288  # 12GB
```

```sql
SET sirius_gpu_memory_limit = 12288;
```

---

### host_memory_limit

**Description**: Maximum host memory for staging in MB

| Property | Value |
|----------|-------|
| **Type** | Integer (MB) |
| **Default** | Auto-detect (75% of available RAM) |
| **Range** | 1024 - System RAM |
| **Restart Required** | No |

**Guidelines**:
- Set to 75% of available RAM
- Used for spilling from GPU
- Leave memory for OS and other processes

**Example**:
```ini
[memory]
host_memory_limit = 49152  # 48GB
```

---

### disk_memory_limit

**Description**: Maximum disk space for spilling in MB

| Property | Value |
|----------|-------|
| **Type** | Integer (MB) |
| **Default** | -1 (unlimited) |
| **Range** | -1 (unlimited) or > 1024 |
| **Restart Required** | No |

**Guidelines**:
- -1 for unlimited (uses available disk space)
- Set limit to prevent disk exhaustion
- Requires fast storage (NVMe recommended)

**Example**:
```ini
[memory]
disk_memory_limit = 102400  # 100GB
```

---

### enable_spilling

**Description**: Enable multi-tier memory spilling (GPU → HOST → DISK)

| Property | Value |
|----------|-------|
| **Type** | Boolean |
| **Default** | true |
| **Values** | true, false |
| **Restart Required** | No |

**Guidelines**:
- Always true for production
- Disabling causes OOM for large queries
- Only disable for testing/debugging

**Example**:
```ini
[memory]
enable_spilling = true
```

```sql
SET sirius_enable_spilling = true;
```

---

## CUDA Configuration

### cuda_streams_per_executor

**Description**: Number of CUDA streams per executor thread

| Property | Value |
|----------|-------|
| **Type** | Integer |
| **Default** | 1 |
| **Range** | 1-4 |
| **Restart Required** | Yes |

**Guidelines**:
- 1 is optimal for most cases
- More streams = more concurrency but also overhead
- Only increase if profiling shows benefit

**Example**:
```ini
[cuda]
cuda_streams_per_executor = 1
```

---

### enable_cuda_graphs

**Description**: Enable CUDA graph optimization (experimental)

| Property | Value |
|----------|-------|
| **Type** | Boolean |
| **Default** | false |
| **Values** | true, false |
| **Restart Required** | Yes |

**Guidelines**:
- Experimental feature
- Can reduce kernel launch overhead
- May cause issues with dynamic workloads

**Example**:
```ini
[cuda]
enable_cuda_graphs = false
```

---

### gpu_device_id

**Description**: GPU device ID to use (0-indexed)

| Property | Value |
|----------|-------|
| **Type** | Integer |
| **Default** | 0 |
| **Range** | 0 - (num_gpus - 1) |
| **Restart Required** | Yes |

**Guidelines**:
- For multi-GPU systems
- Check available GPUs with `nvidia-smi`
- Set to use specific GPU

**Example**:
```ini
[cuda]
gpu_device_id = 0
```

```bash
# Or via environment variable
CUDA_VISIBLE_DEVICES=0 duckdb
```

---

## Logging Configuration

### log_level

**Description**: Logging verbosity level

| Property | Value |
|----------|-------|
| **Type** | String |
| **Default** | INFO |
| **Values** | DEBUG, INFO, WARNING, ERROR |
| **Restart Required** | No |

**Guidelines**:
- **DEBUG**: Verbose (development)
- **INFO**: Important events (production default)
- **WARNING**: Warnings and errors
- **ERROR**: Errors only

**Example**:
```ini
[logging]
log_level = INFO
```

```sql
SET sirius_log_level = 'DEBUG';
```

---

### log_file

**Description**: Path to log file

| Property | Value |
|----------|-------|
| **Type** | String (path) |
| **Default** | /tmp/sirius.log |
| **Values** | Any valid file path |
| **Restart Required** | No |

**Guidelines**:
- Use absolute paths
- Ensure directory exists and is writable
- Rotate logs in production

**Example**:
```ini
[logging]
log_file = /var/log/sirius/sirius.log
```

```sql
SET sirius_log_file = '/var/log/sirius/sirius.log';
```

---

### enable_console_logging

**Description**: Also log to console (stderr)

| Property | Value |
|----------|-------|
| **Type** | Boolean |
| **Default** | true |
| **Values** | true, false |
| **Restart Required** | No |

**Example**:
```ini
[logging]
enable_console_logging = true
```

---

### log_sql_queries

**Description**: Log all SQL queries executed

| Property | Value |
|----------|-------|
| **Type** | Boolean |
| **Default** | false |
| **Values** | true, false |
| **Restart Required** | No |

**Guidelines**:
- Enable for debugging query issues
- May produce large log files
- Contains query text (may include sensitive data)

**Example**:
```ini
[logging]
log_sql_queries = false
```

---

## Execution Configuration

### scan_batch_size

**Description**: Number of rows per batch when scanning tables

| Property | Value |
|----------|-------|
| **Type** | Integer |
| **Default** | 100000 |
| **Range** | 1000 - 10000000 |
| **Restart Required** | No |

**Guidelines**:
- 100K is good default
- Lower (50K) for wide tables (many columns)
- Higher (200K) for narrow tables
- Adjust so batches fit in GPU memory

**Example**:
```ini
[execution]
scan_batch_size = 100000
```

```sql
SET sirius_scan_batch_size = 50000;
```

---

### enable_fallback

**Description**: Fallback to DuckDB CPU execution on errors

| Property | Value |
|----------|-------|
| **Type** | Boolean |
| **Default** | true |
| **Values** | true, false |
| **Restart Required** | No |

**Guidelines**:
- **true**: Development (automatic fallback)
- **false**: Production (catch unsupported queries)
- Disable to identify GPU-incompatible queries

**Example**:
```ini
[execution]
enable_fallback = true
```

```sql
SET sirius_enable_fallback = false;
```

---

### enable_query_caching

**Description**: Cache compiled query plans (experimental)

| Property | Value |
|----------|-------|
| **Type** | Boolean |
| **Default** | false |
| **Values** | true, false |
| **Restart Required** | No |

**Guidelines**:
- Experimental feature
- Reuses plans for identical queries
- Can improve repeated query performance

**Example**:
```ini
[execution]
enable_query_caching = false
```

---

### enable_operator_fusion

**Description**: Fuse adjacent operators into single GPU kernel (experimental)

| Property | Value |
|----------|-------|
| **Type** | Boolean |
| **Default** | false |
| **Values** | true, false |
| **Restart Required** | No |

**Guidelines**:
- Experimental optimization
- Can reduce kernel launch overhead
- May improve performance for simple operators

**Example**:
```ini
[execution]
enable_operator_fusion = false
```

---

## Hardware Configuration

### numa_node

**Description**: NUMA node affinity for memory allocation

| Property | Value |
|----------|-------|
| **Type** | Integer |
| **Default** | -1 (auto) |
| **Range** | -1 (auto) or 0 - (num_numa_nodes - 1) |
| **Restart Required** | Yes |

**Guidelines**:
- -1 for automatic NUMA affinity
- Set specific node for NUMA-aware systems
- Match node to GPU's PCIe root

**Example**:
```ini
[hardware]
numa_node = 0
```

---

### num_gpus

**Description**: Number of GPUs to use (multi-GPU planned)

| Property | Value |
|----------|-------|
| **Type** | Integer |
| **Default** | 1 |
| **Range** | 1-8 |
| **Restart Required** | Yes |

**Status**: Currently only 1 GPU supported. Multi-GPU is planned.

**Example**:
```ini
[hardware]
num_gpus = 1
```

---

## Monitoring Configuration

### enable_monitoring

**Description**: Enable performance monitoring and statistics collection

| Property | Value |
|----------|-------|
| **Type** | Boolean |
| **Default** | false |
| **Values** | true, false |
| **Restart Required** | No |

**Guidelines**:
- Enable for performance analysis
- Minimal overhead (~1-2%)
- View stats with `SELECT * FROM sirius_stats();`

**Example**:
```ini
[monitoring]
enable_monitoring = true
```

```sql
SET sirius_enable_monitoring = true;
SELECT * FROM sirius_execution_stats();
```

---

### stats_sample_rate

**Description**: Sampling rate for statistics (0.0-1.0)

| Property | Value |
|----------|-------|
| **Type** | Float |
| **Default** | 1.0 (all queries) |
| **Range** | 0.0 - 1.0 |
| **Restart Required** | No |

**Guidelines**:
- 1.0 = monitor all queries
- 0.1 = monitor 10% of queries
- Lower for high-throughput systems

**Example**:
```ini
[monitoring]
stats_sample_rate = 1.0
```

---

## Configuration Profiles

### Default Profile

```ini
# Balanced configuration for general use
[threading]
pipeline_executor_threads = 4
task_creator_threads = 2
downgrade_executor_threads = 2
duckdb_scan_executor_threads = 4

[memory]
gpu_memory_limit = auto
host_memory_limit = auto
disk_memory_limit = -1
enable_spilling = true

[cuda]
cuda_streams_per_executor = 1
gpu_device_id = 0

[logging]
log_level = INFO
log_file = /tmp/sirius.log
enable_console_logging = true

[execution]
scan_batch_size = 100000
enable_fallback = true
```

### High-Throughput Profile

```ini
# Optimized for many concurrent queries
[threading]
pipeline_executor_threads = 8
task_creator_threads = 4
duckdb_scan_executor_threads = 8

[execution]
scan_batch_size = 50000  # Smaller batches
```

### Memory-Constrained Profile

```ini
# For systems with limited GPU memory
[memory]
gpu_memory_limit = 4096   # 4GB
host_memory_limit = 16384  # 16GB
enable_spilling = true

[execution]
scan_batch_size = 50000  # Smaller batches
```

### Development Profile

```ini
# For development and debugging
[logging]
log_level = DEBUG
log_sql_queries = true
enable_console_logging = true

[execution]
enable_fallback = true

[monitoring]
enable_monitoring = true
```

---

## Environment Variables

All config options can be set via environment variables:

```bash
# Format: SIRIUS_<SECTION>_<OPTION> (uppercase with underscores)

export SIRIUS_THREADING_PIPELINE_EXECUTOR_THREADS=6
export SIRIUS_MEMORY_GPU_MEMORY_LIMIT=8192
export SIRIUS_LOGGING_LOG_LEVEL=DEBUG

duckdb
```

---

## SQL Configuration Commands

### View Current Settings

```sql
-- All Sirius settings
SELECT * FROM duckdb_settings() WHERE name LIKE 'sirius%';

-- Specific setting
SELECT * FROM duckdb_settings() WHERE name = 'sirius_log_level';
```

### Modify Settings

```sql
-- Set option
SET sirius_log_level = 'DEBUG';
SET sirius_gpu_memory_limit = 8192;

-- Reset to default
RESET sirius_log_level;

-- Reset all Sirius options
RESET sirius%;
```

---

## Configuration Precedence

When multiple configuration sources exist, the precedence is:

1. **SQL SET commands** (highest priority, session-only)
2. **Environment variables**
3. **Configuration file**
4. **Built-in defaults** (lowest priority)

**Example**:
```bash
# Config file: gpu_memory_limit = 8192
# Environment: SIRIUS_MEMORY_GPU_MEMORY_LIMIT=12288
# SQL: SET sirius_gpu_memory_limit = 16384;

# Result: 16384 (SQL takes precedence)
```

---

## Validation

Invalid configurations are caught at startup or when set:

```sql
-- Invalid: threads out of range
SET sirius_pipeline_executor_threads = 999;
-- Error: Value must be between 1 and 64

-- Invalid: memory exceeds available
SET sirius_gpu_memory_limit = 999999;
-- Error: Exceeds available GPU memory

-- Invalid: unknown log level
SET sirius_log_level = 'VERBOSE';
-- Error: Must be DEBUG, INFO, WARNING, or ERROR
```

---

## See Also

- [Configuration System](../05-core-components/configuration.md) - Detailed configuration guide
- [Performance Tips](../appendices/performance-tips.md) - Configuration tuning
- [Building and Testing](../07-development/building-and-testing.md) - Environment setup
