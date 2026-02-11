# Configuration System

This document describes Sirius's configuration system, including global settings, per-connection context, and configuration file formats.

## Overview

Sirius uses a hierarchical configuration system:

1. **Global Configuration** (`sirius_config`) - System-wide settings
2. **Connection Context** (`SiriusContext`) - Per-connection state
3. **Configuration File** - YAML/INI format for persistent settings

---

## Configuration Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│ Configuration File (sirius.cfg)                          │
│ - Loaded at startup                                      │
│ - Persistent settings                                    │
└─────────────────────────────────────────────────────────┘
                    ↓ (loads into)
┌─────────────────────────────────────────────────────────┐
│ sirius_config (Global Singleton)                        │
│ - Thread pool sizes                                      │
│ - Memory limits                                          │
│ - Logging configuration                                  │
│ - Hardware topology                                      │
└─────────────────────────────────────────────────────────┘
                    ↓ (used by)
┌─────────────────────────────────────────────────────────┐
│ SiriusContext (Per-Connection)                          │
│ - Active queries                                         │
│ - Memory reservations                                    │
│ - Query-specific overrides                               │
└─────────────────────────────────────────────────────────┘
```

---

## Global Configuration (sirius_config)

### File Location

**Default**: Searches in order:
1. Path specified by `SIRIUS_CONFIG` environment variable
2. `./sirius.cfg` (current directory)
3. `~/.sirius/sirius.cfg` (user home)
4. `/etc/sirius/sirius.cfg` (system-wide)

**Override in SQL**:
```sql
SET sirius_config_path = '/path/to/custom/sirius.cfg';
```

### Configuration File Format

#### Example Configuration

```ini
# Thread Pool Configuration
[threading]
pipeline_executor_threads = 4
task_creator_threads = 2
downgrade_executor_threads = 2
duckdb_scan_executor_threads = 4

# Memory Configuration (in MB)
[memory]
gpu_memory_limit = 8192      # 8GB
host_memory_limit = 32768    # 32GB
disk_memory_limit = -1       # Unlimited

# CUDA Configuration
[cuda]
cuda_streams_per_executor = 1
enable_cuda_graphs = false

# Logging Configuration
[logging]
log_level = INFO             # DEBUG, INFO, WARNING, ERROR
log_file = /tmp/sirius.log
enable_console_logging = true

# Execution Configuration
[execution]
scan_batch_size = 100000
enable_spilling = true
enable_fallback = true

# Hardware Configuration
[hardware]
gpu_device_id = 0            # Which GPU to use
num_gpus = 1                 # Number of GPUs (multi-GPU future)
```

#### YAML Format (Alternative)

```yaml
threading:
  pipeline_executor_threads: 4
  task_creator_threads: 2
  downgrade_executor_threads: 2
  duckdb_scan_executor_threads: 4

memory:
  gpu_memory_limit: 8192
  host_memory_limit: 32768
  disk_memory_limit: -1

cuda:
  cuda_streams_per_executor: 1
  enable_cuda_graphs: false

logging:
  log_level: INFO
  log_file: /tmp/sirius.log
  enable_console_logging: true

execution:
  scan_batch_size: 100000
  enable_spilling: true
  enable_fallback: true

hardware:
  gpu_device_id: 0
  num_gpus: 1
```

---

## Configuration Options Reference

### Threading Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `pipeline_executor_threads` | int | 4 | Threads for GPU pipeline execution |
| `task_creator_threads` | int | 2 | Threads for dynamic task generation |
| `downgrade_executor_threads` | int | 2 | Threads for memory tier management |
| `duckdb_scan_executor_threads` | int | 4 | Threads for CPU-based table scans |

**Guidelines**:
- `pipeline_executor_threads`: 4-8 for GPU-bound workloads
- `task_creator_threads`: 2-4 is usually sufficient
- `downgrade_executor_threads`: 2-4 for memory management
- `duckdb_scan_executor_threads`: Match I/O parallelism needs
- **Total threads**: Don't exceed 2x physical CPU cores

### Memory Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `gpu_memory_limit` | int (MB) | Auto-detect | Maximum GPU memory usage |
| `host_memory_limit` | int (MB) | Auto-detect | Maximum host memory for staging |
| `disk_memory_limit` | int (MB) | -1 (unlimited) | Maximum disk space for spilling |
| `enable_spilling` | bool | true | Enable multi-tier memory spilling |

**Guidelines**:
- `gpu_memory_limit`: Set to 75-80% of available GPU memory
- `host_memory_limit`: Set to 75% of available RAM
- `disk_memory_limit`: -1 for unlimited, or set limit based on disk space

**Auto-detection**:
```cpp
// Sirius auto-detects available memory if not specified
gpu_memory_limit = detect_gpu_memory() * 0.75;
host_memory_limit = detect_host_memory() * 0.75;
```

### CUDA Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `cuda_streams_per_executor` | int | 1 | CUDA streams per executor thread |
| `enable_cuda_graphs` | bool | false | Enable CUDA graph optimization (experimental) |
| `gpu_device_id` | int | 0 | GPU device ID to use |

**Guidelines**:
- `cuda_streams_per_executor`: Usually 1 is optimal
- `enable_cuda_graphs`: Experimental, can reduce kernel launch overhead
- `gpu_device_id`: For multi-GPU systems, specify which GPU

### Logging Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `log_level` | string | INFO | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `log_file` | string | /tmp/sirius.log | Path to log file |
| `enable_console_logging` | bool | true | Also log to console (stderr) |
| `log_sql_queries` | bool | false | Log all SQL queries |

**Log Levels**:
- **DEBUG**: Verbose, all operations (development)
- **INFO**: Important events (default)
- **WARNING**: Warnings and errors
- **ERROR**: Errors only

### Execution Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `scan_batch_size` | int | 100000 | Rows per batch when scanning |
| `enable_spilling` | bool | true | Enable memory spilling to host/disk |
| `enable_fallback` | bool | true | Fallback to DuckDB on errors |
| `enable_query_caching` | bool | false | Cache query plans (experimental) |

**Guidelines**:
- `scan_batch_size`: Adjust based on table width (fewer rows for wide tables)
- `enable_spilling`: Always true for production
- `enable_fallback`: true for development, false for production (catch errors early)

### Hardware Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `gpu_device_id` | int | 0 | Which GPU to use (0-indexed) |
| `num_gpus` | int | 1 | Number of GPUs (multi-GPU planned) |
| `numa_node` | int | -1 | NUMA node affinity (-1 = auto) |

---

## Per-Connection Context (SiriusContext)

### SiriusContext Structure

**File**: `src/sirius_context.cpp`

```cpp
class SiriusContext {
public:
    // Associated DuckDB context
    ClientContext& duckdb_context;

    // Sirius engine instance
    unique_ptr<sirius_engine> engine;

    // Configuration (copy of global with overrides)
    unique_ptr<sirius_config> config;

    // Memory management
    shared_ptr<sirius_memory_reservation_manager> memory_manager;

    // Active queries
    vector<shared_ptr<sirius_prepared_statement_data>> active_queries;

    // Per-connection state
    unordered_map<string, Value> session_variables;
};
```

### Context Lifecycle

```
DuckDB Connection Created
     ↓
Load Sirius Extension
     ↓
SiriusContext::Create()
├─ Initialize GPU context
├─ Load global config
├─ Create memory manager
└─ Initialize task executors
     ↓
Execute Queries (multiple)
├─ gpu_execution() calls
├─ Query planning
└─ GPU execution
     ↓
DuckDB Connection Closed
     ↓
SiriusContext::Destroy()
├─ Cleanup active queries
├─ Free GPU memory
└─ Shutdown task executors
```

### Context-Specific Configuration

Override global config per connection:

```sql
-- Create connection
LOAD 'sirius';

-- Override config for this connection
SET sirius_gpu_memory_limit = 4096;  -- Use only 4GB on this connection
SET sirius_log_level = 'DEBUG';      -- Verbose logging

-- Run query with overridden config
SELECT * FROM gpu_execution('SELECT ...');
```

---

## Runtime Configuration (SQL)

### Setting Options

```sql
-- Set configuration option
SET sirius_<option_name> = <value>;

-- Examples:
SET sirius_log_level = 'DEBUG';
SET sirius_gpu_memory_limit = 8192;
SET sirius_enable_fallback = true;
```

### Viewing Options

```sql
-- View all Sirius options
SELECT * FROM duckdb_settings() WHERE name LIKE 'sirius%';

-- View specific option
SELECT * FROM duckdb_settings() WHERE name = 'sirius_log_level';
```

### Resetting Options

```sql
-- Reset to default
RESET sirius_log_level;

-- Reset all Sirius options
RESET sirius%;
```

---

## Configuration Validation

### Validation at Load

Sirius validates configuration at startup:

```cpp
void sirius_config::validate() {
    // Thread counts
    if (pipeline_executor_threads < 1 || pipeline_executor_threads > 64) {
        throw ConfigurationException("pipeline_executor_threads must be 1-64");
    }

    // Memory limits
    if (gpu_memory_limit > detect_gpu_memory()) {
        throw ConfigurationException(
            "gpu_memory_limit exceeds available GPU memory");
    }

    // Log level
    if (log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}) {
        throw ConfigurationException("Invalid log_level");
    }
}
```

### Runtime Validation

```sql
-- Invalid setting
SET sirius_gpu_memory_limit = 999999;
-- Error: Exceeds available GPU memory
```

---

## Advanced Configuration

### Hardware Topology Detection

Sirius auto-detects hardware at startup:

```cpp
struct HardwareTopology {
    // GPU information
    int num_gpus;
    vector<size_t> gpu_memory_sizes;
    vector<int> gpu_compute_capabilities;

    // CPU information
    int num_cpu_cores;
    int num_numa_nodes;
    size_t host_memory_size;

    // Storage information
    size_t available_disk_space;
    bool has_nvme;
};
```

**Usage**:
```sql
-- View detected hardware
SELECT * FROM sirius_hardware_info();
```

**Output**:
```
┌──────────┬────────────┬────────────┬──────────────┐
│ num_gpus │ gpu_memory │ cpu_cores  │ host_memory  │
├──────────┼────────────┼────────────┼──────────────┤
│        1 │   16384 MB │         64 │    524288 MB │
└──────────┴────────────┴────────────┴──────────────┘
```

### Profile-Based Configuration

Create configuration profiles for different workloads:

**profile_scan_heavy.cfg**:
```ini
[threading]
duckdb_scan_executor_threads = 8
pipeline_executor_threads = 2

[execution]
scan_batch_size = 200000
```

**profile_compute_heavy.cfg**:
```ini
[threading]
pipeline_executor_threads = 6
duckdb_scan_executor_threads = 2

[execution]
scan_batch_size = 50000
```

**Load profile**:
```sql
SET sirius_config_path = '/path/to/profile_scan_heavy.cfg';
```

---

## Configuration Best Practices

### 1. Start with Defaults

Default configuration is tuned for balanced workloads:

```ini
# Good starting point
pipeline_executor_threads = 4
task_creator_threads = 2
downgrade_executor_threads = 2
duckdb_scan_executor_threads = 4
gpu_memory_limit = auto
host_memory_limit = auto
```

### 2. Tune for Workload

**I/O-Heavy Workloads**:
```ini
duckdb_scan_executor_threads = 8  # More I/O parallelism
scan_batch_size = 200000          # Larger batches
```

**Compute-Heavy Workloads**:
```ini
pipeline_executor_threads = 6     # More GPU parallelism
scan_batch_size = 50000           # Smaller batches
```

**Memory-Constrained**:
```ini
gpu_memory_limit = 6144           # Conservative limit
enable_spilling = true            # Always enable
scan_batch_size = 50000           # Smaller batches
```

### 3. Monitor and Adjust

```sql
-- Enable monitoring
SET sirius_enable_monitoring = true;

-- Run query
SELECT * FROM gpu_execution('...');

-- View statistics
SELECT * FROM sirius_execution_stats();
```

**Metrics to Monitor**:
- GPU utilization: Should be 70-100% during query
- Memory usage: Should stay below limits
- Thread pool saturation: Check for queued tasks
- Spilling frequency: Minimize but allow when needed

### 4. Environment-Specific Config

**Development**:
```ini
log_level = DEBUG
enable_fallback = true
enable_console_logging = true
```

**Production**:
```ini
log_level = WARNING
enable_fallback = false  # Catch unsupported queries
log_file = /var/log/sirius/sirius.log
```

---

## Programmatic Configuration

### C++ API

```cpp
#include "sirius_config.hpp"

// Get global config
auto& config = sirius_config::get_instance();

// Modify settings
config.pipeline_executor_threads = 6;
config.gpu_memory_limit = 8192;

// Reload from file
config.load_from_file("/path/to/sirius.cfg");
```

### Python API (via DuckDB)

```python
import duckdb

conn = duckdb.connect()
conn.execute("LOAD 'sirius'")

# Set configuration
conn.execute("SET sirius_log_level = 'DEBUG'")
conn.execute("SET sirius_gpu_memory_limit = 8192")

# Run query
result = conn.execute("""
    SELECT * FROM gpu_execution('SELECT * FROM large_table')
""").fetchall()
```

---

## Troubleshooting Configuration

### Issue 1: Configuration Not Loading

**Problem**: Changes to config file not reflected

**Solution**:
```sql
-- Force reload
SET sirius_config_path = '/path/to/sirius.cfg';

-- Or restart DuckDB
```

### Issue 2: Invalid Configuration

**Problem**: Error on startup

**Check logs**:
```bash
cat /tmp/sirius.log
# Look for ConfigurationException
```

**Validate manually**:
```bash
# Test config file
sirius-config-validator sirius.cfg
```

### Issue 3: Performance Not Improved

**Problem**: Configuration changes don't help

**Debug**:
```sql
-- View actual settings
SELECT * FROM duckdb_settings() WHERE name LIKE 'sirius%';

-- View execution stats
SELECT * FROM sirius_execution_stats();

-- Check if settings applied
```

---

## Configuration File Templates

### Minimal Configuration

```ini
# Minimal config - uses mostly defaults
[logging]
log_level = INFO
log_file = /tmp/sirius.log
```

### Production Configuration

```ini
# Production-ready configuration
[threading]
pipeline_executor_threads = 4
task_creator_threads = 2
downgrade_executor_threads = 2
duckdb_scan_executor_threads = 4

[memory]
gpu_memory_limit = 12288
host_memory_limit = 49152
disk_memory_limit = 102400
enable_spilling = true

[logging]
log_level = WARNING
log_file = /var/log/sirius/sirius.log
enable_console_logging = false
log_sql_queries = false

[execution]
scan_batch_size = 100000
enable_fallback = false
enable_query_caching = true

[hardware]
gpu_device_id = 0
```

### Development Configuration

```ini
# Development configuration
[threading]
pipeline_executor_threads = 2
task_creator_threads = 1
downgrade_executor_threads = 1
duckdb_scan_executor_threads = 2

[memory]
gpu_memory_limit = 4096
host_memory_limit = 8192
enable_spilling = true

[logging]
log_level = DEBUG
log_file = ./sirius_dev.log
enable_console_logging = true
log_sql_queries = true

[execution]
enable_fallback = true
scan_batch_size = 50000
```

---

## See Also

- [System Overview](../02-architecture/system-overview.md) - Architecture context
- [Performance Tips](../appendices/performance-tips.md) - Tuning guidelines
- [Config Options Reference](../08-reference/config-options.md) - Complete option list
- [Building and Testing](../07-development/building-and-testing.md) - Environment setup
