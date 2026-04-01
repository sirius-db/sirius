---
name: config-optimizer
description: >
  Use this skill to find the optimal Sirius configuration for TPC-H workloads at any scale factor.
  Trigger when the user wants to tune performance, optimize config parameters, find the best thread
  count, batch size, or cache mode, or benchmark different Sirius configurations against each other.
  Also use when the user mentions "config tuning", "parameter sweep", or "optimal settings".
---

# Sirius Config Optimizer

Tune Sirius configuration parameters for optimal TPC-H performance at various scale factors. Systematically explore the configuration space, evaluate each configuration with profiling, and report the best settings.

**Reference:** See `.claude/skills/_shared/build-and-query.md` for shared infrastructure (build modes, query execution).

## Configuration Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `sirius.executor.pipeline.num_threads` | 1–10 | Number of threads (streams) for GPU pipeline tasks |
| `sirius.executor.duckdb_scan.num_threads` | 1–10 | Number of threads for DuckDB scan operator |
| `sirius.executor.duckdb_scan.cache` | `none`, `parquet`, `table_gpu`, `table_host` | Cache mode: none, parquet row groups in pinned memory, scanned table on GPU, scanned table in pinned memory |
| `sirius.operator_params.scan_task_batch_size` | 500MB–5GB | Batch size in bytes for scan tasks |
| `sirius.operator_params.concat_batch_bytes` | 500MB–5GB | Batch size in bytes for concatenation tasks |
| `sirius.operator_params.hash_partition_bytes` | 500MB–5GB | Batch size in bytes for hash partition tasks |

## Workflow

1. **Gather context:**
   - Ask the user for the baseline configuration file (or check `$SIRIUS_CONFIG_FILE`)
   - Ask for target TPC-H scale factor and dataset location
   - If the dataset doesn't exist, use the `/dataset-manager` skill to generate it

2. **Establish baseline:**
   - Use the `/profile-analyzer` skill to evaluate baseline performance
   - Collect metrics: query latency, GPU utilization, memory usage

3. **Explore the configuration space** using a systematic approach (grid search, random search, or Bayesian optimization). For each configuration:
   - Update the config file using `scripts/patch_config.py`
   - Run the same TPC-H queries and collect performance metrics
   - Compare against baseline and previous configurations
   - Back up the configuration and results for each test

4. **Analyze results** to identify the optimal configuration.

5. **Report** the optimal configuration and its performance metrics, along with insights.

### Changing Configuration

Use `scripts/patch_config.py` to modify parameters:

```bash
cd scripts
pixi run python patch_config.py sirius.cfg \
    --opt sirius.executor.pipeline.num_threads=4 \
    --opt sirius.executor.duckdb_scan.cache=parquet \
    --opt sirius.operator_params.scan_task_batch_size=536870912
```

Arguments:
- `config_file_path` — Path to the Sirius configuration file
- `--opt` — Key-value pair for the config parameter

## Examples

```bash
# Example 1: Tune for SF10
# User: "Find the best config for TPC-H SF10"
# 1. Generate data if needed
# 2. Run baseline with current config
# 3. Sweep thread counts: 1, 2, 4, 6, 8
# 4. Sweep cache modes: none, parquet, table_gpu
# 5. Fine-tune batch sizes around best thread/cache combo

# Example 2: Compare two specific configs
# User: "Compare 4 threads vs 8 threads on SF100"
# 1. Run SF100 queries with num_threads=4, collect timings
# 2. Run SF100 queries with num_threads=8, collect timings
# 3. Present comparison table with speedup ratios
```

## Before Running

- **Ask the user** if they want to rebuild the code if there are upstream changes
- Build with pixi: `pixi run -e clang make release`
- Ensure `SIRIUS_CONFIG_FILE` is set and points to a valid config file

## Output

Create an optimization report with:
- Baseline configuration and performance metrics
- Each configuration tested, parameters changed, and resulting metrics
- Optimal configuration saved to `optimal_sirius.cfg`
- Performance report saved to `optimal_config_report.txt`
- Insights and recommendations for future tuning
