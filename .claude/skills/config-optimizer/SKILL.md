---
name: config-optimizer
description: find the optimal configuration for sirius running TPCH at different scale factors.
---

# SIRIUS Config Optimizer Skill

You are tuning sirius configurations for optimal performance on TPC-H datasets at various scale factors. To find the best configuration, will change configuration parameters listed below, use `optimization advisor` to evaluate the performance and maximize gpu utilizations in different operaotions.

## Configuration Parameters
- `sirius.executor.pipeline.num_threads`: [1-10] change number of threads(streams) used for gpu tasks in pipeline executor.
- `sirius.executor.duckdb_scan.num_threads`: [1-10] change number of threads used for duckdb scan operator.
- `sirius.executor.duckdb_scan.cache`: ["none", "parquet", "table_gpu", "table_host"] cache mode, cache noting, cache parquet rowgorups in pinned memory, cache scanned table in pgu, and cache scanned table in pinned memory, respectively.
- `sirius.operator_params.scan_task_batch_size`: [500MB - 5GB] the batch size in bytes for scan tasks, which determines how much data is processed in each scan task.
- `sirius.operator_params.concat_batch_bytes`: [500MB - 5GB] the batch size in bytes for concatenation tasks, which determines how much data is processed in each concatenation task.
- `sirius.operator_params.hash_partition_bytes`: [500MB - 5GB] the batch size in bytes for hash_partition_bytes tasks, which determines how much data is processed in each concatenation task.


## Workflow

1. Ask user for baseline configuration if not set by `$SIRIUS_CONFIG_FILE`.
2. Ask user for target TPC-H scale factor and dataset location.
    - `dataset manager` skill to generate TPC-H dataset at user's specified scale factor with optimized parquet files.
4. Use `optimization advisor` to evaluate the performance of the baseline configuration on the target dataset, and collect performance metrics (e.g. query latency, gpu utilization, memory usage).
5. Use a systematic approach (e.g. grid search, random search, or Bayesian optimization) to explore the configuration space by changing the parameters listed above. For each configuration:
    - Update the sirius configuration file with the new parameters.
    - Run the same TPC-H queries and collect performance metrics.
    - Compare the performance metrics with the baseline configuration and previous configurations to identify improvements or regressions.
    - Backup the configuration file and results for each configuration tested for future reference.

To change configs use `scripts/patch_config.py`.

```bash
cd scripts
pixi run python patch_config.py sirius.cfg \\
        --opt sirius.executor.pipeline.num_threads=4 \\
        --opt sirius.executor.duckdb_scan.cache=true \\
        --opt sirius.operator_params.scan_task_batch_size=536870912
```

Arguments:
- `confi_file_path` — Path to the sirius configuration file to be updated.
- `--opt` — key-value pair for the config


6. After testing a sufficient number of configurations, analyze the results to identify the optimal configuration that provides the best performance for the target TPC-H dataset.
7. Report the optimal configuration and its performance metrics to the user, along with any insights.


## Before Running

- **Ask the user** if they want to rebuild the code using there are upstream changes.
- Use the pixi environment: `pixi run make` to build the code
- Use pixi for package management

## Output

Create optimization report with the following information:
- Baseline configuration and performance metrics
- Each configuration tested, the parameters changed, and the resulting performance metrics
- Optimal configuration in `optimal_sirius.cfg` file and its performance metrics report in `optimal_config_report.txt`
- Insights and recommendations for future tuning efforts based on the observed performance trends.
