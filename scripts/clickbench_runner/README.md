# ClickBench Runner

This directory contains the files needed to run the ClickBench benchmark on Sirius. At a high level, the main file is `runner.py` which has the following arguments: 
```
usage: runner.py [-h] --duckdb_db_path DUCKDB_DB_PATH --result_save_path RESULT_SAVE_PATH [--reload_dataset] [--dataset_path DATASET_PATH] [--buffer_size BUFFER_SIZE] [--num_warm_runs NUM_WARM_RUNS]

Script to run clickbench queries against a DuckDB database.

options:
  -h, --help            show this help message and exit
  --duckdb_db_path DUCKDB_DB_PATH
                        Path to the DuckDB database file.
  --result_save_path RESULT_SAVE_PATH
                        Path to save the result
  --reload_dataset      If specified, reload the dataset into the DuckDB database.
  --dataset_path DATASET_PATH
                        Path to the dataset file to load into DuckDB. This must be specified if --reload_dataset is used or if the table doesn't exist.
  --buffer_size BUFFER_SIZE
                        Size of the buffer to allocate on the GPU
  --num_warm_runs NUM_WARM_RUNS
                        Number of warm runs of the query to perform
```

**Before running the scripts, please make sure that SIRIUS_HOME_PATH is set properly**

## Example

Thus to load the clickbench dataset and to run the queries with 3 warm runs you can run the command:
```
python3 runner.py --duckdb_db_path ~/datasets/clickbench/complete_clickbench.duckdb  --result_save_path "gnode1_clickbench.csv" --buffer_size "15 GB" --num_warm_runs 3 --reload_dataset --dataset_path /home/devesh/datasets/clickbench/hits.csv 
```

It will save the average warmup times in the specified result file. Note that a time of -1.0 indicates that sirius failed to run the query and instead used Duckdb to actually execute the query. 