import argparse
import os
import duckdb
import io
import sys
import subprocess
import time
import pandas as pd
import tempfile
import numpy as np

CLICKBENCH_TABLE_NAME = "hits"
SIRIUS_FAILURE_MESSAGE = "Error in GPUExecuteQuery"
RUN_TIME_LINE = "Run Time (s):"

current_path = os.path.abspath(__file__)
sirius_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
default_sirius_exec_path = os.path.join(sirius_path, "build/release/duckdb")
default_duckdb_db_path = os.path.join(sirius_path, "clickbench.duckdb")
default_result_save_path = os.path.join(os.path.dirname(current_path), "result.csv")
default_output_save_path = os.path.join(os.path.dirname(current_path), "output.txt")

def load_args():
    parser = argparse.ArgumentParser(description="Script to run clickbench queries against a DuckDB database.")
    parser.add_argument("--sirius_exec_path", type=str, default=default_sirius_exec_path, help="Path to the Sirius executable.")
    parser.add_argument("--duckdb_db_path", type=str, default=default_duckdb_db_path, help="Path to the DuckDB database file.")
    parser.add_argument("--result_save_path", type=str, default=default_result_save_path, help="Path to save the benchmark result.")
    parser.add_argument("--output_save_path", type=str, default=default_output_save_path, help="Path to save the output (query result).")
    parser.add_argument('--reload_dataset', action='store_true', help='If specified, reload the dataset into the DuckDB database.')
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file to load into DuckDB. This must be specified if --reload_dataset is used or if the table doesn't exist.")
    parser.add_argument("--caching_region_size", type=str, default= "19 GB", help="Size of the caching region.")
    parser.add_argument("--processing_region_size", type=str, default= "19 GB", help="Size of the processing region.")
    parser.add_argument("--num_warm_runs", type=int, default=2, help="Number of warm runs of the query to perform.")

    return parser.parse_args()

def run_duckdb_query(db_connection, query):
    # Temporarily update stdout/stderr to a string buffer
    old_stdout, old_stderr = sys.stdout, sys.stderr
    string_output = io.StringIO()
    sys.stdout = string_output
    sys.stderr = string_output

    # Run the query
    query = db_connection.sql(query)
    if query is not None: # This happens for queries that don't return results
        query.show()

    # Reset stdout/stderr
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    return string_output.getvalue()

def table_exists(db_connection, table_name):
    all_tables = run_duckdb_query(db_connection, "SHOW TABLES;")
    return table_name.lower() in all_tables.lower()

def load_dataset(db_connection, dataset_path):
    # First drop any existing tables
    curr_file_dir = os.path.dirname(os.path.abspath(__file__))
    run_duckdb_query(db_connection, f"DROP TABLE IF EXISTS {CLICKBENCH_TABLE_NAME};")

    # Now create the table
    create_table_sql_path = os.path.join(curr_file_dir, "create.sql")
    with open(create_table_sql_path, 'r') as reader:
        create_table_query = reader.read().strip()
    run_duckdb_query(db_connection, create_table_query)

    # Finally load the dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}. Please provide a valid path.")
    load_query = f"COPY {CLICKBENCH_TABLE_NAME} FROM '{dataset_path}' (QUOTE '')"
    run_duckdb_query(db_connection, load_query)

def benchmark_query(args, query_to_run, query_label):
    # Create the complete command to run to benchmark this query
    query_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".query")
    query_temp_file.close()
    query_temp_file_path = query_temp_file.name
    with open(query_temp_file_path, 'w+') as writer:
        writer.write(".timer on\n")
        writer.write(f"call gpu_buffer_init('{args.caching_region_size}', '{args.processing_region_size}');\n")
        for _ in range(args.num_warm_runs + 1):
            writer.write(f'call gpu_processing("{query_to_run}");\n')

    # Redirect the result to a temporary file
    result_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".result")
    result_temp_file.close()
    result_temp_file_path = result_temp_file.name

    # Now run the command
    command_to_run_str = f"{args.sirius_exec_path} {args.duckdb_db_path} -f {query_temp_file_path} > {result_temp_file_path} 2>&1"
    subprocess.run(command_to_run_str, shell=True)

    # Now read the result
    with open(result_temp_file_path, 'r') as reader, open(args.output_save_path, 'a') as writer:
        command_result = reader.read()
        writer.write(f"Output of {query_label}:\n")
        writer.write(command_result)
        writer.write("\n")

    # Clean up the temporary files
    os.remove(query_temp_file_path)
    os.remove(result_temp_file_path)

    # Now get the warm run time
    result_lines = command_result.split("\n")
    query_run_times = []
    for curr_line in result_lines:
        curr_line = curr_line.strip()
        if RUN_TIME_LINE in curr_line:
            run_time_line_parts = curr_line.split(" ")
            query_run_times.append(float(run_time_line_parts[4]))

    # Return the average time
    return SIRIUS_FAILURE_MESSAGE in command_result, -1 if len(query_run_times) < 3 else min(query_run_times[2:])

def main():
    args = load_args()

    # First ensure we have built the latest version of sirius
    starting_dir = os.getcwd()
    if not os.path.exists(args.sirius_exec_path):
        print("Building the latest version of sirius...")
        os.chdir(os.path.dirname(args.sirius_dir_path))
        subprocess.run("make -j$(nproc)", shell=True, check=True, capture_output=True)
        os.chdir(starting_dir)

    # Create a connection to the duckdb database
    db_initialize_connection = duckdb.connect(args.duckdb_db_path)

    # See if we need to reload the dataset
    if not table_exists(db_initialize_connection, CLICKBENCH_TABLE_NAME) or args.reload_dataset:
        print("Loading clickbench dataset into DuckDB")
        load_dataset(db_initialize_connection, args.dataset_path)
    db_initialize_connection.close()

    # Now load the queries to run
    curr_file_dir = os.path.dirname(os.path.abspath(__file__))
    queries_file_path = os.path.join(curr_file_dir, "queries.sql")
    with open(queries_file_path, 'r') as reader:
        queries = reader.read().split("\n")

    query_result = []
    if os.path.exists(args.output_save_path):
        os.remove(args.output_save_path)
    for query_idx, query in enumerate(queries):
        # Run the query
        print("Benchmarking query", query_idx + 1)
        fallback, query_run_time = benchmark_query(args, query, f"query{query_idx + 1}")
        query_result.append({
            "query" : query_idx + 1,
            "query_time_sec" : query_run_time,
            "fallback": fallback
        })

    # Save the results to the specified file
    result_df = pd.DataFrame(query_result)
    result_df.to_csv(args.result_save_path, index=False)
    print(f"Benchmark times saved to {args.result_save_path}.")
    print(f"Benchmark output saved to {args.output_save_path}.")

if __name__ == "__main__":
    main()