# =============================================================================
# Copyright 2025, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

import duckdb
import time
import os
import sys
from queries import QUERIES


def _log(msg, warmup=False):
    """Helper function to print messages unless in warmup mode."""
    if not warmup:
        print(msg)


def _wrap_gpu_processing(sql):
    """Wrap SQL in gpu_execution call."""
    # Escape quotes in the SQL
    escaped_sql = sql.replace('"', '\\"')
    return f'call gpu_processing("{escaped_sql}");'


def _verify_results(duckdb_rows, sirius_rows, query_name):
    """Verify that DuckDB and Sirius results match."""
    try:
        # Check row counts
        if len(duckdb_rows) != len(sirius_rows):
            print(
                f"❌ {query_name}: Row count mismatch - DuckDB: {len(duckdb_rows)}, Sirius: {len(sirius_rows)}"
            )
            return False

        # Sort both for comparison (in case order differs slightly)
        duckdb_rows_sorted = sorted(duckdb_rows, key=lambda x: str(x))
        sirius_rows_sorted = sorted(sirius_rows, key=lambda x: str(x))

        # Compare rows
        for i, (duck_row, sirius_row) in enumerate(
            zip(duckdb_rows_sorted, sirius_rows_sorted)
        ):
            if duck_row != sirius_row:
                print(f"❌ {query_name}: Row {i} mismatch")
                print(f"   DuckDB:  {duck_row}")
                print(f"   Sirius:  {sirius_row}")
                return False
            if duck_row == 0:
                print(f"❌ {query_name}: Results are empty")
                return False

        print(f"✓ {query_name}: Results match ({len(duckdb_rows)} rows)")
        return True
    except Exception as e:
        print(f"❌ {query_name}: Error during verification - {e}")
        return False


def execute_query(con, query_num, use_gpu=False, warmup=False, verify_mode=False):
    """
    Execute a single TPC-H query.

    Args:
        con: Database connection
        query_num: Query number (1-22)
        use_gpu: If True, wrap query with gpu_processing()
        warmup: If True, suppress output
        verify_mode: If True, return result for verification (don't consume it)

    Returns:
        Result object if verify_mode=True, None otherwise
    """
    query_name = f"q{query_num}"
    if query_name not in QUERIES:
        raise ValueError(f"Query {query_name} not found")

    sql = QUERIES[query_name]
    if use_gpu:
        sql = _wrap_gpu_processing(sql)

    result = con.execute(sql)

    if verify_mode:
        return result
    else:
        # Consume the result
        result.fetchall()
        _log(f"Q{query_num} done", warmup)
        return None


def run_all(con, use_gpu=False, warmup=False):
    """
    Run all TPC-H queries.

    Args:
        con: Database connection
        use_gpu: If True, use gpu_processing() for all queries
        warmup: If True, suppress output
    """
    for i in range(1, 23):
        execute_query(con, i, use_gpu=use_gpu, warmup=warmup)


def verify_all(con):
    """
    Verify that Sirius (GPU) and DuckDB produce the same results for all queries.

    Args:
        con: Database connection
        warmup: If True, suppress output

    Returns:
        Dictionary mapping query numbers to verification status (True/False)
    """
    results = {}

    for i in range(1, 23):
        query_name = f"Q{i}"
        print(f"Verifying {query_name}...")

        try:
            # Run with DuckDB
            duckdb_result = execute_query(
                con, i, use_gpu=False, verify_mode=True
            ).fetchall()

            # Run with Sirius
            sirius_result = execute_query(
                con, i, use_gpu=True, verify_mode=True
            ).fetchall()

            # Verify results match
            match = _verify_results(duckdb_result, sirius_result, query_name)
            results[i] = match

        except Exception as e:
            print(f"❌ {query_name}: Exception - {e}")
            results[i] = False

    # Print summary
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Verification Summary: {passed}/{total} queries passed")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    con = duckdb.connect(
        "performance_test.duckdb", config={"allow_unsigned_extensions": "true"}
    )
    # con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    extension_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "build/release/extension/sirius/sirius.duckdb_extension",
    )
    con.execute("load '{}'".format(extension_path))

    SF = sys.argv[1]

    print("Initializing GPU buffer...")
    command = f"call gpu_buffer_init('{SF} GB', '{SF} GB')"
    con.execute(command)

    print("Initializing Sirius...")
    run_all(con, use_gpu=True, warmup=True)

    run_all(con, use_gpu=False, warmup=True)
    print("Executing DuckDB queries...")
    start_time = time.time()
    run_all(con, use_gpu=False, warmup=False)
    end_time = time.time()
    print("DuckDB Execution time:", end_time - start_time, "seconds")

    print("Executing Sirius queries...")
    start_time = time.time()
    run_all(con, use_gpu=True, warmup=False)
    end_time = time.time()
    print("Sirius Execution time:", end_time - start_time, "seconds")

    verify_all(con)

    con.close()
