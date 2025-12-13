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
import os
import sys
import math
from duckdb_queries import QUERIES, get_query_count

# Queries that use gpu_processing wrapper
SIRIUS_QUERIES = [
    'call gpu_processing("' + q.rstrip(';') + '");' for q in QUERIES
]

def compare_values(val1, val2, tolerance=1e-6):
    """Compare two values with tolerance for floating point numbers."""
    if val1 is None and val2 is None:
        return True
    if val1 is None or val2 is None:
        return False
    if isinstance(val1, float) and isinstance(val2, float):
        if math.isnan(val1) and math.isnan(val2):
            return True
        if abs(val1) < 1e-10 and abs(val2) < 1e-10:
            return True
        if abs(val1) > 1e-10:
            return abs(val1 - val2) / abs(val1) < tolerance
        return abs(val1 - val2) < tolerance
    return val1 == val2

def compare_rows(row1, row2, tolerance=1e-6):
    """Compare two rows."""
    if len(row1) != len(row2):
        return False
    for v1, v2 in zip(row1, row2):
        if not compare_values(v1, v2, tolerance):
            return False
    return True

def compare_results(result1, result2, ordered=True, tolerance=1e-6):
    """Compare two result sets."""
    if len(result1) != len(result2):
        return False, f"Row count mismatch: {len(result1)} vs {len(result2)}"

    if ordered:
        for i, (r1, r2) in enumerate(zip(result1, result2)):
            if not compare_rows(r1, r2, tolerance):
                return False, f"Row {i} mismatch: {r1} vs {r2}"
    else:
        # For unordered comparison, sort both result sets
        # Convert to tuples for sorting
        try:
            sorted1 = sorted([tuple(r) for r in result1])
            sorted2 = sorted([tuple(r) for r in result2])
            for i, (r1, r2) in enumerate(zip(sorted1, sorted2)):
                if not compare_rows(r1, r2, tolerance):
                    return False, f"Sorted row {i} mismatch: {r1} vs {r2}"
        except TypeError:
            # If sorting fails, fall back to ordered comparison
            for i, (r1, r2) in enumerate(zip(result1, result2)):
                if not compare_rows(r1, r2, tolerance):
                    return False, f"Row {i} mismatch: {r1} vs {r2}"

    return True, "Results match"

def run_correctness_test(con, query_indices=None, verbose=True):
    """Run correctness tests for specified queries.

    Args:
        con: DuckDB connection with Sirius extension loaded
        query_indices: List of query indices to test, or None for all queries
        verbose: Whether to print detailed output

    Returns:
        Tuple of (passed_count, failed_count, error_count, results_dict)
    """
    if query_indices is None:
        query_indices = range(get_query_count())

    passed = 0
    failed = 0
    errors = 0
    results = {}

    # Queries that have ORDER BY (results should be ordered)
    ordered_queries = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42}

    for i in query_indices:
        query_name = f"Q{i}"
        if verbose:
            print(f"Testing {query_name}...", end=" ")

        try:
            # Run DuckDB query
            duckdb_result = con.execute(QUERIES[i]).fetchall()

            # Run Sirius query
            sirius_result = con.execute(SIRIUS_QUERIES[i]).fetchall()

            # Compare results
            is_ordered = i in ordered_queries
            match, message = compare_results(duckdb_result, sirius_result, ordered=is_ordered)

            if match:
                passed += 1
                results[query_name] = {"status": "PASSED", "message": message}
                if verbose:
                    print("PASSED")
            else:
                failed += 1
                results[query_name] = {"status": "FAILED", "message": message}
                if verbose:
                    print(f"FAILED - {message}")
                    if len(duckdb_result) <= 5:
                        print(f"  DuckDB result: {duckdb_result}")
                        print(f"  Sirius result: {sirius_result}")

        except Exception as e:
            errors += 1
            results[query_name] = {"status": "ERROR", "message": str(e)}
            if verbose:
                print(f"ERROR - {e}")

    return passed, failed, errors, results

if __name__ == "__main__":
    con = duckdb.connect('clickbench_test.duckdb', config={"allow_unsigned_extensions": "true"})
    extension_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'build/release/extension/sirius/sirius.duckdb_extension')
    con.execute("load '{}'".format(extension_path))

    gpu_buffer_size = sys.argv[1] if len(sys.argv) > 1 else "2"

    print("Initializing GPU buffer...")
    command = f"call gpu_buffer_init('{gpu_buffer_size} GB', '{gpu_buffer_size} GB')"
    con.execute(command)

    # Parse query indices if provided
    query_indices = None
    if len(sys.argv) > 2:
        query_indices = [int(x) for x in sys.argv[2].split(',')]

    print("\n" + "=" * 60)
    print("ClickBench Correctness Test: DuckDB vs Sirius")
    print("=" * 60 + "\n")

    passed, failed, errors, results = run_correctness_test(con, query_indices)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Errors:  {errors}")
    print(f"Total:   {passed + failed + errors}")

    if failed > 0 or errors > 0:
        print("\nFailed/Error queries:")
        for query_name, result in results.items():
            if result["status"] != "PASSED":
                print(f"  {query_name}: {result['status']} - {result['message']}")

    con.close()

    # Exit with error code if any tests failed
    sys.exit(1 if (failed > 0 or errors > 0) else 0)
