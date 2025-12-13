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

"""
Basic SELECT/FROM correctness test for Sirius.
Tests basic column selection without any aggregation, filtering, or ordering.
"""

import duckdb
import os
import sys
import math

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

def compare_results(result1, result2, tolerance=1e-6):
    """Compare two result sets (unordered)."""
    if len(result1) != len(result2):
        return False, f"Row count mismatch: {len(result1)} vs {len(result2)}"

    # Convert to comparable format and sort
    def to_tuple(row):
        return tuple(str(x) if x is not None else None for x in row)

    try:
        sorted1 = sorted([to_tuple(r) for r in result1])
        sorted2 = sorted([to_tuple(r) for r in result2])

        for i, (r1, r2) in enumerate(zip(sorted1, sorted2)):
            if r1 != r2:
                return False, f"Row {i} mismatch:\n  DuckDB: {r1}\n  Sirius: {r2}"
    except Exception as e:
        return False, f"Comparison error: {e}"

    return True, "Results match"


# Basic SELECT queries - testing different column types
SELECT_QUERIES = [
    # Single integer column
    ("SELECT CounterID FROM hits", "Single INT column"),

    # Single bigint column
    ("SELECT WatchID FROM hits", "Single BIGINT column"),

    # Single smallint column
    ("SELECT JavaEnable FROM hits", "Single SMALLINT column"),

    # Single text column
    ("SELECT Title FROM hits", "Single TEXT column"),

    # Single varchar column
    ("SELECT UserAgentMinor FROM hits", "Single VARCHAR column"),

    # Single char column
    ("SELECT HitColor FROM hits", "Single CHAR column"),

    # Single timestamp column
    ("SELECT EventTime FROM hits", "Single TIMESTAMP column"),

    # Single date column
    ("SELECT EventDate FROM hits", "Single DATE column"),

    # Multiple integer columns
    ("SELECT CounterID, RegionID, ClientIP FROM hits", "Multiple INT columns"),

    # Multiple bigint columns
    ("SELECT WatchID, UserID, FUniqID FROM hits", "Multiple BIGINT columns"),

    # Mixed integer types
    ("SELECT WatchID, CounterID, JavaEnable FROM hits", "Mixed integer types"),

    # Multiple text columns
    ("SELECT Title, URL, Referer FROM hits", "Multiple TEXT columns"),

    # Mixed types: int + text
    ("SELECT CounterID, Title FROM hits", "INT + TEXT"),

    # Mixed types: bigint + text
    ("SELECT WatchID, URL FROM hits", "BIGINT + TEXT"),

    # Mixed types: timestamp + int
    ("SELECT EventTime, CounterID FROM hits", "TIMESTAMP + INT"),

    # Mixed types: date + text
    ("SELECT EventDate, Title FROM hits", "DATE + TEXT"),

    # Many columns of same type
    ("SELECT CounterID, RegionID, ClientIP, RefererRegionID, URLRegionID FROM hits", "5 INT columns"),

    # Many columns of mixed types
    ("SELECT WatchID, CounterID, Title, EventTime, JavaEnable FROM hits", "5 mixed type columns"),

    # All numeric types together
    ("SELECT WatchID, CounterID, JavaEnable, ResolutionWidth, ParamPrice FROM hits", "All numeric types"),

    # Multiple string types
    ("SELECT Title, URL, Referer, UserAgentMinor, HitColor FROM hits", "Multiple string types"),

    # Temporal types together
    ("SELECT EventTime, EventDate, ClientEventTime, LocalEventTime FROM hits", "Multiple temporal types"),

    # Wide selection (many columns)
    ("SELECT WatchID, JavaEnable, Title, CounterID, ClientIP, RegionID, UserID, OS, UserAgent, URL FROM hits", "10 columns wide"),
]


def run_select_tests(con, verbose=True):
    """Run SELECT correctness tests.

    Args:
        con: DuckDB connection with Sirius extension loaded
        verbose: Whether to print detailed output

    Returns:
        Tuple of (passed_count, failed_count, error_count, results_dict)
    """
    passed = 0
    failed = 0
    errors = 0
    results = {}

    for i, (query, description) in enumerate(SELECT_QUERIES):
        test_name = f"T{i}: {description}"
        if verbose:
            print(f"Testing {test_name}...", end=" ")

        try:
            # Run DuckDB query
            duckdb_result = con.execute(query).fetchall()

            # Run Sirius query
            sirius_query = f'call gpu_processing("{query}");'
            sirius_result = con.execute(sirius_query).fetchall()

            # Compare results
            match, message = compare_results(duckdb_result, sirius_result)

            if match:
                passed += 1
                results[test_name] = {"status": "PASSED", "message": message, "rows": len(duckdb_result)}
                if verbose:
                    print(f"PASSED ({len(duckdb_result)} rows)")
            else:
                failed += 1
                results[test_name] = {"status": "FAILED", "message": message, "query": query}
                if verbose:
                    print(f"FAILED - {message}")
                    print(f"  Query: {query}")
                    if len(duckdb_result) <= 10:
                        print(f"  DuckDB ({len(duckdb_result)} rows): {duckdb_result[:5]}")
                        print(f"  Sirius ({len(sirius_result)} rows): {sirius_result[:5]}")

        except Exception as e:
            errors += 1
            results[test_name] = {"status": "ERROR", "message": str(e), "query": query}
            if verbose:
                print(f"ERROR - {e}")
                print(f"  Query: {query}")

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

    print("\n" + "=" * 70)
    print("Basic SELECT/FROM Correctness Test: DuckDB vs Sirius")
    print("=" * 70 + "\n")

    passed, failed, errors, results = run_select_tests(con)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Errors:  {errors}")
    print(f"Total:   {passed + failed + errors}")

    if failed > 0 or errors > 0:
        print("\nFailed/Error tests:")
        for test_name, result in results.items():
            if result["status"] != "PASSED":
                print(f"  {test_name}: {result['status']}")
                print(f"    {result['message']}")
                if 'query' in result:
                    print(f"    Query: {result['query']}")

    con.close()

    # Exit with error code if any tests failed
    sys.exit(1 if (failed > 0 or errors > 0) else 0)
