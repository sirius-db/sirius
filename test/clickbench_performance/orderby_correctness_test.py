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
ORDER BY Correctness Test

This script tests ORDER BY functionality in isolation, derived from ClickBench queries.
It strips away aggregations and complex operations to focus purely on sorting correctness.
"""

import duckdb
import os
import sys
import math

# ORDER BY test queries - simplified from ClickBench to focus on sorting logic
# Each entry: (name, query, description)
ORDERBY_QUERIES = [
    # ==========================================================================
    # Single column ORDER BY tests
    # ==========================================================================

    # Integer column sorting
    ("int_asc",
     "SELECT WatchID, RegionID FROM hits ORDER BY RegionID ASC LIMIT 100;",
     "Single integer column ASC"),

    ("int_desc",
     "SELECT WatchID, RegionID FROM hits ORDER BY RegionID DESC LIMIT 100;",
     "Single integer column DESC"),

    ("bigint_asc",
     "SELECT WatchID, UserID FROM hits ORDER BY UserID ASC LIMIT 100;",
     "Single bigint column ASC"),

    ("bigint_desc",
     "SELECT WatchID, UserID FROM hits ORDER BY UserID DESC LIMIT 100;",
     "Single bigint column DESC"),

    ("smallint_asc",
     "SELECT WatchID, AdvEngineID FROM hits ORDER BY AdvEngineID ASC LIMIT 100;",
     "Single smallint column ASC"),

    ("smallint_desc",
     "SELECT WatchID, AdvEngineID FROM hits ORDER BY AdvEngineID DESC LIMIT 100;",
     "Single smallint column DESC"),

    # String column sorting (from Q24, Q25)
    ("string_asc",
     "SELECT WatchID, SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY SearchPhrase ASC LIMIT 100;",
     "Single string column ASC"),

    ("string_desc",
     "SELECT WatchID, SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY SearchPhrase DESC LIMIT 100;",
     "Single string column DESC"),

    # Timestamp sorting (from Q23, Q24)
    ("timestamp_asc",
     "SELECT WatchID, EventTime FROM hits ORDER BY EventTime ASC LIMIT 100;",
     "Single timestamp column ASC"),

    ("timestamp_desc",
     "SELECT WatchID, EventTime FROM hits ORDER BY EventTime DESC LIMIT 100;",
     "Single timestamp column DESC"),

    # Date sorting
    ("date_asc",
     "SELECT WatchID, EventDate FROM hits ORDER BY EventDate ASC LIMIT 100;",
     "Single date column ASC"),

    ("date_desc",
     "SELECT WatchID, EventDate FROM hits ORDER BY EventDate DESC LIMIT 100;",
     "Single date column DESC"),

    # ==========================================================================
    # Multi-column ORDER BY tests (from Q26, Q11)
    # ==========================================================================

    ("multi_timestamp_string",
     "SELECT WatchID, EventTime, SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime, SearchPhrase LIMIT 100;",
     "Multi-column: timestamp + string (from Q26)"),

    ("multi_int_int",
     "SELECT WatchID, RegionID, CounterID FROM hits ORDER BY RegionID ASC, CounterID ASC LIMIT 100;",
     "Multi-column: int + int ASC"),

    ("multi_int_int_desc",
     "SELECT WatchID, RegionID, CounterID FROM hits ORDER BY RegionID DESC, CounterID DESC LIMIT 100;",
     "Multi-column: int + int DESC"),

    ("multi_int_asc_desc",
     "SELECT WatchID, RegionID, CounterID FROM hits ORDER BY RegionID ASC, CounterID DESC LIMIT 100;",
     "Multi-column: int ASC + int DESC"),

    ("multi_int_desc_asc",
     "SELECT WatchID, RegionID, CounterID FROM hits ORDER BY RegionID DESC, CounterID ASC LIMIT 100;",
     "Multi-column: int DESC + int ASC"),

    ("multi_string_int",
     "SELECT WatchID, MobilePhoneModel, MobilePhone FROM hits WHERE MobilePhoneModel <> '' ORDER BY MobilePhoneModel ASC, MobilePhone DESC LIMIT 100;",
     "Multi-column: string + int (from Q11)"),

    ("multi_3col",
     "SELECT WatchID, RegionID, CounterID, UserID FROM hits ORDER BY RegionID, CounterID, UserID LIMIT 100;",
     "Multi-column: 3 columns"),

    # ==========================================================================
    # ORDER BY with LIMIT and OFFSET (from Q38, Q39, Q40, Q41, Q42)
    # ==========================================================================

    ("limit_small",
     "SELECT WatchID, RegionID FROM hits ORDER BY RegionID DESC LIMIT 10;",
     "Small LIMIT"),

    ("limit_medium",
     "SELECT WatchID, RegionID FROM hits ORDER BY RegionID DESC LIMIT 100;",
     "Medium LIMIT"),

    ("limit_large",
     "SELECT WatchID, RegionID FROM hits ORDER BY RegionID DESC LIMIT 1000;",
     "Large LIMIT"),

    ("offset_small",
     "SELECT WatchID, RegionID FROM hits ORDER BY RegionID DESC LIMIT 10 OFFSET 100;",
     "OFFSET 100 (from Q40)"),

    ("offset_medium",
     "SELECT WatchID, RegionID FROM hits ORDER BY RegionID DESC LIMIT 10 OFFSET 1000;",
     "OFFSET 1000 (from Q38, Q39)"),

    ("offset_large",
     "SELECT WatchID, RegionID FROM hits ORDER BY RegionID DESC LIMIT 10 OFFSET 10000;",
     "OFFSET 10000 (from Q41)"),

    # ==========================================================================
    # ORDER BY with WHERE clause (from Q23, Q24, Q25)
    # ==========================================================================

    ("where_like",
     "SELECT WatchID, URL, EventTime FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 100;",
     "WHERE LIKE + ORDER BY (from Q23)"),

    ("where_neq_string",
     "SELECT WatchID, SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY SearchPhrase LIMIT 100;",
     "WHERE <> '' + ORDER BY (from Q25)"),

    ("where_eq_int",
     "SELECT WatchID, CounterID, EventTime FROM hits WHERE CounterID = 62 ORDER BY EventTime LIMIT 100;",
     "WHERE = + ORDER BY"),

    ("where_range",
     "SELECT WatchID, EventDate, EventTime FROM hits WHERE EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' ORDER BY EventTime LIMIT 100;",
     "WHERE range + ORDER BY (from Q36)"),

    ("where_complex",
     "SELECT WatchID, EventDate, EventTime FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 ORDER BY EventTime LIMIT 100;",
     "Complex WHERE + ORDER BY (from Q36-Q42)"),

    # ==========================================================================
    # ORDER BY with expressions (from Q42)
    # ==========================================================================

    ("expr_date_trunc",
     "SELECT WatchID, DATE_TRUNC('minute', EventTime) AS M FROM hits ORDER BY M LIMIT 100;",
     "ORDER BY date_trunc expression (from Q42)"),

    ("expr_extract",
     "SELECT WatchID, extract(minute FROM EventTime) AS m FROM hits ORDER BY m LIMIT 100;",
     "ORDER BY extract expression (from Q18)"),

    ("expr_arithmetic",
     "SELECT WatchID, ClientIP, ClientIP - 1 AS c1 FROM hits ORDER BY c1 DESC LIMIT 100;",
     "ORDER BY arithmetic expression (from Q35)"),

    # ==========================================================================
    # ORDER BY with NULL handling
    # ==========================================================================

    ("nulls_string",
     "SELECT WatchID, Title FROM hits ORDER BY Title LIMIT 100;",
     "ORDER BY nullable string column"),

    ("nulls_string_desc",
     "SELECT WatchID, Title FROM hits ORDER BY Title DESC LIMIT 100;",
     "ORDER BY nullable string column DESC"),

    # ==========================================================================
    # Full table scans with ORDER BY (stability test)
    # ==========================================================================

    ("full_scan_topk",
     "SELECT WatchID, UserID FROM hits ORDER BY UserID DESC LIMIT 50;",
     "Full table scan with top-k"),

    ("full_scan_multi",
     "SELECT WatchID, RegionID, CounterID FROM hits ORDER BY RegionID, CounterID LIMIT 50;",
     "Full table scan with multi-column sort"),

    # ==========================================================================
    # Large result set ORDER BY
    # ==========================================================================

    ("large_result",
     "SELECT WatchID, RegionID FROM hits WHERE RegionID < 100 ORDER BY WatchID LIMIT 5000;",
     "Large result set sorting"),

    # ==========================================================================
    # ORDER BY on computed/aliased columns
    # ==========================================================================

    ("alias_simple",
     "SELECT WatchID, RegionID AS r FROM hits ORDER BY r LIMIT 100;",
     "ORDER BY aliased column"),

    ("alias_expr",
     "SELECT WatchID, ResolutionWidth + ResolutionHeight AS total FROM hits ORDER BY total DESC LIMIT 100;",
     "ORDER BY aliased expression"),
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


def compare_ordered_results(result1, result2, tolerance=1e-6):
    """Compare two ordered result sets (ORDER BY results must match exactly)."""
    if len(result1) != len(result2):
        return False, f"Row count mismatch: {len(result1)} vs {len(result2)}"

    for i, (r1, r2) in enumerate(zip(result1, result2)):
        if not compare_rows(r1, r2, tolerance):
            return False, f"Row {i} mismatch: {r1} vs {r2}"

    return True, "Results match"


def run_orderby_correctness_test(con, query_names=None, verbose=True):
    """Run ORDER BY correctness tests.

    Args:
        con: DuckDB connection with Sirius extension loaded
        query_names: List of query names to test, or None for all queries
        verbose: Whether to print detailed output

    Returns:
        Tuple of (passed_count, failed_count, error_count, results_dict)
    """
    passed = 0
    failed = 0
    errors = 0
    results = {}

    queries_to_test = ORDERBY_QUERIES
    if query_names:
        name_set = set(query_names)
        queries_to_test = [(n, q, d) for n, q, d in ORDERBY_QUERIES if n in name_set]

    for name, query, description in queries_to_test:
        if verbose:
            print(f"Testing {name}: {description}...", end=" ")

        try:
            # Run DuckDB query
            duckdb_result = con.execute(query).fetchall()

            # Run Sirius query
            sirius_query = f'call gpu_processing("{query.rstrip(";")}")'
            sirius_result = con.execute(sirius_query).fetchall()

            # Compare results (ORDER BY results must be ordered)
            match, message = compare_ordered_results(duckdb_result, sirius_result)

            if match:
                passed += 1
                results[name] = {"status": "PASSED", "message": message, "description": description}
                if verbose:
                    print("PASSED")
            else:
                failed += 1
                results[name] = {"status": "FAILED", "message": message, "description": description}
                if verbose:
                    print(f"FAILED - {message}")
                    if len(duckdb_result) <= 10:
                        print(f"  DuckDB result: {duckdb_result[:5]}...")
                        print(f"  Sirius result: {sirius_result[:5]}...")
                    else:
                        # Find first mismatch
                        for i, (r1, r2) in enumerate(zip(duckdb_result, sirius_result)):
                            if not compare_rows(r1, r2):
                                print(f"  First mismatch at row {i}:")
                                print(f"    DuckDB: {r1}")
                                print(f"    Sirius: {r2}")
                                break

        except Exception as e:
            errors += 1
            results[name] = {"status": "ERROR", "message": str(e), "description": description}
            if verbose:
                print(f"ERROR - {e}")

    return passed, failed, errors, results


def print_summary(passed, failed, errors, results):
    """Print test summary."""
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Errors:  {errors}")
    print(f"Total:   {passed + failed + errors}")

    if failed > 0 or errors > 0:
        print("\n" + "-" * 70)
        print("Failed/Error tests:")
        print("-" * 70)
        for name, result in results.items():
            if result["status"] != "PASSED":
                print(f"  [{result['status']}] {name}: {result['description']}")
                print(f"           {result['message']}")


def list_tests():
    """List all available tests."""
    print("Available ORDER BY tests:")
    print("=" * 70)
    for name, query, description in ORDERBY_QUERIES:
        print(f"  {name:25s} - {description}")
    print("=" * 70)
    print(f"Total: {len(ORDERBY_QUERIES)} tests")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ORDER BY Correctness Test for Sirius")
    parser.add_argument("--gpu-buffer", default="2", help="GPU buffer size in GB (default: 2)")
    parser.add_argument("--tests", help="Comma-separated list of test names to run")
    parser.add_argument("--list", action="store_true", help="List all available tests")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (less output)")
    parser.add_argument("--db", default="clickbench_test.duckdb", help="Database file path")

    args = parser.parse_args()

    if args.list:
        list_tests()
        sys.exit(0)

    # Connect to database
    db_path = args.db
    if not os.path.isabs(db_path):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_path)

    con = duckdb.connect(db_path, config={"allow_unsigned_extensions": "true"})

    # Load Sirius extension
    extension_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'build/release/extension/sirius/sirius.duckdb_extension')
    con.execute(f"load '{extension_path}'")

    # Initialize GPU buffer
    print("Initializing GPU buffer...")
    command = f"call gpu_buffer_init('{args.gpu_buffer} GB', '{args.gpu_buffer} GB')"
    con.execute(command)

    print("\n" + "=" * 70)
    print("ORDER BY Correctness Test: DuckDB vs Sirius")
    print("=" * 70 + "\n")

    # Parse test names if provided
    query_names = None
    if args.tests:
        query_names = [x.strip() for x in args.tests.split(',')]

    passed, failed, errors, results = run_orderby_correctness_test(
        con, query_names, verbose=not args.quiet)

    print_summary(passed, failed, errors, results)

    con.close()

    # Exit with error code if any tests failed
    sys.exit(1 if (failed > 0 or errors > 0) else 0)
