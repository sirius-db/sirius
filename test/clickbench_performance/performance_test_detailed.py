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

from sirius_queries import QUERY_FUNCS
from duckdb_queries import QUERIES

def run_duckdb_queries(con, warmup=False):
    times = []
    for i, query in enumerate(QUERIES):
        start = time.time()
        try:
            con.execute(query)
            elapsed = time.time() - start
            times.append(elapsed)
            if not warmup:
                print(f"  Q{i:2d}: {elapsed:.4f}s")
        except Exception as e:
            times.append(None)
            if not warmup:
                print(f"  Q{i:2d}: FAILED - {e}")
    return times

def run_sirius_queries(con, warmup=False):
    times = []
    for i, qfunc in enumerate(QUERY_FUNCS):
        start = time.time()
        try:
            qfunc(con)
            elapsed = time.time() - start
            times.append(elapsed)
            if not warmup:
                print(f"  Q{i:2d}: {elapsed:.4f}s")
        except Exception as e:
            times.append(None)
            if not warmup:
                print(f"  Q{i:2d}: FAILED - {e}")
    return times

if __name__ == "__main__":
    gpu_buffer_size = sys.argv[1] if len(sys.argv) > 1 else "2"

    db_path = 'clickbench_test.duckdb'
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found. Please run generate_test_data.py first.")
        sys.exit(1)

    print(f"Using database: {db_path}")
    con = duckdb.connect(db_path, config={"allow_unsigned_extensions": "true"})

    extension_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'build/release/extension/sirius/sirius.duckdb_extension')
    con.execute(f"load '{extension_path}'")

    print(f"Initializing GPU buffer ({gpu_buffer_size} GB)...")
    con.execute(f"call gpu_buffer_init('{gpu_buffer_size} GB', '{gpu_buffer_size} GB')")

    print("\n=== Warmup ===")
    run_sirius_queries(con, warmup=True)
    run_duckdb_queries(con, warmup=True)
    print("Warmup done\n")

    print("=== DuckDB (CPU) ===")
    duck_times = run_duckdb_queries(con)
    duck_valid = [t for t in duck_times if t is not None]
    duck_total = sum(duck_valid)
    print(f"  Total: {duck_total:.4f}s ({len(duck_valid)}/{len(duck_times)} queries succeeded)\n")

    print("=== Sirius (GPU) ===")
    sirius_times = run_sirius_queries(con)
    sirius_valid = [t for t in sirius_times if t is not None]
    sirius_total = sum(sirius_valid)
    print(f"  Total: {sirius_total:.4f}s ({len(sirius_valid)}/{len(sirius_times)} queries succeeded)\n")

    print("=== Speedup per Query ===")
    for i in range(len(QUERIES)):
        d, s = duck_times[i], sirius_times[i]
        if d is not None and s is not None and s > 0:
            speedup = d / s
            print(f"  Q{i:2d}: {speedup:6.2f}x  (DuckDB: {d:.4f}s, Sirius: {s:.4f}s)")
        elif d is not None and s is None:
            print(f"  Q{i:2d}: Sirius FAILED")
        elif d is None and s is not None:
            print(f"  Q{i:2d}: DuckDB FAILED")
        else:
            print(f"  Q{i:2d}: Both FAILED")

    if sirius_total > 0:
        print(f"\n  Overall Speedup: {duck_total/sirius_total:.2f}x")

    con.close()
