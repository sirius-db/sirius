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
from sirius_queries import run_sirius
from duckdb_queries import run_duckdb

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

    print("Warming up Sirius...")
    run_sirius(con, warmup=True)

    print("Warming up DuckDB...")
    run_duckdb(con, warmup=True)

    print("\nExecuting DuckDB queries...")
    start_time = time.time()
    run_duckdb(con, warmup=False)
    duckdb_time = time.time() - start_time
    print(f"DuckDB Total Execution time: {duckdb_time:.3f} seconds")

    print("\nExecuting Sirius queries...")
    start_time = time.time()
    run_sirius(con, warmup=False)
    sirius_time = time.time() - start_time
    print(f"Sirius Total Execution time: {sirius_time:.3f} seconds")

    print("\n" + "=" * 50)
    print("Performance Summary")
    print("=" * 50)
    print(f"DuckDB Total Time:  {duckdb_time:.3f} seconds")
    print(f"Sirius Total Time:  {sirius_time:.3f} seconds")
    print(f"Speedup:            {duckdb_time / sirius_time:.2f}x")

    con.close()
