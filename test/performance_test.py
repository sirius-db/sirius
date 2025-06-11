import duckdb
import time
import os
import sys
from sirius_queries import run_sirius
from duckdb_queries import run_duckdb

if __name__ == "__main__":
  con = duckdb.connect('performance_test.duckdb', config={"allow_unsigned_extensions": "true"})
  # con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
  con.execute("load '/mnt/nvme/sirius/build/release/extension/sirius/sirius.duckdb_extension'")
  
  SF = sys.argv[1]
  
  print("Initializing GPU buffer...")
  command = f"call gpu_buffer_init('{SF} GB', '{SF} GB')"
  con.execute(command)
  
  print("Initializing Sirius...")
  run_sirius(con, warmup=True)

  run_duckdb(con, warmup=True)
  print("Executing DuckDB queries...")
  start_time = time.time()
  run_duckdb(con, warmup=False)
  end_time = time.time()
  print("DuckDB Execution time:", end_time - start_time, "seconds")

  print("Executing Sirius queries...")
  start_time = time.time()
  run_sirius(con, warmup=False)
  end_time = time.time()
  print("Sirius Execution time:", end_time - start_time, "seconds")
  con.close()