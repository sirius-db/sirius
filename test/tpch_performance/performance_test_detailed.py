import duckdb
import time
import os
import sys

from sirius_queries import q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19, q20, q21, q22
from duckdb_queries import q1 as dq1, q2 as dq2, q3 as dq3, q4 as dq4, q5 as dq5, q6 as dq6, q7 as dq7, q8 as dq8, q9 as dq9, q10 as dq10, q11 as dq11, q12 as dq12, q13 as dq13, q14 as dq14, q15 as dq15, q16 as dq16, q17 as dq17, q18 as dq18, q19 as dq19, q20 as dq20, q21 as dq21, q22 as dq22

sirius_queries = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19, q20, q21, q22]
duckdb_queries = [dq1, dq2, dq3, dq4, dq5, dq6, dq7, dq8, dq9, dq10, dq11, dq12, dq13, dq14, dq15, dq16, dq17, dq18, dq19, dq20, dq21, dq22]

def run_queries(con, queries, name, warmup=False):
    times = []
    for i, q in enumerate(queries):
        start = time.time()
        try:
            q(con)
            elapsed = time.time() - start
            times.append(elapsed)
            if not warmup:
                print(f"  Q{i+1:2d}: {elapsed:.4f}s")
        except Exception as e:
            times.append(None)
            if not warmup:
                print(f"  Q{i+1:2d}: FAILED - {e}")
    return times

if __name__ == "__main__":
    SF = sys.argv[1] if len(sys.argv) > 1 else "1"
    
    # 使用正确的数据库路径
    db_path = f'test_datasets/tpch_sf{SF}.duckdb'
    if not os.path.exists(db_path):
        db_path = 'performance_test.duckdb'
    
    print(f"Using database: {db_path}")
    con = duckdb.connect(db_path, config={"allow_unsigned_extensions": "true"})
    
    extension_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'build/release/extension/sirius/sirius.duckdb_extension')
    con.execute(f"load '{extension_path}'")
    
    print(f"Initializing GPU buffer ({SF} GB)...")
    con.execute(f"call gpu_buffer_init('{SF} GB', '{SF} GB')")
    
    print("\n=== Warmup ===")
    run_queries(con, sirius_queries, "Sirius", warmup=True)
    run_queries(con, duckdb_queries, "DuckDB", warmup=True)
    print("Warmup done\n")
    
    print("=== DuckDB (CPU) ===")
    duck_times = run_queries(con, duckdb_queries, "DuckDB")
    duck_total = sum(t for t in duck_times if t)
    print(f"  Total: {duck_total:.4f}s\n")
    
    print("=== Sirius (GPU) ===")
    sirius_times = run_queries(con, sirius_queries, "Sirius")
    sirius_total = sum(t for t in sirius_times if t)
    print(f"  Total: {sirius_total:.4f}s\n")
    
    print("=== Speedup ===")
    for i in range(22):
        d, s = duck_times[i], sirius_times[i]
        if d and s and s > 0:
            print(f"  Q{i+1:2d}: {d/s:.2f}x")
    print(f"  Overall: {duck_total/sirius_total:.2f}x")
    
    con.close()
