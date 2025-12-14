#!/usr/bin/env python3
import subprocess
import re
import statistics
import json
from pathlib import Path
import time


# --------------------------------------------
# -- DuckPGQ/CPU Benchmark
# -- (run on the instance, not on the docker image)
# -- Install DuckDB v1.1.3:
# -- wget https://github.com/duckdb/duckdb/releases/download/v1.1.3/duckdb_cli-linux-amd64.zip
# -- unzip duckdb_cli-linux-amd64.zip
# -- sudo mv duckdb /usr/local/bin/
# -- To test performance:
# -- 1. copy test dataset (e.g., snb_*k.duckdb) to instance root
# -- 2. run python3 benchmark_duckpgq.py
# --------------------------------------------


# Configuration
GRAPH_SIZES = ['1k', '10k', '50k', '100k', '500k']
NUM_RUNS = 5
DUCKDB_BINARY = '/usr/local/bin/duckdb'
DB_DIR = '/home/andy'

results = {}

def extract_time_from_output(output):
    """Extract execution time from DuckDB .timer output"""
    patterns = [
        r'Run Time \(s\):\s*real\s+([\d.]+)',  # "Run Time (s): real 0.002"
        r'Run Time[^:]*:\s*([\d.]+)\s*s',       # "Run Time: 0.123s"
        r'Run Time[^:]*:\s*([\d.]+)',           # "Run Time (s): 0.123"
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return float(match.group(1))

    return None

def create_duckpgq_sql_script(db_file, query_type):
    """Create SQL script for DuckPGQ"""

    # Setup + warmup
    sql_script = f"""
ATTACH '{db_file}' AS snb;
USE snb;

INSTALL duckpgq FROM community;
LOAD duckpgq;

CREATE OR REPLACE PROPERTY GRAPH snb
  VERTEX TABLES (Person)
  EDGE TABLES (
    Person_knows_Person SOURCE KEY (source) REFERENCES Person (id)
                        DESTINATION KEY (destination) REFERENCES Person (id)
                        LABEL knows
  );

-- Warmup
FROM GRAPH_TABLE (snb
    MATCH (p:person WHERE p.id = 14)-[k:knows]->(p2:person)
    COLUMNS (p2.id)
);
"""

    # Actual queries
    queries = {
        'direct_neighbors': """
.timer on
FROM GRAPH_TABLE (snb
    MATCH (p:person WHERE p.id = 14)-[k:knows]->(p2:person)
    COLUMNS (p2.id)
);
.timer off
""",
        'bfs': """
.timer on
FROM GRAPH_TABLE (snb
  MATCH p = ANY SHORTEST (p1:person WHERE p1.id = 14)-[k:knows]->*(p2:person)
  COLUMNS (p2.id, path_length(p))
);
.timer off
""",
        'any_shortest': """
.timer on
FROM GRAPH_TABLE (snb
    MATCH p = ANY SHORTEST (p:person WHERE p.id = 14)-[k:knows]->*(p2:person)
    COLUMNS (p2.id, path_length(p))
);
.timer off
"""
    }

    sql_script += queries.get(query_type, "")
    return sql_script

def run_duckpgq_query(db_file, query_type):
    """Run DuckPGQ query and return execution time"""

    sql_script = create_duckpgq_sql_script(db_file, query_type)
    script_file = f'{DB_DIR}/benchmark_duckpgq_{query_type}.sql'

    with open(script_file, 'w') as f:
        f.write(sql_script)

    cmd = f'{DUCKDB_BINARY} < {script_file}'

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        full_output = result.stdout + "\n" + result.stderr
        time_val = extract_time_from_output(full_output)

        if time_val is None:
            # Debug output
            debug_file = f'{DB_DIR}/debug_duckpgq_{query_type}.txt'
            with open(debug_file, 'w') as f:
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)
            print(f"\n      No timing found. Output saved to {debug_file}")
            print(f"      Last 200 chars: ...{full_output[-200:]}")

        Path(script_file).unlink(missing_ok=True)
        return time_val

    except Exception as e:
        print(f"\n      Exception: {e}")
        Path(script_file).unlink(missing_ok=True)
        return None

# Main benchmark loop
print("=" * 80)
print("DUCKPGQ BENCHMARK - ALL DATASETS")
print("=" * 80)
print(f"DuckDB binary: {DUCKDB_BINARY}")
print(f"Database directory: {DB_DIR}")
print(f"Runs per query: {NUM_RUNS}")
print(f"Datasets: {', '.join(GRAPH_SIZES)}")
print("=" * 80)

query_types = ['direct_neighbors', 'bfs', 'any_shortest']

for size in GRAPH_SIZES:
    db_file = f'{DB_DIR}/snb_{size}.duckdb'

    if not Path(db_file).exists():
        print(f"\n⚠️  Skipping {size} - file not found: {db_file}")
        continue

    print(f"\n{'='*80}")
    print(f"Testing graph size: {size.upper()}")
    print(f"Database: {db_file}")
    print(f"{'='*80}")

    if size not in results:
        results[size] = {}

    for query_type in query_types:
        print(f"\n  Query: {query_type}")

        duckpgq_times = []
        for run in range(NUM_RUNS):
            print(f"    Run {run+1}/{NUM_RUNS}...", end=' ', flush=True)
            start_time = time.time()
            time_val = run_duckpgq_query(db_file, query_type)
            elapsed = time.time() - start_time

            if time_val:
                duckpgq_times.append(time_val)
                print(f"{time_val:.4f}s (wall: {elapsed:.1f}s) ✓")
            else:
                print(f"FAILED (wall: {elapsed:.1f}s) ✗")

            time.sleep(1)

        if duckpgq_times:
            median_time = statistics.median(duckpgq_times)
            min_time = min(duckpgq_times)
            max_time = max(duckpgq_times)

            results[size][query_type] = {
                'median': median_time,
                'min': min_time,
                'max': max_time,
                'all_times': duckpgq_times,
                'num_successful': len(duckpgq_times)
            }

            print(f"    ✓ Median: {median_time:.4f}s, Min: {min_time:.4f}s, Max: {max_time:.4f}s")
        else:
            print(f"    ✗ All runs failed - check debug files")

# Save results
output_file = f'{DB_DIR}/duckpgq_benchmark_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*80)
print("RESULTS SAVED")
print("="*80)
print(f"✓ JSON: {output_file}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"{'Size':<10} {'Query':<25} {'Median (s)':<15} {'Min (s)':<12} {'Max (s)':<12}")
print("-"*80)

for size in GRAPH_SIZES:
    if size not in results:
        continue
    for query_type in query_types:
        if query_type in results[size] and results[size][query_type].get('median'):
            data = results[size][query_type]
            print(f"{size:<10} {query_type:<25} {data['median']:<15.4f} {data['min']:<12.4f} {data['max']:<12.4f}")

print("\n✓ DuckPGQ benchmark complete!")