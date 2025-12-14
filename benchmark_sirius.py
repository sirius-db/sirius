#!/usr/bin/env python3
import subprocess
import re
import statistics
import json
from pathlib import Path
import time

# Configuration
GRAPH_SIZES = ['1k', '10k', '50k', '100k', '500k']
NUM_RUNS = 5
SIRIUS_DIR = '/home/andy/sirius'
DOCKER_IMAGE = 'siriusdb/sirius_dependencies_x86_64:stable'

results = {}

def extract_time_from_output(output):
    """Extract execution time from DuckDB .timer output"""
    # Look for "Run Time (s): real X.XXX"
    patterns = [
        r'Run Time \(s\):\s*real\s+([\d.]+)',   # "Run Time (s): real 0.002"
        r'Run Time[^:]*:\s*([\d.]+)\s*s',       # "Run Time: 0.123s"
        r'Run Time[^:]*:\s*([\d.]+)',           # "Run Time (s): 0.123"
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return float(match.group(1))

    return None

def create_sirius_sql_script(query_type):
    """Create SQL script for Sirius with .timer"""

    # Setup + warmup (no timing)
    sql_script = """
CALL gpu_buffer_init('8GB', '8GB');
CALL gpu_processing("SELECT * FROM Person");
CALL gpu_processing("SELECT * FROM Person_knows_Person");

-- Warmup
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p:person WHERE p.id = 14)-[k:Person_knows_Person]->(p2:person)
        COLUMNS (p2.id)
    )
");
"""

    # Actual queries with .timer
    queries = {
        'direct_neighbors': """
.timer on
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p:person WHERE p.id = 14)-[k:Person_knows_Person]->(p2:person)
        COLUMNS (p2.id, distance, predecessor)
    )
");
.timer off
""",
        'bfs': """
.timer on
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p:person WHERE p.id = 14)-[k:Person_knows_Person]->*(p2:person)
        COLUMNS (p2.id, distance, predecessor)
    )
");
.timer off
""",
        'any_shortest': """
.timer on
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH p = ANY SHORTEST (p:person WHERE p.id = 14)-[k:Person_knows_Person]->*(p2:person)
        COLUMNS (p2.id, distance, predecessor)
    )
");
.timer off
""",
        'weighted_shortest': """
.timer on
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH SHORTEST (p:person WHERE p.id = 14)-[k:Person_knows_Person]->*(p2:person)
        COLUMNS (p2.id, distance, predecessor)
    )
");
.timer off
"""
    }

    sql_script += queries.get(query_type, "")
    return sql_script

def run_sirius_query(db_file, query_type):
    """Run Sirius query in Docker and return execution time"""

    sql_script = create_sirius_sql_script(query_type)

    # Write script file
    script_file = f'{SIRIUS_DIR}/benchmark_{query_type}.sql'
    with open(script_file, 'w') as f:
        f.write(sql_script)

    # Docker command - pipe SQL file into duckdb
    docker_cmd = [
        'docker', 'run', '--rm', '--gpus', 'all',
        '-v', f'{SIRIUS_DIR}:/workspace',
        DOCKER_IMAGE,
        'bash', '-c',
        f'cd /workspace && ./build/release/duckdb {db_file} < benchmark_{query_type}.sql'
    ]

    try:
        result = subprocess.run(docker_cmd, capture_output=True, text=True)

        # Combine stdout and stderr
        full_output = result.stdout + "\n" + result.stderr

        # Debug: save full output if no time found
        time_val = extract_time_from_output(full_output)

        if time_val is None:
            # Save output for debugging
            debug_file = f'{SIRIUS_DIR}/debug_output_{query_type}.txt'
            with open(debug_file, 'w') as f:
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)
            print(f"\n      No timing found. Output saved to {debug_file}")
            print(f"      Last 200 chars of output: ...{full_output[-200:]}")

        Path(script_file).unlink(missing_ok=True)
        return time_val

    except Exception as e:
        print(f"\n      Exception: {e}")
        Path(script_file).unlink(missing_ok=True)
        return None

# Main benchmark loop
print("=" * 80)
print("SIRIUS BENCHMARK - ALL DATASETS")
print("=" * 80)
print(f"Sirius directory: {SIRIUS_DIR}")
print(f"Docker image: {DOCKER_IMAGE}")
print(f"Runs per query: {NUM_RUNS}")
print(f"Datasets: {', '.join(GRAPH_SIZES)}")
print("=" * 80)

query_types = ['direct_neighbors', 'bfs', 'any_shortest', 'weighted_shortest']

for size in GRAPH_SIZES:
    db_file = f'snb_{size}.duckdb'
    db_file_full = f'{SIRIUS_DIR}/{db_file}'

    if not Path(db_file_full).exists():
        print(f"\n⚠️  Skipping {size} - file not found: {db_file_full}")
        continue

    print(f"\n{'='*80}")
    print(f"Testing graph size: {size.upper()}")
    print(f"Database: {db_file}")
    print(f"{'='*80}")

    if size not in results:
        results[size] = {}

    for query_type in query_types:
        print(f"\n  Query: {query_type}")

        sirius_times = []
        for run in range(NUM_RUNS):
            print(f"    Run {run+1}/{NUM_RUNS}...", end=' ', flush=True)
            start_time = time.time()
            time_val = run_sirius_query(db_file, query_type)
            elapsed = time.time() - start_time

            if time_val:
                sirius_times.append(time_val)
                print(f"{time_val:.4f}s (wall: {elapsed:.1f}s) ✓")
            else:
                print(f"FAILED (wall: {elapsed:.1f}s) ✗")

            time.sleep(2)

        if sirius_times:
            median_time = statistics.median(sirius_times)
            min_time = min(sirius_times)
            max_time = max(sirius_times)

            results[size][query_type] = {
                'median': median_time,
                'min': min_time,
                'max': max_time,
                'all_times': sirius_times,
                'num_successful': len(sirius_times)
            }

            print(f"    ✓ Median: {median_time:.4f}s, Min: {min_time:.4f}s, Max: {max_time:.4f}s")
        else:
            print(f"    ✗ All runs failed - check debug_output_*.txt files")

# Save results
output_file = f'{SIRIUS_DIR}/sirius_benchmark_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*80)
print("RESULTS SAVED")
print("="*80)
print(f"✓ JSON: {output_file}")

# Summary table
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

print("\n✓ Benchmark complete!")