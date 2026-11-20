#!/usr/bin/env python3
"""DuckDB-only TPC-H Parquet cache benchmark using the Sirius query texts."""

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'test/tpch_performance'))
from performance_test import TPCH_TABLES, _build_views_sql, _resolve_parquet_files  # noqa: E402
from queries import queries_for_scale_factor  # noqa: E402


def log(message):
    print(f'[{datetime.now(timezone.utc).isoformat(timespec="seconds")}] {message}', flush=True)


def parquet_files(dataset):
    files = []
    for table in TPCH_TABLES:
        table_files = _resolve_parquet_files(str(dataset), table)
        if not table_files:
            raise RuntimeError(f'No Parquet files found for {table} in {dataset}')
        files.extend(table_files)
    return files


def sample_resident_bytes(files):
    samples = [next(path for path in files if '/lineitem/' in path),
               next(path for path in files if '/orders/' in path)]
    result = subprocess.run(
        ['fincore', '--bytes', '--raw', '--noheadings', '--output', 'RES', *samples],
        check=True, capture_output=True, text=True,
    )
    return sum(int(line.strip()) for line in result.stdout.splitlines())


def evict_os_page_cache(files):
    os.sync()
    for path in files:
        with open(path, 'rb') as stream:
            os.posix_fadvise(stream.fileno(), 0, os.fstat(stream.fileno()).st_size,
                             os.POSIX_FADV_DONTNEED)
    resident = sample_resident_bytes(files)
    log(f'POSIX_FADV_DONTNEED applied to {len(files)} Parquet files; '
        f'sampled lineitem/orders residency after eviction: {resident} bytes')
    if resident > 1024 * 1024:
        raise RuntimeError(f'OS page-cache eviction check failed: {resident} sampled bytes remain')
    return resident


def open_duckdb(dataset, temp_directory, memory_limit_gib):
    connection = duckdb.connect(':memory:', config={
        'autoload_known_extensions': 'false',
        'autoinstall_known_extensions': 'false',
    })
    for setting in ('enable_external_file_cache', 'cache_local_files',
                    'parquet_metadata_cache'):
        connection.execute(f'SET {setting} = true')
        assert connection.execute(f"SELECT current_setting('{setting}')").fetchone()[0] is True
    if memory_limit_gib is not None:
        connection.execute(f"SET memory_limit = '{memory_limit_gib}GiB'")
    connection.execute(f"SET temp_directory = '{temp_directory}'")
    loaded = connection.execute(
        "SELECT count(*) FROM duckdb_extensions() "
        "WHERE extension_name = 'sirius' AND loaded"
    ).fetchone()[0]
    if loaded:
        raise RuntimeError('Sirius is loaded in the DuckDB-only benchmark')
    for statement in _build_views_sql(str(dataset)).split(';\n'):
        statement = statement.strip()
        if statement:
            connection.execute(statement)
    return connection


def run_query(connection, sql):
    start = time.perf_counter()
    result = connection.execute(sql)
    elapsed = time.perf_counter() - start
    row_count = 0
    while True:
        batch = result.fetchmany(65536)
        if not batch:
            break
        row_count += len(batch)
    blocks, cache_bytes = connection.execute(
        'SELECT count(*), coalesce(sum(nr_bytes), 0) '
        'FROM duckdb_external_file_cache() WHERE loaded'
    ).fetchone()
    return elapsed, row_count, blocks, cache_bytes


def coherent_gpu_memory_mode():
    for line in Path('/proc/driver/nvidia/params').read_text().splitlines():
        if line.startswith('CoherentGPUMemoryMode:'):
            return line.split(':', 1)[1].strip().strip('"')
    raise RuntimeError('NVIDIA coherent GPU memory mode is unavailable')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale-factor', type=int,
                        choices=(100, 300, 1000, 3000, 10000), required=True)
    parser.add_argument('--profile', choices=('cold', 'hot', 'paired'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-limit-gib', type=int)
    parser.add_argument('--require-coherent-mode', choices=('driver', 'numa'))
    args = parser.parse_args()
    coherent_mode = coherent_gpu_memory_mode()
    if args.require_coherent_mode and coherent_mode != args.require_coherent_mode:
        raise SystemExit(f'Expected NVIDIA coherent GPU memory mode '
                         f'{args.require_coherent_mode}, found {coherent_mode}')
    dataset = Path(f'/mnt/datasets/tpch_parquet_sf{args.scale_factor}')
    if not dataset.is_dir():
        raise SystemExit(f'Missing dataset: {dataset}')
    files = parquet_files(dataset)
    sql_by_query = queries_for_scale_factor(str(args.scale_factor))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.scale_factor == 10000:
        temp_directory = Path('/mnt/datasets/duckdb-tmp-20261119') / output.parent.name / args.profile
    else:
        temp_directory = output / 'duckdb-tmp'
    temp_directory.mkdir(parents=True, exist_ok=True)
    info = {
        'duckdb_version': duckdb.__version__,
        'dataset': str(dataset),
        'parquet_file_count': len(files),
        'profile': args.profile,
        'iterations': 1 if args.profile == 'cold' else 2,
        'coherent_gpu_memory_mode': coherent_mode,
        'settings': {
            'enable_external_file_cache': True,
            'cache_local_files': True,
            'parquet_metadata_cache': True,
            'autoload_known_extensions': False,
            'autoinstall_known_extensions': False,
        },
        'sirius_loaded': False,
        'cold_connection_scope': 'fresh in-memory DuckDB database per query',
        'os_cache_eviction': 'sync + POSIX_FADV_DONTNEED on every dataset Parquet file',
        'memory_limit_gib': args.memory_limit_gib,
        'effective_memory_limit': None,
        'temp_directory': str(temp_directory),
        'timing': 'connection.execute(sql), excluding view setup and result fetch',
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    (output / 'metadata.json').write_text(json.dumps(info, indent=2) + '\n')
    log(f'Starting SF{args.scale_factor} {args.profile}: {len(files)} Parquet files, '
        f'DuckDB {duckdb.__version__}, Sirius unloaded')
    connection = None
    try:
        with (output / 'runtimes.csv').open('w', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['engine', 'query', 'iteration', 'runtime_s', 'result_rows',
                             'external_cache_blocks', 'external_cache_bytes',
                             'sampled_os_cache_bytes_after_eviction'])
            stream.flush()
            if args.profile == 'hot':
                connection = open_duckdb(dataset, temp_directory, args.memory_limit_gib)
                info['effective_memory_limit'] = connection.execute(
                    "SELECT current_setting('memory_limit')"
                ).fetchone()[0]
                (output / 'metadata.json').write_text(json.dumps(info, indent=2) + '\n')
                initial_residency = evict_os_page_cache(files)
            for qnum in range(1, 23):
                if args.profile in ('cold', 'paired'):
                    connection = open_duckdb(dataset, temp_directory, args.memory_limit_gib)
                    if info['effective_memory_limit'] is None:
                        info['effective_memory_limit'] = connection.execute(
                            "SELECT current_setting('memory_limit')"
                        ).fetchone()[0]
                        (output / 'metadata.json').write_text(json.dumps(info, indent=2) + '\n')
                    initial_residency = evict_os_page_cache(files)
                for iteration in range(info['iterations']):
                    log(f'Running q{qnum} iteration {iteration}')
                    elapsed, count, blocks, cache_bytes = run_query(
                        connection, sql_by_query[f'q{qnum}']
                    )
                    writer.writerow(['duckdb', f'q{qnum}', iteration, f'{elapsed:.6f}',
                                     count, blocks, cache_bytes,
                                     initial_residency if iteration == 0 else ''])
                    stream.flush()
                    log(f'q{qnum} iteration {iteration}: {elapsed:.3f}s, '
                        f'{count} rows, {cache_bytes} external-cache bytes')
                if args.profile in ('cold', 'paired'):
                    connection.close()
                    connection = None
                    gc.collect()
        info['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        rows = list(csv.DictReader((output / 'runtimes.csv').open(newline='')))
        info['query_time_s'] = [round(sum(float(row['runtime_s']) for row in rows
                                          if row['iteration'] == str(iteration)), 6)
                                for iteration in range(info['iterations'])]
        (output / 'metadata.json').write_text(json.dumps(info, indent=2) + '\n')
        log(f'Completed SF{args.scale_factor} {args.profile}: {info["query_time_s"]}')
    finally:
        if connection is not None:
            connection.close()


if __name__ == '__main__':
    main()
