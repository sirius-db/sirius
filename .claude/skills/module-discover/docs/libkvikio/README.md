# libkvikio — Module Reference

**Version**: 26.02.00
**Location**: `.pixi/envs/default/include/kvikio/`
**Namespace**: `kvikio`
**Conda package**: `libkvikio` (from rapidsai channel)

## Overview

KvikIO (pronounced "quick I/O") is a RAPIDS library providing GPU-accelerated file I/O. It wraps NVIDIA's cuFile/GDS (GPUDirect Storage) APIs with a C++ header-only interface that supports synchronous, asynchronous, parallel, and batched I/O between files and GPU memory. It also supports POSIX fallback, remote file access (S3, HTTP), and memory-mapped I/O.

## Module Map

| Module | Status | Description | Key APIs Used |
|--------|--------|-------------|---------------|
| file_io | UNUSED | Core file I/O via cuFile/GDS with sync, async, and parallel reads/writes | — |
| batch | UNUSED | Batched cuFile I/O operations | — |
| memory | UNUSED | GPU buffer registration and bounce buffer pools | — |
| remote | UNUSED | Remote file access (S3, HTTP, WebHDFS) | — |
| mmap | UNUSED | Memory-mapped file access | — |
| config | UNUSED | Global defaults, compatibility mode, and driver configuration | — |
| shim | UNUSED | Dynamic loading shims for CUDA and cuFile libraries | — |

## Our Usage Summary

We use **0 of 7 modules**. libkvikio is a **transitive dependency** — it is pulled in by libcudf via the RAPIDS conda ecosystem but is not directly referenced anywhere in the Sirius codebase (`src/`, `test/`, `cucascade/`, or CMake files).

### Why it's installed
- libcudf depends on libkvikio for its Parquet/ORC I/O subsystem (`cudf::io`)
- The dependency is declared in the pixi lock file but not in our direct dependencies

### Potential relevance
- If Sirius ever needs to read Parquet files directly on GPU (bypassing DuckDB's scanner), kvikio's `FileHandle` could enable GDS-accelerated reads
- The `RemoteHandle` module could enable direct GPU reads from S3/HTTP without staging through CPU

## Files That Reference This Library

| Source File | Context | Notes |
|-------------|---------|-------|
| `pixi.lock` | Transitive dependency | Pulled in by libcudf |
| `test/tpch_performance/pixi.lock` | Transitive dependency | Same |
| `cucascade/.clang-format` | Include ordering regex | Groups kvikio with other RAPIDS libs |
