/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include <rmm/cuda_stream_view.hpp>

#include <cudf/types.hpp>

#include <optional>
#include <string>
#include <vector>

namespace cucascade {
class data_batch;
}

namespace sirius {

/**
 * @brief Null bitmask copied to host for per-row null checking.
 *
 * Used internally by debug utilities. Designed with offset awareness
 * for future phases that access row-level data from sliced columns.
 */
struct host_column_nulls {
  std::vector<cudf::bitmask_type> mask;
  bool has_nulls{false};

  /**
   * @brief Check if a specific row is null.
   * @param row Row index (caller must add col.offset() for sliced columns)
   */
  bool is_null(int row) const;
};

/**
 * @brief Copy a column's null bitmask from device to host.
 *
 * @param col Column whose null mask to copy
 * @param stream CUDA stream for async memcpy
 * @return host_column_nulls with mask data (empty if column has no nulls)
 */
host_column_nulls copy_null_mask_to_host(cudf::column_view const& col,
                                         rmm::cuda_stream_view stream);

/**
 * @brief Log schema metadata for a data batch.
 *
 * Outputs column names, types, null counts, and total row count
 * as a structured [SIRIUS_DIAG] block in sirius.log.
 * Safe to call from any pipeline thread -- output is buffered into
 * a single string and emitted in one atomic SIRIUS_LOG_DEBUG call.
 *
 * @param batch     The data batch to inspect (must be in GPU tier)
 * @param stream    CUDA stream for synchronization (per INFRA-01)
 * @param col_names Optional column names (cudf::table_view has no names)
 */
void debug_schema(cucascade::data_batch const& batch,
                  rmm::cuda_stream_view stream,
                  std::vector<std::string> const& col_names = {});

/**
 * @brief Log per-column null counts and percentages.
 *
 * Uses column_view::null_count() metadata only -- no GPU kernel launched (per NULL-02).
 * Outputs a [SIRIUS_DIAG] block with null analysis.
 *
 * @param batch     The data batch to inspect (must be in GPU tier)
 * @param stream    CUDA stream for synchronization (per INFRA-01)
 * @param col_names Optional column names
 */
void debug_nulls(cucascade::data_batch const& batch,
                 rmm::cuda_stream_view stream,
                 std::vector<std::string> const& col_names = {});

/**
 * @brief Output format for debug_head row display.
 */
enum class DebugFormat { ALIGNED, CSV };

/**
 * @brief Log the first N rows of a data batch as [SIRIUS_DIAG] output.
 *
 * Copies only the first N rows from GPU to host (via cudf::slice zero-copy
 * view) and formats them in either fixed-width aligned columns or CSV.
 * Supports all numeric types (INT8-64, UINT8-64, FLOAT32/64), BOOL8,
 * STRING (with truncation), DECIMAL32/64/128 (fixed-point format),
 * TIMESTAMP (all resolutions), and DATE (TIMESTAMP_DAYS).
 *
 * @param batch          The data batch to inspect (must be in GPU tier)
 * @param n              Number of rows to display (clamped to actual row count)
 * @param stream         CUDA stream for device-to-host copies
 * @param format         Output format: ALIGNED (default) or CSV
 * @param col_names      Optional column names (falls back to col[N])
 * @param max_string_len Maximum display length for STRING values. Longer
 *                       strings truncated with '...' suffix. Pass 0 for no
 *                       truncation (per D-02).
 */
void debug_head(cucascade::data_batch const& batch,
                cudf::size_type n,
                rmm::cuda_stream_view stream,
                DebugFormat format = DebugFormat::ALIGNED,
                std::vector<std::string> const& col_names = {},
                cudf::size_type max_string_len = 50);

/**
 * @brief Log per-column min, max, sum statistics as [SIRIUS_DIAG] output.
 *
 * Computes statistics entirely on GPU using cudf::minmax and cudf::reduce.
 * No full column data is copied to host. Numeric columns (INT8-64, UINT8-64,
 * FLOAT32/64) show min, max, and sum. Non-numeric columns (BOOL, STRING,
 * DECIMAL, TIMESTAMP, DATE) display as "(non-numeric, skipped)".
 *
 * @param batch     The data batch to inspect (must be in GPU tier)
 * @param stream    CUDA stream for GPU reduction operations
 * @param col_names Optional column names (falls back to col[N])
 */
void debug_stats(cucascade::data_batch const& batch,
                 rmm::cuda_stream_view stream,
                 std::vector<std::string> const& col_names = {});

/**
 * @brief Log per-column xxhash_64 fingerprints as [SIRIUS_DIAG] output.
 *
 * Computes a deterministic 64-bit checksum per column using cudf::hashing::xxhash_64
 * to hash all rows, then cudf::reduce with bitwise XOR to collapse to a single value.
 * Computation stays entirely on GPU -- no column data is copied to host.
 * Same data in same order produces same checksum across runs (per D-12).
 *
 * Output format per column: "col[N] checksum: 0xABCD1234EF567890 nulls=2" (per D-11).
 *
 * @param batch     The data batch to checksum (must be in GPU tier)
 * @param stream    CUDA stream for GPU operations
 * @param col_names Optional column names (falls back to col[N])
 */
void debug_checksum(cucascade::data_batch const& batch,
                    rmm::cuda_stream_view stream,
                    std::vector<std::string> const& col_names = {});

/**
 * @brief Compare two data batches and log schema/value differences.
 *
 * Checks schema (column count, types) first, then row count, then
 * per-column value comparison on host. Reports per-column diff count
 * and first max_diff_rows differing row indices. Skips value comparison
 * if either batch exceeds max_rows to prevent OOM.
 *
 * @param batch_a       First batch (must be in GPU tier)
 * @param batch_b       Second batch (must be in GPU tier)
 * @param stream        CUDA stream for device-to-host copies
 * @param max_diff_rows Max differing row indices shown per column (default 10)
 * @param max_rows      Skip value comparison if either batch exceeds this (default 10M)
 * @param col_names     Optional column names
 */
void debug_diff(cucascade::data_batch const& batch_a,
                cucascade::data_batch const& batch_b,
                rmm::cuda_stream_view stream,
                cudf::size_type max_diff_rows = 10,
                cudf::size_type max_rows = 10'000'000,
                std::vector<std::string> const& col_names = {});

/**
 * @brief Log N randomly selected rows from a data batch.
 *
 * Generates random row indices on host via std::mt19937, uses cudf::gather
 * to extract those rows from GPU, then formats output using the same
 * pipeline as debug_head (aligned columns or CSV).
 *
 * @param batch          The data batch to sample (must be in GPU tier)
 * @param n              Number of rows to sample (clamped to batch size)
 * @param stream         CUDA stream for device-to-host copies
 * @param format         Output format: ALIGNED (default) or CSV
 * @param col_names      Optional column names
 * @param max_string_len Maximum display length for STRING values (0 = no limit)
 * @param seed           Optional RNG seed for reproducible sampling
 */
void debug_sample(cucascade::data_batch const& batch,
                  cudf::size_type n,
                  rmm::cuda_stream_view stream,
                  DebugFormat format = DebugFormat::ALIGNED,
                  std::vector<std::string> const& col_names = {},
                  cudf::size_type max_string_len = 50,
                  std::optional<uint64_t> seed = std::nullopt);

}  // namespace sirius
