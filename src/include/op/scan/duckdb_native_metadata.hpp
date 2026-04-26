/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

// Level-1 metadata-operator split for the DuckDB-native scan path.
//
// Today, `duckdb_native_scan_global_state` walks the per-column segment trees
// inline in its constructor and decides viability there. This couples three
// concerns: walking metadata, deciding viability, and running the scan.
//
// The parquet path already split these via
// `sirius_parquet_metadata_scan_operator` → `partitioned_parquet_metadata`
// → consumer scans. This header introduces the analogous data structure and
// free function for the DuckDB-native path:
//
//   1. `partitioned_duckdb_native_metadata` — what the walk produces (segment
//      scans, viability, per-RG decode budget). Carries an `io_object`
//      placeholder so consumers don't need a new signature when PR H lands.
//
//   2. `walk_duckdb_native_metadata` — pure free function that walks segment
//      trees and returns the metadata struct. PR E's scan-task global state
//      consumes the result; PR H promotes this function body into a real
//      `sirius_duckdb_native_metadata_scan_operator` without changing
//      consumer signatures.
//
// The free-function shape lets PR E ship the structural refactor at ~150 LOC
// of overhead without paying for planner/pipeline plumbing — that's PR H's
// territory once sirius_io's prefetching cache is the IO backend.

#pragma once

#include "helper/logical_type.hpp"
#include "op/scan/direct_block_scan.hpp"

#include <duckdb/common/types.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/storage_index.hpp>
#include <duckdb/storage/table/row_group.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace sirius::op::scan {

/// @brief Output of `walk_duckdb_native_metadata` — a self-contained snapshot
/// of everything the scan-task global state needs to construct itself without
/// re-walking segment trees.
struct partitioned_duckdb_native_metadata {
  /// @brief Empty placeholder; PR H replaces with `sirius::io::sirius_io_object`
  /// for prefetch wiring. Carrying the field now means consumer signatures
  /// don't change when PR H lands.
  struct io_object_placeholder {};
  std::optional<io_object_placeholder> io_object;

  /// @brief One entry per projected column, in projection order. Each contains
  /// data + validity segment scans across all `row_groups` below.
  std::vector<column_scan_result> column_scans;

  /// @brief Row groups visited for `column_scans` (parallel to the per-column
  /// segment vectors inside each `column_scan_result`).
  std::vector<duckdb::RowGroup*> row_groups;

  /// @brief Per-row-group decoded-byte budget summed across projected columns.
  /// Drives batch sizing in the consumer; not used for viability.
  std::vector<std::size_t> rg_decoded_bytes;

  /// @brief True iff every walked segment is GPU-decodable and validity
  /// compressions are within the supported set. False reasons live in
  /// `viability_failure_reason`.
  bool viable = false;

  /// @brief Empty when `viable`. Non-empty diagnostic string otherwise; the
  /// scan-task creator may log it before falling back to `duckdb_scan_task`.
  std::string viability_failure_reason;
};

/// @brief Walk the segment trees of `projected_cols` across `row_groups` and
/// return a metadata snapshot.
///
/// Visits each row group's data tree and validity tree per column, populates
/// `column_scan_result` for each, and decides per-segment compression
/// viability (data-side + validity-side). Does NOT consult `LogicalGet`
/// fields like `dynamic_filters` / `extra_info.sample_options` — those are
/// scan-operator concerns checked by the caller before invoking this walk.
///
/// @param storage          Live DuckDB data table.
/// @param projected_cols   Storage column indices to walk (rowid columns must
///                         be filtered upstream — this walker assumes every
///                         entry corresponds to a real `StandardColumnData`).
/// @param projected_types  Logical types of `projected_cols`, parallel.
/// @param row_groups       Row groups to visit; non-empty.
/// @param context          DuckDB client context (for `BufferManager` access).
partitioned_duckdb_native_metadata walk_duckdb_native_metadata(
  duckdb::DataTable& storage,
  std::vector<duckdb::StorageIndex> const& projected_cols,
  std::vector<sirius::logical_type> const& projected_types,
  std::vector<duckdb::RowGroup*> const& row_groups,
  duckdb::ClientContext& context);

}  // namespace sirius::op::scan
