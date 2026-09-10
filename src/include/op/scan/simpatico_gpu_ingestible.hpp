/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// A .hpln file as a scan SOURCE rather than only a pin-time representation.
//
// A .hpln holds N independently compressed chunks, and this source is one file, one split per
// chunk, coalesced into decode-sized batches. Column projection is honoured -- an unread column is
// never decoded. The query's pushed-down filter is used twice: the file's own per-group zone maps
// drop chunks that cannot match before anything is read, and narrow a surviving chunk to the
// 1024-row decode chunks whose bounds could match; what is left is applied exactly after the
// decode, because DuckDB removes a pushed-down filter from the plan and holds the source
// responsible for it. The decode itself is the pinned-chunk path unchanged: a chunk's payload
// stages into pinned host memory byte-for-byte (read_hpln_chunks_into_pinned) and is then
// reconstructed and decompressed exactly as a pinned chunk is (compression_converters.cpp,
// decompress_host_to_gpu). What this class supplies is the split-provider shape the scan operator
// drives.

// sirius
#include <compression/compressed_scan.hpp>
#include <helper/logical_type.hpp>
#include <op/scan/gpu_ingestible.hpp>
#include <scan_manager/pinned_chunk_stats.hpp>
#include <sirius_config.hpp>

// cudf
#include <cudf/types.hpp>

// duckdb
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>
#include <duckdb/planner/expression.hpp>
#include <duckdb/planner/table_filter.hpp>

// standard library
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace cucascade::memory {
class memory_space;
}  // namespace cucascade::memory

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// simpatico_ingestible_table_info
//===----------------------------------------------------------------------===//
/**
 * @brief Bind data for a scan over one .hpln file; factory input for
 *        @c simpatico_gpu_ingestible.
 *
 * Everything here comes from @c read_hpln_schema, which parses the header and the file's
 * `logical_types` segment without staging any payload and without a GPU -- so a bind costs one
 * tail read plus one header read, which is what makes the format bindable at all.
 */
class simpatico_ingestible_table_info : public ingestible_table_info {
 public:
  /// The .hpln path, and the pinned-cache identity for this table. Exactly one entry: a table is
  /// one file, however many chunks that file holds.
  std::vector<std::string> resolved_file_paths;
  std::vector<std::string> names;
  /// The engine's logical types, from the file when it declares them and derived from the cuDF
  /// physical types otherwise -- lossy for DECIMAL precision, nullability and time zones.
  duckdb::vector<duckdb::LogicalType> types;
  /// What each column decodes to in cuDF. Sizing the decoded table needs these rather than
  /// @ref types: a DECIMAL is INT64 to the engine and may be DECIMAL32 in the file.
  std::vector<cudf::data_type> physical_types;
  /// Rows over the whole file, summed over its chunks.
  std::int64_t num_rows = 0;
  /// Rows in each chunk, in file order. Its size is the number of splits the walk emits, and its
  /// entries size each split's reservation -- both of which have to be known before anything is
  /// decoded.
  std::vector<std::int64_t> chunk_rows;
  /// Byte budget a batch of chunks is coalesced up to; 0 means no budget, as it does for every
  /// other source. A chunk always forms at least one batch on its own, since a chunk is the
  /// smallest decodable unit of the file.
  std::size_t approximate_batch_size = sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE;

  /// The scan's pushed-down filter, keyed by POSITION into @ref duckdb_column_ids (the remapping
  /// create_table_filter_set does). Owned rather than borrowed: the physical scan operator this
  /// is built from is destroyed as the leaf replaces it. Null when the query has no filter.
  ///
  /// Accepting it is a promise to APPLY it: `read_simpatico` sets `filter_pushdown`, so DuckDB
  /// deletes the predicate from the plan and no operator above the scan will re-check it.
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters;
  /// The scan's columns as DuckDB names them, positional with @ref column_ids. Needed to resolve
  /// a filter key onto a file column.
  duckdb::vector<duckdb::ColumnIndex> duckdb_column_ids;
  /// Types of ALL the file's columns, as the filter's constants are typed.
  duckdb::vector<sirius::logical_type> returned_types;
  /// Per-(column, chunk, group) min/max from the file's `zone_maps` segment, positional with the
  /// FILE's columns. Empty means "serve unpruned"; see @ref sirius::hpln_bind_schema.
  scan_manager::group_bounds_arena group_bounds;

  /// File column indices to decode, in the order the scan emits them. A scan that reads two of
  /// ten columns must emit two: the plan projects by output POSITION, so emitting the file's full
  /// width would silently shift every reference. Filled with the identity by
  /// @ref bind_simpatico_file, so a caller that wants the whole file need not set it.
  std::vector<std::size_t> column_ids;

  /// Where the payload stages before it is fetched to the GPU. Pinned, so the H2D copy is a DMA
  /// rather than a staged bounce, and so a pin can adopt the same blob untouched. Required: an
  /// ingest with nowhere to stage cannot be served, and refusing at construction beats failing on
  /// the first split.
  cucascade::memory::memory_space* host_space = nullptr;

  /// Backend the file is read through, resolved at bind. Null means the local filesystem, which
  /// is what a host test that drives the ingestible without a scan manager gets; an `s3://` path
  /// with no io_context is refused by the transport rather than read from the local filesystem.
  std::shared_ptr<io::sirius_ioctx> io_ctx;

  simpatico_ingestible_table_info() = default;

  [[nodiscard]] std::span<std::string const> column_names() const override { return names; }

  [[nodiscard]] std::span<std::string const> file_paths() const override
  {
    return resolved_file_paths;
  }
};

/// Bind @p path: read its schema and build the table info a @c simpatico_gpu_ingestible is
/// constructed from. Throws std::runtime_error if the file is missing or is not a readable .hpln.
[[nodiscard]] std::unique_ptr<simpatico_ingestible_table_info> bind_simpatico_file(
  std::string const& path,
  cucascade::memory::memory_space& host_space,
  std::shared_ptr<io::sirius_ioctx> io_ctx = nullptr);

//===----------------------------------------------------------------------===//
// simpatico_scan_info
//===----------------------------------------------------------------------===//
/**
 * @brief One unit of .hpln scan work: a run of the file's chunks.
 *
 * The walk emits one chunk per split and the coalescer bundles them, so a batch carries several
 * ids and decodes to their concatenation. Order within @ref chunk_ids is the order the rows are
 * emitted in, so a batch that reordered them would return the file's rows shuffled between
 * chunks.
 *
 * Carries no byte ranges to advise. Where parquet's split names row groups so the sequencer can
 * prefetch them, a .hpln reader locates every chunk from the trailer, so there is nothing to hand
 * the prefetcher that it does not already read in one pass.
 */
class simpatico_scan_info : public scan_info {
 public:
  std::string path;
  /// Backend this split reads through, resolved once on the walk. Carried on the split rather
  /// than read from a member so a concurrent walk and materialize never race for it.
  std::shared_ptr<io::sirius_ioctx> io_ctx;
  /// File-order chunk ids this split decodes, ascending.
  std::vector<std::size_t> chunk_ids;
  /// Positional with @ref chunk_ids: the chunk's surviving 1024-row decode chunk ids, or an empty
  /// vector meaning "decode the chunk whole". A subsetted chunk is fetched and decoded as a
  /// smaller table describing only those rows (simpatico::build_chunk_subset_header), so the
  /// pruned rows cross neither the file read nor PCIe nor the decode.
  std::vector<std::vector<std::uint32_t>> decode_chunks;
  /// Rows the split's chunks decode to, summed.
  std::int64_t num_rows = 0;
  /// Decoded size of the columns this split produces; drives the memory reservation.
  std::size_t decoded_bytes = 0;

  [[nodiscard]] std::size_t estimated_bytes() const noexcept override { return decoded_bytes; }
};

//===----------------------------------------------------------------------===//
// simpatico_gpu_ingestible
//===----------------------------------------------------------------------===//
/**
 * @brief Scan source over a .hpln file.
 *
 * The metadata walk emits one split per SURVIVING chunk of the file and the coalescer bundles
 * consecutive chunks up to a byte budget, so a batch is decode-sized rather than chunk-sized.
 * Every split decodes @c simpatico_ingestible_table_info::column_ids and nothing else.
 *
 * Which chunks survive, and which of a survivor's 1024-row decode chunks do, is decided once at
 * construction from the file's zone maps and this scan's filter. Bounds narrow what is READ; they
 * do not test rows. Testing them is the decode's job where it can -- @c decompress_chunk evaluates
 * the filter's bounds while a column decompresses and hands back only the surviving rows -- and
 * @ref post_filter_and_project's otherwise, which covers every chunk whose decode did not carry
 * the WHOLE filter. That fallback is not optional: DuckDB deleted the predicate from the plan.
 */
class simpatico_gpu_ingestible : public gpu_ingestible {
 public:
  explicit simpatico_gpu_ingestible(std::unique_ptr<simpatico_ingestible_table_info> info);

  ~simpatico_gpu_ingestible() override;

  [[nodiscard]] std::unique_ptr<batch_coalescer> create_batch_coalescer() const override;

  [[nodiscard]] bool has_processed_all_metadata() const override;

  metadata_scan_task_t next_split_provider(io::ioctx_resolver resolve) override;

  filtered_table materialize_metadata_to_table(
    scan_info const& info,
    ::cucascade::memory::memory_space const& mem_space,
    rmm::cuda_stream_view stream,
    bool like_swar_fastpath,
    std::shared_ptr<const sirius::like_multiliteral_cache> like_cache) override;

  std::unique_ptr<cudf::table> post_filter_and_project(
    filtered_table&& input,
    ::cucascade::memory::memory_space const& mem_space,
    rmm::cuda_stream_view stream,
    bool like_swar_fastpath,
    std::shared_ptr<const sirius::like_multiliteral_cache> like_cache,
    std::unique_ptr<cudf::column>* survivors,
    std::span<std::size_t const> elided) override;

  [[nodiscard]] ingestible_table_info const& table_info() const noexcept override { return *_info; }

  [[nodiscard]] std::vector<std::size_t> materialized_column_order() const override;

  [[nodiscard]] bool has_row_filter() const noexcept override
  {
    return _filter_expression != nullptr;
  }

  /// Chunks the zone maps dropped outright, and decode chunks dropped inside a surviving one.
  /// Reported for tests and for the log line: a scan that prunes nothing still returns the right
  /// rows, so the only way to know pruning happened is to count it.
  struct prune_stats {
    std::size_t chunks_total{0};
    std::size_t chunks_pruned{0};
    std::size_t decode_chunks_total{0};
    std::size_t decode_chunks_pruned{0};
  };
  [[nodiscard]] prune_stats pruning() const noexcept { return _prune_stats; }
  /// Splits whose subset header was refused and served whole (see @ref
  /// materialize_metadata_to_table).
  [[nodiscard]] std::size_t subset_refusals() const noexcept
  {
    return _subset_refusals.load(std::memory_order_relaxed);
  }

  /// What the decode-time filtering did, counted over the chunks this scan decoded.
  ///
  /// Like @ref prune_stats, this exists because the layer is invisible in the answer: a decode
  /// that filtered nothing returns the same rows as one that filtered everything, so counting is
  /// the only way to tell that it ran at all.
  struct pushdown_stats {
    /// Chunks a decode-time request was attached to.
    std::size_t chunks_offered{0};
    /// ... of those, the ones whose decode carried the WHOLE filter and handed back compacted
    /// columns, so the post-decode filter was skipped for them.
    std::size_t chunks_row_filtered{0};
    /// ... and the ones where too many rows survived for compaction to pay for itself. The decode
    /// then returns ordinary full-width columns and the post-decode filter still applies.
    std::size_t chunks_unprofitable{0};
  };
  [[nodiscard]] pushdown_stats decode_filtering() const noexcept
  {
    return {_pushdown_offered.load(std::memory_order_relaxed),
            _pushdown_row_filtered.load(std::memory_order_relaxed),
            _pushdown_unprofitable.load(std::memory_order_relaxed)};
  }

 private:
  /// One chunk that survived the zone-map pass, with what of it is worth reading.
  struct live_chunk {
    std::size_t id{0};
    /// Empty when every decode chunk survives: an unpruned chunk must take the ordinary whole-file
    /// path rather than a subset header describing all of it, which would cost the synthesis for
    /// nothing.
    std::vector<std::uint32_t> decode_chunks;
    /// Rows the chunk will actually produce -- fewer than the chunk's own when subsetted, which
    /// is what the reservation and the batch budget have to be sized by.
    std::int64_t num_rows{0};
  };

  /// Decide, once, which chunks survive the file's zone maps and which of their decode chunks do.
  void plan_pruning();

  std::unique_ptr<simpatico_ingestible_table_info> _info;
  /// The pushed-down filter as one conjunction over batch positions, or null when there is none.
  /// Applied in @ref post_filter_and_project: zone maps bound rows, they do not test them.
  duckdb::unique_ptr<duckdb::Expression> _filter_expression;
  /// The same filter as the DECODE can use it -- bounds a column evaluates while it decompresses,
  /// so a rejected row is never fully reconstructed. Null when nothing of the filter survived the
  /// analysis, or when the experimental gate is off. Shared by every chunk and narrowed per chunk
  /// by @c decompression_pushdown_scan::for_chunk, which is where a compression plan is consulted.
  std::shared_ptr<const sirius::decompression_pushdown_scan> _pushdown_scan;
  /// Set once a decode reports that too many rows survived for compaction to pay for itself.
  /// Selectivity barely varies across one file's chunks, so one such chunk predicts the rest and
  /// the remaining ones stop paying for the attempt -- the same conclusion the pinned path draws
  /// from @c pushdown_outcome::selection_unprofitable.
  std::atomic<bool> _row_selection_dropped{false};
  std::atomic<std::size_t> _pushdown_offered{0};
  std::atomic<std::size_t> _pushdown_row_filtered{0};
  std::atomic<std::size_t> _pushdown_unprofitable{0};
  /// The chunks a split may be emitted for, ascending. Not simply 0..N: a pruned chunk is absent.
  std::vector<live_chunk> _live_chunks;
  prune_stats _prune_stats;
  std::atomic<std::size_t> _subset_refusals{0};
  /// Next unclaimed entry of @ref _live_chunks. An atomic for the same reason parquet's file index
  /// is: the driver may call @ref next_split_provider from several dispatcher threads, and a chunk
  /// handed to two of them would be emitted twice -- which shows up as a plausible row count, not
  /// as a failure.
  std::atomic<std::size_t> _next_chunk{0};
};

[[nodiscard]] std::shared_ptr<simpatico_gpu_ingestible> make_ingestible(
  std::unique_ptr<simpatico_ingestible_table_info> info);

}  // namespace sirius::op::scan
