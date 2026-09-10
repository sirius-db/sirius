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

// sirius
#include <compression/compressed_representation.hpp>
#include <compression/compressed_scan.hpp>
#include <compression/decompression_pushdown_policy.hpp>
#include <compression/simpatico_file_ingest.hpp>
#include <expression/ast/from_duckdb.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <log/logging.hpp>
#include <op/scan/owning_table_view.hpp>
#include <op/scan/scan_filter_analysis.hpp>
#include <op/scan/scan_utils.hpp>
#include <op/scan/simpatico_gpu_ingestible.hpp>
#include <scan_manager/sirius_scan_manager.hpp>

// cudf
#include <cudf/concatenate.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>

// cucascade
#include <cucascade/memory/memory_space.hpp>

// simpatico
#include <api/compressed_table_io.hpp>
#include <api/simpatico_codegen.hpp>
#include <codegen/jit/fused_tree.hpp>

// standard library
#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>

namespace sirius::op::scan {

namespace {

/// Decoded footprint of the columns this scan emits, for the reservation.
///
/// Exact for fixed-width columns: the header carries the row count and the decoded type. A
/// variable-width column has no such answer without decoding it, so it contributes only its
/// offsets -- a floor, not an estimate. A file with strings therefore under-reserves until the
/// format records decoded sizes the way parquet's SizeStatistics do.
std::size_t estimate_decoded_bytes(simpatico_ingestible_table_info const& info,
                                   std::int64_t num_rows)
{
  auto const rows   = static_cast<std::size_t>(std::max<std::int64_t>(num_rows, 0));
  std::size_t total = 0;
  for (auto const column : info.column_ids) {
    auto const& dtype = info.physical_types[column];
    total += rows * (cudf::is_fixed_width(dtype) ? cudf::size_of(dtype) : sizeof(std::int32_t));
  }
  return total;
}

/// Bundle per-chunk splits into decode-sized batches.
///
/// A chunk is whatever size the writer chose, and one decode per chunk is one launch, one pinned
/// staging allocation and one reservation each -- so a file of many small chunks pays per chunk
/// rather than per byte. This accumulates them the way parquet_batch_coalescer accumulates row
/// groups: keep adding until the next one would exceed the byte budget, then seal. A chunk larger
/// than the budget on its own still forms a batch, because splitting one is not possible -- a
/// chunk is the smallest decodable unit of the file.
///
/// The cudf row ceiling is a hard limit rather than a preference: a batch whose chunks sum past
/// size_type rows cannot be concatenated into one table at all.
class simpatico_batch_coalescer : public batch_coalescer {
 public:
  explicit simpatico_batch_coalescer(std::size_t cap) : _cap(cap) {}

  std::vector<std::unique_ptr<scan_info>> push(std::unique_ptr<scan_info> info) override
  {
    std::vector<std::unique_ptr<scan_info>> out;
    auto* split = dynamic_cast<simpatico_scan_info*>(info.get());
    if (split == nullptr) { return out; }

    static constexpr std::int64_t kCudfMaxRows = std::numeric_limits<cudf::size_type>::max();
    bool const byte_cap_hit =
      _current && _cap > 0 && _current->decoded_bytes + split->decoded_bytes > _cap;
    bool const row_cap_hit = _current && _current->num_rows + split->num_rows > kCudfMaxRows;
    if (byte_cap_hit || row_cap_hit) { out.push_back(std::move(_current)); }

    if (!_current) {
      _current       = std::make_unique<simpatico_scan_info>();
      _current->path = split->path;
    }
    // Ascending ids keep a batch's rows in file order and its reads sequential. The walk hands
    // chunks out in order, but several dispatcher threads may claim them, so the coalescer can
    // see them out of order and must not preserve that. The surviving-decode-chunk list is
    // inserted at the SAME position: it names rows of its own chunk, so a list that drifted onto
    // a neighbour would decode the wrong rows and still return a plausible count.
    for (std::size_t i = 0; i < split->chunk_ids.size(); ++i) {
      auto const at = static_cast<std::size_t>(std::lower_bound(_current->chunk_ids.begin(),
                                                                _current->chunk_ids.end(),
                                                                split->chunk_ids[i]) -
                                               _current->chunk_ids.begin());
      _current->chunk_ids.insert(_current->chunk_ids.begin() + static_cast<std::ptrdiff_t>(at),
                                 split->chunk_ids[i]);
      _current->decode_chunks.insert(
        _current->decode_chunks.begin() + static_cast<std::ptrdiff_t>(at),
        i < split->decode_chunks.size() ? split->decode_chunks[i] : std::vector<std::uint32_t>{});
    }
    _current->num_rows += split->num_rows;
    _current->decoded_bytes += split->decoded_bytes;
    return out;
  }

  std::vector<std::unique_ptr<scan_info>> flush() override
  {
    std::vector<std::unique_ptr<scan_info>> out;
    if (_current) { out.push_back(std::move(_current)); }
    return out;
  }

 private:
  std::size_t _cap;
  std::unique_ptr<simpatico_scan_info> _current;
};

/// Positions of `width` minus `elided`, ascending. Empty when nothing would be left: a zero-column
/// table carries no row count, so dropping everything loses the batch's length.
std::vector<std::size_t> kept_positions(std::size_t width, std::span<std::size_t const> elided)
{
  if (elided.empty() || elided.size() >= width) { return {}; }
  std::vector<std::size_t> kept;
  kept.reserve(width - elided.size());
  for (std::size_t pos = 0; pos < width; ++pos) {
    if (std::find(elided.begin(), elided.end(), pos) == elided.end()) { kept.push_back(pos); }
  }
  return kept;
}

/// Rows of one simpatico decode chunk -- the granularity a chunk subset addresses.
constexpr std::size_t kDecodeChunkRows = static_cast<std::size_t>(::codegen::kChunkSize);

/// Ceiling division that does not overflow for the sizes involved here.
constexpr std::size_t ceil_div(std::size_t a, std::size_t b)
{
  return b == 0 ? 0 : (a + b - 1) / b;
}

}  // namespace

//===----------------------------------------------------------------------===//
// bind
//===----------------------------------------------------------------------===//
std::unique_ptr<simpatico_ingestible_table_info> bind_simpatico_file(
  std::string const& path,
  cucascade::memory::memory_space& host_space,
  std::shared_ptr<io::sirius_ioctx> io_ctx)
{
  // The bind reads through the same transport the scan will: an `s3://` file has to be bindable
  // before it can be scanned, and binding it locally is not an option.
  sirius::hpln_open_options options;
  options.io_ctx = io_ctx;
  auto schema    = sirius::read_hpln_schema(path, options);

  auto info                 = std::make_unique<simpatico_ingestible_table_info>();
  info->resolved_file_paths = {path};
  info->names               = std::move(schema.names);
  info->types               = std::move(schema.types);
  info->physical_types      = std::move(schema.physical_types);
  info->num_rows            = schema.num_rows;
  info->chunk_rows          = std::move(schema.chunk_rows);
  info->group_bounds        = std::move(schema.group_bounds);
  info->host_space          = &host_space;
  info->io_ctx              = std::move(io_ctx);
  // Whole file by default; a caller with a narrower projection overwrites this.
  info->column_ids.resize(info->names.size());
  std::iota(info->column_ids.begin(), info->column_ids.end(), std::size_t{0});
  return info;
}

//===----------------------------------------------------------------------===//
// construction
//===----------------------------------------------------------------------===//
simpatico_gpu_ingestible::simpatico_gpu_ingestible(
  std::unique_ptr<simpatico_ingestible_table_info> info)
  : _info(std::move(info))
{
  if (!_info) {
    throw std::invalid_argument("[simpatico_gpu_ingestible] table_info must be non-null");
  }
  if (_info->resolved_file_paths.size() != 1) {
    throw std::invalid_argument("[simpatico_gpu_ingestible] expects exactly one .hpln path, got " +
                                std::to_string(_info->resolved_file_paths.size()));
  }
  if (_info->host_space == nullptr) {
    throw std::invalid_argument(
      "[simpatico_gpu_ingestible] table_info.host_space must be non-null");
  }
  if (_info->chunk_rows.empty()) {
    throw std::invalid_argument(
      "[simpatico_gpu_ingestible] table_info.chunk_rows must name at least one chunk; a walk over "
      "no chunks emits no split and the pipeline waits forever for a completion that never fires");
  }
  if (_info->column_ids.empty()) {
    throw std::invalid_argument(
      "[simpatico_gpu_ingestible] table_info.column_ids must name at "
      "least one column");
  }
  // An out-of-range index would otherwise reach simpatico::decompress, whose selection is
  // unchecked -- it reads a neighbouring column's buffers and returns them as data.
  for (auto const column : _info->column_ids) {
    if (column >= _info->names.size()) {
      throw std::invalid_argument("[simpatico_gpu_ingestible] column index " +
                                  std::to_string(column) +
                                  " is out of range "
                                  "for a file with " +
                                  std::to_string(_info->names.size()) + " columns");
    }
  }

  // The decode emits _info->column_ids in order, so the batch position of filter key `i` (a
  // position into duckdb_column_ids, which create_table_filter_set already remapped to) is `i`.
  if (_info->table_filters != nullptr && !_info->table_filters->filters.empty()) {
    std::vector<std::optional<std::size_t>> batch_position(_info->duckdb_column_ids.size());
    for (std::size_t i = 0; i < batch_position.size(); ++i) {
      batch_position[i] = i;
    }
    // Throws for a predicate shape Sirius cannot lower, which fails the query over to DuckDB --
    // where read_simpatico has no CPU reader. That is still the right failure: silently dropping
    // a conjunct DuckDB deleted from the plan would return rows the query excluded.
    _filter_expression = sirius::op::convert_table_filters_to_expression(
      *_info->table_filters, _info->duckdb_column_ids, _info->returned_types, batch_position);

    // The same filter once more, in the form a DECODE can act on: inclusive bounds per column,
    // evaluated while that column decompresses, so a row the predicate rejects is never fully
    // reconstructed. This is additive to the pruning above rather than a replacement for it --
    // the two act on different columns. A group survives if ANY of its rows could match, so after
    // pruning on a clustered column its survivors are nearly all matches and the decode declines
    // to compact on that column alone; what pays here is the predicate on a column the file is
    // NOT clustered by, which the zone maps cannot narrow at all.
    //
    // No equality set is requested: a dictionary-answered equality substitutes the column's
    // values with the BOOL8 answer, which only pays for a column the query never emits, and this
    // source declares no projection pushdown -- every decoded column is an output column.
    // Requesting one would hand the plan a boolean where it expects values.
    auto const analysis = sirius::op::analyze_scan_filters(
      *_info->table_filters, _info->duckdb_column_ids, _info->returned_types);
    // Slot k of the decode is _info->column_ids[k], which is exactly the primary index the
    // analysis keyed its ranges by (the plan generator builds column_ids from the same
    // duckdb_column_ids).
    auto request = sirius::op::build_pushdown_request(analysis, _info->column_ids);
    // Off-gate the request carries nothing this source can use: row dropping is gated, and the
    // equality answers that are not are never asked for here. Dropping it entirely keeps the
    // gate-off path byte-identical to the one before decode-time filtering existed.
    if (!request.empty() && sirius::decompression_pushdown_enabled()) {
      _pushdown_scan =
        std::make_shared<const sirius::decompression_pushdown_scan>(std::move(request));
    }
  }

  plan_pruning();
}

//===----------------------------------------------------------------------===//
// pruning
//===----------------------------------------------------------------------===//
void simpatico_gpu_ingestible::plan_pruning()
{
  auto const n_chunks       = _info->chunk_rows.size();
  _prune_stats              = prune_stats{};
  _prune_stats.chunks_total = n_chunks;
  for (auto const rows : _info->chunk_rows) {
    _prune_stats.decode_chunks_total +=
      ceil_div(static_cast<std::size_t>(std::max<std::int64_t>(rows, 0)), kDecodeChunkRows);
  }

  auto const serve_everything = [&] {
    _live_chunks.clear();
    _live_chunks.reserve(n_chunks);
    for (std::size_t c = 0; c < n_chunks; ++c) {
      _live_chunks.push_back({c, {}, _info->chunk_rows[c]});
    }
  };

  auto const& arena = _info->group_bounds;
  if (_info->table_filters == nullptr || _info->table_filters->filters.empty() || arena.empty() ||
      arena.group_rows() == 0 || arena.chunk_count() != n_chunks) {
    serve_everything();
    return;
  }

  // Lower each filter once for the whole file rather than once per group cell -- the same reason
  // build_survivor_row_ranges does: this is the release-mode line for dropping data, and it is
  // cross-checked against chunk_provably_empty by a randomized test.
  struct lowered_entry {
    scan_manager::lowered_bound_filter filter;
    std::size_t file_column;
  };
  std::vector<lowered_entry> lowered;
  for (auto const& [key, filter] : _info->table_filters->filters) {
    if (!filter) { continue; }
    if (key >= _info->duckdb_column_ids.size()) { continue; }
    auto const& column_id = _info->duckdb_column_ids[key];
    if (!column_id.HasPrimaryIndex() || column_id.IsRowIdColumn() || column_id.IsEmptyColumn() ||
        column_id.IsVirtualColumn()) {
      continue;
    }
    auto const file_column = static_cast<std::size_t>(column_id.GetPrimaryIndex());
    // The type the BOUNDS are in, not the type the file declares: a column the capture could not
    // represent round-trips as "no statistics", and lowering against the declared type would
    // build a filter with nothing to evaluate it on.
    std::optional<duckdb::LogicalType> stats_type;
    for (std::size_t c = 0; c < n_chunks && !stats_type.has_value(); ++c) {
      auto const cell = arena.cell(file_column, c);
      if (!cell.empty()) { stats_type = cell.type; }
    }
    if (!stats_type.has_value()) { continue; }
    // A filter this cannot lower simply does not prune; the post-decode pass still applies it.
    if (auto low = scan_manager::lowered_bound_filter::lower(*filter, *stats_type)) {
      lowered.push_back({std::move(*low), file_column});
    }
  }
  if (lowered.empty()) {
    serve_everything();
    return;
  }

  std::size_t const group_rows = arena.group_rows();
  // Sub-chunk narrowing addresses 1024-row decode chunks, so a group that is not a whole number of
  // them cannot be turned into a chunk-id list. Whole-chunk pruning is unaffected.
  std::size_t const groups_per_decode_span =
    (group_rows % kDecodeChunkRows == 0) ? group_rows / kDecodeChunkRows : 0;

  _live_chunks.clear();
  _live_chunks.reserve(n_chunks);
  for (std::size_t c = 0; c < n_chunks; ++c) {
    auto const rows     = static_cast<std::size_t>(std::max<std::int64_t>(_info->chunk_rows[c], 0));
    auto const n_groups = ceil_div(rows, group_rows);
    auto const n_decode = ceil_div(rows, kDecodeChunkRows);

    // A group survives unless SOME filter proves it empty -- the same AND-over-filters the
    // chunk-level pass applies.
    std::vector<bool> keep(n_groups, true);
    bool any_evidence = false;
    for (auto const& le : lowered) {
      auto const bounds = arena.cell(le.file_column, c);
      // A cell whose shape disagrees with the chunk's own row count describes different rows;
      // using it would drop rows there is no evidence about.
      if (bounds.size() != n_groups) { continue; }
      any_evidence        = true;
      bool const has_null = !bounds.column_has_no_nulls;
      for (std::size_t g = 0; g < n_groups; ++g) {
        if (!keep[g] || bounds.valid[g] == 0) { continue; }  // an absent cell never prunes
        if (le.filter.provably_empty(bounds.mins[g], bounds.maxs[g], has_null, false)) {
          keep[g] = false;
        }
      }
    }
    if (!any_evidence || n_groups == 0) {
      _live_chunks.push_back({c, {}, _info->chunk_rows[c]});
      continue;
    }

    auto const kept_groups = static_cast<std::size_t>(std::count(keep.begin(), keep.end(), true));
    if (kept_groups == 0) {
      // Nothing in this chunk can match: it is never read, never fetched and never decoded.
      ++_prune_stats.chunks_pruned;
      _prune_stats.decode_chunks_pruned += n_decode;
      continue;
    }
    if (kept_groups == n_groups || groups_per_decode_span == 0) {
      _live_chunks.push_back({c, {}, _info->chunk_rows[c]});
      continue;
    }

    std::vector<std::uint32_t> decode_chunks;
    decode_chunks.reserve(kept_groups * groups_per_decode_span);
    for (std::size_t g = 0; g < n_groups; ++g) {
      if (!keep[g]) { continue; }
      auto const first = g * groups_per_decode_span;
      auto const last  = std::min((g + 1) * groups_per_decode_span, n_decode);
      for (auto d = first; d < last; ++d) {
        decode_chunks.push_back(static_cast<std::uint32_t>(d));
      }
    }
    if (decode_chunks.empty()) {
      // Every surviving group lies past the chunk's decode chunks -- impossible for consistent
      // metadata, and dropping the chunk on it would be dropping rows on no evidence.
      _live_chunks.push_back({c, {}, _info->chunk_rows[c]});
      continue;
    }
    _prune_stats.decode_chunks_pruned += n_decode - decode_chunks.size();
    auto const kept_rows = scan_manager::surviving_decode_chunk_rows(decode_chunks, rows);
    _live_chunks.push_back({c, std::move(decode_chunks), static_cast<std::int64_t>(kept_rows)});
  }

  // An all-pruned scan must not become a zero-split scan: zero splits means zero tasks, and the
  // pipeline waits for a completion that never fires. Keep chunk 0 whole and let the post-decode
  // filter empty it -- the same sentinel build_cached_scan_plan keeps.
  if (_live_chunks.empty()) {
    _live_chunks.push_back({0, {}, _info->chunk_rows.front()});
    _prune_stats.chunks_pruned = n_chunks - 1;
    _prune_stats.decode_chunks_pruned =
      _prune_stats.decode_chunks_total -
      ceil_div(static_cast<std::size_t>(std::max<std::int64_t>(_info->chunk_rows.front(), 0)),
               kDecodeChunkRows);
  }

  if (_prune_stats.chunks_pruned > 0 || _prune_stats.decode_chunks_pruned > 0) {
    SIRIUS_LOG_INFO(
      "[simpatico_gpu_ingestible] '{}' zone maps pruned {}/{} chunks and {}/{} decode chunks",
      _info->resolved_file_paths.front(),
      _prune_stats.chunks_pruned,
      _prune_stats.chunks_total,
      _prune_stats.decode_chunks_pruned,
      _prune_stats.decode_chunks_total);
  }
}

simpatico_gpu_ingestible::~simpatico_gpu_ingestible() = default;

//===----------------------------------------------------------------------===//
// split-provider interface
//===----------------------------------------------------------------------===//
bool simpatico_gpu_ingestible::has_processed_all_metadata() const
{
  return _next_chunk.load(std::memory_order_relaxed) >= _live_chunks.size();
}

simpatico_gpu_ingestible::metadata_scan_task_t simpatico_gpu_ingestible::next_split_provider(
  io::ioctx_resolver resolve)
{
  // fetch_add rather than load-then-store: several dispatcher threads may reach here, and two of
  // them reading the same cursor would emit one chunk twice and skip another -- which lands as a
  // wrong answer with a right-looking row count, not as a failure.
  auto const cursor = _next_chunk.fetch_add(1, std::memory_order_relaxed);
  if (cursor >= _live_chunks.size()) { return nullptr; }

  // Resolve the backend HERE, on the walk, the way the parquet source does: the resolver routes
  // by path (`s3://` -> rest, local -> uring/kvikio), and resolving once per split keeps
  // materialize free of the scan manager. The bind already resolved the same path, so this is a
  // map lookup rather than a build.
  auto io_ctx = _info->io_ctx;
  if (resolve) {
    io_ctx = resolve(_info->resolved_file_paths.front());
  } else if (!io_ctx) {
    // No scan manager wired (host tests drive the ingestible directly). A local file still reads
    // through the filesystem; a scheme path is refused by the transport rather than read wrongly.
    SIRIUS_LOG_DEBUG(
      "[simpatico_gpu_ingestible] '{}': no ioctx resolver; reading through the filesystem",
      _info->resolved_file_paths.front());
  }

  // The walk is over the SURVIVING chunks, not over the file's: a chunk the zone maps ruled out
  // never becomes a split, so it is never read, fetched or decoded.
  return [this, cursor, io_ctx = std::move(io_ctx)]() -> std::unique_ptr<scan_info> {
    auto const& live     = _live_chunks[cursor];
    auto split           = std::make_unique<simpatico_scan_info>();
    split->path          = _info->resolved_file_paths.front();
    split->io_ctx        = io_ctx;
    split->chunk_ids     = {live.id};
    split->decode_chunks = {live.decode_chunks};
    // The narrowed row count, so the reservation and the batch budget are sized by what the
    // decode will actually return rather than by the chunk it came from.
    split->num_rows      = live.num_rows;
    split->decoded_bytes = estimate_decoded_bytes(*_info, split->num_rows);
    return split;
  };
}

//===----------------------------------------------------------------------===//
// materialize
//===----------------------------------------------------------------------===//
filtered_table simpatico_gpu_ingestible::materialize_metadata_to_table(
  scan_info const& info,
  cucascade::memory::memory_space const& mem_space,
  rmm::cuda_stream_view stream,
  bool /*like_swar_fastpath*/,
  std::shared_ptr<const like_multiliteral_cache> /*like_cache*/)
{
  auto const& split = static_cast<simpatico_scan_info const&>(info);
  if (split.chunk_ids.empty()) {
    throw std::runtime_error("[simpatico_gpu_ingestible] '" + split.path +
                             "' split names no chunks");
  }

  // Stage byte-for-byte into pinned host memory, then fetch the leaf buffers the reader asks for
  // straight out of those blocks. Nothing is decoded on the way in, so a served table costs one
  // file read plus one decode rather than a decode and a re-compress.
  sirius::hpln_open_options options;
  options.io_ctx = split.io_ctx;
  auto ingested =
    sirius::read_hpln_chunks_into_pinned(split.path, *_info->host_space, split.chunk_ids, options);

  rmm::device_async_resource_ref mr(mem_space.get_default_allocator());
  std::vector<std::unique_ptr<cudf::table>> decoded;
  decoded.reserve(ingested.size());
  // Cleared by any chunk the decode did not fully filter; see the decision below.
  bool row_filtered_batch = _pushdown_scan != nullptr;
  for (std::size_t i = 0; i < ingested.size(); ++i) {
    auto const& blob = *ingested[i].blob;
    simpatico::payload_fetch_fn whole_fetch =
      [&blob](std::uint64_t off, std::size_t sz, void* dst, rmm::cuda_stream_view s) {
        copy_pinned_blocks_to_device(*blob.payload, off, dst, sz, s);
      };

    // Zone maps ruled some of this chunk's 1024-row decode chunks out. Synthesize a header
    // describing only the survivors and gather only their bytes: the reader and the decode see an
    // ordinary, smaller table and need no knowledge that a subset is in play. Every failure here
    // degrades to decoding the chunk whole, which is correct -- just less selective -- so nothing
    // below throws on a refusal.
    std::vector<std::uint8_t> subset_header;
    std::vector<simpatico::gather_range> gather;
    bool use_subset = false;
    if (i < split.decode_chunks.size() && !split.decode_chunks[i].empty()) {
      simpatico::payload_host_read_fn read_metadata =
        [&blob](std::uint64_t off, std::uint64_t size, void* dst) {
          copy_pinned_blocks_to_host(*blob.payload, off, dst, size);
          return true;
        };
      std::vector<std::uint8_t> subsetted;
      auto const error = simpatico::build_chunk_subset_header(blob.header,
                                                              split.decode_chunks[i],
                                                              read_metadata,
                                                              subset_header,
                                                              gather,
                                                              /*max_gap_bytes=*/0,
                                                              /*out_payload_bytes=*/nullptr,
                                                              &subsetted);
      // A column the format cannot address per chunk is emitted WHOLE, and a whole column beside
      // a compacted one has a different row count -- the two cannot be one cudf::table. So the
      // subset is usable only when every column this scan READS was compacted. Unread columns may
      // be whole: their buffers are never fetched.
      auto const refusing_column = [&]() -> std::optional<std::size_t> {
        for (auto const column : _info->column_ids) {
          if (column >= subsetted.size() || subsetted[column] == 0) { return column; }
        }
        return std::nullopt;
      }();
      use_subset = error.empty() && !refusing_column.has_value();
      if (!use_subset) {
        _subset_refusals.fetch_add(1, std::memory_order_relaxed);
        if (error.empty()) {
          SIRIUS_LOG_DEBUG(
            "[simpatico_gpu_ingestible] '{}' chunk {}: column {} ({}) is not chunk-addressable; "
            "decoding the chunk whole",
            split.path,
            split.chunk_ids[i],
            *refusing_column,
            *refusing_column < _info->names.size() ? _info->names[*refusing_column] : "?");
        } else {
          SIRIUS_LOG_WARN(
            "[simpatico_gpu_ingestible] '{}' chunk {}: chunk subset refused ({}); decoding the "
            "chunk whole",
            split.path,
            split.chunk_ids[i],
            error);
        }
      }
    }

    simpatico::payload_fetch_fn const gathered_fetch =
      gathered_payload_fetch{*blob.payload, std::move(gather)};

    std::string read_error;
    // Reconstruct only the columns this scan reads: a projection of a wide file then never pulls
    // the other columns' compressed bytes across PCIe at all.
    auto const compressed = simpatico::read_compressed_table_subset_from_memory(
      use_subset ? std::span<const std::uint8_t>{subset_header} : blob.header,
      use_subset ? gathered_fetch : whole_fetch,
      _info->column_ids,
      stream,
      mr,
      &read_error);
    if (!read_error.empty()) {
      throw std::runtime_error("[simpatico_gpu_ingestible] '" + split.path + "' chunk " +
                               std::to_string(split.chunk_ids[i]) +
                               " could not be reconstructed: " + read_error);
    }

    // `compressed` already holds only the projected columns, in the requested order.
    std::vector<std::size_t> selection(compressed.num_columns());
    std::iota(selection.begin(), selection.end(), std::size_t{0});

    // This scan's filter narrowed to what THIS chunk can answer. `compressed` is the table the
    // decode will see -- the compacted subset when the zone maps narrowed the chunk to some of its
    // decode chunks -- so the narrowing, and everything the decode decides off a plan tree, is
    // about the rows that will actually decode rather than about the chunk they came from.
    std::shared_ptr<const sirius::decompression_pushdown_scan> chunk_scan;
    if (_pushdown_scan && !_row_selection_dropped.load(std::memory_order_relaxed)) {
      chunk_scan = _pushdown_scan->for_chunk(compressed, selection);
    }

    if (!chunk_scan) {
      // Decode on the caller's stream rather than on the decode thread pool. The pool overlaps
      // columns, but its buffers are freed on pool streams and would have to be re-bound to
      // `stream` before anything downstream may read them. It also keeps the fetch above and the
      // decode on one stream, so no synchronize is needed between them.
      decoded.push_back(simpatico::decompress(compressed, selection, stream, mr));
      row_filtered_batch = false;
      continue;
    }

    // The fetch above only ENQUEUED its H2D copies on `stream`, and decompress_chunk decodes on
    // its own stream pool; sync so no pool stream reads bytes that have not landed yet.
    stream.synchronize();
    auto result = sirius::decompress_chunk(compressed, selection, chunk_scan.get(), stream, mr);
    _pushdown_offered.fetch_add(1, std::memory_order_relaxed);
    if (result.outcome.row_filtered) {
      _pushdown_row_filtered.fetch_add(1, std::memory_order_relaxed);
    } else {
      // Anything less than the whole filter leaves rows the predicate rejects in this chunk, and
      // the batch is one table: one such chunk makes the post-decode filter mandatory for all of
      // it. Re-checking the conjuncts a compacted chunk already applied is idempotent.
      row_filtered_batch = false;
    }
    if (result.outcome.selection_unprofitable) {
      _pushdown_unprofitable.fetch_add(1, std::memory_order_relaxed);
      _row_selection_dropped.store(true, std::memory_order_relaxed);
    }
    if (!result.outcome.predicate_columns.empty()) {
      // A column answered in place arrives as the BOOL8 answer instead of its values, which this
      // source cannot emit: it declares no projection pushdown, so every decoded column is an
      // output column. The request is built without equality sets precisely so this cannot
      // happen -- reaching it means the request grew a source whose output shape nothing here
      // handles, and serving the batch would hand the plan a boolean where it expects values.
      throw std::runtime_error(
        "[simpatico_gpu_ingestible] '" + split.path + "' chunk " +
        std::to_string(split.chunk_ids[i]) +
        " decoded a column as a predicate answer, which a .hpln scan never asks for");
    }
    // Re-point the decoded buffers onto `stream`: they were produced on pool streams, while the
    // concatenate below and everything downstream are ordered by `stream`.
    decoded.push_back(sirius::rebind_table_stream(std::move(result.table), stream));
  }

  // The fetches above only ENQUEUED their H2D copies, and the pinned staging blobs die with this
  // scope -- a host free is not stream-ordered, so the bytes have to have landed first.
  stream.synchronize();

  // One table per chunk, concatenated in chunk-id order. The batch is a contiguous run of the
  // file, so concatenating in any other order would silently reshuffle its rows.
  std::unique_ptr<cudf::table> table;
  if (decoded.size() == 1) {
    table = std::move(decoded.front());
  } else {
    std::vector<cudf::table_view> views;
    views.reserve(decoded.size());
    for (auto const& t : decoded) {
      views.push_back(t->view());
    }
    table = cudf::concatenate(views, stream, mr);
  }

  SIRIUS_LOG_DEBUG(
    "[simpatico_gpu_ingestible] '{}' chunks={} decoded rows={} cols={} row_filtered={}",
    split.path,
    split.chunk_ids.size(),
    table->num_rows(),
    table->num_columns(),
    row_filtered_batch);

  // ROW_FILTERED only when EVERY chunk of this batch came back carrying the whole filter --
  // decompress_chunk sets row_filtered only where the request covered the filter with no conjunct
  // dropped and the compaction actually applied. post_filter_and_project then skips the filter
  // for this batch, which is why anything weaker has to read as "not filtered": the scan is the
  // only thing left that applies the predicate at all.
  return filtered_table{
    .table = owning_table_view{std::move(table)},
    .state = row_filtered_batch ? filter_state::ROW_FILTERED : filter_state::UNFILTERED};
}

//===----------------------------------------------------------------------===//
// batch_coalescer
//===----------------------------------------------------------------------===//
std::unique_ptr<batch_coalescer> simpatico_gpu_ingestible::create_batch_coalescer() const
{
  return std::make_unique<simpatico_batch_coalescer>(_info->approximate_batch_size);
}

//===----------------------------------------------------------------------===//
// post_filter_and_project
//===----------------------------------------------------------------------===//
std::unique_ptr<cudf::table> simpatico_gpu_ingestible::post_filter_and_project(
  filtered_table&& input,
  cucascade::memory::memory_space const& mem_space,
  rmm::cuda_stream_view stream,
  bool like_swar_fastpath,
  std::shared_ptr<const like_multiliteral_cache> like_cache,
  std::unique_ptr<cudf::column>* /*survivors*/,
  std::span<std::size_t const> elided)
{
  rmm::device_async_resource_ref mr(mem_space.get_default_allocator());
  auto table = std::move(input.table);

  // The zone maps BOUNDED the rows -- they did not test them, and a surviving group still holds
  // rows the predicate rejects. This is where the filter is actually applied, and it must be:
  // read_simpatico declares filter_pushdown, so DuckDB deleted the predicate from the plan and
  // nothing above the scan re-checks it.
  //
  // Every column the decode emitted is an output column (the source declares no projection
  // pushdown, so the scan is full-width and a projection sits above it), which is why the filter's
  // batch positions are the file's column order and no projection is folded in here.
  if (_filter_expression && input.state != filter_state::ROW_FILTERED &&
      input.state != filter_state::ROW_FILTERED_AND_PROJECTED) {
    auto filter_ast = sirius::ast::from_duckdb(*_filter_expression);
    sirius::expression_evaluator exec(filter_ast.get(),
                                      mr,
                                      stream,
                                      strategy_from_config(),
                                      sirius::expression_evaluator::default_min_ast_size,
                                      like_swar_fastpath,
                                      std::move(like_cache));
    auto filtered = owning_table_view{exec.select(table.view())};
    // The select only ENQUEUED its reads; record before the input's read-lock owner is dropped.
    table.record_reader_event(stream);
    table = std::move(filtered);
  }

  // Drop the positions the caller is about to overwrite. `survivors` stays untouched, which is
  // what can_report_survivors() == false promises.
  if (auto const kept =
        kept_positions(static_cast<std::size_t>(table.view().num_columns()), elided);
      !kept.empty()) {
    table.select_columns(kept);
  }
  return table.release(stream, mr);
}

//===----------------------------------------------------------------------===//
// materialized_column_order
//===----------------------------------------------------------------------===//
std::vector<std::size_t> simpatico_gpu_ingestible::materialized_column_order() const
{
  // The decode emits exactly the requested columns, in the order requested, so the selection is
  // already the answer.
  return _info->column_ids;
}

std::shared_ptr<simpatico_gpu_ingestible> make_ingestible(
  std::unique_ptr<simpatico_ingestible_table_info> info)
{
  return std::make_shared<simpatico_gpu_ingestible>(std::move(info));
}

}  // namespace sirius::op::scan
