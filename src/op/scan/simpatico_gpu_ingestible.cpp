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
#include <compression/simpatico_file_ingest.hpp>
#include <log/logging.hpp>
#include <op/scan/owning_table_view.hpp>
#include <op/scan/simpatico_gpu_ingestible.hpp>

// cudf
#include <cudf/concatenate.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>

// cucascade
#include <cucascade/memory/memory_space.hpp>

// simpatico
#include <api/compressed_table_io.hpp>
#include <api/simpatico_codegen.hpp>

// standard library
#include <algorithm>
#include <limits>
#include <numeric>
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
    // see them out of order and must not preserve that.
    _current->chunk_ids.insert(
      _current->chunk_ids.end(), split->chunk_ids.begin(), split->chunk_ids.end());
    std::sort(_current->chunk_ids.begin(), _current->chunk_ids.end());
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

}  // namespace

//===----------------------------------------------------------------------===//
// bind
//===----------------------------------------------------------------------===//
std::unique_ptr<simpatico_ingestible_table_info> bind_simpatico_file(
  std::string const& path, cucascade::memory::memory_space& host_space)
{
  auto schema = sirius::read_hpln_schema(path);

  auto info                 = std::make_unique<simpatico_ingestible_table_info>();
  info->resolved_file_paths = {path};
  info->names               = std::move(schema.names);
  info->types               = std::move(schema.types);
  info->physical_types      = std::move(schema.physical_types);
  info->num_rows            = schema.num_rows;
  info->chunk_rows          = std::move(schema.chunk_rows);
  info->host_space          = &host_space;
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
}

simpatico_gpu_ingestible::~simpatico_gpu_ingestible() = default;

//===----------------------------------------------------------------------===//
// split-provider interface
//===----------------------------------------------------------------------===//
bool simpatico_gpu_ingestible::has_processed_all_metadata() const
{
  return _next_chunk.load(std::memory_order_relaxed) >= _info->chunk_rows.size();
}

simpatico_gpu_ingestible::metadata_scan_task_t simpatico_gpu_ingestible::next_split_provider(
  io::ioctx_resolver /*resolve*/)
{
  // fetch_add rather than load-then-store: several dispatcher threads may reach here, and two of
  // them reading the same cursor would emit one chunk twice and skip another -- which lands as a
  // wrong answer with a right-looking row count, not as a failure.
  auto const chunk = _next_chunk.fetch_add(1, std::memory_order_relaxed);
  if (chunk >= _info->chunk_rows.size()) { return nullptr; }

  // The resolver is unused because the ingest reads the file through the filesystem rather than
  // through an io_context datasource, which is also why a remote path cannot be read yet.
  return [this, chunk]() -> std::unique_ptr<scan_info> {
    auto split           = std::make_unique<simpatico_scan_info>();
    split->path          = _info->resolved_file_paths.front();
    split->chunk_ids     = {chunk};
    split->num_rows      = _info->chunk_rows[chunk];
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
  auto ingested =
    sirius::read_hpln_chunks_into_pinned(split.path, *_info->host_space, split.chunk_ids);

  rmm::device_async_resource_ref mr(mem_space.get_default_allocator());
  std::vector<std::unique_ptr<cudf::table>> decoded;
  decoded.reserve(ingested.size());
  for (std::size_t i = 0; i < ingested.size(); ++i) {
    auto const& blob = *ingested[i].blob;
    simpatico::payload_fetch_fn fetch =
      [&blob](std::uint64_t off, std::size_t sz, void* dst, rmm::cuda_stream_view s) {
        copy_pinned_blocks_to_device(*blob.payload, off, dst, sz, s);
      };

    std::string read_error;
    auto const compressed =
      simpatico::read_compressed_table_from_memory(blob.header, fetch, stream, mr, &read_error);
    if (!read_error.empty()) {
      throw std::runtime_error("[simpatico_gpu_ingestible] '" + split.path + "' chunk " +
                               std::to_string(split.chunk_ids[i]) +
                               " could not be reconstructed: " + read_error);
    }

    // Decode on the caller's stream rather than on the decode thread pool. The pool overlaps
    // columns, but its buffers are freed on pool streams and would have to be re-bound to `stream`
    // before anything downstream may read them. It also keeps the fetch above and the decode on
    // one stream, so no synchronize is needed between them.
    decoded.push_back(simpatico::decompress(compressed, _info->column_ids, stream, mr));
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

  SIRIUS_LOG_DEBUG("[simpatico_gpu_ingestible] '{}' chunks={} decoded rows={} cols={}",
                   split.path,
                   split.chunk_ids.size(),
                   table->num_rows(),
                   table->num_columns());

  return filtered_table{.table = owning_table_view{std::move(table)},
                        .state = filter_state::UNFILTERED};
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
  bool /*like_swar_fastpath*/,
  std::shared_ptr<const like_multiliteral_cache> /*like_cache*/,
  std::unique_ptr<cudf::column>* /*survivors*/,
  std::span<std::size_t const> elided)
{
  // Every column the decode emitted is an output column and there is no filter to apply, so the
  // only work left is dropping the positions the caller is about to overwrite. `survivors` stays
  // untouched, which is what can_report_survivors() == false promises.
  rmm::device_async_resource_ref mr(mem_space.get_default_allocator());
  auto table = std::move(input.table);
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
