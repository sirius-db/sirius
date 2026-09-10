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
#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>

// cucascade
#include <cucascade/memory/memory_space.hpp>

// simpatico
#include <api/compressed_table_io.hpp>
#include <api/simpatico_codegen.hpp>

// standard library
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace sirius::op::scan {

namespace {

/// Decoded footprint of the whole file, for the reservation.
///
/// Exact for fixed-width columns: the header carries the row count and the decoded type, and
/// nothing is projected away. A variable-width column has no such answer without decoding it, so
/// it contributes only its offsets -- a floor, not an estimate. A file with strings therefore
/// under-reserves until the format records decoded sizes the way parquet's SizeStatistics do.
std::size_t estimate_decoded_bytes(simpatico_ingestible_table_info const& info)
{
  auto const rows   = static_cast<std::size_t>(std::max<std::int64_t>(info.num_rows, 0));
  std::size_t total = 0;
  for (auto const& dtype : info.physical_types) {
    total += rows * (cudf::is_fixed_width(dtype) ? cudf::size_of(dtype) : sizeof(std::int32_t));
  }
  return total;
}

/// Pass every split straight through.
///
/// A coalescer earns its keep by bundling many small metadata units into one decode-sized batch;
/// a file that holds one chunk offers exactly one unit, and splitting or merging it would change
/// what a batch means without changing what is read.
class simpatico_batch_coalescer : public batch_coalescer {
 public:
  std::vector<std::unique_ptr<scan_info>> push(std::unique_ptr<scan_info> info) override
  {
    std::vector<std::unique_ptr<scan_info>> out;
    if (info) { out.push_back(std::move(info)); }
    return out;
  }

  std::vector<std::unique_ptr<scan_info>> flush() override { return {}; }
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
  info->host_space          = &host_space;
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
}

simpatico_gpu_ingestible::~simpatico_gpu_ingestible() = default;

//===----------------------------------------------------------------------===//
// split-provider interface
//===----------------------------------------------------------------------===//
bool simpatico_gpu_ingestible::has_processed_all_metadata() const
{
  return _split_claimed.load(std::memory_order_relaxed);
}

simpatico_gpu_ingestible::metadata_scan_task_t simpatico_gpu_ingestible::next_split_provider(
  io::ioctx_resolver /*resolve*/)
{
  // Exchange rather than load-then-store: several dispatcher threads may reach here and exactly
  // one of them owns the file's single split.
  if (_split_claimed.exchange(true, std::memory_order_relaxed)) { return nullptr; }

  // The resolver is unused because the ingest reads the file through the filesystem rather than
  // through an io_context datasource, which is also why a remote path cannot be read yet.
  return [this]() -> std::unique_ptr<scan_info> {
    auto split           = std::make_unique<simpatico_scan_info>();
    split->path          = _info->resolved_file_paths.front();
    split->decoded_bytes = estimate_decoded_bytes(*_info);
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

  // Stage byte-for-byte into pinned host memory, then fetch the leaf buffers the reader asks for
  // straight out of those blocks. Nothing is decoded on the way in, so a served table costs one
  // file read plus one decode rather than a decode and a re-compress.
  auto ingested = sirius::read_hpln_into_pinned(split.path, *_info->host_space);

  simpatico::payload_fetch_fn fetch =
    [&ingested](std::uint64_t off, std::size_t sz, void* dst, rmm::cuda_stream_view s) {
      copy_pinned_blocks_to_device(*ingested.blob->payload, off, dst, sz, s);
    };

  rmm::device_async_resource_ref mr(mem_space.get_default_allocator());
  std::string read_error;
  auto const compressed = simpatico::read_compressed_table_from_memory(
    ingested.blob->header, fetch, stream, mr, &read_error);
  if (!read_error.empty()) {
    throw std::runtime_error("[simpatico_gpu_ingestible] '" + split.path +
                             "' could not be reconstructed: " + read_error);
  }

  // The fetch above only ENQUEUED its H2D copies, and the pinned staging blob dies with this
  // scope -- a host free is not stream-ordered, so the bytes have to have landed first.
  stream.synchronize();

  // Decode on the caller's stream rather than on the decode thread pool. The pool overlaps
  // columns, but its buffers are freed on pool streams and would have to be re-bound to `stream`
  // before anything downstream may read them; one chunk per file is not enough work to be worth
  // that. It also keeps the fetch above and the decode on one stream, so no synchronize is needed
  // between them.
  std::vector<std::size_t> selection(static_cast<std::size_t>(compressed.num_columns()));
  std::iota(selection.begin(), selection.end(), std::size_t{0});
  auto table = simpatico::decompress(compressed, selection, stream, mr);

  SIRIUS_LOG_DEBUG("[simpatico_gpu_ingestible] '{}' decoded rows={} cols={}",
                   split.path,
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
  return std::make_unique<simpatico_batch_coalescer>();
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
  // Every column of the file is an output column and there is no filter to apply, so the only work
  // left is dropping the positions the caller is about to overwrite. `survivors` stays untouched,
  // which is what can_report_survivors() == false promises.
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
  // The decode emits the file's columns in file order and nothing is projected away, so storage
  // index and emission position are the same number.
  std::vector<std::size_t> order(_info->names.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  return order;
}

std::shared_ptr<simpatico_gpu_ingestible> make_ingestible(
  std::unique_ptr<simpatico_ingestible_table_info> info)
{
  return std::make_shared<simpatico_gpu_ingestible>(std::move(info));
}

}  // namespace sirius::op::scan
