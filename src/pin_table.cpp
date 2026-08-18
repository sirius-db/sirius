/*
 * Copyright 2025, Sirius Contributors.
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

#include "pin_table.hpp"

#include "compression/compressed_representation.hpp"
#include "compression/device_compressed_blob.hpp"
#include "cudf/cudf_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "helper/numeric_narrowing.hpp"
#include "helper/type_conversions.hpp"
#include "io/io_context.hpp"
#include "log/logging.hpp"
#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "op/scan/gpu_ingestible.hpp"
#include "scan_manager/pinned_chunk_stats.hpp"
#include "scan_manager/round_robin_strategy.hpp"

#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <api/compressed_table_io.hpp>
#include <api/simpatico_codegen.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace duckdb {

namespace {

std::mutex& recorded_calls_mutex()
{
  static std::mutex m;
  return m;
}

std::vector<PinTableArgs>& recorded_calls_storage()
{
  static std::vector<PinTableArgs> calls;
  return calls;
}

std::vector<std::string>& recorded_unpin_calls_storage()
{
  static std::vector<std::string> calls;
  return calls;
}

}  // namespace

void pin_table_to(const PinTableArgs& args)
{
  std::lock_guard<std::mutex> lock(recorded_calls_mutex());
  recorded_calls_storage().push_back(args);
}

void unpin_table_to(const std::string& name)
{
  std::lock_guard<std::mutex> lock(recorded_calls_mutex());
  recorded_unpin_calls_storage().push_back(name);
}

namespace pin_table_testing {

const std::vector<PinTableArgs>& recorded_calls() { return recorded_calls_storage(); }

void clear_recorded_calls()
{
  std::lock_guard<std::mutex> lock(recorded_calls_mutex());
  recorded_calls_storage().clear();
}

const std::vector<std::string>& recorded_unpin_calls() { return recorded_unpin_calls_storage(); }

void clear_recorded_unpin_calls()
{
  std::lock_guard<std::mutex> lock(recorded_calls_mutex());
  recorded_unpin_calls_storage().clear();
}

}  // namespace pin_table_testing

}  // namespace duckdb

namespace sirius {

// A coalescer change that splits a row group across batches or reorders batches
// must fail the pin loudly here instead of silently misaligning the per-chunk
// visibility masks built over these boundaries (full contract in pin_table.hpp).
void validate_duckdb_pin_chunk(const op::scan::scan_info& batch,
                               std::size_t chunk_rows,
                               std::size_t rows_before_chunk)
{
  auto const* native = dynamic_cast<const op::scan::duckdb_native_scan_info*>(&batch);
  if (native == nullptr) { return; }

  std::size_t next_row = rows_before_chunk;
  for (auto const& rg : native->row_groups) {
    if (static_cast<std::size_t>(rg.row_group_start) != next_row) {
      throw std::runtime_error(
        "[pin_table] duckdb-native batch breaks the whole-row-group contiguity invariant: row "
        "group " +
        std::to_string(rg.row_group_index) + " starts at row " +
        std::to_string(rg.row_group_start) + " but the pin has materialized rows [0, " +
        std::to_string(next_row) + ")");
    }
    next_row += rg.row_count;
  }
  if (next_row - rows_before_chunk != chunk_rows) {
    throw std::runtime_error(
      "[pin_table] duckdb-native chunk decoded " + std::to_string(chunk_rows) +
      " rows but its row-group metadata covers " + std::to_string(next_row - rows_before_chunk));
  }
}

cudf::data_type pin_native_type(cudf::data_type decoded_type,
                                duckdb::LogicalType const* declared_type)
{
  if (declared_type == nullptr) { return decoded_type; }
  auto const native = sirius::try_get_cudf_type(sirius::from_duckdb(*declared_type));
  return native.value_or(decoded_type);
}

namespace {

/// Per-materialized-batch sink: receives one GPU-resident table, its GPU placement, the
// stream it was decoded on, the chunk's stored-column metadata, and the chunk's zone-map
// capture (empty when capture is off).
/// The driver does NOT synchronize — the sink owns the sync, because the host-streaming path
/// must synchronize while the GPU table (and the gpu_table_representation wrapping it) is
/// still alive, after the D2H conversion has been enqueued on the same stream.
using pin_batch_sink =
  std::function<void(std::unique_ptr<cudf::table>,
                     cucascade::memory::memory_space* target,
                     rmm::cuda_stream_view stream,
                     std::vector<pinned_column_storage_meta> column_storage,
                     std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>> chunk_stats)>;

struct narrowed_pin_chunk {
  std::unique_ptr<cudf::table> table;
  std::vector<bool> columns;
};

narrowed_pin_chunk narrow_pin_chunk(std::unique_ptr<cudf::table> table,
                                    duckdb::vector<duckdb::LogicalType> const& column_types,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  if (column_types.size() != static_cast<std::size_t>(table->num_columns())) {
    throw std::invalid_argument(
      "[pin_table] compressed materialization requires one logical type per cached column");
  }

  std::vector<bool> narrowed_columns(static_cast<std::size_t>(table->num_columns()), false);
  auto columns = table->release();
  for (std::size_t column_idx = 0; column_idx < columns.size(); ++column_idx) {
    auto const logical = sirius::from_duckdb(column_types[column_idx]);
    auto const range =
      compute_exact_numeric_range(columns[column_idx]->view(), logical, stream, mr);
    if (!range) { continue; }
    auto target = choose_narrow_physical_type(logical, *range);
    if (!target || columns[column_idx]->type() == *target) { continue; }
    auto const actual   = columns[column_idx]->type();
    columns[column_idx] = cast_through_rep(columns[column_idx]->view(), *target, stream, mr);
    narrowed_columns[column_idx] = true;
    SIRIUS_LOG_DEBUG("[compressed_materialization] pin column {} narrowed: {} -> {}",
                     column_idx,
                     cudf::type_to_name(actual),
                     cudf::type_to_name(*target));
  }
  return {std::make_unique<cudf::table>(std::move(columns)), std::move(narrowed_columns)};
}

/// Shared driver behind @ref materialize_all_batches, @ref materialize_pin_to_host, and
/// @ref materialize_all_batches_compressed: walk the
/// ingestible's metadata + batch coalescer to completion, materialize each emitted batch onto a
/// round-robin GPU, and hand it to @p on_batch. The single-threaded, deterministic round-robin
/// placement means re-pinning the same source yields identical placement (required by
/// insert_pinned_entry's merge path on the GPU tier) and bounds peak GPU residency to ~one batch
/// (the host tier frees each table in @p on_batch before the next is materialized).
void materialize_pin_batches(op::scan::gpu_ingestible& ingestible,
                             std::span<cucascade::memory::memory_space* const> gpu_spaces,
                             io::sirius_ioctx& io_ctx,
                             duckdb::vector<duckdb::LogicalType> const& pinned_column_types,
                             pin_materialization_options options,
                             const pin_batch_sink& on_batch)
{
  if (gpu_spaces.empty()) {
    throw std::invalid_argument("[materialize_pin_batches] gpu_spaces must be non-empty");
  }

  // GPU placement via the same strategy the query path uses
  // (sirius_scan_manager::prepare_for_query). A fresh strategy per call starts its
  // cursor at 0, so re-pinning the same source yields identical placement — required
  // by insert_pinned_entry's merge path. The strategy hands out device ids; map each
  // back to the memory_space to materialize into.
  std::vector<int> device_ids;
  std::unordered_map<int, cucascade::memory::memory_space*> space_by_device;
  device_ids.reserve(gpu_spaces.size());
  for (auto* sp : gpu_spaces) {
    device_ids.push_back(sp->get_device_id());
    space_by_device.emplace(sp->get_device_id(), sp);
  }
  scan_manager::round_robin_strategy placement(std::move(device_ids));

  // next_split_provider takes the io_ctx by shared_ptr; sirius_ioctx derives
  // std::enable_shared_from_this and the scan manager owns it via a shared_ptr, so
  // this hands the metadata reads a valid owning reference for the read's duration.
  auto io_ctx_sp = io_ctx.shared_from_this();

  auto coalescer = ingestible.create_batch_coalescer();

  // Rows materialized so far, in emission order — feeds the chunk-contiguity
  // validation (duckdb-native pins only).
  std::size_t rows_materialized = 0;

  // Materialize one coalesced batch into a GPU-resident cudf::table and hand it to on_batch
  // together with its GPU placement + the decode stream. Mirrors
  // load_balancing_scan_batch_coalescer::process_provider_inputs, minus the connector/balancer
  // and the post-decode step: there is no downstream scan operator at pin time, so the
  // unfiltered table is delivered to the sink directly. The device guard stays in scope across
  // the on_batch call, so the sink runs on the correct device.
  auto handle_batch = [&](std::unique_ptr<op::scan::scan_info> batch) {
    if (!batch) { return; }
    // round_robin_strategy ignores pipeline_id/data; it returns the next device id
    // from its cursor. gpu_spaces is non-empty (checked above), so the strategy always
    // yields a device. space_by_device round-trips it to the materialization target.
    int const gpu_id =
      placement.get_next_gpu(/*pipeline_id=*/0, /*data=*/nullptr, /*hint=*/{}).value();
    auto* target = space_by_device.at(gpu_id);
    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{gpu_id}};
    // Borrow a real (non-default) stream from the target memory_space's pool: the
    // duckdb-native decoder's cudaMemcpyBatchAsync rejects the legacy default
    // stream. The pool owns the stream for the memory_space's lifetime (which
    // outlives the pinned data), so the materialized buffers' deallocation stream
    // stays valid after this call.
    auto stream = target->acquire_stream();

    // A pin caches the table UNFILTERED: it has no row filter (no WHERE), and the
    // ingestible's projection_ids are identity into column_ids, so the reader already
    // emits exactly the pinned columns in column_ids order. materialize_metadata_to_table
    // thus yields the cache-ready table directly — no scan_operator_input wrapper and no
    // post_filter_and_project (which would only apply a row filter or re-project to a
    // query's output layout, neither of which a pin needs). `batch` is borrowed by const
    // ref and outlives the call.
    auto materialized     = ingestible.materialize_metadata_to_table(*batch, *target, stream);
    auto tbl              = materialized.table.release(stream, target->get_default_allocator());
    auto const chunk_rows = static_cast<std::size_t>(tbl->num_rows());
    validate_duckdb_pin_chunk(*batch, chunk_rows, rows_materialized);
    rows_materialized += chunk_rows;
    // Zone-map capture, while the decode device guard + stream are active and the
    // GPU table is alive. The scalar downloads inside are synchronous, so the
    // capture is complete before the sink converts or frees the table. Clean
    // unsupported-metadata cases degrade to null cells inside; CUDA errors
    // propagate and abort the pin like any other pin-time CUDA failure.
    std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>> chunk_stats;
    std::vector<bool> narrowed_columns(static_cast<std::size_t>(tbl->num_columns()), false);
    if (options.capture_chunk_stats && !pinned_column_types.empty()) {
      chunk_stats = scan_manager::compute_pinned_chunk_stats(
        tbl->view(), pinned_column_types, stream, target->get_default_allocator());
    }
    // Record declared-native identity before narrowing; decoder type is only the fallback when no
    // declared mapping is available.
    std::vector<cudf::data_type> native_types;
    native_types.reserve(static_cast<std::size_t>(tbl->num_columns()));
    for (cudf::size_type i = 0; i < tbl->num_columns(); ++i) {
      auto const idx = static_cast<std::size_t>(i);
      native_types.push_back(
        pin_native_type(tbl->get_column(i).type(),
                        idx < pinned_column_types.size() ? &pinned_column_types[idx] : nullptr));
    }
    if (options.enable_compressed_materialization) {
      auto narrowed = narrow_pin_chunk(
        std::move(tbl), pinned_column_types, stream, target->get_default_allocator());
      tbl              = std::move(narrowed.table);
      narrowed_columns = std::move(narrowed.columns);
    }
    // Record the chunk's stored-column metadata from the exact table the sink stores — for a
    // chunk the sink compresses, these are the types compress_with_plan receives and
    // decompression reproduces.
    std::vector<pinned_column_storage_meta> column_storage;
    column_storage.reserve(static_cast<std::size_t>(tbl->num_columns()));
    for (cudf::size_type i = 0; i < tbl->num_columns(); ++i) {
      column_storage.push_back({tbl->get_column(i).type(),
                                narrowed_columns[static_cast<std::size_t>(i)],
                                native_types[static_cast<std::size_t>(i)]});
    }
    on_batch(std::move(tbl), target, stream, std::move(column_storage), std::move(chunk_stats));
  };

  while (!ingestible.has_processed_all_metadata()) {
    // A pin reads its fixed ingestible on the one ioctx it was given; route every file to it.
    auto task = ingestible.next_split_provider([io_ctx_sp](std::string_view) { return io_ctx_sp; });
    if (!task) { continue; }
    auto info = task();
    if (!info) { continue; }
    for (auto& b : coalescer->push(std::move(info))) {
      handle_batch(std::move(b));
    }
  }
  for (auto& b : coalescer->flush()) {
    handle_batch(std::move(b));
  }
}

// Streams for cross-column encode parallelism, one pool per thread and device —
// the same accessor the decode path uses, so a thread that both pins and
// decodes holds one pool rather than two.
simpatico::stream_pool& compress_pool() { return simpatico::thread_device_stream_pool(4); }

// Diagnostic for a compression failure inside a pin sink, shared by both drivers. Reports the real
// blast radius: the sink latches compression off for every remaining chunk of the pin, not only the
// failed chunk. When that chunk narrowed carriers before compression, the diagnostic names those
// columns: a plan block with a width-explicit op (@c bitextract / @c bitjoin packs a fixed total
// field width) is authored against the native element width and cannot encode a narrowed column,
// which is the one failure mode narrowing itself can cause.
std::string compression_failure_warning(std::string_view what,
                                        compression_pin_config const& compression,
                                        std::span<pinned_column_storage_meta const> chunk_storage)
{
  std::string message{"compression failed: "};
  message += what;
  message +=
    "; pinning the remainder of this table uncompressed (the failure latches compression "
    "off for every later chunk)";

  std::string narrowed;
  for (std::size_t i = 0; i < chunk_storage.size(); ++i) {
    if (!chunk_storage[i].narrowed) { continue; }
    if (!narrowed.empty()) { narrowed += ", "; }
    narrowed +=
      i < compression.column_names.size() ? compression.column_names[i] : std::to_string(i);
  }
  if (!narrowed.empty()) {
    message += ". This pin narrowed carriers before compression (" + narrowed +
               "); a plan block with a width-explicit op (bitextract/bitjoin) is authored against "
               "the native element width and cannot encode a narrowed column";
  }
  return message;
}

/// Shared compress step for the host and device pin drivers: compress @p tbl per
/// @p compression on @p stream, and when the batch qualifies (compression on and
/// >= the size threshold) AND the compressed footprint saves enough (<=
/// max_compressed_fraction of the original), invoke @p stage to copy the compressed
/// buffers into the caller's tier storage. Returns true iff @p stage ran (the batch
/// was pinned compressed); false means pin uncompressed. Throws on a compression /
/// encode failure so the caller can latch its per-chunk fallback.
///
/// @p stage runs while the compressed_table (which owns the device payload buffers
/// enumerated in @c buffers) and the compress stream pool are still alive, so it
/// must copy the buffers out and synchronize @p stream before returning. It is
/// called as stage(ct&&, header&&, buffers, payload_bytes, uncompressed_bytes,
/// column_sizes).
template <typename StageFn>
bool compress_and_stage_batch(cudf::table const& tbl,
                              compression_pin_config const& compression,
                              rmm::cuda_stream_view stream,
                              std::string_view log_tag,
                              StageFn&& stage)
{
  nvtx3::scoped_range nvtx_range{"sirius::pin::compress_and_stage"};
  if (tbl.num_columns() == 0) { return false; }
  // Total device footprint of the batch (includes string chars/offsets and null
  // masks), so string columns count toward the threshold.
  const std::size_t uncompressed_bytes = tbl.alloc_size();
  if (uncompressed_bytes < compression.min_batch_size_bytes) { return false; }

  // `tbl` was decoded on `stream` (the caller's materialize stream). The pool
  // streams are NOT ordered after `stream`, so synchronize first to ensure the
  // table is fully resident before the pool streams read it — mirrors the
  // parallel decompress path in compression_converters.cpp.
  stream.synchronize();
  auto ct = simpatico::compress_with_plan(tbl.view(),
                                          compression.plan_dsl,
                                          compress_pool(),
                                          rmm::mr::get_current_device_resource_ref(),
                                          compression.column_names);

  // Build the structural header and enumerate the payload buffers (no bytes copied yet).
  std::vector<std::uint8_t> header;
  std::vector<simpatico::payload_buffer_ref> buffers;
  std::uint64_t payload_bytes = 0;
  const std::string hdr_err =
    simpatico::build_compressed_table_header(ct, header, buffers, payload_bytes, stream);
  if (!hdr_err.empty()) { throw std::runtime_error("build_compressed_table_header: " + hdr_err); }

  // Keep the compressed form only if it saves enough: compare the total compressed
  // footprint (header + payload) against the batch's original device size.
  const std::size_t compressed_bytes = header.size() + payload_bytes;
  if (uncompressed_bytes > 0 &&
      static_cast<double>(compressed_bytes) >
        compression.max_compressed_fraction * static_cast<double>(uncompressed_bytes)) {
    SIRIUS_LOG_DEBUG("[{}] compressed {}B > {:.0f}% of {}B original; pinning uncompressed",
                     log_tag,
                     compressed_bytes,
                     compression.max_compressed_fraction * 100.0,
                     uncompressed_bytes);
    return false;
  }

  // Exact per-column footprints, so a projection over this chunk can size itself by
  // summing the columns it selects rather than scaling the whole-chunk totals.
  auto column_sizes = std::make_shared<sirius::per_column_byte_sizes>();
  column_sizes->compressed.assign(static_cast<std::size_t>(tbl.num_columns()), 0);
  column_sizes->uncompressed.reserve(static_cast<std::size_t>(tbl.num_columns()));
  for (cudf::size_type i = 0; i < tbl.num_columns(); ++i) {
    column_sizes->uncompressed.push_back(tbl.get_column(i).alloc_size());
  }
  for (auto const& b : buffers) {
    if (b.column_index < column_sizes->compressed.size()) {
      column_sizes->compressed[b.column_index] += b.size_bytes;
    }
  }

  {
    nvtx3::scoped_range stage_range{"sirius::compression::stage_payload"};
    stage(std::move(ct),
          std::move(header),
          buffers,
          payload_bytes,
          uncompressed_bytes,
          std::move(column_sizes));
  }
  return true;
}

}  // namespace

materialized_pin materialize_all_batches(
  op::scan::gpu_ingestible& ingestible,
  std::span<cucascade::memory::memory_space* const> gpu_spaces,
  io::sirius_ioctx& io_ctx,
  duckdb::vector<duckdb::LogicalType> const& pinned_column_types,
  pin_materialization_options options)
{
  materialized_pin out;
  materialize_pin_batches(
    ingestible,
    gpu_spaces,
    io_ctx,
    pinned_column_types,
    options,
    [&](std::unique_ptr<cudf::table> tbl,
        cucascade::memory::memory_space* target,
        rmm::cuda_stream_view stream,
        std::vector<pinned_column_storage_meta> column_storage,
        std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>> chunk_stats) {
      // Cached GPU batches are stored with a null writer stream, so the data
      // must be fully resident before it can be served or host-converted.
      stream.synchronize();
      out.base_row_count_per_chunk.push_back(static_cast<std::size_t>(tbl->num_rows()));
      out.tables.emplace_back(std::move(tbl));
      out.chunk_memory_spaces.push_back(target);
      out.column_storage.emplace_back(std::move(column_storage));
      if (!chunk_stats.empty()) { out.chunk_stats.emplace_back(std::move(chunk_stats)); }
    });
  return out;
}

host_pin_result materialize_pin_to_host(
  op::scan::gpu_ingestible& ingestible,
  std::span<cucascade::memory::memory_space* const> gpu_spaces,
  const std::unordered_map<int, cucascade::memory::memory_space*>& host_space_by_gpu,
  io::sirius_ioctx& io_ctx,
  duckdb::vector<duckdb::LogicalType> const& pinned_column_types,
  compression_pin_config const& compression,
  pin_materialization_options options)
{
  auto& registry = converter_registry::get();
  host_pin_result out;
  bool compression_failed = false;

  materialize_pin_batches(
    ingestible,
    gpu_spaces,
    io_ctx,
    pinned_column_types,
    options,
    [&](std::unique_ptr<cudf::table> tbl,
        cucascade::memory::memory_space* src_space,
        rmm::cuda_stream_view stream,
        std::vector<pinned_column_storage_meta> column_storage,
        std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>> chunk_stats) {
      auto* target_host_space    = host_space_by_gpu.at(src_space->get_device_id());
      bool compressed_this_chunk = false;
      out.base_row_count_per_chunk.push_back(static_cast<std::size_t>(tbl->num_rows()));
      out.column_storage.emplace_back(std::move(column_storage));
      if (!chunk_stats.empty()) { out.chunk_stats.emplace_back(std::move(chunk_stats)); }

      if (compression.enabled && !compression_failed && tbl && !compression.plan_dsl.empty()) {
        try {
          compressed_this_chunk = compress_and_stage_batch(
            *tbl,
            compression,
            stream,
            "materialize_pin_to_host",
            [&](simpatico::compressed_table&& /*ct*/,
                std::vector<std::uint8_t>&& header,
                std::vector<simpatico::payload_buffer_ref> const& buffers,
                std::uint64_t payload_bytes,
                std::size_t uncompressed_bytes,
                std::shared_ptr<const sirius::per_column_byte_sizes> column_sizes) {
              // Allocate the pinned payload from the target host space's chunked
              // pool (the same tracked pool the uncompressed path uses), reserved
              // against the host budget so the pinned footprint is accounted, then
              // stage every compressed buffer device->pinned. Sync before `ct`
              // (which owns the device buffers) leaves scope.
              auto* host_mr =
                target_host_space->get_memory_resource_of<cucascade::memory::Tier::HOST>();
              if (host_mr == nullptr) {
                throw std::runtime_error(
                  "target host space has no fixed_size_host_memory_resource");
              }
              auto payload_res = target_host_space->make_reservation_or_null(payload_bytes);
              auto blob        = std::make_shared<sirius::pinned_compressed_blob>();
              blob->header     = std::move(header);
              blob->payload = host_mr->allocate_multiple_blocks(payload_bytes, payload_res.get());
              blob->payload_bytes = payload_bytes;
              for (auto const& b : buffers) {
                if (b.size_bytes > 0 && b.device_ptr != nullptr) {
                  sirius::copy_device_to_pinned_blocks(b.device_ptr,
                                                       *blob->payload,
                                                       b.offset,
                                                       static_cast<std::size_t>(b.size_bytes),
                                                       stream);
                }
              }
              stream.synchronize();

              out.chunks.emplace_back(std::make_shared<sirius::compressed_host_representation>(
                *target_host_space,
                std::move(blob),
                compression.column_names,
                static_cast<std::size_t>(payload_bytes),
                uncompressed_bytes,
                static_cast<int64_t>(tbl->num_rows()),
                std::move(column_sizes)));
            });
        } catch (const std::exception& e) {
          compression_failed = true;
          SIRIUS_LOG_WARN(
            "[materialize_pin_to_host] {}",
            compression_failure_warning(e.what(), compression, out.column_storage.back()));
        }
      }

      if (!compressed_this_chunk) {
        cucascade::gpu_table_representation gpu_repr(std::move(tbl), *src_space, stream);
        auto host_reservation =
          target_host_space->make_reservation_or_null(gpu_repr.get_size_in_bytes());
        if (host_reservation == nullptr) {
          SIRIUS_LOG_WARN(
            "materialize_pin_to_host: host reservation failed ({} bytes) — proceeding without "
            "reservation, converter may OOM",
            gpu_repr.get_size_in_bytes());
        }
        auto host_repr = host_reservation != nullptr
                           ? registry.convert<cucascade::host_data_representation>(
                               gpu_repr, *host_reservation, stream)
                           : registry.convert<cucascade::host_data_representation>(
                               gpu_repr, target_host_space, stream);
        stream.synchronize();
        out.chunks.emplace_back(std::move(host_repr));
      }
    });

  return out;
}

device_pin_result materialize_all_batches_compressed(
  op::scan::gpu_ingestible& ingestible,
  std::span<cucascade::memory::memory_space* const> gpu_spaces,
  io::sirius_ioctx& io_ctx,
  duckdb::vector<duckdb::LogicalType> const& pinned_column_types,
  compression_pin_config const& compression,
  pin_materialization_options options)
{
  device_pin_result out;
  bool compression_failed = false;

  // The device insert path stores no statistics sidecar (device_pin_result carries none), so a
  // capture would be computed and dropped — force it off.
  options.capture_chunk_stats = false;

  materialize_pin_batches(
    ingestible,
    gpu_spaces,
    io_ctx,
    pinned_column_types,
    options,
    [&](std::unique_ptr<cudf::table> tbl,
        cucascade::memory::memory_space* src_space,
        rmm::cuda_stream_view stream,
        std::vector<pinned_column_storage_meta> column_storage,
        std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>> /*chunk_stats*/) {
      std::shared_ptr<sirius::compressed_device_representation> compressed_chunk;
      const std::int64_t chunk_rows = tbl->num_rows();
      out.base_row_count_per_chunk.push_back(static_cast<std::size_t>(chunk_rows));
      out.column_storage.emplace_back(std::move(column_storage));

      if (compression.enabled && !compression_failed && tbl && !compression.plan_dsl.empty()) {
        try {
          compress_and_stage_batch(
            *tbl,
            compression,
            stream,
            "materialize_all_batches_compressed",
            [&](simpatico::compressed_table&& /*ct*/,
                std::vector<std::uint8_t>&& header,
                std::vector<simpatico::payload_buffer_ref> const& buffers,
                std::uint64_t payload_bytes,
                std::size_t uncompressed_bytes,
                std::shared_ptr<const sirius::per_column_byte_sizes> column_sizes) {
              // Free the uncompressed source now that the batch is compressed, BEFORE
              // allocating the payload, so it reuses that space (avoids a pin-time peak
              // spike at large scale factors).
              tbl.reset();
              auto blob = std::make_shared<sirius::compressed_device_blob>();

              // Lay every compressed leaf out in one contiguous device buffer at an
              // ALIGNED offset (not the header's dense offsets): nvcomp's batched
              // decode requires aligned input pointers, and the padding after each
              // buffer also absorbs the few-word read-ahead of bitpacked decode. The
              // slab hands these offsets back at reconstruct time so the cached table's
              // leaves are views into this buffer — the only D2D copy, no query re-fetch.
              //
              // Each slice is sized to the leaf's reconstructed footprint (alloc_bytes),
              // NOT its compressed size (size_bytes): read_compressed_table_from_memory
              // allocates each leaf column at its DECODED element count, and a decode
              // kernel reads/writes the whole column — a slice sized only to size_bytes
              // would let the leaf run off its end into the next slice (or past the
              // payload for the last leaf) and fault the context.
              constexpr std::size_t kLeafAlign = rmm::CUDA_ALLOCATION_ALIGNMENT;  // 256
              auto const align_up              = [](std::uint64_t n, std::uint64_t a) {
                return (n + a - 1) & ~(a - 1);
              };
              // The slab hands leaves out positionally: the k-th leaf_mr allocation
              // during the re-read gets offsets[k]. read_compressed_table_from_memory
              // allocates one leaf column per enumerated buffer IN ORDER — EXCEPT a
              // zero-footprint buffer (alloc_bytes == 0, e.g. an empty "output" leaf of
              // an all-null chunk): cudf::make_numeric_column(size 0) allocates nothing,
              // so rmm never calls the slab for it and the cursor does not advance. Give
              // an offset slot only to buffers that actually allocate, or every leaf
              // after the empty one is handed the wrong slice and the decode faults.
              blob->offsets.reserve(buffers.size());
              std::vector<std::size_t> slot_src;  // offsets[k] holds buffers[slot_src[k]]
              slot_src.reserve(buffers.size());
              std::uint64_t cur = 0;
              for (std::size_t i = 0; i < buffers.size(); ++i) {
                auto const& b         = buffers[i];
                std::uint64_t const n = std::max(b.size_bytes, b.alloc_bytes);
                if (n == 0) continue;  // read won't allocate this leaf -> no slot
                cur = align_up(cur, kLeafAlign);
                blob->offsets.push_back(cur);
                slot_src.push_back(i);
                cur += n;
              }
              std::size_t const payload_capacity =
                align_up(cur, kLeafAlign) + kLeafAlign;  // tail slop for the last buffer
              blob->payload =
                rmm::device_buffer(payload_capacity, stream, src_space->get_default_allocator());
              // Zero first: inter-leaf alignment padding and the tail slop are never
              // written by the copies below, and a bitpacked decode reads a few bytes
              // past a leaf's logical end — zeros keep those reads benign.
              CUCASCADE_CUDA_TRY(
                cudaMemsetAsync(blob->payload.data(), 0, payload_capacity, stream.value()));
              for (std::size_t k = 0; k < blob->offsets.size(); ++k) {
                auto const& b = buffers[slot_src[k]];
                if (b.size_bytes > 0 && b.device_ptr != nullptr) {
                  CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
                    static_cast<std::byte*>(blob->payload.data()) + blob->offsets[k],
                    b.device_ptr,
                    static_cast<std::size_t>(b.size_bytes),
                    cudaMemcpyDeviceToDevice,
                    stream.value()));
                }
              }
              blob->slab_mr = sirius::slab_memory_resource{
                static_cast<std::byte*>(blob->payload.data()), &blob->offsets, &blob->slab_cursor};

              auto noop_fetch = [](std::uint64_t, std::size_t, void*, rmm::cuda_stream_view) {};
              std::string read_err;
              // Leaf buffers come from the slab (placed as views into the contiguous
              // payload — zero copy). Codec decode scratch comes from the source GPU
              // pool instead, so it neither disturbs the slab's positional (idx-based)
              // placement nor leaks into the pinned payload.
              blob->table = simpatico::read_compressed_table_from_memory(
                header,
                noop_fetch,
                stream,
                /*mr (scratch)=*/src_space->get_default_allocator(),
                &read_err,
                /*leaf_mr=*/blob->slab_mr);
              if (!read_err.empty()) {
                throw std::runtime_error("[materialize_all_batches_compressed] " + read_err);
              }
              stream.synchronize();

              compressed_chunk = std::make_shared<sirius::compressed_device_representation>(
                *src_space,
                std::move(blob),
                compression.column_names,
                static_cast<std::size_t>(payload_bytes),
                uncompressed_bytes,
                chunk_rows,
                std::move(column_sizes));
            });
        } catch (const std::exception& e) {
          if (!tbl) { throw; }  // source released inside callback — no fallback possible
          compression_failed = true;
          SIRIUS_LOG_WARN(
            "[materialize_all_batches_compressed] {}",
            compression_failure_warning(e.what(), compression, out.column_storage.back()));
        }
      }

      if (compressed_chunk) {
        out.chunks.push_back(device_pin_chunk{
          .compressed = std::move(compressed_chunk), .columns = {}, .memory_space = src_space});
      } else {
        // Retain the uncompressed GPU table in place (device pin holds all
        // chunks on the GPU by definition). Sync so the table is fully resident
        // before it is stored (its writer stream is not tracked downstream), then
        // split it into per-column device columns so a mixed pin stores every
        // chunk — compressed or not — in one ordered vector.
        stream.synchronize();
        auto cols = tbl->release();
        std::vector<std::shared_ptr<cudf::column>> shared_cols;
        shared_cols.reserve(cols.size());
        for (auto& col : cols) {
          shared_cols.emplace_back(std::move(col));
        }
        out.chunks.push_back(device_pin_chunk{
          .compressed = nullptr, .columns = std::move(shared_cols), .memory_space = src_space});
      }
    });

  return out;
}

}  // namespace sirius
