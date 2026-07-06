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
#include "data/sirius_converter_registry.hpp"
#include "io/io_context.hpp"
#include "log/logging.hpp"
#include "op/scan/gpu_ingestible.hpp"
#include "scan_manager/round_robin_strategy.hpp"

#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <api/compressed_table_io.hpp>
#include <api/simpatico_codegen.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
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

namespace {

/// Per-materialized-batch sink: receives one GPU-resident table, its GPU placement, and the
/// stream it was decoded on. The driver does NOT synchronize — the sink owns the sync, because
/// the host-streaming path must synchronize while the GPU table (and the gpu_table_representation
/// wrapping it) is still alive, after the D2H conversion has been enqueued on the same stream.
using pin_batch_sink = std::function<void(std::unique_ptr<cudf::table>,
                                          cucascade::memory::memory_space* target,
                                          rmm::cuda_stream_view stream)>;

/// Shared driver behind @ref materialize_all_batches and @ref materialize_pin_to_host: walk the
/// ingestible's metadata + batch coalescer to completion, materialize each emitted batch onto a
/// round-robin GPU, and hand it to @p on_batch. The single-threaded, deterministic round-robin
/// placement means re-pinning the same source yields identical placement (required by
/// insert_pinned_entry's merge path on the GPU tier) and bounds peak GPU residency to ~one batch
/// (the host tier frees each table in @p on_batch before the next is materialized).
void materialize_pin_batches(op::scan::gpu_ingestible& ingestible,
                             std::span<cucascade::memory::memory_space* const> gpu_spaces,
                             io::sirius_ioctx& io_ctx,
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
    auto materialized = ingestible.materialize_metadata_to_table(*batch, *target, stream);
    auto tbl          = materialized.table.release(stream, target->get_default_allocator());
    on_batch(std::move(tbl), target, stream);
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

}  // namespace

materialized_pin materialize_all_batches(
  op::scan::gpu_ingestible& ingestible,
  std::span<cucascade::memory::memory_space* const> gpu_spaces,
  io::sirius_ioctx& io_ctx)
{
  materialized_pin out;
  materialize_pin_batches(ingestible,
                          gpu_spaces,
                          io_ctx,
                          [&](std::unique_ptr<cudf::table> tbl,
                              cucascade::memory::memory_space* target,
                              rmm::cuda_stream_view stream) {
                            // Cached GPU batches are stored with a null writer stream, so the data
                            // must be fully resident before it can be served or host-converted.
                            stream.synchronize();
                            out.tables.emplace_back(std::move(tbl));
                            out.chunk_memory_spaces.push_back(target);
                          });
  return out;
}

std::vector<std::shared_ptr<cucascade::host_data_representation>> materialize_pin_to_host(
  op::scan::gpu_ingestible& ingestible,
  std::span<cucascade::memory::memory_space* const> gpu_spaces,
  const std::unordered_map<int, cucascade::memory::memory_space*>& host_space_by_gpu,
  io::sirius_ioctx& io_ctx)
{
  auto& registry = converter_registry::get();
  std::vector<std::shared_ptr<cucascade::host_data_representation>> host_chunks;

  materialize_pin_batches(
    ingestible,
    gpu_spaces,
    io_ctx,
    [&](std::unique_ptr<cudf::table> tbl,
        cucascade::memory::memory_space* src_space,
        rmm::cuda_stream_view stream) {
      // Stream this freshly-materialized batch straight to pinned host memory and let the GPU
      // table free before the next batch is materialized — so peak GPU residency stays at ~one
      // batch and the whole table never needs to fit in GPU memory. The chunk is pinned on the
      // source GPU's NUMA-local host space (host_space_by_gpu), so on multi-GPU systems the
      // chunks land round-robin across NUMA nodes. The conversion reuses the decode stream; the
      // GPU->HOST converter (convert_gpu_to_host_fast) synchronizes internally before returning,
      // so the host copy is complete once convert() returns. The explicit sync below is
      // belt-and-suspenders before gpu_repr (which owns the GPU table's buffers) leaves scope.
      auto* target_host_space = host_space_by_gpu.at(src_space->get_device_id());
      cucascade::gpu_table_representation gpu_repr(std::move(tbl), *src_space, stream);
      auto host_repr =
        registry.convert<cucascade::host_data_representation>(gpu_repr, target_host_space, stream);
      stream.synchronize();
      host_chunks.emplace_back(std::move(host_repr));
    });

  return host_chunks;
}

host_pin_result materialize_pin_to_host_with_compression(
  op::scan::gpu_ingestible& ingestible,
  std::span<cucascade::memory::memory_space* const> gpu_spaces,
  const std::unordered_map<int, cucascade::memory::memory_space*>& host_space_by_gpu,
  io::sirius_ioctx& io_ctx,
  compression_pin_config const& compression)
{
  auto& registry = converter_registry::get();
  host_pin_result out;

  materialize_pin_batches(
    ingestible,
    gpu_spaces,
    io_ctx,
    [&](std::unique_ptr<cudf::table> tbl,
        cucascade::memory::memory_space* src_space,
        rmm::cuda_stream_view stream) {
      auto* target_host_space    = host_space_by_gpu.at(src_space->get_device_id());
      bool compressed_this_chunk = false;

      if (compression.enabled && tbl && tbl->num_columns() > 0 && !compression.plan_dsl.empty()) {
        try {
          // Total device footprint of the batch (includes string chars/offsets
          // and null masks), so string columns count toward the threshold.
          std::size_t uncompressed_bytes = tbl->alloc_size();
          if (uncompressed_bytes >= compression.min_batch_size_bytes) {
            auto ct = simpatico::compress_with_plan(tbl->view(),
                                                    compression.plan_dsl,
                                                    stream,
                                                    rmm::mr::get_current_device_resource_ref(),
                                                    compression.column_names);

            // Build the structural header and enumerate the payload buffers
            // (no bytes copied yet).
            std::vector<std::uint8_t> header;
            std::vector<simpatico::payload_buffer_ref> buffers;
            std::uint64_t payload_bytes = 0;
            const std::string hdr_err =
              simpatico::build_compressed_table_header(ct, header, buffers, payload_bytes, stream);
            if (!hdr_err.empty()) {
              throw std::runtime_error("build_compressed_table_header: " + hdr_err);
            }

            // Keep the compressed form only if it saves enough: compare the
            // total compressed footprint (header + payload) against the batch's
            // original device size. Otherwise discard it and pin uncompressed.
            const std::size_t original_bytes   = tbl->alloc_size();
            const std::size_t compressed_bytes = header.size() + payload_bytes;
            if (original_bytes > 0 &&
                static_cast<double>(compressed_bytes) >
                  compression.max_compressed_fraction * static_cast<double>(original_bytes)) {
              SIRIUS_LOG_DEBUG(
                "[materialize_pin_to_host_with_compression] compressed {}B > {:.0f}% of {}B "
                "original; pinning uncompressed",
                compressed_bytes,
                compression.max_compressed_fraction * 100.0,
                original_bytes);
            } else {
              // Allocate the pinned payload from the target host space's chunked
              // pool (the same tracked pool the uncompressed path uses) and stage
              // every compressed buffer device->pinned. Sync before `ct` (which
              // owns the device buffers) leaves scope.
              auto* host_mr =
                target_host_space->get_memory_resource_of<cucascade::memory::Tier::HOST>();
              if (host_mr == nullptr) {
                throw std::runtime_error(
                  "target host space has no fixed_size_host_memory_resource");
              }
              auto blob           = std::make_shared<sirius::pinned_compressed_blob>();
              blob->header        = std::move(header);
              blob->payload       = host_mr->allocate_multiple_blocks(payload_bytes);
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

              out.compressed_chunks.emplace_back(
                std::make_shared<sirius::compressed_host_representation>(
                  *target_host_space,
                  std::move(blob),
                  compression.column_names,
                  static_cast<std::size_t>(payload_bytes),
                  uncompressed_bytes,
                  static_cast<int64_t>(tbl->num_rows())));
              compressed_this_chunk = true;
            }
          }
        } catch (const std::exception& e) {
          SIRIUS_LOG_WARN(
            "[materialize_pin_to_host_with_compression] compression failed: {}; "
            "falling back to uncompressed for this chunk",
            e.what());
        }
      }

      if (!compressed_this_chunk) {
        cucascade::gpu_table_representation gpu_repr(std::move(tbl), *src_space, stream);
        auto host_repr = registry.convert<cucascade::host_data_representation>(
          gpu_repr, target_host_space, stream);
        stream.synchronize();
        out.host_chunks.emplace_back(std::move(host_repr));
      }
    });

  return out;
}

device_pin_result materialize_pin_to_device_with_compression(
  op::scan::gpu_ingestible& ingestible,
  std::span<cucascade::memory::memory_space* const> gpu_spaces,
  io::sirius_ioctx& io_ctx,
  compression_pin_config const& compression)
{
  device_pin_result out;

  materialize_pin_batches(
    ingestible,
    gpu_spaces,
    io_ctx,
    [&](std::unique_ptr<cudf::table> tbl,
        cucascade::memory::memory_space* src_space,
        rmm::cuda_stream_view stream) {
      bool compressed_this_chunk = false;

      if (compression.enabled && tbl && tbl->num_columns() > 0 && !compression.plan_dsl.empty()) {
        try {
          // Total device footprint of the batch (includes string chars/offsets
          // and null masks), so string columns count toward the threshold.
          std::size_t uncompressed_bytes = tbl->alloc_size();
          if (uncompressed_bytes >= compression.min_batch_size_bytes) {
            auto ct = simpatico::compress_with_plan(tbl->view(),
                                                    compression.plan_dsl,
                                                    stream,
                                                    rmm::mr::get_current_device_resource_ref(),
                                                    compression.column_names);

            std::vector<std::uint8_t> header;
            std::vector<simpatico::payload_buffer_ref> buffers;
            std::uint64_t payload_bytes = 0;
            const std::string hdr_err =
              simpatico::build_compressed_table_header(ct, header, buffers, payload_bytes, stream);
            if (!hdr_err.empty()) {
              throw std::runtime_error("build_compressed_table_header: " + hdr_err);
            }

            // Keep the compressed form only if it saves enough (header + payload
            // vs the batch's original device size); otherwise discard and pin
            // uncompressed.
            const std::size_t original_bytes   = tbl->alloc_size();
            const std::size_t compressed_bytes = header.size() + payload_bytes;
            if (original_bytes > 0 &&
                static_cast<double>(compressed_bytes) >
                  compression.max_compressed_fraction * static_cast<double>(original_bytes)) {
              SIRIUS_LOG_DEBUG(
                "[materialize_pin_to_device_with_compression] compressed {}B > {:.0f}% of {}B "
                "original; pinning uncompressed",
                compressed_bytes,
                compression.max_compressed_fraction * 100.0,
                original_bytes);
            } else {
              // Keep the compressed payload in one contiguous device buffer on the
              // source GPU; copy each compressed leaf buffer device->device, then
              // sync before `ct` (owning the source device buffers) leaves scope.
              rmm::device_buffer payload(static_cast<std::size_t>(payload_bytes),
                                         stream,
                                         src_space->get_default_allocator());
              for (auto const& b : buffers) {
                if (b.size_bytes > 0 && b.device_ptr != nullptr) {
                  cudaMemcpyAsync(static_cast<std::byte*>(payload.data()) + b.offset,
                                  b.device_ptr,
                                  static_cast<std::size_t>(b.size_bytes),
                                  cudaMemcpyDeviceToDevice,
                                  stream.value());
                }
              }
              stream.synchronize();

              auto blob           = std::make_shared<sirius::device_compressed_blob>();
              blob->header        = std::move(header);
              blob->payload       = std::move(payload);
              blob->payload_bytes = payload_bytes;

              out.compressed_chunks.emplace_back(
                std::make_shared<sirius::compressed_device_representation>(
                  *src_space,
                  std::move(blob),
                  compression.column_names,
                  static_cast<std::size_t>(payload_bytes),
                  uncompressed_bytes,
                  static_cast<int64_t>(tbl->num_rows())));
              compressed_this_chunk = true;
            }
          }
        } catch (const std::exception& e) {
          SIRIUS_LOG_WARN(
            "[materialize_pin_to_device_with_compression] compression failed: {}; "
            "falling back to uncompressed for this chunk",
            e.what());
        }
      }

      if (!compressed_this_chunk) {
        // Retain the uncompressed GPU table in place (device pin holds all
        // chunks on the GPU by definition). Sync so the table is fully resident
        // before it is stored (its writer stream is not tracked downstream).
        stream.synchronize();
        out.tables.emplace_back(std::move(tbl));
        out.chunk_memory_spaces.push_back(src_space);
      }
    });

  return out;
}

}  // namespace sirius
