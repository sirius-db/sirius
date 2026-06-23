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

// sirius
#include <data/cached_data_representation.hpp>
#include <data/host_parquet_representation.hpp>
#include <data/host_parquet_representation_converters.hpp>
#include <log/logging.hpp>
#include <op/scan/cached_ranges.hpp>

// cucascade
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>

// cudf
#include "cudf/cudf_utils.hpp"

#include <cudf/utilities/span.hpp>

// rmm
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

// cuda runtime (FIX-02 carryover: cudaGetLastError sticky-state consume)
#include <cuda_runtime_api.h>

// standard library
#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace sirius {

namespace detail {

/**
 * @brief Convert host_parquet_representation to gpu_table_representation
 *
 * Multi-GPU stream-correctness pattern: the prior implementation set
 *   rmm::cuda_set_device_raii target_device_raii(target_device_id)
 * but then called `cudf::io::read_parquet(opts, stream, mr_ref)` using the
 * CALLER-supplied `stream`. Under `num_gpus == 2`, the caller's stream may be
 * bound to a non-target device (e.g. the pipeline-executor's GPU-0 stream
 * while `target_device_id == 1`), producing `cudaErrorInvalidValue` inside
 * cudf's internal H2D path. Same root cause as cucascade's built-in
 * convert_host_fast_to_gpu / convert_gpu_to_gpu before they were fixed.
 *
 * Fix pattern:
 *   1. Sync caller's stream so any upstream work on it is flushed.
 *   2. Enter `rmm::cuda_set_device_raii` for the target device.
 *   3. Acquire a target-bound stream from `target_memory_space->acquire_stream()`.
 *   4. Use the TARGET-bound stream for read_parquet + apply_post_convert +
 *      apply_partition_inject + final sync (never the caller's stream).
 *   5. Consume sticky cuda errors before returning.
 */
std::unique_ptr<cucascade::idata_representation>
convert_host_parquet_to_gpu_with_prefetched_data_source(
  cucascade::idata_representation& source,
  cucascade::memory::memory_space const* target_memory_space,
  rmm::cuda_stream_view stream)
{
  // STRUCTURAL GAP (io rebase): sirius::op::scan::prefetched_data_source — the
  // in-memory cudf::io::datasource adapter that wrapped a cache_ranges and fed
  // cudf::io::read_parquet from prefetched host blocks — was removed
  // (prefetched_data_source.cpp deleted, the type is no longer declared) with no
  // replacement. The cached-host-parquet -> GPU conversion below cannot be
  // reconstructed until that adapter is reimplemented, so this path is stubbed
  // to throw and the dependent body is disabled.
  (void)source;
  (void)target_memory_space;
  (void)stream;
  throw std::runtime_error(
    "convert_host_parquet_to_gpu_with_prefetched_data_source: prefetched_data_source "
    "adapter was removed in the io rebase; cached-host-parquet GPU conversion is "
    "unavailable until it is reimplemented.");
#if 0   // disabled: depends on the removed prefetched_data_source adapter
  auto& host_src                         = source.cast<host_parquet_representation>();
  auto const& post_filter_projection_ids = host_src.get_post_filter_projection_ids();

  rmm::device_async_resource_ref mr_ref(target_memory_space->get_default_allocator());
  rmm::cuda_device_id target_device_id(target_memory_space->get_device_id());

  // [mgpu-probe] entry instrumentation (08-07 gap-closure).
  // Captures the device/stream identity the converter OBSERVES before any
  // RAII switch or target-stream acquire. If current_device != target_device_id
  // at entry, hypothesis A (upstream frame in wrong device context) is confirmed.
  // If current_device == target_device_id but stream handle differs from target
  // stream at exit, hypothesis D (mr_ref captured before RAII) is in play.
  {
    int current_device = -1;
    (void)cudaGetDevice(&current_device);
    SIRIUS_LOG_INFO(
      "[mgpu-probe] host_parquet_to_gpu entry current_device={} stream={} "
      "target_device_id={} memspace_device_id={}",
      current_device,
      static_cast<void*>(stream.value()),
      target_device_id.value(),
      target_memory_space->get_device_id());
  }

  // --- Sync caller stream BEFORE switching device. The caller's stream may
  // --- be bound to a different device than target_device_id; syncing it
  // --- under its own device is safe (cuda_stream_view::synchronize wraps
  // --- cudaStreamSynchronize which is cross-device tolerant for the WAIT
  // --- itself). This flushes any upstream work that produced the source.
  stream.synchronize();

  // --- Enter target device, then acquire a target-bound stream from the
  // --- target memory space's pool. From this point on we use target_stream
  // --- for every cudf H2D/compute call.
  rmm::cuda_set_device_raii target_device_raii(target_device_id);
  auto target_stream = target_memory_space->acquire_stream();

  // Build a cache_ranges from the packed host blocks and the column-chunk byte-range descriptors.
  // The block pointers are raw std::byte* owned by the allocation; cache_ranges does not take
  // ownership of them (the allocation still lives for the duration of this call).
  auto const& allocation = host_src.get_column_chunks();
  auto block_ptrs =
    std::vector<std::byte*>(allocation->get_blocks().begin(), allocation->get_blocks().end());

  auto ranges =
    std::make_unique<sirius::op::scan::cache_ranges>(host_src.get_column_chunk_byte_ranges(),
                                                     std::move(block_ptrs),
                                                     allocation->block_size(),
                                                     target_memory_space->get_device_id(),
                                                     host_src.get_device_id());  // NUMA node id

  auto data_source = std::make_unique<sirius::op::scan::prefetched_data_source>(
    std::move(ranges), host_src.get_file_size(), host_src.get_fallback_datasource());

  // Point the reader options at our in-memory datasource and call cudf::io::read_parquet.
  auto opts = host_src.get_reader_options();
  opts.set_source(cudf::io::source_info{data_source.get()});
  // set_row_groups expects one inner vector per source; we have a single source.
  opts.set_row_groups({std::vector<cudf::size_type>(host_src.get_row_group_indices().begin(),
                                                    host_src.get_row_group_indices().end())});

  // Pick the filter-tree entry whose cudf::scalar device buffers live on the
  // target device. The planner built one tree per configured GPU; here we
  // select the one that matches this task's target so cudf's AST evaluation
  // (inside cudf::io::read_parquet) reads scalars from memory the current
  // device can address.
  auto const& filter_by_device = host_src.get_filter_expression_by_device();
  if (filter_by_device && !filter_by_device->empty()) {
    auto it = filter_by_device->find(target_device_id.value());
    if (it == filter_by_device->end()) { it = filter_by_device->begin(); }
    opts.set_filter(it->second.back());
  }

  auto [table, md] = cudf::io::read_parquet(opts, target_stream, mr_ref);

  // Apply the post-convert hook (used by iceberg scan for V2 delete filtering).
  if (host_src.has_post_convert_fn()) {
    table = host_src.apply_post_convert(std::move(table), target_stream);
  }

  // Reshape the cuDF table to the pipeline-expected layout: inject hive partition columns and,
  // under schema evolution, NULL-fill any columns missing from this file. The closure is built
  // by parquet_scan_task_global_state (init_hive_partitions / build_schema_reconciliation).
  if (host_src.has_partition_inject_fn()) {
    table = host_src.apply_partition_inject(std::move(table), target_stream);
  }

  target_stream.synchronize();

  // Consume any sticky CUDA state before returning so a later call-site does
  // not surface a stray error against us (matches Pattern 2 hygiene).
  (void)cudaGetLastError();

  // [mgpu-probe] exit instrumentation (08-07 gap-closure).
  // Captures the device context right after the target-bound read + inject
  // chain completes and the sticky-error consume runs. If we reach here and
  // the upstream caller STILL fails at cuda_memcpy.cu:42, the hazard is not
  // inside this function and is either upstream (hypothesis A/C) or on the
  // post-return column-projection path below.
  {
    int current_device_exit = -1;
    (void)cudaGetDevice(&current_device_exit);
    SIRIUS_LOG_INFO(
      "[mgpu-probe] host_parquet_to_gpu exit current_device={} target_stream={} "
      "target_device_id={}",
      current_device_exit,
      static_cast<void*>(target_stream.value()),
      target_device_id.value());
  }

  // Now we need to prune the post-filter columns from the table, if there are any.
  if (!post_filter_projection_ids.empty()) {
    auto columns = table->release();
    std::vector<std::unique_ptr<cudf::column>> projected_columns;
    projected_columns.reserve(post_filter_projection_ids.size());
    for (auto const id : post_filter_projection_ids) {
      projected_columns.push_back(std::move(columns[id]));
    }
    table = std::make_unique<cudf::table>(std::move(projected_columns));
  }

  // STREAM-LINEAGE: read_parquet + apply_post_convert + apply_partition_inject
  // all wrote on `target_stream`. The target_stream.synchronize() above flushes
  // their work in-stream order, but the target stream itself is the writer
  // identity that any cross-device reader (cucascade::convert_gpu_to_gpu) must
  // observe — record the writer event so the subsequent reader's
  // cudaStreamWaitEvent receives a real ordering primitive.
  auto repr = std::make_unique<cucascade::gpu_table_representation>(
    std::move(table),
    *const_cast<cucascade::memory::memory_space*>(target_memory_space),
    target_stream);
  return repr;
#endif  // disabled: depends on the removed prefetched_data_source adapter
}

/**
 * @brief Convert host_parquet_representation to host_parquet_representation (cross-host copy)
 */
std::unique_ptr<cucascade::idata_representation> convert_host_parquet_to_host_parquet(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view /* stream */)
{
  auto& host_src       = source.cast<host_parquet_representation>();
  auto const data_size = host_src.get_size_in_bytes();

  assert(source.get_device_id() != target_memory_space->get_device_id());
  auto* mr = target_memory_space
               ->get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  if (mr == nullptr) {
    throw std::runtime_error(
      "Target HOST memory_space does not have a fixed_size_host_memory_resource");
  }

  auto const& src_allocation  = host_src.get_column_chunks();
  auto dst_allocation         = mr->allocate_multiple_blocks(data_size);
  size_t src_block_index      = 0;
  size_t src_block_offset     = 0;
  size_t dst_block_index      = 0;
  size_t dst_block_offset     = 0;
  size_t const src_block_size = src_allocation->block_size();
  size_t const dst_block_size = dst_allocation->block_size();
  size_t copied               = 0;
  while (copied < data_size) {
    size_t remaining     = data_size - copied;
    size_t src_avail     = src_block_size - src_block_offset;
    size_t dst_avail     = dst_block_size - dst_block_offset;
    size_t bytes_to_copy = std::min({remaining, src_avail, dst_avail});
    auto* src_ptr        = src_allocation->at(src_block_index).data() + src_block_offset;
    auto* dst_ptr        = dst_allocation->at(dst_block_index).data() + dst_block_offset;
    std::memcpy(dst_ptr, src_ptr, bytes_to_copy);
    copied += bytes_to_copy;
    src_block_offset += bytes_to_copy;
    dst_block_offset += bytes_to_copy;
    if (src_block_offset == src_block_size) {
      src_block_index++;
      src_block_offset = 0;
    }
    if (dst_block_offset == dst_block_size) {
      dst_block_index++;
      dst_block_offset = 0;
    }
  }

  using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;
  auto cloned_reader       = std::make_unique<hybrid_scan_reader>(
    host_src.get_parquet_reader()->parquet_metadata(), host_src.get_reader_options());
  auto dst = std::make_unique<host_parquet_representation>(
    const_cast<cucascade::memory::memory_space*>(target_memory_space),
    std::move(dst_allocation),
    std::move(cloned_reader),
    host_src.get_reader_options(),
    host_src.get_row_group_indices(),
    host_src.get_column_chunk_byte_ranges(),
    data_size,
    host_src.get_uncompressed_data_size_in_bytes(),
    host_src.get_file_size(),
    host_src.get_fallback_datasource(),
    host_src.get_filter_expression_by_device(),
    host_src.get_post_filter_projection_ids());
  if (host_src.has_post_convert_fn()) { dst->set_post_convert_fn(host_src.get_post_convert_fn()); }
  if (host_src.has_partition_inject_fn()) {
    dst->set_partition_inject_fn(host_src.get_partition_inject_fn());
    dst->set_partition_values(host_src.get_partition_values());
  }
  if (!host_src.get_data_file_path().empty()) {
    dst->set_data_file_path(host_src.get_data_file_path());
  }
  return dst;
}

}  // namespace detail

void register_parquet_converters(cucascade::representation_converter_registry& registry)
{
  // HOST Parquet -> GPU
  if (!registry.has_converter<host_parquet_representation, cucascade::gpu_table_representation>()) {
    registry.register_converter<host_parquet_representation, cucascade::gpu_table_representation>(
      detail::convert_host_parquet_to_gpu_with_prefetched_data_source);
  }

  // HOST Parquet -> HOST Parquet (cross-host copy)
  if (!registry.has_converter<host_parquet_representation, host_parquet_representation>()) {
    registry.register_converter<host_parquet_representation, host_parquet_representation>(
      detail::convert_host_parquet_to_host_parquet);
  }

  if (!registry
         .has_converter<cached_host_data_representation, cucascade::gpu_table_representation>()) {
    registry
      .register_converter<cached_host_data_representation, cucascade::gpu_table_representation>(
        [&registry](cucascade::idata_representation& source,
                    const cucascade::memory::memory_space* target_memory_space,
                    rmm::cuda_stream_view stream) {
          auto r = source.cast<cached_host_data_representation>().get_representation();
          return registry.convert<cucascade::gpu_table_representation>(
            *r, target_memory_space, stream);
        });
  }

  if (!registry.has_converter<cached_host_parquet_representation,
                              cucascade::gpu_table_representation>()) {
    registry
      .register_converter<cached_host_parquet_representation, cucascade::gpu_table_representation>(
        [&registry](cucascade::idata_representation& source,
                    const cucascade::memory::memory_space* target_memory_space,
                    rmm::cuda_stream_view stream) {
          auto r = source.cast<cached_host_parquet_representation>().get_representation();
          return registry.convert<cucascade::gpu_table_representation>(
            *r, target_memory_space, stream);
        });
  }
}

}  // namespace sirius
