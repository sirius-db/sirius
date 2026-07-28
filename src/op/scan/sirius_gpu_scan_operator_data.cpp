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
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <data/sirius_converter_registry.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>

#include <algorithm>

namespace sirius::op::scan {

void scan_operator_input::prepare_for_processing(
  const ::cucascade::memory::memory_space* requested_memory_space, rmm::cuda_stream_view stream)
{
  gpu_memory_space = const_cast<::cucascade::memory::memory_space*>(requested_memory_space);
  if (!std::holds_alternative<std::shared_ptr<cucascade::data_batch>>(materialization_info)) {
    prefetch(io::cache::prefetching_stage::just_in_time);
    return;
  }
  auto batch = std::get<std::shared_ptr<cucascade::data_batch>>(materialization_info);

  if (batch && requested_memory_space) {
    bool needs_upload = false;
    {
      auto ro          = batch->to_read_only();
      auto const* data = ro.get_data();
      // Convert when the data is not on the GPU tier, OR when it is on the GPU
      // tier but not already a plain gpu_table_representation (e.g. a
      // compressed_device_representation, which must be decompressed in place).
      const bool is_gpu_table =
        dynamic_cast<const ::cucascade::gpu_table_representation*>(data) != nullptr;
      needs_upload = data != nullptr &&
                     (ro.get_current_tier() != ::cucascade::memory::Tier::GPU || !is_gpu_table);
    }
    if (needs_upload) {
      auto& registry = ::sirius::converter_registry::get();
      auto mut       = batch->to_mutable();
      mut.convert_to<::cucascade::gpu_table_representation>(
        registry, requested_memory_space, stream);
    }
  }

  if (batch) {
    auto ro          = batch->to_read_only();
    gpu_memory_space = ro.get_memory_space();
  }
}

std::size_t scan_operator_input::get_estimated_size_in_bytes() const
{
  if (std::holds_alternative<std::unique_ptr<scan_info>>(materialization_info)) {
    return std::get<std::unique_ptr<scan_info>>(materialization_info)->estimated_bytes();
  }
  if (std::holds_alternative<std::shared_ptr<cucascade::data_batch>>(materialization_info)) {
    auto batch = std::get<std::shared_ptr<cucascade::data_batch>>(materialization_info);

    auto ro          = batch->to_read_only();
    auto const* data = ro.get_data();
    if (!data) { return 0; }
    // prepare_for_processing decompresses a compressed cache batch, so the task's
    // working set (hence its reservation) is the UNCOMPRESSED footprint, not the
    // resident compressed payload. get_size_in_bytes() reports only the compressed
    // bytes for a compressed entry; using it under-reserves, and the on-demand
    // decompress then over-allocates into the downgrade path — which cannot evict
    // the pinned GPU entry, so the query deadlocks. Uncompressed batches report
    // equal sizes, so this is a no-op for them.
    return std::max(data->get_size_in_bytes(), data->get_uncompressed_data_size_in_bytes());
  }
  return 0;
}

std::size_t scan_operator_input::get_estimated_working_set_size_in_bytes() const
{
  if (std::holds_alternative<std::unique_ptr<scan_info>>(materialization_info)) {
    auto const decode_bytes =
      std::get<std::unique_ptr<scan_info>>(materialization_info)->estimated_working_set_bytes();
    if (mvcc_keep_mask.has_mask()) {
      // A partially visible insert-delta split is mask-filtered right after
      // decode: the decoded input and the compacted output (up to input-sized)
      // coexist at peak, alongside the BOOL8 expansion column (1 B/row) and
      // the uploaded bitmask words — the same envelope as the cached branch
      // below.
      return 2 * decode_bytes + mvcc_keep_mask.row_count + mvcc_keep_mask.view().size_bytes();
    }
    return decode_bytes;
  }
  auto const batch_bytes = get_estimated_size_in_bytes();
  if (mvcc_keep_mask.has_mask()) {
    // A masked resident chunk is filtered by copy at materialize: the input
    // batch and the compacted output (up to input-sized) coexist at peak,
    // alongside the BOOL8 expansion column (1 B/row) and the uploaded bitmask
    // words. A pending row filter needs no extra headroom: its phase peaks at
    // mask output + predicate + compacted output, inside the same envelope.
    return 2 * batch_bytes + mvcc_keep_mask.row_count + mvcc_keep_mask.view().size_bytes();
  }
  if (row_filter_pending) {
    // post_filter_and_project filters by copy: the materialized input and the
    // compacted output (up to input-sized) coexist at peak. The BOOL8
    // predicate column (1 B/row) hides inside the 2x conservatism (any
    // projected column is >= 4 B/row).
    return 2 * batch_bytes;
  }
  // An unmasked, unfiltered chunk serves a zero-copy view whose output copy is
  // at most batch-sized, so plain batch_bytes stays accurate.
  return batch_bytes;
}

}  // namespace sirius::op::scan
