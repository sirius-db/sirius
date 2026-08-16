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
#include <cudf/table/table.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <data/sirius_converter_registry.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace sirius::op::scan {

static_assert(std::is_nothrow_move_constructible_v<cudf::column>);

void scan_operator_input::prepare_for_processing(
  const ::cucascade::memory::memory_space* requested_memory_space, rmm::cuda_stream_view stream)
{
  gpu_memory_space = const_cast<::cucascade::memory::memory_space*>(requested_memory_space);
  if (!std::holds_alternative<std::shared_ptr<cucascade::data_batch>>(materialization_info)) {
    prefetch(io::cache::prefetching_stage::just_in_time);
    return;
  }
  auto batch = std::get<std::shared_ptr<cucascade::data_batch>>(materialization_info);

  if (batch && requested_memory_space && !stolen_table && !stolen_table_consumed) {
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
      // Conversion produces a fresh owned table for this split. Raw GPU pins already use a plain
      // gpu_table_representation, so they never reach this branch. Filter-free conversions may
      // therefore transfer their columns without touching shared pin storage. A carrier-converting
      // split retains the whole source until all casts succeed; a non-converting split can detach
      // immediately because no allocating GPU operation follows the take.
      if (!mvcc_keep_mask.has_mask() && !row_filter_pending) {
        if (auto* gpu_rep = dynamic_cast<::cucascade::gpu_table_representation*>(mut.get_data())) {
          auto& space        = gpu_rep->get_memory_space();
          stolen_table_bytes = gpu_rep->get_size_in_bytes();
          if (needs_carrier_conversion) {
            converted_table_steal_pending = true;
          } else {
            stolen_table = gpu_rep->release_table(stream);
            // The batch cannot hold null data and its size/view queries dereference the table, so
            // leave a valid empty placeholder.
            mut.set_data(std::make_unique<::cucascade::gpu_table_representation>(
              std::make_unique<cudf::table>(), space, rmm::cuda_stream_view{}));
          }
        }
      }
    }
  }

  if (batch) {
    auto ro          = batch->to_read_only();
    gpu_memory_space = ro.get_memory_space();
  }
}

std::unique_ptr<cudf::table> scan_operator_input::transactionally_steal_converted_table(
  std::size_t output_width,
  const converted_table_builder& builder,
  rmm::cuda_stream_view stream) const
{
  // This gate is deliberately narrower than the generic resident path. Only prepare's own fresh
  // conversion may set pending; raw GPU pins and any split that needs filtering stay view-backed.
  if (!converted_table_steal_pending || !needs_carrier_conversion || stolen_table_consumed ||
      stolen_table || !is_resident() || mvcc_keep_mask.has_mask() || row_filter_pending) {
    return nullptr;
  }

  auto batch = get_cached_batch();
  if (!batch) { return nullptr; }

  // Keep the exclusive lock and the complete source table through every potentially allocating
  // builder operation. In particular, rmm::out_of_memory unwinds only the replacement columns and
  // this lock; the wrapper still owns every source column for the scheduler's retry.
  auto mut = batch->to_mutable();
  mut.rebind_stream(stream);
  auto* gpu_rep = dynamic_cast<::cucascade::gpu_table_representation*>(mut.get_data());
  if (gpu_rep == nullptr) { return nullptr; }

  auto const source_view = gpu_rep->get_table_view();
  if (static_cast<std::size_t>(source_view.num_columns()) != output_width) { return nullptr; }

  auto& space    = gpu_rep->get_memory_space();
  auto empty_rep = std::make_unique<::cucascade::gpu_table_representation>(
    std::make_unique<cudf::table>(), space, rmm::cuda_stream_view{});
  auto replacements = builder(source_view);
  if (replacements.size() != output_width) {
    throw std::runtime_error(
      "[scan_operator_input::transactionally_steal_converted_table] builder returned the wrong "
      "number of replacement columns");
  }
  for (auto const& replacement : replacements) {
    if (replacement && replacement->size() != source_view.num_rows()) {
      throw std::runtime_error(
        "[scan_operator_input::transactionally_steal_converted_table] replacement column has the "
        "wrong row count");
    }
  }

  // Commit point: pending provenance guarantees this is the owned-table arm, so release_table
  // moves rather than materializes. No allocating GPU operation follows this source surrender.
  auto source_table = gpu_rep->release_table(stream);
  if (!source_table) {
    throw std::runtime_error(
      "[scan_operator_input::transactionally_steal_converted_table] converted source table was "
      "already released");
  }
  for (std::size_t column_idx = 0; column_idx < replacements.size(); ++column_idx) {
    if (!replacements[column_idx]) { continue; }
    auto& destination = source_table->get_column(static_cast<cudf::size_type>(column_idx));
    std::destroy_at(std::addressof(destination));
    std::construct_at(std::addressof(destination), std::move(*replacements[column_idx]));
  }
  mut.set_data(std::move(empty_rep));
  converted_table_steal_pending = false;
  stolen_table_consumed         = true;
  return source_table;
}

std::size_t scan_operator_input::get_estimated_size_in_bytes() const
{
  if (std::holds_alternative<std::unique_ptr<scan_info>>(materialization_info)) {
    return std::get<std::unique_ptr<scan_info>>(materialization_info)->estimated_bytes();
  }
  if (std::holds_alternative<std::shared_ptr<cucascade::data_batch>>(materialization_info)) {
    // Once prepare_for_processing has taken the wrapper's table the batch only
    // holds an empty placeholder; answer from the stolen table so OOM-retry
    // estimates keep covering the live data.
    if (stolen_table_bytes > 0) { return stolen_table_bytes; }
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
