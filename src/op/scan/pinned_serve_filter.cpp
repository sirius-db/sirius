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

#include "op/scan/pinned_serve_filter.hpp"

#include "data/sirius_converter_registry.hpp"
#include "log/logging.hpp"
#include "op/scan/pinned_block_gather.hpp"
#include "op/sirius_dynamic_filter.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>
#include <nvtx3/nvtx3.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/cudf/host_table.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace sirius::op::scan {

namespace {

/// Fast-path admissibility of one served column: plain fixed-width, null-free, no children. A
/// null MASK with zero null COUNT is fine (parquet schemas are nullable even for null-free data);
/// an all-valid mask carries no information, so the gathered output legitimately omits it.
bool column_is_gatherable(cucascade::memory::column_metadata const& meta)
{
  if ((meta.has_null_mask && meta.null_count > 0) || !meta.children.empty() || !meta.has_data) {
    return false;
  }
  auto const dt = cudf::data_type{static_cast<cudf::type_id>(meta.type_id)};
  return cudf::is_fixed_width(dt);
}

cudf::data_type column_data_type(cucascade::memory::column_metadata const& meta)
{
  auto const id = static_cast<cudf::type_id>(meta.type_id);
  return cudf::is_fixed_point(cudf::data_type{id}) ? cudf::data_type{id, meta.scale}
                                                   : cudf::data_type{id};
}

}  // namespace

pinned_serve_filter_result try_serve_filtered_upload(
  cucascade::data_batch& batch,
  sirius_dynamic_filter_set const& filters,
  cucascade::memory::memory_space const* target_space,
  rmm::cuda_stream_view stream)
{
  pinned_serve_filter_result result;
  if (target_space == nullptr) { return result; }

  std::unique_ptr<cucascade::gpu_table_representation> filtered_rep;
  {
    auto ro          = batch.to_read_only();
    auto const* data = ro.get_data();
    if (data == nullptr || ro.get_current_tier() != cucascade::memory::Tier::HOST) {
      SIRIUS_LOG_DEBUG("[pinned_serve_filter] split skipped: not a HOST-tier batch.");
      return result;
    }
    auto const* host = dynamic_cast<cucascade::host_data_representation const*>(data);
    if (host == nullptr) {
      SIRIUS_LOG_DEBUG("[pinned_serve_filter] split skipped: not a plain host representation.");
      return result;
    }
    auto const& tbl = *host->get_host_table();
    if (tbl.columns.empty() || tbl.allocation == nullptr) { return result; }
    auto const num_rows = tbl.columns.front().num_rows;
    if (num_rows <= 0) { return result; }

    for (auto const& meta : tbl.columns) {
      if (!column_is_gatherable(meta) || meta.num_rows != num_rows) {
        SIRIUS_LOG_DEBUG(
          "[pinned_serve_filter] split skipped: column not gatherable (type_id={}, null_count={}, "
          "children={}).",
          meta.type_id,
          meta.null_count,
          meta.children.size());
        return result;
      }
    }

    // Served cached batches are laid out output-columns-first in output order
    // (gpu_ingestible::materialized_column_order), and the channel keys filters by output
    // position — so a filter for output column p tests served column p.
    auto const device_id = target_space->get_device_id();
    std::size_t key_col  = 0;
    std::shared_ptr<sirius_dynamic_filter const> key_filter;
    sirius_mask_applicable const* mask_capable = nullptr;
    for (auto const col_idx : filters.filtered_columns()) {
      if (col_idx >= tbl.columns.size()) { continue; }
      for (auto const& f : filters.filters_for_column(col_idx)) {
        auto const* capable = dynamic_cast<sirius_mask_applicable const*>(f.get());
        if (capable == nullptr || !f->is_available_on_device(device_id)) { continue; }
        key_col      = col_idx;
        key_filter   = f;
        mask_capable = capable;
        break;
      }
      if (key_filter) { break; }
    }
    if (!key_filter) {
      SIRIUS_LOG_DEBUG(
        "[pinned_serve_filter] split skipped: no device-local mask-capable filter in the channel "
        "({} filtered columns, {} served columns).",
        filters.filtered_columns().size(),
        tbl.columns.size());
      return result;
    }

    nvtx3::scoped_range nvtx_range{"dynfilter::serve_filtered_upload"};
    auto const mr = target_space->get_default_allocator();

    // 1. Upload only the key column via the regular converter and probe the filter on device.
    auto key_host = host->slice(std::vector<std::size_t>{key_col});
    auto key_gpu  = sirius::converter_registry::get().convert<cucascade::gpu_table_representation>(
      *key_host, target_space, stream);
    if (!key_gpu) { return result; }
    auto const key_view = key_gpu->get_table_view().column(0);
    auto const mask     = mask_capable->compute_mask(key_view, device_id, stream, mr);
    if (!mask) { return result; }

    // 2. Survivor indices + the serve-side selectivity gate.
    auto seq       = cudf::sequence(static_cast<cudf::size_type>(num_rows),
                              cudf::numeric_scalar<cudf::size_type>(0, true, stream),
                              cudf::numeric_scalar<cudf::size_type>(1, true, stream),
                              stream,
                              mr);
    auto survivors = cudf::apply_boolean_mask(
      cudf::table_view(std::vector<cudf::column_view>{seq->view()}), mask->view(), stream, mr);
    auto const kept = survivors->num_rows();
    result.rows_in  = static_cast<std::size_t>(num_rows);
    result.rows_out = static_cast<std::size_t>(kept);
    if (kept == num_rows ||
        static_cast<double>(kept) > k_serve_filter_keep_threshold * static_cast<double>(num_rows)) {
      return result;  // not selective enough to beat the sequential bulk copy
    }

    // 3. Gather surviving rows of every other column straight from the mapped pinned blocks.
    auto const host_blocks = tbl.allocation->get_blocks();
    rmm::device_uvector<std::byte const*> device_blocks(host_blocks.size(), stream, mr);
    if (cudaMemcpyAsync(device_blocks.data(),
                        host_blocks.data(),
                        host_blocks.size() * sizeof(std::byte*),
                        cudaMemcpyHostToDevice,
                        stream.value()) != cudaSuccess) {
      return result;
    }
    auto const block_size      = tbl.allocation->block_size();
    auto const* survivor_index = survivors->view().column(0).data<cudf::size_type>();

    std::vector<std::unique_ptr<cudf::column>> out_columns;
    out_columns.reserve(tbl.columns.size());
    std::size_t gathered_bytes = 0;
    for (std::size_t c = 0; c < tbl.columns.size(); ++c) {
      auto const& meta = tbl.columns[c];
      auto const dt    = column_data_type(meta);
      if (c == key_col) {
        // Already device-resident from the probe; compact it there instead of re-reading host.
        auto compacted = cudf::apply_boolean_mask(
          cudf::table_view(std::vector<cudf::column_view>{key_view}), mask->view(), stream, mr);
        out_columns.push_back(std::move(compacted->release().front()));
        continue;
      }
      auto const width = static_cast<std::size_t>(cudf::size_of(dt));
      rmm::device_buffer out_buf(static_cast<std::size_t>(kept) * width, stream, mr);
      gather_fixed_width_from_host_blocks(device_blocks.data(),
                                          block_size,
                                          meta.data_offset,
                                          width,
                                          survivor_index,
                                          kept,
                                          static_cast<std::byte*>(out_buf.data()),
                                          stream);
      gathered_bytes += out_buf.size();
      out_columns.push_back(
        std::make_unique<cudf::column>(dt, kept, std::move(out_buf), rmm::device_buffer{}, 0));
    }

    result.bytes_moved   = tbl.columns[key_col].data_size + gathered_bytes;
    result.bytes_avoided = tbl.data_size - result.bytes_moved;

    // The gather kernels read the pinned host blocks asynchronously; the blocks are owned by the
    // representation this function is about to replace. Drain the stream before releasing the
    // read accessor so the host memory cannot be freed underneath in-flight kernels.
    stream.synchronize();

    filtered_rep = std::make_unique<cucascade::gpu_table_representation>(
      std::make_unique<cudf::table>(std::move(out_columns)),
      const_cast<cucascade::memory::memory_space&>(*target_space),
      stream);
  }

  auto mut = batch.to_mutable();
  mut.set_data(std::move(filtered_rep));
  result.applied = true;
  SIRIUS_LOG_DEBUG(
    "[pinned_serve_filter] served split filtered pre-transfer: kept {} of {} rows, moved {} B, "
    "avoided {} B.",
    result.rows_out,
    result.rows_in,
    result.bytes_moved,
    result.bytes_avoided);
  return result;
}

}  // namespace sirius::op::scan
