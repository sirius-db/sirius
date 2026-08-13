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

#include "op/scan/owning_table_view.hpp"

#include <data/data_batch_utils.hpp>
#include <op/scan/gpu_ingestible.hpp>
#include <op/scan/host_keep_mask.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>

#include <stdexcept>
#include <string>
#include <utility>

namespace sirius::op::scan {

filtered_table gpu_ingestible::materialize_table(const op::scan::scan_operator_input& split,
                                                 rmm::cuda_stream_view stream)
{
  auto* mem_space = split.gpu_memory_space;
  if (split.has_scan_metadata()) [[likely]] {
    split.prefetch(io::cache::prefetching_stage::disposable);
    auto materialized = materialize_metadata_to_table(split.get_scan_info(), *mem_space, stream);
    if (split.mvcc_keep_mask.has_mask()) {
      // Only insert-delta splits carry a visibility mask here; disk-walk
      // splits never do. Sync before returning for the same reason as the
      // cached branch below.
      auto const& mask = split.mvcc_keep_mask;
      auto view        = materialized.table.view();
      if (mask.row_count != static_cast<std::size_t>(view.num_rows())) {
        throw std::runtime_error("[gpu_ingestible::materialize_table] mvcc keep-mask covers " +
                                 std::to_string(mask.row_count) +
                                 " rows but the materialized split has " +
                                 std::to_string(view.num_rows()));
      }
      auto masked = apply_host_keep_bitmask(
        view, mask.view(), mask.row_count, stream, mem_space->get_default_allocator());
      stream.synchronize();
      return {.table = owning_table_view{std::move(masked)}, .state = materialized.state};
    }
    return materialized;
  } else {
    if (split.stolen_table) {
      // prepare_for_processing took ownership of the wrapper batch's per-query
      // table; move it straight into the scan output — the owned-table
      // owning_table_view releases by moving columns, so no copy is made.
      split.stolen_table_consumed = true;
      return {.table = owning_table_view{std::move(split.stolen_table)},
              .state = filter_state::UNFILTERED};
    }
    if (split.stolen_table_consumed) {
      throw std::runtime_error(
        "[gpu_ingestible::materialize_table] the split's stolen table was already consumed; "
        "refusing to re-materialize from the emptied wrapper batch");
    }
    auto batch  = split.get_cached_batch();
    auto rbatch = batch->to_read_only();
    auto view   = get_cudf_table_view(rbatch);
    if (split.mvcc_keep_mask.has_mask()) {
      auto const& mask = split.mvcc_keep_mask;
      if (mask.row_count != static_cast<std::size_t>(view.num_rows())) {
        throw std::runtime_error("[gpu_ingestible::materialize_table] mvcc keep-mask covers " +
                                 std::to_string(mask.row_count) +
                                 " rows but the cached chunk has " +
                                 std::to_string(view.num_rows()));
      }
      auto masked = apply_host_keep_bitmask(
        view, mask.view(), mask.row_count, stream, mem_space->get_default_allocator());
      // The pinned mask words feed a true-async H2D DMA, and for a HOST-tier
      // chunk `rbatch` owns the staged device copy the filter kernel reads —
      // neither release is ordered against the stream once this branch
      // returns. Await the mask work before dropping them, the same
      // discipline the duckdb-native decoder uses for its staging buffers
      // (submit_and_await).
      stream.synchronize();
      return {.table = owning_table_view{std::move(masked)}, .state = filter_state::UNFILTERED};
    }
    return {.table = owning_table_view{std::move(rbatch), view}, .state = filter_state::UNFILTERED};
  }
}

}  // namespace sirius::op::scan
