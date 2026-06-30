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
#include <op/scan/sirius_gpu_scan_operator_data.hpp>

namespace sirius::op::scan {

filtered_table gpu_ingestible::materialize_table(const op::scan::scan_operator_input& split,
                                                 rmm::cuda_stream_view stream)
{
  auto* mem_space = split.gpu_memory_space;
  if (split.has_scan_metadata()) [[likely]] {
    split.prefetch(io::cache::prefetching_stage::disposable);
    return materialize_metadata_to_table(split.get_scan_info(), *mem_space, stream);
  } else {
    auto batch  = split.get_cached_batch();
    auto rbatch = batch->to_read_only();
    auto view   = get_cudf_table_view(rbatch);
    return {.table = owning_table_view{std::move(rbatch), view}, .state = filter_state::UNFILTERED};
  }
}

}  // namespace sirius::op::scan
