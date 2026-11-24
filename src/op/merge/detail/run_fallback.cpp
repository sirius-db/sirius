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

#include "op/merge/detail/run_fallback.hpp"

#include "compression/simpatico_compressed_representation.hpp"
#include "data/sirius_converter_registry.hpp"
#include "downgrade/spill_io_stats.hpp"

#include <rmm/cuda_device.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/cuda/event.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/memory/common.hpp>

#include <chrono>
#include <stdexcept>

namespace sirius::op::merge::detail {

bool convert_run_batch_to_gpu(cucascade::data_batch& batch,
                              cucascade::memory::memory_space& gpu_space,
                              rmm::cuda_stream_view stream)
{
  if (gpu_space.get_tier() != cucascade::memory::Tier::GPU) {
    throw std::invalid_argument("convert_run_batch_to_gpu: the target space is not a GPU space");
  }

  // Blocking, like every other in-place conversion of a repository batch (the upgrade in
  // lock_and_prepare_batch, a blocking downgrade): to_mutable() waits for the batch's readers,
  // their recorded device reads and any other exclusive holder (a downgrade of this batch, or
  // another task converting it). The caller must hold no lock on this batch itself.
  cucascade::mutable_data_batch mut = batch.to_mutable();

  // Decide under the exclusive lock: whoever held it before us may have converted the batch.
  // A compressed device representation is GPU-tier too but is not readable through views, so
  // only an uncompressed table in `gpu_space` counts as already converted.
  auto const* space = mut.get_memory_space();
  if (space != nullptr && space->get_id() == gpu_space.get_id() &&
      dynamic_cast<cucascade::gpu_table_representation const*>(mut.get_data()) != nullptr) {
    return false;
  }

  ::cuda::stream_ref const stream_ref{stream.value()};
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{gpu_space.get_device_id()}};

  // A device-resident source (compressed, or a table on another GPU) may still be being written.
  if (cudaEvent_t const writer_event = mut.get_data()->get_writer_event();
      writer_event != nullptr) {
    cucascade::cuda::cuda_event_view{writer_event}.wait(stream_ref);
  }

  // The target-space overload allocates from `gpu_space`'s allocator, so the bytes are charged
  // to the reservation the merge task has attached; an OOM propagates and leaves the batch as
  // it was. convert_to() synchronizes `stream` before it destroys the source representation.
  auto const source_tier  = mut.get_data()->get_current_tier();
  auto const source_bytes = mut.get_data()->get_size_in_bytes();
  auto const compressed   = is_simpatico_compressed_representation(mut.get_data());
  auto const started      = std::chrono::steady_clock::now();
  mut.convert_to<cucascade::gpu_table_representation>(
    sirius::converter_registry::get(), &gpu_space, stream_ref);
  spill::io_stats::get().record_reload(source_tier,
                                       source_bytes,
                                       mut.get_data()->get_size_in_bytes(),
                                       spill::micros_since(started),
                                       compressed);
  return true;
}

}  // namespace sirius::op::merge::detail
