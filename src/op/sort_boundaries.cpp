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

#include "op/sort_boundaries.hpp"

#include "log/logging.hpp"
#include "sirius/exception.hpp"

#include <cudf/contiguous_split.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime.h>

#include <cucascade/memory/memory_space.hpp>

#include <utility>

namespace sirius::op {

sort_boundaries::sort_boundaries(cudf::table_view const& boundaries,
                                 cucascade::memory::memory_space const& owner,
                                 ::cuda::stream_ref stream)
  : _num_rows(boundaries.num_rows())
{
  // One contiguous buffer plus host metadata, whatever the key types are.
  auto packed = cudf::pack(boundaries, stream, owner.get_default_allocator());
  if (!packed.metadata || !packed.gpu_data) {
    throw internal_exception("Sort boundaries: cudf::pack returned no data");
  }
  _metadata = std::move(*packed.metadata);
  _host_data.resize(packed.gpu_data->size());
  if (!_host_data.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(_host_data.data(),
                                  packed.gpu_data->data(),
                                  _host_data.size(),
                                  cudaMemcpyDeviceToHost,
                                  stream.get()));
  }
  // Tasks on other streams read these rows as soon as the caller publishes this object, so both
  // the host copy and the owner's copy must be complete before the constructor returns.
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));

  auto view =
    cudf::unpack(_metadata.data(), static_cast<std::uint8_t const*>(packed.gpu_data->data()));
  _copies.emplace(&owner, device_copy{std::move(packed.gpu_data), view});
}

cudf::table_view sort_boundaries::view_for(cucascade::memory::memory_space const& space,
                                           ::cuda::stream_ref stream)
{
  std::lock_guard lock(_mutex);
  if (auto it = _copies.find(&space); it != _copies.end()) { return it->second.view; }

  auto data =
    std::make_unique<rmm::device_buffer>(_host_data.size(), stream, space.get_default_allocator());
  if (!_host_data.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      data->data(), _host_data.data(), _host_data.size(), cudaMemcpyHostToDevice, stream.get()));
  }
  // Complete before it is stored: later tasks on this GPU read it from their own streams.
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));

  auto view = cudf::unpack(_metadata.data(), static_cast<std::uint8_t const*>(data->data()));
  SIRIUS_LOG_DEBUG("[sort_boundaries] copied {} boundary rows ({} bytes) to GPU {}",
                   _num_rows,
                   _host_data.size(),
                   space.get_device_id());
  return _copies.emplace(&space, device_copy{std::move(data), view}).first->second.view;
}

std::size_t sort_boundaries::num_copies() const
{
  std::lock_guard lock(_mutex);
  return _copies.size();
}

}  // namespace sirius::op
