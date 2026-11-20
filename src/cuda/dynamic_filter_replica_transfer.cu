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

#include "memory/runtime_stream_pool.hpp"

#include <cudf/utilities/error.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime.h>

#include <cucascade/error.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <log/logging.hpp>
#include <memory/device_transfer.hpp>
#include <op/dynamic_filter/dynamic_filter_replica_transfer.hpp>

#include <cstddef>
#include <exception>
#include <stdexcept>
#include <string_view>

namespace sirius::op::detail {

replica_transfer_route enqueue_replica_copy(
  void* destination,
  rmm::cuda_device_id destination_device,
  void const* source,
  cucascade::memory::memory_space const& source_space,
  std::size_t bytes,
  ::cuda::stream_ref destination_stream,
  cucascade::memory::memory_space const& host_staging_space,
  replica_transfer_policy policy)
{
  if (bytes != 0 && (destination == nullptr || source == nullptr)) {
    throw std::invalid_argument("[enqueue_replica_copy] nonempty copies require non-null pointers");
  }
  return sirius::memory::enqueue_device_copy({static_cast<std::byte*>(destination), bytes},
                                             destination_device,
                                             destination_stream,
                                             {static_cast<std::byte const*>(source), bytes},
                                             rmm::cuda_device_id{source_space.get_device_id()},
                                             &source_space,
                                             sirius::memory::source_already_complete{},
                                             &host_staging_space,
                                             policy);
}

bool replica_fanout_failures::skip_if_recoverable(std::string_view filter_name,
                                                  int source_device,
                                                  int destination_device) noexcept
{
  auto const failure = std::current_exception();
  try {
    std::rethrow_exception(failure);
  } catch (cucascade::cuda_error const&) {
  } catch (::cuda::cuda_error const&) {
  } catch (cudf::cuda_error const&) {
  } catch (rmm::cuda_error const&) {
  } catch (std::logic_error const&) {
  } catch (std::exception const& e) {
    sirius::log::log_noexcept([&] {
      SIRIUS_LOG_WARN(
        "{} replica GPU {} -> GPU {} unavailable: {}. That GPU will skip this optional filter.",
        filter_name,
        source_device,
        destination_device,
        e.what());
    });
    return true;
  } catch (...) {
  }
  record_fatal(filter_name, source_device, destination_device);
  return false;
}

void replica_fanout_failures::record_fatal(std::string_view filter_name,
                                           int source_device,
                                           int destination_device) noexcept
{
  auto const failure = std::current_exception();
  if (!_fatal) {
    _fatal = failure;
    return;
  }
  sirius::log::log_noexcept([&] {
    try {
      std::rethrow_exception(failure);
    } catch (std::exception const& error) {
      SIRIUS_LOG_ERROR(
        "{} replica GPU {} -> GPU {} failed during cleanup: {}. Preserving the first fatal error.",
        filter_name,
        source_device,
        destination_device,
        error.what());
    } catch (...) {
      SIRIUS_LOG_ERROR(
        "{} replica GPU {} -> GPU {} failed during cleanup with a non-standard exception. "
        "Preserving the first fatal error.",
        filter_name,
        source_device,
        destination_device);
    }
  });
}

void replica_fanout_failures::rethrow_if_fatal() const
{
  if (_fatal) { std::rethrow_exception(_fatal); }
}

}  // namespace sirius::op::detail
