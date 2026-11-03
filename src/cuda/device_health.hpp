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

#pragma once
#include <cudf/utilities/error.hpp>

#include <rmm/error.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>

#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sirius {
// These failures invalidate a shared CUDA context. Allocation exhaustion and
// invalid SQL/operator input remain ordinary per-query failures.
inline bool fatal_cuda_status(cudaError_t status) noexcept
{
  switch (status) {
    case cudaErrorIllegalInstruction:
    case cudaErrorMisalignedAddress:
    case cudaErrorInvalidAddressSpace:
    case cudaErrorInvalidPc:
    case cudaErrorHardwareStackError:
    case cudaErrorIllegalAddress:
    case cudaErrorAssert:
    case cudaErrorLaunchFailure:
    case cudaErrorLaunchTimeout:
    case cudaErrorECCUncorrectable:
    case cudaErrorContextIsDestroyed:
    case cudaErrorUnknown: return true;
    default: return false;
  }
}
class fatal_device_error : public std::runtime_error {
 public:
  explicit fatal_device_error(cudaError_t status)
    : std::runtime_error(std::string("Sirius CUDA context unavailable: ") +
                         cudaGetErrorName(status)),
      status_(status)
  {
  }
  [[nodiscard]] cudaError_t error_code() const noexcept { return status_; }

 private:
  cudaError_t status_;
};
inline void check_cuda_health(cudaError_t status)
{
  if (fatal_cuda_status(status)) throw fatal_device_error(status);
}
namespace detail {
// RMM's pinned version does not retain cudaError_t. Decode only its CUDA macro's
// status field, and only for RMM exception types. Never inspect arbitrary SQL/error text.
inline std::optional<cudaError_t> fatal_rmm_cuda_status(std::string_view message) noexcept
{
  constexpr std::string_view allocation_prefix = "std::bad_alloc: ";
  if (message.starts_with(allocation_prefix)) message.remove_prefix(allocation_prefix.size());
  if (!message.starts_with("CUDA error at: ") &&
      !message.starts_with("CUDA error (failed to allocate "))
    return std::nullopt;
  auto const status_start = message.rfind(": ");
  if (status_start == std::string_view::npos) return std::nullopt;
  message.remove_prefix(status_start + 2);
  auto const status_end = message.find(' ');
  auto const name       = message.substr(0, status_end);
  // Match the complete status token, not occurrences in the file path or error description.
  for (auto status : {cudaErrorIllegalInstruction,
                      cudaErrorMisalignedAddress,
                      cudaErrorInvalidAddressSpace,
                      cudaErrorInvalidPc,
                      cudaErrorHardwareStackError,
                      cudaErrorIllegalAddress,
                      cudaErrorAssert,
                      cudaErrorLaunchFailure,
                      cudaErrorLaunchTimeout,
                      cudaErrorECCUncorrectable,
                      cudaErrorContextIsDestroyed,
                      cudaErrorUnknown}) {
    if (name == cudaGetErrorName(status)) return status;
  }
  return std::nullopt;
}
}  // namespace detail

// Return the fatal status as well as recognizing it, so health diagnostics never
// have to recover a code from an arbitrary exception message.
inline std::optional<cudaError_t> fatal_device_status(std::exception_ptr error) noexcept
{
  auto status = cudaSuccess;
  try {
    if (error) std::rethrow_exception(error);
  } catch (fatal_device_error const& e) {
    status = e.error_code();
  } catch (cudf::cuda_error const& e) {
    status = e.error_code();
  } catch (::cuda::cuda_error const& e) {
    status = static_cast<cudaError_t>(e.status());
  } catch (thrust::system_error const& e) {
    // Other backends can reuse the same numeric values; only CUDA codes apply.
    if (e.code().category() == thrust::cuda_category())
      status = static_cast<cudaError_t>(e.code().value());
  } catch (rmm::cuda_error const& e) {
    return detail::fatal_rmm_cuda_status(e.what());
  } catch (rmm::bad_alloc const& e) {
    // RMM_CUDA_TRY_ALLOC also uses bad_alloc for non-OOM CUDA failures.
    return detail::fatal_rmm_cuda_status(e.what());
  } catch (...) {
  }
  return fatal_cuda_status(status) ? std::optional{status} : std::nullopt;
}
}  // namespace sirius
