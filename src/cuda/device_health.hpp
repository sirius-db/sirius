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
#include <cuda_runtime_api.h>

#include <exception>
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
                         cudaGetErrorName(status))
  {
  }
};
inline void check_cuda_health(cudaError_t status)
{
  if (fatal_cuda_status(status)) throw fatal_device_error(status);
}
inline bool fatal_device_exception(std::exception_ptr error) noexcept
{
  try {
    if (error) std::rethrow_exception(error);
  } catch (fatal_device_error const&) {
    return true;
  } catch (std::exception const& e) {
    // cuDF/RMM wrap CUDA statuses in exceptions after consuming the sticky error.
    // Their error names preserve the fatal classification even after that reset.
    std::string_view message(e.what());
    for (auto name : {"cudaErrorIllegalInstruction",
                      "cudaErrorMisalignedAddress",
                      "cudaErrorInvalidAddressSpace",
                      "cudaErrorInvalidPc",
                      "cudaErrorHardwareStackError",
                      "cudaErrorIllegalAddress",
                      "cudaErrorAssert",
                      "cudaErrorLaunchFailure",
                      "cudaErrorLaunchTimeout",
                      "cudaErrorECCUncorrectable",
                      "cudaErrorContextIsDestroyed",
                      "cudaErrorUnknown"})
      if (message.find(name) != std::string_view::npos) return true;
  } catch (...) {
  }
  return false;
}
}  // namespace sirius
