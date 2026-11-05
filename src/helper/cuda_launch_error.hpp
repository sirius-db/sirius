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

#pragma once

#include <cuda_runtime_api.h>

namespace sirius {

/**
 * @brief Whether a kernel launch that failed with @p error may succeed when retried.
 *
 * The launch either lacked resources (`cudaErrorLaunchOutOfResources`) or was rejected with
 * `cudaErrorInvalidValue`. `gpu_pipeline_task` reschedules a task whose kernel launch failed this
 * way; accumulated Bloom work ends its optional attempt as transient.
 */
[[nodiscard]] constexpr bool is_retryable_launch_error(cudaError_t error) noexcept
{
  return error == cudaErrorLaunchOutOfResources || error == cudaErrorInvalidValue;
}

}  // namespace sirius
