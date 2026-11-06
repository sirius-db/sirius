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

#include "op/join/kept_build_marks.hpp"

#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>

namespace sirius::op::kept_build {

void mark_rows(cudf::device_span<std::uint8_t> flags,
               cudf::device_span<cudf::size_type const> indices,
               ::cuda::stream_ref stream)
{
  if (indices.empty()) { return; }
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   indices.begin(),
                   indices.end(),
                   [flags = flags.data()] __device__(cudf::size_type row) { flags[row] = 1; });
}

}  // namespace sirius::op::kept_build
