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

#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream_ref>

#include <cstdint>

namespace sirius::op::kept_build {

/**
 * @brief Set `flags[i] = 1` for every row index `i` in @p indices, in place.
 *
 * Only ever writes the value 1, so concurrent calls on different streams against the same buffer
 * are benign: the flags are an OR-accumulation of matches over every probe batch.
 *
 * @param flags One byte per build row, 0 = not yet matched.
 * @param indices Build row indices, each in `[0, flags.size())`.
 */
void mark_rows(cudf::device_span<std::uint8_t> flags,
               cudf::device_span<cudf::size_type const> indices,
               ::cuda::stream_ref stream);

}  // namespace sirius::op::kept_build
