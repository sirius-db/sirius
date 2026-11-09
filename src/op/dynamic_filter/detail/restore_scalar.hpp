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

#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <memory>

namespace sirius::op::detail {

/**
 * @brief Converts a zone-map bound at a narrowed carrier to the key's recorded storage type
 *
 * The bound goes through a one-row column and `cast_through_rep`, so a DATE bound takes the same
 * INT32 tunnel the build column would. Reads the result back, so it waits on @p stream.
 */
[[nodiscard]] std::unique_ptr<cudf::scalar> restore_scalar(cudf::scalar const& bound,
                                                           cudf::data_type target,
                                                           ::cuda::stream_ref stream,
                                                           rmm::device_async_resource_ref mr);

}  // namespace sirius::op::detail
