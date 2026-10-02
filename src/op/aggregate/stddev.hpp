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
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

namespace sirius::op {
// Partial state follows cuDF MERGE_M2's contract: STRUCT(INT64 count, DOUBLE mean, DOUBLE M2).
// M2 is the sum of squared deviations, not a finalized variance or standard deviation.
std::unique_ptr<cudf::column> make_stddev_state(std::unique_ptr<cudf::column> count,
                                                std::unique_ptr<cudf::column> mean,
                                                std::unique_ptr<cudf::column> m2,
                                                ::cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr);
std::unique_ptr<cudf::column> local_stddev_state(cudf::column_view input,
                                                 ::cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr);
std::unique_ptr<cudf::column> merge_stddev_states(cudf::column_view states,
                                                  ::cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr);
std::unique_ptr<cudf::column> finalize_stddev(cudf::column_view states,
                                              ::cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr);
}  // namespace sirius::op
