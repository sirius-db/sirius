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

#include "op/sort_comparison_config.hpp"

#include <cudf/table/table.hpp>

#include <cstdint>

namespace sirius::op::sampled_merge {

// GPU helpers used by merge tasks, with an explicit stream/resource. Callers
// register idle outputs, retain inputs through device completion and
// provide the ordinary task fence on both success and exception paths.
// Samples/splitters contain SQL keys (in comparison order), then two nonnullable
// UINT64 columns: frozen contribution ordinal and original row ordinal.
std::unique_ptr<cudf::table> gather_samples(cudf::table_view input,
                                            uint64_t ordinal,
                                            uint64_t stride,
                                            sort_comparison_config const& comparison,
                                            ::cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr);

// All sample backing, concatenation, sort workspace, selected indices and splitter
// copy must fit the caller's admission. Native overflow is rejected before any
// assembly. This helper makes no claim that its sample working set is bounded
// in bytes; the operator admits the post-coalescing envelope or fails normally.
std::unique_ptr<cudf::table> order_sample_splitters(std::vector<cudf::table_view> const& samples,
                                                    uint64_t sample_step,
                                                    sort_comparison_config const& comparison,
                                                    ::cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr);

// Accepts a contiguous splitter window. Returns device INT32 offsets, without
// endpoints. No host key/offset readback: extraction will copy a whole window
// once and add its adjacent boundaries/endpoints before owning slice copies.
std::unique_ptr<cudf::column> discover_cuts(cudf::table_view input,
                                            uint64_t ordinal,
                                            cudf::table_view splitters,
                                            sort_comparison_config const& comparison,
                                            ::cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr);

}  // namespace sirius::op::sampled_merge
