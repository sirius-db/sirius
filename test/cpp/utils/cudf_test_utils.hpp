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

#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace sirius {
namespace test {

// Stream-aware variants to enforce stream ordering with async allocations
bool cudf_tables_have_equal_contents_on_stream(const cudf::table& left,
                                               const cudf::table& right,
                                               rmm::cuda_stream_view stream_view);
void expect_cudf_tables_equal_on_stream(const cudf::table& left,
                                        const cudf::table& right,
                                        rmm::cuda_stream_view stream_view);

}  // namespace test
}  // namespace sirius
