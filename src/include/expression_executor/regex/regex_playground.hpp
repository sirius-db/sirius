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

#include <cudf/column/column_factories.hpp>
#include <cudf/transform.hpp>
#include <cudf/copying.hpp>
#include <rmm/cuda_stream_view.hpp>

namespace sirius {
namespace expression {

class regex_playground {
public:
    static std::unique_ptr<cudf::column> jit_transform_clickbench_q28_regex(const cudf::column_view& input);
};

} // namespace expression
} // namespace sirius
