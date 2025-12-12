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

#include "expression_executor/regex/regex_playground.hpp"
#include "expression_executor/regex/regex_interpreter.hpp"

#include <optional>

namespace sirius {
namespace regex {

std::unique_ptr<cudf::column>
regex_playground::jit_transform_clickbench_q28_regex(const cudf::column_view& input) {
    auto& cache = RegexUdfCache::Instance();
    const auto& udf = cache.GetOrCreate("(^https?://(?:www\\.)?([^/]+)/.*$)", "(\\1)");
    return cudf::transform({input},
                           udf.source,
                           cudf::data_type{cudf::type_id::STRING},
                           false,
                           std::nullopt,
                           cudf::null_aware::YES);
}

}  // namespace regex
}  // namespace sirius
