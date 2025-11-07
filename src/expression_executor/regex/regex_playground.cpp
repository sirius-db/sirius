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

namespace sirius {
namespace expression {

std::unique_ptr<cudf::column>
regex_playground::jit_transform_clickbench_q28_regex(const cudf::column_view& input) {
    auto udf = R"***(
__device__ void extract_domain(cudf::string_view* out, cuda::std::optional<cudf::string_view> const url_opt) {
    // Skip null
    if (!url_opt.has_value()) {
        return;
    }
    cudf::string_view url = url_opt.value();

    // For "http"
    if (!(url.length() >= 4 && url[0] == 'h' && url[1] == 't' && url[2] == 't' && url[3] == 'p')) {
        *out = url;
        return;
    }
    cudf::string_view next = url.substr(4, url.length() - 4);

    // For "s?"
    if (!next.empty() && next[0] == 's') {
        next = next.substr(1, next.length() - 1);
    }

    // For "://"
    if (!(next.length() >= 3 && next[0] == ':' && next[1] == '/' && next[2] == '/')) {
        *out = url;
        return;
    }
    next = next.substr(3, next.length() - 3);

    // For "(?:www\.)?"
    if (next.length() >= 4 && next[0] == 'w' && next[1] == 'w' && next[2] == 'w' && next[3] == '.') {
        next = next.substr(4, next.length() - 4);
    }

    // For "([^/]+)"
    if (next.empty() || next[0] == '/') {
        *out = url;
        return;
    }

    // For "/.*" and "\1"
    auto pos = next.find('/');
    *out = (pos == cudf::string_view::npos) ? url : next.substr(0, pos);
}
)***";

    return cudf::transform({input}, udf, cudf::data_type{cudf::type_id::STRING}, false,
                            std::nullopt, cudf::null_aware::YES);
}

} // namespace expression
} // namespace sirius
