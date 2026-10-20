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

// clang-format off
#include "expression_evaluator/regex/regex_playground.hpp"
// clang-format on

#include <cudf/cudf_utils.hpp>

namespace sirius {
namespace regex {

std::unique_ptr<cudf::column> regex_playground::jit_transform_clickbench_q28_regex(
  const cudf::column_view& input, ::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  auto udf = R"***(
__device__ void extract_domain(cuda::std::optional<cudf::string_view>* out, cuda::std::optional<cudf::string_view> const url_opt) {
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
    // Only consume the optional prefix when the required ([^/]+) group will remain non-empty.
    // Otherwise the generic regex engine backtracks and captures "www." as the host.
    if (next.length() > 4 && next[0] == 'w' && next[1] == 'w' && next[2] == 'w' && next[3] == '.' && next[4] != '/') {
        next = next.substr(4, next.length() - 4);
    }

    // For "([^/]+)/"
    if (next.empty() || next[0] == '/') {
        *out = url;
        return;
    }
    auto pos = next.find('/');
    if (pos == cudf::string_view::npos) {
        *out = url;
        return;
    }
    *out = next.substr(0, pos);

    // For "/.*$": RE2's . excludes newlines and its $ only matches at the end of the text,
    // so any newline after the host, trailing or not, means the pattern does not match.
    next = next.substr(pos + 1, next.length() - pos - 1);
    if (next.find('\n') != cudf::string_view::npos) {
        *out = url;
        return;
    }

}
)***";

  cudf::transform_input ti = input;
#if CUDF_VERSION_NUM >= 2610
  cudf::transform_output const output{cudf::data_type{cudf::type_id::STRING},
                                      cudf::output_nullability::PRESERVE};
  auto result  = cudf::transform(udf,
                                cudf::udf_source_type::CUDA,
                                cudf::null_aware::YES,
                                std::nullopt,
                                std::span(&ti, 1),
                                std::span(&output, 1),
                                std::vector<std::unique_ptr<cudf::column>>{},
                                std::nullopt,
                                stream,
                                mr);
  auto columns = result->release();
  return std::move(columns.front());
#else
  return cudf::transform_extended(std::span(&ti, 1),
                                  udf,
                                  cudf::data_type{cudf::type_id::STRING},
                                  cudf::udf_source_type::CUDA,
                                  std::nullopt,
                                  cudf::null_aware::YES,
                                  std::nullopt,
                                  cudf::output_nullability::PRESERVE,
                                  stream,
                                  mr);
#endif
}

}  // namespace regex
}  // namespace sirius
