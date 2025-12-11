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
namespace regex {

std::unique_ptr<cudf::column> regex_playground::jit_transform_clickbench_q28_regex(
  const cudf::column_view& input)
{
  auto udf = R"***(
__device__ void extract_domain(cuda::std::optional<cudf::string_view>* out, cuda::std::optional<cudf::string_view> const url_opt) {
    // Skip null
    if (!url_opt.has_value()) {
        return;
    }
    cudf::string_view url = url_opt.value();
    auto len = url.length();
    int32_t pos = 0;
    int32_t g1_start = -1;
    int32_t g1_end = -1;
    // ^ start anchor
    // Literal "http"
    if (!(len - pos >= 4 && 
          url[pos + 0] == 'h' && url[pos + 1] == 't' && url[pos + 2] == 't' && url[pos + 3] == 'p'))
    {
        *out = url;
        return;
    }
    pos += 4;
    // Quantifier ?
    {
        int32_t save_pos = pos;
        if (pos < static_cast<int32_t>(len) && url[pos] == 's') {
            ++pos;
        } else {
            pos = save_pos;
        }
    }
    // Literal "://"
    if (!(len - pos >= 3 && 
          url[pos + 0] == ':' && url[pos + 1] == '/' && url[pos + 2] == '/'))
    {
        *out = url;
        return;
    }
    pos += 3;
    // Quantifier ?
    {
        int32_t save_pos = pos;
        if (len - pos >= 4 && 
            url[pos + 0] == 'w' && url[pos + 1] == 'w' && url[pos + 2] == 'w' && url[pos + 3] == '.') {
            pos += 4;
        } else {
            pos = save_pos;
        }
    }
    // Capturing group 1
    g1_start = pos;
    // Quantifier +
    if (pos >= static_cast<int32_t>(len) || url[pos] == '/')
    {
        *out = url;
        return;
    }
    while (pos < static_cast<int32_t>(len) && url[pos] != '/') {
        ++pos;
    }
    g1_end = pos;
    // Literal "/"
    if (!(len - pos >= 1 && 
          url[pos + 0] == '/'))
    {
        *out = url;
        return;
    }
    pos += 1;
    // Quantifier *
    while (pos < static_cast<int32_t>(len) && url[pos] != '\n') {
        ++pos;
    }
    // $ end anchor
    if (pos != static_cast<int32_t>(len))
    {
        *out = url;
        return;
    }
    // Build replacement on success
    if (g1_start >= 0 && g1_end >= g1_start) {
        *out = url.substr(g1_start, g1_end - g1_start);
    } else {
        *out = url;
    }
}
)***";

  cudf::transform_input ti = input;
  return cudf::transform_extended(std::span(&ti, 1),
                                  udf,
                                  cudf::data_type{cudf::type_id::STRING},
                                  cudf::udf_source_type::CUDA,
                                  std::nullopt,
                                  cudf::null_aware::YES);
}

}  // namespace regex
}  // namespace sirius
