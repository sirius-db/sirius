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
#include <variant>
#include "expression_evaluator/regex/regex_playground.hpp"
// clang-format on

#include <cudf/cudf_utils.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <vector>

namespace sirius {
namespace regex {

std::unique_ptr<cudf::column> regex_playground::jit_transform_clickbench_q28_regex(
  const cudf::column_view& input, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  // libcudf's $ anchor matches before one final LF. Replacing that match retains the unmatched LF,
  // which a string-view transform cannot synthesize after extracting a non-contiguous domain.
  // Route the whole batch through the generic implementation when any row ends in LF.
  auto final_newline_rows = cudf::strings::ends_with(
    cudf::strings_column_view(input), cudf::string_scalar("\n", true, stream, mr), stream, mr);
  auto any_final_newline = cudf::reduce(final_newline_rows->view(),
                                        *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
                                        cudf::data_type{cudf::type_id::BOOL8},
                                        stream,
                                        mr);
  auto const& any_final_newline_scalar =
    static_cast<cudf::scalar_type_t<bool> const&>(*any_final_newline);
  if (any_final_newline_scalar.is_valid(stream) && any_final_newline_scalar.value(stream)) {
    auto regex_prog = cudf::strings::regex_program::create(R"(^https?://(?:www\.)?([^/]+)/.*$)");
    return cudf::strings::replace_with_backrefs(
      cudf::strings_column_view(input), *regex_prog, R"(\1)", stream, mr);
  }

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

    // For "/.*", an internal newline triggers a mismatch. A final newline is handled by the
    // batch-level generic fallback above because libcudf's $ anchor preserves it.
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
