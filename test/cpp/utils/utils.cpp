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

#include "utils.hpp"

#include <cudf/column/column_factories.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace sirius {

std::mt19937_64& global_rng()
{
  static std::random_device rd;
  static std::mt19937_64 gen(rd());
  return gen;
}

template <typename T>
std::unique_ptr<cudf::column> create_numeric_column_with_random_data(
  size_t num_rows,
  const cudf::data_type& dtype,
  const std::optional<std::pair<int, int>>& range,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto& gen = global_rng();
  auto col  = cudf::make_numeric_column(dtype, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  auto view = col->mutable_view();

  auto dist = range.has_value() ? std::uniform_int_distribution<T>(range->first, range->second)
                                : std::uniform_int_distribution<T>(0, 1000);
  std::vector<T> h_data(num_rows);
  for (size_t r = 0; r < num_rows; ++r)
    h_data[r] = dist(gen);

  cudaMemcpy(view.data<T>(), h_data.data(), sizeof(T) * num_rows, cudaMemcpyHostToDevice);
  return col;
}

std::unique_ptr<cudf::table> create_cudf_table_with_random_data(
  size_t num_rows,
  const std::vector<cudf::data_type>& column_types,
  const std::vector<std::optional<std::pair<int, int>>>& ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  bool use_int64_string_offsets)
{
  auto& gen = global_rng();
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(column_types.size());

  for (int c = 0; c < column_types.size(); ++c) {
    const auto& dtype = column_types[c];
    switch (dtype.id()) {
      case cudf::type_id::INT32: {
        cols.push_back(std::move(
          create_numeric_column_with_random_data<int32_t>(num_rows, dtype, ranges[c], stream, mr)));
        break;
      }
      case cudf::type_id::INT64: {
        cols.push_back(std::move(
          create_numeric_column_with_random_data<int64_t>(num_rows, dtype, ranges[c], stream, mr)));
        break;
      }
      case cudf::type_id::STRING: {
        auto dist = ranges[c].has_value()
                      ? std::uniform_int_distribution<int>(ranges[c]->first, ranges[c]->second)
                      : std::uniform_int_distribution<int>(0, 1000);
        std::vector<char> h_chars;
        std::vector<int64_t> h_offsets(num_rows + 1, 0);
        for (size_t r = 0; r < num_rows; ++r) {
          std::string h_str = "str_" + std::to_string(dist(gen));
          h_chars.insert(h_chars.end(), h_str.begin(), h_str.end());
          h_offsets[r + 1] = h_offsets[r] + static_cast<int64_t>(h_str.size());
        }

        std::unique_ptr<cudf::column> offsets_col;
        if (use_int64_string_offsets) {
          offsets_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                                  h_offsets.size(),
                                                  cudf::mask_state::UNALLOCATED,
                                                  stream,
                                                  mr);
          cudaMemcpy(offsets_col->mutable_view().data<int64_t>(),
                     h_offsets.data(),
                     sizeof(int64_t) * h_offsets.size(),
                     cudaMemcpyHostToDevice);
        } else {
          std::vector<cudf::size_type> offsets32(h_offsets.size(), 0);
          std::transform(h_offsets.begin(), h_offsets.end(), offsets32.begin(), [](int64_t value) {
            return static_cast<cudf::size_type>(value);
          });
          offsets_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                  h_offsets.size(),
                                                  cudf::mask_state::UNALLOCATED,
                                                  stream,
                                                  mr);
          cudaMemcpy(offsets_col->mutable_view().data<cudf::size_type>(),
                     offsets32.data(),
                     sizeof(cudf::size_type) * offsets32.size(),
                     cudaMemcpyHostToDevice);
        }

        rmm::device_buffer d_chars(h_chars.data(), h_chars.size(), stream, mr);

        auto col = cudf::make_strings_column(
          num_rows, std::move(offsets_col), std::move(d_chars), 0, rmm::device_buffer{});
        cols.push_back(std::move(col));
        break;
      }
      default:
        throw std::runtime_error("Unsupported cudf::data_type in `make_random_cudf_table()`");
    }
  }

  return std::make_unique<cudf::table>(std::move(cols));
}

}  // namespace sirius
