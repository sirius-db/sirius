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

#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/device_scalar.hpp>

#include <cuda_runtime_api.h>

#include <op/aggregate/sorted_keys.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace sirius::op {

namespace {

void check_cuda(cudaError_t status)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("sorted_keys CUDA error: ") + cudaGetErrorString(status));
  }
}

template <typename T>
__global__ void find_descent_kernel(T const* values, cudf::size_type num_rows, int* descended)
{
  auto const stride = static_cast<cudf::size_type>(blockDim.x * gridDim.x);
  for (auto i = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x) + 1;
       i < num_rows;
       i += stride) {
    if (values[i] < values[i - 1]) {
      *descended = 1;
      return;
    }
  }
}

template <typename T>
bool typed_ascending(cudf::column_view const& keys, ::cuda::stream_ref stream)
{
  constexpr int block_size = 256;
  auto const num_rows      = keys.size();
  int const num_blocks =
    static_cast<int>(std::min<std::int64_t>((num_rows + block_size - 1) / block_size, 4096));
  int const not_descended = 0;
  rmm::device_scalar<int> descended(not_descended, stream);
  find_descent_kernel<T>
    <<<num_blocks, block_size, 0, stream.get()>>>(keys.data<T>(), num_rows, descended.data());
  check_cuda(cudaGetLastError());
  return descended.value(stream) == 0;
}

/// Comparison type for the typed pass: the column's storage type with its signedness, or nullopt
/// when the column needs cuDF's comparator.
enum class compare_as { i8, i16, i32, i64, i128, u8, u16, u32, u64 };

std::optional<compare_as> typed_compare(cudf::data_type type)
{
  using cudf::type_id;
  switch (type.id()) {
    case type_id::INT8: return compare_as::i8;
    case type_id::INT16: return compare_as::i16;
    case type_id::INT32:
    case type_id::TIMESTAMP_DAYS:
    case type_id::DURATION_DAYS:
    case type_id::DECIMAL32: return compare_as::i32;
    case type_id::INT64:
    case type_id::TIMESTAMP_SECONDS:
    case type_id::TIMESTAMP_MILLISECONDS:
    case type_id::TIMESTAMP_MICROSECONDS:
    case type_id::TIMESTAMP_NANOSECONDS:
    case type_id::DURATION_SECONDS:
    case type_id::DURATION_MILLISECONDS:
    case type_id::DURATION_MICROSECONDS:
    case type_id::DURATION_NANOSECONDS:
    case type_id::DECIMAL64: return compare_as::i64;
    case type_id::DECIMAL128: return compare_as::i128;
    case type_id::BOOL8:
    case type_id::UINT8: return compare_as::u8;
    case type_id::UINT16: return compare_as::u16;
    case type_id::UINT32: return compare_as::u32;
    case type_id::UINT64: return compare_as::u64;
    default: return std::nullopt;
  }
}

/// Calls @p fn with a value of the comparison type for @p compare.
template <typename Fn>
decltype(auto) with_compare_type(compare_as compare, Fn&& fn)
{
  switch (compare) {
    case compare_as::i8: return fn(std::int8_t{});
    case compare_as::i16: return fn(std::int16_t{});
    case compare_as::i32: return fn(std::int32_t{});
    case compare_as::i64: return fn(std::int64_t{});
    case compare_as::i128: return fn(__int128_t{});
    case compare_as::u8: return fn(std::uint8_t{});
    case compare_as::u16: return fn(std::uint16_t{});
    case compare_as::u32: return fn(std::uint32_t{});
    case compare_as::u64: return fn(std::uint64_t{});
  }
  throw std::logic_error("sorted_keys: unknown comparison type");
}

}  // namespace

bool keys_ascending(cudf::column_view const& keys, ::cuda::stream_ref stream)
{
  if (keys.size() < 2) { return true; }
  auto const compare = keys.has_nulls() ? std::nullopt : typed_compare(keys.type());
  if (!compare) {
    return cudf::is_sorted(
      cudf::table_view({keys}), {cudf::order::ASCENDING}, {cudf::null_order::AFTER}, stream);
  }
  return with_compare_type(*compare,
                           [&](auto tag) { return typed_ascending<decltype(tag)>(keys, stream); });
}

}  // namespace sirius::op
