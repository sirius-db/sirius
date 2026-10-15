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

// sirius
#include <expression_evaluator/decimal_divide.hpp>
#include <helper/decimal_divide.hpp>
#include <sirius/exception.hpp>

// cudf
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

// rmm
#include <rmm/device_uvector.hpp>

// standard library
#include <array>
#include <cstdint>
#include <type_traits>

namespace sirius {
namespace {

constexpr int threads_per_block = 256;
constexpr int warp_size         = 32;

static_assert(threads_per_block % warp_size == 0, "every warp writes one whole mask word");
static_assert(sizeof(cudf::bitmask_type) * 8 == warp_size);

/// Indices into the kernel's counters.
enum counter : int { null_rows = 0, overflow_rows = 1, counter_count = 2 };

/// One row per thread. Each warp builds the validity word of its 32 rows with a ballot, so the
/// result mask needs no second pass.
template <typename Dividend, typename Divisor>
__global__ void decimal_divide_kernel(Dividend const* __restrict__ dividend,
                                      cudf::bitmask_type const* __restrict__ dividend_mask,
                                      cudf::size_type dividend_offset,
                                      Divisor const* __restrict__ divisor,
                                      cudf::bitmask_type const* __restrict__ divisor_mask,
                                      cudf::size_type divisor_offset,
                                      cudf::size_type num_rows,
                                      int exponent,
                                      __int128_t* __restrict__ out,
                                      cudf::bitmask_type* __restrict__ out_mask,
                                      unsigned long long* __restrict__ counters)
{
  auto const row  = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto const lane = static_cast<int>(threadIdx.x % warp_size);
  bool valid      = false;
  if (row < num_rows) {
    auto const i = static_cast<cudf::size_type>(row);
    bool const present =
      (dividend_mask == nullptr || cudf::bit_is_set(dividend_mask, dividend_offset + i)) &&
      (divisor_mask == nullptr || cudf::bit_is_set(divisor_mask, divisor_offset + i));
    if (present) {
      __int128_t quotient = 0;
      auto const status   = decimal_divide_round(static_cast<__int128_t>(dividend[i]),
                                               static_cast<__int128_t>(divisor[i]),
                                               exponent,
                                               &quotient);
      if (status == decimal_divide_status::ok) {
        out[i] = quotient;
        valid  = true;
      } else if (status == decimal_divide_status::overflow) {
        atomicAdd(&counters[overflow_rows], 1ULL);
      }
    }
  }
  auto const word       = __ballot_sync(0xffffffffU, valid);
  auto const warp_start = row - lane;
  if (lane == 0 && warp_start < num_rows) {
    out_mask[warp_start / warp_size] = word;
    auto const rows_in_word = num_rows - warp_start < warp_size ? num_rows - warp_start : warp_size;
    auto const nulls        = rows_in_word - __popc(word);
    if (nulls > 0) { atomicAdd(&counters[null_rows], static_cast<unsigned long long>(nulls)); }
  }
}

/// The scale of @p column in fractional digits: 0 for integers.
int fractional_digits(cudf::column_view const& column)
{
  return cudf::is_fixed_point(column.type()) ? -column.type().scale() : 0;
}

/// Calls @p fn with a null pointer of the storage type of @p column.
template <typename Fn>
decltype(auto) dispatch_storage(cudf::column_view const& column, Fn&& fn)
{
  switch (column.type().id()) {
    case cudf::type_id::INT32:
    case cudf::type_id::DECIMAL32: return fn(static_cast<int32_t const*>(nullptr));
    case cudf::type_id::INT64:
    case cudf::type_id::DECIMAL64: return fn(static_cast<int64_t const*>(nullptr));
    case cudf::type_id::DECIMAL128: return fn(static_cast<__int128_t const*>(nullptr));
    default:
      throw invalid_input_exception(
        "[sirius_decimal_divide] unsupported input type id {}; expected INT32, INT64 or DECIMAL",
        static_cast<int>(column.type().id()));
  }
}

}  // namespace

std::unique_ptr<cudf::column> decimal_divide(cudf::column_view const& dividend,
                                             cudf::column_view const& divisor,
                                             int32_t result_scale,
                                             ::cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  if (dividend.size() != divisor.size()) {
    throw invalid_input_exception(
      "[sirius_decimal_divide] the inputs have {} and {} rows", dividend.size(), divisor.size());
  }
  if (result_scale < 0 || result_scale > decimal_divide_max_digits) {
    throw invalid_input_exception("[sirius_decimal_divide] result scale {} is outside 0..{}",
                                  result_scale,
                                  decimal_divide_max_digits);
  }
  auto const num_rows = dividend.size();
  auto const out_type =
    cudf::data_type{cudf::type_id::DECIMAL128, static_cast<cudf::size_type>(-result_scale)};
  auto result =
    cudf::make_fixed_point_column(out_type, num_rows, cudf::mask_state::UNINITIALIZED, stream, mr);
  if (num_rows == 0) { return result; }

  // dividend * 10^exponent / divisor is the unscaled quotient at the result scale.
  auto const exponent = result_scale + fractional_digits(divisor) - fractional_digits(dividend);
  rmm::device_uvector<unsigned long long> counters(counter_count, stream, mr);
  CUDF_CUDA_TRY(
    cudaMemsetAsync(counters.data(), 0, counter_count * sizeof(unsigned long long), stream.get()));

  auto const num_blocks = static_cast<uint32_t>(
    (static_cast<int64_t>(num_rows) + threads_per_block - 1) / threads_per_block);
  auto* out      = result->mutable_view().data<__int128_t>();
  auto* out_mask = result->mutable_view().null_mask();
  dispatch_storage(dividend, [&](auto dividend_tag) {
    using Dividend = std::remove_const_t<std::remove_pointer_t<decltype(dividend_tag)>>;
    dispatch_storage(divisor, [&](auto divisor_tag) {
      using Divisor = std::remove_const_t<std::remove_pointer_t<decltype(divisor_tag)>>;
      decimal_divide_kernel<Dividend, Divisor>
        <<<num_blocks, threads_per_block, 0, stream.get()>>>(dividend.data<Dividend>(),
                                                             dividend.null_mask(),
                                                             dividend.offset(),
                                                             divisor.data<Divisor>(),
                                                             divisor.null_mask(),
                                                             divisor.offset(),
                                                             num_rows,
                                                             exponent,
                                                             out,
                                                             out_mask,
                                                             counters.data());
    });
  });
  CUDF_CHECK_CUDA(stream.get());

  std::array<unsigned long long, counter_count> host_counters{};
  CUDF_CUDA_TRY(cudaMemcpyAsync(host_counters.data(),
                                counters.data(),
                                counter_count * sizeof(unsigned long long),
                                cudaMemcpyDeviceToHost,
                                stream.get()));
  stream.sync();
  if (host_counters[overflow_rows] > 0) {
    throw invalid_input_exception(
      "[sirius_decimal_divide] {} quotients need more than {} digits at scale {}",
      host_counters[overflow_rows],
      decimal_divide_max_digits,
      result_scale);
  }
  result->set_null_count(static_cast<cudf::size_type>(host_counters[null_rows]));
  return result;
}

}  // namespace sirius
