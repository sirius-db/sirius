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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/utility>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <expression_evaluator/integer_cast.hpp>
#include <sirius/exception.hpp>

#include <type_traits>

namespace sirius {
namespace {

struct convert_integer {
  template <typename Source, typename Target>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           cudf::data_type target,
                                           bool try_cast,
                                           ::cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr) const
  {
    if constexpr (std::is_integral_v<Source> && std::is_integral_v<Target> &&
                  !std::is_same_v<Source, bool> && !std::is_same_v<Target, bool>) {
      auto output =
        cudf::make_numeric_column(target, input.size(), cudf::mask_state::UNALLOCATED, stream, mr);
      auto* dst            = output->mutable_view().data<Target>();
      auto const* src      = input.data<Source>();
      auto const* validity = input.null_mask();
      auto const offset    = input.offset();

      auto begin = thrust::make_counting_iterator<cudf::size_type>(0);
      if (!try_cast) {
        if (input.size() == 0) { return output; }
        // A successful strict cast preserves input validity. Only overflow needs reporting.
        rmm::device_scalar<int> overflow(stream, mr);
        overflow.set_value_to_zero_async(stream);
        auto* flag = overflow.data();
        thrust::for_each_n(rmm::exec_policy_nosync(stream, mr),
                           begin,
                           input.size(),
                           [=] __device__(cudf::size_type i) {
                             if (validity && !cudf::bit_is_set(validity, offset + i)) {
                               dst[i] = Target{};
                               return;
                             }
                             auto const value = src[i];
                             if (!cuda::std::in_range<Target>(value)) { atomicExch(flag, 1); }
                             dst[i] = static_cast<Target>(value);
                           });
        if (input.has_nulls()) {
          output->set_null_mask(cudf::copy_bitmask(input, stream, mr), input.null_count());
        }
        if (overflow.value(stream)) {
          throw not_implemented_exception("Integer value out of range; falling back to DuckDB");
        }
        return output;
      }

      auto [mask, nulls] = cudf::detail::valid_if(
        begin,
        begin + input.size(),
        [=] __device__(cudf::size_type i) {
          dst[i] = Target{};
          if (validity && !cudf::bit_is_set(validity, offset + i)) { return false; }
          auto value = src[i];
          if (!cuda::std::in_range<Target>(value)) { return false; }
          dst[i] = static_cast<Target>(value);
          return true;
        },
        stream,
        cudf::memory_resources{mr, mr});
      if (nulls != 0) { output->set_null_mask(std::move(mask), nulls); }
      return output;
    } else {
      throw internal_exception("Expected native integer carriers for checked integer cast");
    }
  }
};

}  // namespace

std::unique_ptr<cudf::column> cast_checked_integer(cudf::column_view const& input,
                                                   cudf::data_type target,
                                                   bool try_cast,
                                                   ::cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  return cudf::double_type_dispatcher(
    input.type(), target, convert_integer{}, input, target, try_cast, stream, mr);
}

}  // namespace sirius
