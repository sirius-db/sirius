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

#include "op/aggregate/aggregate_op_util.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/sort.h>

#include <type_traits>

namespace sirius::op {

namespace {

template <typename T, typename = void>
struct storage_of {
  using type = cudf::device_storage_type_t<T>;
};

template <typename T>
struct storage_of<T, std::enable_if_t<cudf::is_chrono<T>()>> {
  using type = typename T::rep;
};

struct ascending_fn {
  template <typename T>
  bool operator()(cudf::column_view const& keys, ::cuda::stream_ref stream) const
  {
    if constexpr (cudf::is_fixed_width<T>() && !std::is_floating_point_v<T>) {
      // The integer the type stores its values in: the chrono or fixed-point representation, or
      // the type itself. One typed pass over the values, which stops at the first descent.
      using storage      = typename storage_of<T>::type;
      auto const* values = keys.data<storage>();
      return thrust::is_sorted(rmm::exec_policy(stream), values, values + keys.size());
    } else {
      return cudf::is_sorted(
        cudf::table_view({keys}), {cudf::order::ASCENDING}, {cudf::null_order::AFTER}, stream);
    }
  }
};

}  // namespace

bool keys_ascending(cudf::column_view const& keys, ::cuda::stream_ref stream)
{
  if (keys.size() < 2) { return true; }
  // cuDF's comparator handles nulls, floating-point (NaN ordering) and the non-fixed-width types;
  // its generic row comparison reads a plain integer column at a fraction of memory bandwidth.
  if (keys.has_nulls()) {
    return cudf::is_sorted(
      cudf::table_view({keys}), {cudf::order::ASCENDING}, {cudf::null_order::AFTER}, stream);
  }
  return cudf::type_dispatcher(keys.type(), ascending_fn{}, keys, stream);
}

}  // namespace sirius::op
