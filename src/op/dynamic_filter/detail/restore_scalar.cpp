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

#include "op/dynamic_filter/detail/restore_scalar.hpp"

#include "helper/numeric_narrowing.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>

namespace sirius::op::detail {

std::unique_ptr<cudf::scalar> restore_scalar(cudf::scalar const& bound,
                                             cudf::data_type target,
                                             ::cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  auto const one      = cudf::make_column_from_scalar(bound, 1, stream, mr);
  auto const restored = sirius::cast_through_rep(one->view(), target, stream, mr);
  return cudf::get_element(restored->view(), 0, stream, mr);
}

}  // namespace sirius::op::detail
