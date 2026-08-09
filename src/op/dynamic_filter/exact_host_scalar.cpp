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

#include <op/dynamic_filter/exact_host_scalar.hpp>

namespace sirius::op {

int exact_host_scalar::compare(exact_host_scalar const& other, cudf::order order) const noexcept
{
  // Every admitted alternative fits std::int64_t exactly, so widening loses nothing.
  auto const to_int64 = [](value_type const& v) {
    return std::visit([](auto value) { return static_cast<std::int64_t>(value); }, v);
  };
  auto const lhs = to_int64(_value);
  auto const rhs = to_int64(other._value);
  int const cmp  = lhs < rhs ? -1 : (lhs > rhs ? 1 : 0);
  return order == cudf::order::DESCENDING ? -cmp : cmp;
}

int exact_host_key_tuple::lex_compare(exact_host_key_tuple const& other,
                                      std::span<top_n_key_semantics const> semantics) const noexcept
{
  for (std::size_t i = 0; i < _components.size(); ++i) {
    auto const& sem = semantics[i];
    auto const& lhs = _components[i];
    auto const& rhs = other._components[i];

    int cmp = 0;
    if (lhs && rhs) {
      cmp = lhs->compare(*rhs, sem.order);
    } else if (lhs.has_value() != rhs.has_value()) {
      // cuDF contract: null precedence in the ascending frame, then reverse for DESCENDING.
      cmp = (sem.null_order == cudf::null_order::BEFORE) == !lhs.has_value() ? -1 : 1;
      if (sem.order == cudf::order::DESCENDING) { cmp = -cmp; }
    }
    if (cmp != 0) { return cmp; }
  }
  return 0;
}

}  // namespace sirius::op
