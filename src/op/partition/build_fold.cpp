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

#include "op/partition/build_fold.hpp"

#include "memory/size_arithmetic.hpp"
#include "op/partition/config.hpp"

#include <cudf/types.hpp>

#include <limits>

namespace sirius::op {

bool decide_build_fold(build_fold_inputs const& inputs,
                       build_fold_limits limits,
                       build_fold_policy policy,
                       bool publishes_filters) noexcept
{
  constexpr auto cudf_row_limit =
    static_cast<std::uint64_t>(std::numeric_limits<cudf::size_type>::max());
  // Every folding policy needs a concatenation cuDF can hold and a resident build within budget.
  // clang-format off
  bool const fits = inputs.rows &&
                    *inputs.rows <= cudf_row_limit &&
                    inputs.resident_bytes < limits.max_resident_bytes;
  // clang-format on
  switch (policy) {
    // The legacy policy has now-anachronistic coupling: CONCAT-folding depends on whether filters
    // can be published for this join, since, previously, dynamic filters were only constructed for
    // joins in BUILD_PROBE mode, so the policy effectively encouraged BUILD_PROBE mode in order to
    // reap the benefits of dynamic filtering. Now, dynamic filters are constructed for all join
    // modes, so the coupling is no longer necessary. For the time being, keep continuity with
    // existing production, but revisit in the future.
    case build_fold_policy::legacy_filter_coupled: return publishes_filters && fits;
    case build_fold_policy::never: return false;
    case build_fold_policy::bounded:
      // clang-format off
      return fits &&
             memory::saturating_mul(*inputs.rows, config::BUILD_PROBE_HASH_TABLE_BYTES_PER_ROW) <=
               inputs.resident_bytes &&
             inputs.resident_bytes > inputs.concat_batch_bytes;
      // clang-format on
  }
  return false;
}

}  // namespace sirius::op
