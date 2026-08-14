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

#include "late_mat/defer_directive.hpp"

#include <algorithm>

namespace sirius::late_mat {

bool deferred_scan_output::defers(std::size_t position) const noexcept
{
  return std::find(output_positions.begin(), output_positions.end(), position) !=
         output_positions.end();
}

bool port_materialize_directive::matches(std::vector<cudf::data_type> const& schema) const
{
  return !empty() && schema == expected_schema;
}

bool port_materialize_directive::valid() const
{
  if (empty()) { return false; }
  if (output_positions.size() != origins.size()) { return false; }
  if (output_positions.size() != restored_types.size()) { return false; }

  for (std::size_t i = 0; i < output_positions.size(); ++i) {
    auto const pos = output_positions[i];
    if (pos >= expected_schema.size()) { return false; }
    // Ascending and distinct, so the rowid is unambiguously the first and a
    // position cannot be restored twice.
    if (i > 0 && pos <= output_positions[i - 1]) { return false; }
    // The schema has to carry what the scan side substituted, or the two halves
    // disagree about what is riding where.
    auto const want = (i == 0) ? kRowidType : kPlaceholderType;
    if (expected_schema[pos].id() != want) { return false; }
    if (!origins[i].has_origin()) { return false; }
  }
  return true;
}

bool defer_pair::valid() const
{
  return !scan.empty() && port.valid() && scan.output_positions == port.output_positions;
}

defer_pair make_defer_pair(std::vector<cudf::data_type> const& schema,
                           std::vector<std::size_t> const& positions,
                           std::vector<column_origin> const& origins)
{
  defer_pair pair;
  if (positions.empty() || positions.size() != origins.size()) { return pair; }

  auto substituted = schema;
  for (std::size_t i = 0; i < positions.size(); ++i) {
    auto const pos = positions[i];
    if (pos >= schema.size()) { return {}; }
    if (i > 0 && pos <= positions[i - 1]) { return {}; }
    substituted[pos] = cudf::data_type{(i == 0) ? kRowidType : kPlaceholderType};
    pair.port.restored_types.push_back(schema[pos]);
  }

  pair.scan.output_positions = positions;
  pair.port.output_positions = positions;
  pair.port.origins          = origins;
  pair.port.expected_schema  = std::move(substituted);
  // Built, then checked — so a caller cannot install a pair that only looks
  // right because the builder was trusted.
  if (!pair.valid()) { return {}; }
  return pair;
}

}  // namespace sirius::late_mat
