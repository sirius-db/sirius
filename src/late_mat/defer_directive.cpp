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
#include <atomic>

namespace sirius::late_mat {

namespace {
std::atomic<std::uint64_t>& install_counter() noexcept
{
  static std::atomic<std::uint64_t> installed{0};
  return installed;
}
}  // namespace

std::uint64_t deferrals_installed() noexcept
{
  return install_counter().load(std::memory_order_relaxed);
}

void note_deferral_installed() noexcept
{
  install_counter().fetch_add(1, std::memory_order_relaxed);
}

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
  if (std::find(output_positions.begin(), output_positions.end(), rowid_at) ==
      output_positions.end()) {
    return false;
  }

  for (std::size_t i = 0; i < output_positions.size(); ++i) {
    auto const pos = output_positions[i];
    if (pos >= expected_schema.size()) { return false; }
    // Ascending and distinct, so a position cannot be restored twice.
    if (i > 0 && pos <= output_positions[i - 1]) { return false; }
    // The schema has to carry what the scan side substituted, or the two halves
    // disagree about what is riding where.
    auto const want = (pos == rowid_at) ? kRowidType : kPlaceholderType;
    if (expected_schema[pos].id() != want) { return false; }
    if (!origins[i].has_origin()) { return false; }
  }
  return true;
}

bool defer_pair::valid() const
{
  // The halves speak different coordinate systems by design, so what has to
  // agree is the SIZE of the bundle, not the positions themselves.
  return !scan.empty() && port.valid() &&
         scan.output_positions.size() == port.output_positions.size();
}

defer_pair make_defer_pair(std::vector<cudf::data_type> const& scan_schema,
                           std::vector<std::size_t> const& scan_positions,
                           std::vector<cudf::data_type> const& port_schema,
                           std::vector<std::size_t> const& port_positions,
                           std::vector<column_origin> const& origins)
{
  defer_pair pair;
  if (scan_positions.empty() || scan_positions.size() != origins.size() ||
      scan_positions.size() != port_positions.size()) {
    return pair;
  }

  /// One deferred column, carrying both of its coordinates. The scan side
  /// arrives ascending (the walk reports lifetimes in output order); the port
  /// side may be in any order, so it is sorted below rather than assumed.
  struct deferred_column {
    std::size_t port_position;
    cudf::data_type restored;
    column_origin origin;
  };
  std::vector<deferred_column> columns;
  columns.reserve(scan_positions.size());

  auto scan_substituted = scan_schema;
  for (std::size_t i = 0; i < scan_positions.size(); ++i) {
    auto const scan_position = scan_positions[i];
    auto const port_position = port_positions[i];
    if (scan_position >= scan_schema.size() || port_position >= port_schema.size()) { return {}; }
    if (i > 0 && scan_position <= scan_positions[i - 1]) { return {}; }
    // A column's type may not change on the ride: it is what the port must hand
    // back, and the scan is what gave it up.
    if (port_schema[port_position] != scan_schema[scan_position]) { return {}; }
    // The rowid rides at the FIRST deferred scan position; the rest become
    // placeholders, which exist only to keep the arity and the positions after
    // them where they were.
    scan_substituted[scan_position] = cudf::data_type{(i == 0) ? kRowidType : kPlaceholderType};
    columns.push_back(deferred_column{port_position, scan_schema[scan_position], origins[i]});
  }

  // Where the rowid ends up at the port is wherever THAT column travelled to —
  // not the front of the bundle, which the ride may have reordered.
  pair.port.rowid_at    = columns.front().port_position;
  auto port_substituted = port_schema;
  for (auto const& column : columns) {
    port_substituted[column.port_position] =
      cudf::data_type{(column.port_position == pair.port.rowid_at) ? kRowidType : kPlaceholderType};
  }

  std::sort(columns.begin(), columns.end(), [](auto const& lhs, auto const& rhs) {
    return lhs.port_position < rhs.port_position;
  });
  for (auto& column : columns) {
    pair.port.output_positions.push_back(column.port_position);
    pair.port.restored_types.push_back(column.restored);
    pair.port.origins.push_back(std::move(column.origin));
  }

  pair.scan.output_positions = scan_positions;
  pair.port.expected_schema  = std::move(port_substituted);
  // Built, then checked — so a caller cannot install a pair that only looks
  // right because the builder was trusted.
  if (!pair.valid()) { return {}; }
  return pair;
}

}  // namespace sirius::late_mat
