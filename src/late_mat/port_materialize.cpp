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

#include "late_mat/port_materialize.hpp"

#include <nvtx3/nvtx3.hpp>

#include "helper/numeric_narrowing.hpp"
#include "late_mat/materialize.hpp"
#include "late_mat/prepared_selection.hpp"
#include "scan_manager/late_mat_resolver.hpp"

#include <cudf/column/column.hpp>
#include <cudf/unary.hpp>

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius::late_mat {

namespace {

std::vector<cudf::data_type> schema_of(cudf::table_view const& batch)
{
  std::vector<cudf::data_type> schema;
  schema.reserve(static_cast<std::size_t>(batch.num_columns()));
  for (auto const& column : batch) {
    schema.push_back(column.type());
  }
  return schema;
}

/// Restore one bundle's columns: resolve its pinned layout, build the selection
/// from ITS rowid column, and gather. Writes each restored column into
/// @p restored_by_position, keyed by the port position it belongs at.
///
/// One bundle per pinned table, because a rowid means nothing outside the table
/// it indexes: a rider's ids address ITS pin, and mixing the two would gather
/// arbitrary rows that look entirely plausible.
void restore_bundle(std::vector<std::size_t> const& output_positions,
                    std::size_t rowid_at,
                    std::vector<column_origin> const& origins,
                    std::vector<cudf::data_type> const& restored_types,
                    cudf::table_view const& batch,
                    std::map<std::size_t, std::unique_ptr<cudf::column>>& restored_by_position,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr)
{
  auto const rowids = batch.column(static_cast<cudf::size_type>(rowid_at));
  // Nothing an outer join could null is ever deferred, so a null here means the
  // rowid column stopped being a rowid somewhere on the ride — the one failure
  // mode that must never be papered over with a plausible row.
  if (rowids.null_count() != 0) {
    throw std::runtime_error(
      "late_mat::materialize_at_port: the rowid column carries nulls; a null rowid must "
      "materialize a null value, which is not implemented");
  }

  auto layout = scan_manager::resolve_pinned_layout(origins.front());
  if (!layout) {
    throw std::runtime_error(
      "late_mat::materialize_at_port: the origin pin no longer resolves; the deferred values "
      "exist nowhere else");
  }

  // A narrow rowid rode as UINT32 to halve the bytes on the ride; the gather
  // speaks one width, so widen it here. The cast is over the batch AT THE PORT,
  // which is the far, already-reduced end — the cheap place to pay it.
  std::unique_ptr<cudf::column> widened;
  if (rowids.type().id() == kNarrowRowidType) {
    widened = cudf::cast(rowids, cudf::data_type{kRowidType}, stream, mr);
  }
  auto const ids = widened ? widened->view() : rowids;

  // The ids are BORROWED from the batch (or from `widened`, which outlives this
  // call) — that is what lets an uncompressed column gather straight off them.
  prepared_selection selection(
    std::move(*layout),
    row_id_list{ids.data<std::uint64_t>(), ids.size(), /*sorted_unique=*/false});

  for (std::size_t i = 0; i < output_positions.size(); ++i) {
    auto column = scan_manager::resolve_pinned_column(origins[i]);
    if (!column) {
      throw std::runtime_error("late_mat::materialize_at_port: origin column " + std::to_string(i) +
                               " no longer resolves");
    }
    auto produced = materialize(*column, selection, stream, mr);
    // A narrow-stored pin may come back a strictly narrower carrier of the
    // restored type, and only that; anything else is not what was deferred.
    if (produced->type() != restored_types[i]) {
      if (!sirius::can_restore_to(produced->type(), restored_types[i])) {
        throw std::runtime_error("late_mat::materialize_at_port: origin column " +
                                 std::to_string(i) +
                                 " came back as a different type than the scan gave up");
      }
      produced = sirius::cast_through_rep(produced->view(), restored_types[i], stream, mr);
    }
    restored_by_position.emplace(output_positions[i], std::move(produced));
  }
}

}  // namespace

bool port_directive_matches(port_materialize_directive const& directive,
                            cudf::table_view const& batch)
{
  return directive.matches(schema_of(batch));
}

std::unique_ptr<cudf::table> materialize_at_port(port_materialize_directive const& directive,
                                                 cudf::table_view const& batch,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  nvtx3::scoped_range nvtx_range{"sirius::late_mat::materialize_at_port"};
  if (!port_directive_matches(directive, batch)) {
    throw std::runtime_error(
      "late_mat::materialize_at_port: the batch is not the one this directive was installed for");
  }

  // Position-keyed rather than parallel-to-positions: with riders the restored
  // columns come from several bundles and interleave arbitrarily at the port.
  std::map<std::size_t, std::unique_ptr<cudf::column>> restored_by_position;
  restore_bundle(directive.output_positions,
                 directive.rowid_at,
                 directive.origins,
                 directive.restored_types,
                 batch,
                 restored_by_position,
                 stream,
                 mr);
  for (auto const& rider : directive.riders) {
    restore_bundle(rider.output_positions,
                   rider.rowid_at,
                   rider.origins,
                   rider.restored_types,
                   batch,
                   restored_by_position,
                   stream,
                   mr);
  }

  // Splice: every other position is copied through. The copy is of the batch as
  // it rides — the deferred columns are still rowids and placeholders here, so
  // the wide data is not among what is copied.
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(static_cast<std::size_t>(batch.num_columns()));
  for (cudf::size_type position = 0; position < batch.num_columns(); ++position) {
    auto found = restored_by_position.find(static_cast<std::size_t>(position));
    if (found != restored_by_position.end()) {
      columns.push_back(std::move(found->second));
      continue;
    }
    columns.push_back(std::make_unique<cudf::column>(batch.column(position), stream, mr));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace sirius::late_mat
