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

#include "late_mat/materialize.hpp"
#include "late_mat/prepared_selection.hpp"
#include "scan_manager/late_mat_resolver.hpp"

#include <cudf/column/column.hpp>

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
  if (!port_directive_matches(directive, batch)) {
    throw std::runtime_error(
      "late_mat::materialize_at_port: the batch is not the one this directive was installed for");
  }

  auto const rowids = batch.column(static_cast<cudf::size_type>(directive.rowid_position()));
  // v1 defers nothing an outer join could null, so a null here means the rowid
  // column stopped being a rowid somewhere on the ride — the one failure mode
  // that must never be papered over with a plausible row.
  if (rowids.null_count() != 0) {
    throw std::runtime_error(
      "late_mat::materialize_at_port: the rowid column carries nulls; a null rowid must "
      "materialize a null value, which is not implemented");
  }

  auto layout = scan_manager::resolve_pinned_layout(directive.origins.front());
  if (!layout) {
    throw std::runtime_error(
      "late_mat::materialize_at_port: the origin pin no longer resolves; the deferred values "
      "exist nowhere else");
  }

  // The ids are BORROWED from the batch, which outlives this call — that is what
  // lets an uncompressed column gather straight off them.
  prepared_selection selection(
    std::move(*layout),
    row_id_list{rowids.data<std::uint64_t>(), rowids.size(), /*sorted_unique=*/false});

  std::vector<std::unique_ptr<cudf::column>> restored;
  restored.reserve(directive.output_positions.size());
  for (std::size_t i = 0; i < directive.output_positions.size(); ++i) {
    auto column = scan_manager::resolve_pinned_column(directive.origins[i]);
    if (!column) {
      throw std::runtime_error("late_mat::materialize_at_port: origin column " + std::to_string(i) +
                               " no longer resolves");
    }
    auto produced = materialize(*column, selection, stream, mr);
    if (produced->type() != directive.restored_types[i]) {
      throw std::runtime_error("late_mat::materialize_at_port: origin column " + std::to_string(i) +
                               " came back as a different type than the scan gave up");
    }
    restored.push_back(std::move(produced));
  }

  // Splice: every other position is copied through. The copy is of the batch as
  // it rides — the deferred columns are still a rowid and placeholders here, so
  // the wide data is not among what is copied.
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(static_cast<std::size_t>(batch.num_columns()));
  std::size_t next_restored = 0;
  for (cudf::size_type position = 0; position < batch.num_columns(); ++position) {
    if (next_restored < directive.output_positions.size() &&
        directive.output_positions[next_restored] == static_cast<std::size_t>(position)) {
      columns.push_back(std::move(restored[next_restored]));
      ++next_restored;
      continue;
    }
    columns.push_back(std::make_unique<cudf::column>(batch.column(position), stream, mr));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace sirius::late_mat
