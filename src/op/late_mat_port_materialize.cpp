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

#include "op/late_mat_port_materialize.hpp"

#include "late_mat/late_materializer.hpp"
#include "log/logging.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::op {

namespace {

/// Placeholder-signature check, hardened per review finding F1: the ENTIRE table schema
/// must equal the expected post-substitution schema (producer's planned
/// types with deferred positions as UINT64/INT8). A directive without a full
/// schema never matches — no partial-signature matching exists anymore.
bool matches_placeholder_signature(cudf::table_view const& view,
                                   late_mat::port_materialize_directive const& d)
{
  if (d.expected_types.empty()) { return false; }
  if (static_cast<std::size_t>(view.num_columns()) != d.expected_types.size()) { return false; }
  for (cudf::size_type c = 0; c < view.num_columns(); ++c) {
    if (view.column(c).type().id() != d.expected_types[static_cast<std::size_t>(c)]) {
      return false;
    }
  }
  return true;
}

}  // namespace

::cucascade::read_only_data_batch late_mat_apply_port_directive(
  ::cucascade::read_only_data_batch ro,
  late_mat::port_materialize_directive const& directive,
  rmm::cuda_stream_view stream)
{
  auto const* gpu_rep = dynamic_cast<::cucascade::gpu_table_representation const*>(ro.get_data());
  if (gpu_rep == nullptr) { return ro; }
  if (!matches_placeholder_signature(gpu_rep->get_table_view(), directive)) { return ro; }

  // Review finding F3: re-check every origin's generation at USE time (the directive's
  // views were resolved at install; a re-pin since then must fail closed,
  // per the origin contract in late_mat/column_origin.hpp).
  for (auto const& origin : directive.origins) {
    if (origin.resolve() == nullptr) {
      throw std::runtime_error(
        "[late_mat_apply_port_directive] origin generation changed since the directive was "
        "installed (re-pin?); refusing stale pinned views");
    }
  }

  // Matched: this is a deferred batch. Upgrade, materialize, splice, downgrade.
  auto mut  = ::cucascade::data_batch::readonly_to_mutable(std::move(ro));
  auto* rep = dynamic_cast<::cucascade::gpu_table_representation*>(mut.get_data());
  if (rep == nullptr) {
    throw std::runtime_error(
      "[late_mat_apply_port_directive] representation changed across the lock upgrade");
  }
  auto& space = rep->get_memory_space();
  auto mr     = space.get_default_allocator();

  auto view             = rep->get_table_view();
  auto const rowid_view = view.column(static_cast<cudf::size_type>(directive.rowid_position));
  if (rowid_view.null_count() != 0) {
    throw std::runtime_error("[late_mat_apply_port_directive] rowid column carries nulls");
  }

  // Review finding F4: a 0-row deferred batch never calls prepare/materialize — splice
  // empty columns of the materialized types and return.
  if (rowid_view.size() == 0) {
    auto owned = rep->release_table(stream);
    auto cols  = owned->release();
    for (std::size_t i = 0; i < directive.positions.size(); ++i) {
      cols[directive.positions[i]] = cudf::make_empty_column(directive.columns[i].dtype);
    }
    mut.set_data(std::make_unique<::cucascade::gpu_table_representation>(
      std::make_unique<cudf::table>(std::move(cols)), space, stream));
    return ::cucascade::data_batch::mutable_to_readonly(std::move(mut));
  }

  // Review-F1 content check: rowid values must address the pinned table. One
  // minmax reduce + tiny D2H at a pipeline-boundary port (negligible); an
  // out-of-range id — the loud face of any signature coincidence — throws
  // here instead of deep inside the materializer.
  {
    auto const total_rows =
      directive.layout.batch_row_start.empty() ? 0 : directive.layout.batch_row_start.back();
    auto const minmax = cudf::minmax(rowid_view, stream, mr);
    auto const max_id =
      static_cast<cudf::numeric_scalar<std::uint64_t> const&>(*minmax.second).value(stream);
    if (max_id >= static_cast<std::uint64_t>(total_rows)) {
      throw std::runtime_error(
        "[late_mat_apply_port_directive] rowid max " + std::to_string(max_id) +
        " out of range for pinned table with " + std::to_string(total_rows) +
        " rows — placeholder signature matched a non-deferred batch?");
    }
  }

  // One prepared selection serves every deferred column of this batch. The
  // rowid values are pin-order global ids in post-join/partition order:
  // unsorted, duplicates legal (gather semantics restore caller order).
  late_mat::row_id_list ids;
  ids.ids           = rowid_view.data<std::uint64_t>();
  ids.count         = static_cast<std::int64_t>(rowid_view.size());
  ids.sorted_unique = false;
  auto const sel    = late_mat::prepare_selection(directive.layout, ids, stream, mr);

  // Take ownership of the table (owned on the v1 paths — CONCAT/join outputs;
  // a view-backed batch materializes here, correct if slower) and splice the
  // materialized columns over the placeholders.
  auto owned = rep->release_table(stream);
  auto cols  = owned->release();
  std::vector<std::unique_ptr<cudf::column>> placeholders;  // kept alive past the enqueues
  placeholders.reserve(directive.positions.size());
  for (std::size_t i = 0; i < directive.positions.size(); ++i) {
    auto materialized = late_mat::materialize(directive.columns[i], *sel, stream, mr);
    placeholders.push_back(std::exchange(cols[directive.positions[i]], std::move(materialized)));
  }
  // NO host sync here (sync-removal review item): all reads of the placeholder buffers
  // are enqueued on `stream` above, so freeing them is safe iff their
  // deallocation is stream-ordered on the SAME stream. The common case
  // already is (lock_or_prepare_batch's opportunistic rebind moved the
  // batch's buffers onto this task's stream), but the rebind is best-effort —
  // so rebind each placeholder's buffers to `stream` explicitly before
  // letting them die. Frees then enqueue after the reads, no sync needed.
  for (auto& placeholder : placeholders) {
    auto contents = placeholder->release();
    if (contents.data) { contents.data->set_stream(stream); }
    if (contents.null_mask) { contents.null_mask->set_stream(stream); }
    // fixed-width UINT64/INT8 placeholders have no children
  }
  placeholders.clear();

  mut.set_data(std::make_unique<::cucascade::gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(cols)), space, stream));
  return ::cucascade::data_batch::mutable_to_readonly(std::move(mut));
}

}  // namespace sirius::op
