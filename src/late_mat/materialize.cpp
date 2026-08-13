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

#include "late_mat/materialize.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <stdexcept>
#include <string>
#include <utility>

namespace sirius::late_mat {

namespace {

/// A batch's surviving rows, as a column of its own.
///
/// Dense batches are copied rather than gathered: the gather map would be the
/// identity, and building one to apply it is strictly more work than the copy.
std::unique_ptr<cudf::column> materialize_batch(batch_source const& source,
                                                batch_selection const& selection,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  if (source.is_compressed()) {
    throw std::runtime_error(
      "late_mat::materialize: a compressed origin needs a decompression entry point that takes a "
      "selection; refusing rather than decoding the batch full width");
  }

  auto const source_table = cudf::table_view{{source.uncompressed}};
  if (selection.dense) {
    auto copied = std::make_unique<cudf::table>(source_table, stream, mr);
    return std::move(copied->release().front());
  }

  // The map is the batch-local index list the selection already built, so no
  // conversion happens here. DONT_CHECK because prepare_selection has already
  // refused any id outside the batch — checking again would cost a pass to
  // re-establish something already guaranteed.
  auto const map = cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                     static_cast<cudf::size_type>(selection.survivors),
                                     selection.local_indices.data(),
                                     nullptr,
                                     0};
  auto gathered =
    cudf::gather(source_table, map, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  return std::move(gathered->release().front());
}

}  // namespace

std::unique_ptr<cudf::column> materialize(pinned_column_view const& column,
                                          prepared_selection const& selection,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  if (column.batches.size() != selection.batches.size()) {
    throw std::runtime_error(
      "late_mat::materialize: the column's batches do not match the "
      "layout the selection was prepared against");
  }
  for (std::size_t b = 0; b < column.batches.size(); ++b) {
    if (column.batches[b].num_rows != selection.layout.batch_rows[b]) {
      throw std::runtime_error("late_mat::materialize: batch " + std::to_string(b) +
                               " has a different row count than the prepared layout");
    }
  }

  if (selection.total_survivors == 0) { return cudf::make_empty_column(column.dtype); }

  // In pin order, so the assembled column is in pinned-table order.
  std::vector<std::unique_ptr<cudf::column>> pieces;
  pieces.reserve(column.batches.size());
  for (std::size_t b = 0; b < column.batches.size(); ++b) {
    if (selection.batches[b].survivors == 0) { continue; }
    pieces.push_back(materialize_batch(column.batches[b], selection.batches[b], stream, mr));
  }

  std::unique_ptr<cudf::column> assembled;
  if (pieces.size() == 1) {
    assembled = std::move(pieces.front());
  } else {
    std::vector<cudf::column_view> views;
    views.reserve(pieces.size());
    for (auto const& p : pieces) {
      views.push_back(p->view());
    }
    assembled = cudf::concatenate(views, stream, mr);
  }

  if (!selection.needs_restore()) { return assembled; }

  // Back into the caller's order, repeats included. This gather is over the
  // materialized column — narrow, and already reduced to the surviving rows —
  // which is why deduplicating up front was worth a sort.
  auto const ranks = cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(selection.original_count),
                                       selection.restore_rank.data(),
                                       nullptr,
                                       0};
  auto restored    = cudf::gather(cudf::table_view{{assembled->view()}},
                               ranks,
                               cudf::out_of_bounds_policy::DONT_CHECK,
                               stream,
                               mr);
  return std::move(restored->release().front());
}

}  // namespace sirius::late_mat
