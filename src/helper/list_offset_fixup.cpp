/*
 * Copyright 2025, Sirius Contributors.
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

#include "cudf/list_offset_fixup.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <rmm/device_buffer.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace sirius {

namespace {

/// True if @p view holds anything that may carry cuCascade's INT64-promoted offsets.
bool view_has_list(const cudf::column_view& view)
{
  if (view.type().id() == cudf::type_id::LIST) { return true; }
  for (auto it = view.child_begin(); it != view.child_end(); ++it) {
    if (view_has_list(*it)) { return true; }
  }
  return false;
}

bool table_has_list(const cudf::table_view& view)
{
  for (const auto& col : view) {
    if (view_has_list(col)) { return true; }
  }
  return false;
}

/// Rebuild @p col with INT32 list offsets if it is a LIST column, recursing into the values
/// child first so nested (array-of-array) lists are normalized too. Non-LIST columns are
/// returned unchanged. Moves the values child and null mask; only the offsets buffer is
/// reallocated (by cudf::cast). Assembled via the column constructor (not make_lists_column)
/// to avoid purge_nonempty_nulls, which would collapse the fixed stride of null array rows.
std::unique_ptr<cudf::column> fixup_list_offsets(std::unique_ptr<cudf::column> col,
                                                 rmm::cuda_stream_view stream,
                                                 const rmm::device_async_resource_ref& mr)
{
  if (col->type().id() != cudf::type_id::LIST) { return col; }

  auto const num_rows   = col->size();
  auto const null_count = col->null_count();

  auto contents = col->release();
  auto offsets  = std::move(contents.children[cudf::lists_column_view::offsets_column_index]);
  auto child    = std::move(contents.children[cudf::lists_column_view::child_column_index]);

  // Normalize nested lists before reassembling this level
  child = fixup_list_offsets(std::move(child), stream, mr);

  if (offsets->type().id() == cudf::type_id::INT64) {
    offsets = cudf::cast(offsets->view(), cudf::data_type{cudf::type_id::INT32}, stream, mr);
  }

  std::vector<std::unique_ptr<cudf::column>> children;
  children.reserve(2);
  children.push_back(std::move(offsets));
  children.push_back(std::move(child));

  rmm::device_buffer null_mask =
    contents.null_mask ? std::move(*contents.null_mask) : rmm::device_buffer{};

  return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::LIST},
                                        num_rows,
                                        rmm::device_buffer{},  // LIST has no flat data buffer
                                        std::move(null_mask),
                                        null_count,
                                        std::move(children));
}

}  // namespace

void normalize_gpu_list_offsets(cucascade::mutable_data_batch& batch,
                                const cucascade::memory::memory_space* gpu_space,
                                rmm::cuda_stream_view stream)
{
  if (gpu_space == nullptr) { return; }
  auto* data = batch.get_data();
  if (data == nullptr) { return; }

  auto& gpu_rep = data->cast<cucascade::gpu_table_representation>();

  // Return if there is no LIST column
  if (!table_has_list(gpu_rep.get_table_view())) { return; }

  auto mr    = gpu_space->get_default_allocator();
  auto table = gpu_rep.release_table(stream);

  auto columns = table->release();
  for (auto& col : columns) {
    col = fixup_list_offsets(std::move(col), stream, mr);
  }

  batch.set_data(std::make_unique<cucascade::gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(columns)),
    *const_cast<cucascade::memory::memory_space*>(gpu_space),
    stream));
}

}  // namespace sirius
