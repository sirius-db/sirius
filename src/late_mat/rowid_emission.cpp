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

#include "rowid_emission.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>

#include <rmm/device_buffer.hpp>

#include <stdexcept>
#include <string>

namespace sirius::late_mat {

namespace {

[[noreturn]] void fail(std::string const& what)
{
  throw std::runtime_error("late_mat rowid emission: " + what);
}

template <typename W>
std::unique_ptr<cudf::column> emit_typed(rowid_emission_request const& req,
                                         std::int64_t n_rows,
                                         cudf::type_id out_type,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  if (req.mask == nullptr) {
    // Dense: batch row k IS global row range.start + k.
    if (n_rows != req.range.rows) {
      fail("dense batch rows " + std::to_string(n_rows) + " != origin range rows " +
           std::to_string(req.range.rows));
    }
    return cudf::sequence(static_cast<cudf::size_type>(n_rows),
                          cudf::numeric_scalar<W>(static_cast<W>(req.range.start), true, stream),
                          cudf::numeric_scalar<W>(W{1}, true, stream),
                          stream,
                          mr);
  }
  // Fused-compacted: expand the captured wave-1 mask to batch-local survivor
  // ids (shipped kernel), then add the chunk's global base.
  if (req.mask->chunk_offsets == nullptr || req.mask->survivor_count != n_rows) {
    fail("compacted batch rows " + std::to_string(n_rows) +
         " do not match the captured mask (survivors=" +
         std::to_string(req.mask->survivor_count) + ", counted=" +
         std::string(req.mask->chunk_offsets != nullptr ? "yes" : "no") + ")");
  }
  if (req.mask->num_rows != req.range.rows) {
    fail("captured mask covers " + std::to_string(req.mask->num_rows) +
         " rows but the origin range has " + std::to_string(req.range.rows));
  }
  rmm::device_buffer local_ids(sizeof(std::int32_t) * static_cast<std::size_t>(n_rows), stream,
                               mr);
  sirius::codegen::mask_to_row_indices(*req.mask,
                                       static_cast<std::int32_t*>(local_ids.data()), stream);
  cudf::column_view const local_view{cudf::data_type{cudf::type_id::INT32},
                                     static_cast<cudf::size_type>(n_rows),
                                     local_ids.data(),
                                     nullptr,
                                     0};
  // local_ids feeds the binary op enqueued on `stream` and was allocated on
  // `stream` — its stream-ordered free at scope exit is safe.
  return cudf::binary_operation(
    local_view,
    cudf::numeric_scalar<W>(static_cast<W>(req.range.start), true, stream),
    cudf::binary_operator::ADD,
    cudf::data_type{out_type},
    stream,
    mr);
}

}  // namespace

std::unique_ptr<cudf::column> emit_rowid_column(rowid_emission_request const& req,
                                                std::int64_t n_rows,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  if (n_rows < 0 || req.range.rows < 0 || req.range.start < 0) {
    fail("negative geometry");
  }
  if (req.width == rowid_width::u32) {
    // The planner asserts the TABLE total; re-check this batch's own span so
    // a mis-planned narrow request can never silently truncate.
    if (req.range.start + req.range.rows > (std::int64_t{1} << 32)) {
      fail("u32 rowid requested but the batch span [" + std::to_string(req.range.start) +
           ", +" + std::to_string(req.range.rows) + ") exceeds 2^32");
    }
    return emit_typed<std::uint32_t>(req, n_rows, cudf::type_id::UINT32, stream, mr);
  }
  return emit_typed<std::uint64_t>(req, n_rows, cudf::type_id::UINT64, stream, mr);
}

}  // namespace sirius::late_mat
