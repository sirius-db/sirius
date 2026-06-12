// SPDX-License-Identifier: Apache-2.0
#include "codegen/plan/column_copy.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace simpatico {
namespace {

std::unique_ptr<cudf::column> copy_column_view_impl(cudf::column_view const& view,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  cudf::data_type const dt = view.type();
  bool const is_fixed      = (dt.id() == cudf::type_id::UINT8 || dt.id() == cudf::type_id::INT8 ||
                         dt.id() == cudf::type_id::INT16 || dt.id() == cudf::type_id::UINT16 ||
                         dt.id() == cudf::type_id::INT32 || dt.id() == cudf::type_id::UINT32 ||
                         dt.id() == cudf::type_id::INT64 || dt.id() == cudf::type_id::UINT64 ||
                         dt.id() == cudf::type_id::FLOAT32 || dt.id() == cudf::type_id::FLOAT64);
  if (is_fixed) {
    cudf::size_type const n = view.size();
    auto col = cudf::make_fixed_width_column(dt, n, cudf::mask_state::UNALLOCATED, stream, mr);
    if (n > 0) {
      size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(cudf::size_of(dt));
      cudaMemcpyAsync(col->mutable_view().head<void>(),
                      view.head<void>(),
                      bytes,
                      cudaMemcpyDeviceToDevice,
                      stream.value());
      cudaStreamSynchronize(stream.value());
    }
    return col;
  }
  // Non-fixed-width (e.g. STRING, LIST): copy via gather with identity indices
  // to avoid relying on allocate_like/copy_range_in_place (API varies by libcudf version).
  cudf::size_type const n = view.size();
  if (n == 0) { return cudf::empty_like(view); }
  auto indices_col = cudf::make_fixed_width_column(
    cudf::data_type(cudf::type_id::INT32), n, cudf::mask_state::UNALLOCATED, stream, mr);
  std::vector<int32_t> h_indices(static_cast<size_t>(n));
  for (cudf::size_type i = 0; i < n; ++i) {
    h_indices[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }
  cudaMemcpyAsync(indices_col->mutable_view().head<int32_t>(),
                  h_indices.data(),
                  static_cast<size_t>(n) * sizeof(int32_t),
                  cudaMemcpyHostToDevice,
                  stream.value());
  cudaStreamSynchronize(stream.value());
  cudf::table_view single(std::vector<cudf::column_view>{view});
  auto gathered =
    cudf::gather(single, indices_col->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  auto cols = gathered->release();
  return std::move(cols[0]);
}

}  // namespace

std::unique_ptr<cudf::column> copy_column_view(cudf::column_view const& view,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  if (view.type().id() == cudf::type_id::LIST && view.num_children() >= 2) {
    return copy_column_view_impl(view.child(1), stream, mr);
  }
  return copy_column_view_impl(view, stream, mr);
}

std::unique_ptr<cudf::column> copy_column_view_as_uint8(cudf::column_view const& view,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  cudf::column_view data_view = view;
  if (view.num_children() >= 2) { data_view = view.child(1); }
  cudf::size_type const n  = data_view.size();
  size_t const elem_size   = static_cast<size_t>(cudf::size_of(data_view.type()));
  size_t const total_bytes = static_cast<size_t>(n) * elem_size;
  auto col                 = cudf::make_fixed_width_column(cudf::data_type(cudf::type_id::UINT8),
                                           static_cast<cudf::size_type>(total_bytes),
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
  if (total_bytes > 0) {
    cudaMemcpyAsync(col->mutable_view().head<void>(),
                    data_view.head<void>(),
                    total_bytes,
                    cudaMemcpyDeviceToDevice,
                    stream.value());
    cudaStreamSynchronize(stream.value());
  }
  return col;
}

}  // namespace simpatico
