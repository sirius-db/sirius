// SPDX-License-Identifier: Apache-2.0
// Shared helpers for the ALP and ALP-RD operators (both compile with nvcc).
#ifndef SIMPATICO_OPERATORS_ALP_COMMON_CUH
#define SIMPATICO_OPERATORS_ALP_COMMON_CUH

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>

#include <cstdint>
#include <memory>

namespace simpatico {

// The (positions, values) exception layout shared by the ALP and ALP-RD
// operators: `positions` holds the global row indices of the exceptions and
// `values` their payloads (raw floats for plain ALP, left parts for ALP-RD).
struct alp_exception_columns {
  std::unique_ptr<cudf::column> positions;  // INT32
  std::unique_ptr<cudf::column> values;     // value_type_id
  cudf::size_type count = 0;
};

// Reduce the per-row exception flags (`d_flags[i] != 0` == exception) to a
// count, then stream-compact the exception row indices into `positions` and the
// corresponding `values_source[i]` payloads into `values`. All work is enqueued
// on `stream`; the thrust::reduce return makes `count` valid on the host on
// return, but the caller must sync before reading the columns from another
// stream. `ValueT` is both the source element type and the `values` column type
// (which must equal `value_type_id`).
template <typename ValueT>
alp_exception_columns compact_exceptions(const uint8_t* d_flags,
                                         cudf::size_type n,
                                         const ValueT* values_source,
                                         cudf::type_id value_type_id,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  auto exec         = rmm::exec_policy_nosync(stream, mr);
  int64_t exc_count = thrust::reduce(exec,
                                     thrust::device_pointer_cast(d_flags),
                                     thrust::device_pointer_cast(d_flags + n),
                                     int64_t{0},
                                     thrust::plus<int64_t>{});

  alp_exception_columns out;
  out.count  = static_cast<cudf::size_type>(exc_count);
  out.values = cudf::make_fixed_width_column(
    cudf::data_type(value_type_id), out.count, cudf::mask_state::UNALLOCATED, stream, mr);
  out.positions = cudf::make_fixed_width_column(
    cudf::data_type(cudf::type_id::INT32), out.count, cudf::mask_state::UNALLOCATED, stream, mr);

  if (out.count > 0) {
    auto counting = thrust::make_counting_iterator(int32_t{0});
    thrust::copy_if(exec,
                    counting,
                    counting + n,
                    thrust::device_pointer_cast(d_flags),
                    thrust::device_pointer_cast(out.positions->mutable_view().data<int32_t>()),
                    [] __device__(uint8_t f) { return f != 0; });
    thrust::copy_if(exec,
                    thrust::device_pointer_cast(values_source),
                    thrust::device_pointer_cast(values_source + n),
                    thrust::device_pointer_cast(d_flags),
                    thrust::device_pointer_cast(out.values->mutable_view().data<ValueT>()),
                    [] __device__(uint8_t f) { return f != 0; });
  }
  return out;
}

}  // namespace simpatico

#endif  // SIMPATICO_OPERATORS_ALP_COMMON_CUH
