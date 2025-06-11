#include "duckdb/common/exception.hpp"
#include "expression_executor/gpu_dispatcher.hpp"
#include "gpu_columns.hpp"
#include <rmm/device_uvector.hpp>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace duckdb
{
namespace sirius
{

// Helper (template) functors to reduce bloat
//----------Numerics----------//
template <typename T>
struct MaterializeNumeric
{
  static std::unique_ptr<cudf::column> Do(const T* input_data,
                                          const uint64_t* row_ids,
                                          uint64_t row_id_count,
                                          rmm::device_async_resource_ref mr,
                                          rmm::cuda_stream_view stream = rmm::cuda_stream_default)
  {
    // Define the thrust execution policy
    rmm::mr::thrust_allocator<uint8_t> thrust_allocator(stream, mr);
    auto exec = thrust::cuda::par(thrust_allocator).on(stream);

    // Allocate output buffer
    rmm::device_uvector<T> output_data(row_id_count, stream, mr);

    // Gather the input data
    thrust::gather(exec, row_ids, row_ids + row_id_count, input_data, output_data.begin());

    // Construct and return a cudf::column
    return std::make_unique<cudf::column>(std::move(output_data),
                                          rmm::device_buffer(0, stream, mr),
                                          0);
  }
};

//----------Strings----------//
struct MaterializeString
{
  // Functor for calculating the string lengths
  struct GatherLengths
  {
    // Fields
    const uint64_t* input_offsets;

    // Constructor
    explicit GatherLengths(const uint64_t* input_offset)
        : input_offsets(input_offset)
    {}

    // Operator
    __device__ __forceinline__ int64_t operator()(uint64_t row_id) const
    {
      return static_cast<int64_t>(input_offsets[row_id + 1] - input_offsets[row_id]);
    }
  };

  // Functor for copying string data
  struct CopyData
  {
    // Fields
    const uint64_t* row_ids;
    const uint64_t* input_offsets;
    const uint8_t* input_data;
    int64_t* output_offsets;
    uint8_t* output_data;

    // Constructor
    CopyData(const uint64_t* row_ids,
             const uint64_t* input_offsets,
             const uint8_t* input_data,
             int64_t* output_offsets,
             uint8_t* output_data)
        : row_ids(row_ids)
        , input_offsets(input_offsets)
        , input_data(input_data)
        , output_offsets(output_offsets)
        , output_data(output_data)
    {}

    // Operator
    __device__ __forceinline__ void operator()(cudf::size_type output_idx)
    {
      const auto row_id        = row_ids[output_idx];
      const auto output_offset = output_offsets[output_idx];
      const auto input_offset  = input_offsets[row_id];
      const auto next_offset   = input_offsets[row_id + 1];
      const auto length        = next_offset - input_offset;

      memcpy(output_data + output_offset, input_data + input_offset, length * sizeof(uint8_t));
    }
  };

  static std::unique_ptr<cudf::column> Do(const uint8_t* input_data,
                                          const uint64_t* input_offsets,
                                          const uint64_t* row_ids,
                                          uint64_t row_id_count,
                                          rmm::device_async_resource_ref mr,
                                          rmm::cuda_stream_view stream = rmm::cuda_stream_default)
  {
    static_assert(std::is_same_v<int32_t, cudf::size_type>); // Sanity check

    // Define the thrust execution policy
    rmm::mr::thrust_allocator<uint8_t> thrust_allocator(stream, mr);
    auto exec             = thrust::cuda::par(thrust_allocator).on(stream);
    uint64_t offset_count = row_id_count + 1;

    // Allocate temporary string length and output offsets buffer
    rmm::device_uvector<int64_t> temp_string_lengths(offset_count, stream, mr);
    rmm::device_uvector<int64_t> output_offsets(offset_count, stream, mr);
    // Set the last string length to 0, so that the exclusive scan places the total sum at the end
    CUDF_CUDA_TRY(cudaMemset(temp_string_lengths.data() + row_id_count, 0, sizeof(int64_t)));

    // Gather the string lengths
    thrust::transform(exec,
                      row_ids,
                      row_ids + row_id_count,
                      temp_string_lengths.begin(),
                      GatherLengths{input_offsets});

    // Calculate the output offsets
    thrust::exclusive_scan(exec,
                           temp_string_lengths.begin(),
                           temp_string_lengths.end(),
                           output_offsets.begin(),
                           static_cast<int64_t>(0));
    const auto output_bytes = output_offsets.back_element(stream);

    // Allocate output data buffer
    rmm::device_uvector<uint8_t> output_data(output_bytes, stream, mr);

    // Gather the string data
    thrust::for_each_n(
      exec,
      thrust::counting_iterator<cudf::size_type>(0),
      row_id_count,
      CopyData(row_ids, input_offsets, input_data, output_offsets.data(), output_data.data()));

    // Return a cudf::column
    auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT64},
                                                      static_cast<cudf::size_type>(offset_count),
                                                      output_offsets.release(),
                                                      rmm::device_buffer{0, stream, mr},
                                                      0);
    return cudf::make_strings_column(static_cast<cudf::size_type>(row_id_count),
                                     std::move(offsets_col),
                                     output_data.release(),
                                     0,
                                     rmm::device_buffer{0, stream, mr});
  }
};

// Materialize a GPUColumn directly into a cudf::column
std::unique_ptr<cudf::column> GpuDispatcher::DispatchMaterialize(const GPUColumn* input,
                                                                 rmm::device_async_resource_ref mr,
                                                                 rmm::cuda_stream_view stream)
{
  D_ASSERT(input->row_ids != nullptr);
  D_ASSERT(input->row_ids->size() > 0);

  const auto* input_data    = input->data_wrapper.data;
  const auto* input_offsets = input->data_wrapper.offset; // Maybe unused

  switch (input->data_wrapper.type.id())
  {
    case GPUColumnTypeId::INT32:
      return MaterializeNumeric<int32_t>::Do(reinterpret_cast<const int32_t*>(input_data),
                                             input->row_ids,
                                             input->row_id_count,
                                             mr,
                                             stream);
    case GPUColumnTypeId::INT64:
      return MaterializeNumeric<uint64_t>::Do(reinterpret_cast<const uint64_t*>(input_data),
                                              input->row_ids,
                                              input->row_id_count,
                                              mr,
                                              stream);
    case GPUColumnTypeId::FLOAT32:
      return MaterializeNumeric<float_t>::Do(reinterpret_cast<const float_t*>(input_data),
                                             input->row_ids,
                                             input->row_id_count,
                                             mr,
                                             stream);
    case GPUColumnTypeId::FLOAT64:
      return MaterializeNumeric<double_t>::Do(reinterpret_cast<const double_t*>(input_data),
                                              input->row_ids,
                                              input->row_id_count,
                                              mr,
                                              stream);
    case GPUColumnTypeId::BOOLEAN:
      return MaterializeNumeric<bool>::Do(reinterpret_cast<const bool*>(input_data),
                                          input->row_ids,
                                          input->row_id_count,
                                          mr,
                                          stream);
    case GPUColumnTypeId::DATE:
      return MaterializeNumeric<cudf::timestamp_D>::Do(
        reinterpret_cast<const cudf::timestamp_D*>(input_data),
        input->row_ids,
        input->row_id_count,
        mr,
        stream);
    case GPUColumnTypeId::VARCHAR:
      return MaterializeString::Do(input_data,
                                   input_offsets,
                                   input->row_ids,
                                   input->row_id_count,
                                   mr,
                                   stream);
    default:
      throw InternalException("Unsupported sirius column type in `Dispatch[Materialize]`: %d",
                              static_cast<int>(input->data_wrapper.type.id()));
  }
}

} // namespace sirius
} // namespace duckdb