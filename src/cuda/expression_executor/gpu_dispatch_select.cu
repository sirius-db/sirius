#include "expression_executor/gpu_dispatcher.hpp"
#include "gpu_buffer_manager.hpp"
#include <cub/cub.cuh>
#include <tuple>

namespace duckdb
{
namespace sirius
{

//----------Select----------//
std::tuple<uint64_t*, uint64_t> GpuDispatcher::DispatchSelect(const cudf::column_view& bitmap,
                                                              rmm::device_async_resource_ref mr,
                                                              rmm::cuda_stream_view stream)
{
  // The row ids are owned by the query executor and so must be managed by the buffer manager
  auto* gpu_buffer_manager = &GPUBufferManager::GetInstance();
  uint64_t* row_ids = gpu_buffer_manager->customCudaMalloc<uint64_t>(bitmap.size(), 0, false);
  rmm::device_scalar<uint64_t> d_num_selected(0, stream, mr);

  size_t temp_storage_bytes = 0;
  uint64_t num_selected     = 0;
  cub::DeviceSelect::Flagged(nullptr,
                             temp_storage_bytes,
                             thrust::make_counting_iterator<uint64_t>(0),
                             bitmap.data<bool>(),
                             row_ids,
                             d_num_selected.data(),
                             bitmap.size(),
                             stream);
  rmm::device_buffer temp_storage(temp_storage_bytes, stream, mr);
  cub::DeviceSelect::Flagged(temp_storage.data(),
                             temp_storage_bytes,
                             thrust::make_counting_iterator<uint64_t>(0),
                             bitmap.data<bool>(),
                             row_ids,
                             d_num_selected.data(),
                             bitmap.size(),
                             stream);
  num_selected = d_num_selected.value(stream);
  return std::make_tuple(row_ids, num_selected);
}

} // namespace sirius
} // namespace duckdb