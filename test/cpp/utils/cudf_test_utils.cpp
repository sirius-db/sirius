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

#include "catch.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/cuda_stream.hpp>

namespace sirius {
namespace test {

static inline bool host_mem_equal(const uint8_t* a, const uint8_t* b, size_t n)
{
  if (a == b) return true;
  if ((a == nullptr) || (b == nullptr)) return false;
  return std::memcmp(a, b, n) == 0;
}

// Returns true if two cudf tables have identical packed metadata and data bytes
bool cudf_tables_have_equal_contents(const cudf::table& left, const cudf::table& right)
{
  if (left.num_rows() != right.num_rows()) return false;
  if (left.num_columns() != right.num_columns()) return false;

  rmm::cuda_stream local_stream;
  auto stream_view  = local_stream.view();
  auto left_packed  = cudf::pack(left, stream_view);
  auto right_packed = cudf::pack(right, stream_view);
  local_stream.synchronize();

  // Compare metadata size and bytes
  if (left_packed.metadata->size() != right_packed.metadata->size()) return false;
  if (!host_mem_equal(left_packed.metadata->data(),
                      right_packed.metadata->data(),
                      left_packed.metadata->size())) {
    return false;
  }

  // Compare GPU data sizes
  auto left_bytes  = left_packed.gpu_data->size();
  auto right_bytes = right_packed.gpu_data->size();
  if (left_bytes != right_bytes) return false;

  if (left_bytes == 0) return true;

  // Copy device buffers to host and compare
  std::vector<uint8_t> left_host(left_bytes);
  std::vector<uint8_t> right_host(right_bytes);
  cudaMemcpy(left_host.data(), left_packed.gpu_data->data(), left_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(right_host.data(), right_packed.gpu_data->data(), right_bytes, cudaMemcpyDeviceToHost);

  return host_mem_equal(left_host.data(), right_host.data(), left_bytes);
}

void expect_cudf_tables_equal(const cudf::table& left, const cudf::table& right)
{
  REQUIRE(cudf_tables_have_equal_contents(left, right));
}

}  // namespace test
}  // namespace sirius
