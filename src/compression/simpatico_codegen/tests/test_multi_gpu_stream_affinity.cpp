// SPDX-License-Identifier: Apache-2.0
//
// Serial compression keeps the cache untouched; threaded decodes on devices
// 0 -> 1 -> 0 verify that cached streams remain device-local. Self-skips when
// fewer than two GPUs are visible.

#include "api/simpatico_codegen.hpp"
#include "test_utils.hpp"

#include <cudf/table/table.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

constexpr int kColumnThreads = 4;
constexpr int kNumColumns    = 4;
constexpr int kNumRows       = 65536;

char const* const kPlanDsl =
  "input -> delta -> differences\n"
  "delta.differences -> bitpack\n"
  "---\n"
  "input -> delta -> differences\n"
  "delta.differences -> bitpack\n"
  "---\n"
  "input -> delta -> differences\n"
  "delta.differences -> bitpack\n"
  "---\n"
  "input -> delta -> differences\n"
  "delta.differences -> bitpack\n";

void decode_on_current_device(int device)
{
  auto const label = "device " + std::to_string(device);

  auto const mr = rmm::mr::get_current_device_resource_ref();

  rmm::cuda_stream serial_stream{rmm::cuda_stream::flags::non_blocking};
  auto input = make_int32_table(kNumColumns, kNumRows, 13 + device);

  // Keep compression off the internal stream cache.
  auto compressed =
    simpatico::compress_with_plan(input->view(), kPlanDsl, serial_stream.view(), mr);
  serial_stream.synchronize();

  expect(compressed.num_columns() == static_cast<std::size_t>(kNumColumns),
         (label + ": compressed column count").c_str());
  expect(compressed.num_rows() == kNumRows, (label + ": compressed row count").c_str());

  auto decoded = simpatico::decompress(compressed, kColumnThreads, mr);
  expect(decoded != nullptr, (label + ": decode returned null").c_str());
  expect(decoded->num_columns() == kNumColumns, (label + ": decoded column count").c_str());
  expect(decoded->num_rows() == kNumRows, (label + ": decoded row count").c_str());

  for (int column = 0; column < kNumColumns; ++column) {
    expect(columns_equal(input->view().column(column), decoded->view().column(column)),
           (label + ": column " + std::to_string(column) + " byte mismatch").c_str());
  }

  // Surface asynchronous device faults.
  if (auto const status = cudaDeviceSynchronize(); status != cudaSuccess) {
    throw std::runtime_error(label +
                             ": cudaDeviceSynchronize failed: " + cudaGetErrorString(status));
  }
}

void decode_on_device(int device)
{
  rmm::cuda_set_device_raii const device_guard{rmm::cuda_device_id{device}};
  decode_on_current_device(device);
}

}  // namespace

int main()
{
  int device_count       = 0;
  auto const count_error = cudaGetDeviceCount(&device_count);
  if (count_error != cudaSuccess) {
    std::fprintf(stderr,
                 "test_multi_gpu_stream_affinity: cudaGetDeviceCount failed: %s\n",
                 cudaGetErrorString(count_error));
    return 1;
  }
  if (device_count < 2) {
    std::printf("test_multi_gpu_stream_affinity: SKIP (needs >= 2 GPUs, found %d)\n", device_count);
    return 0;
  }

  try {
    decode_on_device(0);
    decode_on_device(1);
    decode_on_device(0);
    std::printf("test_multi_gpu_stream_affinity: PASS\n");
    return 0;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "test_multi_gpu_stream_affinity: FAIL: %s\n", error.what());
    return 1;
  }
}
