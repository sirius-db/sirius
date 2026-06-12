// Shared test utilities for codegen tests.
#pragma once
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Detect the compute capability of the current CUDA device.
// Initialises the CUDA driver and picks device 0 if no context exists.
// Falls back to sm_89 (Ada) if detection fails.
inline int detect_arch_cc() noexcept
{
  if (cuInit(0) != CUDA_SUCCESS) return 89;
  CUdevice dev = 0;
  // If a context is active, use its device; otherwise use device 0.
  if (cuCtxGetDevice(&dev) != CUDA_SUCCESS) {
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 89;
  }
  int major = 0, minor = 0;
  if (cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) !=
        CUDA_SUCCESS ||
      cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) !=
        CUDA_SUCCESS) {
    return 89;
  }
  return major * 10 + minor;
}

// ---------------------------------------------------------------------------
// Table factories
// ---------------------------------------------------------------------------

inline std::unique_ptr<cudf::table> make_int32_table(int num_cols, int num_rows, int seed)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(static_cast<std::size_t>(num_cols));
  for (int c = 0; c < num_cols; ++c) {
    std::vector<std::int32_t> host(static_cast<std::size_t>(num_rows));
    for (int r = 0; r < num_rows; ++r)
      host[static_cast<std::size_t>(r)] =
        static_cast<std::int32_t>((r * 17 + c * 1013 + seed) % 1000);
    auto col = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, num_rows, cudf::mask_state::UNALLOCATED);
    if (cudaMemcpy(col->mutable_view().head<std::int32_t>(),
                   host.data(),
                   host.size() * sizeof(std::int32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("make_int32_table: cudaMemcpy failed");
    cols.push_back(std::move(col));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

inline std::unique_ptr<cudf::table> make_f32_table(int num_cols, int num_rows, int seed)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(static_cast<std::size_t>(num_cols));
  for (int c = 0; c < num_cols; ++c) {
    std::vector<float> host(static_cast<std::size_t>(num_rows));
    for (int r = 0; r < num_rows; ++r)
      host[static_cast<std::size_t>(r)] =
        static_cast<float>((r * 17 + c * 1013 + seed) % 1000) * 0.5f -
        static_cast<float>((r + seed) % 7);
    auto col = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::FLOAT32}, num_rows, cudf::mask_state::UNALLOCATED);
    if (cudaMemcpy(col->mutable_view().head<float>(),
                   host.data(),
                   host.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("make_f32_table: cudaMemcpy failed");
    cols.push_back(std::move(col));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

inline std::unique_ptr<cudf::table> make_f64_table(int num_cols, int num_rows, int seed)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(static_cast<std::size_t>(num_cols));
  for (int c = 0; c < num_cols; ++c) {
    std::vector<double> host(static_cast<std::size_t>(num_rows));
    for (int r = 0; r < num_rows; ++r)
      host[static_cast<std::size_t>(r)] =
        static_cast<double>((r * 17 + c * 1013 + seed) % 1000) * 0.25 -
        static_cast<double>((r + seed) % 13);
    auto col = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::FLOAT64}, num_rows, cudf::mask_state::UNALLOCATED);
    if (cudaMemcpy(col->mutable_view().head<double>(),
                   host.data(),
                   host.size() * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("make_f64_table: cudaMemcpy failed");
    cols.push_back(std::move(col));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

// Byte-exact comparison of the full fixed-width payload (correct for all
// element widths, verifies lossless roundtrips bit-for-bit).
inline bool columns_equal(cudf::column_view a, cudf::column_view b)
{
  if (a.type() != b.type() || a.size() != b.size()) return false;
  std::size_t nbytes =
    static_cast<std::size_t>(a.size()) * static_cast<std::size_t>(cudf::size_of(a.type()));
  std::vector<std::uint8_t> ha(nbytes), hb(nbytes);
  cudaMemcpy(ha.data(), a.head<std::uint8_t>(), nbytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), b.head<std::uint8_t>(), nbytes, cudaMemcpyDeviceToHost);
  return ha == hb;
}

inline void expect(bool cond, char const* msg)
{
  if (!cond) throw std::runtime_error(msg);
}
