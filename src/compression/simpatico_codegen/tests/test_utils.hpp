// Shared test utilities for codegen tests.
#pragma once
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

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

inline std::unique_ptr<cudf::table> make_int64_table(int num_cols, int num_rows, int seed)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(static_cast<std::size_t>(num_cols));
  for (int c = 0; c < num_cols; ++c) {
    std::vector<std::int64_t> host(static_cast<std::size_t>(num_rows));
    for (int r = 0; r < num_rows; ++r)
      host[static_cast<std::size_t>(r)] =
        static_cast<std::int64_t>((r * 17 + c * 1013 + seed) % 1000);
    auto col = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64}, num_rows, cudf::mask_state::UNALLOCATED);
    if (cudaMemcpy(col->mutable_view().head<std::int64_t>(),
                   host.data(),
                   host.size() * sizeof(std::int64_t),
                   cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("make_int64_table: cudaMemcpy failed");
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

// ---------------------------------------------------------------------------
// String helpers
// ---------------------------------------------------------------------------

// A STRING column with heavy repetition (dictionary/BWT/nvcomp friendly).
inline std::unique_ptr<cudf::column> make_string_column(int num_rows, rmm::cuda_stream_view stream)
{
  static char const* const words[] = {
    "apple", "banana", "cherry", "apple", "date", "banana", "apple", "elderberry", "fig", "banana"};
  constexpr int kNumWords = 10;

  std::vector<char> chars;
  std::vector<std::int32_t> offsets(static_cast<std::size_t>(num_rows) + 1, 0);
  for (int r = 0; r < num_rows; ++r) {
    char const* w = words[r % kNumWords];
    for (char const* p = w; *p; ++p)
      chars.push_back(*p);
    offsets[static_cast<std::size_t>(r) + 1] = static_cast<std::int32_t>(chars.size());
  }

  auto offsets_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows + 1, cudf::mask_state::UNALLOCATED, stream);
  if (cudaMemcpyAsync(offsets_col->mutable_view().head<std::int32_t>(),
                      offsets.data(),
                      offsets.size() * sizeof(std::int32_t),
                      cudaMemcpyHostToDevice,
                      stream.value()) != cudaSuccess)
    throw std::runtime_error("make_string_column: offsets copy failed");

  rmm::device_buffer chars_buf(chars.size(), stream);
  if (!chars.empty() &&
      cudaMemcpyAsync(
        chars_buf.data(), chars.data(), chars.size(), cudaMemcpyHostToDevice, stream.value()) !=
        cudaSuccess)
    throw std::runtime_error("make_string_column: chars copy failed");
  stream.synchronize();

  return cudf::make_strings_column(
    num_rows, std::move(offsets_col), std::move(chars_buf), 0, rmm::device_buffer{});
}

// Single-column STRING table wrapping make_string_column.
inline std::unique_ptr<cudf::table> make_string_table(int num_rows, rmm::cuda_stream_view stream)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(make_string_column(num_rows, stream));
  return std::make_unique<cudf::table>(std::move(cols));
}

// Byte-exact STRING comparison (offsets + chars).
inline bool strings_equal(cudf::column_view a, cudf::column_view b, rmm::cuda_stream_view stream)
{
  if (a.size() != b.size()) return false;
  cudf::strings_column_view sa(a), sb(b);
  auto const ca = sa.chars_size(stream);
  auto const cb = sb.chars_size(stream);
  if (ca != cb) return false;

  std::size_t const n_off = static_cast<std::size_t>(a.size()) + 1;
  std::vector<std::int32_t> oa(n_off), ob(n_off);
  cudaMemcpy(oa.data(),
             sa.offsets().head<std::int32_t>(),
             n_off * sizeof(std::int32_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(ob.data(),
             sb.offsets().head<std::int32_t>(),
             n_off * sizeof(std::int32_t),
             cudaMemcpyDeviceToHost);
  if (oa != ob) return false;

  std::vector<std::uint8_t> pa(static_cast<std::size_t>(ca)), pb(static_cast<std::size_t>(cb));
  if (ca > 0) {
    cudaMemcpy(
      pa.data(), sa.chars_begin(stream), static_cast<std::size_t>(ca), cudaMemcpyDeviceToHost);
    cudaMemcpy(
      pb.data(), sb.chars_begin(stream), static_cast<std::size_t>(cb), cudaMemcpyDeviceToHost);
  }
  return pa == pb;
}

// Equality for any supported column type (STRING or fixed-width).
inline bool columns_equal_any(cudf::column_view a,
                              cudf::column_view b,
                              rmm::cuda_stream_view stream)
{
  if (a.type() != b.type() || a.size() != b.size()) return false;
  if (a.type().id() == cudf::type_id::STRING) return strings_equal(a, b, stream);
  return columns_equal(a, b);
}
