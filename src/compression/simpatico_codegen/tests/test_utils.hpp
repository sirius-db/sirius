// Shared test utilities for codegen tests.
#pragma once
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
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

// Per-row validity of a column as host bools (true = valid). A column without
// a null mask (or with null_count 0) is all-valid, so a (mask, 0-null) column
// and a maskless column compare equal. Honors the view's row offset.
inline std::vector<bool> host_validity_bits(cudf::column_view v)
{
  std::vector<bool> valid(static_cast<std::size_t>(v.size()), true);
  if (v.null_mask() == nullptr || v.null_count() == 0) return valid;
  std::size_t const first_word = static_cast<std::size_t>(v.offset()) / 32;
  std::size_t const last_word  = static_cast<std::size_t>(v.offset() + v.size() + 31) / 32;
  std::vector<std::uint32_t> words(last_word - first_word);
  cudaMemcpy(words.data(),
             reinterpret_cast<std::uint32_t const*>(v.null_mask()) + first_word,
             words.size() * sizeof(std::uint32_t),
             cudaMemcpyDeviceToHost);
  for (cudf::size_type r = 0; r < v.size(); ++r) {
    std::size_t const bit = static_cast<std::size_t>(v.offset() + r);
    valid[static_cast<std::size_t>(r)] = (words[bit / 32 - first_word] >> (bit % 32)) & 1u;
  }
  return valid;
}

// Validity equality: same null_count and the same per-row valid/null pattern.
inline bool validity_equal(cudf::column_view a, cudf::column_view b)
{
  if (a.null_count() != b.null_count()) return false;
  return host_validity_bits(a) == host_validity_bits(b);
}

// Byte-exact comparison of the full fixed-width payload (correct for all
// element widths, verifies lossless roundtrips bit-for-bit), plus validity:
// a roundtrip that loses or moves nulls fails even when the data bytes match.
inline bool columns_equal(cudf::column_view a, cudf::column_view b)
{
  if (a.type() != b.type() || a.size() != b.size()) return false;
  if (!validity_equal(a, b)) return false;
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

// Semantic STRING comparison: same validity pattern, and byte-exact content
// for every VALID row. Row-wise via offsets so sliced views (whose offsets
// child spans the parent) and differing null-row padding compare correctly;
// handles INT32 and INT64 offsets.
inline bool strings_equal(cudf::column_view a, cudf::column_view b, rmm::cuda_stream_view stream)
{
  if (a.size() != b.size()) return false;
  if (!validity_equal(a, b)) return false;
  auto const n = a.size();
  if (n == 0) return true;

  // Per-row byte ranges [o[r], o[r+1]) of the view's rows within the parent
  // chars buffer (a view's offsets child spans the parent, so index by
  // view offset).
  auto read_offsets = [](cudf::column_view col) {
    cudf::strings_column_view s(col);
    std::vector<std::int64_t> o(static_cast<std::size_t>(col.size()) + 1);
    auto off = s.offsets();
    if (off.type().id() == cudf::type_id::INT64) {
      cudaMemcpy(o.data(),
                 off.head<std::int64_t>() + col.offset(),
                 o.size() * sizeof(std::int64_t),
                 cudaMemcpyDeviceToHost);
    } else {
      std::vector<std::int32_t> o32(o.size());
      cudaMemcpy(o32.data(),
                 off.head<std::int32_t>() + col.offset(),
                 o32.size() * sizeof(std::int32_t),
                 cudaMemcpyDeviceToHost);
      std::copy(o32.begin(), o32.end(), o.begin());
    }
    return o;
  };
  auto const oa = read_offsets(a);
  auto const ob = read_offsets(b);

  auto read_span = [&stream](cudf::column_view col, std::vector<std::int64_t> const& o) {
    cudf::strings_column_view s(col);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(o.back() - o.front()));
    if (!bytes.empty()) {
      cudaMemcpy(bytes.data(),
                 s.chars_begin(stream) + o.front(),
                 bytes.size(),
                 cudaMemcpyDeviceToHost);
    }
    return bytes;
  };
  auto const pa = read_span(a, oa);
  auto const pb = read_span(b, ob);
  auto const valid = host_validity_bits(a);  // equals host_validity_bits(b) per validity_equal

  for (cudf::size_type r = 0; r < n; ++r) {
    if (!valid[static_cast<std::size_t>(r)]) continue;  // null rows may pad differently
    auto const idx = static_cast<std::size_t>(r);
    auto const la  = oa[idx + 1] - oa[idx];
    if (la != ob[idx + 1] - ob[idx]) return false;
    if (la > 0 && std::memcmp(pa.data() + (oa[idx] - oa.front()),
                              pb.data() + (ob[idx] - ob.front()),
                              static_cast<std::size_t>(la)) != 0)
      return false;
  }
  return true;
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

// STRING column from explicit values, with optional per-row validity (false = null).
// A null row contributes an empty slice to the chars buffer. The null mask is padded
// to cuDF's bitmask allocation size (an undersized mask makes kernels read OOB).
inline std::unique_ptr<cudf::column> make_strings_column(std::vector<std::string> const& values,
                                                         std::vector<bool> const& valid,
                                                         rmm::cuda_stream_view stream)
{
  int const n          = static_cast<int>(values.size());
  bool const has_nulls = !valid.empty();
  std::vector<char> chars;
  std::vector<std::int32_t> offsets(static_cast<std::size_t>(n) + 1, 0);
  for (int r = 0; r < n; ++r) {
    if (!has_nulls || valid[static_cast<std::size_t>(r)])
      for (char c : values[static_cast<std::size_t>(r)])
        chars.push_back(c);
    offsets[static_cast<std::size_t>(r) + 1] = static_cast<std::int32_t>(chars.size());
  }
  auto offsets_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, n + 1, cudf::mask_state::UNALLOCATED, stream);
  cudaMemcpyAsync(offsets_col->mutable_view().head<std::int32_t>(),
                  offsets.data(),
                  offsets.size() * sizeof(std::int32_t),
                  cudaMemcpyHostToDevice,
                  stream.value());
  rmm::device_buffer chars_buf(chars.size(), stream);
  if (!chars.empty())
    cudaMemcpyAsync(
      chars_buf.data(), chars.data(), chars.size(), cudaMemcpyHostToDevice, stream.value());

  if (!has_nulls) {
    stream.synchronize();
    return cudf::make_strings_column(
      n, std::move(offsets_col), std::move(chars_buf), 0, rmm::device_buffer{});
  }
  std::size_t const mask_bytes = cudf::bitmask_allocation_size_bytes(n);
  std::vector<std::uint32_t> words(static_cast<std::size_t>((n + 31) / 32), 0u);
  int nulls = 0;
  for (int r = 0; r < n; ++r) {
    if (valid[static_cast<std::size_t>(r)])
      words[static_cast<std::size_t>(r / 32)] |= (1u << (r % 32));
    else
      ++nulls;
  }
  rmm::device_buffer mask_buf(mask_bytes, stream);
  cudaMemsetAsync(mask_buf.data(), 0, mask_bytes, stream.value());
  cudaMemcpyAsync(mask_buf.data(),
                  words.data(),
                  words.size() * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice,
                  stream.value());
  stream.synchronize();
  return cudf::make_strings_column(
    n, std::move(offsets_col), std::move(chars_buf), nulls, std::move(mask_buf));
}

// Single-column STRING table from explicit values (+ optional validity).
inline std::unique_ptr<cudf::table> make_strings_table(std::vector<std::string> const& values,
                                                       std::vector<bool> const& valid,
                                                       rmm::cuda_stream_view stream)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(make_strings_column(values, valid, stream));
  return std::make_unique<cudf::table>(std::move(cols));
}
