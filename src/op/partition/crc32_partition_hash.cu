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

#include "op/partition/crc32_partition_hash.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace sirius {
namespace op {

namespace {

constexpr int kBlock = 256;

// How a column's per-row bytes are folded into the running CRC.
enum fold_kind : int {
  kFixed = 0,  // raw little-endian value bytes (`width`)
  kString,     // raw char bytes bounded by `offsets`
  kSplit,      // DecimalV2/V3(27,9): int128 -> int64 integer part + int32 fractional part
};

// Per-column device descriptor. Fixed/split: `data` is the offset-adjusted base (row i at
// `data + i*width`). String: `data` is the char base, bytes bounded by `offsets[i]..offsets[i+1]`.
// The null mask is indexed by absolute row (`mask_offset + i`).
struct col_desc {
  uint8_t const* data            = nullptr;
  cudf::size_type const* offsets = nullptr;  // strings only, else nullptr
  cudf::bitmask_type const* mask = nullptr;
  cudf::size_type mask_offset    = 0;
  int width                      = 0;  // byte count for kFixed/kSplit
  int kind                       = kFixed;
};

// DecimalV2 scaling: value is stored as an int128 equal to (number * 10^9).
constexpr int64_t kOneBillion = 1000000000;

// Build the reflected CRC-32 table (poly 0xEDB88320) in shared memory — stateless, so there is no
// global table to initialize or keep in sync.
__device__ __forceinline__ void build_crc_table(uint32_t* tab)
{
  // Threads split the 256 entries; each entry is one byte run through 8 polynomial rounds.
  for (int j = threadIdx.x; j < 256; j += blockDim.x) {
    uint32_t c = static_cast<uint32_t>(j);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
    }
    tab[j] = c;
  }
}

// Table-driven zlib CRC-32, i.e. `crc32(seed, p, len)`. StarRocks folds each field with one such
// call, chaining the result as the next seed.
__device__ __forceinline__ uint32_t crc32_bytes(uint32_t seed,
                                                uint8_t const* p,
                                                int len,
                                                uint32_t const* tab)
{
  uint32_t crc = seed ^ 0xFFFFFFFFu;
  for (int i = 0; i < len; ++i) {
    crc = (crc >> 8) ^ tab[(crc ^ p[i]) & 0xFFu];
  }
  return crc ^ 0xFFFFFFFFu;
}

// StarRocks feeds 4 zero bytes for a null row, regardless of column type.
__device__ __forceinline__ uint32_t crc32_null(uint32_t seed, uint32_t const* tab)
{
  uint32_t const null_bytes = 0;
  return crc32_bytes(seed, reinterpret_cast<uint8_t const*>(&null_bytes), 4, tab);
}

// DecimalV2 / DecimalV3(27,9) fold: int64 integer part then int32 fractional part of the int128
// value, matching StarRocks' `DecimalV2Value::int_value()` / `frac_value()` (C++ truncated
// division).
__device__ __forceinline__ uint32_t crc32_decimal_split(uint32_t seed,
                                                        uint8_t const* p,
                                                        uint32_t const* tab)
{
  __int128 v             = *reinterpret_cast<__int128 const*>(p);
  int64_t const int_val  = static_cast<int64_t>(v / static_cast<__int128>(kOneBillion));
  int32_t const frac_val = static_cast<int32_t>(v % static_cast<__int128>(kOneBillion));
  uint32_t h             = crc32_bytes(seed, reinterpret_cast<uint8_t const*>(&int_val), 8, tab);
  return crc32_bytes(h, reinterpret_cast<uint8_t const*>(&frac_val), 4, tab);
}

// Fold all key columns into one per-row CRC-32 in a single pass: seed 0 in a register, columns
// folded left→right, one hash write per row (no intermediate hash-buffer round-trips).
__global__ void fold_all(col_desc const* __restrict__ cols,
                         int ncols,
                         uint32_t* __restrict__ hashes,
                         cudf::size_type n)
{
  __shared__ uint32_t tab[256];
  build_crc_table(tab);
  __syncthreads();

  auto idx          = static_cast<cudf::size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto const stride = static_cast<cudf::size_type>(gridDim.x) * blockDim.x;
  for (; idx < n; idx += stride) {        // grid-stride: each thread hashes one or more rows
    uint32_t h = 0;                       // seed
    for (int cc = 0; cc < ncols; ++cc) {  // fold each key column into this row's hash
      col_desc const d = cols[cc];
      if (!cudf::bit_value_or(d.mask, d.mask_offset + idx, true)) {
        h = crc32_null(h, tab);  // null row
      } else if (d.kind == kString) {
        auto const begin = d.offsets[idx];
        auto const len   = d.offsets[idx + 1] - begin;
        // An empty string contributes nothing (StarRocks hashes 0 bytes).
        if (len > 0) { h = crc32_bytes(h, d.data + begin, static_cast<int>(len), tab); }
      } else if (d.kind == kSplit) {
        h = crc32_decimal_split(h, d.data + static_cast<std::size_t>(idx) * d.width, tab);
      } else {
        h = crc32_bytes(h, d.data + static_cast<std::size_t>(idx) * d.width, d.width, tab);
      }
    }
    hashes[idx] = h;  // one global write per row
  }
}

cudf::size_type grid_for(cudf::size_type n)
{
  // Widen before the ceiling division: `n` can be near INT32_MAX, which cuDF permits.
  int64_t const blocks = (static_cast<int64_t>(n) + kBlock - 1) / kBlock;

  // `fold_all` is a grid-stride loop, so any positive grid is correct. Size it to the device's
  // resident-block capacity (SMs × max active blocks/SM) rather than one block per kBlock rows:
  // for large inputs the latter launches far more blocks than the GPU can run at once, and there
  // is no legacy grid-X ceiling to work around on the CUDA 12+ hardware we target. Fall back to the
  // full ceiling if the device queries fail.
  int device = 0, sm_count = 0, blocks_per_sm = 0;
  if (cudaGetDevice(&device) == cudaSuccess &&
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) == cudaSuccess &&
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, fold_all, kBlock, 0) ==
        cudaSuccess &&
      sm_count > 0 && blocks_per_sm > 0) {
    int64_t const resident = static_cast<int64_t>(sm_count) * blocks_per_sm;
    return static_cast<cudf::size_type>(std::min(blocks, resident));
  }
  return static_cast<cudf::size_type>(blocks);
}

// Fixed-width byte count for a supported non-string type id.
int fixed_width_bytes(cudf::type_id id)
{
  switch (id) {
    case cudf::type_id::BOOL8:
    case cudf::type_id::INT8:
    case cudf::type_id::UINT8: return 1;
    case cudf::type_id::INT16:
    case cudf::type_id::UINT16: return 2;
    case cudf::type_id::INT32:
    case cudf::type_id::UINT32:
    case cudf::type_id::FLOAT32:
    case cudf::type_id::DECIMAL32: return 4;
    case cudf::type_id::INT64:
    case cudf::type_id::UINT64:
    case cudf::type_id::FLOAT64:
    case cudf::type_id::DECIMAL64: return 8;
    case cudf::type_id::DECIMAL128: return 16;
    default: return 0;
  }
}

bool is_decimal(cudf::type_id id)
{
  return id == cudf::type_id::DECIMAL32 || id == cudf::type_id::DECIMAL64 ||
         id == cudf::type_id::DECIMAL128;
}

// Offset-adjusted byte base for a fixed-width/decimal column: element i lives at `base + i*width`.
uint8_t const* fixed_base(cudf::column_view const& col, int width)
{
  return col.head<uint8_t>() + static_cast<std::size_t>(col.offset()) * width;
}

// Resolve one key column into its device descriptor, throwing if it is a type/layout the kernel
// cannot hash bit-exactly. `spec` is the column's `decimal_key` (or nullptr if none was supplied).
// Called for every column before any kernel launch, so a rejection never unwinds mid-flight.
col_desc resolve_column(cudf::column_view const& col,
                        crc32_partition_hash::decimal_key const* spec,
                        rmm::cuda_stream_view stream)
{
  auto const id = col.type().id();
  col_desc d;
  d.mask        = col.null_mask();
  d.mask_offset = col.offset();

  if (id == cudf::type_id::STRING) {
    if (spec != nullptr) {
      throw std::invalid_argument(
        "crc32_partition_hash: decimal_key given for a non-decimal column");
    }
    d.kind = kString;
    // A 0-row strings column may be the childless make_empty_column representation, whose
    // `offsets()` throws; there is nothing to hash, so leave the pointers null.
    if (col.size() > 0) {
      cudf::strings_column_view const sv(col);
      if (sv.offsets().type().id() != cudf::type_id::INT32) {
        throw std::invalid_argument(
          "crc32_partition_hash: INT64 (large) string offsets are not supported");
      }
      // `offsets()` is the unshifted offsets child; a sliced strings column would need its offset
      // folded into the indexing. Partition inputs are freshly materialized, so require offset 0.
      if (col.offset() != 0) {
        throw std::invalid_argument(
          "crc32_partition_hash: sliced string columns (offset != 0) are not supported");
      }
      d.data    = reinterpret_cast<uint8_t const*>(sv.chars_begin(stream));
      d.offsets = sv.offsets().data<cudf::size_type>();
    }
    return d;
  }

  if (is_decimal(id)) {
    if (spec == nullptr) {
      throw std::invalid_argument(
        "crc32_partition_hash: DECIMAL key columns require a decimal_key (cudf lacks precision and "
        "the DecimalV2/V3 distinction)");
    }
    auto const scale = col.type().scale();
    bool split       = false;
    if (spec->is_decimal_v2) {
      // StarRocks stores DecimalV2 as an int128 scaled by 10^9; it must arrive as DECIMAL128(-9).
      if (id != cudf::type_id::DECIMAL128 || scale != -9) {
        throw std::invalid_argument(
          "crc32_partition_hash: DecimalV2 keys must be represented as DECIMAL128 with scale -9");
      }
      split = true;
    } else if (id == cudf::type_id::DECIMAL128 && spec->precision == 27 && scale == -9) {
      // DecimalV3(27,9) reuses the DecimalV2 split fold for on-disk compatibility.
      split = true;
    }
    d.width = fixed_width_bytes(id);
    d.kind  = split ? kSplit : kFixed;
    d.data  = fixed_base(col, d.width);
    return d;
  }

  // Plain fixed-width: integers, bool, float/double.
  auto const width = fixed_width_bytes(id);
  if (width == 0) {
    throw std::invalid_argument("crc32_partition_hash: unsupported key column type id " +
                                std::to_string(static_cast<int>(id)));
  }
  if (spec != nullptr) {
    throw std::invalid_argument("crc32_partition_hash: decimal_key given for a non-decimal column");
  }
  d.kind  = kFixed;
  d.width = width;
  d.data  = fixed_base(col, width);
  return d;
}

}  // namespace

rmm::device_uvector<uint32_t> crc32_partition_hash::compute(
  cudf::table_view const& keys,
  std::vector<decimal_key> const& decimal_keys,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const ncols = keys.num_columns();
  if (ncols == 0) {
    throw std::invalid_argument("crc32_partition_hash::compute: `keys` has no columns");
  }

  // Index the decimal specs by column, rejecting out-of-range and duplicate entries.
  std::unordered_map<cudf::size_type, decimal_key const*> spec_by_col;
  spec_by_col.reserve(decimal_keys.size());
  for (auto const& k : decimal_keys) {
    if (k.column < 0 || k.column >= ncols) {
      throw std::invalid_argument("crc32_partition_hash: decimal_key column index out of range");
    }
    if (!spec_by_col.emplace(k.column, &k).second) {
      throw std::invalid_argument("crc32_partition_hash: duplicate decimal_key for a column");
    }
  }

  // Resolve (and validate) every column before allocating output or launching, so a rejection never
  // unwinds while an earlier column's kernel is still reading caller-owned inputs on the stream.
  std::vector<col_desc> h_descs(ncols);
  for (cudf::size_type c = 0; c < ncols; ++c) {
    auto const it = spec_by_col.find(c);
    h_descs[c] =
      resolve_column(keys.column(c), it == spec_by_col.end() ? nullptr : it->second, stream);
  }

  auto const n = keys.num_rows();
  rmm::device_uvector<uint32_t> hashes(static_cast<std::size_t>(n), stream, mr);
  if (n == 0) { return hashes; }

  // Upload descriptors and fold all columns in one pass. The kernel writes every row, so no
  // zero-init is needed; `d_descs` is freed in stream order, after the kernel reads it.
  rmm::device_uvector<col_desc> d_descs(static_cast<std::size_t>(ncols), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_descs.data(),
                                h_descs.data(),
                                static_cast<std::size_t>(ncols) * sizeof(col_desc),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  fold_all<<<grid_for(n), kBlock, 0, stream.value()>>>(d_descs.data(), ncols, hashes.data(), n);
  CUDF_CHECK_CUDA(stream.value());
  return hashes;
}

}  // namespace op
}  // namespace sirius
