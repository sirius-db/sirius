/**
 * Dictionary compressor: STRING column -> dictionary column (encode); decompress via decode.
 * Stores the encoded dictionary column to avoid copying keys chars from cuDF (invalid pointers).
 */

#include "codegen/plan/representation.hpp"
#include "codegen/util/cuda_check.hpp"
#include "codegen/util/nvtx.hpp"
#include "constant_width_offsets.hpp"
#include "decode/decode_session.hpp"
#include "util/host_observation.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction/approx_distinct_count.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_reduce.cuh>
#include <cuda/functional>
#include <cuda_runtime.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace simpatico {

namespace {

// Structural row bound only: a cudf column cannot exceed size_type rows, and
// the INT32 dictionary codes index the KEY SET (K distinct values), not rows,
// so any representable column encodes. Audit notes (iteration 6): the decode
// gathers do their addressing in int64 (key_base/nbytes below) and the
// constant-width fast path self-guards on nbytes > size_type max; the historic
// `1 << 28` "sanity bound" predated the HyperLogLog cardinality gate below,
// which now handles the real hazard (dictionary::encode's illegal access on
// huge HIGH-CARDINALITY inputs) by distinct FRACTION rather than row count —
// the row cap only silently forced narrow pins (>= 2^28 rows/chunk, e.g. q12's
// 5-col lineitem pin at ~276M and 2-3-col orders pins at ~600-900M) to raw.
// Encode transients (cudf's hash set + INT32 indices) scale O(n); an
// allocation failure throws and the pin falls back to an uncompressed chunk,
// visibly (see the per-pin coverage line in pin_table.cpp).
constexpr size_t MAX_INDICES = static_cast<size_t>(std::numeric_limits<cudf::size_type>::max());

// cudf::dictionary::encode faults with a context-corrupting illegal access on
// very large, very-high-cardinality strings — inputs a dictionary can't help
// anyway. Skip such columns before the full encode, gating on an estimate of
// the *full-column* distinct fraction: a prefix probe mis-reads long columns
// with moderate absolute cardinality (e.g. 1M distinct over 100M rows is ~0.23
// unique in a 256K-row prefix but 0.01 over the whole column — an ideal
// dictionary target). A fixed-memory HyperLogLog sketch gives the true
// fraction in one pass without materializing the keys column that faults.
constexpr size_t kDictCardCheckMinRows = 1 << 20;
constexpr double kDictMaxCardFraction  = 0.5;

struct matching_key_width {
  cudf::detail::input_offsetalator offsets;

  __device__ int64_t operator()(cudf::size_type i) const
  {
    auto const width = offsets[1] - offsets[0];
    return width > 0 && offsets[i + 1] - offsets[i] == width ? width : 0;
  }
};

// The positive uniform key width, or zero for variable-width or empty keys. One CUB reduction
// writes the result beside its scratch in a single allocation, read back through the calling
// thread's pinned staging slab; returns after @p stream completes.
int64_t measure_constant_key_width(cudf::strings_column_view const& keys,
                                   ::cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  if (keys.size() <= 0) return 0;
  matching_key_width const matching_width{
    cudf::detail::offsetalator_factory::make_input_iterator(keys.offsets(), keys.offset())};
  auto reduce = [&](void* scratch, std::size_t& scratch_bytes, int64_t* result) {
    throw_if_cuda_error(
      cub::DeviceReduce::TransformReduce(scratch,
                                         scratch_bytes,
                                         thrust::counting_iterator<cudf::size_type>(0),
                                         result,
                                         keys.size(),
                                         cuda::minimum<int64_t>{},
                                         matching_width,
                                         std::numeric_limits<int64_t>::max(),
                                         stream.get()),
      "dictionary key width reduction");
  };
  std::size_t scratch_bytes = 0;
  reduce(nullptr, scratch_bytes, nullptr);
  if (scratch_bytes > std::numeric_limits<std::size_t>::max() - sizeof(int64_t)) {
    throw std::overflow_error("dictionary key width scratch size overflow");
  }
  // CUB's queried size includes padding to align the scratch following the result.
  rmm::device_buffer storage(sizeof(int64_t) + scratch_bytes, stream, mr);
  auto* const result = static_cast<int64_t*>(storage.data());
  reduce(static_cast<char*>(storage.data()) + sizeof(int64_t), scratch_bytes, result);
  int64_t width = 0;
  read_device_bytes_completed(&width, result, sizeof width, stream);
  return width;
}

// Compile-time width lets the stitch loop fully unroll into registers — with a
// runtime width the byte shuffle spills to local memory.
template <int W>
void padded_gather_chunks(
  uint4 const* pool, int32_t const* ix, char* out, int64_t nbytes, decode_frame& frame)
{
  int64_t const nchunks = (nbytes + 15) / 16;
  thrust::for_each_n(rmm::exec_policy_nosync(frame.stream(), frame.mr()),
                     thrust::counting_iterator<int64_t>(0),
                     nchunks,
                     [=] __device__(int64_t t) {
                       int64_t const base = t * 16;
                       int const n = nbytes - base < 16 ? static_cast<int>(nbytes - base) : 16;
                       char b[16];
                       int64_t row       = base / W;
                       int64_t row_start = row * W;
                       for (int i = 0; i < n;) {
                         uint4 const v = pool[ix[row]];
                         char a[16];
                         memcpy(a, &v, 16);
                         int const in_row = static_cast<int>(base + i - row_start);
                         int const take   = W - in_row < n - i ? W - in_row : n - i;
#pragma unroll
                         for (int j = 0; j < W; ++j)
                           if (j < take) b[i + j] = a[in_row + j];
                         i += take;
                         row_start += W;
                         ++row;
                       }
                       if (n == 16) {
                         uint4 v;
                         memcpy(&v, b, 16);
                         *reinterpret_cast<uint4*>(out + base) = v;
                       } else {
                         for (int i = 0; i < n; ++i)
                           out[base + i] = b[i];
                       }
                     });
}

// Device view of an equals_any predicate's needles packed for one upload: `offsets` holds
// count + 1 byte offsets into `chars`, the needles' bytes back to back.
struct needle_view {
  std::int32_t const* offsets;
  char const* chars;
  std::int32_t count;
};

// The upload's device storage together with its view; the storage is released in stream order once
// the owner goes out of scope, so it must outlive the enqueue of every kernel that reads the view.
struct uploaded_needles {
  rmm::device_buffer storage;
  needle_view view;
};

// Per-key membership test: `lut[k]` is true iff key `k` equals some needle byte for byte, the same
// comparison as cuDF's STRING EQUAL. Keys are read through an offsetalator so INT32 and INT64 key
// offsets share one instantiation; the offsets are absolute into `key_chars`.
struct key_matches_any_needle {
  cudf::detail::input_offsetalator key_offsets;
  char const* key_chars;
  needle_view needles;

  __device__ bool operator()(cudf::size_type k) const
  {
    auto const key_begin = key_offsets[k];
    auto const key_size  = key_offsets[k + 1] - key_begin;
    for (std::int32_t j = 0; j < needles.count; ++j) {
      auto const needle_begin = needles.offsets[j];
      if (needles.offsets[j + 1] - needle_begin != key_size) continue;
      bool equal = true;
      for (int64_t b = 0; b < key_size && equal; ++b) {
        equal = key_chars[key_begin + b] == needles.chars[needle_begin + b];
      }
      if (equal) return true;
    }
    return false;
  }
};

// Stage the needles in frame-owned host storage, which lives until the session drains, and upload
// them once. A pageable host-to-device copy is staged by the runtime without waiting for the
// stream, so this is the whole cost of the predicate's host side.
uploaded_needles upload_needles(std::vector<std::string> const& needles, decode_frame& frame)
{
  auto const count = needles.size();
  if (count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::length_error("dictionary predicate: too many values");
  }
  std::size_t total_chars = 0;
  for (auto const& needle : needles) {
    if (needle.size() >
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) - total_chars) {
      throw std::length_error("dictionary predicate: values exceed the packed size limit");
    }
    total_chars += needle.size();
  }
  // Offsets first so they stay naturally aligned at the start of both blocks; the chars follow.
  auto const offsets_bytes = (count + 1) * sizeof(std::int32_t);
  auto const total_bytes   = offsets_bytes + total_chars;
  auto const host          = frame.host_array<char>(total_bytes);
  auto* const host_chars   = host.data() + offsets_bytes;
  std::vector<std::int32_t> offsets(count + 1);
  std::int32_t cursor = 0;
  for (std::size_t j = 0; j < count; ++j) {
    offsets[j] = cursor;
    if (!needles[j].empty()) std::memcpy(host_chars + cursor, needles[j].data(), needles[j].size());
    cursor += static_cast<std::int32_t>(needles[j].size());
  }
  offsets[count] = cursor;
  std::memcpy(host.data(), offsets.data(), offsets_bytes);
  rmm::device_buffer storage(total_bytes, frame.stream(), frame.mr());
  throw_if_cuda_error(
    cudaMemcpyAsync(
      storage.data(), host.data(), total_bytes, cudaMemcpyHostToDevice, frame.stream().get()),
    "dictionary predicate: upload values");
  needle_view const view{static_cast<std::int32_t const*>(storage.data()),
                         static_cast<char const*>(storage.data()) + offsets_bytes,
                         static_cast<std::int32_t>(count)};
  return {std::move(storage), view};
}

// Constant-width null-free decode: analytic offsets + flat byte gather (skips cudf's
// batched-memcpy gather, offsets scan, and null-mask pass). nullptr = ineligible.
std::unique_ptr<cudf::column> try_decode_constant_width(cudf::strings_column_view const& keys,
                                                        cudf::column_view const& indices,
                                                        std::int64_t cached_width,
                                                        decode_frame& frame)
{
  auto const stream = frame.stream();
  auto const mr     = frame.mr();
  if (indices.null_count() > 0 || keys.parent().null_count() > 0) return nullptr;
  if (indices.type().id() != cudf::type_id::INT32) return nullptr;
  int64_t const width =
    cached_width < 0 ? measure_constant_key_width(keys, stream, mr) : cached_width;
  if (width <= 0) return nullptr;
  auto const n_rows    = indices.size();
  int64_t const nbytes = static_cast<int64_t>(n_rows) * width;
  if (nbytes > std::numeric_limits<cudf::size_type>::max()) return nullptr;

  auto offsets = make_constant_width_offsets(n_rows, static_cast<std::int32_t>(width), stream, mr);
  rmm::device_buffer chars(nbytes, stream, mr);
  auto output =
    cudf::make_strings_column(n_rows,
                              std::move(offsets),
                              std::move(chars),
                              0,
                              cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr));

  auto* out      = output->mutable_view().head<char>();
  auto const* kc = keys.chars_begin(stream);
  auto const* ix = indices.data<int32_t>();
  // One aligned 16B store per thread, assembled in registers from the rows
  // overlapping the chunk. Key-slice loads depend on pool size: small pools
  // are L1-resident so direct byte loads are fastest; a large pool (L2) is
  // first padded to a 16B stride so each overlapped row is one aligned uint4
  // load instead of per-byte L2 round trips.
  int64_t const nchunks = (nbytes + 15) / 16;
  auto const n_keys     = keys.size();
  bool const big_pool   = static_cast<int64_t>(n_keys) * width > (1 << 20);
  if (width <= 16 && big_pool) {
    rmm::device_buffer padded(static_cast<std::size_t>(n_keys) * 16, stream, mr);
    {
      auto* p = static_cast<char*>(padded.data());
      thrust::for_each_n(rmm::exec_policy_nosync(stream, mr),
                         thrust::counting_iterator<int64_t>(0),
                         static_cast<int64_t>(n_keys) * 16,
                         [=] __device__(int64_t i) {
                           int64_t const k = i / 16, o = i % 16;
                           p[i] = o < width ? kc[k * width + o] : 0;
                         });
    }
    auto const* pool = reinterpret_cast<uint4 const*>(padded.data());
    switch (width) {
      case 1: padded_gather_chunks<1>(pool, ix, out, nbytes, frame); break;
      case 2: padded_gather_chunks<2>(pool, ix, out, nbytes, frame); break;
      case 3: padded_gather_chunks<3>(pool, ix, out, nbytes, frame); break;
      case 4: padded_gather_chunks<4>(pool, ix, out, nbytes, frame); break;
      case 5: padded_gather_chunks<5>(pool, ix, out, nbytes, frame); break;
      case 6: padded_gather_chunks<6>(pool, ix, out, nbytes, frame); break;
      case 7: padded_gather_chunks<7>(pool, ix, out, nbytes, frame); break;
      case 8: padded_gather_chunks<8>(pool, ix, out, nbytes, frame); break;
      case 9: padded_gather_chunks<9>(pool, ix, out, nbytes, frame); break;
      case 10: padded_gather_chunks<10>(pool, ix, out, nbytes, frame); break;
      case 11: padded_gather_chunks<11>(pool, ix, out, nbytes, frame); break;
      case 12: padded_gather_chunks<12>(pool, ix, out, nbytes, frame); break;
      case 13: padded_gather_chunks<13>(pool, ix, out, nbytes, frame); break;
      case 14: padded_gather_chunks<14>(pool, ix, out, nbytes, frame); break;
      case 15: padded_gather_chunks<15>(pool, ix, out, nbytes, frame); break;
      case 16: padded_gather_chunks<16>(pool, ix, out, nbytes, frame); break;
    }
  } else {
    thrust::for_each_n(rmm::exec_policy_nosync(stream, mr),
                       thrust::counting_iterator<int64_t>(0),
                       nchunks,
                       [=] __device__(int64_t t) {
                         int64_t const base = t * 16;
                         int const n = nbytes - base < 16 ? static_cast<int>(nbytes - base) : 16;
                         char b[16];
                         int64_t row       = base / width;
                         int64_t row_start = row * width;
                         for (int i = 0; i < n;) {
                           auto const key_base = static_cast<int64_t>(ix[row]) * width;
                           int const in_row    = static_cast<int>(base + i - row_start);
                           int64_t const left  = width - in_row;
                           int const take      = left < n - i ? static_cast<int>(left) : n - i;
                           for (int j = 0; j < take; ++j)
                             b[i + j] = kc[key_base + in_row + j];
                           i += take;
                           row_start += width;
                           ++row;
                         }
                         if (n == 16) {
                           uint4 v;
                           memcpy(&v, b, 16);
                           *reinterpret_cast<uint4*>(out + base) = v;
                         } else {
                           for (int i = 0; i < n; ++i)
                             out[base + i] = b[i];
                         }
                       });
  }
  return output;
}

// Small variable-width dictionary: every (key, destination byte shift) pair is pre-rendered as
// masked 32-bit words in shared memory, so a row's chars are a few word ORs into a shared stage
// that each tile of kSmallDictTile rows then writes out with aligned 16B stores. The word table
// and the stage are sized from the longest key and stay under the 48 KiB default together.
constexpr int kSmallDictThreads    = 256;
constexpr int kSmallDictRowsPerThr = 4;
constexpr int kSmallDictTile       = kSmallDictThreads * kSmallDictRowsPerThr;
constexpr int kSmallDictMaxKeys    = 256;
constexpr int kSmallDictMaxLen     = 32;
constexpr int kSmallDictMaxTable   = 8 << 10;

// Words per pre-rendered key: a key of up to max_len bytes shifted by up to 3 bytes.
constexpr int small_dict_words_per_key(int max_len) { return (max_len + 3) / 4 + 1; }
constexpr int small_dict_stage_bytes(int max_len) { return 32 + kSmallDictTile * max_len; }

// The tile's keys, read as one int4 per thread when the indices are 16B-aligned and the tile full.
template <bool Vec>
__device__ void load_tile_keys(int32_t const* __restrict__ ix,
                               cudf::size_type n,
                               int64_t first_row,
                               int32_t (&key)[kSmallDictRowsPerThr])
{
  if (Vec && first_row + kSmallDictRowsPerThr <= n) {
    int4 const v = *reinterpret_cast<int4 const*>(ix + first_row);
    key[0] = v.x, key[1] = v.y, key[2] = v.z, key[3] = v.w;
    return;
  }
#pragma unroll
  for (int j = 0; j < kSmallDictRowsPerThr; ++j)
    key[j] = first_row + j < n ? ix[first_row + j] : -1;
}

// Shared key offsets rebased to the first key; a key of -1 (past the end) has length zero.
__device__ void load_key_offsets(int32_t const* __restrict__ key_offsets,
                                 int32_t num_keys,
                                 int32_t* koff)
{
  int32_t const k0 = key_offsets[0];
  for (int k = static_cast<int>(threadIdx.x); k <= num_keys; k += kSmallDictThreads)
    koff[k] = key_offsets[k] - k0;
}

__device__ int32_t key_length(int32_t const* koff, int32_t key)
{
  return key < 0 ? 0 : koff[key + 1] - koff[key];
}

template <bool Vec>
__global__ void __launch_bounds__(kSmallDictThreads)
  small_dict_tile_bytes(int32_t const* __restrict__ ix,
                        cudf::size_type n,
                        int32_t const* __restrict__ key_offsets,
                        int32_t num_keys,
                        int64_t num_tiles,
                        int32_t* __restrict__ tile_bytes)
{
  using block_reduce = cub::BlockReduce<int32_t, kSmallDictThreads>;
  __shared__ typename block_reduce::TempStorage reduce_storage;
  __shared__ int32_t koff[kSmallDictMaxKeys + 1];
  load_key_offsets(key_offsets, num_keys, koff);
  __syncthreads();
  for (int64_t tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
    int32_t key[kSmallDictRowsPerThr];
    load_tile_keys<Vec>(ix, n, tile * kSmallDictTile + threadIdx.x * kSmallDictRowsPerThr, key);
    int32_t sum = 0;
#pragma unroll
    for (int j = 0; j < kSmallDictRowsPerThr; ++j)
      sum += key_length(koff, key[j]);
    sum = block_reduce(reduce_storage).Sum(sum);
    if (threadIdx.x == 0) tile_bytes[tile] = sum;
    __syncthreads();
  }
}

// Writes the strings offsets and chars of every tile, starting each at its scanned tile base.
// Dynamic shared memory: the stage (small_dict_stage_bytes) followed by the word table.
template <bool Vec>
__global__ void __launch_bounds__(kSmallDictThreads)
  small_dict_gather(int32_t const* __restrict__ ix,
                    cudf::size_type n,
                    int32_t const* __restrict__ key_offsets,
                    char const* __restrict__ key_chars,
                    int32_t num_keys,
                    int32_t max_len,
                    int64_t num_tiles,
                    int32_t const* __restrict__ tile_base,
                    int32_t* __restrict__ out_offsets,
                    char* __restrict__ out)
{
  using block_scan = cub::BlockScan<int32_t, kSmallDictThreads>;
  __shared__ typename block_scan::TempStorage scan_storage;
  __shared__ int32_t koff[kSmallDictMaxKeys + 1];
  extern __shared__ __align__(16) uint32_t dynamic_smem[];
  int const stage_words   = small_dict_stage_bytes(max_len) / 4;
  int const words_per_key = small_dict_words_per_key(max_len);
  uint32_t* const stage32 = dynamic_smem;
  uint32_t* const table   = dynamic_smem + stage_words;
  char* const stage       = reinterpret_cast<char*>(stage32);

  int const tid = static_cast<int>(threadIdx.x);
  load_key_offsets(key_offsets, num_keys, koff);
  for (int w = tid; w < stage_words; w += kSmallDictThreads)
    stage32[w] = 0;
  __syncthreads();
  // table[(k * 4 + s) * words_per_key + i]: word i of key k when its first byte lands at byte s.
  int32_t const k0 = key_offsets[0];
  for (int e = tid; e < num_keys * 4 * words_per_key; e += kSmallDictThreads) {
    int const k = e / (4 * words_per_key), s = (e / words_per_key) % 4, i = e % words_per_key;
    int const len = key_length(koff, k);
    uint32_t v    = 0;
    for (int b = 0; b < 4; ++b) {
      int const j = 4 * i + b - s;
      if (j >= 0 && j < len)
        v |= static_cast<uint32_t>(static_cast<unsigned char>(key_chars[k0 + koff[k] + j]))
             << (8 * b);
    }
    table[e] = v;
  }
  __syncthreads();

  for (int64_t tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
    int64_t const first_row = tile * kSmallDictTile + tid * kSmallDictRowsPerThr;
    int32_t key[kSmallDictRowsPerThr];
    load_tile_keys<Vec>(ix, n, first_row, key);
    int32_t const base = tile_base[tile];
    int32_t len[kSmallDictRowsPerThr];
#pragma unroll
    for (int j = 0; j < kSmallDictRowsPerThr; ++j)
      len[j] = key_length(koff, key[j]);
    int32_t pos[kSmallDictRowsPerThr];
    int32_t bytes;
    block_scan(scan_storage).ExclusiveSum(len, pos, bytes);

    if (Vec && first_row + kSmallDictRowsPerThr <= n) {
      *reinterpret_cast<int4*>(out_offsets + first_row) =
        make_int4(base + pos[0], base + pos[1], base + pos[2], base + pos[3]);
    } else {
#pragma unroll
      for (int j = 0; j < kSmallDictRowsPerThr; ++j)
        if (first_row + j < n) out_offsets[first_row + j] = base + pos[j];
    }
    if (tile == num_tiles - 1 && tid == 0) out_offsets[n] = base + bytes;

    // Rows share their boundary words, so each ORs its words into the zeroed stage.
    int64_t const abase = base & ~int64_t{15};
    int const shift     = static_cast<int>(base - abase);
#pragma unroll
    for (int j = 0; j < kSmallDictRowsPerThr; ++j) {
      if (key[j] < 0) continue;
      int const d         = shift + pos[j];
      int const s         = d & 3;
      uint32_t const* src = table + (key[j] * 4 + s) * words_per_key;
      for (int i = 0; i < (s + len[j] + 3) >> 2; ++i)
        atomicOr(&stage32[(d >> 2) + i], src[i]);
    }
    __syncthreads();

    int const end = shift + bytes;
    for (int lo = tid * 16; lo < end; lo += kSmallDictThreads * 16) {
      int const hi = lo + 16;
      if (lo >= shift && hi <= end) {
        *reinterpret_cast<uint4*>(out + abase + lo) = *reinterpret_cast<uint4 const*>(stage + lo);
      } else {
        for (int b = max(lo, shift); b < min(hi, end); ++b)
          out[abase + b] = stage[b];
      }
      *reinterpret_cast<uint4*>(stage + lo) = make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
  }
}

// Blocks that fit on the device at once; the small-dictionary kernels loop over tiles from there.
template <typename Kernel>
int64_t resident_blocks(Kernel kernel, std::size_t dynamic_smem)
{
  int device = 0, sms = 0, per_sm = 0;
  throw_if_cuda_error(cudaGetDevice(&device), "dictionary decode: current device");
  throw_if_cuda_error(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device),
                      "dictionary decode: SM count");
  throw_if_cuda_error(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kSmallDictThreads, dynamic_smem),
    "dictionary decode: occupancy");
  return int64_t{sms} * std::max(per_sm, 1);
}

// Null-free decode of a small variable-width dictionary: per-tile byte counts and their scan give
// each tile's chars base, then small_dict_gather writes the offsets and chars. nullptr =
// ineligible.
std::unique_ptr<cudf::column> try_decode_small_dictionary(cudf::strings_column_view const& keys,
                                                          cudf::column_view const& indices,
                                                          decode_frame& frame)
{
  auto const stream = frame.stream();
  auto const mr     = frame.mr();
  if (indices.null_count() > 0 || keys.parent().null_count() > 0) return nullptr;
  if (indices.type().id() != cudf::type_id::INT32) return nullptr;
  if (keys.offsets().type().id() != cudf::type_id::INT32) return nullptr;
  auto const num_keys = keys.size();
  auto const n_rows   = indices.size();
  if (num_keys <= 0 || num_keys > kSmallDictMaxKeys || n_rows == 0) return nullptr;

  auto const* const d_key_offsets = keys.offsets().data<int32_t>() + keys.offset();
  std::vector<int32_t> key_offsets(static_cast<std::size_t>(num_keys) + 1);
  read_device_bytes_completed(
    key_offsets.data(), d_key_offsets, key_offsets.size() * sizeof(int32_t), stream);
  int32_t max_len = 0;
  for (cudf::size_type k = 0; k < num_keys; ++k)
    max_len = std::max(max_len, key_offsets[k + 1] - key_offsets[k]);
  auto const table_bytes =
    static_cast<std::size_t>(num_keys) * 4 * small_dict_words_per_key(max_len) * sizeof(uint32_t);
  if (max_len <= 0 || max_len > kSmallDictMaxLen || table_bytes > kSmallDictMaxTable ||
      static_cast<int64_t>(n_rows) * max_len > std::numeric_limits<cudf::size_type>::max())
    return nullptr;

  auto const* const ix = indices.data<int32_t>();
  bool const vec       = reinterpret_cast<std::uintptr_t>(ix) % 16 == 0;
  auto const num_tiles = (static_cast<int64_t>(n_rows) + kSmallDictTile - 1) / kSmallDictTile;
  rmm::device_uvector<int32_t> tile_base(static_cast<std::size_t>(num_tiles) + 1, stream, mr);
  {
    auto const kernel = vec ? small_dict_tile_bytes<true> : small_dict_tile_bytes<false>;
    auto const grid   = std::min(num_tiles, resident_blocks(kernel, 0));
    kernel<<<static_cast<unsigned>(grid), kSmallDictThreads, 0, stream.get()>>>(
      ix, n_rows, d_key_offsets, num_keys, num_tiles, tile_base.data());
    throw_if_cuda_error(cudaGetLastError(), "dictionary decode: tile byte counts");
  }
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, mr),
                         tile_base.begin(),
                         tile_base.end(),
                         tile_base.begin(),
                         int32_t{0});
  int32_t total_bytes = 0;
  read_device_bytes_completed(
    &total_bytes, tile_base.data() + num_tiles, sizeof total_bytes, stream);

  auto offsets = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, n_rows + 1, cudf::mask_state::UNALLOCATED, stream, mr);
  rmm::device_buffer chars(static_cast<std::size_t>(total_bytes), stream, mr);
  {
    auto const kernel = vec ? small_dict_gather<true> : small_dict_gather<false>;
    auto const smem   = static_cast<std::size_t>(small_dict_stage_bytes(max_len)) + table_bytes;
    auto const grid   = std::min(num_tiles, resident_blocks(kernel, smem));
    kernel<<<static_cast<unsigned>(grid), kSmallDictThreads, smem, stream.get()>>>(
      ix,
      n_rows,
      d_key_offsets,
      keys.chars_begin(stream),
      num_keys,
      max_len,
      num_tiles,
      tile_base.data(),
      offsets->mutable_view().data<int32_t>(),
      static_cast<char*>(chars.data()));
    throw_if_cuda_error(cudaGetLastError(), "dictionary decode: small-dictionary gather");
  }
  return cudf::make_strings_column(
    n_rows,
    std::move(offsets),
    std::move(chars),
    0,
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr));
}

std::unique_ptr<dictionary_compressed_representation> dictionary_compress_impl(
  cudf::column_view const& col, ::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  if (col.type().id() != cudf::type_id::STRING) {
    throw std::runtime_error("dictionary_compressor: column must be STRING, got '" +
                             type_id_to_name(col.type()) + "'");
  }
  auto const n = col.size();
  if (n < 0) { throw std::runtime_error("dictionary_compressor: column size is negative"); }
  if (static_cast<size_t>(n) > MAX_INDICES) {
    throw std::runtime_error("dictionary_compressor: column size exceeds maximum");
  }
  if (n == 0) {
    auto empty_dict = cudf::make_empty_column(cudf::data_type(cudf::type_id::DICTIONARY32));
    return dictionary_compressed_representation::from_encoded_column(
      std::move(empty_dict), stream, mr);
  }

  if (static_cast<size_t>(n) > kDictCardCheckMinRows) {
    cudf::approx_distinct_count sketch(cudf::table_view{{col}},
                                       12,  // precision -> ~1.6% standard error
                                       cudf::null_policy::INCLUDE,
                                       cudf::nan_policy::NAN_IS_NULL,
                                       stream);
    auto const keys = sketch.estimate(stream);
    if (static_cast<double>(keys) > kDictMaxCardFraction * static_cast<double>(n)) {
      throw std::runtime_error("dictionary_compressor: cardinality too high (skipping)");
    }
  }

  auto dict_col = cudf::dictionary::encode(col, cudf::data_type(cudf::type_id::INT32), stream, mr);
  return dictionary_compressed_representation::from_encoded_column(std::move(dict_col), stream, mr);
}

}  // namespace

std::unique_ptr<cudf::column> make_constant_width_offsets(cudf::size_type rows,
                                                          std::int32_t width,
                                                          ::cuda::stream_ref stream,
                                                          rmm::device_async_resource_ref mr)
{
  if (rows < 0 || width < 0) {
    throw std::invalid_argument("constant-width offsets: negative row count or width");
  }
  if (rows == std::numeric_limits<cudf::size_type>::max() ||
      static_cast<std::int64_t>(rows) * width > std::numeric_limits<cudf::size_type>::max()) {
    throw std::overflow_error(
      "constant-width offsets: offset count or total bytes exceed the INT32 range");
  }
  auto offsets = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, rows + 1, cudf::mask_state::UNALLOCATED, stream, mr);
  auto* const data = offsets->mutable_view().data<std::int32_t>();
  thrust::tabulate(
    rmm::exec_policy_nosync(stream, mr), data, data + rows + 1, [=] __device__(int64_t i) {
      return static_cast<std::int32_t>(i * width);
    });
  return offsets;
}

std::unique_ptr<dictionary_compressed_representation>
dictionary_compressed_representation::from_encoded_column(std::unique_ptr<cudf::column> dict_col,
                                                          ::cuda::stream_ref stream,
                                                          rmm::device_async_resource_ref mr)
{
  if (!dict_col) throw std::invalid_argument("dictionary construction: missing column");
  auto result = std::make_unique<dictionary_compressed_representation>(std::move(dict_col));
  try {
    auto const& column = *result->dict_column;
    result->constant_key_width =
      column.size() > 0
        ? measure_constant_key_width(cudf::dictionary_column_view(column.view()).keys(), stream, mr)
        : 0;
    stream.sync();
  } catch (...) {
    // Keep the column alive until pending work that reads it drains.
    (void)cudaStreamSynchronize(stream.get());
    throw;
  }
  return result;
}

std::unique_ptr<cudf::column> dictionary_compressed_representation::decompress(
  decode_frame& frame) const
{
  auto const stream = frame.stream();
  auto const mr     = frame.mr();
  nvtx_scoped_range r{"dictionary_decompress"};
  // Decode from the stored dictionary column.
  if (dict_column == nullptr) { throw std::invalid_argument("dictionary decode: missing column"); }
  if (dict_column->size() == 0) {
    return cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  }
  // The encoded column is null-free (validity lives in the plan tree's sidecar), so the
  // constant-width fast path always applies when the keys allow it.
  cudf::dictionary_column_view dv(dict_column->view());
  if (auto output = try_decode_constant_width(
        cudf::strings_column_view(dv.keys()), dv.indices(), constant_key_width, frame))
    return output;
  if (auto output =
        try_decode_small_dictionary(cudf::strings_column_view(dv.keys()), dv.indices(), frame))
    return output;
  return cudf::dictionary::decode(dict_column->view(), stream, mr);
}

std::unique_ptr<cudf::column> dictionary_compressed_representation::decompress_predicate(
  decode_predicate const& pred, decode_frame& frame) const
{
  auto const stream = frame.stream();
  auto const mr     = frame.mr();
  nvtx_scoped_range r{"dictionary_decompress_predicate"};
  if (dict_column == nullptr) {
    throw std::invalid_argument("dictionary predicate: missing column");
  }
  if (!pred.active()) { return nullptr; }

  auto const n_rows = dict_column->size();
  auto const bool_t = cudf::data_type{cudf::type_id::BOOL8};
  if (n_rows == 0) { return cudf::make_empty_column(bool_t); }

  cudf::dictionary_column_view const dv(dict_column->view());
  auto const keys    = dv.keys();
  auto const n_keys  = keys.size();
  auto const indices = dv.indices();

  // The index lookup below reads indices[i] unconditionally, so anything other
  // than the INT32 index type encode produces is left to the generic path.
  if (indices.type().id() != cudf::type_id::INT32) { return nullptr; }
  if (keys.type().id() != cudf::type_id::STRING) { return nullptr; }
  // A null key would make the per-key comparison null rather than false, and the
  // OR-accumulate below would then propagate it; leave that shape to the generic path
  // instead of carrying a tri-state accumulate.
  if (keys.null_count() > 0) { return nullptr; }

  // One bool per distinct value: key in equals_any. The key set is the column's whole
  // distinct-value population (four entries for l_shipinstruct), so this pass is noise next to the
  // row-length pass below, which is the entire point: this is the work that replaces the decode
  // gather. No cuDF scalar is constructed, so nothing here waits for the stream or borrows from the
  // process-global pinned pool; the needles and the table live until the row pass has been queued.
  auto const needles = upload_needles(pred.equals_any, frame);
  auto lut =
    cudf::make_fixed_width_column(bool_t, n_keys, cudf::mask_state::UNALLOCATED, stream, mr);
  cudf::strings_column_view const key_strings(keys);
  thrust::tabulate(rmm::exec_policy_nosync(stream, mr),
                   lut->mutable_view().begin<bool>(),
                   lut->mutable_view().end<bool>(),
                   key_matches_any_needle{cudf::detail::offsetalator_factory::make_input_iterator(
                                            key_strings.offsets(), key_strings.offset()),
                                          key_strings.chars_begin(stream),
                                          needles.view});

  // The dictionary column is null-free: row validity is reattached by the caller from the
  // plan tree's sidecar, so the result carries no mask of its own.
  auto output =
    cudf::make_fixed_width_column(bool_t, n_rows, cudf::mask_state::UNALLOCATED, stream, mr);

  auto* d_out       = output->mutable_view().data<bool>();
  auto const* d_lut = lut->view().data<bool>();
  auto const* d_idx = indices.data<int32_t>();
  // Clamp rather than trust the index: a corrupt file must not read outside the LUT.
  thrust::transform(rmm::exec_policy_nosync(stream, mr),
                    d_idx,
                    d_idx + n_rows,
                    d_out,
                    [d_lut, n_keys] __device__(int32_t code) {
                      return code >= 0 && code < n_keys ? d_lut[code] : false;
                    });
  return output;
}

std::unique_ptr<compressed_representation> dictionary_compressor::compress(
  cudf::column_view column_to_compress,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  return dictionary_compress_impl(column_to_compress, stream, mr);
}

}  // namespace simpatico
