// SPDX-License-Identifier: Apache-2.0
// GPU OnPair encoder. The host trains the dictionary on a strided sample of the characters
// with the reference trainer (onpair_cpp); everything that scales with the data runs on the
// GPU:
//   1. each thread greedily parses one segment of the characters, longest prefix first,
//      against an open-addressing table of token fingerprints, in shared memory when it
//      fits: the token lengths that can start with the next two bytes come from a 64K-entry
//      mask, and a fingerprint hit is verified against the token's bytes. Codes go to a
//      per-segment scratch slot in one pass and are compacted after a scan of the counts;
//      segments too poorly compressible for their slot are re-parsed in place;
//   2. the dictionary is pruned to the tokens the codes use and renumbered densely;
//   3. the codes are bitpacked to ceil(log2(used tokens)) bits, and the per-K sidecars and
//      the dictionary planes the FastPair decoder reads are built.

#include "onpair_representation.hpp"

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda_runtime.h>
#include <thrust/scan.h>

#include <onpair/encoding/training/trainer.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace simpatico {
namespace {

inline void check_cuda(cudaError_t e, char const* what)
{
  if (e != cudaSuccess) {
    cudaGetLastError();
    throw std::runtime_error(std::string("onpair encode: ") + what + ": " + cudaGetErrorString(e));
  }
}

constexpr std::uint32_t kMaxToken = 16;
constexpr int kParseThreads       = 512;

__host__ __device__ inline std::uint64_t low_mask(std::uint32_t bytes)
{
  return bytes >= 8 ? ~std::uint64_t{0} : (std::uint64_t{1} << (8 * bytes)) - 1;
}

__host__ __device__ inline std::uint32_t slot_hash(std::uint64_t lo,
                                                   std::uint64_t hi,
                                                   std::uint32_t len)
{
  // 32-bit multiplies only: the parse probes several lengths per token and is bound by
  // the instructions it issues, of which 64-bit multiplies take several each.
  std::uint32_t h = static_cast<std::uint32_t>(lo) * 0x9E3779B1u +
                    static_cast<std::uint32_t>(lo >> 32) * 0x85EBCA77u +
                    static_cast<std::uint32_t>(hi) * 0xC2B2AE3Du +
                    static_cast<std::uint32_t>(hi >> 32) * 0x27D4EB2Fu + len * 0x165667B1u;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  return h ^ (h >> 13);
}

struct parse_table {
  /// Open addressing over the tokens of 2+ bytes: (fingerprint << 20) | ((length - 1) << 16)
  /// | token, 0 when empty.
  std::uint32_t const* fingerprints;
  std::uint32_t slot_mask;
  bool shared;                   ///< The slots fit in shared memory, so the parse stages them.
  std::uint16_t const* byte_id;  ///< [256] the single-byte token for each byte.
  std::uint8_t const* dict16;    ///< [tokens * 16] zero-padded token bytes, to verify a match.
  /// [65536] bit L-1 set if a token of length L >= 2 starts with the two bytes (LSB first).
  std::uint16_t const* pair_len_mask;
};

/// Fingerprint slots fit in shared memory up to this many (64 KiB).
constexpr std::uint32_t kMaxSharedSlots = 16384;

__host__ __device__ inline std::uint32_t slot_fingerprint(std::uint32_t hash) { return hash >> 20; }

/// Loads the 16 bytes at @p p (zero past @p avail) with aligned 8-byte loads only, so no
/// load touches a word that holds no requested byte.
__device__ inline void load_window(std::uint8_t const* p,
                                   std::uint32_t avail,
                                   std::uint64_t& lo,
                                   std::uint64_t& hi)
{
  auto const addr  = reinterpret_cast<std::uintptr_t>(p);
  auto const* base = reinterpret_cast<std::uint64_t const*>(addr & ~std::uintptr_t{7});
  auto const shift = static_cast<std::uint32_t>(addr & 7) * 8;
  auto const last  = addr + min(avail, kMaxToken) - 1;
  auto word        = [&](int k) -> std::uint64_t {
    auto const w = reinterpret_cast<std::uintptr_t>(base + k);
    return w <= last ? __ldg(base + k) : 0;
  };
  std::uint64_t const w0 = word(0), w1 = word(1), w2 = shift ? word(2) : 0;
  lo = shift ? (w0 >> shift) | (w1 << (64 - shift)) : w0;
  hi = shift ? (w1 >> shift) | (w2 << (64 - shift)) : w1;
  if (avail < 16) {
    hi &= low_mask(avail > 8 ? avail - 8 : 0);
    lo &= low_mask(avail);
  }
}

/// The lengths, as bit L-1, of the tokens of 2+ bytes that may prefix the window.
__device__ inline std::uint32_t candidate_lengths(parse_table const& t,
                                                  std::uint64_t lo,
                                                  std::uint32_t avail)
{
  if (avail < 2) return 0;
  return __ldg(t.pair_len_mask + (lo & 0xffff)) & ((1u << min(avail, kMaxToken)) - 1);
}

/// The longest dictionary token that prefixes the window; returns (length << 16) | token.
/// Probes the fingerprint slots @p fps; only a fingerprint hit reads the token to verify it.
__device__ inline std::uint32_t longest_match(parse_table const& t,
                                              std::uint32_t const* fps,
                                              std::uint64_t lo,
                                              std::uint64_t hi,
                                              std::uint32_t avail)
{
  auto const first   = static_cast<std::uint32_t>(lo & 0xff);
  std::uint32_t lens = candidate_lengths(t, lo, avail);
  while (lens) {
    std::uint32_t const len = 32 - __clz(lens);
    lens &= ~(1u << (len - 1));
    std::uint64_t const klo  = lo & low_mask(len);
    std::uint64_t const khi  = len > 8 ? hi & low_mask(len - 8) : 0;
    std::uint32_t const h    = slot_hash(klo, khi, len);
    std::uint32_t const want = (slot_fingerprint(h) << 20) | ((len - 1) << 16);
    for (std::uint32_t s = h & t.slot_mask;; s = (s + 1) & t.slot_mask) {
      std::uint32_t const f = fps[s];
      if (f == 0) break;
      if ((f & 0xffff0000u) != want) continue;
      auto const* tok    = reinterpret_cast<ulonglong2 const*>(t.dict16) + (f & 0xffff);
      ulonglong2 const v = __ldg(tok);
      if (v.x == klo && v.y == khi) return (len << 16) | (f & 0xffff);
    }
  }
  return (1u << 16) | t.byte_id[first];
}

/// Greedily parses the bytes [pos, end) of @p chars, appending the codes to out[n..cap)
/// when Write; returns n plus the number of codes.
template <bool Write>
__device__ inline std::uint32_t parse_range(std::uint8_t const* chars,
                                            std::uint64_t pos,
                                            std::uint64_t end,
                                            parse_table const& table,
                                            std::uint32_t const* fps,
                                            std::uint32_t n,
                                            std::uint32_t cap,
                                            std::uint16_t* out)
{
  while (pos < end) {
    auto const avail = static_cast<std::uint32_t>(min(end - pos, std::uint64_t{kMaxToken}));
    std::uint64_t lo, hi;
    load_window(chars + pos, avail, lo, hi);
    std::uint32_t const m = longest_match(table, fps, lo, hi, avail);
    if constexpr (Write) {
      if (n < cap) out[n] = static_cast<std::uint16_t>(m & 0xffff);
    }
    ++n;
    pos += m >> 16;
  }
  return n;
}

/// The fingerprint slots to probe: staged in shared memory when they fit.
__device__ inline std::uint32_t const* stage_table(parse_table const& table, std::uint32_t* smem)
{
  if (!table.shared) return table.fingerprints;
  auto const* src = reinterpret_cast<uint4 const*>(table.fingerprints);
  auto* dst       = reinterpret_cast<uint4*>(smem);
  for (std::uint32_t i = threadIdx.x; i < (table.slot_mask + 1) / 4; i += blockDim.x)
    dst[i] = src[i];
  __syncthreads();
  return smem;
}

/// The first pass (Overflow = false) parses every segment, writes its first @p cap codes
/// to its scratch slot (codes + seg * cap) and its code count to seg_counts, and raises
/// @p overflowed if any segment has more. The overflow pass re-parses just those
/// segments, writing their codes at seg_start in the final codes. Segment k is the bytes
/// [k * segment_bytes, (k + 1) * segment_bytes), or, given @p offsets, the strings
/// [seg_first[k], seg_first[k + 1]), each parsed on its own.
template <bool Overflow, typename Offset>
__global__ void __launch_bounds__(kParseThreads) parse_segments(std::uint8_t const* chars,
                                                                std::uint64_t n_chars,
                                                                std::uint32_t segment_bytes,
                                                                Offset const* offsets,
                                                                std::uint32_t const* seg_first,
                                                                std::uint64_t n_segments,
                                                                parse_table table,
                                                                std::uint32_t cap,
                                                                std::uint32_t* seg_counts,
                                                                std::uint32_t const* seg_start,
                                                                std::uint16_t* codes,
                                                                std::uint32_t* overflowed)
{
  extern __shared__ uint4 fps_storage[];
  auto const* fps = stage_table(table, reinterpret_cast<std::uint32_t*>(fps_storage));

  std::uint64_t const seg = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (seg >= n_segments) return;
  if (Overflow && seg_counts[seg] <= cap) return;
  std::uint16_t* out          = Overflow ? codes + seg_start[seg] : codes + seg * cap;
  std::uint32_t const out_cap = Overflow ? ~0u : cap;
  std::uint32_t n             = 0;
  if (offsets == nullptr) {
    std::uint64_t const begin = seg * segment_bytes;
    n                         = parse_range<true>(chars,
                          begin,
                          std::min<std::uint64_t>(begin + segment_bytes, n_chars),
                          table,
                          fps,
                          0,
                          out_cap,
                          out);
  } else {
    for (std::uint64_t r = seg_first[seg]; r < seg_first[seg + 1]; ++r)
      n = parse_range<true>(chars,
                            static_cast<std::uint64_t>(offsets[r]),
                            static_cast<std::uint64_t>(offsets[r + 1]),
                            table,
                            fps,
                            n,
                            out_cap,
                            out);
  }
  if constexpr (!Overflow) {
    seg_counts[seg] = n;
    if (n > cap) *overflowed = 1;
  }
}

/// seg_first[k], k <= n_segments: the first string starting at or past byte k *
/// segment_bytes, n_strings for k = n_segments. Segments so start on string starts and
/// hold about segment_bytes, more where a string is longer.
template <typename Offset>
__global__ void align_segments(Offset const* offsets,
                               std::uint64_t n_strings,
                               std::uint32_t segment_bytes,
                               std::uint64_t n_segments,
                               std::uint32_t* seg_first)
{
  std::uint64_t const k = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (k > n_segments) return;
  if (k == n_segments) {
    seg_first[k] = static_cast<std::uint32_t>(n_strings);
    return;
  }
  std::uint64_t const target = k * segment_bytes;
  std::uint64_t lo = 0, hi = n_strings;
  while (lo < hi) {
    std::uint64_t const mid = (lo + hi) / 2;
    if (static_cast<std::uint64_t>(offsets[mid]) < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  seg_first[k] = static_cast<std::uint32_t>(lo);
}

/// Moves each segment's codes from its scratch slot to seg_start, one warp per segment.
__global__ void compact_codes(std::uint16_t const* scratch,
                              std::uint32_t cap,
                              std::uint32_t const* seg_counts,
                              std::uint32_t const* seg_start,
                              std::uint64_t n_segments,
                              std::uint16_t* codes)
{
  std::uint64_t const seg = (std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x) / 32;
  if (seg >= n_segments) return;
  std::uint32_t const n = seg_counts[seg];
  if (n > cap) return;
  std::uint16_t const* src = scratch + seg * cap;
  std::uint16_t* dst       = codes + seg_start[seg];
  for (std::uint32_t i = threadIdx.x & 31; i < n; i += 32)
    dst[i] = src[i];
}

/// Counts the codes of n_probe segments of probe_len bytes, one every @p stride bytes.
__global__ void __launch_bounds__(kParseThreads) probe_segments(std::uint8_t const* chars,
                                                                std::uint64_t n_chars,
                                                                std::uint64_t stride,
                                                                std::uint32_t probe_len,
                                                                std::uint64_t n_probe,
                                                                parse_table table,
                                                                unsigned long long* n_codes)
{
  extern __shared__ uint4 fps_storage[];
  auto const* fps = stage_table(table, reinterpret_cast<std::uint32_t*>(fps_storage));

  std::uint64_t const seg = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  std::uint32_t n         = 0;
  if (seg < n_probe) {
    std::uint64_t const begin = seg * stride;
    n                         = parse_range<false>(
      chars, begin, std::min<std::uint64_t>(begin + probe_len, n_chars), table, fps, 0, 0, nullptr);
  }
  for (int o = 16; o > 0; o >>= 1)
    n += __shfl_down_sync(0xffffffffu, n, o);
  if ((threadIdx.x & 31) == 0 && n) atomicAdd(n_codes, static_cast<unsigned long long>(n));
}

__global__ void mark_used(std::uint16_t const* codes, std::uint64_t n, std::uint32_t* used)
{
  for (std::uint64_t i = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x; i < n;
       i += std::uint64_t{gridDim.x} * blockDim.x) {
    // Few distinct codes, so most stores would hit a line other SMs are writing too.
    if (!used[codes[i]]) used[codes[i]] = 1;
  }
}

/// Writes the pruned dictionary planes of @p d tokens for w-byte first reads: source token
/// t becomes token remap[t].
__global__ void build_planes(std::uint32_t const* used,
                             std::uint32_t const* remap,
                             std::uint32_t source_size,
                             std::uint8_t const* src_dict16,
                             std::uint8_t const* src_lens,
                             std::uint32_t d,
                             int w,
                             std::uint8_t* dict,
                             std::uint8_t* new_lens)
{
  std::uint32_t const t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= source_size || !used[t]) return;
  std::size_t const i = remap[t];
  for (int b = 0; b < 16; ++b) {
    std::uint8_t const v = src_dict16[std::size_t{t} * 16 + b];
    if (w == 16) {
      dict[i * 16 + b] = v;
    } else {
      dict[(b < 8 ? i : d + i) * 8 + (b & 7)] = v;
    }
  }
  new_lens[i] = src_lens[t];
}

__global__ void pack_nibbles(std::uint8_t const* lens, std::uint32_t d, std::uint8_t* lens_nib)
{
  std::uint32_t const j = blockIdx.x * blockDim.x + threadIdx.x;
  if (2 * j >= d) return;
  std::uint32_t v = lens[2 * j] - 1;
  if (2 * j + 1 < d) v |= (lens[2 * j + 1] - 1u) << 4;
  lens_nib[j] = static_cast<std::uint8_t>(v);
}

/// Bytes each group of 32 codes decodes to, and the number of codes longer than 8 bytes.
__global__ void group_bytes(std::uint16_t const* codes,
                            std::uint64_t n,
                            std::uint8_t const* src_lens,
                            std::uint32_t* bytes_out,
                            unsigned long long* n_long)
{
  std::uint64_t const g        = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  std::uint64_t const n_groups = (n + 31) / 32;
  std::uint32_t sum = 0, longs = 0;
  if (g < n_groups) {
    std::uint64_t const end = std::min<std::uint64_t>(n, g * 32 + 32);
    for (std::uint64_t i = g * 32; i < end; ++i) {
      std::uint32_t const l = src_lens[codes[i]];
      sum += l;
      longs += l > 8;
    }
    bytes_out[g] = sum;
  }
  for (int o = 16; o > 0; o >>= 1)
    longs += __shfl_down_sync(0xffffffffu, longs, o);
  if ((threadIdx.x & 31) == 0 && longs) atomicAdd(n_long, static_cast<unsigned long long>(longs));
}

/// sidecar[b] = output offset of batch b of kOnPairBatchCodes codes; sidecar[n_batches] =
/// total.
__global__ void build_sidecar(std::uint32_t const* group_prefix,
                              std::uint64_t n_groups,
                              std::uint64_t n_batches,
                              std::uint32_t* sidecar)
{
  std::uint64_t const b = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (b > n_batches) return;
  sidecar[b] = group_prefix[std::min<std::uint64_t>(b * kOnPairK, n_groups)];
}

/// LSB-first bitpacking of remap[codes[i]], one output word per thread; codes past n are 0.
__global__ void pack_codes(std::uint16_t const* codes,
                           std::uint64_t n,
                           std::uint32_t const* remap,
                           std::uint32_t bits,
                           std::uint64_t n_words,
                           std::uint32_t* packed)
{
  std::uint64_t const w = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (w >= n_words) return;
  std::uint64_t const bit0 = w * 32;
  std::uint64_t const i0   = bit0 / bits;
  std::uint64_t const i1   = std::min<std::uint64_t>((bit0 + 31) / bits + 1, n);
  std::uint32_t acc        = 0;
  for (std::uint64_t i = i0; i < i1; ++i) {
    std::uint64_t const c = remap[codes[i]];
    std::int64_t const s  = static_cast<std::int64_t>(i * bits) - static_cast<std::int64_t>(bit0);
    acc |= static_cast<std::uint32_t>(s >= 0 ? c << s : c >> -s);
  }
  packed[w] = acc;
}

unsigned grid_for(std::uint64_t n, unsigned threads)
{
  return static_cast<unsigned>(std::max<std::uint64_t>(1, (n + threads - 1) / threads));
}

/// Calls @p f with the INT32 or INT64 data of @p offsets.
template <typename F>
void with_offsets(cudf::column_view offsets, F&& f)
{
  if (offsets.type().id() == cudf::type_id::INT32) {
    f(offsets.data<std::int32_t>());
  } else {
    f(offsets.data<std::int64_t>());
  }
}

}  // namespace

/// A trained dictionary as the host-side tables the parse uses; immutable once cached.
struct onpair_dictionary {
  std::uint32_t source_size = 0;
  std::vector<std::uint32_t> fingerprints;  ///< Power-of-two, see parse_table.
  std::vector<std::uint16_t> byte_id, pair_len_mask;
  std::vector<std::uint8_t> dict16, lens;
  double fit_bytes_per_code = 0;  ///< Probe bytes per code on the data it was trained for.
};

namespace {

using host_dictionary = onpair_dictionary;

host_dictionary build_host_dictionary(onpair::Dictionary const& dict)
{
  host_dictionary h;
  h.source_size            = static_cast<std::uint32_t>(dict.num_tokens());
  std::uint32_t slot_count = 1;
  while (slot_count < 2 * h.source_size)
    slot_count <<= 1;
  h.fingerprints.assign(slot_count, 0);
  h.byte_id.assign(256, 0);
  h.pair_len_mask.assign(65536, 0);
  h.dict16.assign(std::size_t{h.source_size} * kMaxToken, 0);
  h.lens.resize(h.source_size);
  for (std::uint32_t t = 0; t < h.source_size; ++t) {
    std::uint32_t const b = dict.offsets[t], len = dict.offsets[t + 1] - b;
    if (len == 0 || len > kMaxToken) { throw std::runtime_error("onpair encode: bad token"); }
    std::memcpy(h.dict16.data() + std::size_t{t} * kMaxToken, dict.bytes.data() + b, len);
    h.lens[t]                = static_cast<std::uint8_t>(len);
    std::uint8_t const first = dict.bytes[b];
    if (len == 1) {
      h.byte_id[first] = static_cast<std::uint16_t>(t);
      continue;
    }
    h.pair_len_mask[first | (std::uint32_t{dict.bytes[b + 1]} << 8)] |=
      static_cast<std::uint16_t>(1u << (len - 1));
    std::uint64_t lo = 0, hi = 0;
    std::memcpy(&lo, h.dict16.data() + std::size_t{t} * kMaxToken, 8);
    std::memcpy(&hi, h.dict16.data() + std::size_t{t} * kMaxToken + 8, 8);
    std::uint32_t const hash = slot_hash(lo, hi, len);
    std::uint32_t s          = hash & (slot_count - 1);
    while (h.fingerprints[s] != 0)
      s = (s + 1) & (slot_count - 1);
    h.fingerprints[s] = (slot_fingerprint(hash) << 20) | ((len - 1) << 16) | t;
  }
  return h;
}

/// A host dictionary's tables uploaded for one encode.
struct device_dictionary {
  rmm::device_buffer fingerprints, byte_id, dict16, lens, pair_len_mask;
  parse_table table;

  device_dictionary(host_dictionary const& h,
                    ::cuda::stream_ref stream,
                    rmm::device_async_resource_ref mr)
  {
    auto upload = [&](void const* p, std::size_t bytes) {
      rmm::device_buffer b(bytes, stream, mr);
      check_cuda(cudaMemcpyAsync(b.data(), p, bytes, cudaMemcpyHostToDevice, stream.get()),
                 "upload");
      return b;
    };
    fingerprints  = upload(h.fingerprints.data(), h.fingerprints.size() * sizeof(std::uint32_t));
    byte_id       = upload(h.byte_id.data(), 256 * sizeof(std::uint16_t));
    dict16        = upload(h.dict16.data(), h.dict16.size());
    lens          = upload(h.lens.data(), h.lens.size());
    pair_len_mask = upload(h.pair_len_mask.data(), h.pair_len_mask.size() * sizeof(std::uint16_t));
    table         = parse_table{static_cast<std::uint32_t const*>(fingerprints.data()),
                        static_cast<std::uint32_t>(h.fingerprints.size() - 1),
                        h.fingerprints.size() <= kMaxSharedSlots,
                        static_cast<std::uint16_t const*>(byte_id.data()),
                        static_cast<std::uint8_t const*>(dict16.data()),
                        static_cast<std::uint16_t const*>(pair_len_mask.data())};
  }
};

/// The dynamic shared memory a parse kernel stages @p table in; raises @p kernel's limit to it.
template <typename Kernel>
int parse_shared_bytes(Kernel kernel, parse_table const& table)
{
  if (!table.shared) return 0;
  auto const smem = static_cast<int>((table.slot_mask + 1) * sizeof(std::uint32_t));
  check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
             "set shared memory");
  return smem;
}

constexpr std::uint32_t kProbeSegment = 256;

/// Bytes per code of @p table over the strided probe of the characters.
double probe_bytes_per_code(std::uint8_t const* chars,
                            std::uint64_t n_chars,
                            std::size_t probe_bytes,
                            parse_table const& table,
                            unsigned long long* d_count,
                            ::cuda::stream_ref stream)
{
  std::uint64_t const segs    = (n_chars + kProbeSegment - 1) / kProbeSegment;
  std::uint64_t const n_probe = std::clamp<std::uint64_t>(probe_bytes / kProbeSegment, 1, segs);
  std::uint64_t const stride =
    n_probe == segs ? kProbeSegment : n_chars / n_probe / kProbeSegment * kProbeSegment;
  std::uint64_t const probed = std::min<std::uint64_t>(n_chars, n_probe * kProbeSegment);
  auto const s               = stream.get();
  check_cuda(cudaMemsetAsync(d_count, 0, sizeof(unsigned long long), s), "memset");
  probe_segments<<<grid_for(n_probe, kParseThreads),
                   kParseThreads,
                   parse_shared_bytes(probe_segments, table),
                   s>>>(chars, n_chars, stride, kProbeSegment, n_probe, table, d_count);
  check_cuda(cudaGetLastError(), "probe");
  unsigned long long n_codes = 0;
  check_cuda(cudaMemcpyAsync(&n_codes, d_count, sizeof(n_codes), cudaMemcpyDeviceToHost, s),
             "read probe");
  stream.sync();
  return n_codes ? static_cast<double>(probed) / static_cast<double>(n_codes) : 0.0;
}

/// Most recently used first; bounded, as each entry probes on every cache miss.
class dictionary_cache {
 public:
  static dictionary_cache& instance()
  {
    static dictionary_cache cache;
    return cache;
  }

  std::vector<std::shared_ptr<host_dictionary const>> snapshot(int bits)
  {
    std::lock_guard lock(mutex_);
    std::vector<std::shared_ptr<host_dictionary const>> out;
    for (auto const& e : entries_)
      if (e.bits == bits) out.push_back(e.dict);
    return out;
  }

  void touch(std::shared_ptr<host_dictionary const> const& dict, int bits)
  {
    std::lock_guard lock(mutex_);
    auto it = std::find_if(
      entries_.begin(), entries_.end(), [&](entry const& e) { return e.dict == dict; });
    if (it != entries_.end()) entries_.erase(it);
    entries_.insert(entries_.begin(), entry{bits, dict});
    if (entries_.size() > kCapacity) entries_.pop_back();
  }

 private:
  static constexpr std::size_t kCapacity = 16;
  struct entry {
    int bits;
    std::shared_ptr<host_dictionary const> dict;
  };
  std::mutex mutex_;
  std::vector<entry> entries_;
};

}  // namespace

onpair_training_sample sample_onpair_training_data(std::uint8_t const* chars,
                                                   std::size_t n_chars,
                                                   onpair_encoder_config const& cfg,
                                                   ::cuda::stream_ref stream)
{
  if (n_chars == 0 || cfg.sample_slice_bytes == 0) { return {}; }
  auto const s = stream.get();

  // ── Sample: n_slices runs of slice bytes, evenly strided over the characters ──────────
  std::size_t const slice  = std::min(cfg.sample_slice_bytes, n_chars);
  std::size_t const want   = std::max<std::size_t>(1, cfg.sample_bytes / slice);
  std::size_t const n_full = n_chars / slice;
  onpair_training_sample out;
  auto& sample         = out.bytes;
  auto& sample_offsets = out.offsets;
  sample_offsets.push_back(0);
  if (n_full <= want) {
    sample.resize(n_chars);
    check_cuda(cudaMemcpyAsync(sample.data(), chars, n_chars, cudaMemcpyDeviceToHost, s),
               "copy sample");
    for (std::size_t o = slice; o < n_chars; o += slice)
      sample_offsets.push_back(static_cast<std::uint32_t>(o));
    sample_offsets.push_back(static_cast<std::uint32_t>(n_chars));
  } else {
    std::size_t const stride = n_chars / want;
    sample.resize(want * slice);
    check_cuda(cudaMemcpy2DAsync(
                 sample.data(), slice, chars, stride, slice, want, cudaMemcpyDeviceToHost, s),
               "copy sample");
    for (std::size_t i = 1; i <= want; ++i)
      sample_offsets.push_back(static_cast<std::uint32_t>(i * slice));
  }
  stream.sync();
  return out;
}

std::shared_ptr<onpair_dictionary const> train_onpair_dictionary(
  onpair_training_sample const& sample, onpair_encoder_config const& cfg)
{
  if (sample.offsets.size() < 2) {
    throw std::invalid_argument("onpair train: the sample holds no characters");
  }
  onpair::encoding::TrainingConfig tc;
  tc.bits            = static_cast<onpair::BitWidth>(cfg.bits);
  tc.threshold       = onpair::encoding::DynamicThreshold{1.0};
  tc.seed            = cfg.seed;
  auto const trained = onpair::encoding::train(
    sample.bytes.data(), sample.offsets.data(), sample.offsets.size() - 1, tc);
  return std::make_shared<onpair_dictionary const>(build_host_dictionary(trained.dict));
}

std::size_t onpair_dictionary_bytes(onpair_dictionary const& dict)
{
  return dict.fingerprints.size() * sizeof(std::uint32_t) +
         (dict.byte_id.size() + dict.pair_len_mask.size()) * sizeof(std::uint16_t) +
         dict.dict16.size() + dict.lens.size();
}

std::unique_ptr<onpair_compressed_representation> onpair_compressed_representation::encode(
  std::uint8_t const* chars,
  std::size_t n_chars,
  cudf::data_type type,
  cudf::size_type n_rows,
  onpair_encoder_config const& cfg,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  cudf::column_view string_offsets)
{
  if (n_chars > 0xffffffffull) {
    throw std::invalid_argument("onpair encode: the u32 sidecar holds at most 4 GiB of chars");
  }
  if (cfg.bits < 9 || cfg.bits > 16 || cfg.segment_bytes == 0 || cfg.sample_slice_bytes == 0) {
    throw std::invalid_argument("onpair encode: bad encoder configuration");
  }
  auto const s       = stream.get();
  bool const aligned = cfg.string_aligned;
  if (aligned) {
    auto const id = string_offsets.type().id();
    if (string_offsets.size() < 1 || string_offsets.has_nulls() ||
        (id != cudf::type_id::INT32 && id != cudf::type_id::INT64)) {
      throw std::invalid_argument("onpair encode: string_aligned needs INT32 or INT64 offsets");
    }
    std::int64_t ends[2] = {0, 0};
    with_offsets(string_offsets, [&](auto const* offsets) {
      using Offset = std::remove_cv_t<std::remove_pointer_t<decltype(offsets)>>;
      Offset v[2];
      check_cuda(cudaMemcpyAsync(&v[0], offsets, sizeof(Offset), cudaMemcpyDeviceToHost, s),
                 "read offsets");
      check_cuda(
        cudaMemcpyAsync(
          &v[1], offsets + string_offsets.size() - 1, sizeof(Offset), cudaMemcpyDeviceToHost, s),
        "read offsets");
      stream.sync();
      ends[0] = v[0];
      ends[1] = v[1];
    });
    if (ends[0] != 0 || ends[1] != static_cast<std::int64_t>(n_chars)) {
      throw std::invalid_argument("onpair encode: string offsets do not span the characters");
    }
  }
  auto rep                = std::make_unique<onpair_compressed_representation>(type, n_rows);
  rep->uncompressed_bytes = n_chars;
  rep->string_aligned     = aligned;
  if (n_chars == 0) {
    std::uint32_t const zero = 0;
    rep->code_bits           = 1;
    rep->codes_buf           = rmm::device_buffer(&zero, sizeof(zero), stream, mr);
    rep->sidecar_buf         = rmm::device_buffer(&zero, sizeof(zero), stream, mr);
    stream.sync();
    return rep;
  }

  // ── Reuse a cached dictionary that still fits, or train one ───────────────────────────
  auto& cache = dictionary_cache::instance();
  rmm::device_uvector<unsigned long long> probe_count(1, stream, mr);
  std::shared_ptr<host_dictionary const> hdict = cfg.dictionary;
  std::optional<device_dictionary> ddict;
  if (hdict) {
    ddict.emplace(*hdict, stream, mr);
  } else if (cfg.reuse_dictionaries) {
    for (auto const& candidate : cache.snapshot(cfg.bits)) {
      ddict.emplace(*candidate, stream, mr);
      double const fit = probe_bytes_per_code(
        chars, n_chars, cfg.probe_bytes, ddict->table, probe_count.data(), stream);
      if (fit >= cfg.reuse_min_fit * candidate->fit_bytes_per_code) {
        hdict = candidate;
        break;
      }
      ddict.reset();
    }
  }
  if (!hdict) {
    auto fresh = std::make_shared<host_dictionary>(
      *train_onpair_dictionary(sample_onpair_training_data(chars, n_chars, cfg, stream), cfg));
    ddict.emplace(*fresh, stream, mr);
    if (cfg.reuse_dictionaries) {
      fresh->fit_bytes_per_code = probe_bytes_per_code(
        chars, n_chars, cfg.probe_bytes, ddict->table, probe_count.data(), stream);
    }
    hdict = std::move(fresh);
  }
  if (cfg.reuse_dictionaries && !cfg.dictionary) { cache.touch(hdict, cfg.bits); }
  parse_table const table         = ddict->table;
  std::uint32_t const source_size = hdict->source_size;
  auto const& d_dict16            = ddict->dict16;
  auto const& d_lens              = ddict->lens;

  // ── Parse on the GPU ──────────────────────────────────────────────────────────────────
  auto seg_bytes = static_cast<std::uint32_t>(cfg.segment_bytes);
  if (cfg.min_segments_per_sm > 0) {
    int device = 0, sms = 0;
    check_cuda(cudaGetDevice(&device), "get device");
    check_cuda(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "get SMs");
    std::uint64_t const want = cfg.min_segments_per_sm * static_cast<std::uint64_t>(sms);
    while (seg_bytes > 64 && n_chars / seg_bytes < want)
      seg_bytes /= 2;
  }
  std::uint64_t const n_segs = (n_chars + seg_bytes - 1) / seg_bytes;
  rmm::device_uvector<std::uint32_t> seg_counts(n_segs + 1, stream, mr);
  rmm::device_uvector<std::uint32_t> seg_start(n_segs + 1, stream, mr);
  check_cuda(cudaMemsetAsync(seg_counts.data() + n_segs, 0, sizeof(std::uint32_t), s), "memset");
  // Each segment's codes go to a scratch slot of cap codes, which holds them unless the
  // segment averages under 4 bytes per code; the rare ones that do not fit are re-parsed.
  std::uint32_t const cap = std::max<std::uint32_t>(1, seg_bytes / 4);
  rmm::device_uvector<std::uint16_t> scratch(n_segs * cap, stream, mr);
  rmm::device_uvector<std::uint32_t> overflowed(1, stream, mr);
  check_cuda(cudaMemsetAsync(overflowed.data(), 0, sizeof(std::uint32_t), s), "memset");
  rmm::device_uvector<std::uint32_t> seg_first(aligned ? n_segs + 1 : 0, stream, mr);
  if (aligned) {
    auto const n_strings = static_cast<std::uint64_t>(string_offsets.size() - 1);
    with_offsets(string_offsets, [&](auto const* offsets) {
      align_segments<<<grid_for(n_segs + 1, 256), 256, 0, s>>>(
        offsets, n_strings, seg_bytes, n_segs, seg_first.data());
    });
  }
  auto parse = [&](auto overflow, auto const* offsets, std::uint16_t* out) {
    using Offset = std::remove_cv_t<std::remove_pointer_t<decltype(offsets)>>;
    auto kernel  = parse_segments<decltype(overflow)::value, Offset>;
    kernel<<<grid_for(n_segs, kParseThreads),
             kParseThreads,
             parse_shared_bytes(kernel, table),
             s>>>(chars,
                  n_chars,
                  seg_bytes,
                  offsets,
                  seg_first.data(),
                  n_segs,
                  table,
                  cap,
                  seg_counts.data(),
                  seg_start.data(),
                  out,
                  overflowed.data());
  };
  auto parse_pass = [&](auto overflow, std::uint16_t* out) {
    if (aligned) {
      with_offsets(string_offsets, [&](auto const* offsets) { parse(overflow, offsets, out); });
    } else {
      parse(overflow, static_cast<std::int32_t const*>(nullptr), out);
    }
  };
  parse_pass(std::false_type{}, scratch.data());
  check_cuda(cudaGetLastError(), "parse pass");
  thrust::exclusive_scan(
    rmm::exec_policy_nosync(stream, mr), seg_counts.begin(), seg_counts.end(), seg_start.begin());
  std::uint32_t counts[2] = {0, 0};  // total codes, any segment overflowed
  check_cuda(
    cudaMemcpyAsync(
      &counts[0], seg_start.data() + n_segs, sizeof(std::uint32_t), cudaMemcpyDeviceToHost, s),
    "read code count");
  check_cuda(cudaMemcpyAsync(
               &counts[1], overflowed.data(), sizeof(std::uint32_t), cudaMemcpyDeviceToHost, s),
             "read overflow");
  stream.sync();
  std::uint64_t const n_codes = counts[0];
  rmm::device_uvector<std::uint16_t> codes(n_codes, stream, mr);
  compact_codes<<<grid_for(n_segs * 32, 256), 256, 0, s>>>(
    scratch.data(), cap, seg_counts.data(), seg_start.data(), n_segs, codes.data());
  if (counts[1]) { parse_pass(std::true_type{}, codes.data()); }
  check_cuda(cudaGetLastError(), "compact");
  scratch.release();

  // ── Prune, bitpack, sidecar and dictionary planes ─────────────────────────────────────
  rmm::device_uvector<std::uint32_t> used(source_size, stream, mr);
  rmm::device_uvector<std::uint32_t> remap(source_size, stream, mr);
  check_cuda(cudaMemsetAsync(used.data(), 0, source_size * sizeof(std::uint32_t), s), "memset");
  mark_used<<<std::min<unsigned>(grid_for(n_codes, 256), 4096), 256, 0, s>>>(
    codes.data(), n_codes, used.data());
  thrust::exclusive_scan(
    rmm::exec_policy_nosync(stream, mr), used.begin(), used.end(), remap.begin());

  std::uint64_t const n_groups = (n_codes + 31) / 32;
  rmm::device_uvector<std::uint32_t> grp(n_groups + 1, stream, mr);
  rmm::device_uvector<std::uint32_t> grp_prefix(n_groups + 1, stream, mr);
  rmm::device_uvector<unsigned long long> n_long(1, stream, mr);
  check_cuda(cudaMemsetAsync(grp.data() + n_groups, 0, sizeof(std::uint32_t), s), "memset");
  check_cuda(cudaMemsetAsync(n_long.data(), 0, sizeof(unsigned long long), s), "memset");
  group_bytes<<<grid_for(n_groups, 256), 256, 0, s>>>(
    codes.data(),
    n_codes,
    static_cast<std::uint8_t const*>(d_lens.data()),
    grp.data(),
    n_long.data());
  thrust::exclusive_scan(
    rmm::exec_policy_nosync(stream, mr), grp.begin(), grp.end(), grp_prefix.begin());

  std::uint32_t tail[3];  // remap[last], used[last], total bytes
  unsigned long long longs = 0;
  check_cuda(
    cudaMemcpyAsync(&tail[0], remap.data() + source_size - 1, 4, cudaMemcpyDeviceToHost, s),
    "read remap");
  check_cuda(cudaMemcpyAsync(&tail[1], used.data() + source_size - 1, 4, cudaMemcpyDeviceToHost, s),
             "read used");
  check_cuda(cudaMemcpyAsync(&tail[2], grp_prefix.data() + n_groups, 4, cudaMemcpyDeviceToHost, s),
             "read total");
  check_cuda(cudaMemcpyAsync(&longs, n_long.data(), sizeof(longs), cudaMemcpyDeviceToHost, s),
             "read long count");
  stream.sync();
  if (tail[2] != n_chars) {
    throw std::runtime_error("onpair encode: codes decode to " + std::to_string(tail[2]) +
                             " bytes, expected " + std::to_string(n_chars));
  }
  std::uint32_t const d = tail[0] + tail[1];
  std::uint32_t bits    = 1;
  while ((std::uint64_t{1} << bits) < d)
    ++bits;

  // The paper's rule: split reads unless over 60% of the codes are longer than 8 bytes.
  rep->w            = n_codes && static_cast<double>(longs) / n_codes > 0.6 ? 16 : 8;
  rep->dict_buf     = rmm::device_buffer(std::size_t{d} * 16, stream, mr);
  rep->lens_nib_buf = rmm::device_buffer((d + 1) / 2, stream, mr);
  rmm::device_uvector<std::uint8_t> new_lens(d, stream, mr);
  build_planes<<<grid_for(source_size, 256), 256, 0, s>>>(
    used.data(),
    remap.data(),
    source_size,
    static_cast<std::uint8_t const*>(d_dict16.data()),
    static_cast<std::uint8_t const*>(d_lens.data()),
    d,
    rep->w,
    static_cast<std::uint8_t*>(rep->dict_buf.data()),
    new_lens.data());
  pack_nibbles<<<grid_for((d + 1) / 2, 256), 256, 0, s>>>(
    new_lens.data(), d, static_cast<std::uint8_t*>(rep->lens_nib_buf.data()));

  std::uint64_t const n_batches = (n_codes + kOnPairBatchCodes - 1) / kOnPairBatchCodes;
  std::uint64_t const n_words   = n_batches * kOnPairBatchCodes * bits / 32 + 1;
  rep->codes_buf                = rmm::device_buffer(n_words * sizeof(std::uint32_t), stream, mr);
  pack_codes<<<grid_for(n_words, 256), 256, 0, s>>>(
    codes.data(),
    n_codes,
    remap.data(),
    bits,
    n_words,
    static_cast<std::uint32_t*>(rep->codes_buf.data()));
  rep->sidecar_buf = rmm::device_buffer((n_batches + 1) * sizeof(std::uint32_t), stream, mr);
  build_sidecar<<<grid_for(n_batches + 1, 256), 256, 0, s>>>(
    grp_prefix.data(), n_groups, n_batches, static_cast<std::uint32_t*>(rep->sidecar_buf.data()));
  check_cuda(cudaGetLastError(), "finish kernels");
  stream.sync();

  rep->dict_size = d;
  rep->n_tokens  = n_codes;
  rep->code_bits = bits;
  return rep;
}

}  // namespace simpatico
