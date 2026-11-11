// SPDX-License-Identifier: Apache-2.0
// GPU decoder for OnPair codes, after the FastPair decoder (arXiv 2609.15034, Algorithm 1):
// lanes take K codes each from the bitpacked code stream, read each token's length (a nibble
// table) and first W bytes (the low dictionary plane), place the tokens with a warp prefix sum,
// queue the remaining bytes of tokens longer than W (hoisting H rounds of those reads), emit to
// shared scratch and drain to global memory with aligned 16-byte stores.
//
// Fixed input: a warp decodes a batch of 32*K codes; the sidecar stores each batch's output
// offset.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace simpatico::onpair_kernels {

constexpr int kThreads            = 256;
constexpr int kWarpsPerBlock      = kThreads / 32;
constexpr std::uint32_t kMaxToken = 16;

template <int W>
struct low_bytes;
template <>
struct low_bytes<8> {
  using type = uint2;
};
template <>
struct low_bytes<16> {
  using type = uint4;
};

/// Device pointers shared by both layouts.
template <int W>
struct dictionary_planes {
  std::uint32_t const* codes;  ///< Bitpacked LSB-first, padded so a warp may over-read.
  std::uint32_t bits;
  typename low_bytes<W>::type const* lo;  ///< First W bytes of every token.
  uint2 const* hi;                        ///< Bytes 8..15 (split reads only).
  std::uint8_t const* lens_nib;           ///< Token length minus one, two per byte.
};

__device__ __forceinline__ std::uint32_t warp_inclusive_scan(std::uint32_t x, std::uint32_t lane)
{
#pragma unroll
  for (std::uint32_t offset = 1; offset < 32; offset <<= 1) {
    std::uint32_t const y = __shfl_up_sync(0xffffffffu, x, offset);
    if (lane >= offset) { x += y; }
  }
  return x;
}

/// The first index in [lo, hi) at which the monotone @p past holds, or hi, searched 32 ways
/// at a time. Every lane of the warp calls it with the same bounds.
template <typename Past>
__device__ __forceinline__ std::uint64_t warp_lower_bound(std::uint64_t lo,
                                                          std::uint64_t hi,
                                                          std::uint32_t lane,
                                                          Past past)
{
  while (hi - lo > 32u) {
    std::uint64_t const stride = (hi - lo + 31u) / 32u;
    std::uint64_t const probe  = lo + lane * stride;
    bool const after           = probe >= hi || past(probe);
    auto const before          = static_cast<std::uint32_t>(__popc(~__ballot_sync(~0u, after)));
    if (before < 32u && lo + before * stride < hi) { hi = lo + before * stride; }
    lo += before > 0u ? (before - 1u) * stride + 1u : 0u;
  }
  bool const after = lo + lane >= hi || past(lo + lane);
  return lo + static_cast<std::uint32_t>(__popc(~__ballot_sync(~0u, after)));
}

/// The decoded length of @p code, from two 4-bit lengths minus one per byte.
__device__ __forceinline__ std::uint32_t token_length(std::uint8_t const* lens_nib,
                                                      std::uint32_t code)
{
  return ((__ldg(lens_nib + (code >> 1)) >> ((code & 1u) * 4u)) & 15u) + 1u;
}

/// Reads codes `first + lane` for a round starting on a word boundary (`first * bits` % 32 == 0):
/// one coalesced load of `bits` words and two shuffles.
__device__ __forceinline__ std::uint32_t read_round_aligned(std::uint32_t const* codes,
                                                            std::uint32_t bits,
                                                            std::uint64_t first,
                                                            std::uint32_t lane)
{
  std::uint64_t const word0 = first * bits / 32u;
  std::uint32_t const word  = lane < bits ? __ldg(codes + word0 + lane) : 0u;
  std::uint32_t const bit   = lane * bits;
  std::uint32_t const w_lo  = __shfl_sync(0xffffffffu, word, bit >> 5);
  std::uint32_t const w_hi  = __shfl_sync(0xffffffffu, word, ((bit >> 5) + 1u) & 31u);
  std::uint32_t const c     = __funnelshift_r(w_lo, w_hi, bit & 31u);
  return bits >= 32u ? c : c & ((1u << bits) - 1u);
}

/// Reads the length and low bytes of one code per lane.
template <int W>
__device__ __forceinline__ void gather_token(dictionary_planes<W> const& p,
                                             std::uint32_t code,
                                             bool valid,
                                             std::uint32_t& len,
                                             typename low_bytes<W>::type& lo)
{
  lo  = __ldg(p.lo + code);
  len = valid ? token_length(p.lens_nib, code) : 0u;
}

/// Emits K tokens per lane at `base + off[k]` of `scratch`, queueing the bytes past W.
template <int K, int W, int H, bool Queue>
__device__ __forceinline__ void emit_tokens(dictionary_planes<W> const& p,
                                            std::uint32_t const (&code)[K],
                                            std::uint32_t const (&len)[K],
                                            std::uint32_t const (&off)[K],
                                            typename low_bytes<W>::type const (&lo)[K],
                                            std::uint8_t* scratch,
                                            std::uint32_t* queue,
                                            std::uint32_t lane)
{
  std::uint32_t n                   = 0;
  [[maybe_unused]] std::uint32_t q0 = 0;
  [[maybe_unused]] uint2 hi0        = make_uint2(0u, 0u);
  if constexpr (Queue) {
    std::uint32_t const below = (1u << lane) - 1u;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      bool const is_long    = len[k] > static_cast<std::uint32_t>(W);
      std::uint32_t const m = __ballot_sync(0xffffffffu, is_long);
      if (is_long) {
        queue[n + __popc(m & below)] = (code[k] << 16) | ((off[k] + W) << 4) | (len[k] - W);
      }
      n += __popc(m);
    }
    __syncwarp();
    if constexpr (H > 0) {
      if (lane < n) {
        q0  = queue[lane];
        hi0 = __ldg(p.hi + (q0 >> 16));
      }
    }
  }

#pragma unroll
  for (int k = 0; k < K; ++k) {
    std::uint32_t const nlo = len[k] < static_cast<std::uint32_t>(W) ? len[k] : W;
    auto const* bytes       = reinterpret_cast<std::uint8_t const*>(&lo[k]);
    std::uint8_t* dst       = scratch + off[k];
#pragma unroll
    for (int j = 0; j < W; ++j) {
      if (static_cast<std::uint32_t>(j) < nlo) { dst[j] = bytes[j]; }
    }
  }

  if constexpr (Queue) {
    std::uint32_t const rounds = (n + 31u) >> 5;
    for (std::uint32_t r = 0; r < rounds; ++r) {
      std::uint32_t const idx = lane + 32u * r;
      if (idx < n) {
        std::uint32_t q;
        uint2 v;
        if (H > 0 && r == 0) {
          q = q0;
          v = hi0;
        } else {
          q = queue[idx];
          v = __ldg(p.hi + (q >> 16));
        }
        std::uint32_t const qlen = q & 15u;
        std::uint8_t* dst        = scratch + ((q >> 4) & 0xfffu);
        auto const* bytes        = reinterpret_cast<std::uint8_t const*>(&v);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (static_cast<std::uint32_t>(j) < qlen) { dst[j] = bytes[j]; }
        }
      }
    }
  }
}

template <int W>
constexpr bool uses_queue()
{
  return W < 16;
}

template <int K>
constexpr std::uint32_t fixed_input_scratch_bytes()
{
  return 32u * K * kMaxToken + 16u;
}

/// Decodes fixed-input batch @p batch into @p scratch and returns its byte count.
template <int K, int W, int H>
__device__ __forceinline__ std::uint32_t decode_batch(dictionary_planes<W> const& p,
                                                      std::uint64_t batch,
                                                      std::uint64_t n_codes,
                                                      std::uint32_t lane,
                                                      std::uint8_t* scratch,
                                                      std::uint32_t* queue)
{
  constexpr std::uint32_t kBatch = 32u * K;
  std::uint32_t code[K], len[K], off[K];
  typename low_bytes<W>::type lo[K];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    std::uint64_t const first = batch * kBatch + 32u * k;
    code[k]                   = read_round_aligned(p.codes, p.bits, first, lane);
    gather_token<W>(p, code[k], first + lane < n_codes, len[k], lo[k]);
  }
  std::uint32_t total = 0;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    std::uint32_t const incl = warp_inclusive_scan(len[k], lane);
    off[k]                   = total + incl - len[k];
    total += __shfl_sync(0xffffffffu, incl, 31);
  }
  emit_tokens<K, W, H, uses_queue<W>()>(p, code, len, off, lo, scratch, queue, lane);
  __syncwarp();
  return total;
}

/// Fixed input: one warp per batch of 32*K codes; `sidecar[b]` is batch b's output offset.
template <int K, int W, int H>
__global__ void __launch_bounds__(kThreads, 4)
  decode_fixed_input(dictionary_planes<W> p,
                     std::uint32_t const* __restrict__ sidecar,
                     std::uint8_t* __restrict__ out,
                     std::uint64_t n_codes,
                     std::uint64_t num_batches)
{
  constexpr std::uint32_t kBatch = 32u * K;
  constexpr bool kQueue          = uses_queue<W>();
  __shared__ __align__(16) std::uint8_t s_scratch[kWarpsPerBlock][fixed_input_scratch_bytes<K>()];
  __shared__ std::uint32_t s_queue[kWarpsPerBlock][kQueue ? kBatch : 1u];

  std::uint32_t const lane  = threadIdx.x & 31u;
  std::uint32_t const warp  = threadIdx.x >> 5;
  std::uint64_t const batch = static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (batch >= num_batches) { return; }

  std::uint32_t const cursor = __ldg(sidecar + batch);
  std::uint32_t const shift  = cursor & 15u;
  std::uint8_t* scratch      = s_scratch[warp] + shift;
  std::uint32_t const total =
    decode_batch<K, W, H>(p, batch, n_codes, lane, scratch, s_queue[warp]);

  std::uint8_t* dst        = out + cursor;
  std::uint32_t const head = min((16u - shift) & 15u, total);
  std::uint32_t const body = (total - head) >> 4;
  std::uint32_t const tail = head + (body << 4);
  if (lane < head) { dst[lane] = scratch[lane]; }
  for (std::uint32_t i = lane; i < body; i += 32u) {
    uint4 const v = *reinterpret_cast<uint4 const*>(scratch + head + 16u * i);
    __stcs(reinterpret_cast<uint4*>(dst + head + 16u * i), v);
  }
  if (lane < total - tail) { dst[tail + lane] = scratch[tail + lane]; }
}

/// Fixed input, ranges only: copies decoded bytes [src[i], src[i] + dst[i + 1] - dst[i]) to
/// out + dst[i] for the n_ranges ranges, which ascend and do not overlap. Each warp takes
/// ranges_per_warp consecutive ranges, finds their batches in the sidecar and decodes each
/// batch it reaches once, so the work follows the ranges rather than the column; a batch
/// shared by two warps' ranges is decoded by both.
template <int K, int W, int H>
__global__ void __launch_bounds__(kThreads, 4)
  decode_fixed_input_ranges(dictionary_planes<W> p,
                            std::uint32_t const* __restrict__ sidecar,
                            std::uint64_t total_bytes,
                            std::uint64_t n_codes,
                            std::uint64_t num_batches,
                            std::int64_t const* __restrict__ src,
                            std::int32_t const* __restrict__ dst,
                            std::uint64_t n_ranges,
                            std::uint64_t ranges_per_warp,
                            std::uint8_t* __restrict__ out)
{
  constexpr std::uint32_t kBatch = 32u * K;
  constexpr bool kQueue          = uses_queue<W>();
  __shared__ __align__(16) std::uint8_t s_scratch[kWarpsPerBlock][fixed_input_scratch_bytes<K>()];
  __shared__ std::uint32_t s_queue[kWarpsPerBlock][kQueue ? kBatch : 1u];

  std::uint32_t const lane = threadIdx.x & 31u;
  std::uint32_t const warp = threadIdx.x >> 5;
  std::uint64_t const first =
    (static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp) * ranges_per_warp;
  if (first >= n_ranges) { return; }
  std::uint64_t const last =
    first + ranges_per_warp < n_ranges ? first + ranges_per_warp : n_ranges;

  // Batch starts ascend strictly (every code decodes to at least one byte); start(num_batches)
  // is the end of the bytes.
  auto start = [&](std::uint64_t b) -> std::uint64_t {
    return b < num_batches ? __ldg(sidecar + b) : total_bytes;
  };

  std::uint8_t* scratch = s_scratch[warp];
  std::uint64_t decoded = num_batches;  // none yet
  std::uint64_t begin = 0, end = 0;     // the decoded batch's bytes
  for (std::uint64_t base = first; base < last; base += 32u) {
    // A lane per range: each round decodes the batch holding the lowest lane's next byte,
    // which, the ranges ascending, is the lowest next byte of all.
    std::uint64_t const i = base + lane;
    bool const valid      = i < last;
    auto x                = valid ? static_cast<std::uint64_t>(src[i]) : 0u;
    std::int32_t const d  = valid ? dst[i] : 0;
    std::uint64_t const e = x + static_cast<std::uint32_t>(valid ? dst[i + 1] - d : 0);
    std::uint8_t* const o = out + d - static_cast<std::int64_t>(x);
    for (std::uint32_t pending = __ballot_sync(~0u, x < e); pending != 0u;
         pending               = __ballot_sync(~0u, x < e)) {
      std::uint64_t const next = __shfl_sync(~0u, x, __ffs(pending) - 1u);
      if (decoded == num_batches || next >= end) {
        // Most next batches are within 32 of the last one.
        std::uint64_t const from = decoded == num_batches ? 0 : decoded + 1;
        std::uint64_t const near = from + lane;
        std::uint32_t const hits = __ballot_sync(~0u, near > num_batches || start(near) > next);
        std::uint64_t const batch =
          (from > 0 && hits != 0u
             ? from + __ffs(hits) - 1u
             : warp_lower_bound(from, num_batches, lane, [&](auto b) { return start(b) > next; })) -
          1u;
        __syncwarp();
        decode_batch<K, W, H>(p, batch, n_codes, lane, scratch, s_queue[warp]);
        decoded = batch;
        begin   = __ldg(sidecar + batch);
        end     = start(batch + 1);
      }
      std::uint64_t const stop = x < end ? (e < end ? e : end) : x;
      bool const wide          = stop - x > 32u;
      if (!wide) {
        for (std::uint64_t y = x; y < stop; ++y)
          o[y] = scratch[y - begin];
      }
      for (std::uint32_t w = __ballot_sync(~0u, wide); w != 0u; w &= w - 1u) {
        std::uint32_t const t  = __ffs(w) - 1u;
        std::uint64_t const ws = __shfl_sync(~0u, x, t);
        std::uint64_t const we = __shfl_sync(~0u, stop, t);
        auto* const wo         = reinterpret_cast<std::uint8_t*>(
          __shfl_sync(~0u, reinterpret_cast<unsigned long long>(o), t));
        for (std::uint64_t y = ws + lane; y < we; y += 32u)
          wo[y] = scratch[y - begin];
      }
      x = stop;
    }
  }
}

/// Literals to compare string-aligned codes against: literal l is lit_bytes[l] bytes long
/// and parses to the codes lit_codes[lit_first[l], lit_first[l + 1]).
struct code_literals {
  std::uint32_t const* lit_bytes;
  std::uint32_t const* lit_first;
  std::uint16_t const* lit_codes;
  std::uint32_t n;
};

/// String-aligned codes, equality without decoding: out[r] = 1 iff row r, the bytes
/// [offsets[r], offsets[r + 1]), equals one of the literals. A string-aligned parse gives
/// equal strings equal codes and starts every row on a code, so a row as long as a literal
/// whose codes from its start are the literal's is that literal. Each warp takes
/// batches_per_warp consecutive batches of 32*K codes: it finds a batch's code byte offsets
/// from the code lengths alone, then the rows starting in the batch (the last batch also
/// those starting at the end of the bytes) a lane each, continuing from the previous
/// batch's rows; only a row as long as a literal reads codes.
template <int K, typename Offset>
__global__ void __launch_bounds__(kThreads)
  equal_fixed_input(std::uint32_t const* __restrict__ codes,
                    std::uint32_t bits,
                    std::uint8_t const* __restrict__ lens_nib,
                    std::uint32_t const* __restrict__ sidecar,
                    std::uint64_t total_bytes,
                    std::uint64_t n_codes,
                    std::uint64_t num_batches,
                    std::uint64_t batches_per_warp,
                    Offset const* __restrict__ offsets,
                    std::uint64_t n_rows,
                    code_literals list,
                    std::uint8_t* __restrict__ out)
{
  constexpr std::uint32_t kBatch = 32u * K;
  __shared__ std::uint16_t s_off[kWarpsPerBlock][kBatch];

  std::uint32_t const lane = threadIdx.x & 31u;
  std::uint32_t const warp = threadIdx.x >> 5;
  std::uint64_t const first_batch =
    (static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp) * batches_per_warp;
  if (first_batch >= num_batches) { return; }
  std::uint64_t const last_batch =
    first_batch + batches_per_warp < num_batches ? first_batch + batches_per_warp : num_batches;

  // The first row starting at or past the first batch.
  std::uint64_t const begin0 = __ldg(sidecar + first_batch);
  std::uint64_t cursor       = warp_lower_bound(
    0, n_rows, lane, [&](auto r) { return static_cast<std::uint64_t>(offsets[r]) >= begin0; });

  auto code_at = [&](std::uint64_t c) {
    std::uint64_t const bit = c * bits;
    std::uint32_t const w   = __funnelshift_r(__ldg(codes + (bit >> 5)),
                                            __ldg(codes + (bit >> 5) + 1),
                                            static_cast<std::uint32_t>(bit & 31u));
    return w & ((1u << bits) - 1u);
  };
  for (std::uint64_t batch = first_batch; batch < last_batch; ++batch) {
    std::uint64_t const begin = __ldg(sidecar + batch);
    std::uint64_t const stop =
      batch + 1 < num_batches ? __ldg(sidecar + batch + 1) : total_bytes + 1;
    std::uint32_t total = 0;
    __syncwarp();
#pragma unroll
    for (int k = 0; k < K; ++k) {
      std::uint64_t const first   = batch * kBatch + 32u * k;
      std::uint32_t const code    = read_round_aligned(codes, bits, first, lane);
      std::uint32_t const len     = first + lane < n_codes ? token_length(lens_nib, code) : 0u;
      std::uint32_t const incl    = warp_inclusive_scan(len, lane);
      s_off[warp][32u * k + lane] = static_cast<std::uint16_t>(total + incl - len);
      total += __shfl_sync(0xffffffffu, incl, 31);
    }
    __syncwarp();
    // The rows ascend, so those starting in the batch are a prefix of each round of 32.
    for (;;) {
      std::uint64_t const r = cursor + lane;
      std::uint64_t const s = r < n_rows ? static_cast<std::uint64_t>(offsets[r]) : stop;
      bool const in         = s < stop;
      if (in) {
        auto const len = static_cast<std::uint32_t>(static_cast<std::uint64_t>(offsets[r + 1]) - s);
        bool match     = false;
        for (std::uint32_t l = 0; l < list.n && !match; ++l) {
          if (__ldg(list.lit_bytes + l) != len) { continue; }
          if (len == 0) {
            match = true;
            break;
          }
          // The code starting at s: the first whose offset reaches it.
          auto const target = static_cast<std::uint16_t>(s - begin);
          std::uint32_t a = 0, b = kBatch;
          while (a < b) {
            std::uint32_t const mid = (a + b) / 2;
            if (s_off[warp][mid] < target) {
              a = mid + 1;
            } else {
              b = mid;
            }
          }
          std::uint64_t const c     = batch * kBatch + a;
          std::uint32_t const first = __ldg(list.lit_first + l);
          std::uint32_t const m     = __ldg(list.lit_first + l + 1) - first;
          match                     = c + m <= n_codes;
          for (std::uint32_t j = 0; j < m && match; ++j)
            match = code_at(c + j) == __ldg(list.lit_codes + first + j);
        }
        out[r] = match;
      }
      auto const taken = static_cast<std::uint32_t>(__popc(__ballot_sync(~0u, in)));
      cursor += taken;
      if (taken < 32u) { break; }
    }
  }
}

}  // namespace simpatico::onpair_kernels
