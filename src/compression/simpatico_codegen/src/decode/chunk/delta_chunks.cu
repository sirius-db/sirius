// SPDX-License-Identifier: Apache-2.0
//
// Delta chunk decoding: the shared decode_delta_chunks primitive and the decoder of a
// `delta -> bitpack` or `delta -> rle -> bitpack` root plan.
//
// Layout of one delta subtree (see the Delta / Rle sections of encode/jit/renderer.cpp and
// decode/jit/renderer.cpp). Every node is chunked like the delta's input: chunk c of each node
// belongs to elements [c * 1024, c * 1024 + len) of the input.
//   delta.delta_first        one element per chunk: the chunk's first value, so a chunk decodes
//                            without its predecessors.
//   differences              len - 1 differences per chunk (0 for a chunk of one element);
//                            value[0] = delta_first, value[i + 1] = value[i] + difference[i],
//                            in the element's unsigned (wrapping) arithmetic.
//   differences -> bitpack   chunk c of the bitpack node holds the chunk's len - 1 differences.
//   differences -> rle       rle_runs_offsets: int32, one per chunk plus one; chunk c owns runs
//                            [offsets[c], offsets[c + 1]) (runs never span chunks). Chunk c of
//                            the `runs` bitpack node holds the chunk's run lengths and chunk c of
//                            the `values` node its run values, one per run; the lengths add up to
//                            len - 1.

#include "decode/chunk/chunk_decoders.hpp"

#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace simpatico::chunk {

namespace {

/// Per-chunk decode parameters, uploaded for the chunks of one call.
struct chunk_meta {
  std::uint64_t first;  ///< delta_first, zero-extended from the element width
  std::int32_t n;       ///< differences in the chunk (no rle) or runs in the chunk (rle)
  std::int32_t cap;     ///< most elements the chunk can hold (1024 unless it is the last one)
};

constexpr int kChunkThreads = static_cast<int>(kChunkRows);
/// Chunks decoded per pass; bounds the int64 scratch of the unpacked children.
constexpr std::int64_t kMaxChunksPerPass = 2048;

/// Inclusive prefix sum of src[0, 1024) across the block (one thread per element), using tmp as
/// the other buffer. src must be visible to the block on entry; the sums are in src, and visible,
/// on return.
__device__ __forceinline__ void block_inclusive_sum(std::uint64_t* src, std::uint64_t* tmp, int tid)
{
  // log2(1024) = 10 passes: an even number, so the result lands back in src.
  for (int offset = 1; offset < kChunkThreads; offset <<= 1) {
    tmp[tid] = src[tid] + (tid >= offset ? src[tid - offset] : 0);
    __syncthreads();
    std::uint64_t* const swap = src;
    src                       = tmp;
    tmp                       = swap;
  }
}

/// One block per chunk, one thread per element. `values` (and `runs`) hold one 1024-slot block
/// per chunk: the chunk's differences, or with `runs != nullptr` its run values and run lengths.
/// Element i of chunk c is position pos_base + c * 1024 + i; positions in [lo, hi) are written
/// to out[position - lo]. Src is the delta's element type (its width wraps the sum and its
/// signedness extends it into Out).
template <typename Src, typename Out>
__global__ void delta_kernel(const chunk_meta* __restrict__ meta,
                             const std::int64_t* __restrict__ values,
                             const std::int64_t* __restrict__ runs,
                             std::int64_t pos_base,
                             std::int64_t lo,
                             std::int64_t hi,
                             Out* __restrict__ out)
{
  __shared__ std::uint64_t sa[kChunkThreads];
  __shared__ std::uint64_t sb[kChunkThreads];

  const int tid               = static_cast<int>(threadIdx.x);
  const std::int64_t c        = blockIdx.x;
  const std::int64_t chunk_at = pos_base + c * kChunkRows;
  // The same for every thread of the block, so the barriers below stay block-wide.
  if (chunk_at >= hi || chunk_at + kChunkRows <= lo) { return; }
  const chunk_meta m           = meta[c];
  const std::int64_t* const in = values + c * kChunkRows;

  int ndiff = 0;
  if (runs != nullptr) {
    const int nruns = m.n;
    sa[tid]         = tid < nruns ? static_cast<std::uint64_t>(runs[c * kChunkRows + tid]) : 0;
    __syncthreads();
    block_inclusive_sum(sa, sb, tid);  // sa[r] = differences up to and including run r
    if (nruns > 0) {
      const std::uint64_t total = sa[nruns - 1];
      ndiff = total < kChunkThreads ? static_cast<int>(total) : kChunkThreads - 1;
    }
    std::uint64_t diff = 0;
    const int j        = tid - 1;
    if (j >= 0 && j < ndiff) {
      // The run holding difference j: the first whose end is past j (j < sa[nruns - 1]).
      int first = 0;
      int last  = nruns - 1;
      while (first < last) {
        const int mid = (first + last) >> 1;
        if (sa[mid] > static_cast<std::uint64_t>(j)) {
          last = mid;
        } else {
          first = mid + 1;
        }
      }
      diff = static_cast<std::uint64_t>(in[first]);
    }
    sb[tid] = diff;
  } else {
    ndiff   = m.n;
    sb[tid] = (tid >= 1 && tid - 1 < ndiff) ? static_cast<std::uint64_t>(in[tid - 1]) : 0;
  }
  __syncthreads();
  block_inclusive_sum(sb, sa, tid);  // sb[i] = sum of the differences before element i

  const int len          = ndiff + 1 < m.cap ? ndiff + 1 : m.cap;
  const std::int64_t pos = chunk_at + tid;
  if (tid < len && pos >= lo && pos < hi) {
    out[pos - lo] = static_cast<Out>(static_cast<Src>(m.first + sb[tid]));
  }
}

void cuda_try(cudaError_t status, char const* what)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string{"delta chunk decode: "} + what +
                             " failed: " + cudaGetErrorString(status));
  }
}

/// Buffer `name` of the own-rep leaf of node `node`.
std::optional<buffer_ref> find_own_buffer(hpln_column_description const& column,
                                          std::uint32_t node,
                                          std::string_view name)
{
  for (std::size_t l = 0; l < column.leaves.size(); ++l) {
    auto const& leaf = column.leaves[l];
    if (leaf.node_index != node || leaf.slot != kSelfRepSlot) { continue; }
    for (std::size_t b = 0; b < leaf.buffers.size(); ++b) {
      if (leaf.buffers[b].name == name) { return buffer_ref{l, b}; }
    }
  }
  return std::nullopt;
}

std::unique_ptr<bitpack_state, void (*)(bitpack_state*)> no_bitpack_state()
{
  return {nullptr, [](bitpack_state*) {}};
}

}  // namespace

struct delta_state {
  buffer_ref first;         ///< delta.delta_first
  buffer_ref runs_offsets;  ///< rle.rle_runs_offsets (with_rle)
  buffer_ref diff_counts;   ///< chunk_count of the differences bitpack node (!with_rle)
  bool with_rle{false};
  bool is_unsigned{false};      ///< signedness of the delta's element type
  std::size_t elem_size{0};     ///< bytes per delta_first element; 0 when there are no chunks
  std::int64_t num_chunks{0};   ///< chunks of the delta node
  std::int64_t num_values{-1};  ///< elements of the delta's input when known (a root delta)
  // Bitpack children: the differences (!with_rle), or the run values and run lengths.
  std::unique_ptr<bitpack_state, void (*)(bitpack_state*)> values{no_bitpack_state()};
  std::unique_ptr<bitpack_state, void (*)(bitpack_state*)> runs{no_bitpack_state()};
  // Per-call transients, kept (grow-only) so a call allocates nothing in the steady state.
  std::vector<std::uint8_t> host_firsts;
  std::vector<std::int32_t> host_counts;
  std::vector<chunk_meta> host_meta;
  rmm::device_buffer device_meta;
  rmm::device_buffer values_scratch;
  rmm::device_buffer runs_scratch;

  [[nodiscard]] std::size_t device_bytes() const noexcept
  {
    return device_meta.size() + values_scratch.size() + runs_scratch.size();
  }

  /// Reads and uploads the per-chunk parameters of chunks [chunk_begin, chunk_end).
  const chunk_meta* stage_meta(leaf_access& leaves,
                               std::int64_t chunk_begin,
                               std::int64_t chunk_end,
                               rmm::cuda_stream_view stream)
  {
    auto const n = static_cast<std::size_t>(chunk_end - chunk_begin);
    host_firsts.resize(n * elem_size);
    leaves.host_bytes(first.leaf,
                      first.buffer,
                      static_cast<std::uint64_t>(chunk_begin) * elem_size,
                      n * elem_size,
                      host_firsts.data(),
                      stream);
    host_meta.assign(n, chunk_meta{});
    if (with_rle) {
      host_counts.resize(n + 1);
      leaves.host_bytes(runs_offsets.leaf,
                        runs_offsets.buffer,
                        static_cast<std::uint64_t>(chunk_begin) * sizeof(std::int32_t),
                        (n + 1) * sizeof(std::int32_t),
                        host_counts.data(),
                        stream);
      for (std::size_t i = 0; i < n; ++i) {
        host_meta[i].n = host_counts[i + 1] - host_counts[i];
      }
    } else {
      host_counts.resize(n);
      leaves.host_bytes(diff_counts.leaf,
                        diff_counts.buffer,
                        static_cast<std::uint64_t>(chunk_begin) * sizeof(std::int32_t),
                        n * sizeof(std::int32_t),
                        host_counts.data(),
                        stream);
      for (std::size_t i = 0; i < n; ++i) {
        host_meta[i].n = host_counts[i];
      }
    }
    for (std::size_t i = 0; i < n; ++i) {
      auto& m = host_meta[i];
      // A chunk of len elements has len - 1 differences, in at most as many runs.
      if (m.n < 0 || m.n >= kChunkRows) {
        throw std::runtime_error("delta chunk decode: corrupt chunk metadata");
      }
      std::memcpy(&m.first, host_firsts.data() + i * elem_size, elem_size);  // little-endian
      m.cap = static_cast<std::int32_t>(kChunkRows);
      if (num_values >= 0) {
        auto const before = (chunk_begin + static_cast<std::int64_t>(i)) * kChunkRows;
        m.cap = static_cast<std::int32_t>(std::clamp<std::int64_t>(num_values - before, 0, m.cap));
      }
    }
    auto const meta_bytes = n * sizeof(chunk_meta);
    if (device_meta.size() < meta_bytes) {
      device_meta = rmm::device_buffer(meta_bytes, stream, leaves.mr());
    }
    // host_meta stays alive (and unchanged) until the next call on this state.
    cuda_try(
      cudaMemcpyAsync(
        device_meta.data(), host_meta.data(), meta_bytes, cudaMemcpyHostToDevice, stream.value()),
      "chunk metadata upload");
    return static_cast<const chunk_meta*>(device_meta.data());
  }

  /// Decodes chunks [chunk_begin, chunk_end) (non-empty, within the node) and writes the
  /// elements at positions [lo, hi), counted from the first element of chunk_begin, to
  /// out[position - lo].
  template <typename Src, typename Out>
  void decode(leaf_access& leaves,
              std::int64_t chunk_begin,
              std::int64_t chunk_end,
              std::int64_t lo,
              std::int64_t hi,
              Out* out,
              rmm::cuda_stream_view stream)
  {
    auto const* const meta = stage_meta(leaves, chunk_begin, chunk_end, stream);
    auto const pass_chunks = std::min(chunk_end - chunk_begin, kMaxChunksPerPass);
    auto const scratch = static_cast<std::size_t>(pass_chunks * kChunkRows) * sizeof(std::int64_t);
    if (values_scratch.size() < scratch) {
      values_scratch = rmm::device_buffer(scratch, stream, leaves.mr());
    }
    if (with_rle && runs_scratch.size() < scratch) {
      runs_scratch = rmm::device_buffer(scratch, stream, leaves.mr());
    }
    auto* const values_out = static_cast<std::int64_t*>(values_scratch.data());
    auto* const runs_out   = with_rle ? static_cast<std::int64_t*>(runs_scratch.data()) : nullptr;

    for (auto begin = chunk_begin; begin < chunk_end; begin += kMaxChunksPerPass) {
      auto const end = std::min(chunk_end, begin + kMaxChunksPerPass);
      unpack_bitpack_chunks(leaves, *values, begin, end, values_out, stream);
      if (with_rle) { unpack_bitpack_chunks(leaves, *runs, begin, end, runs_out, stream); }
      delta_kernel<Src, Out>
        <<<static_cast<unsigned>(end - begin), kChunkThreads, 0, stream.value()>>>(
          meta + (begin - chunk_begin),
          values_out,
          runs_out,
          (begin - chunk_begin) * kChunkRows,
          lo,
          hi,
          out);
      cuda_try(cudaGetLastError(), "delta kernel launch");
    }
  }

  void check_range(std::int64_t chunk_begin, std::int64_t chunk_end) const
  {
    if (chunk_begin < 0 || chunk_end < chunk_begin || chunk_end > num_chunks) {
      throw std::invalid_argument("delta chunk decode: chunks [" + std::to_string(chunk_begin) +
                                  ", " + std::to_string(chunk_end) +
                                  ") are not within the node's " + std::to_string(num_chunks));
    }
  }
};

namespace {

void delete_delta_state(delta_state* state) { delete state; }

/// Decoder of a `delta` root: values are written in the column's storage type, which is the
/// delta's element type (chrono and decimal columns are encoded as their same-width integers).
class delta_decoder final : public chunk_decoder {
 public:
  explicit delta_decoder(std::unique_ptr<delta_state, void (*)(delta_state*)> state)
    : _state(std::move(state))
  {
  }

  std::int64_t decode(leaf_access& leaves,
                      chunk_span const& span,
                      column_write_target const& dst,
                      rmm::cuda_stream_view stream) override
  {
    if (span.take <= 0) { return 0; }
    // Only the chunks the rows touch, which may be fewer than the span offers.
    auto const chunk_begin = span.chunk_begin + span.skip_head / kChunkRows;
    auto const skip        = span.skip_head % kChunkRows;
    auto const chunk_end =
      std::min(span.chunk_end, chunk_begin + (skip + span.take + kChunkRows - 1) / kChunkRows);
    _state->check_range(chunk_begin, chunk_end);
    if (chunk_end == chunk_begin) {
      throw std::invalid_argument("delta chunk decode: the span holds no chunk for its rows");
    }
    auto const width = static_cast<std::size_t>(cudf::size_of(leaves.column().dtype));
    if (_state->elem_size != width) {
      throw unsupported_chunk_decode("delta chunk decode: the delta element is " +
                                     std::to_string(_state->elem_size) + " bytes, the column's " +
                                     std::to_string(width));
    }
    auto* const out =
      static_cast<std::byte*>(dst.data) + static_cast<std::size_t>(dst.dst_row_offset) * width;
    auto run = [&]<typename T>(T*) {
      _state->decode<T, T>(
        leaves, chunk_begin, chunk_end, skip, skip + span.take, reinterpret_cast<T*>(out), stream);
    };
    switch (width) {
      case 1: run(static_cast<std::uint8_t*>(nullptr)); break;
      case 2: run(static_cast<std::uint16_t*>(nullptr)); break;
      case 4: run(static_cast<std::uint32_t*>(nullptr)); break;
      default: run(static_cast<std::uint64_t*>(nullptr)); break;
    }
    return span.take;
  }

  [[nodiscard]] std::size_t state_bytes() const noexcept override { return _state->device_bytes(); }

 private:
  std::unique_ptr<delta_state, void (*)(delta_state*)> _state;
};

}  // namespace

std::unique_ptr<delta_state, void (*)(delta_state*)> make_delta_state(
  hpln_column_description const& column, std::uint32_t delta_node)
{
  auto const unsupported = [](char const* why) {
    return unsupported_chunk_decode(std::string{"delta chunk decode: "} + why);
  };
  auto const& nodes = column.nodes;
  if (delta_node >= nodes.size() || nodes[delta_node].op != "delta") {
    throw unsupported("the node is not a delta");
  }
  auto const first = find_own_buffer(column, delta_node, "delta_first");
  auto const child = find_child_node(column, delta_node, "differences");
  if (!first || !child || nodes[delta_node].edges.size() != 1) {
    throw unsupported("the delta node has no delta_first or no differences child");
  }

  std::unique_ptr<delta_state, void (*)(delta_state*)> state{new delta_state{},
                                                             &delete_delta_state};
  state->first = *first;
  if (nodes[*child].op == "rle") {
    auto const offsets     = find_own_buffer(column, *child, "rle_runs_offsets");
    auto const runs_node   = find_child_node(column, *child, "runs");
    auto const values_node = find_child_node(column, *child, "values");
    auto const runs        = runs_node ? find_bitpack(column, *runs_node) : std::nullopt;
    auto const values      = values_node ? find_bitpack(column, *values_node) : std::nullopt;
    if (!offsets || !runs || !values || nodes[*child].edges.size() != 2) {
      throw unsupported("the rle node is not rle_runs_offsets + bitpacked runs and values");
    }
    auto const offsets_bytes = column.leaves[offsets->leaf].buffers[offsets->buffer].size;
    if (offsets_bytes % sizeof(std::int32_t) != 0) {
      throw std::runtime_error("delta chunk decode: rle_runs_offsets is not int32");
    }
    state->with_rle     = true;
    state->runs_offsets = *offsets;
    state->num_chunks =
      offsets_bytes == 0 ? 0 : static_cast<std::int64_t>(offsets_bytes / sizeof(std::int32_t)) - 1;
    state->values = make_bitpack_state(*values);
    state->runs   = make_bitpack_state(*runs);
  } else {
    auto const diffs = find_bitpack(column, *child);
    if (nodes[*child].op != "bitpack" || !diffs) {
      throw unsupported("the differences are neither bitpack nor rle");
    }
    state->diff_counts = diffs->chunk_count;
    state->num_chunks  = static_cast<std::int64_t>(
      column.leaves[diffs->chunk_bits.leaf].buffers[diffs->chunk_bits.buffer].size);
    state->values = make_bitpack_state(*diffs);
  }

  auto const& first_leaf = column.leaves[first->leaf];
  auto const first_bytes = first_leaf.buffers[first->buffer].size;
  if (state->num_chunks != 0) {
    auto const chunks = static_cast<std::uint64_t>(state->num_chunks);
    state->elem_size  = first_bytes / chunks;
    if (first_bytes % chunks != 0) {
      throw std::runtime_error("delta chunk decode: delta_first is not one element per chunk");
    }
    if (state->elem_size != 1 && state->elem_size != 2 && state->elem_size != 4 &&
        state->elem_size != 8) {
      throw unsupported("the delta element is not 1, 2, 4 or 8 bytes");
    }
  }
  state->is_unsigned = cudf::is_unsigned(tag_to_dtype(first_leaf.type_tag));
  // The delta of a root plan covers the column's rows; a nested one (string offsets) has its
  // own length, which the header does not carry.
  if (!nodes.empty() && nodes[0].edges.size() == 1 && nodes[0].edges[0].second == delta_node) {
    state->num_values = column.num_rows;
  }
  return state;
}

void decode_delta_chunks(leaf_access& leaves,
                         delta_state& state,
                         std::int64_t chunk_begin,
                         std::int64_t chunk_end,
                         std::int64_t* out,
                         rmm::cuda_stream_view stream)
{
  state.check_range(chunk_begin, chunk_end);
  if (chunk_end == chunk_begin) { return; }
  // One 1024-value block per chunk; the kernel stops at each chunk's own length.
  auto const hi = (chunk_end - chunk_begin) * kChunkRows;
  auto run      = [&]<typename T>(T*) {
    state.decode<T, std::int64_t>(leaves, chunk_begin, chunk_end, 0, hi, out, stream);
  };
  switch (state.elem_size) {
    case 1:
      state.is_unsigned ? run(static_cast<std::uint8_t*>(nullptr))
                        : run(static_cast<std::int8_t*>(nullptr));
      break;
    case 2:
      state.is_unsigned ? run(static_cast<std::uint16_t*>(nullptr))
                        : run(static_cast<std::int16_t*>(nullptr));
      break;
    case 4:
      state.is_unsigned ? run(static_cast<std::uint32_t*>(nullptr))
                        : run(static_cast<std::int32_t*>(nullptr));
      break;
    default: run(static_cast<std::int64_t*>(nullptr)); break;
  }
}

std::unique_ptr<chunk_decoder> make_delta_decoder(hpln_column_description const& column,
                                                  bool with_rle)
{
  // nodes[0] is the column input; its one edge leads to the delta node.
  if (column.nodes.empty() || column.nodes[0].edges.size() != 1) {
    throw unsupported_chunk_decode("delta chunk decode: the plan has no single root operator");
  }
  if (!cudf::is_fixed_width(column.dtype) ||
      static_cast<std::size_t>(cudf::size_of(column.dtype)) > sizeof(std::int64_t)) {
    throw unsupported_chunk_decode(
      "delta chunk decode: the column is not fixed width of at most 8 bytes");
  }
  auto state = make_delta_state(column, column.nodes[0].edges[0].second);
  if (state->with_rle != with_rle) {
    throw unsupported_chunk_decode(with_rle
                                     ? "delta chunk decode: the differences are not rle"
                                     : "delta chunk decode: the differences are not bitpack");
  }
  // Reject a width mismatch now, so the cursor reports the column unsupported up front.
  if (state->num_chunks != 0 &&
      state->elem_size != static_cast<std::size_t>(cudf::size_of(column.dtype))) {
    throw unsupported_chunk_decode(
      "delta chunk decode: the delta element width differs from the column's");
  }
  return std::make_unique<delta_decoder>(std::move(state));
}
}  // namespace simpatico::chunk
