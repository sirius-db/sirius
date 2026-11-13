// SPDX-License-Identifier: Apache-2.0
// ANS block-range decoder: decodes only the nvCOMP ANS blocks of a frame (see
// operators/nvcomp_batched_codec.hpp for the frame layout) that cover a requested range of
// uncompressed bytes.
//
// The covering blocks are decoded in batches, each one nvCOMP call. A batch holds at most
// kBlocksPerBatch blocks, at most kScratchBytes of blocks that cannot decode straight into the
// destination, and, when the compressed bytes are not usable in place, at most about
// kStagedBytes of them. What the state keeps between calls is therefore bounded, whatever the
// range.
//
//   frame layout (little-endian):
//     u64  num_blocks
//     u64  block_size                 // uncompressed bytes per block (last is smaller)
//     u64  comp_sizes[num_blocks]     // compressed bytes per block
//     <pad to 256>
//     compressed block data, each block 256-byte (kFrameAlign) aligned

#include "codegen/util/cuda_check.hpp"
#include "decode/chunk/chunk_decoders.hpp"
#include "operators/nvcomp_batched_codec.hpp"
#include "util/nvcomp_payload_format.hpp"

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <nvcomp/ans.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace simpatico::chunk {

namespace {

constexpr std::uint64_t kFrameHeaderBytes = 16;

inline std::uint64_t align_up(std::uint64_t x, std::uint64_t a) { return (x + a - 1) / a * a; }

inline bool is_aligned(void const* p, std::size_t alignment)
{
  return alignment <= 1 || reinterpret_cast<std::uintptr_t>(p) % alignment == 0;
}

/// Most blocks one batch decodes.
constexpr std::size_t kBlocksPerBatch = 64;
/// Most scratch one batch uses for blocks that do not decode straight into the destination (at
/// least one block).
constexpr std::uint64_t kScratchBytes = std::uint64_t{1} << 20;
/// Most compressed bytes one batch copies or reads into the state's staging buffer (at least one
/// block).
constexpr std::uint64_t kStagedBytes = std::uint64_t{1} << 20;

/// The first `count` entries of the four nvCOMP input tables (compressed pointers, compressed
/// sizes, uncompressed sizes, uncompressed pointers) of one batch. Passed to the kernel by
/// value: a launch copies its arguments before it returns, so the tables reach the device
/// without a host buffer that would have to outlive an asynchronous copy.
struct table_batch {
  std::uint64_t count;
  std::uint64_t comp_ptrs[kBlocksPerBatch];
  std::uint64_t comp_bytes[kBlocksPerBatch];
  std::uint64_t uncomp_bytes[kBlocksPerBatch];
  std::uint64_t uncomp_ptrs[kBlocksPerBatch];
};

/// Writes a batch into the tables: `stride` u64 each, laid out back to back.
__global__ void write_tables_kernel(table_batch batch,
                                    std::uint64_t* __restrict__ tables,
                                    std::uint64_t stride)
{
  const std::uint64_t i = threadIdx.x;
  if (i >= batch.count) { return; }
  tables[i]              = batch.comp_ptrs[i];
  tables[stride + i]     = batch.comp_bytes[i];
  tables[2 * stride + i] = batch.uncomp_bytes[i];
  tables[3 * stride + i] = batch.uncomp_ptrs[i];
}

inline void check(nvcompStatus_t s, char const* what)
{
  if (s != nvcompSuccess) {
    throw std::runtime_error(std::string("chunk ANS decode: ") + what + " failed (status " +
                             std::to_string(static_cast<int>(s)) + ")");
  }
}

// Grow-only device scratch owned by the state: reused across calls so the stream-ordered work
// of one call never outlives its buffers.
void* ensure(std::optional<rmm::device_buffer>& buffer,
             std::size_t bytes,
             rmm::cuda_stream_view stream,
             rmm::device_async_resource_ref mr)
{
  if (!buffer.has_value() || buffer->size() < bytes) {
    buffer.reset();
    buffer.emplace(bytes, stream, mr);
  }
  return buffer->data();
}

}  // namespace

struct ans_state {
  buffer_ref ans{};
  std::uint64_t frame_bytes{0};         ///< size of the leaf buffer holding the frame
  std::uint64_t uncompressed_bytes{0};  ///< decoded size of the whole leaf

  // Parsed frame header; filled by the first decode that needs it.
  bool header_parsed{false};
  std::uint64_t block_bytes{0};              ///< uncompressed bytes per block (last is shorter)
  std::vector<std::uint64_t> block_offsets;  ///< frame offset of each block's compressed bytes
  std::vector<std::uint64_t> block_sizes;    ///< compressed bytes of each block

  nvcompAlignmentRequirements_t alignments{1, 1, 1};

  // Grow-only, each sized by one batch: see kBlocksPerBatch, kScratchBytes and kStagedBytes.
  std::optional<rmm::device_buffer> tables;   ///< per-block pointer / size tables
  std::optional<rmm::device_buffer> temp;     ///< nvCOMP temporary storage
  std::optional<rmm::device_buffer> scratch;  ///< blocks not decoded straight into dst
  std::optional<rmm::device_buffer> staged;   ///< compressed bytes not usable in place
};

namespace {

void delete_ans_state(ans_state* state) { delete state; }

void parse_frame_header(leaf_access& leaves, ans_state& state, rmm::cuda_stream_view stream)
{
  auto const corrupt = [](char const* what) {
    return std::runtime_error(std::string("chunk ANS decode: corrupt frame (") + what + ")");
  };
  if (state.frame_bytes < kFrameHeaderBytes) { throw corrupt("shorter than its header"); }

  std::array<std::uint64_t, 2> head{};
  leaves.host_bytes(state.ans.leaf, state.ans.buffer, 0, kFrameHeaderBytes, head.data(), stream);
  std::uint64_t const num_blocks  = head[0];
  std::uint64_t const block_bytes = head[1];
  if (block_bytes == 0 || block_bytes > nvcompANSDecompressionMaxAllowedChunkSize) {
    throw corrupt("block size");
  }
  if (num_blocks != (state.uncompressed_bytes + block_bytes - 1) / block_bytes) {
    throw corrupt("block count does not match the uncompressed size");
  }
  if (num_blocks > (state.frame_bytes - kFrameHeaderBytes) / sizeof(std::uint64_t)) {
    throw corrupt("block size table past the end");
  }

  std::vector<std::uint64_t> sizes(num_blocks);
  leaves.host_bytes(state.ans.leaf,
                    state.ans.buffer,
                    kFrameHeaderBytes,
                    num_blocks * sizeof(std::uint64_t),
                    sizes.data(),
                    stream);

  std::vector<std::uint64_t> offsets(num_blocks);
  std::uint64_t cursor =
    align_up(kFrameHeaderBytes + num_blocks * sizeof(std::uint64_t), detail::kFrameAlign);
  for (std::uint64_t i = 0; i < num_blocks; ++i) {
    if (cursor > state.frame_bytes || sizes[i] > state.frame_bytes - cursor) {
      throw corrupt("block past the end");
    }
    offsets[i] = cursor;
    cursor     = align_up(cursor + sizes[i], detail::kFrameAlign);
  }

  check(nvcompBatchedANSDecompressGetRequiredAlignments(nvcompBatchedANSDecompressDefaultOpts,
                                                        &state.alignments),
        "get_required_alignments");

  state.block_bytes   = block_bytes;
  state.block_sizes   = std::move(sizes);
  state.block_offsets = std::move(offsets);
  state.header_parsed = true;
}

}  // namespace

std::size_t ans_state_bytes(ans_state const& state) noexcept
{
  std::size_t bytes = 0;
  for (auto const* buffer : {&state.tables, &state.temp, &state.scratch, &state.staged}) {
    if (buffer->has_value()) { bytes += (*buffer)->size(); }
  }
  return bytes;
}

std::unique_ptr<ans_state, void (*)(ans_state*)> make_ans_state(
  hpln_column_description const& column, buffer_ref ans)
{
  if (ans.leaf >= column.leaves.size() || ans.buffer >= column.leaves[ans.leaf].buffers.size()) {
    throw unsupported_chunk_decode("chunk ANS decode: no such leaf buffer");
  }
  auto const& leaf       = column.leaves[ans.leaf];
  auto const* const meta = std::get_if<leaf_meta::ans>(&leaf.meta);
  if (leaf.kind != OpId::Ans || meta == nullptr) {
    throw unsupported_chunk_decode("chunk ANS decode: the leaf is not an ANS leaf");
  }
  if (auto const error = detail::nvcomp_payload_compatibility_error("ans", meta->payload_format);
      !error.empty()) {
    throw unsupported_chunk_decode("chunk ANS decode: " + error);
  }

  std::unique_ptr<ans_state, void (*)(ans_state*)> state{new ans_state{}, &delete_ans_state};
  state->ans                = ans;
  state->frame_bytes        = leaf.buffers[ans.buffer].size;
  state->uncompressed_bytes = meta->uncompressed_size;
  return state;
}

void decode_ans_range(leaf_access& leaves,
                      ans_state& state,
                      std::uint64_t byte_begin,
                      std::uint64_t byte_end,
                      void* dst,
                      rmm::cuda_stream_view stream)
{
  if (byte_begin > byte_end || byte_end > state.uncompressed_bytes) {
    throw std::out_of_range("chunk ANS decode: byte range [" + std::to_string(byte_begin) + ", " +
                            std::to_string(byte_end) + ") outside the " +
                            std::to_string(state.uncompressed_bytes) + " uncompressed bytes");
  }
  if (byte_begin == byte_end) { return; }
  if (!state.header_parsed) { parse_frame_header(leaves, state, stream); }

  cudaStream_t const s            = stream.value();
  auto const mr                   = leaves.mr();
  std::uint64_t const block_bytes = state.block_bytes;
  std::uint64_t const first_block = byte_begin / block_bytes;
  std::uint64_t const last_block  = (byte_end - 1) / block_bytes;
  std::size_t const num_blocks    = static_cast<std::size_t>(last_block - first_block + 1);

  // The covering blocks' compressed bytes: one contiguous span of the frame. They are used in
  // place when they are on the device and aligned for nvCOMP (blocks sit kFrameAlign apart, so
  // they all share the span's misalignment, if any); otherwise each batch's bytes are copied, or
  // read from the payload source, into the state's staging buffer.
  auto const comp_begin = [&](std::size_t i) { return state.block_offsets[first_block + i]; };
  auto const comp_end   = [&](std::size_t i) {
    return state.block_offsets[first_block + i] + state.block_sizes[first_block + i];
  };
  std::uint64_t const span_begin = comp_begin(0);
  std::uint64_t const span_bytes = comp_end(num_blocks - 1) - span_begin;
  auto const* const resident =
    unstaged_leaf_bytes(leaves, state.ans.leaf, state.ans.buffer, span_begin, span_bytes, stream);
  bool const in_place = resident != nullptr && is_aligned(resident, state.alignments.input);

  // A block lying wholly inside the range decodes straight into dst; the (at most two) partial
  // blocks at the ends, and any block whose place in dst is misaligned for nvCOMP, decode into
  // scratch and have their part of the range copied out.
  auto* const out                = static_cast<std::byte*>(dst);
  std::uint64_t const slot_bytes = align_up(block_bytes, detail::kFrameAlign);
  auto const block_begin         = [&](std::size_t i) { return (first_block + i) * block_bytes; };
  auto const block_end           = [&](std::size_t i) {
    return std::min(block_begin(i) + block_bytes, state.uncompressed_bytes);
  };
  auto const direct = [&](std::size_t i) {
    return block_begin(i) >= byte_begin && block_end(i) <= byte_end &&
           is_aligned(out + (block_begin(i) - byte_begin), state.alignments.output);
  };

  // Batches of consecutive blocks, each within the limits above.
  struct block_batch {
    std::size_t first, count, scratch_slots;
  };
  std::vector<block_batch> batches;
  std::size_t const max_scratch_slots =
    static_cast<std::size_t>(std::max<std::uint64_t>(1, kScratchBytes / slot_bytes));
  for (std::size_t i = 0; i < num_blocks; ++i) {
    std::size_t const needs_scratch = direct(i) ? 0 : 1;
    if (!batches.empty()) {
      auto& batch     = batches.back();
      bool const full = batch.count == kBlocksPerBatch ||
                        batch.scratch_slots + needs_scratch > max_scratch_slots ||
                        (!in_place && comp_end(i) - comp_begin(batch.first) > kStagedBytes);
      if (!full) {
        ++batch.count;
        batch.scratch_slots += needs_scratch;
        continue;
      }
    }
    batches.push_back({i, 1, needs_scratch});
  }
  std::size_t max_count         = 0;
  std::size_t max_scratch       = 0;
  std::uint64_t max_staged      = 0;
  auto const batch_staged_bytes = [&](block_batch const& batch) {
    return comp_end(batch.first + batch.count - 1) - comp_begin(batch.first);
  };
  for (auto const& batch : batches) {
    max_count   = std::max(max_count, batch.count);
    max_scratch = std::max(max_scratch, batch.scratch_slots);
    if (!in_place) { max_staged = std::max(max_staged, batch_staged_bytes(batch)); }
  }

  // Device tables: the four inputs, then the actual-size and status outputs (not read back,
  // as in the whole-frame decode), max_count u64 each.
  static_assert(sizeof(void*) == sizeof(std::uint64_t) &&
                sizeof(std::size_t) == sizeof(std::uint64_t));
  std::size_t const table_bytes = max_count * sizeof(std::uint64_t);
  auto* const tables = static_cast<std::byte*>(ensure(state.tables, 6 * table_bytes, stream, mr));
  auto* const scratch =
    max_scratch == 0
      ? nullptr
      : static_cast<std::byte*>(ensure(state.scratch, max_scratch * slot_bytes, stream, mr));
  auto* const staged =
    in_place ? nullptr : static_cast<std::byte*>(ensure(state.staged, max_staged, stream, mr));
  if (!is_aligned(scratch, state.alignments.output) ||
      !is_aligned(staged, state.alignments.input)) {
    throw std::runtime_error("chunk ANS decode: the memory resource's storage is misaligned");
  }

  std::size_t temp_bytes = 0;
  check(SIMPATICO_NVCOMP_GET_TEMP_SIZE(nvcompBatchedANSDecompressGetTempSize,
                                       s,
                                       max_count,
                                       static_cast<std::size_t>(block_bytes),
                                       nvcompBatchedANSDecompressDefaultOpts,
                                       &temp_bytes,
                                       max_count * static_cast<std::size_t>(block_bytes)),
        "decompress_get_temp_size");
  void* const temp = ensure(state.temp, temp_bytes, stream, mr);

  // Everything below is stream ordered, so a batch may reuse the tables, the scratch and the
  // staging buffer of the batch before it.
  struct scratch_copy {
    std::byte const* src;
    std::byte* dst;
    std::uint64_t bytes;
  };
  std::vector<scratch_copy> copies;
  for (auto const& batch : batches) {
    std::uint64_t const batch_begin = comp_begin(batch.first);
    std::byte const* compressed     = staged;
    if (in_place) {
      compressed = resident + (batch_begin - span_begin);
    } else if (resident != nullptr) {
      throw_if_cuda_error(cudaMemcpyAsync(staged,
                                          resident + (batch_begin - span_begin),
                                          batch_staged_bytes(batch),
                                          cudaMemcpyDeviceToDevice,
                                          s),
                          "chunk ANS decode: realign compressed blocks");
    } else {
      read_leaf_bytes(leaves,
                      state.ans.leaf,
                      state.ans.buffer,
                      batch_begin,
                      batch_staged_bytes(batch),
                      staged,
                      stream);
    }

    table_batch table;
    table.count      = batch.count;
    std::size_t slot = 0;
    copies.clear();
    for (std::size_t k = 0; k < batch.count; ++k) {
      std::size_t const i = batch.first + k;
      table.comp_ptrs[k] =
        reinterpret_cast<std::uint64_t>(compressed + (comp_begin(i) - batch_begin));
      table.comp_bytes[k]   = state.block_sizes[first_block + i];
      table.uncomp_bytes[k] = block_end(i) - block_begin(i);
      auto* target          = out + (block_begin(i) - byte_begin);
      if (!direct(i)) {
        target                   = scratch + slot++ * slot_bytes;
        std::uint64_t const from = std::max(block_begin(i), byte_begin);
        std::uint64_t const to   = std::min(block_end(i), byte_end);
        auto const* const src    = target + (from - block_begin(i));
        auto* const copy_to      = out + (from - byte_begin);
        // Whole blocks in neighbouring slots are one run of bytes on both sides.
        if (!copies.empty() && copies.back().src + copies.back().bytes == src &&
            copies.back().dst + copies.back().bytes == copy_to) {
          copies.back().bytes += to - from;
        } else {
          copies.push_back({src, copy_to, to - from});
        }
      }
      table.uncomp_ptrs[k] = reinterpret_cast<std::uint64_t>(target);
    }
    write_tables_kernel<<<1, kBlocksPerBatch, 0, s>>>(
      table, reinterpret_cast<std::uint64_t*>(tables), max_count);
    throw_if_cuda_error(cudaGetLastError(), "chunk ANS decode: block table kernel launch");

    check(nvcompBatchedANSDecompressAsync(
            reinterpret_cast<void const* const*>(tables),
            reinterpret_cast<std::size_t const*>(tables + table_bytes),
            reinterpret_cast<std::size_t const*>(tables + 2 * table_bytes),
            reinterpret_cast<std::size_t*>(tables + 4 * table_bytes),
            batch.count,
            temp,
            temp_bytes,
            reinterpret_cast<void* const*>(tables + 3 * table_bytes),
            nvcompBatchedANSDecompressDefaultOpts,
            reinterpret_cast<nvcompStatus_t*>(tables + 5 * table_bytes),
            s),
          "decompress_async");

    for (auto const& copy : copies) {
      throw_if_cuda_error(
        cudaMemcpyAsync(copy.dst, copy.src, copy.bytes, cudaMemcpyDeviceToDevice, s),
        "chunk ANS decode: copy partial block");
    }
  }
}
}  // namespace simpatico::chunk
