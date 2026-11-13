// SPDX-License-Identifier: Apache-2.0
//
// str_split chunk decoder: the plan
//   input -> str_split -> offsets, chars
//   str_split.chars -> ans
//   str_split.offsets -> delta -> differences
//   str_split.offsets.differences -> bitpack -> chunk_min, chunk_count, chunk_bits, packed
// keeps a STRING column as its cudf offsets and chars (see operators/str_split_compressor.cu).
//   offsets  num_rows + 1 INT32 or INT64 elements, chunked by 1024 ELEMENTS: element e (the
//            start of row e, the end of row e - 1) is element e % 1024 of chunk e / 1024 of the
//            delta node (see delta_chunks.cu). Rows [r, r + n) therefore need offsets chunks
//            [r / 1024, (r + n) / 1024 + 1): one past the rows' own chunks when the last row
//            ends a chunk.
//   chars    the column's chars as one nvCOMP ANS frame of 64 KiB blocks (see ans_blocks.cu).
//
// Null rows: the column's validity is detached before the encoder sees it, so a null row keeps
// whatever offsets its slot holds and decodes like any other row, as in the whole-table decode.
//
// A call walks the offsets in windows of at most kWindowChunks chunks: decode the window, then
// one device pass finds the rows whose chars still fit the budget, writes their rebased offsets
// and records the cut in a small device record. The host reads that record back (a few bytes)
// to learn the cut, and stops at the first window the budget cuts; when the budget holds all
// the column's chars nothing can be cut and the record is read once, after the last window.
// The chars of all the rows taken are then decoded by one decode_ans_range into the destination.

#include "codegen/util/cuda_check.hpp"
#include "decode/chunk/chunk_decoders.hpp"
#include "util/host_observation.hpp"

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda_runtime.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <variant>

namespace simpatico::chunk {

namespace {

/// Offsets chunks decoded per window; bounds the int64 scratch at 2 MiB.
constexpr std::int64_t kWindowChunks = 256;

/// What a call's device passes report to the host.
struct cut_record {
  std::int64_t base;  ///< source offset of the call's first row
  std::int64_t end;   ///< source offset past the last row the latest window took
  std::int64_t fit;   ///< rows the latest window took
  std::int64_t bad;   ///< nonzero once any window saw corrupt offsets
};

// One window: offsets src[0, count) are the starts of rows [0, count - 1) and the end of the
// last one. Thread i checks offset i, writes destination offset i (source offset i moved so
// that the call's base lands on dst_base) when row i - 1 is within the budget, and the thread
// at the cut records it. Row 0's start is written only when row 0 itself fits. Corrupt offsets
// (negative, decreasing, past the chars) set `bad`; the writes stay within the count offsets.
// The record is written without atomics: every writer of a field stores the same value unless
// the offsets are corrupt, and then `bad` is all the host looks at.
struct fit_offsets_fn {
  std::int64_t const* src;
  std::int64_t count;
  void* dst;
  std::uint8_t dst_width;
  bool first_window;  ///< the call's base is src[0]; otherwise it is in the record
  std::int64_t dst_base;
  std::uint64_t budget;
  std::uint64_t chars_bytes;
  cut_record* record;

  __device__ void operator()(std::int64_t i) const
  {
    auto const value = src[i];
    if ((i == 0 && value < 0) || (i + 1 < count && src[i + 1] < value) ||
        (i + 1 == count && static_cast<std::uint64_t>(value) > chars_bytes)) {
      record->bad = 1;
    }
    auto const base = first_window ? src[0] : record->base;
    if (first_window && i == 0) { record->base = base; }
    // The largest source offset the budget reaches (saturating), and never past the chars: a
    // corrupt offset beyond them is not written.
    auto const reach = static_cast<std::uint64_t>(base) + budget;
    auto const limit = reach < budget || reach > chars_bytes ? chars_bytes : reach;
    auto const fits  = [&](std::int64_t offset) {
      return static_cast<std::uint64_t>(offset) <= limit;
    };
    if (!fits(value)) { return; }
    if (i + 1 == count || !fits(src[i + 1])) {
      record->fit = i;
      record->end = value;
      if (i == 0) { return; }
    }
    auto const moved = value - base + dst_base;
    if (dst_width == 4) {
      static_cast<std::int32_t*>(dst)[i] = static_cast<std::int32_t>(moved);
    } else {
      static_cast<std::int64_t*>(dst)[i] = moved;
    }
  }
};

class str_split_decoder final : public chunk_decoder {
 public:
  str_split_decoder(std::unique_ptr<delta_state, void (*)(delta_state*)> offsets,
                    std::unique_ptr<ans_state, void (*)(ans_state*)> chars,
                    std::uint64_t chars_bytes)
    : _offsets_state(std::move(offsets)), _chars_state(std::move(chars)), _chars_bytes(chars_bytes)
  {
  }

  std::int64_t decode(leaf_access& leaves,
                      chunk_span const& span,
                      column_write_target const& dst,
                      rmm::cuda_stream_view stream) override
  {
    auto const num_rows  = leaves.column().num_rows;
    auto const first_row = span.first_row();
    if (span.take < 0 || span.skip_head < 0 || first_row < 0 || first_row + span.take > num_rows) {
      throw std::invalid_argument("str_split chunk decode: rows [" + std::to_string(first_row) +
                                  ", +" + std::to_string(span.take) +
                                  ") are not within the column's " + std::to_string(num_rows));
    }
    if (span.take == 0) { return 0; }
    if (first_row + span.take > span.chunk_end * kChunkRows) {
      throw std::invalid_argument("str_split chunk decode: the span's chunks do not hold its rows");
    }
    // A destination without chars storage takes the rows whose chars are empty.
    if ((dst.data == nullptr && dst.data_capacity != 0) || dst.offsets == nullptr ||
        (dst.offset_width != 4 && dst.offset_width != 8)) {
      throw std::invalid_argument("str_split chunk decode: bad STRING destination");
    }

    // Chars budget: what is left of the destination, and what a 4-byte offset can address.
    std::uint64_t budget =
      dst.data_capacity > dst.dst_byte_offset ? dst.data_capacity - dst.dst_byte_offset : 0;
    if (dst.offset_width == 4) {
      std::uint64_t const max_end = std::numeric_limits<std::int32_t>::max();
      budget = std::min(budget, max_end > dst.dst_byte_offset ? max_end - dst.dst_byte_offset : 0);
    }

    auto const mr           = leaves.mr();
    auto const last_element = first_row + span.take;  // the end of the last row offered
    auto const chunks_end   = last_element / kChunkRows + 1;
    auto* const dst_offsets = static_cast<std::byte*>(dst.offsets) +
                              static_cast<std::size_t>(dst.dst_row_offset) * dst.offset_width;

    if (_record.size() == 0) { _record = rmm::device_buffer(sizeof(cut_record), stream, mr); }
    auto* const record = static_cast<cut_record*>(_record.data());
    throw_if_cuda_error(cudaMemsetAsync(record, 0, sizeof(cut_record), stream.value()),
                        "str_split chunk decode: clearing the cut record");
    // No valid offset is past the chars, so a budget that holds them all cuts nothing.
    bool const all_fit = budget >= _chars_bytes;

    cut_record cut{};
    auto read_cut = [&] {
      read_device_bytes_completed(&cut, record, sizeof(cut_record), stream);
      if (cut.bad != 0) { throw std::runtime_error("str_split chunk decode: corrupt offsets"); }
    };

    std::int64_t rows = 0;  // rows taken so far
    while (rows < span.take) {
      // Offsets elements [element, window_last] of chunks [chunk_begin, chunk_end).
      auto const element     = first_row + rows;
      auto const chunk_begin = element / kChunkRows;
      auto const chunk_end   = std::min(chunks_end, chunk_begin + kWindowChunks);
      auto const window_last = std::min(last_element, chunk_end * kChunkRows - 1);
      auto const window_rows = window_last - element;

      auto const scratch_bytes =
        static_cast<std::size_t>((chunk_end - chunk_begin) * kChunkRows) * sizeof(std::int64_t);
      if (_scratch.size() < scratch_bytes) {
        _scratch = rmm::device_buffer(scratch_bytes, stream, mr);
      }
      auto* const chunk_offsets = static_cast<std::int64_t*>(_scratch.data());
      decode_delta_chunks(leaves, *_offsets_state, chunk_begin, chunk_end, chunk_offsets, stream);

      // The window's first offset is the previous window's last: written again, the same.
      thrust::for_each_n(
        rmm::exec_policy_nosync(stream, mr),
        thrust::counting_iterator<std::int64_t>{0},
        window_rows + 1,
        fit_offsets_fn{chunk_offsets + (element - chunk_begin * kChunkRows),
                       window_rows + 1,
                       dst_offsets + static_cast<std::size_t>(rows) * dst.offset_width,
                       dst.offset_width,
                       rows == 0,
                       static_cast<std::int64_t>(dst.dst_byte_offset),
                       budget,
                       _chars_bytes,
                       record});
      if (all_fit) {
        rows += window_rows;
        continue;
      }
      read_cut();
      rows += cut.fit;
      if (cut.fit != window_rows) { break; }
    }
    if (all_fit) { read_cut(); }
    // Source chars [chars_from, chars_to) of the rows taken.
    auto const chars_from = cut.base;
    auto const chars_to   = rows == 0 ? cut.base : cut.end;

    if (chars_to != chars_from) {
      decode_ans_range(leaves,
                       *_chars_state,
                       static_cast<std::uint64_t>(chars_from),
                       static_cast<std::uint64_t>(chars_to),
                       static_cast<std::byte*>(dst.data) + dst.dst_byte_offset,
                       stream);
    }
    return rows;
  }

  [[nodiscard]] std::size_t state_bytes() const noexcept override
  {
    return _scratch.size() + _record.size() + delta_state_bytes(*_offsets_state) +
           ans_state_bytes(*_chars_state);
  }

 private:
  std::unique_ptr<delta_state, void (*)(delta_state*)> _offsets_state;
  std::unique_ptr<ans_state, void (*)(ans_state*)> _chars_state;
  std::uint64_t _chars_bytes;  ///< uncompressed bytes of the chars leaf

  rmm::device_buffer _scratch;  ///< grow-only: one window's offsets chunks, as int64
  rmm::device_buffer _record;   ///< one cut_record, allocated by the first decode
};

}  // namespace

std::unique_ptr<chunk_decoder> make_str_split_decoder(hpln_column_description const& column)
{
  auto const unsupported = [](char const* why) {
    return unsupported_chunk_decode(std::string{"str_split chunk decode: "} + why);
  };
  if (column.dtype.id() != cudf::type_id::STRING || column.num_rows < 0) {
    throw unsupported("the column is not STRING");
  }
  // nodes[0] is the column input; its one edge leads to the str_split node.
  if (column.nodes.empty() || column.nodes[0].edges.size() != 1) {
    throw unsupported("the plan has no single root operator");
  }
  auto const root = column.nodes[0].edges[0].second;
  if (root >= column.nodes.size() || column.nodes[root].op != "str_split") {
    throw unsupported("the plan root is not a str_split");
  }
  auto const offsets_node = find_child_node(column, root, "offsets");
  auto const chars_node   = find_child_node(column, root, "chars");
  if (!offsets_node || !chars_node || column.nodes[*offsets_node].op != "delta" ||
      column.nodes[*chars_node].op != "ans") {
    throw unsupported("the offsets are not delta or the chars are not ans");
  }

  // One offset per row plus one: ceil((num_rows + 1) / 1024) chunks of bitpacked differences.
  auto const diffs_node = find_child_node(column, *offsets_node, "differences");
  auto const diffs      = diffs_node ? find_bitpack(column, *diffs_node) : std::nullopt;
  if (!diffs || column.nodes[*diffs_node].op != "bitpack") {
    throw unsupported("the offsets' differences are not a bitpack leaf");
  }
  auto const offsets_chunks =
    column.leaves[diffs->chunk_bits.leaf].buffers[diffs->chunk_bits.buffer].size;
  if (column.num_rows != 0 &&
      offsets_chunks != static_cast<std::uint64_t>(column.num_rows / kChunkRows + 1)) {
    throw unsupported("the offsets do not match the rows");
  }

  // The chars: the ans node's own leaf, one buffer holding the frame.
  std::optional<buffer_ref> chars;
  for (std::size_t l = 0; l < column.leaves.size(); ++l) {
    auto const& leaf = column.leaves[l];
    if (leaf.node_index == *chars_node && leaf.slot == kSelfRepSlot && leaf.kind == OpId::Ans &&
        leaf.buffers.size() == 1) {
      chars = buffer_ref{l, 0};
      break;
    }
  }
  if (!chars) { throw unsupported("the chars have no ANS leaf"); }

  auto offsets_state     = make_delta_state(column, *offsets_node);
  auto chars_state       = make_ans_state(column, *chars);
  auto const* const meta = std::get_if<leaf_meta::ans>(&column.leaves[chars->leaf].meta);
  return std::make_unique<str_split_decoder>(
    std::move(offsets_state), std::move(chars_state), meta->uncompressed_size);
}
}  // namespace simpatico::chunk
