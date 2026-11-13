// SPDX-License-Identifier: Apache-2.0
//
// Dictionary chunk decoder: the plan
//   input -> dictionary -> keys_offsets, keys_chars, indices
//   dictionary.indices -> bitpack -> chunk_min, chunk_count, chunk_bits, packed
// keeps a STRING column as its distinct values (the keys) plus one INT32 key index per row.
//   keys_offsets  terminal leaf of the dictionary node (output port "keys_offsets"): one buffer of
//                 num_keys + 1 INT32 or INT64 offsets into keys_chars
//   keys_chars    terminal leaf (output port "keys_chars"): one buffer of UINT8 bytes
//   indices       a bitpack node, 1024 rows per chunk (see bitpack_chunks.cu)
//
// Null rows: the column's validity is detached before the encoder sees it (strip_validity), so
// cudf::dictionary::encode gives a null row the index of whatever string its slot holds - an
// ordinary key, usually the empty one. A null row therefore decodes like any other row, which is
// what the whole-table decode produces too.
//
// The keys are read once, on the first decode: the offsets go to the host (to validate them and
// to know the longest key), and both the offsets and the chars stay where the leaf_access staged
// them, at their stored width, for the cursor's life.
//
// A call walks its rows in windows of at most kWindowChunks index chunks, so its scratch is
// bounded however many rows it takes. The chars appended so far are carried from window to
// window on the device; the host reads back one record per call, and the rows that fit a window
// only when the longest key says they might not all fit.
//
// Corrupt data fails closed: bad key offsets on the first decode, an index chunk shorter than its
// rows before anything is written, and an index outside the keys (which writes no chars) when
// the call's record is read.

#include "codegen/util/cuda_check.hpp"
#include "decode/chunk/chunk_decoders.hpp"
#include "util/host_observation.hpp"

#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_scan.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace simpatico::chunk {

namespace {

/// Index chunks decoded per window; bounds the int64 scratch (2 MiB of indices, 2 MiB of ends).
constexpr std::int64_t kWindowChunks = 256;

/// What a call's device passes carry from window to window and report to the host.
struct call_record {
  std::int64_t chars;  ///< chars of the rows the windows before the current one appended
  std::int64_t bad;    ///< nonzero once a row's index was outside the keys
};

/// The key offsets as stored: num_keys + 1 INT32 or INT64, at any address when !aligned (a leaf
/// buffer can start at any payload offset).
struct key_offsets {
  std::uint8_t const* bytes{nullptr};
  std::uint8_t width{4};
  bool aligned{false};

  __device__ __forceinline__ std::int64_t operator[](std::int64_t i) const
  {
    auto const* const at = bytes + i * width;
    if (aligned) {
      return width == 4 ? std::int64_t{*reinterpret_cast<std::int32_t const*>(at)}
                        : *reinterpret_cast<std::int64_t const*>(at);
    }
    std::uint64_t v = 0;
    for (std::uint8_t k = 0; k < width; ++k) {
      v |= static_cast<std::uint64_t>(at[k]) << (8 * k);
    }
    return width == 4 ? std::int64_t{static_cast<std::int32_t>(v)} : static_cast<std::int64_t>(v);
  }
};

// Length of the key row `i` of the window points at. An index outside the keys is corrupt data:
// it is recorded and counts as an empty key, so nothing is read or written out of bounds.
struct key_length_fn {
  std::int64_t const* indices;  ///< first row of the window
  key_offsets keys;
  std::int64_t num_keys;
  call_record* record;

  __device__ std::int64_t operator()(std::int64_t i) const
  {
    auto const key = indices[i];
    if (key < 0 || key >= num_keys) {
      record->bad = 1;
      return 0;
    }
    return keys[key + 1] - keys[key];
  }
};

struct fits_fn {
  call_record const* record;
  std::int64_t budget;

  __device__ bool operator()(std::int64_t end) const { return record->chars + end <= budget; }
};

// Writes offset i of the window's rows: base, then base + the chars of rows [0, i), where base
// is where the windows before left the destination chars.
struct write_offsets_fn {
  std::int64_t const* ends;  ///< ends[i] = chars of rows [0, i] of the window
  call_record const* record;
  void* dst;
  std::uint8_t dst_width;
  std::int64_t base;  ///< dst_byte_offset

  __device__ void operator()(std::int64_t i) const
  {
    auto const value = base + record->chars + (i == 0 ? 0 : ends[i - 1]);
    if (dst_width == 4) {
      static_cast<std::int32_t*>(dst)[i] = static_cast<std::int32_t>(value);
    } else {
      static_cast<std::int64_t*>(dst)[i] = value;
    }
  }
};

// Copies the key of row i of the window to its place in the destination chars.
struct gather_chars_fn {
  std::int64_t const* indices;  ///< first row of the window
  std::int64_t const* ends;
  call_record const* record;
  key_offsets keys;
  std::int64_t num_keys;
  std::uint8_t const* key_chars;
  std::uint8_t* dst;  ///< destination chars at dst_byte_offset

  __device__ void operator()(std::int64_t i) const
  {
    auto const key = indices[i];
    if (key < 0 || key >= num_keys) { return; }
    auto const begin  = keys[key];
    auto const length = keys[key + 1] - begin;
    auto* const out   = dst + record->chars + (i == 0 ? 0 : ends[i - 1]);
    for (std::int64_t b = 0; b < length; ++b) {
      out[b] = key_chars[begin + b];
    }
  }
};

// Adds a window's chars to the record, for the next window.
struct advance_fn {
  std::int64_t const* ends;
  std::int64_t rows;
  call_record* record;

  __device__ void operator()(std::int64_t) const { record->chars += ends[rows - 1]; }
};

/// Where the dictionary's buffers are, and the shape of its keys.
struct dictionary_locator {
  buffer_ref keys_offsets;
  std::uint8_t keys_offsets_width{4};
  std::uint64_t keys_offsets_bytes{0};
  buffer_ref keys_chars;
  std::uint64_t keys_chars_bytes{0};
  bitpack_locator indices;
};

class dictionary_decoder final : public chunk_decoder {
 public:
  explicit dictionary_decoder(dictionary_locator const& locator)
    : _loc(locator), _indices_state(make_bitpack_state(locator.indices))
  {
  }

  std::int64_t decode(leaf_access& leaves,
                      chunk_span const& span,
                      column_write_target const& dst,
                      rmm::cuda_stream_view stream) override
  {
    auto const num_rows = leaves.column().num_rows;
    if (span.take < 0 || span.skip_head < 0 || span.first_row() < 0 ||
        span.first_row() + span.take > num_rows) {
      throw std::invalid_argument(
        "dictionary chunk decode: rows [" + std::to_string(span.first_row()) + ", +" +
        std::to_string(span.take) + ") are not within the column's " + std::to_string(num_rows));
    }
    if (span.take == 0) { return 0; }
    // A destination with no chars capacity needs no chars buffer: it takes empty rows only.
    if ((dst.data == nullptr && dst.data_capacity != 0) || dst.offsets == nullptr ||
        (dst.offset_width != 4 && dst.offset_width != 8)) {
      throw std::invalid_argument("dictionary chunk decode: bad STRING destination");
    }
    // Only the chunks the rows touch, which may be fewer than the span offers.
    auto const chunk_begin = span.chunk_begin + span.skip_head / kChunkRows;
    auto const skip        = span.skip_head % kChunkRows;
    auto const chunks_end =
      std::min(span.chunk_end, chunk_begin + (skip + span.take + kChunkRows - 1) / kChunkRows);
    if (chunks_end <= chunk_begin || skip + span.take > (chunks_end - chunk_begin) * kChunkRows) {
      throw std::invalid_argument(
        "dictionary chunk decode: the span's chunks do not hold its rows");
    }
    load_keys(leaves, stream);
    auto const mr = leaves.mr();

    // Chars budget: what is left of the destination, and what a 4-byte offset can address.
    std::uint64_t budget =
      dst.data_capacity > dst.dst_byte_offset ? dst.data_capacity - dst.dst_byte_offset : 0;
    if (dst.offset_width == 4) {
      std::uint64_t const max_end = std::numeric_limits<std::int32_t>::max();
      budget = std::min(budget, max_end > dst.dst_byte_offset ? max_end - dst.dst_byte_offset : 0);
    }
    auto const signed_budget = static_cast<std::int64_t>(
      std::min<std::uint64_t>(budget, std::numeric_limits<std::int64_t>::max()));

    if (_record.size() == 0) { _record = rmm::device_buffer(sizeof(call_record), stream, mr); }
    auto* const record = static_cast<call_record*>(_record.data());
    throw_if_cuda_error(cudaMemsetAsync(record, 0, sizeof(call_record), stream.value()),
                        "dictionary chunk decode: clearing the call record");

    auto* const dst_offsets = static_cast<std::byte*>(dst.offsets) +
                              static_cast<std::size_t>(dst.dst_row_offset) * dst.offset_width;
    auto* const dst_chars = static_cast<std::uint8_t*>(dst.data) + dst.dst_byte_offset;
    auto const base       = static_cast<std::int64_t>(dst.dst_byte_offset);

    std::int64_t rows = 0;  // rows taken so far
    while (rows < span.take) {
      // Rows [rows, rows + window_rows) of the call: those left in chunks [begin, end).
      auto const position    = skip + rows;
      auto const begin       = chunk_begin + position / kChunkRows;
      auto const end         = std::min(chunks_end, begin + kWindowChunks);
      auto const head        = position % kChunkRows;
      auto const window_rows = std::min(span.take - rows, (end - begin) * kChunkRows - head);

      reserve(_indices,
              static_cast<std::size_t>((end - begin) * kChunkRows) * sizeof(std::int64_t),
              leaves,
              stream);
      auto* const chunk_indices = static_cast<std::int64_t*>(_indices.data());
      unpack_bitpack_chunks(leaves, *_indices_state, begin, end, chunk_indices, stream);
      // One index per row: a corrupt count below a chunk's rows would leave stale indices in the
      // scratch. The counts are the node's, so the first window checks the whole call, before
      // anything is written.
      if (rows == 0) { require_bitpack_values(*_indices_state, chunk_begin, skip + span.take); }
      auto const* const indices = chunk_indices + head;

      // ends[i] = chars of rows [0, i] of the window.
      reserve(_ends, static_cast<std::size_t>(window_rows) * sizeof(std::int64_t), leaves, stream);
      auto* const ends = static_cast<std::int64_t*>(_ends.data());
      thrust::transform_inclusive_scan(rmm::exec_policy_nosync(stream, mr),
                                       thrust::counting_iterator<std::int64_t>{0},
                                       thrust::counting_iterator<std::int64_t>{window_rows},
                                       ends,
                                       key_length_fn{indices, _keys, _num_keys, record},
                                       thrust::plus<std::int64_t>{});

      // Rows of the window that fit: all of them when even the longest key would, for every row
      // so far, without asking the device; otherwise the ends within the budget (they do not
      // decrease), which costs one sync.
      auto fit = window_rows;
      if (_max_key_length != 0 && static_cast<std::uint64_t>(rows + window_rows) >
                                    budget / static_cast<std::uint64_t>(_max_key_length)) {
        fit = static_cast<std::int64_t>(thrust::count_if(
          rmm::exec_policy(stream, mr), ends, ends + window_rows, fits_fn{record, signed_budget}));
      }
      if (fit == 0) { break; }

      // The window's first offset is the previous window's last: written again, the same.
      thrust::for_each_n(
        rmm::exec_policy_nosync(stream, mr),
        thrust::counting_iterator<std::int64_t>{0},
        fit + 1,
        write_offsets_fn{ends,
                         record,
                         dst_offsets + static_cast<std::size_t>(rows) * dst.offset_width,
                         dst.offset_width,
                         base});
      if (_max_key_length != 0) {
        thrust::for_each_n(
          rmm::exec_policy_nosync(stream, mr),
          thrust::counting_iterator<std::int64_t>{0},
          fit,
          gather_chars_fn{indices, ends, record, _keys, _num_keys, _key_chars, dst_chars});
      }
      rows += fit;
      if (fit != window_rows) { break; }
      if (rows < span.take) {
        thrust::for_each_n(rmm::exec_policy_nosync(stream, mr),
                           thrust::counting_iterator<std::int64_t>{0},
                           1,
                           advance_fn{ends, window_rows, record});
      }
    }

    call_record result{};
    read_device_bytes_completed(&result, record, sizeof(call_record), stream);
    if (result.bad != 0) {
      throw std::runtime_error("dictionary chunk decode: an index is outside the keys");
    }
    return rows;
  }

  [[nodiscard]] std::size_t state_bytes() const noexcept override
  {
    return _indices.size() + _ends.size() + _record.size() + bitpack_state_bytes(*_indices_state);
  }

 private:
  // Grow-only scratch, so a call allocates nothing in the steady state.
  static void reserve(rmm::device_buffer& buffer,
                      std::size_t bytes,
                      leaf_access& leaves,
                      rmm::cuda_stream_view stream)
  {
    if (buffer.size() < bytes) { buffer = rmm::device_buffer(bytes, stream, leaves.mr()); }
  }

  /// Reads the keys on the first call: the offsets are validated on the host, and offsets and
  /// chars are staged whole, to stay with the leaf_access as long as the cursor lives.
  void load_keys(leaf_access& leaves, rmm::cuda_stream_view stream)
  {
    if (_keys_loaded) { return; }
    std::size_t const width = _loc.keys_offsets_width;
    auto const count        = static_cast<std::size_t>(_loc.keys_offsets_bytes / width);
    if (count == 0) { throw std::runtime_error("dictionary chunk decode: no key offsets"); }
    std::vector<std::uint8_t> raw(count * width);
    leaves.host_bytes(
      _loc.keys_offsets.leaf, _loc.keys_offsets.buffer, 0, raw.size(), raw.data(), stream);
    auto const offset_at = [&](std::size_t i) {
      std::int64_t wide = 0;
      if (width == 4) {
        std::int32_t narrow;
        std::memcpy(&narrow, raw.data() + i * width, sizeof(narrow));
        wide = narrow;
      } else {
        std::memcpy(&wide, raw.data() + i * width, sizeof(wide));
      }
      return wide;
    };

    std::int64_t max_length = 0;
    auto previous           = offset_at(0);
    if (previous < 0) { throw std::runtime_error("dictionary chunk decode: negative key offset"); }
    for (std::size_t i = 1; i < count; ++i) {
      auto const offset = offset_at(i);
      if (offset < previous) {
        throw std::runtime_error("dictionary chunk decode: decreasing key offsets");
      }
      max_length = std::max(max_length, offset - previous);
      previous   = offset;
    }
    if (static_cast<std::uint64_t>(previous) > _loc.keys_chars_bytes) {
      throw std::runtime_error("dictionary chunk decode: key offsets exceed keys_chars");
    }

    key_offsets keys;
    keys.bytes = reinterpret_cast<std::uint8_t const*>(
      leaves.device_bytes(_loc.keys_offsets.leaf, _loc.keys_offsets.buffer, 0, raw.size(), stream));
    keys.width   = _loc.keys_offsets_width;
    keys.aligned = reinterpret_cast<std::uintptr_t>(keys.bytes) % width == 0;
    if (_loc.keys_chars_bytes != 0) {
      _key_chars = reinterpret_cast<std::uint8_t const*>(leaves.device_bytes(
        _loc.keys_chars.leaf, _loc.keys_chars.buffer, 0, _loc.keys_chars_bytes, stream));
    }
    _keys           = keys;
    _num_keys       = static_cast<std::int64_t>(count) - 1;
    _max_key_length = max_length;
    _keys_loaded    = true;
  }

  dictionary_locator _loc;
  std::unique_ptr<bitpack_state, void (*)(bitpack_state*)> _indices_state;

  bool _keys_loaded{false};
  key_offsets _keys;                        ///< staged by the leaf_access, which owns them
  std::uint8_t const* _key_chars{nullptr};  ///< staged by the leaf_access, which owns it
  std::int64_t _num_keys{0};
  std::int64_t _max_key_length{0};

  rmm::device_buffer _indices;  ///< scratch: one window's index chunks, unpacked
  rmm::device_buffer _ends;     ///< scratch: inclusive scan of one window's key lengths
  rmm::device_buffer _record;   ///< one call_record, allocated by the first decode
};

/// The terminal leaf holding output port `output` of node `node`: one buffer. Throws
/// unsupported_chunk_decode if there is none.
buffer_ref find_output_leaf(hpln_column_description const& column,
                            std::uint32_t node,
                            std::string_view output)
{
  auto const& outputs = column.nodes[node].outputs;
  auto const port     = std::find(outputs.begin(), outputs.end(), output);
  if (port != outputs.end()) {
    auto const slot = static_cast<std::int32_t>(port - outputs.begin());
    for (std::size_t l = 0; l < column.leaves.size(); ++l) {
      auto const& leaf = column.leaves[l];
      if (leaf.node_index == node && leaf.slot == slot && leaf.buffers.size() == 1) {
        return buffer_ref{l, 0};
      }
    }
  }
  throw unsupported_chunk_decode("dictionary chunk decode: no stored '" + std::string{output} +
                                 "' leaf");
}

}  // namespace

std::unique_ptr<chunk_decoder> make_dictionary_decoder(hpln_column_description const& column)
{
  if (column.dtype.id() != cudf::type_id::STRING || column.num_rows < 0) {
    throw unsupported_chunk_decode("dictionary chunk decode: the column is not STRING");
  }
  // nodes[0] is the column input; its one edge leads to the dictionary node.
  if (column.nodes.empty() || column.nodes[0].edges.size() != 1) {
    throw unsupported_chunk_decode("dictionary chunk decode: the plan has no single root operator");
  }
  auto const root = column.nodes[0].edges[0].second;
  if (root >= column.nodes.size() || column.nodes[root].op != "dictionary") {
    throw unsupported_chunk_decode("dictionary chunk decode: the plan root is not a dictionary");
  }
  auto const indices_node = find_child_node(column, root, "indices");
  auto const indices      = indices_node ? find_bitpack(column, *indices_node) : std::nullopt;
  if (!indices) {
    throw unsupported_chunk_decode("dictionary chunk decode: the indices are not a bitpack leaf");
  }

  dictionary_locator locator;
  locator.indices      = *indices;
  locator.keys_offsets = find_output_leaf(column, root, "keys_offsets");
  locator.keys_chars   = find_output_leaf(column, root, "keys_chars");

  auto const& offsets_range = column.leaves[locator.keys_offsets.leaf].buffers[0];
  auto const offsets_type   = tag_to_dtype(offsets_range.type_tag).id();
  if (offsets_type != cudf::type_id::INT32 && offsets_type != cudf::type_id::INT64) {
    throw unsupported_chunk_decode("dictionary chunk decode: key offsets are not INT32 or INT64");
  }
  locator.keys_offsets_width = offsets_type == cudf::type_id::INT32 ? 4 : 8;
  locator.keys_offsets_bytes = offsets_range.size;
  if (offsets_range.size < locator.keys_offsets_width ||
      offsets_range.size % locator.keys_offsets_width != 0) {
    throw unsupported_chunk_decode("dictionary chunk decode: key offsets are empty or ragged");
  }
  auto const& chars_range = column.leaves[locator.keys_chars.leaf].buffers[0];
  auto const chars_type   = tag_to_dtype(chars_range.type_tag).id();
  if (chars_type != cudf::type_id::UINT8 && chars_type != cudf::type_id::INT8) {
    throw unsupported_chunk_decode("dictionary chunk decode: key chars are not bytes");
  }
  locator.keys_chars_bytes = chars_range.size;

  // One index per row: ceil(num_rows / 1024) chunks, of at most 8-byte elements.
  auto const index_chunks =
    column.leaves[indices->chunk_bits.leaf].buffers[indices->chunk_bits.buffer].size;
  auto const expected_chunks =
    static_cast<std::uint64_t>((column.num_rows + kChunkRows - 1) / kChunkRows);
  if ((column.num_rows != 0 && index_chunks != expected_chunks) ||
      static_cast<std::size_t>(cudf::size_of(indices->value_type)) > sizeof(std::int64_t)) {
    throw unsupported_chunk_decode("dictionary chunk decode: the indices do not match the rows");
  }
  return std::make_unique<dictionary_decoder>(locator);
}
}  // namespace simpatico::chunk
