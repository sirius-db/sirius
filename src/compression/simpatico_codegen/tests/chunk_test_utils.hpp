// SPDX-License-Identifier: Apache-2.0
// Shared utilities for the chunk-decode tests: compress one column with a plan, describe the
// in-memory compressed_table as a parsed .hpln column, serve its leaf bytes through a
// chunk::leaf_access with a request log, and compare decoded spans against a full decode.
#pragma once

#include "api/chunk_decode.hpp"
#include "api/simpatico_codegen.hpp"
#include "decode/chunk/chunk_decoders.hpp"
#include "test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace simpatico::chunk_test {

// ---------------------------------------------------------------------------
// Spill plans — verbatim copies of the plan strings in Sirius's
// src/compression/spill_compression.cpp (Simpatico cannot include Sirius).
// ---------------------------------------------------------------------------

inline constexpr char const* kBitpackPlan =
  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
inline constexpr char const* kDeltaBitpackPlan =
  "input -> delta -> differences\n"
  "delta.differences -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
inline constexpr char const* kDeltaRleBitpackPlan =
  "input -> delta -> differences\n"
  "delta.differences -> rle -> runs, values\n"
  "delta.differences.runs -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
  "delta.differences.values -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
inline constexpr char const* kDictionaryPlan =
  "input -> dictionary -> keys_offsets, keys_chars, indices\n"
  "dictionary.indices -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
inline constexpr char const* kStringSplitPlan =
  "input -> str_split -> offsets, chars\n"
  "str_split.chars -> ans\n"
  "str_split.offsets -> delta -> differences\n"
  "str_split.offsets.differences -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
inline constexpr char const* kStoredPlan = "input -> identity\n";

// ---------------------------------------------------------------------------
// Compress / full decode
// ---------------------------------------------------------------------------

/// Compresses one column with `plan_dsl` (a single-column plan) into a one-column table.
inline compressed_table compress_column_with_plan(
  cudf::column_view column,
  std::string plan_dsl,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref())
{
  ::cuda::stream_ref const sref = stream;
  return compress_with_plan(cudf::table_view{{column}}, plan_dsl, sref, mr);
}

/// The oracle: column `column` of `table` decoded whole by simpatico::decompress.
inline std::unique_ptr<cudf::column> full_decode(
  compressed_table const& table,
  std::size_t column,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref())
{
  std::array<std::size_t, 1> const selected{column};
  ::cuda::stream_ref const sref = stream;
  auto decoded = decompress(table, std::span<const std::size_t>{selected}, sref, mr);
  if (decoded == nullptr || decoded->num_columns() != 1) {
    throw std::runtime_error("full_decode: decompress did not return one column");
  }
  auto columns = decoded->release();
  stream.synchronize();
  return std::move(columns.front());
}

// ---------------------------------------------------------------------------
// Description + flattened payload of one column of an in-memory table
// ---------------------------------------------------------------------------

/// Describes column `column` of `table` as describe_hpln_header would describe a .hpln holding
/// only that column: the validity mask (if any) first, then every leaf buffer in
/// compressed_table::describe() leaf/buffer order, laid out contiguously from offset 0.
/// flatten_payload() produces the payload these offsets index.
inline hpln_column_description describe_from_table(
  compressed_table const& table,
  std::size_t column,
  rmm::cuda_stream_view stream = cudf::get_default_stream())
{
  if (column >= table.columns.size()) {
    throw std::out_of_range("describe_from_table: column out of range");
  }
  auto const& col = table.columns[column];
  if (col.plan_tree == nullptr) {
    throw std::runtime_error("describe_from_table: column has no plan tree");
  }
  auto const& tree              = *col.plan_tree;
  ::cuda::stream_ref const sref = stream;
  auto const leaves             = table.describe(sref).at(column);

  hpln_column_description out;
  out.name     = col.name;
  out.dtype    = col.dtype;
  out.num_rows = col.num_rows;

  std::uint64_t payload_offset = 0;
  out.validity                 = static_cast<hpln_validity>(tree.validity.kind);
  if (tree.validity.kind != validity_kind::all_valid) { out.null_count = tree.validity.null_count; }
  if (tree.validity.kind == validity_kind::mask) {
    out.validity_size           = tree.validity.mask.size();
    out.validity_payload_offset = payload_offset;
    payload_offset += out.validity_size;
  }

  out.nodes.reserve(tree.nodes.size());
  for (auto const& node : tree.nodes) {
    hpln_node_description described;
    described.op         = node.op;
    described.is_bitjoin = node.attrs.bitjoin.has_value();
    for (auto const& edge : node.children) {
      described.edges.emplace_back(edge.channel, edge.child);
    }
    described.outputs = node.output_names;
    out.nodes.push_back(std::move(described));
  }

  out.leaves.reserve(leaves.size());
  for (auto const& leaf : leaves) {
    hpln_leaf_description ld;
    ld.node_index = leaf.node_index;
    ld.slot       = leaf.slot;
    ld.kind       = leaf.kind;
    ld.type_tag   = leaf.type_tag;
    ld.meta       = leaf.meta;
    ld.buffers.reserve(leaf.buffers.size());
    for (auto const& buffer : leaf.buffers) {
      ld.buffers.push_back(
        hpln_buffer_range{buffer.name, buffer.type_tag, buffer.size_bytes, payload_offset});
      payload_offset += buffer.size_bytes;
    }
    out.leaves.push_back(std::move(ld));
  }
  return out;
}

/// Total payload bytes of a description built by describe_from_table.
inline std::uint64_t payload_size(hpln_column_description const& column)
{
  std::uint64_t bytes = column.validity_size;
  for (auto const& leaf : column.leaves) {
    for (auto const& buffer : leaf.buffers) {
      bytes += buffer.size;
    }
  }
  return bytes;
}

/// Copies the validity mask and the leaf buffers of column `column` into one device buffer, in
/// the order (and at the offsets) describe_from_table assigns.
inline rmm::device_buffer flatten_payload(
  compressed_table const& table,
  std::size_t column,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref())
{
  auto const description        = describe_from_table(table, column, stream);
  auto const& tree              = *table.columns[column].plan_tree;
  ::cuda::stream_ref const sref = stream;
  auto const leaves             = table.describe(sref).at(column);

  rmm::device_buffer payload(payload_size(description), stream, mr);
  auto* const base = static_cast<std::byte*>(payload.data());
  auto copy        = [&](std::uint64_t offset, void const* src, std::uint64_t size) {
    if (size == 0) { return; }
    cuda_check(cudaMemcpyAsync(base + offset, src, size, cudaMemcpyDeviceToDevice, stream.value()));
  };
  if (description.validity == hpln_validity::mask) {
    copy(description.validity_payload_offset, tree.validity.mask.data(), description.validity_size);
  }
  for (std::size_t l = 0; l < leaves.size(); ++l) {
    for (std::size_t b = 0; b < leaves[l].buffers.size(); ++b) {
      auto const& range = description.leaves[l].buffers[b];
      copy(range.payload_offset, leaves[l].buffers[b].device_ptr, range.size);
    }
  }
  stream.synchronize();
  return payload;
}

/// Synchronous host copy of `count` elements of device memory.
template <typename T>
std::vector<T> read_device(void const* src,
                           std::size_t count,
                           rmm::cuda_stream_view stream = cudf::get_default_stream())
{
  std::vector<T> host(count);
  if (count != 0) {
    cuda_check(
      cudaMemcpyAsync(host.data(), src, count * sizeof(T), cudaMemcpyDeviceToHost, stream.value()));
  }
  stream.synchronize();
  return host;
}

/// flatten_payload as host bytes (what a .hpln payload region of this column would hold).
inline std::vector<std::byte> flatten_payload_host(
  compressed_table const& table,
  std::size_t column,
  rmm::cuda_stream_view stream = cudf::get_default_stream())
{
  auto const payload = flatten_payload(table, column, stream);
  return read_device<std::byte>(payload.data(), payload.size(), stream);
}

// ---------------------------------------------------------------------------
// leaf_access over a flattened device payload
// ---------------------------------------------------------------------------

/// chunk::leaf_access over a device-resident flattened payload (zero copy), recording every
/// request so a test can assert that a decode touched only the ranges it needed. A request
/// outside its buffer throws std::out_of_range.
class device_leaf_access : public chunk::leaf_access {
 public:
  /// (leaf, buffer, offset, size) of one device_bytes / host_bytes call.
  using request = std::tuple<std::size_t, std::size_t, std::uint64_t, std::uint64_t>;
  /// (offset, size) of one validity_bytes call.
  using validity_request = std::pair<std::uint64_t, std::uint64_t>;

  /// `payload` must be laid out as `column`'s payload offsets say (see flatten_payload).
  device_leaf_access(hpln_column_description column,
                     rmm::device_buffer payload,
                     rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref())
    : _column(std::move(column)), _payload(std::move(payload)), _mr(mr)
  {
    if (payload_size(_column) > _payload.size()) {
      throw std::invalid_argument("device_leaf_access: payload smaller than the description");
    }
  }

  /// Describes and flattens column `column` of `table`.
  device_leaf_access(compressed_table const& table,
                     std::size_t column,
                     rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                     rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref())
    : device_leaf_access(
        describe_from_table(table, column, stream), flatten_payload(table, column, stream, mr), mr)
  {
  }

  [[nodiscard]] const hpln_column_description& column() const noexcept override { return _column; }

  const std::byte* device_bytes(std::size_t leaf,
                                std::size_t buffer,
                                std::uint64_t offset,
                                std::uint64_t size,
                                rmm::cuda_stream_view /*stream*/) override
  {
    auto const* bytes = locate(leaf, buffer, offset, size);
    _device_requests.emplace_back(leaf, buffer, offset, size);
    _requests.emplace_back(leaf, buffer, offset, size);
    return bytes;
  }

  void host_bytes(std::size_t leaf,
                  std::size_t buffer,
                  std::uint64_t offset,
                  std::uint64_t size,
                  void* dst,
                  rmm::cuda_stream_view stream) override
  {
    auto const* bytes = locate(leaf, buffer, offset, size);
    _host_requests.emplace_back(leaf, buffer, offset, size);
    _requests.emplace_back(leaf, buffer, offset, size);
    if (size != 0) {
      cuda_check(cudaMemcpyAsync(dst, bytes, size, cudaMemcpyDeviceToHost, stream.value()));
    }
    stream.synchronize();
  }

  const std::byte* validity_bytes(std::uint64_t offset,
                                  std::uint64_t size,
                                  rmm::cuda_stream_view /*stream*/) override
  {
    if (_column.validity != hpln_validity::mask) {
      throw std::logic_error("device_leaf_access: the column stores no validity mask");
    }
    if (offset > _column.validity_size || size > _column.validity_size - offset) {
      throw std::out_of_range("device_leaf_access: validity request out of range");
    }
    _validity_requests.emplace_back(offset, size);
    return base() + _column.validity_payload_offset + offset;
  }

  [[nodiscard]] rmm::device_async_resource_ref mr() const noexcept override { return _mr; }

  /// Every device_bytes and host_bytes request, in call order.
  [[nodiscard]] std::vector<request> const& requests() const noexcept { return _requests; }
  [[nodiscard]] std::vector<request> const& device_requests() const noexcept
  {
    return _device_requests;
  }
  [[nodiscard]] std::vector<request> const& host_requests() const noexcept
  {
    return _host_requests;
  }
  [[nodiscard]] std::vector<validity_request> const& validity_requests() const noexcept
  {
    return _validity_requests;
  }
  void clear_requests()
  {
    _requests.clear();
    _device_requests.clear();
    _host_requests.clear();
    _validity_requests.clear();
  }

  /// Sum of the sizes of the logged requests for one buffer (overlaps counted twice).
  [[nodiscard]] std::uint64_t requested_bytes(std::size_t leaf, std::size_t buffer) const
  {
    std::uint64_t bytes = 0;
    for (auto const& [l, b, offset, size] : _requests) {
      if (l == leaf && b == buffer) { bytes += size; }
    }
    return bytes;
  }

  /// True if every logged request for the buffer lies within bytes [begin, end) of it.
  [[nodiscard]] bool requests_within(std::size_t leaf,
                                     std::size_t buffer,
                                     std::uint64_t begin,
                                     std::uint64_t end) const
  {
    return std::all_of(_requests.begin(), _requests.end(), [&](request const& r) {
      auto const& [l, b, offset, size] = r;
      return l != leaf || b != buffer || size == 0 || (offset >= begin && offset + size <= end);
    });
  }

  /// The flattened payload this access serves.
  [[nodiscard]] rmm::device_buffer const& payload() const noexcept { return _payload; }

 private:
  [[nodiscard]] const std::byte* base() const noexcept
  {
    return static_cast<const std::byte*>(_payload.data());
  }

  [[nodiscard]] const std::byte* locate(std::size_t leaf,
                                        std::size_t buffer,
                                        std::uint64_t offset,
                                        std::uint64_t size) const
  {
    auto const& range = _column.leaves.at(leaf).buffers.at(buffer);
    if (offset > range.size || size > range.size - offset) {
      throw std::out_of_range("device_leaf_access: request past the end of buffer '" + range.name +
                              "' of leaf " + std::to_string(leaf));
    }
    return base() + range.payload_offset + offset;
  }

  hpln_column_description _column;
  rmm::device_buffer _payload;
  rmm::device_async_resource_ref _mr;
  std::vector<request> _requests;
  std::vector<request> _device_requests;
  std::vector<request> _host_requests;
  std::vector<validity_request> _validity_requests;
};

// ---------------------------------------------------------------------------
// Comparing decoded rows with the oracle
// ---------------------------------------------------------------------------

struct compare_options {
  /// false: null rows are compared by validity only. true: the bytes under null rows must match
  /// the oracle too (fixed width), and STRING offsets must equal the oracle's rebased offsets.
  bool null_rows_too{false};
};

namespace detail {

inline bool mismatch(std::string const& what)
{
  std::fprintf(stderr, "chunk_test mismatch: %s\n", what.c_str());
  return false;
}

inline bool range_ok(cudf::column_view oracle, cudf::size_type row_begin, cudf::size_type rows)
{
  return row_begin >= 0 && rows >= 0 && row_begin <= oracle.size() &&
         rows <= oracle.size() - row_begin;
}

}  // namespace detail

/// Bits [bit_begin, bit_begin + bits) of a device bitmask, as host bools (true = valid).
inline std::vector<bool> read_mask_bits(cudf::bitmask_type const* mask,
                                        std::size_t bit_begin,
                                        std::size_t bits,
                                        rmm::cuda_stream_view stream = cudf::get_default_stream())
{
  std::vector<bool> out(bits);
  if (bits == 0) { return out; }
  std::size_t const first_word = bit_begin / 32;
  std::size_t const last_word  = (bit_begin + bits + 31) / 32;
  auto const words =
    read_device<cudf::bitmask_type>(mask + first_word, last_word - first_word, stream);
  for (std::size_t i = 0; i < bits; ++i) {
    std::size_t const bit = bit_begin + i;
    out[i]                = ((words[bit / 32 - first_word] >> (bit % 32)) & 1U) != 0;
  }
  return out;
}

/// True if bit dst_bit_offset + i of `mask` equals the validity of oracle row row_begin + i for
/// every i in [0, rows).
inline bool compare_null_mask(cudf::column_view oracle,
                              cudf::bitmask_type const* mask,
                              std::size_t dst_bit_offset,
                              cudf::size_type row_begin,
                              cudf::size_type rows,
                              rmm::cuda_stream_view stream = cudf::get_default_stream())
{
  if (!detail::range_ok(oracle, row_begin, rows)) {
    return detail::mismatch("null mask: row range outside the oracle");
  }
  stream.synchronize();
  auto const expected = host_validity_bits(oracle);
  auto const got = read_mask_bits(mask, dst_bit_offset, static_cast<std::size_t>(rows), stream);
  for (cudf::size_type i = 0; i < rows; ++i) {
    auto const row = static_cast<std::size_t>(row_begin + i);
    if (got[static_cast<std::size_t>(i)] != expected[row]) {
      return detail::mismatch("null mask: row " + std::to_string(row) + " expected " +
                              (expected[row] ? "valid" : "null"));
    }
  }
  return true;
}

/// True if the fixed-width values at dst_data rows [dst_row_offset, dst_row_offset + rows) equal
/// oracle rows [row_begin, row_begin + rows), byte for byte.
inline bool compare_fixed_width_rows(cudf::column_view oracle,
                                     void const* dst_data,
                                     std::size_t dst_row_offset,
                                     cudf::size_type row_begin,
                                     cudf::size_type rows,
                                     rmm::cuda_stream_view stream = cudf::get_default_stream(),
                                     compare_options options      = {})
{
  if (!detail::range_ok(oracle, row_begin, rows)) {
    return detail::mismatch("fixed width: row range outside the oracle");
  }
  auto const width    = static_cast<std::size_t>(cudf::size_of(oracle.type()));
  auto const count    = static_cast<std::size_t>(rows) * width;
  auto const expected = read_device<std::uint8_t>(
    oracle.head<std::uint8_t>() + static_cast<std::size_t>(oracle.offset() + row_begin) * width,
    count,
    stream);
  auto const got = read_device<std::uint8_t>(
    static_cast<std::uint8_t const*>(dst_data) + dst_row_offset * width, count, stream);
  auto const valid = host_validity_bits(oracle);
  for (cudf::size_type i = 0; i < rows; ++i) {
    auto const row = static_cast<std::size_t>(row_begin + i);
    if (!options.null_rows_too && !valid[row]) { continue; }
    auto const at = static_cast<std::size_t>(i) * width;
    if (std::memcmp(expected.data() + at, got.data() + at, width) != 0) {
      return detail::mismatch("fixed width: value of row " + std::to_string(row) + " differs");
    }
  }
  return true;
}

/// True if the STRING rows written at dst (rows + 1 offsets of `offset_width` bytes starting at
/// offset index dst_row_offset, rebased to dst_byte_offset; chars at dst_chars + dst_byte_offset)
/// equal oracle rows [row_begin, row_begin + rows).
inline bool compare_string_rows(cudf::column_view oracle,
                                void const* dst_chars,
                                void const* dst_offsets,
                                std::uint8_t offset_width,
                                std::size_t dst_row_offset,
                                std::size_t dst_byte_offset,
                                cudf::size_type row_begin,
                                cudf::size_type rows,
                                rmm::cuda_stream_view stream = cudf::get_default_stream(),
                                compare_options options      = {})
{
  if (!detail::range_ok(oracle, row_begin, rows)) {
    return detail::mismatch("string: row range outside the oracle");
  }
  if (offset_width != 4 && offset_width != 8) {
    return detail::mismatch("string: offset width must be 4 or 8");
  }
  if (rows == 0) { return true; }
  auto const count = static_cast<std::size_t>(rows) + 1;

  auto read_offsets = [&](void const* base, std::size_t first, bool wide) {
    std::vector<std::int64_t> out(count);
    if (wide) {
      out =
        read_device<std::int64_t>(static_cast<std::int64_t const*>(base) + first, count, stream);
    } else {
      auto const narrow =
        read_device<std::int32_t>(static_cast<std::int32_t const*>(base) + first, count, stream);
      std::copy(narrow.begin(), narrow.end(), out.begin());
    }
    return out;
  };

  cudf::strings_column_view const strings{oracle};
  ::cuda::stream_ref const sref = stream;
  auto const expected_offsets   = read_offsets(strings.offsets().head<void>(),
                                             static_cast<std::size_t>(oracle.offset() + row_begin),
                                             strings.offsets().type().id() == cudf::type_id::INT64);
  auto const got_offsets        = read_offsets(dst_offsets, dst_row_offset, offset_width == 8);

  if (got_offsets.front() != static_cast<std::int64_t>(dst_byte_offset)) {
    return detail::mismatch("string: first offset is " + std::to_string(got_offsets.front()) +
                            ", expected dst_byte_offset " + std::to_string(dst_byte_offset));
  }
  for (std::size_t i = 0; i + 1 < count; ++i) {
    if (got_offsets[i + 1] < got_offsets[i]) {
      return detail::mismatch("string: offsets decrease at row " +
                              std::to_string(static_cast<std::size_t>(row_begin) + i));
    }
  }

  auto const expected_chars = read_device<std::uint8_t>(
    strings.chars_begin(sref) + expected_offsets.front(),
    static_cast<std::size_t>(expected_offsets.back() - expected_offsets.front()),
    stream);
  auto const got_chars =
    read_device<std::uint8_t>(static_cast<std::uint8_t const*>(dst_chars) + got_offsets.front(),
                              static_cast<std::size_t>(got_offsets.back() - got_offsets.front()),
                              stream);
  auto const valid = host_validity_bits(oracle);

  for (std::size_t i = 0; i + 1 < count; ++i) {
    auto const row = static_cast<std::size_t>(row_begin) + i;
    if (options.null_rows_too) {
      if (got_offsets[i + 1] - got_offsets.front() !=
          expected_offsets[i + 1] - expected_offsets.front()) {
        return detail::mismatch("string: rebased offset after row " + std::to_string(row) +
                                " differs");
      }
    } else if (!valid[row]) {
      continue;
    }
    auto const length = expected_offsets[i + 1] - expected_offsets[i];
    if (got_offsets[i + 1] - got_offsets[i] != length) {
      return detail::mismatch("string: length of row " + std::to_string(row) + " differs");
    }
    if (length > 0 &&
        std::memcmp(expected_chars.data() + (expected_offsets[i] - expected_offsets.front()),
                    got_chars.data() + (got_offsets[i] - got_offsets.front()),
                    static_cast<std::size_t>(length)) != 0) {
      return detail::mismatch("string: chars of row " + std::to_string(row) + " differ");
    }
  }
  return true;
}

/// True if the `rows` rows a decode appended at `dst` (at dst.dst_row_offset, and
/// dst.dst_byte_offset for STRING chars) equal oracle rows [row_begin, row_begin + rows): values
/// or offsets + chars by the oracle's dtype, and validity when dst.null_mask is set. With no
/// null mask the oracle rows must all be valid.
inline bool compare_rows(cudf::column_view oracle,
                         column_write_target const& dst,
                         cudf::size_type row_begin,
                         cudf::size_type rows,
                         rmm::cuda_stream_view stream = cudf::get_default_stream(),
                         compare_options options      = {})
{
  if (!detail::range_ok(oracle, row_begin, rows)) {
    return detail::mismatch("row range outside the oracle");
  }
  auto const dst_row = static_cast<std::size_t>(dst.dst_row_offset);
  if (dst.null_mask != nullptr) {
    if (!compare_null_mask(oracle, dst.null_mask, dst_row, row_begin, rows, stream)) {
      return false;
    }
  } else {
    stream.synchronize();
    auto const valid = host_validity_bits(oracle);
    for (cudf::size_type i = 0; i < rows; ++i) {
      if (!valid[static_cast<std::size_t>(row_begin + i)]) {
        return detail::mismatch("the range has nulls but the destination has no null mask");
      }
    }
  }
  if (oracle.type().id() == cudf::type_id::STRING) {
    return compare_string_rows(oracle,
                               dst.data,
                               dst.offsets,
                               dst.offset_width,
                               dst_row,
                               dst.dst_byte_offset,
                               row_begin,
                               rows,
                               stream,
                               options);
  }
  return compare_fixed_width_rows(oracle, dst.data, dst_row, row_begin, rows, stream, options);
}

// ---------------------------------------------------------------------------
// Seeded random columns
// ---------------------------------------------------------------------------

struct column_options {
  cudf::size_type rows{4096};
  std::uint64_t seed{1};
  double null_fraction{0.0};       ///< 0 ⇒ no null mask; >= 1 ⇒ every row null
  bool sorted{false};              ///< ascending values
  std::int64_t cardinality{0};     ///< > 0 ⇒ at most this many distinct values; 0 ⇒ high
  std::int64_t run_length{1};      ///< each drawn value repeats this many consecutive rows
  std::int32_t decimal_scale{-2};  ///< scale of DECIMAL32/64/128 columns
  std::size_t min_length{0};       ///< STRING: shortest value, bytes
  std::size_t max_length{24};      ///< STRING: longest value, bytes
};

/// Per-row validity (true = valid) drawn with `null_fraction`; empty when null_fraction <= 0.
inline std::vector<bool> random_validity(cudf::size_type rows,
                                         double null_fraction,
                                         std::uint64_t seed)
{
  if (null_fraction <= 0.0) { return {}; }
  std::vector<bool> valid(static_cast<std::size_t>(rows));
  std::mt19937_64 rng{seed ^ 0x9E3779B97F4A7C15ULL};
  std::bernoulli_distribution is_null{std::min(null_fraction, 1.0)};
  for (auto&& bit : valid) {
    bit = !is_null(rng);
  }
  return valid;
}

/// A fixed-width column of `dtype` holding `rows` values copied from host `bytes`
/// (rows * size_of(dtype) bytes). `valid` is empty (no null mask) or one bool per row.
inline std::unique_ptr<cudf::column> make_fixed_width_column_from_host(
  cudf::data_type dtype,
  cudf::size_type rows,
  void const* bytes,
  std::vector<bool> const& valid,
  rmm::cuda_stream_view stream = cudf::get_default_stream())
{
  auto column = cudf::make_fixed_width_column(dtype, rows, cudf::mask_state::UNALLOCATED, stream);
  auto const size = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cudf::size_of(dtype));
  if (size != 0) {
    cuda_check(cudaMemcpyAsync(
      column->mutable_view().head<void>(), bytes, size, cudaMemcpyHostToDevice, stream.value()));
  }
  if (!valid.empty()) {
    if (valid.size() != static_cast<std::size_t>(rows)) {
      throw std::invalid_argument("make_fixed_width_column_from_host: validity size != rows");
    }
    auto const mask_bytes = cudf::bitmask_allocation_size_bytes(rows);
    std::vector<cudf::bitmask_type> words(mask_bytes / sizeof(cudf::bitmask_type), 0U);
    cudf::size_type nulls = 0;
    for (std::size_t r = 0; r < valid.size(); ++r) {
      if (valid[r]) {
        words[r / 32] |= cudf::bitmask_type{1} << (r % 32);
      } else {
        ++nulls;
      }
    }
    rmm::device_buffer mask(mask_bytes, stream);
    if (mask_bytes != 0) {
      cuda_check(cudaMemcpyAsync(
        mask.data(), words.data(), mask_bytes, cudaMemcpyHostToDevice, stream.value()));
    }
    column->set_null_mask(std::move(mask), nulls);
  }
  stream.synchronize();
  return column;
}

/// Typed convenience over make_fixed_width_column_from_host; sizeof(T) must be size_of(dtype).
template <typename T>
std::unique_ptr<cudf::column> make_fixed_width_column(
  cudf::data_type dtype,
  std::vector<T> const& values,
  std::vector<bool> const& valid = {},
  rmm::cuda_stream_view stream   = cudf::get_default_stream())
{
  if (sizeof(T) != static_cast<std::size_t>(cudf::size_of(dtype))) {
    throw std::invalid_argument("make_fixed_width_column: element size != size_of(dtype)");
  }
  return make_fixed_width_column_from_host(
    dtype, static_cast<cudf::size_type>(values.size()), values.data(), valid, stream);
}

namespace detail {

// `rows` draws of `draw`, honoring cardinality, run_length and sorted.
template <typename T, typename Draw>
std::vector<T> draw_values(column_options const& options, std::mt19937_64& rng, Draw&& draw)
{
  auto const rows = static_cast<std::size_t>(options.rows);
  std::vector<T> pool;
  if (options.cardinality > 0) {
    pool.reserve(static_cast<std::size_t>(options.cardinality));
    for (std::int64_t i = 0; i < options.cardinality; ++i) {
      pool.push_back(draw(rng));
    }
  }
  auto next = [&]() -> T {
    if (pool.empty()) { return draw(rng); }
    return pool[std::uniform_int_distribution<std::size_t>{0, pool.size() - 1}(rng)];
  };
  auto const run = static_cast<std::size_t>(std::max<std::int64_t>(options.run_length, 1));
  std::vector<T> values;
  values.reserve(rows);
  while (values.size() < rows) {
    T value             = next();
    std::size_t const n = std::min(run, rows - values.size());
    values.insert(values.end(), n, value);
  }
  if (options.sorted) { std::sort(values.begin(), values.end()); }
  return values;
}

}  // namespace detail

/// A seeded random column of type `id`: any fixed-width integer, floating, decimal or chrono
/// type (INT32, INT64, DECIMAL64, TIMESTAMP_*, FLOAT64, ...) or STRING.
inline std::unique_ptr<cudf::column> make_random_column(
  cudf::type_id id,
  column_options const& options = {},
  rmm::cuda_stream_view stream  = cudf::get_default_stream())
{
  std::mt19937_64 rng{options.seed};
  auto const valid = random_validity(options.rows, options.null_fraction, options.seed);

  if (id == cudf::type_id::STRING) {
    static constexpr char kAlphabet[] =
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-";
    auto const max_length = std::max(options.min_length, options.max_length);
    std::uniform_int_distribution<std::size_t> length{options.min_length, max_length};
    std::uniform_int_distribution<std::size_t> letter{0, sizeof(kAlphabet) - 2};
    auto const values = detail::draw_values<std::string>(options, rng, [&](std::mt19937_64& g) {
      std::string value(length(g), '\0');
      for (auto& c : value) {
        c = kAlphabet[letter(g)];
      }
      return value;
    });
    ::cuda::stream_ref const sref = stream;
    return make_strings_column(values, valid, sref);
  }

  bool const is_decimal = id == cudf::type_id::DECIMAL32 || id == cudf::type_id::DECIMAL64 ||
                          id == cudf::type_id::DECIMAL128;
  auto const dtype = is_decimal ? cudf::data_type{id, options.decimal_scale} : cudf::data_type{id};
  if (!cudf::is_fixed_width(dtype) || id == cudf::type_id::BOOL8) {
    throw std::invalid_argument("make_random_column: unsupported type");
  }
  auto const width = static_cast<std::size_t>(cudf::size_of(dtype));

  // Integer storage values. The range leaves headroom below the type's width so deltas of
  // neighbours never overflow it, and chrono values look like real dates / event times.
  int const bits  = std::min<int>(static_cast<int>(width) * 8 - 2, 40);
  std::int64_t lo = -(std::int64_t{1} << bits);
  std::int64_t hi = (std::int64_t{1} << bits) - 1;
  if (cudf::is_unsigned(dtype)) { lo = 0; }
  if (cudf::is_chrono(dtype)) {
    lo = width == 4 ? 15'000 : 1'600'000'000'000'000LL;
    hi = lo + (width == 4 ? 8'000 : (std::int64_t{1} << 40));
  }
  if (id == cudf::type_id::FLOAT32) {
    lo = -(std::int64_t{1} << 22);
    hi = (std::int64_t{1} << 22) - 1;
  }
  std::uniform_int_distribution<std::int64_t> value{lo, hi};
  auto const values =
    detail::draw_values<std::int64_t>(options, rng, [&](std::mt19937_64& g) { return value(g); });

  std::vector<std::uint8_t> bytes(values.size() * width);
  for (std::size_t r = 0; r < values.size(); ++r) {
    auto* const at = bytes.data() + r * width;
    if (id == cudf::type_id::FLOAT64) {
      double const d = static_cast<double>(values[r]) * 0.01;
      std::memcpy(at, &d, sizeof(d));
    } else if (id == cudf::type_id::FLOAT32) {
      float const f = static_cast<float>(values[r]) * 0.25F;
      std::memcpy(at, &f, sizeof(f));
    } else if (width == 16) {
      __int128 const wide = values[r];
      std::memcpy(at, &wide, sizeof(wide));
    } else {
      std::memcpy(at, &values[r], width);  // little-endian narrowing
    }
  }
  return make_fixed_width_column_from_host(dtype, options.rows, bytes.data(), valid, stream);
}

}  // namespace simpatico::chunk_test
