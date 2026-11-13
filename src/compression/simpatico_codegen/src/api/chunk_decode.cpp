// SPDX-License-Identifier: Apache-2.0
// Chunk-decode cursor: plan classification, on-demand payload staging and decoder dispatch.

#include "api/chunk_decode.hpp"

#include "codegen/util/cuda_check.hpp"
#include "decode/chunk/chunk_decoders.hpp"
#include "util/host_observation.hpp"

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace simpatico {

namespace chunk {

namespace {

/// Throws unless [offset, offset + size) lies within a region of `limit` bytes.
void check_range(std::uint64_t offset, std::uint64_t size, std::uint64_t limit, char const* what)
{
  if (offset > limit || size > limit - offset) {
    throw std::out_of_range(std::string{"chunk decode: "} + what + " range [" +
                            std::to_string(offset) + ", +" + std::to_string(size) +
                            ") exceeds its " + std::to_string(limit) + " bytes");
  }
}

/// Byte ranges of a payload staged in device buffers, keyed by payload offset (leaf buffers and
/// the validity mask are disjoint ranges of one payload). A request not covered by one staged
/// range gets a buffer of exactly the requested bytes, filled from the staged ranges it overlaps
/// (device-to-device) and from the source for the rest, so nothing beyond the requested bytes is
/// ever read. No indexed range contains another, so begins and ends grow together and the only
/// range that can cover a request is the last one starting at or before it.
class staged_ranges {
 public:
  /// Device pointer to payload bytes [begin, begin + size) if one staged range covers them.
  [[nodiscard]] const std::byte* find(std::uint64_t begin, std::uint64_t size) const
  {
    auto const it = _staged.upper_bound(begin);
    if (it == _staged.begin()) { return nullptr; }
    auto const prev = std::prev(it);
    if (prev->second.end < begin + size) { return nullptr; }
    return static_cast<const std::byte*>(prev->second.bytes.data()) + (begin - prev->first);
  }

  /// Device pointer to payload bytes [begin, begin + size), staging what is missing. size > 0.
  const std::byte* stage(payload_source& payload,
                         std::uint64_t begin,
                         std::uint64_t size,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
  {
    if (auto const* staged = find(begin, size)) { return staged; }
    auto const end = begin + size;

    auto it = _staged.upper_bound(begin);
    if (it != _staged.begin() && std::prev(it)->second.end > begin) { --it; }

    rmm::device_buffer bytes{size, stream, mr};
    auto* const out = static_cast<std::byte*>(bytes.data());
    auto pos        = begin;
    auto read       = [&](std::uint64_t upto) {
      payload.read_to_device(pos, upto - pos, out + (pos - begin), stream);
      pos = upto;
    };
    for (; it != _staged.end() && it->first < end; ++it) {
      if (it->first > pos) { read(it->first); }
      auto const upto = std::min(end, it->second.end);
      if (upto > pos) {
        throw_if_cuda_error(cudaMemcpyAsync(out + (pos - begin),
                                            static_cast<const std::byte*>(it->second.bytes.data()) +
                                              (pos - it->first),
                                            upto - pos,
                                            cudaMemcpyDeviceToDevice,
                                            stream.value()),
                            "chunk decode: copy of staged bytes");
        pos = upto;
      }
    }
    if (pos < end) { read(end); }

    // Ranges the new one contains leave the index; their bytes stay alive for the pointers
    // already handed out.
    for (it = _staged.lower_bound(begin); it != _staged.end() && it->second.end <= end;) {
      _retired.push_back(std::move(it->second.bytes));
      it = _staged.erase(it);
    }
    _bytes_held += size;
    return static_cast<const std::byte*>(
      _staged.insert_or_assign(begin, range{end, std::move(bytes)}).first->second.bytes.data());
  }

  /// Frees every staged range.
  void clear() noexcept
  {
    _staged.clear();
    _retired.clear();
    _bytes_held = 0;
  }

  [[nodiscard]] std::size_t bytes_held() const noexcept { return _bytes_held; }

 private:
  struct range {
    std::uint64_t end{0};
    rmm::device_buffer bytes;
  };

  std::map<std::uint64_t, range> _staged;  ///< keyed by the range's first payload byte
  std::vector<rmm::device_buffer> _retired;
  std::size_t _bytes_held{0};
};

/// True for a leaf whose buffers a cursor keeps staged for its whole life: a dictionary node's
/// own leaf (the keys).
bool is_kept_leaf(hpln_column_description const& column, hpln_leaf_description const& leaf)
{
  return leaf.node_index < column.nodes.size() && column.nodes[leaf.node_index].op == "dictionary";
}

/// leaf_access over a payload_source. A device-resident payload is served zero copy. Any other
/// payload is staged on demand, only the bytes asked for.
///
/// Staged bytes live for one decode call: decompress_column_chunks calls release_transient() when
/// the decoder and write_validity have returned, so a cursor read window after window holds none
/// of the windows behind it. The exception is the buffers of a dictionary node's own leaves (the
/// keys), which the dictionary decoder points into for its whole life: those are staged once and
/// kept until the access dies.
///
/// host_bytes never adds to what is staged.
class staged_leaf_access final : public leaf_access {
 public:
  staged_leaf_access(std::shared_ptr<const hpln_description> description,
                     std::size_t column,
                     std::shared_ptr<payload_source> payload,
                     rmm::device_async_resource_ref mr)
    : _description{std::move(description)}, _column{column}, _payload{std::move(payload)}, _mr{mr}
  {
    auto const& col = this->column();
    _kept_leaf.reserve(col.leaves.size());
    for (auto const& leaf : col.leaves) {
      _kept_leaf.push_back(is_kept_leaf(col, leaf));
    }
  }

  [[nodiscard]] const hpln_column_description& column() const noexcept override
  {
    return _description->columns[_column];
  }

  const std::byte* device_bytes(std::size_t leaf,
                                std::size_t buffer,
                                std::uint64_t offset,
                                std::uint64_t size,
                                rmm::cuda_stream_view stream) override
  {
    auto const& range = buffer_range(leaf, buffer);
    check_range(offset, size, range.size, "leaf buffer");
    auto const begin = range.payload_offset + offset;
    if (auto const* base = _payload->device_base()) { return base + begin; }
    if (size == 0) { return nullptr; }
    return ranges_of(leaf).stage(*_payload, begin, size, stream, _mr);
  }

  /// Bytes of a leaf buffer in place, for a device-resident payload; nullptr otherwise, and for
  /// size == 0. Stages nothing.
  [[nodiscard]] const std::byte* resident_bytes(std::size_t leaf,
                                                std::size_t buffer,
                                                std::uint64_t offset,
                                                std::uint64_t size) const
  {
    auto const& range = buffer_range(leaf, buffer);
    check_range(offset, size, range.size, "leaf buffer");
    auto const* base = _payload->device_base();
    return base != nullptr && size != 0 ? base + range.payload_offset + offset : nullptr;
  }

  /// Stream-ordered copy of leaf-buffer bytes to device memory at dst: device to device when the
  /// payload is device resident or the bytes are already staged, otherwise read from the source
  /// straight into dst. Stages nothing.
  void read_bytes(std::size_t leaf,
                  std::size_t buffer,
                  std::uint64_t offset,
                  std::uint64_t size,
                  void* dst,
                  rmm::cuda_stream_view stream)
  {
    auto const& range = buffer_range(leaf, buffer);
    check_range(offset, size, range.size, "leaf buffer");
    if (size == 0) { return; }
    auto const begin = range.payload_offset + offset;

    auto const* src = _payload->device_base();
    src             = src != nullptr ? src + begin : ranges_of(leaf).find(begin, size);
    if (src == nullptr) {
      _payload->read_to_device(begin, size, dst, stream);
      return;
    }
    throw_if_cuda_error(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream.value()),
                        "chunk decode: copy of leaf bytes");
  }

  /// Copies from the staged bytes when they cover the request. Otherwise the bytes pass through a
  /// device buffer that is freed once the stream has completed (payload_source only reads to the
  /// device). Waits for the stream either way, also when the copy fails.
  void host_bytes(std::size_t leaf,
                  std::size_t buffer,
                  std::uint64_t offset,
                  std::uint64_t size,
                  void* dst,
                  rmm::cuda_stream_view stream) override
  {
    auto const& range = buffer_range(leaf, buffer);
    check_range(offset, size, range.size, "leaf buffer");
    if (size == 0) { return; }
    auto const begin = range.payload_offset + offset;

    rmm::device_buffer bounce;
    const std::byte* src = nullptr;
    if (auto const* base = _payload->device_base()) {
      src = base + begin;
    } else if (src = ranges_of(leaf).find(begin, size); src == nullptr) {
      bounce = rmm::device_buffer{size, stream, _mr};
      _payload->read_to_device(begin, size, bounce.data(), stream);
      src = static_cast<const std::byte*>(bounce.data());
    }
    // Returns, or throws, only after the stream has completed, so `bounce` outlives its copies.
    read_device_bytes_completed(dst, src, size, stream);
  }

  const std::byte* validity_bytes(std::uint64_t offset,
                                  std::uint64_t size,
                                  rmm::cuda_stream_view stream) override
  {
    auto const& col = column();
    check_range(offset, size, col.validity_size, "validity mask");
    auto const begin = col.validity_payload_offset + offset;
    if (auto const* base = _payload->device_base()) { return base + begin; }
    if (size == 0) { return nullptr; }
    return _transient.stage(*_payload, begin, size, stream, _mr);
  }

  [[nodiscard]] rmm::device_async_resource_ref mr() const noexcept override { return _mr; }

  /// Frees everything staged except the kept dictionary keys. Pointers into the freed bytes are
  /// dead; work already enqueued on the stream they were staged on is unaffected, as with any
  /// stream-ordered free.
  void release_transient() noexcept { _transient.clear(); }

  /// Device bytes currently staged.
  [[nodiscard]] std::size_t bytes_held() const noexcept
  {
    return _kept.bytes_held() + _transient.bytes_held();
  }

 private:
  [[nodiscard]] hpln_buffer_range const& buffer_range(std::size_t leaf, std::size_t buffer) const
  {
    auto const& leaves = column().leaves;
    if (leaf >= leaves.size() || buffer >= leaves[leaf].buffers.size()) {
      throw std::out_of_range("chunk decode: no buffer " + std::to_string(buffer) + " of leaf " +
                              std::to_string(leaf));
    }
    return leaves[leaf].buffers[buffer];
  }

  [[nodiscard]] staged_ranges& ranges_of(std::size_t leaf)
  {
    return _kept_leaf[leaf] ? _kept : _transient;
  }

  std::shared_ptr<const hpln_description> _description;
  std::size_t _column;
  std::shared_ptr<payload_source> _payload;
  rmm::device_async_resource_ref _mr;
  std::vector<bool> _kept_leaf;  ///< per leaf: its buffers are staged for the access's lifetime
  staged_ranges _kept;
  staged_ranges _transient;
};

void check_cursor_arguments(std::shared_ptr<const hpln_description> const& description,
                            std::size_t column,
                            std::shared_ptr<payload_source> const& payload)
{
  if (!description) { throw std::invalid_argument("chunk decode: null hpln_description"); }
  if (!payload) { throw std::invalid_argument("chunk decode: null payload_source"); }
  if (column >= description->columns.size()) {
    throw std::invalid_argument("chunk decode: column " + std::to_string(column) + " of " +
                                std::to_string(description->columns.size()));
  }
}

/// Child of `node` via `channel` when `node` has exactly `num_edges` edges.
std::optional<std::uint32_t> child_of(hpln_column_description const& column,
                                      std::uint32_t node,
                                      std::string_view channel,
                                      std::size_t num_edges)
{
  if (column.nodes[node].edges.size() != num_edges) { return std::nullopt; }
  return find_child_node(column, node, channel);
}

bool is_bitpack(hpln_column_description const& column, std::optional<std::uint32_t> node)
{
  return node && column.nodes[*node].op == "bitpack" && column.nodes[*node].edges.empty() &&
         find_bitpack(column, *node).has_value();
}

/// delta -> differences -> bitpack
bool is_delta_bitpack(hpln_column_description const& column, std::optional<std::uint32_t> node)
{
  return node && column.nodes[*node].op == "delta" &&
         is_bitpack(column, child_of(column, *node, "differences", 1));
}

bool is_delta_type(cudf::data_type type)
{
  return (cudf::is_integral_not_bool(type) || cudf::is_chrono(type) ||
          cudf::is_fixed_point(type)) &&
         cudf::size_of(type) <= sizeof(std::int64_t);
}

}  // namespace

std::unique_ptr<leaf_access> make_staged_leaf_access(
  std::shared_ptr<const hpln_description> description,
  std::size_t column,
  std::shared_ptr<payload_source> payload,
  rmm::device_async_resource_ref mr)
{
  check_cursor_arguments(description, column, payload);
  return std::make_unique<staged_leaf_access>(
    std::move(description), column, std::move(payload), mr);
}

std::size_t staged_bytes_held(leaf_access const& access)
{
  auto const* staged = dynamic_cast<staged_leaf_access const*>(&access);
  return staged != nullptr ? staged->bytes_held() : 0;
}

void release_transient_staging(leaf_access& access)
{
  if (auto* staged = dynamic_cast<staged_leaf_access*>(&access)) { staged->release_transient(); }
}

const std::byte* unstaged_leaf_bytes(leaf_access& leaves,
                                     std::size_t leaf,
                                     std::size_t buffer,
                                     std::uint64_t offset,
                                     std::uint64_t size,
                                     rmm::cuda_stream_view stream)
{
  if (auto const* staged = dynamic_cast<staged_leaf_access const*>(&leaves)) {
    return staged->resident_bytes(leaf, buffer, offset, size);
  }
  return size != 0 ? leaves.device_bytes(leaf, buffer, offset, size, stream) : nullptr;
}

void read_leaf_bytes(leaf_access& leaves,
                     std::size_t leaf,
                     std::size_t buffer,
                     std::uint64_t offset,
                     std::uint64_t size,
                     void* dst,
                     rmm::cuda_stream_view stream)
{
  if (auto* staged = dynamic_cast<staged_leaf_access*>(&leaves)) {
    staged->read_bytes(leaf, buffer, offset, size, dst, stream);
    return;
  }
  if (size == 0) { return; }
  auto const* src = leaves.device_bytes(leaf, buffer, offset, size, stream);
  throw_if_cuda_error(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream.value()),
                      "chunk decode: copy of leaf bytes");
}

value_plan classify_value_plan(hpln_column_description const& column)
{
  auto const& nodes = column.nodes;
  // nodes[0] is the column input; its one edge leads to the plan's first operator.
  if (nodes.empty() || nodes[0].edges.size() != 1) { return value_plan::unsupported; }
  if (std::any_of(nodes.begin(), nodes.end(), [](auto const& n) { return n.is_bitjoin; })) {
    return value_plan::unsupported;
  }
  auto const root = nodes[0].edges[0].second;
  if (root == 0 || root >= nodes.size()) { return value_plan::unsupported; }

  auto const& op        = nodes[root].op;
  auto const type       = column.dtype;
  bool const is_string  = type.id() == cudf::type_id::STRING;
  bool const is_fixed   = cudf::is_fixed_width(type);
  auto const node_count = nodes.size();

  if (op == "bitpack") {
    return is_fixed && node_count == 2 && is_bitpack(column, root) ? value_plan::bitpack
                                                                   : value_plan::unsupported;
  }
  if (op == "delta") {
    if (!is_fixed || !is_delta_type(type)) { return value_plan::unsupported; }
    if (node_count == 3 && is_delta_bitpack(column, root)) { return value_plan::delta_bitpack; }
    auto const rle = child_of(column, root, "differences", 1);
    if (node_count == 5 && rle && nodes[*rle].op == "rle" &&
        is_bitpack(column, child_of(column, *rle, "runs", 2)) &&
        is_bitpack(column, child_of(column, *rle, "values", 2))) {
      return value_plan::delta_rle_bitpack;
    }
    return value_plan::unsupported;
  }
  if (op == "dictionary") {
    return is_string && node_count == 3 && is_bitpack(column, child_of(column, root, "indices", 1))
             ? value_plan::dictionary
             : value_plan::unsupported;
  }
  if (op == "str_split") {
    auto const chars = child_of(column, root, "chars", 2);
    return is_string && node_count == 5 &&
               is_delta_bitpack(column, child_of(column, root, "offsets", 2)) && chars &&
               nodes[*chars].op == "ans" && nodes[*chars].edges.empty()
             ? value_plan::str_split
             : value_plan::unsupported;
  }
  if (op == "identity") {
    return (is_string || is_fixed) && node_count == 2 && nodes[root].edges.empty()
             ? value_plan::stored
             : value_plan::unsupported;
  }
  return value_plan::unsupported;
}

}  // namespace chunk

struct compressed_column_cursor::impl {
  impl(std::shared_ptr<const hpln_description> description,
       std::size_t column,
       std::shared_ptr<payload_source> payload,
       rmm::device_async_resource_ref mr)
    : access{std::move(description), column, std::move(payload), mr}
  {
  }

  chunk::staged_leaf_access access;
  chunk::value_plan plan{chunk::value_plan::unsupported};
  std::unique_ptr<chunk::chunk_decoder> decoder;
};

compressed_column_cursor::compressed_column_cursor(
  std::shared_ptr<const hpln_description> description,
  std::size_t column,
  std::shared_ptr<payload_source> payload,
  rmm::cuda_stream_view /*stream*/,  // nothing is staged eagerly
  rmm::device_async_resource_ref mr)
{
  chunk::check_cursor_arguments(description, column, payload);
  _impl = std::make_unique<impl>(std::move(description), column, std::move(payload), mr);

  auto const& col = _impl->access.column();
  _impl->plan     = chunk::classify_value_plan(col);
  try {
    switch (_impl->plan) {
      case chunk::value_plan::bitpack: _impl->decoder = chunk::make_bitpack_decoder(col); break;
      case chunk::value_plan::delta_bitpack:
        _impl->decoder = chunk::make_delta_decoder(col, false);
        break;
      case chunk::value_plan::delta_rle_bitpack:
        _impl->decoder = chunk::make_delta_decoder(col, true);
        break;
      case chunk::value_plan::dictionary:
        _impl->decoder = chunk::make_dictionary_decoder(col);
        break;
      case chunk::value_plan::str_split: _impl->decoder = chunk::make_str_split_decoder(col); break;
      case chunk::value_plan::stored: _impl->decoder = chunk::make_stored_decoder(col); break;
      case chunk::value_plan::unsupported: break;
    }
  } catch (unsupported_chunk_decode const&) {
    // The plan has the right shape but the decoder refuses this column.
    _impl->decoder.reset();
  }
  if (!_impl->decoder) { _impl->plan = chunk::value_plan::unsupported; }
}

compressed_column_cursor::~compressed_column_cursor()                                   = default;
compressed_column_cursor::compressed_column_cursor(compressed_column_cursor&&) noexcept = default;
compressed_column_cursor& compressed_column_cursor::operator=(compressed_column_cursor&&) noexcept =
  default;

bool compressed_column_cursor::supported() const noexcept { return _impl->decoder != nullptr; }

const hpln_column_description& compressed_column_cursor::column() const noexcept
{
  return _impl->access.column();
}

std::size_t compressed_column_cursor::device_bytes_held() const noexcept
{
  return _impl->access.bytes_held() + (_impl->decoder ? _impl->decoder->state_bytes() : 0);
}

cudf::size_type decompress_column_chunks(compressed_column_cursor& cursor,
                                         cudf::size_type chunk_begin,
                                         cudf::size_type chunk_end,
                                         cudf::size_type skip_head,
                                         cudf::size_type take,
                                         column_write_target const& dst,
                                         rmm::cuda_stream_view stream)
{
  auto const& col = cursor.column();
  chunk::chunk_span const span{chunk_begin, chunk_end, skip_head, take};

  auto const num_chunks = (col.num_rows + kChunkRows - 1) / kChunkRows;
  if (chunk_begin < 0 || chunk_end < chunk_begin || span.chunk_end > num_chunks) {
    throw std::invalid_argument("decompress_column_chunks: chunks [" + std::to_string(chunk_begin) +
                                ", " + std::to_string(chunk_end) +
                                ") are not within the column's " + std::to_string(num_chunks));
  }
  if (skip_head < 0 || skip_head >= kChunkRows || take < 0) {
    throw std::invalid_argument("decompress_column_chunks: skip_head " + std::to_string(skip_head) +
                                " or take " + std::to_string(take) + " out of range");
  }
  if (span.first_row() + span.take > std::min(span.chunk_end * kChunkRows, col.num_rows)) {
    throw std::invalid_argument("decompress_column_chunks: rows [" +
                                std::to_string(span.first_row()) + ", +" + std::to_string(take) +
                                ") run past the chunk span or the column");
  }

  if (dst.dst_row_offset < 0) {
    throw std::invalid_argument("decompress_column_chunks: negative dst_row_offset");
  }
  if (col.dtype.id() == cudf::type_id::STRING) {
    if (dst.offsets == nullptr || (dst.offset_width != 4 && dst.offset_width != 8) ||
        dst.dst_byte_offset > dst.data_capacity) {
      throw std::invalid_argument("decompress_column_chunks: bad STRING destination");
    }
  } else if (cudf::is_fixed_width(col.dtype) && take > 0) {
    auto const rows = static_cast<std::size_t>(dst.dst_row_offset) + static_cast<std::size_t>(take);
    if (dst.data == nullptr || rows * cudf::size_of(col.dtype) > dst.data_capacity) {
      throw std::invalid_argument("decompress_column_chunks: destination holds fewer than " +
                                  std::to_string(rows) + " rows");
    }
  }

  auto& impl = *cursor._impl;
  if (!impl.decoder) {
    throw unsupported_chunk_decode(
      "decompress_column_chunks: the column's plan is not chunk-decodable");
  }
  if (take == 0) { return 0; }

  // Whatever this call stages is freed when it returns, so the cursor's footprint does not grow
  // with the rows read through it.
  struct release_on_exit {
    chunk::staged_leaf_access& access;
    ~release_on_exit() { access.release_transient(); }
  } const release{impl.access};

  auto const rows = impl.decoder->decode(impl.access, span, dst, stream);
  if (rows > 0) { chunk::write_validity(impl.access, span.first_row(), rows, dst, stream); }
  return static_cast<cudf::size_type>(rows);
}

}  // namespace simpatico
