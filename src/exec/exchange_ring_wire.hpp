/*
 * Copyright 2026, Sirius Contributors.
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

#pragma once

#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/memory/column_metadata.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

/**
 * @file
 * @brief How the ring exchange lays a batch out on the wire.
 *
 * A batch travels as one logical byte stream, cut into fixed-size frames. Its header says how to
 * read the stream:
 * - `gpu_columns` and `host_columns`: each column's buffers (null mask, data, and for STRING the
 *   offsets child) sit at the offsets cuCascade's `column_metadata` gives, the same structure the
 *   host and disk tiers describe their tables with. A GPU batch is laid out by plan_stream; a
 *   batch spilled to host travels in its host allocation's own layout. Either kind lands on the
 *   GPU, in host blocks or in a disk file from the same frames.
 * - `compressed`: the stream is a compressed batch's payload, and the header carries its
 *   compression header. It lands as a compressed batch and is decoded on its way to the GPU.
 *
 * The header rides with the batch's first frame. Columns are fixed-width or STRING, as in the
 * direct exchange.
 */
namespace sirius::exec::ring {

enum class batch_kind : std::uint8_t {
  gpu_columns  = 1,  ///< Copied from GPU buffers.
  host_columns = 2,  ///< Copied from a batch spilled to host, raw.
  compressed   = 3,  ///< A compressed batch's payload, sent as is.
};

/// The header of one batch on the wire.
struct wire_batch {
  batch_kind kind{batch_kind::gpu_columns};
  std::uint64_t rows{0};
  /// Bytes the frames carry, in all.
  std::uint64_t stream_bytes{0};
  /// gpu_columns and host_columns: each column's buffers at offsets into the stream.
  std::vector<cucascade::memory::column_metadata> columns;
  /// compressed: the compressed batch's header, its column names, and its decoded size.
  std::vector<std::uint8_t> blob_header;
  std::vector<std::string> blob_columns;
  std::uint64_t uncompressed_bytes{0};
};

[[nodiscard]] std::vector<std::uint8_t> encode_header(wire_batch const& batch);

/// @throws sirius::invalid_input_exception unless @p bytes is a well-formed header: at least one
///         row; for the column kinds, at least one fixed-width or STRING column of that many rows,
///         each buffer the size its type needs and inside the stream.
[[nodiscard]] wire_batch decode_header(std::span<std::uint8_t const> bytes);

/// One buffer of a stream.
struct stream_buffer {
  std::size_t offset;
  std::size_t size;

  bool operator==(stream_buffer const&) const = default;
};

/**
 * @brief The non-empty buffers of @p columns, in walk order: per column its null mask, its data,
 * then its children's, depth first.
 *
 * Every end of the exchange walks buffers in this order: the sender's sources, a GPU landing's
 * allocations, and the columns rebuilt from them.
 */
[[nodiscard]] std::vector<stream_buffer> stream_map(
  std::vector<cucascade::memory::column_metadata> const& columns);

/// The part of buffer @p buffer that a stream range covers.
struct stream_piece {
  std::size_t buffer;         ///< Index into the stream_map.
  std::size_t buffer_offset;  ///< Where in the buffer the piece starts.
  std::size_t range_offset;   ///< Where in the range the piece starts.
  std::size_t size;

  bool operator==(stream_piece const&) const = default;
};

/// The pieces of @p buffers that the stream range [@p begin, @p end) covers, in stream order.
/// Bytes of the range outside every buffer (alignment padding) belong to no piece.
[[nodiscard]] std::vector<stream_piece> pieces_in(std::span<stream_buffer const> buffers,
                                                  std::size_t begin,
                                                  std::size_t end);

/// Stream buffers are aligned to this, so every copy into and out of a frame is aligned.
constexpr std::size_t stream_alignment = 256;

/**
 * @brief Lays @p table out as a stream: each buffer in walk order, aligned to stream_alignment.
 *
 * STRING chars sizes are read on @p stream.
 * @param[out] stream_bytes the stream's length.
 * @throws sirius::invalid_input_exception on a sliced column, or one neither fixed-width nor
 *         STRING.
 */
[[nodiscard]] std::vector<cucascade::memory::column_metadata> plan_stream(
  cudf::table_view const& table, rmm::cuda_stream_view stream, std::size_t& stream_bytes);

/// The device address of each of @p table's buffers in walk order, matching
/// stream_map(@p columns), where @p columns is plan_stream(@p table).
[[nodiscard]] std::vector<void const*> stream_sources(
  cudf::table_view const& table, std::vector<cucascade::memory::column_metadata> const& columns);

/// Whether @p table has a column with a non-zero offset (one level of children checked).
[[nodiscard]] bool is_sliced(cudf::table_view const& table);

}  // namespace sirius::exec::ring
