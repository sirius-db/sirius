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

#include "compression/compressed_disk_representation.hpp"
#include "compression/compressed_representation.hpp"
#include "compression/simpatico_compressed_representation.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <cuda/stream>

#include <api/chunk_decode.hpp>

#include <memory>

namespace sirius::compression {

// Ownership, for the three sources below. A source owns a share of its batch's backing store
// (device blob, pinned host blob, file). The store therefore outlives the batch's own
// representation: when the batch is downgraded or dropped while a source is alive, the old
// tier's memory stays allocated, outside any reservation, until the source is destroyed. Keep a
// source (and anything that holds one, such as a compressed_reader_state) only while reading,
// and drop it before waiting for memory the batch's tier is expected to give back.

/**
 * @brief Payload source over a device-resident compressed batch.
 *
 * Zero copy: device_base() is the lowest address of the batch's payload buffers and
 * read_to_device is a device-to-device copy. The buffers of a device batch are separate
 * allocations or aligned slices of one, not the dense payload a header describes, so the source
 * must be used with a description of the SAME batch on the device tier (describe() of that
 * representation, or open_compressed_payload): only there is a buffer's payload offset its
 * address minus device_base(). Only ranges inside one buffer may be read; the bytes between two
 * buffers belong to nobody.
 *
 * The source holds a clone() of the representation, which shares the device blob: it stays
 * readable after the batch changes tier or is dropped, and the blob stays allocated for as long
 * as the source lives (see the note on ownership below). The caller orders its reads against
 * writers of the batch (a read lock, or a recorded reader event).
 *
 * Rebuilds the batch's header from its table; the device work that takes runs on @p stream.
 *
 * @throws std::invalid_argument if the batch has no table.
 */
std::shared_ptr<simpatico::payload_source> make_payload_source(
  const compressed_device_representation& representation,
  ::cuda::stream_ref stream = cudf::get_default_stream());

/**
 * @brief Payload source over a compressed batch held in pinned host blocks.
 *
 * read_to_device copies the requested byte range host-to-device, block by block. The source
 * holds a clone() of the representation, which shares the pinned blob: it stays readable after
 * the batch changes tier or is dropped, and the blob must not be released before the stream has
 * run the copies (the row-range readers wait for them).
 */
std::shared_ptr<simpatico::payload_source> make_payload_source(
  const compressed_host_representation& representation);

/**
 * @brief Payload source over a compressed batch spilled to a .hpln file.
 *
 * read_to_device reads the requested byte range of the file (payload offsets start at
 * header_bytes()) into device memory. The source holds a clone() of the representation, which
 * shares the file: it stays readable after the batch changes tier or is dropped.
 */
std::shared_ptr<simpatico::payload_source> make_payload_source(
  const compressed_disk_representation& representation);

/**
 * @brief The parsed .hpln header of a compressed representation of any tier.
 *
 * Parses on every call (a device batch stores no header: it is rebuilt from the table, with any
 * device work on @p stream; the other tiers do not use the stream); the caller caches the result
 * per call site (the row-range readers keep it in their reader state).
 *
 * The description of a device batch is specific to that batch on that tier: its payload offsets
 * are device-relative (buffer address minus the lowest buffer address) and payload_bytes is the
 * extent from that base to the end of the last buffer, gaps included. Use it only with a payload
 * source of the same representation; a description of a host or disk batch (dense offsets) is
 * wrong for a device source, and the reverse.
 *
 * @throws std::runtime_error if the header is malformed.
 * @throws std::invalid_argument if @p representation is not one of the three compressed tiers,
 *         or is a device batch without a table.
 */
std::shared_ptr<const simpatico::hpln_description> describe(
  const simpatico_compressed_representation& representation,
  ::cuda::stream_ref stream = cudf::get_default_stream());

/**
 * @brief A compressed batch's parsed header and the payload source it describes.
 */
struct compressed_payload {
  std::shared_ptr<const simpatico::hpln_description> description;
  std::shared_ptr<simpatico::payload_source> source;
};

/**
 * @brief describe() and make_payload_source() of one batch, as a matching pair.
 *
 * For a device batch the header is rebuilt once for both, on @p stream. The other tiers do not
 * use the stream.
 *
 * @throws std::runtime_error if the header is malformed.
 * @throws std::invalid_argument if @p representation is not one of the three compressed tiers,
 *         or is a device batch without a table.
 */
compressed_payload open_compressed_payload(
  const simpatico_compressed_representation& representation, ::cuda::stream_ref stream);

}  // namespace sirius::compression
