/*
 * Copyright 2025, Sirius Contributors.
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
#include "op/scan/table_scan/scan_contract.hpp"
#include "scan_manager/preparation_ledger.hpp"

#include <cstdint>
#include <ios>
#include <memory>
#include <optional>
#include <span>

namespace sirius::scan_manager {
class charging_allocator;
}
namespace sirius::io {
class sirius_datasource;
}
#include <string>
#include <string_view>
#include <vector>

namespace sirius::op::scan {
struct iceberg_delete_set;

/// Byte-level seam for Puffin containers. Implementations may read local files,
/// ioctx datasources, or object-store ranges; the parser never depends on a
/// filesystem stream after the source is supplied.
class puffin_byte_source {
 public:
  virtual ~puffin_byte_source()                       = default;
  virtual bool seek(std::streamoff offset)            = 0;
  virtual bool read(std::span<std::byte> destination) = 0;
  [[nodiscard]] virtual std::streamoff size() const   = 0;
};

std::unique_ptr<puffin_byte_source> make_puffin_byte_source(
  std::unique_ptr<sirius::io::sirius_datasource> source);

// Structural metadata only; adapters interpret blob type, properties and
// snapshot applicability before decoding. Strings borrow the metadata owner.
struct puffin_blob_metadata {
  std::string_view type;
  int64_t offset, length, snapshot_id, sequence_number;
  std::optional<std::string_view> compression_codec;
};
struct puffin_metadata_limits {
  size_t max_footer_bytes = 8ULL * 1024 * 1024;
  size_t max_blobs        = 4096;
};
class puffin_container_metadata {
 public:
  ~puffin_container_metadata();
  size_t blob_count() const;
  puffin_blob_metadata blob(size_t index) const;
  size_t field_count(size_t index) const;
  int64_t field(size_t blob_index, size_t field_index) const;
  // Absent and present-empty properties are distinct. No numeric conversion or
  // provider-specific property interpretation occurs in this structural layer.
  std::optional<std::string_view> property(size_t index, std::string_view name) const;
  std::optional<std::string_view> file_property(std::string_view name) const;
  int64_t file_size() const;
  int64_t footer_start() const;

 private:
  struct impl;
  explicit puffin_container_metadata(std::unique_ptr<impl>);
  std::unique_ptr<impl> impl_;
  friend std::shared_ptr<puffin_container_metadata const> read_puffin_metadata(
    puffin_byte_source&, scan_manager::charging_allocator&, puffin_metadata_limits);
};
// Uses the existing byte source and footer parser. Retains JSON text/parser
// storage in charged backing; no blob payload is read and no visibility is sealed.
std::shared_ptr<puffin_container_metadata const> read_puffin_metadata(
  puffin_byte_source&, scan_manager::charging_allocator&, puffin_metadata_limits = {});

/// Hard ceiling on the positions one deletion vector may decode into, independent of anything the
/// table says about itself.
///
/// A Roaring run container costs 4 bytes on disk and expands to up to 65,536 positions, so a
/// sub-megabyte blob can ask for tens of gigabytes of host memory. The manifest's `record_count`
/// and the Puffin footer's `cardinality` are cross-checked against each other, but BOTH are
/// written by the table, so neither can be the allocation bound: a table declaring 5e9 in both
/// places passes every consistency check and still OOM-kills the planning thread.
///
/// 32M deleted positions in a single data file is already far past any real table, and a valid
/// vector above this bound is DECLINED (the scan falls back to DuckDB, which streams) rather than
/// letting file metadata size a plan-time allocation.
inline constexpr int64_t kMaxDeletionVectorPositions = 32LL * 1024 * 1024;

// Independent payload limit for the provider-neutral decoder. A caller may use
// a smaller bound; neither a catalog count nor an optional footer declaration
// grants permission to allocate beyond these limits.
inline constexpr size_t kMaxDeletionVectorPayloadBytes = 64ULL * 1024 * 1024;
struct deletion_vector_decode_limits {
  size_t max_payload_bytes = kMaxDeletionVectorPayloadBytes;
  size_t max_positions     = kMaxDeletionVectorPositions;
};

// Decoded bytes are not a visibility certificate: the adapter must still prove
// data-file association, snapshot applicability and complete source coverage.
// This owner keeps the actual charged positions alive, without an Iceberg path
// or manifest. Only the decoder constructs it; consumers get immutable positions.
class decoded_delete_positions {
 public:
  std::span<int64_t const> positions() const noexcept { return positions_; }
  decoded_delete_positions(decoded_delete_positions const&)            = delete;
  decoded_delete_positions& operator=(decoded_delete_positions const&) = delete;

 private:
  friend std::shared_ptr<decoded_delete_positions const> decode_deletion_vector_payload(
    std::span<uint8_t const>, scan_manager::charging_allocator&, deletion_vector_decode_limits);
  decoded_delete_positions(scan_manager::charged_block backing, size_t count);
  scan_manager::charged_block backing_;
  std::span<int64_t const> positions_;
};

// Decode one already acquired, bounded deletion-vector-v1 payload (bare or a
// container blob). The caller must keep bytes immutable throughout this call.
// Checks framing, CRC, bitmap integrity and expansion before retained allocation.
// Returns the unique decoded count via positions().size(); optional declared
// cardinality is a separate adapter check, never the allocation bound.
std::shared_ptr<decoded_delete_positions const> decode_deletion_vector_payload(
  std::span<uint8_t const>, scan_manager::charging_allocator&, deletion_vector_decode_limits = {});

/// Ceiling on the positions retained across EVERY deletion vector of one statement.
///
/// kMaxDeletionVectorPositions is per vector, and every decoded vector is held in
/// `IcebergDeleteData::positional_deletes` until execution ends, so N individually-admissible
/// vectors retain 8 * sum(record_count) bytes: 128 vectors at the per-vector ceiling is 32 GiB.
/// Charged from the manifests' summed `record_count` BEFORE any Puffin is opened, and counted per
/// live ENTRY rather than per Puffin path, because one Puffin may carry several vectors.
inline constexpr int64_t kMaxDeletionVectorPositionsPerStatement = 256LL * 1024 * 1024;

/// What a manifest entry claims about the deletion vector it points at. The reader checks every
/// field against the Puffin footer, so a wrong pointer is caught rather than silently followed.
struct DeletionVectorRefView {
  std::string const& puffin_path;
  int64_t content_offset, content_size_in_bytes;
  std::string const& referenced_data_file;
  int64_t file_size_in_bytes, record_count;
};
struct DeletionVectorRef {
  std::string puffin_path;            ///< Path to the Puffin file.
  int64_t content_offset{-1};         ///< Byte offset of the DV blob within the file.
  int64_t content_size_in_bytes{-1};  ///< Byte length of the DV blob.
  std::string referenced_data_file;   ///< Data file the entry says this vector deletes from.
  int64_t file_size_in_bytes{-1};
  int64_t record_count{-1};  ///< Deleted positions the entry claims; -1 if absent.
};

/**
 * @brief Read a V3 deletion vector from a Puffin sidecar file.
 *
 * Validates the Puffin footer's blob descriptor against @p ref before decoding: the spec requires
 * `content_offset`/`content_size_in_bytes` to match the descriptor's `offset`/`length` exactly, so
 * an entry pointing at another entry's vector is rejected instead of deleting the wrong rows. Then
 * verifies the deletion-vector-v1 magic and CRC-32 and deserializes the 64-bit Roaring bitmap.
 *
 * @return Sorted vector of deleted row positions (int64_t).
 * @throws std::runtime_error on I/O errors, a descriptor that contradicts @p ref, magic mismatch,
 *         or CRC failure.
 */
std::vector<int64_t> read_deletion_vector(DeletionVectorRef const& ref);
std::vector<int64_t> read_deletion_vector(DeletionVectorRef const&, puffin_byte_source&);
std::shared_ptr<iceberg_delete_set const> read_deletion_vector_charged(
  DeletionVectorRef const&, sirius::scan_manager::charging_allocator&);

std::shared_ptr<iceberg_delete_set const> read_deletion_vector_charged(
  DeletionVectorRefView const&,
  sirius::scan_manager::charging_allocator&,
  std::string_view canonical_path,
  std::shared_ptr<void const> path_owner,
  scan_contract_id contract = 0);

std::shared_ptr<iceberg_delete_set const> read_deletion_vector_charged(
  DeletionVectorRefView const&,
  sirius::scan_manager::charging_allocator&,
  std::string_view canonical_path,
  std::shared_ptr<void const> path_owner,
  scan_contract_id contract,
  puffin_byte_source& source);

}  // namespace sirius::op::scan
