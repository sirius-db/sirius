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

#include <cstdint>
#include <string>
#include <vector>

namespace sirius::op::scan {

/// What a manifest entry claims about the deletion vector it points at. The reader checks every
/// field against the Puffin footer, so a wrong pointer is caught rather than silently followed.
struct DeletionVectorRef {
  std::string puffin_path;            ///< Path to the Puffin file.
  int64_t content_offset{-1};         ///< Byte offset of the DV blob within the file.
  int64_t content_size_in_bytes{-1};  ///< Byte length of the DV blob.
  std::string referenced_data_file;   ///< Data file the entry says this vector deletes from.
  int64_t record_count{-1};           ///< Deleted positions the entry claims; -1 if absent.
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

}  // namespace sirius::op::scan
