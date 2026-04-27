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

/**
 * @brief Read a V3 deletion vector from a Puffin sidecar file.
 *
 * Reads the blob at the given offset/length from the Puffin file, verifies
 * the deletion-vector-v1 magic and CRC-32, deserializes the 64-bit Roaring
 * bitmap, and returns the deleted row positions as a sorted vector.
 *
 * @param puffin_path          Path to the Puffin file.
 * @param content_offset       Byte offset of the DV blob within the file.
 * @param content_size_in_bytes Byte length of the DV blob.
 * @return Sorted vector of deleted row positions (int64_t).
 * @throws std::runtime_error on I/O errors, magic mismatch, or CRC failure.
 */
std::vector<int64_t> read_deletion_vector(std::string const& puffin_path,
                                          int64_t content_offset,
                                          int64_t content_size_in_bytes);

}  // namespace sirius::op::scan
