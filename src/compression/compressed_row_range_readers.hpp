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

#include <api/chunk_decode.hpp>
#include <cucascade/data/row_range_read.hpp>

#include <memory>
#include <vector>

namespace sirius::compression {

/**
 * @brief Reader state for one (caller, compressed batch).
 *
 * Created by the reader on the first read and handed back on later reads of the same batch, so
 * the parsed header, the payload source and each column's decode state (staged bytes, chunk
 * prefixes, dictionary keys) are built once. Not thread-safe: one reader call at a time.
 */
struct compressed_reader_state final : cucascade::row_range_reader_state {
  std::shared_ptr<const simpatico::hpln_description> description;
  std::shared_ptr<simpatico::payload_source> payload;
  /// One cursor per source column (index = source column), created lazily; null until first used.
  std::vector<std::unique_ptr<simpatico::compressed_column_cursor>> cursors;
};

/**
 * @brief Registers row-range readers for compressed_device_representation,
 * compressed_host_representation and compressed_disk_representation.
 *
 * The readers decode only the 1024-row chunks covering the requested rows. A column whose plan
 * is not chunk-decodable makes the reader throw cucascade::row_range_unsupported_error, so the
 * caller falls back to a whole-batch decode.
 */
void register_compressed_row_range_readers(cucascade::row_range_reader_registry& registry);

}  // namespace sirius::compression
