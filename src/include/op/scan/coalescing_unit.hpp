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

// cudf
#include <cudf/types.hpp>

// standard library
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace sirius::op::scan {

/**
 * @brief The maximum number of characters that can be stored in a cuDF string column.
 *
 * This is the maximum value of `cudf::size_type`, which is used for the offset type in cuDF string
 * columns.
 */
constexpr std::size_t kCudfInt32StringsThreshold =
  static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max());

/**
 * @brief Opaque, format-owned per-batch payload.
 *
 * The coalescer stores it without interpretation; the format that produced it casts it back in
 * `make_batch`.
 */
struct coalescing_payload {
  virtual ~coalescing_payload() = default;
};

/**
 * @brief One unit of coalescing work.
 *
 * A coalescing unit represents a chunk of data that has been parsed and is ready to be coalesced
 * into a larger batch.
 */
struct coalescing_unit {
  std::size_t row_count     = 0;  ///< Drives the drop-if-empty rule
  std::size_t decoded_bytes = 0;  ///< Drives the per-batch byte cap
  std::vector<std::size_t>
    varchar_bytes_per_col;                      ///< Drives the cuDF string offset size limitation
  std::unique_ptr<coalescing_payload> payload;  ///< The opaque result of provider parsing a batch
  /// Opaque grouping key: the coalescer never merges units with different keys into one batch
  /// (empty = ungrouped, the common case). A format uses it to keep batches within a partition /
  /// shard; the coalescer doesn't interpret it.
  std::vector<std::string> group_key;
};

}  // namespace sirius::op::scan
