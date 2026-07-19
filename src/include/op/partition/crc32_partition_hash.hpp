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

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <vector>

namespace sirius {
namespace op {

/**
 * @brief Per-row CRC32 hash matching StarRocks' bucket-shuffle hash bit-exactly.
 *
 * StarRocks partitions `BUCKET_SHUFFLE_HASH_PARTITIONED` data with a CRC32 hash so that
 * GPU-computed partitions land on the same compute node as StarRocks' on-disk bucket layout. This
 * reproduces that hash on the GPU. It is a standalone primitive: nothing here is wired into the
 * partition operator or the physical plan yet.
 *
 * Semantics (mirror `Column::crc32_hash` in StarRocks BE):
 * - Base function is the standard zlib CRC-32 (reflected, poly 0xEDB88320) — NOT the hardware
 *   CRC-32C StarRocks uses for joins/filters.
 * - The per-row hash starts at seed 0; each key column folds into it left→right, each column's
 * result seeding the next (one `crc32(seed, bytes)` call per value/field).
 * - Integer / bool / float / double: raw little-endian value bytes (`sizeof`).
 * - String: raw char bytes only (length is not hashed); an empty string is a no-op.
 * - NULL row: 4 zero bytes, regardless of column type.
 * - Decimal: cuDF's `data_type` lacks precision and the DecimalV2/V3 distinction, both of which
 *   change the fold, so a `decimal_key` must be supplied for every DECIMAL key column:
 *     * DecimalV2 — folds the int128 value's integer part (int64) then fractional part (int32).
 *       Represent as cuDF DECIMAL128 with scale -9 (StarRocks stores DecimalV2 as int128, scale 9).
 *     * DecimalV3 (general) — raw little-endian bytes of the unscaled integer (DECIMAL32/64/128).
 *     * DecimalV3 precision 27, scale 9 — StarRocks' legacy compatibility split fold (same as V2).
 *
 * Not yet supported (throws): date/datetime (StarRocks hashes their string form), INT64 (large)
 * string offsets, sliced string columns (offset != 0), and nested types (array/map/struct/json).
 *
 * The caller must cast key columns so their cuDF types/scales match StarRocks' storage; casting is
 * the caller's job.
 */
class crc32_partition_hash {
 public:
  /**
   * @brief StarRocks decimal metadata for one DECIMAL key column.
   *
   * Required for every DECIMAL key column (a decimal key without a matching spec is rejected);
   * must not be supplied for a non-decimal column.
   */
  struct decimal_key {
    cudf::size_type column;  ///< Index of the DECIMAL key column in `keys`.
    int precision;           ///< StarRocks column precision (absent from `cudf::data_type`).
    bool is_decimal_v2;      ///< Legacy DecimalV2 (int128, scale 9) → int64+int32 split fold.
  };

  /**
   * @brief Compute the per-row StarRocks-compatible CRC32 over the given key columns.
   *
   * @param keys Partition-key columns, in the same order StarRocks folds them.
   * @param decimal_keys One `decimal_key` per DECIMAL key column (empty if there are none).
   * @param stream CUDA stream used for kernel launches and device allocations.
   * @param mr Device memory resource used to allocate the output.
   *
   * @return A device buffer of `keys.num_rows()` uint32 hashes (seed 0). Reducing to a partition
   *         index (e.g. `hash % num_partitions`) is left to the caller.
   */
  static rmm::device_uvector<uint32_t> compute(cudf::table_view const& keys,
                                               std::vector<decimal_key> const& decimal_keys,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr);
};

}  // namespace op
}  // namespace sirius
