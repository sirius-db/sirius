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

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <optional>

namespace sirius::op {

/**
 * @brief Device-side state for the dense direct-address count-join fast path.
 *
 * Holds two histograms over the preserved-side key domain [min_key, min_key + range):
 *  - `presence[k]` — number of preserved-side rows whose key equals `min_key + k`
 *    (exact duplicate handling; a zero slot means the key does not exist and emits no group),
 *  - `counts[k]`   — number of counted-side rows whose key equals `min_key + k` and whose
 *    count column (when present) is non-NULL — exactly the per-key `COUNT(col)` contribution.
 *
 * The final aggregate value per existing key k is
 *  - `presence[k] * counts[k]` for `COUNT(counted_col)` (0 when no counted rows match — the
 *    outer-join "zero count" groups), or
 *  - `presence[k] * max(counts[k], 1)` for `COUNT(*)` (unmatched preserved rows survive an
 *    outer join as one row each).
 *
 * Histogram slots are `uint32_t` by default and widen to `uint64_t` when either side's total
 * row count could overflow a 32-bit slot, so the accumulation is exact at any scale.
 *
 * NULL semantics: callers must skip NULL keys via the column validity masks (handled inside
 * `accumulate_*`); NULL preserved keys form the SQL NULL group, appended by `emit` when
 * `null_group_rows > 0`.
 */
class dense_count_state {
 public:
  /**
   * @brief Allocate and zero the two histograms.
   *
   * @param min_key Smallest non-NULL preserved-side key (array offset origin).
   * @param range   Number of slots: max_key - min_key + 1. Must be > 0.
   * @param wide    Use 64-bit slots (required when either side has >= 2^32 rows).
   * @param stream  Stream for allocation and the zeroing memsets.
   * @param mr      Device memory resource for the histograms.
   */
  dense_count_state(int64_t min_key,
                    int64_t range,
                    bool wide,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr);

  /**
   * @brief Accumulate preserved-side keys into `presence`. NULL keys are skipped (they are
   * accounted for separately as the NULL group). Every non-NULL key must lie in
   * [min_key, min_key + range) — guaranteed when the state was sized from these columns'
   * global min/max.
   */
  void accumulate_preserved(cudf::column_view const& keys, rmm::cuda_stream_view stream);

  /**
   * @brief Accumulate counted-side keys into `counts`.
   *
   * A row contributes iff its key is non-NULL, its key lies inside [min_key, min_key + range)
   * (out-of-range keys cannot match any preserved key, and unmatched counted rows contribute
   * nothing to a preserved-side outer join), and — when @p count_validity_source is given —
   * the count column is non-NULL at that row (exact `COUNT(col)` NULL semantics).
   *
   * @param keys Counted-side join key column.
   * @param count_validity_source The COUNT(col) argument column, used ONLY for its validity
   *        mask; nullptr for COUNT(*) or a provably mask-free column.
   */
  void accumulate_counted(cudf::column_view const& keys,
                          cudf::column_view const* count_validity_source,
                          rmm::cuda_stream_view stream);

  /**
   * @brief Materialize the aggregate output table: one row per key with presence > 0
   * (ascending key order), plus one trailing NULL-key row when @p null_group_rows > 0.
   *
   * @param key_type Output key column type (INT32 or INT64; values fit by construction).
   * @param count_star COUNT(*) semantics (see class comment) instead of COUNT(col).
   * @param null_group_rows Number of NULL-key preserved rows; > 0 appends the NULL group whose
   *        value is `null_group_rows` for COUNT(*) and 0 for COUNT(col).
   * @return Two-column table [key (key_type), value (INT64)].
   */
  std::unique_ptr<cudf::table> emit(cudf::data_type key_type,
                                    bool count_star,
                                    int64_t null_group_rows,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr) const;

  [[nodiscard]] int64_t min_key() const noexcept { return _min_key; }
  [[nodiscard]] int64_t range() const noexcept { return _range; }
  [[nodiscard]] bool wide() const noexcept { return _wide; }

 private:
  int64_t _min_key;
  int64_t _range;
  bool _wide;
  /// Exactly one of each pair is engaged, matching `_wide`.
  std::optional<rmm::device_uvector<uint32_t>> _presence32;
  std::optional<rmm::device_uvector<uint32_t>> _counts32;
  std::optional<rmm::device_uvector<uint64_t>> _presence64;
  std::optional<rmm::device_uvector<uint64_t>> _counts64;
};

/**
 * @brief Build the NULL-group-only / empty output for the degenerate cases (no non-NULL
 * preserved keys). Returns a [key (key_type), value (INT64)] table with one row when
 * @p null_group_rows > 0 (NULL key; value `null_group_rows` for COUNT(*), else 0) and zero
 * rows otherwise.
 */
std::unique_ptr<cudf::table> dense_count_empty_output(cudf::data_type key_type,
                                                      bool count_star,
                                                      int64_t null_group_rows,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr);

}  // namespace sirius::op
