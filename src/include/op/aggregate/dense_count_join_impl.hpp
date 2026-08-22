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
#include <utility>
#include <vector>

namespace sirius::op {

/** @brief Find global non-NULL extrema across INT32 or INT64 key batches.
 *
 * Returns std::nullopt for empty/all-NULL input and synchronizes @p stream once.
 */
std::optional<std::pair<int64_t, int64_t>> dense_count_global_minmax(
  std::vector<cudf::column_view> const& keys,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/** @brief Accumulate preserved-key multiplicities and counted matches in direct-address histograms.
 */
class dense_count_state {
 public:
  dense_count_state(int64_t min_key,
                    int64_t range,
                    bool wide,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr);

  /** @brief Accumulate non-NULL preserved keys, which must lie in the histogram domain. */
  void accumulate_preserved(cudf::column_view const& keys, rmm::cuda_stream_view stream);

  /** @brief Accumulate in-domain counted keys; nullptr validity applies COUNT(*) semantics. */
  void accumulate_counted(cudf::column_view const& keys,
                          cudf::column_view const* count_validity_source,
                          rmm::cuda_stream_view stream);

  /** @brief Emit `[key, BIGINT count]`, optionally checking products for BIGINT overflow. */
  std::unique_ptr<cudf::table> emit(cudf::data_type key_type,
                                    bool count_star,
                                    int64_t null_group_rows,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr,
                                    bool check_product_overflow) const;

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

/** @brief Build the NULL-group-only or empty output when no non-NULL preserved key exists. */
std::unique_ptr<cudf::table> dense_count_empty_output(cudf::data_type key_type,
                                                      bool count_star,
                                                      int64_t null_group_rows,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr);

/** @brief Validate equal-length, non-null INT64 products against BIGINT overflow.
 *
 * Synchronizes @p stream to read the result.
 */
void throw_if_count_product_overflows(cudf::column_view const& lhs,
                                      cudf::column_view const& rhs,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

}  // namespace sirius::op
