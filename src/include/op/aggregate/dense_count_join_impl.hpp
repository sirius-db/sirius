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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::op {

/** @brief COUNT(*) versus COUNT(col), derived once and shared by both execution strategies.
 *
 * For a key with preserved-side multiplicity `P`, counted-side match count `M` and non-NULL COUNT
 * argument count `V`, the emitted value is `P * max(M, 1)` for COUNT(*) and `P * V` for COUNT(col).
 * `unmatched_fill` is the `max(., 1)`/`0` floor; `counted_null_policy` is what makes `M` count `V`.
 */
struct dense_count_semantics {
  int64_t unmatched_fill;                 ///< Matches a preserved key with no counted match adds.
  cudf::null_policy counted_null_policy;  ///< Whether a NULL COUNT argument still counts a match.

  [[nodiscard]] static constexpr dense_count_semantics for_count_star(bool count_star) noexcept
  {
    return count_star ? dense_count_semantics{1, cudf::null_policy::INCLUDE}
                      : dense_count_semantics{0, cudf::null_policy::EXCLUDE};
  }

  /** @brief COUNT value of the preserved-side NULL group: the general formula at `M == 0`. */
  [[nodiscard]] constexpr int64_t null_group_value(int64_t null_group_rows) const noexcept
  {
    return unmatched_fill * null_group_rows;
  }

  /** @brief Upper bound on any group's match count, given the counted-side row count. */
  [[nodiscard]] constexpr int64_t max_matched(int64_t counted_rows) const noexcept
  {
    return std::max(counted_rows, unmatched_fill);
  }
};

/** @brief Per-key multiplicity bounds whose product is what can exceed BIGINT. */
struct dense_count_bounds {
  int64_t max_preserved_multiplicity;  ///< Upper bound on `P` for any key.
  int64_t max_counted_multiplicity;    ///< Upper bound on `M` for any key, after the COUNT(*) fill.

  [[nodiscard]] constexpr bool may_exceed_bigint() const noexcept
  {
    return max_counted_multiplicity != 0 &&
           max_preserved_multiplicity >
             std::numeric_limits<int64_t>::max() / max_counted_multiplicity;
  }
};

/** @brief Validated geometry of the direct-address histogram pair.
 *
 * Only `plan()` can produce one, so an unrepresentable layout does not exist and no consumer needs
 * to re-derive the slot width or re-check the allocation arithmetic.
 */
class dense_count_layout {
 public:
  /** @brief Plan a histogram over `[min_key, max_key]`, or nullopt if it is not representable.
   *
   * Unrepresentable means: an empty domain (`max_key < min_key`), the full 64-bit domain (`range`
   * wraps to zero), `2 * range * slot_bytes` beyond `size_t`, or `range` beyond `int64_t`. The row
   * counts select the slot width only.
   */
  [[nodiscard]] static std::optional<dense_count_layout> plan(int64_t min_key,
                                                              int64_t max_key,
                                                              int64_t preserved_rows,
                                                              int64_t counted_rows) noexcept;

  [[nodiscard]] constexpr int64_t min_key() const noexcept { return _min_key; }
  [[nodiscard]] constexpr std::size_t slots() const noexcept { return _slots; }
  [[nodiscard]] constexpr std::size_t slot_bytes() const noexcept { return _slot_bytes; }
  /** @brief Presence plus counts, in bytes. */
  [[nodiscard]] constexpr std::size_t total_bytes() const noexcept
  {
    return 2 * _slots * _slot_bytes;
  }

 private:
  constexpr dense_count_layout(int64_t min_key, std::size_t slots, std::size_t slot_bytes) noexcept
    : _min_key(min_key), _slots(slots), _slot_bytes(slot_bytes)
  {
  }

  int64_t _min_key;
  std::size_t _slots;
  std::size_t _slot_bytes;
};

/** @brief Find global non-NULL extrema across INT32 or INT64 key batches.
 *
 * Returns std::nullopt for empty/all-NULL input. Synchronizes @p stream twice per non-empty batch,
 * once for that batch's minimum and once for its maximum; only the first read blocks, since it
 * drains every enqueued reduction.
 */
[[nodiscard]] std::optional<std::pair<int64_t, int64_t>> dense_count_global_minmax(
  std::vector<cudf::column_view> const& keys,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/** @brief Accumulate preserved-key multiplicities and counted matches in direct-address histograms.
 */
class dense_count_state {
 public:
  dense_count_state(dense_count_layout const& layout,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr);

  /** @brief Accumulate non-NULL preserved keys, which must lie in the histogram domain. */
  void accumulate_preserved(cudf::column_view const& keys, rmm::cuda_stream_view stream);

  /** @brief Accumulate in-domain counted keys.
   *
   * @param count_argument COUNT argument column, absent for COUNT(*). Only its validity is read,
   *        and it must have the same length as @p keys.
   */
  void accumulate_counted(cudf::column_view const& keys,
                          std::optional<cudf::column_view> const& count_argument,
                          rmm::cuda_stream_view stream);

  /** @brief Emit `[key, BIGINT count]`, validating products when @p bounds allows overflow. */
  [[nodiscard]] std::unique_ptr<cudf::table> emit(cudf::data_type key_type,
                                                  dense_count_semantics semantics,
                                                  int64_t null_group_rows,
                                                  dense_count_bounds bounds,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr) const;

 private:
  template <typename CountT>
  struct histograms {
    /// `2 * slots`, zeroed: presence in the first half, counts in the second.
    rmm::device_uvector<CountT> bins;

    [[nodiscard]] CountT* presence() noexcept { return bins.data(); }
    [[nodiscard]] CountT const* presence() const noexcept { return bins.data(); }
    [[nodiscard]] CountT* counts() noexcept { return bins.data() + bins.size() / 2; }
    [[nodiscard]] CountT const* counts() const noexcept { return bins.data() + bins.size() / 2; }
  };
  using bins_variant = std::variant<histograms<uint32_t>, histograms<uint64_t>>;

  /// Defined in the .cu, where the CUDA error-checking macros it needs are already in scope.
  template <typename CountT>
  [[nodiscard]] static histograms<CountT> make_bins(std::size_t slots,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr);

  dense_count_layout _layout;
  bins_variant _bins;
};

/** @brief Build the `[key, BIGINT count]` table holding only the preserved-side NULL group.
 *
 * One row when @p null_group_rows > 0, an empty table otherwise.
 */
[[nodiscard]] std::unique_ptr<cudf::table> make_null_group_table(cudf::data_type key_type,
                                                                 dense_count_semantics semantics,
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
