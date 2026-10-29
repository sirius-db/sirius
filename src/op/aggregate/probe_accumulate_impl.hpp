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
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace sirius::op {

/// Aggregate kinds a probe-accumulate join can maintain per build row.
enum class accumulate_kind : std::uint8_t {
  MIN,         ///< MIN(col), NULL inputs skipped
  MAX,         ///< MAX(col), NULL inputs skipped
  COUNT,       ///< COUNT(col), NULL inputs skipped
  COUNT_STAR,  ///< COUNT(*)
};

/// One aggregate maintained against a build row.
struct accumulate_spec {
  accumulate_kind kind;
  /// Column of the probe batch the aggregate reads. Ignored by COUNT_STAR.
  cudf::size_type probe_column = 0;
  /// Type of that column. Also the output type of MIN and MAX; COUNT and COUNT_STAR emit INT64.
  cudf::data_type value_type{cudf::type_id::EMPTY};
};

/// Most aggregates one probe-accumulate join can maintain. The probe kernel receives every
/// aggregate's descriptor as one by-value launch argument, which is what bounds this.
inline constexpr std::size_t probe_accumulate_max_specs = 8;

/**
 * @brief Whether MIN, MAX and COUNT over @p type can be maintained with device atomics.
 *
 * True for the 32- and 64-bit integer carriers, which also cover DECIMAL32/DECIMAL64 and the day
 * granularity temporal types. Floating point and narrower integers have no native atomic min/max.
 */
[[nodiscard]] bool probe_accumulate_supports_type(cudf::data_type type) noexcept;

/**
 * @brief Per-build-row aggregate state for a join that never materializes its output.
 *
 * A build table of N unique keys owns N accumulator records. Every probe batch hands the join's
 * (probe row, build row) match pairs to accumulate(), which folds the probe row's values into the
 * matched build row's record with atomics. When all probe batches are done, emit() yields one row
 * per build row that matched at least once: `[build key, aggregate...]`. That is the result of
 * `SELECT key, aggs FROM probe JOIN build USING (key) GROUP BY key`, computed without the join
 * output or a group-by hash table.
 *
 * One record holds every aggregate of a build row, so a match touches one cache line instead of
 * one per aggregate, and all aggregates are folded by a single kernel launch.
 *
 * accumulate() may be called concurrently from several streams. emit() must run after every
 * accumulate() call has completed on the device, and consumes the accumulators: call it once.
 */
class probe_accumulate_state {
 public:
  /// @throws sirius::internal_exception on an unsupported MIN/MAX type or more than
  ///         `probe_accumulate_max_specs` aggregates
  explicit probe_accumulate_state(std::vector<accumulate_spec> specs);
  ~probe_accumulate_state();
  probe_accumulate_state(probe_accumulate_state const&)            = delete;
  probe_accumulate_state& operator=(probe_accumulate_state const&) = delete;

  [[nodiscard]] std::vector<accumulate_spec> const& specs() const noexcept { return _specs; }

  /// Allocate and initialise one record per build row. Must be called once before accumulate().
  void initialize(cudf::size_type num_build_rows,
                  ::cuda::stream_ref stream,
                  rmm::device_async_resource_ref mr);

  /// Whether initialize() has run.
  [[nodiscard]] bool initialized() const noexcept { return _initialized; }

  /**
   * @brief Fold the matched probe rows into their build rows' records.
   *
   * @param probe Probe batch the specs' `probe_column` indices address
   * @param probe_rows Row of @p probe for each match pair
   * @param build_rows Build row (record) for each match pair, aligned with @p probe_rows
   */
  void accumulate(cudf::table_view const& probe,
                  cudf::device_span<cudf::size_type const> probe_rows,
                  cudf::device_span<cudf::size_type const> build_rows,
                  ::cuda::stream_ref stream);

  /**
   * @brief One row per build row that matched at least once.
   *
   * @param build_key Build key column, in the build row order used by accumulate()
   * @return `[build_key, aggregate...]`: MIN/MAX carry their input type and are NULL when every
   *         matched input was NULL; COUNT and COUNT_STAR are INT64
   */
  [[nodiscard]] std::unique_ptr<cudf::table> emit(cudf::column_view const& build_key,
                                                  ::cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr);

  /// Build rows the state was initialised with; 0 before initialize().
  [[nodiscard]] cudf::size_type num_build_rows() const noexcept { return _num_build_rows; }

  /// Device bytes the records occupy for @p num_build_rows rows.
  [[nodiscard]] std::size_t record_bytes_for(cudf::size_type num_build_rows) const noexcept;

  /// Peak device bytes emit() needs beyond the records, for @p num_build_rows rows.
  [[nodiscard]] std::size_t emit_peak_bytes(cudf::size_type num_build_rows) const noexcept;

 private:
  struct layout;
  std::vector<accumulate_spec> _specs;
  std::unique_ptr<layout> _layout;
  rmm::device_buffer _records;
  cudf::size_type _num_build_rows = 0;
  bool _initialized               = false;
};

}  // namespace sirius::op
