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

#include "op/aggregate/group_by_bypass_policy.hpp"

#include "memory/size_arithmetic.hpp"

#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>

#include <rmm/aligned.hpp>

#include <cmath>
#include <limits>

namespace sirius::op::group_by_bypass {

// The header keeps the policy's inputs free of cuDF types; this is where they are tied back.
static_assert(CUDF_MAX_ROWS ==
              static_cast<std::uint64_t>(std::numeric_limits<cudf::size_type>::max()));

namespace {

using sirius::memory::saturating_add;
using sirius::memory::saturating_mul;

constexpr std::uint64_t SATURATED = std::numeric_limits<std::size_t>::max();

constexpr std::uint64_t DEVICE_ALLOCATION_ALIGNMENT = rmm::CUDA_ALLOCATION_ALIGNMENT;

/// Slot width of the cuco set cuDF's hash groupby builds over the key rows.
constexpr std::uint64_t HASH_SLOT_BYTES = sizeof(cudf::size_type);

/// cuDF's hash groupby targets a 0.5 load factor, so the set is sized at twice the row count.
/// Mirrors libcudf 26.08 (pixi.toml pins libcudf==26.08.01); recheck cudf/groupby when bumping
/// cuDF, since nothing fails if it drifts.
constexpr std::uint64_t HASH_SLOTS_PER_ROW = 2;

/// Group gather map entry.
constexpr std::uint64_t GATHER_ENTRY_BYTES = sizeof(cudf::size_type);

/// Round @p value up to @p alignment, saturating instead of wrapping.
[[nodiscard]] std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) noexcept
{
  if (value > SATURATED - (alignment - 1)) { return SATURATED; }
  return rmm::align_up(value, alignment);
}

/// Device bytes a fixed-width column of @p rows × @p width occupies, including allocator padding.
[[nodiscard]] std::uint64_t column_bytes(std::uint64_t rows, std::uint64_t width) noexcept
{
  return align_up(saturating_mul(rows, width), DEVICE_ALLOCATION_ALIGNMENT);
}

/// Upper bound on the device bytes of @p count separately allocated fixed-width columns over
/// @p rows whose widths sum to @p total_width. Each allocation pads by less than one alignment
/// unit, so the sum of the aligned columns is at most rows × total_width + count × (alignment − 1).
[[nodiscard]] std::uint64_t columns_bytes(std::uint64_t rows,
                                          std::uint64_t total_width,
                                          std::uint64_t count) noexcept
{
  if (rows == 0 || count == 0) { return 0; }
  return saturating_add(saturating_mul(rows, total_width),
                        saturating_mul(count, DEVICE_ALLOCATION_ALIGNMENT - 1));
}

/// Device bytes one validity mask over @p rows occupies: cuDF's padded bitmask size, then the
/// allocator's alignment on top.
[[nodiscard]] std::uint64_t mask_bytes(std::uint64_t rows) noexcept
{
  if (rows == 0) { return 0; }
  // No cuDF column can be this long; decide() rejects it before the model is used.
  if (rows > CUDF_MAX_ROWS) { return SATURATED; }
  return align_up(cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(rows)),
                  DEVICE_ALLOCATION_ALIGNMENT);
}

/// Total mask bytes for @p count columns over @p rows.
[[nodiscard]] std::uint64_t masks_bytes(std::uint64_t rows, std::uint64_t count) noexcept
{
  return saturating_mul(mask_bytes(rows), count);
}

}  // namespace

const char* reason_name(decision_reason reason) noexcept
{
  switch (reason) {
    case decision_reason::disabled: return "disabled";
    case decision_reason::multi_gpu: return "multi_gpu";
    case decision_reason::already_one: return "already_one";
    case decision_reason::projected_input: return "projected_input";
    case decision_reason::unknown_metadata: return "unknown_metadata";
    case decision_reason::unsupported_residency: return "unsupported_residency";
    case decision_reason::unsupported_state: return "unsupported_state";
    case decision_reason::unsupported_downstream: return "unsupported_downstream";
    case decision_reason::size_overflow: return "size_overflow";
    case decision_reason::insufficient_budget: return "insufficient_budget";
    case decision_reason::bypass_selected: return "bypass_selected";
  }
  return "unknown";
}

memory_model model_additional_bytes(const candidate_input& in) noexcept
{
  memory_model m;

  auto const state              = in.state.value_or(state_shape{});
  std::uint64_t const rows      = state.total_rows;
  std::uint64_t const key_width = state.key_width_bytes;
  std::uint64_t const agg_width = state.agg_width_bytes;
  std::uint64_t const null_keys = state.nullable_key_columns;
  std::uint64_t const null_aggs = state.nullable_agg_columns;
  std::uint64_t const key_cols  = state.key_columns;
  std::uint64_t const agg_cols  = state.agg_columns;

  // cudf::concatenate materializes one buffer per column over every input row, plus a
  // concatenated validity mask for each column that is nullable in any input.
  std::uint64_t const key_values = columns_bytes(rows, key_width, key_cols);
  std::uint64_t const agg_values = columns_bytes(rows, agg_width, agg_cols);
  m.concat_bytes =
    saturating_add(saturating_add(key_values, agg_values),
                   saturating_add(masks_bytes(rows, null_keys), masks_bytes(rows, null_aggs)));

  // cuDF's hash groupby builds a cuco open-addressing set of row indices over the key rows.
  m.hash_set_bytes =
    align_up(saturating_mul(saturating_mul(rows, HASH_SLOTS_PER_ROW), HASH_SLOT_BYTES),
             DEVICE_ALLOCATION_ALIGNMENT);

  m.gather_map_bytes = column_bytes(rows, GATHER_ENTRY_BYTES);

  // Sparse per-row aggregation results, before the gather down to the dense group set.
  m.sparse_agg_bytes = saturating_add(agg_values, masks_bytes(rows, null_aggs));

  // Worst-case group cardinality is the full partial row count: the model never assumes the
  // result is small. The dense output therefore has the same shape as the concatenated table.
  m.output_bytes = m.concat_bytes;

  // PROJECTION/FILTER/LIMIT steps on the way to the collector allocate new columns while the merge
  // output is still live. Nullability of computed columns is unknown, so every one gets a mask.
  m.downstream_bytes =
    saturating_add(columns_bytes(rows, in.downstream_row_bytes, in.downstream_columns),
                   masks_bytes(rows, in.downstream_columns));

  // Input is already resident; any later conversion cost belongs to the executor's estimate.
  std::uint64_t total = m.concat_bytes;
  total               = saturating_add(total, m.hash_set_bytes);
  total               = saturating_add(total, m.gather_map_bytes);
  total               = saturating_add(total, m.sparse_agg_bytes);
  total               = saturating_add(total, m.output_bytes);
  total               = saturating_add(total, m.downstream_bytes);
  m.additional_needed = total;

  // The declared empirical margin. Computed in double and then bounds-checked, so a large model
  // and a large fraction reject the candidate rather than wrapping to a small requirement.
  double const fraction = (in.headroom_fraction > 0.0) ? in.headroom_fraction : 0.0;
  double const headroom = std::ceil(static_cast<double>(total) * fraction);
  if (!std::isfinite(headroom) || headroom >= static_cast<double>(SATURATED)) {
    m.headroom_bytes = SATURATED;
  } else {
    m.headroom_bytes = static_cast<std::uint64_t>(headroom);
  }

  m.required_bytes = saturating_add(total, m.headroom_bytes);
  m.overflowed     = (m.required_bytes == SATURATED) || (total == SATURATED);
  return m;
}

decision decide(const candidate_input& in) noexcept
{
  decision d;
  d.num_partitions = in.auto_num_partitions;

  // Multi-GPU keeps existing behaviour, including the partition floor and placement.
  if (in.num_admitted_gpus > 1) {
    d.reason = decision_reason::multi_gpu;
    return d;
  }

  // AUTO already chose 1. Preserve it, but claim nothing.
  if (in.auto_num_partitions <= 1) {
    d.reason = decision_reason::already_one;
    return d;
  }

  // AUTO > 1: run the eligibility gates, cheapest and most-common first.
  if (!in.upstream_complete) {
    d.reason = decision_reason::projected_input;
    return d;
  }
  if (!in.single_gpu_resident) {
    d.reason = decision_reason::unsupported_residency;
    return d;
  }
  // Unknown is checked before unsupported: a schema that could not be read says nothing about
  // whether its types are supported, and the log should say which it was.
  if (!in.state.has_value() || !in.admissible_additional_budget.has_value()) {
    d.reason = decision_reason::unknown_metadata;
    return d;
  }
  if (!in.supported_state) {
    d.reason = decision_reason::unsupported_state;
    return d;
  }
  if (!in.supported_downstream) {
    d.reason = decision_reason::unsupported_downstream;
    return d;
  }
  // A concatenated table cannot be indexed past cudf::size_type's range.
  if (in.state->total_rows > CUDF_MAX_ROWS) {
    d.reason = decision_reason::size_overflow;
    return d;
  }

  d.model           = model_additional_bytes(in);
  d.model_evaluated = true;
  if (d.model.overflowed) {
    d.reason = decision_reason::size_overflow;
    return d;
  }
  if (d.model.required_bytes > *in.admissible_additional_budget) {
    d.reason = decision_reason::insufficient_budget;
    return d;
  }

  d.num_partitions = 1;
  d.reason         = decision_reason::bypass_selected;
  return d;
}

}  // namespace sirius::op::group_by_bypass
