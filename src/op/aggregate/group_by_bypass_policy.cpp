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

#include <cmath>
#include <limits>

namespace sirius::op::group_by_bypass {

namespace {

constexpr std::uint64_t SATURATED = std::numeric_limits<std::size_t>::max();

/// rmm::CUDA_ALLOCATION_ALIGNMENT. Hard-coded rather than included so this translation unit stays
/// free of device headers and the policy remains host-testable; the value is asserted against
/// rmm in the unit tests.
constexpr std::uint64_t DEVICE_ALLOCATION_ALIGNMENT = 256;

/// cuDF pads validity bitmasks to this many bytes before allocating.
constexpr std::uint64_t BITMASK_PAD_BYTES = 64;

/// Slot width of the cuco set cuDF's hash groupby builds over the key rows: one `cudf::size_type`.
constexpr std::uint64_t HASH_SLOT_BYTES = 4;

/// cuDF's hash groupby targets a 0.5 load factor, so the set is sized at twice the row count.
constexpr std::uint64_t HASH_SLOTS_PER_ROW = 2;

/// `cudf::size_type` gather map entry.
constexpr std::uint64_t GATHER_ENTRY_BYTES = 4;

[[nodiscard]] std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) noexcept
{
  return static_cast<std::uint64_t>(
    sirius::memory::saturating_add(static_cast<std::size_t>(a), static_cast<std::size_t>(b)));
}

[[nodiscard]] std::uint64_t saturating_mul(std::uint64_t a, std::uint64_t b) noexcept
{
  return static_cast<std::uint64_t>(
    sirius::memory::saturating_mul(static_cast<std::size_t>(a), static_cast<std::size_t>(b)));
}

/// Round @p value up to @p alignment, saturating instead of wrapping.
[[nodiscard]] std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) noexcept
{
  if (value == 0) { return 0; }
  std::uint64_t const bumped = saturating_add(value, alignment - 1);
  if (bumped == SATURATED) { return SATURATED; }
  return (bumped / alignment) * alignment;
}

/// Device bytes a fixed-width column of @p rows × @p width occupies, including allocator padding.
[[nodiscard]] std::uint64_t column_bytes(std::uint64_t rows, std::uint64_t width) noexcept
{
  return align_up(saturating_mul(rows, width), DEVICE_ALLOCATION_ALIGNMENT);
}

/// Device bytes one validity mask over @p rows occupies: cuDF pads the bitmask to 64 bytes, then
/// the allocator pads that to its own alignment.
[[nodiscard]] std::uint64_t mask_bytes(std::uint64_t rows) noexcept
{
  if (rows == 0) { return 0; }
  std::uint64_t const bits = align_up(saturating_add(rows, 7) / 8, BITMASK_PAD_BYTES);
  return align_up(bits, DEVICE_ALLOCATION_ALIGNMENT);
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

  std::uint64_t const rows      = in.total_rows.value_or(0);
  std::uint64_t const key_width = in.key_width_bytes.value_or(0);
  std::uint64_t const agg_width = in.agg_width_bytes.value_or(0);
  std::uint64_t const null_keys = in.nullable_key_columns.value_or(0);
  std::uint64_t const null_aggs = in.nullable_agg_columns.value_or(0);

  // cudf::concatenate materializes one buffer per column over every input row, plus a
  // concatenated validity mask for each column that is nullable in any input.
  std::uint64_t const key_values = column_bytes(rows, key_width);
  std::uint64_t const agg_values = column_bytes(rows, agg_width);
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

  // A PROJECTION/FILTER on the way to the collector allocates a transformed copy while the merge
  // output is still live.
  m.downstream_bytes = in.downstream_materializes_copy ? m.output_bytes : 0;

  // v1 requires the input to already be resident in the target space, so there is nothing to
  // clone or upgrade here. The executor's bytes_to_materialize_input remains the single source
  // of that cost.
  m.materialization_bytes = 0;

  std::uint64_t total = m.materialization_bytes;
  total               = saturating_add(total, m.concat_bytes);
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
    d.num_partitions = in.auto_num_partitions;
    d.reason         = decision_reason::already_one;
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
  if (!in.supported_state) {
    d.reason = decision_reason::unsupported_state;
    return d;
  }
  if (!in.supported_downstream) {
    d.reason = decision_reason::unsupported_downstream;
    return d;
  }
  if (!in.total_rows.has_value() || !in.key_width_bytes.has_value() ||
      !in.agg_width_bytes.has_value() || !in.nullable_key_columns.has_value() ||
      !in.nullable_agg_columns.has_value() || !in.admissible_additional_budget.has_value()) {
    d.reason = decision_reason::unknown_metadata;
    return d;
  }
  // A concatenated table cannot be indexed past cudf::size_type's range.
  if (*in.total_rows > CUDF_MAX_ROWS) {
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

  d.num_partitions      = 1;
  d.reason              = decision_reason::bypass_selected;
  d.prototype_activated = true;
  return d;
}

}  // namespace sirius::op::group_by_bypass
