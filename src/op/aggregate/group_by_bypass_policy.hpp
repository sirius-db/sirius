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

#include <cstddef>
#include <cstdint>
#include <optional>

namespace sirius::op::group_by_bypass {

/// Why a group-by partition count came out the way it did. Every rejection has its own value so a
/// log line identifies the gate that stopped the candidate.
enum class decision_reason : std::uint8_t {
  disabled,                ///< Prototype setting is off (the default). No work was done.
  multi_gpu,               ///< More than one admitted GPU; existing behaviour retained.
  already_one,             ///< The automatic policy already chose 1. Not a prototype activation.
  projected_input,         ///< Upstream is still running; only a projection is available.
  unknown_metadata,        ///< Row/type/batch metadata could not be read reliably.
  unsupported_residency,   ///< Input is not all GPU-resident in one memory space.
  unsupported_state,       ///< Key or aggregate partial-state types outside the v1 whitelist.
  unsupported_downstream,  ///< Downstream is not a bounded result-collection path.
  size_overflow,           ///< Checked arithmetic saturated, or cuDF's row limit was exceeded.
  insufficient_budget,     ///< The modelled requirement does not fit the admissible budget.
  bypass_selected,         ///< The prototype selected P=1.
};

[[nodiscard]] const char* reason_name(decision_reason reason) noexcept;

/// The largest row count `cudf::size_type` can index. A concatenated table above this cannot be
/// built at all, so the candidate is rejected rather than truncated.
inline constexpr std::uint64_t CUDF_MAX_ROWS = 2147483647ULL;

/// Per-term breakdown of the modelled *additional* bytes an unpartitioned merge would need.
/// Resident input bytes are deliberately absent: they are already-live allocations, already
/// charged against the memory space, and adding them here would double-count them.
struct memory_model {
  std::uint64_t concat_bytes     = 0;  ///< cudf::concatenate output (values + validity masks)
  std::uint64_t hash_set_bytes   = 0;  ///< cuco set built over the concatenated key rows
  std::uint64_t gather_map_bytes = 0;  ///< group gather map
  std::uint64_t sparse_agg_bytes = 0;  ///< per-row sparse aggregation results
  std::uint64_t output_bytes     = 0;  ///< dense keys + dense aggregates (worst case: R groups)
  std::uint64_t downstream_bytes = 0;  ///< retained copy along the supported collection path
  std::uint64_t materialization_bytes = 0;  ///< 0 in v1: residency in the target space is required
  std::uint64_t additional_needed     = 0;  ///< Sum of the terms above
  std::uint64_t headroom_bytes        = 0;  ///< Declared empirical margin on top
  std::uint64_t required_bytes        = 0;  ///< additional_needed + headroom_bytes
  bool overflowed = false;                  ///< A term saturated; the candidate must be rejected
};

/// Everything the policy needs, as plain values. Deliberately free of engine, cuDF and cuCascade
/// types so the decision is unit-testable without a GPU, a plan or a repository.
///
/// Unknown quantities are absent optionals, never zeros — "no nullable columns" and "nullability
/// could not be determined" must not collapse to the same input.
struct candidate_input {
  /// The count the existing automatic policy produced. Never recomputed here.
  int auto_num_partitions = 1;

  /// Admitted GPUs for this query (not the machine's physical GPU count).
  int num_admitted_gpus = 1;

  /// Upstream pipeline has finished: every partial batch has actually arrived.
  bool upstream_complete = false;

  /// All input batches are GPU-resident in one memory space, and that space is the merge target.
  bool single_gpu_resident = false;

  /// Group keys and aggregate partial states are all inside the v1 fixed-width whitelist.
  bool supported_state = false;

  /// The merge's downstream is a bounded result-collection path.
  bool supported_downstream = false;

  /// A PROJECTION/FILTER between merge and collector materializes a transformed copy.
  bool downstream_materializes_copy = false;

  /// Total partial rows over all batches. Absent when the metadata could not be read.
  std::optional<std::uint64_t> total_rows;

  /// Summed fixed widths of the group key columns, in bytes.
  std::optional<std::uint64_t> key_width_bytes;

  /// Summed fixed widths of the aggregate partial-state columns, in bytes.
  std::optional<std::uint64_t> agg_width_bytes;

  /// Columns that are nullable in at least one input batch, split by role. Each nullable column
  /// costs a validity mask in every term that materializes that column.
  std::optional<std::size_t> nullable_key_columns;
  std::optional<std::size_t> nullable_agg_columns;

  /// Bytes the target memory space can still hand out: capacity minus live allocations minus
  /// outstanding reservations. See docs/super-sirius/group-by-bypass.md — this value is already
  /// net of both, so callers
  /// must not subtract reserved bytes again.
  std::optional<std::uint64_t> admissible_additional_budget;

  /// Declared empirical margin, as a fraction of the modelled requirement.
  double headroom_fraction = 0.25;
};

/// The outcome. @ref num_partitions is what the caller must apply.
struct decision {
  int num_partitions     = 1;
  decision_reason reason = decision_reason::disabled;
  memory_model model{};
  /// True only when this prototype changed AUTO > 1 into 1. `already_one`
  /// is not an activation.
  bool prototype_activated = false;
  /// True when @ref model was actually computed (i.e. the candidate reached the budget check).
  bool model_evaluated = false;
};

/// Model the additional bytes an unpartitioned merge of @p in would allocate.
/// @pre The caller has already established that the metadata optionals are present.
[[nodiscard]] memory_model model_additional_bytes(const candidate_input& in) noexcept;

/// Apply the decision order from docs/super-sirius/group-by-bypass.md. Pure: no locks, no
/// allocation, no logging.
[[nodiscard]] decision decide(const candidate_input& in) noexcept;

}  // namespace sirius::op::group_by_bypass
