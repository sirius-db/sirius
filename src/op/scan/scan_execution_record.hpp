/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "op/scan/scan_filter_analysis.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace sirius::op::scan {
enum class filter_state;
enum class execution_record_producer { parquet, legacy };
enum class row_position_state { none, identity, runs, per_row, lost };

// A binding retains the exact visibility input, not just a filename or an "applied" bit.
struct applied_visibility {
  std::string file;
  std::shared_ptr<void const> input;
};
using visibility_bindings = std::vector<applied_visibility>;

// The required bindings come from the certified split, independently of the
// producer's completion report. No per-row metadata is involved.
visibility_bindings remaining_visibility(std::span<applied_visibility const> required,
                                         std::span<applied_visibility const> applied,
                                         bool complete);
void finish_visibility(std::span<applied_visibility const> required,
                       visibility_bindings& applied,
                       bool& complete,
                       std::function<void(std::span<applied_visibility const>)> const& apply);

struct execution_requirements {
  std::span<applied_visibility const> visibility;
  // Query-owned immutable ordered layout; width alone cannot distinguish slots.
  std::shared_ptr<void const> layout;
};

// Query-local facts. Position storage is separately optional and never follows from coverage.
struct scan_execution_record {
  uint64_t contract                  = 0;
  uint64_t generation                = 0;
  execution_record_producer producer = execution_record_producer::legacy;
  bool authoritative                 = false;
  std::vector<predicate_progress> predicates;
  bool whole_static_complete   = false;
  bool logical_schema_complete = false;
  bool visibility_complete     = false;
  bool projection_complete     = false;
  std::optional<std::size_t> layout_width;
  std::shared_ptr<void const> layout;
  row_position_state positions = row_position_state::none;
  std::vector<applied_visibility> visibility_applied;
};

struct legacy_execution_facts {
  filter_state state;
  std::span<std::size_t const> answer_columns;
  bool answers_enforced = false;
  // Observed completion of the whole static conjunction, never a request latch
  // or an arbitrary row selection such as a dynamic filter.
  bool row_filtered = false;
};

scan_execution_record translate_legacy_execution(legacy_execution_facts const&,
                                                 residual_filter const& catalogue,
                                                 uint64_t contract,
                                                 uint64_t generation);
void validate_execution_record(scan_execution_record const&,
                               residual_filter const& catalogue,
                               uint64_t contract,
                               uint64_t generation,
                               execution_requirements const* required = nullptr);

struct execution_record_selection {
  scan_execution_record facts;
  bool used_producer         = false;
  bool shadow_mismatch       = false;
  bool shadow_coverage_added = false;
};
// Shadow discrepancies are observable but cannot change the old decision. An
// authoritative record must validate; falling back would hide missing work.
execution_record_selection select_execution_record(
  scan_execution_record const* producer,
  legacy_execution_facts const& legacy,
  residual_filter const& catalogue,
  uint64_t contract,
  uint64_t generation,
  execution_requirements const* required = nullptr);
}  // namespace sirius::op::scan
