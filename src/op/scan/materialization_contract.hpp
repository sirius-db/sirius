/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "op/scan/owning_table_view.hpp"
#include "op/scan/parquet_batch_layout.hpp"
#include "op/scan/scan_filter_analysis.hpp"

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace sirius::op::scan {
// A program is query-local. A reader may report progress only for this exact binding.
struct materialization_program {
  uint64_t contract   = 0;
  uint64_t generation = 0;
  std::vector<cudf::data_type> output_types;
  std::size_t predicates = 0;
  bool needs_visibility  = false;
  // Required by raw importers; nullopt is reserved for the validated legacy bridge.
  std::optional<std::vector<cudf::data_type>> input_types;
  bool validate_output_schema = true;
  // cuDF data_type retains decimal scale, but not bound precision. Arrow imports
  // require this per-input-column binding for decimals (zero for other types).
  std::vector<int32_t> decimal_precisions;
};

struct row_origin {
  std::string identity;
  int64_t file_rows = 0;
};
struct source_position {
  std::size_t file;
  int64_t row;
};
struct row_domain {
  std::vector<row_origin> files;
  std::vector<batch_row_run> runs;
  std::optional<std::vector<source_position>> positions;
  void validate(int64_t rows) const;
  [[nodiscard]] bool available() const { return positions.has_value() || !runs.empty(); }
};

struct materialized_batch {
  owning_table_view table;
  uint64_t contract            = 0;
  uint64_t generation          = 0;
  int64_t rows                 = 0;
  bool logical_schema_complete = false;
  bool visibility_complete     = false;
  bool projection_complete     = false;
  bool row_count_carrier       = false;
  std::vector<predicate_progress> predicates;
  row_domain lineage;
};

// Each step consumes/updates the owned batch on the caller's task stream. Row-changing
// steps must transform lineage too, or discard it after its last semantic consumer.
using materialization_step = std::function<void(materialized_batch&)>;
struct materialization_steps {
  materialization_step mapping;
  materialization_step visibility;
  materialization_step residual;
  materialization_step projection;
};

void validate_materialized_batch(materialized_batch const&, materialization_program const&);
materialized_batch finish_materialization(materialized_batch,
                                          materialization_program const&,
                                          materialization_steps const&);
// A zero-column cudf table cannot carry a positive cardinality. Preserve it with an
// internal carrier until the consumer that understands zero projected columns runs.
void ensure_row_count_carrier(materialized_batch&,
                              ::cuda::stream_ref,
                              rmm::device_async_resource_ref);
}  // namespace sirius::op::scan
