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

#include "duckdb/common/optionally_owned_ptr.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "op/sirius_physical_operator.hpp"

namespace sirius::op {

/// Sirius-specific operator that wraps DuckDB source operators producing
/// pre-computed CPU-side data (COLUMN_DATA_SCAN, EMPTY_RESULT, DUMMY_SCAN).
/// Acts as a scan-pipeline sink whose corresponding cpu_source_task reads
/// the ColumnDataCollection and publishes data batches to the repository.
class sirius_physical_cpu_source : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::CPU_SOURCE;

  /// Construct from a COLUMN_DATA_SCAN operator that owns a ColumnDataCollection.
  sirius_physical_cpu_source(duckdb::vector<sirius::logical_type> types,
                             duckdb::idx_t estimated_cardinality,
                             duckdb::optionally_owned_ptr<duckdb::ColumnDataCollection> collection);

  /// Construct for EMPTY_RESULT / DUMMY_SCAN (no collection).
  sirius_physical_cpu_source(duckdb::vector<sirius::logical_type> types,
                             duckdb::idx_t estimated_cardinality,
                             bool produce_single_row);

  std::optional<task_creation_hint> get_next_task_hint(
    std::optional<std::size_t> /* downstream_request */ = std::nullopt) override
  {
    // A CPU source is single-shot: one task drains the collection. Gate task
    // creation on task_scheduled so concurrent calls from the task creator
    // cannot produce duplicate cpu_source_tasks (which would duplicate batches
    // into the repo). exhausted alone isn't sufficient because it flips only
    // after the task runs, leaving a window where a second task can be created.
    bool expected = false;
    if (!task_scheduled.compare_exchange_strong(expected, true)) { return std::nullopt; }
    // CPU source is single-shot: exactly one task drains the collection. The downstream request
    // is irrelevant — we cap at 1 regardless.
    return task_creation_hint{TaskCreationHint::READY, this, 1};
  }

  bool is_source() const override { return true; }

  //! The ColumnDataCollection to scan (may be nullptr for EMPTY_RESULT)
  duckdb::optionally_owned_ptr<duckdb::ColumnDataCollection> collection;
  //! Whether to produce a single empty row (DUMMY_SCAN behavior)
  bool produce_single_row{false};
  //! Whether a cpu_source_task has been handed out for this operator.
  std::atomic<bool> task_scheduled{false};
  //! Whether the source has been fully consumed
  std::atomic<bool> exhausted{false};
};

}  // namespace sirius::op
