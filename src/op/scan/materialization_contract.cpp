/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/materialization_contract.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace sirius::op::scan {
void row_domain::validate(int64_t rows) const
{
  if (rows < 0 || (positions && !runs.empty()))
    throw std::invalid_argument("invalid materialization row domain");
  for (auto const& file : files)
    if (file.identity.empty() || file.file_rows < 0)
      throw std::invalid_argument("row domain has no source identity");
  if (positions) {
    if (positions->size() != static_cast<std::size_t>(rows))
      throw std::invalid_argument("position count does not match materialized rows");
    for (auto const& p : *positions)
      if (p.file >= files.size() || p.row < 0 || p.row >= files[p.file].file_rows)
        throw std::invalid_argument("position is outside its original file");
  } else if (!runs.empty()) {
    int64_t offset = 0;
    for (auto const& r : runs) {
      if (r.file_index >= files.size() || r.batch_row_offset != offset || r.num_rows < 0 ||
          r.file_row_offset < 0 || r.file_row_offset > files[r.file_index].file_rows ||
          r.num_rows > files[r.file_index].file_rows - r.file_row_offset ||
          r.num_rows > rows - offset)
        throw std::invalid_argument("row run is outside its file or batch");
      offset += r.num_rows;
    }
    if (offset != rows) throw std::invalid_argument("row runs do not cover batch");
  }
}

void validate_materialized_batch(materialized_batch const& batch,
                                 materialization_program const& program)
{
  if (!program.contract || !program.generation || batch.contract != program.contract ||
      batch.generation != program.generation)
    throw std::invalid_argument("materialization binding mismatch");
  if (batch.rows < 0 || batch.rows > std::numeric_limits<cudf::size_type>::max() ||
      batch.table.num_rows() != batch.rows)
    throw std::invalid_argument("materialized cardinality mismatch");
  batch.lineage.validate(batch.rows);
  if (program.validate_output_schema && !program.input_types && !batch.projection_complete)
    throw std::invalid_argument("materialization requires a bound input schema");
  if (batch.logical_schema_complete && !batch.projection_complete && program.input_types) {
    auto const& expected = *program.input_types;
    auto width           = batch.table.view().num_columns();
    if (expected.empty()) {
      if (width != (batch.row_count_carrier ? 1 : 0))
        throw std::invalid_argument("unexpected zero-column input layout");
    } else {
      if (width != static_cast<cudf::size_type>(expected.size()) || batch.row_count_carrier)
        throw std::invalid_argument("materialized input width mismatch");
      for (std::size_t i = 0; i < expected.size(); ++i) {
        bool answer =
          std::any_of(batch.predicates.begin(), batch.predicates.end(), [&](auto const& p) {
            return p.stage == predicate_stage::answer && p.answer_slot == i;
          });
        auto type = answer ? cudf::data_type{cudf::type_id::BOOL8} : expected[i];
        if (batch.table.view().column(i).type() != type)
          throw std::invalid_argument("materialized input type mismatch");
      }
    }
  }
  std::vector<bool> seen(program.predicates);
  for (auto const& p : batch.predicates) {
    if (p.conjunct >= seen.size() || seen[p.conjunct])
      throw std::invalid_argument("unknown or duplicate materialized predicate");
    seen[p.conjunct] = true;
    if (p.stage != predicate_stage::untouched && p.stage != predicate_stage::pruning_only &&
        p.stage != predicate_stage::answer && p.stage != predicate_stage::enforced)
      throw std::invalid_argument("unknown predicate stage");
    if ((p.stage == predicate_stage::answer) != p.answer_slot.has_value())
      throw std::invalid_argument("invalid predicate answer report");
    if (p.answer_slot &&
        (*p.answer_slot >= static_cast<std::size_t>(batch.table.view().num_columns()) ||
         batch.table.view().column(*p.answer_slot).type().id() != cudf::type_id::BOOL8))
      throw std::invalid_argument("predicate answer is not a boolean slot");
  }
}

materialized_batch finish_materialization(materialized_batch batch,
                                          materialization_program const& program,
                                          materialization_steps const& steps)
{
  validate_materialized_batch(batch, program);
  auto run = [&](materialization_step const& step) {
    if (!step) throw std::invalid_argument("materialization has unfinished work without a step");
    step(batch);
    validate_materialized_batch(batch, program);
  };
  if (!batch.logical_schema_complete) {
    run(steps.mapping);
    batch.logical_schema_complete = true;
  }
  if (program.needs_visibility && !batch.visibility_complete) {
    if (!batch.lineage.available() && batch.rows)
      throw std::invalid_argument("visibility needs original row positions");
    run(steps.visibility);
  }
  batch.visibility_complete = true;
  std::size_t enforced      = 0;
  for (auto const& p : batch.predicates)
    enforced += p.stage == predicate_stage::enforced;
  if (enforced != program.predicates) {
    if (!steps.residual) throw std::invalid_argument("missing residual step");
    steps.residual(batch);
    batch.predicates.clear();
    for (std::size_t i = 0; i < program.predicates; ++i)
      batch.predicates.push_back({i, predicate_stage::enforced, {}});
    validate_materialized_batch(batch, program);
  }
  if (!batch.projection_complete) {
    if (!steps.projection) throw std::invalid_argument("missing projection step");
    steps.projection(batch);
    batch.projection_complete = true;
    validate_materialized_batch(batch, program);
  }
  if (program.validate_output_schema && program.output_types.empty()) {
    if (batch.table.view().num_columns() != (batch.row_count_carrier ? 1 : 0))
      throw std::invalid_argument("unexpected zero-column output layout");
  } else if (program.validate_output_schema) {
    if (batch.table.view().num_columns() !=
        static_cast<cudf::size_type>(program.output_types.size()))
      throw std::invalid_argument("materialized output width mismatch");
    for (std::size_t i = 0; i < program.output_types.size(); ++i)
      if (batch.table.view().column(i).type() != program.output_types[i])
        throw std::invalid_argument("materialized output type mismatch");
  }
  return batch;
}

void ensure_row_count_carrier(materialized_batch& batch,
                              ::cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  if (batch.rows < 0 || batch.rows > std::numeric_limits<cudf::size_type>::max())
    throw std::invalid_argument("materialized row count exceeds batch limit");
  if (batch.table.view().num_columns() == 0 && batch.rows) {
    cudf::numeric_scalar<int8_t> value(0, true, stream, mr);
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(cudf::make_column_from_scalar(value, batch.rows, stream, mr));
    batch.table             = owning_table_view(std::make_unique<cudf::table>(std::move(columns)));
    batch.row_count_carrier = true;
  }
}
}  // namespace sirius::op::scan
