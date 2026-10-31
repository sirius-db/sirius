/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/materialization_contract.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>

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

row_domain legacy_row_domain(row_lineage const& lineage, ::cuda::stream_ref stream)
{
  row_domain result;
  if (lineage.state == row_position_state::none || lineage.state == row_position_state::lost)
    return result;
  lineage.validate(lineage.output_length());
  for (auto const& file : lineage.domain->files)
    result.files.push_back({file.identity, file.rows});
  if (lineage.state == row_position_state::per_row) {
    result.positions.emplace();
    std::unordered_set<int64_t> seen;
    for (auto ordinal : lineage.ordinals_on(stream)) {
      if (!seen.insert(ordinal).second) throw std::invalid_argument("duplicate lineage ordinal");
      auto [file, row] = lineage.resolve(ordinal);
      result.positions->push_back({file, row});
    }
  } else {
    auto selections = lineage.runs;
    if (lineage.state == row_position_state::identity && lineage.output_length())
      selections = {{0, lineage.output_length()}};
    int64_t output = 0;
    for (auto [first, count] : selections) {
      int64_t offset = 0;
      for (auto const& run : lineage.domain->file_runs) {
        auto start = std::max(first, offset);
        auto end   = std::min(first + count, offset + run.count);
        if (start < end) {
          result.runs.push_back({lineage.domain->files[run.file].identity,
                                 run.first + start - offset,
                                 output,
                                 end - start,
                                 run.file});
          output += end - start;
        }
        offset += run.count;
      }
    }
  }
  result.validate(lineage.output_length());
  return result;
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
  if (batch.lineage_record) {
    if (batch.lineage.available())
      throw std::invalid_argument("canonical and legacy lineage cannot have independent authority");
    batch.lineage_record->validate(batch.rows);
  }
  if (program.catalogue) {
    if (program.catalogue->contract() != program.contract ||
        program.catalogue->generation() != program.generation ||
        program.catalogue->size() != program.predicates)
      throw std::invalid_argument("materialization predicate catalogue binding mismatch");
    program.catalogue->residual().validate_progress(batch.predicates);
  }
  if (program.required_visibility)
    remaining_visibility(
      *program.required_visibility, batch.visibility_applied, batch.visibility_complete);
  auto const& layout = batch.projection_complete ? program.output_layout : program.input_layout;
  if (layout && batch.layout != layout)
    throw std::invalid_argument("materialized ordered layout mismatch");
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
        bool answer = std::any_of(batch.predicates.begin(),
                                  batch.predicates.end(),
                                  [&](auto const& p) { return p.answer_slot == i; });
        auto type   = answer ? cudf::data_type{cudf::type_id::BOOL8} : expected[i];
        if (batch.table.view().column(i).type() != type)
          throw std::invalid_argument("materialized input type mismatch");
      }
    }
  }
  std::vector<bool> seen(batch.predicates.empty() ? 0 : program.predicates);
  for (auto const& p : batch.predicates) {
    if (p.conjunct >= seen.size() || seen[p.conjunct])
      throw std::invalid_argument("unknown or duplicate materialized predicate");
    seen[p.conjunct] = true;
    if (p.stage != predicate_stage::untouched && p.stage != predicate_stage::pruning_only &&
        p.stage != predicate_stage::answer && p.stage != predicate_stage::enforced)
      throw std::invalid_argument("unknown predicate stage");
    if ((p.stage == predicate_stage::answer && !p.answer_slot) ||
        (p.answer_slot && p.stage != predicate_stage::answer &&
         p.stage != predicate_stage::enforced))
      throw std::invalid_argument("invalid predicate answer report");
    if (p.answer_slot &&
        (*p.answer_slot >= static_cast<std::size_t>(batch.table.view().num_columns()) ||
         batch.table.view().column(*p.answer_slot).type().id() != cudf::type_id::BOOL8))
      throw std::invalid_argument("predicate answer is not a boolean slot");
  }
  if (program.validate_output_schema && batch.projection_complete) {
    auto expected_width = program.output_types.empty() && batch.row_count_carrier
                            ? std::size_t{1}
                            : program.output_types.size();
    if (batch.table.n_columns() != expected_width)
      throw std::invalid_argument("materialized output width mismatch");
    for (std::size_t i = 0; i < program.output_types.size(); ++i)
      if (batch.table.view().column(i).type() != program.output_types[i])
        throw std::invalid_argument("materialized output type mismatch");
  }
}

void finish_materialization_in_place(materialized_batch& batch,
                                     materialization_program const& program,
                                     materialization_steps const& steps)
{
  validate_materialized_batch(batch, program);
  auto positions_available = [&] {
    if (batch.lineage_record)
      return batch.lineage_record->state == row_position_state::identity ||
             batch.lineage_record->state == row_position_state::runs ||
             batch.lineage_record->state == row_position_state::per_row;
    return batch.lineage.available();
  };
  auto run = [&](materialization_step const& step) {
    if (!step) throw std::invalid_argument("materialization has unfinished work without a step");
    step(batch);
    validate_materialized_batch(batch, program);
  };
  if (!batch.logical_schema_complete) {
    run(steps.mapping);
    batch.logical_schema_complete = true;
  }
  if (program.required_visibility) {
    finish_visibility(
      *program.required_visibility,
      batch.visibility_applied,
      batch.visibility_complete,
      [&](std::span<applied_visibility const> pending) {
        if (!positions_available() && batch.rows &&
            (!steps.visibility_requires_positions || steps.visibility_requires_positions(pending)))
          throw std::invalid_argument("visibility needs original row positions");
        if (!steps.bound_visibility) throw std::invalid_argument("missing bound visibility step");
        steps.bound_visibility(batch, pending);
      });
    validate_materialized_batch(batch, program);
  } else if (program.needs_visibility && !batch.visibility_complete) {
    if (!positions_available() && batch.rows)
      throw std::invalid_argument("visibility needs original row positions");
    run(steps.visibility);
  }
  batch.visibility_complete = true;
  std::size_t enforced      = 0;
  for (auto const& p : batch.predicates)
    enforced += p.stage == predicate_stage::enforced;
  if (!batch.whole_static_complete && enforced != program.predicates) {
    if (!steps.residual) throw std::invalid_argument("missing residual step");
    steps.residual(batch);
    batch.whole_static_complete = true;
    // Retain answer layout until projection consumes it, even after enforcing it.
    std::vector<predicate_progress> completed;
    completed.reserve(program.predicates);
    for (std::size_t i = 0; i < program.predicates; ++i)
      completed.push_back({i, predicate_stage::enforced, {}});
    for (auto const& p : batch.predicates) {
      if (p.conjunct >= completed.size())
        throw std::invalid_argument("residual step reported an unknown predicate");
      completed[p.conjunct].answer_slot = p.answer_slot;
    }
    batch.predicates = std::move(completed);
    validate_materialized_batch(batch, program);
  }
  batch.whole_static_complete = true;
  if (!batch.projection_complete) {
    if (!steps.projection) throw std::invalid_argument("missing projection step");
    steps.projection(batch);
    batch.projection_complete = true;
    batch.layout              = program.output_layout;
    for (auto& p : batch.predicates) {
      p.stage = predicate_stage::enforced;
      p.answer_slot.reset();
    }
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
}

materialized_batch finish_materialization(materialized_batch batch,
                                          materialization_program const& program,
                                          materialization_steps const& steps)
{
  finish_materialization_in_place(batch, program, steps);
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
