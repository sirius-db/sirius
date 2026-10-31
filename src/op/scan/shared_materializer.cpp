/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/shared_materializer.hpp"

#include "op/scan/detail/read_retirement.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <shared_mutex>

namespace sirius::op::scan {
namespace {
std::shared_mutex observer_mutex;
std::map<uint64_t, step_observer*> observers;
uint64_t next_observer = 0;
std::atomic<bool> has_observers{false};
thread_local bool notifying = false;
template <class F>
void observe(F&& callback) noexcept
{
  if (notifying || !has_observers.load(std::memory_order_acquire)) return;
  try {
    std::shared_lock lock(observer_mutex);
    if (observers.empty()) return;
    notifying = true;
    try {
      callback(*observers.rbegin()->second);
    } catch (...) {
    }
    notifying = false;
  } catch (...) {
    notifying = false;
  }
}
bool available(row_lineage const& lineage)
{
  return lineage.state == row_position_state::identity ||
         lineage.state == row_position_state::runs || lineage.state == row_position_state::per_row;
}
struct shared_state {
  prepared_split split;
  materializer_inputs inputs;
  decode_fn decode;
  decode_result decoded;
  materialized_batch batch;
};
}  // namespace

std::size_t shared_materialization_workspace_bytes(std::size_t rows,
                                                   std::size_t payload,
                                                   std::size_t deletes,
                                                   std::size_t intervals) noexcept
{
  auto const max    = std::numeric_limits<std::size_t>::max();
  std::size_t bytes = sizeof(int);
  for (auto [count, width] : {std::pair{payload, std::size_t{2}},
                              {rows, std::size_t{64}},
                              {deletes, std::size_t{8}},
                              {intervals, std::size_t{128}}}) {
    if (count > (max - bytes) / width) return max;
    bytes += count * width;
  }
  return bytes;
}

scoped_step_observer::scoped_step_observer(step_observer& observer)
{
  auto flag = std::getenv("SIRIUS_ENABLE_TEST_OPTIONS");
  if (!flag || std::string_view(flag) != "1" || notifying)
    throw std::invalid_argument("scan step observer requires quiescent test installation");
  std::unique_lock lock(observer_mutex);
  registration_ = ++next_observer;
  observers.emplace(registration_, &observer);
  has_observers.store(true, std::memory_order_release);
}
scoped_step_observer::~scoped_step_observer() noexcept
{
  std::unique_lock lock(observer_mutex);
  observers.erase(registration_);
  has_observers.store(!observers.empty(), std::memory_order_release);
}

void apply_materialization_visibility(
  std::span<applied_visibility const> required,
  visibility_bindings& applied,
  bool& complete,
  std::function<void(std::span<applied_visibility const>)> const& apply,
  uint64_t contract,
  uint64_t attempt,
  uint64_t logical_input)
{
  finish_visibility(required, applied, complete, [&](auto pending) {
    if (!apply) throw std::invalid_argument("missing bound visibility step");
    apply(pending);
    step_identity id{contract, attempt, logical_input};
    for (auto const& binding : pending)
      observe([&](auto& observer) { observer.on_visibility(id, binding); });
  });
}

void apply_materialization_residual(materialized_batch& batch,
                                    materialization_program const& program,
                                    semantic_options const& options,
                                    ::cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  if (!program.catalogue) throw std::invalid_argument("residual requires a bound catalogue");
  struct selection_state {
    owning_table_view input;
    std::unique_ptr<cudf::table> output;
    std::unique_ptr<sirius::ast::node> residual;
    std::unique_ptr<expression_evaluator> evaluator;
    std::unique_ptr<cudf::column> survivors;
    std::shared_ptr<row_lineage const> lineage;
  };
  auto state      = std::make_unique<selection_state>();
  state->residual = program.catalogue->residual().against_progress(batch.predicates);
  if (!state->residual) return;
  state->input        = std::move(batch.table);
  state->lineage      = batch.lineage_record;
  auto const unjoined = detail::prepare_unjoined_failure();
  try {
    state->evaluator =
      std::make_unique<expression_evaluator>(state->residual.get(),
                                             mr,
                                             stream,
                                             options.policy,
                                             expression_evaluator::default_min_ast_size,
                                             options.like_swar_fastpath,
                                             options.like_cache);
    bool const preserve_lineage = state->lineage && available(*state->lineage);
    if (options.survivors || preserve_lineage) {
      auto columns = options.filter_projection;
      if (columns.empty()) {
        columns.resize(state->input.n_columns());
        std::iota(columns.begin(), columns.end(), cudf::size_type{0});
      }
      state->output =
        state->evaluator->select_with_survivors(state->input.view(), columns, state->survivors);
      if (preserve_lineage)
        batch.lineage_record = std::make_shared<row_lineage const>(
          state->lineage->select(state->survivors->view(), stream, mr));
    } else {
      state->output = options.filter_projection.empty()
                        ? state->evaluator->select(state->input.view())
                        : state->evaluator->select(state->input.view(), options.filter_projection);
    }
    detail::check_read_checkpoint(detail::read_checkpoint::PAYLOAD_GATHER);
    detail::check_read_checkpoint(detail::read_checkpoint::READER_EVENT);
    state->input.record_reader_event(stream);
    batch.table = owning_table_view{std::move(state->output)};
    batch.rows  = batch.table.num_rows();
    if (options.survivors) *options.survivors = std::move(state->survivors);
    if (!options.filter_projection.empty())
      for (auto& p : batch.predicates)
        p.answer_slot.reset();
    step_identity const id{program.contract, options.attempt, options.logical_input};
    for (std::size_t term = 0; term < program.predicates; ++term)
      if (std::none_of(batch.predicates.begin(), batch.predicates.end(), [&](auto const& p) {
            return p.conjunct == term && p.stage == predicate_stage::enforced;
          }))
        observe([&](auto& observer) { observer.on_residual(id, term); });
  } catch (...) {
    detail::retire_reads(state, stream, unjoined);
    throw;
  }
}

void apply_materialization_projection(materialized_batch& batch,
                                      materialization_program const& program,
                                      scan_plan const& plan,
                                      std::vector<std::string> const& partitions,
                                      semantic_options const& options,
                                      ::cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  if (program.validate_output_schema && program.output_types.empty()) {
    if (batch.table.n_columns()) {
      std::array<std::size_t, 1> carrier{0};
      batch.table             = select_scan_columns(std::move(batch.table), carrier, stream, mr);
      batch.row_count_carrier = true;
    }
  } else if (plan.has_partitions()) {
    batch.table = assemble_scan_output(plan, std::move(batch.table), partitions, stream, mr);
  } else if (!plan.output_layout.empty()) {
    std::vector<std::size_t> positions;
    bool const elide = options.elided.size() < plan.output_layout.size();
    for (std::size_t i = 0; i < plan.output_layout.size(); ++i)
      if (!elide || std::ranges::find(options.elided, i) == options.elided.end())
        positions.push_back(plan.output_layout[i].idx);
    batch.table = select_scan_columns(std::move(batch.table), positions, stream, mr);
  }
  batch.rows = batch.table.num_rows();
  step_identity const id{program.contract, options.attempt, options.logical_input};
  observe([&](auto& observer) { observer.on_assemble(id); });
}

decode_result parquet_cudf_decode(prepared_split const& split,
                                  materializer_inputs const& inputs,
                                  cucascade::memory::memory_space const& space,
                                  ::cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  if (!inputs.reader || !inputs.reader->decode)
    throw std::invalid_argument("default Parquet decoder requires an owned reader task");
  return inputs.reader->decode(split, inputs, space, stream, mr);
}

decode_result materialize_shared(prepared_split const& split,
                                 materializer_inputs const& inputs,
                                 decode_fn const& decode,
                                 cucascade::memory::memory_space const& space,
                                 ::cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  if (!decode || !inputs.mapping.program || !inputs.mapping.evidence || !split.plan)
    throw std::invalid_argument("shared materialization requires certified input and layout");
  if ((!split.slices.empty() || !split.dependencies.empty()) && !split.info &&
      !split.metadata_owner)
    throw std::invalid_argument("prepared metadata views require a retained input owner");
  auto const& program = *inputs.mapping.program;
  auto const& proof   = *inputs.mapping.evidence;
  if (proof.contract != program.contract || proof.generation != program.generation ||
      proof.layout != program.input_layout || proof.plan != split.plan ||
      proof.normalization != inputs.mapping.normalization)
    throw std::invalid_argument("shared mapping certification binding mismatch");
  if (inputs.mapping.normalization &&
      (inputs.mapping.normalization->program().contract != program.contract ||
       inputs.mapping.normalization->program().generation != program.generation))
    throw std::invalid_argument("shared normalization program binding mismatch");
  if (!program.catalogue || program.catalogue->contract() != program.contract ||
      program.catalogue->generation() != program.generation ||
      program.catalogue->size() != program.predicates || !program.input_layout ||
      !program.output_layout)
    throw std::invalid_argument(
      "shared materialization requires bound predicate catalogue and layouts");
  if (inputs.coverage.catalogue && inputs.coverage.catalogue != program.catalogue)
    throw std::invalid_argument("reader coverage belongs to another predicate catalogue");
  std::vector<bool> requested(program.predicates, false);
  for (auto term : inputs.coverage.reader_requested) {
    if (term >= requested.size() || requested[term])
      throw std::invalid_argument("invalid requested reader conjunct");
    requested[term] = true;
  }
  visibility_bindings required;
  for (auto const& visibility : inputs.visibility) {
    if (!visibility.owner || visibility.binding.input != visibility.owner ||
        visibility.binding.file != visibility.owner->data_file)
      throw std::invalid_argument("shared visibility input has mismatched immutable owner");
    required.push_back(visibility.binding);
  }
  remaining_visibility(required, {}, false);  // Reject duplicate requirements before decode.
  if (program.required_visibility) {
    if (program.required_visibility->size() != required.size())
      throw std::invalid_argument("shared visibility requirements mismatch");
  }
  if (program.required_visibility) {
    for (std::size_t i = 0; i < required.size(); ++i)
      if (program.required_visibility->at(i).file != required[i].file ||
          program.required_visibility->at(i).input != required[i].input)
        throw std::invalid_argument("shared visibility requirements mismatch");
  } else if (!required.empty()) {
    throw std::invalid_argument("shared program omitted visibility requirements");
  }
  if (split.stop.stop_requested()) throw std::runtime_error("scan materialization cancelled");
  auto state          = std::make_unique<shared_state>();
  auto const unjoined = detail::prepare_unjoined_failure();
  state->split        = split;
  state->inputs       = inputs;
  state->decode       = decode;
  try {
    state->decoded  = state->decode(state->split, state->inputs, space, stream, mr);
    auto& delivered = state->decoded.batch;
    auto& lineage   = state->decoded.lineage;
    auto rows       = delivered.table.num_rows();
    scan_execution_record facts;
    if (delivered.execution && delivered.execution->authoritative) {
      validate_execution_record(
        *delivered.execution, program.catalogue->residual(), program.contract, program.generation);
      facts = *delivered.execution;
    } else {
      legacy_execution_facts legacy{
        delivered.state, delivered.predicate_columns, delivered.predicates_enforced};
      facts =
        inputs.phase == materialization_phase::decoded
          ? select_execution_record(
              nullptr, legacy, program.catalogue->residual(), program.contract, program.generation)
              .facts
          : translate_legacy_execution(
              legacy, program.catalogue->residual(), program.contract, program.generation);
    }
    auto pending_visibility =
      remaining_visibility(required, facts.visibility_applied, facts.visibility_complete);
    bool const position_consumer =
      inputs.needs_lineage ||
      std::any_of(pending_visibility.begin(), pending_visibility.end(), [&](auto const& binding) {
        return std::any_of(
          inputs.visibility.begin(), inputs.visibility.end(), [&](auto const& input) {
            return input.binding.input == binding.input && !input.owner->positions.empty();
          });
      });
    if (inputs.phase == materialization_phase::complete && delivered.row_positions &&
        position_consumer) {
      if (lineage.state != row_position_state::none)
        throw std::invalid_argument("decoder returned two canonical position representations");
      lineage = consume_reader_positions(std::move(delivered.row_positions), stream, mr);
      if (split.input_domain && *lineage.domain == *split.input_domain)
        lineage.domain = split.input_domain;
    }
    if (inputs.phase == materialization_phase::complete && !position_consumer &&
        delivered.row_positions) {
      delivered.row_positions.reset();
      facts.positions = lineage.state;
      if (delivered.execution) delivered.execution->positions = lineage.state;
    }
    lineage.validate_on(rows, stream, mr);
    if (split.input_domain && lineage.domain && lineage.domain != split.input_domain)
      throw std::invalid_argument("decoder changed its immutable input domain");
    if (facts.projection_complete && !facts.logical_schema_complete)
      throw std::invalid_argument("decoder assembled output before logical normalization");
    if (facts.layout_width &&
        *facts.layout_width != static_cast<std::size_t>(delivered.table.n_columns()))
      throw std::invalid_argument("decoder reported a different layout width");
    if (delivered.execution && delivered.execution->authoritative && !facts.layout)
      throw std::invalid_argument("authoritative decoder omitted its ordered layout");
    if (facts.layout &&
        facts.layout != (facts.projection_complete ? program.output_layout : program.input_layout))
      throw std::invalid_argument("decoder returned a foreign ordered layout");
    if (inputs.validation == validation_policy::qualified_lineage && !inputs.reader &&
        delivered.execution && delivered.execution->authoritative &&
        delivered.execution->positions != lineage.state)
      throw std::invalid_argument("decoder position facts disagree with lineage");
    // The deferred Parquet entry leaves legacy state/cardinality checks at the
    // adapter's original boundary. Empty sets need no positional proof.
    if (inputs.phase == materialization_phase::complete &&
        inputs.validation == validation_policy::legacy_iceberg &&
        delivered.state != filter_state::UNFILTERED) {
      bool const pending_positions =
        std::any_of(pending_visibility.begin(), pending_visibility.end(), [&](auto const& binding) {
          return std::any_of(
            inputs.visibility.begin(), inputs.visibility.end(), [&](auto const& input) {
              return input.binding.input == binding.input && !input.owner->positions.empty();
            });
        });
      if (pending_positions)
        throw sirius::internal_exception(
          inputs.diagnostic_context +
          ": decode returned a filtered table, so row positions no longer identify file rows");
    }
    if (inputs.phase == materialization_phase::complete && lineage.domain) {
      for (auto const& binding : pending_visibility) {
        auto file = std::find_if(lineage.domain->files.begin(),
                                 lineage.domain->files.end(),
                                 [&](auto const& file) { return file.identity == binding.file; });
        if (file == lineage.domain->files.end())
          throw std::invalid_argument("visibility file is outside decoder input domain");
        for (auto const& input : inputs.visibility)
          if (input.binding.input == binding.input) input.owner->validate_file_bounds(file->rows);
      }
    }
    if (required.empty()) {
      facts.visibility_complete = true;
      if (delivered.execution) delivered.execution->visibility_complete = true;
    }
    if (inputs.phase == materialization_phase::decoded && facts.logical_schema_complete) {
      if (program.input_types || program.validate_output_schema) {
        materialized_batch checked;
        checked.contract                = program.contract;
        checked.generation              = program.generation;
        checked.rows                    = rows;
        checked.logical_schema_complete = true;
        checked.projection_complete     = facts.projection_complete;
        checked.layout                  = facts.layout                ? facts.layout
                                          : facts.projection_complete ? program.output_layout
                                                                      : program.input_layout;
        checked.predicates              = facts.predicates;
        checked.visibility_applied      = facts.visibility_applied;
        checked.visibility_complete     = facts.visibility_complete;
        checked.row_count_carrier =
          delivered.table.n_columns() == 1 &&
          (facts.projection_complete ? program.output_types.empty()
                                     : program.input_types && program.input_types->empty());
        checked.table = std::move(delivered.table);
        try {
          validate_materialized_batch(checked, program);
        } catch (...) {
          delivered.table = std::move(checked.table);
          throw;
        }
        delivered.table = std::move(checked.table);
      }
      delivered.required_visibility = program.required_visibility;
      if (!delivered.execution) {
        facts.layout = facts.projection_complete ? program.output_layout : program.input_layout;
        facts.layout_width  = delivered.table.n_columns();
        facts.positions     = lineage.state;
        delivered.execution = std::move(facts);
      }
      // Replacement decoders may enqueue reads of borrowed staging. The default
      // reader owns its completion protocol; do not add a wait to that path.
      if (!inputs.reader) {
        detail::check_read_checkpoint(detail::read_checkpoint::BATCH);
        detail::retire_reads(state, stream, unjoined);
      }
      if (split.stop.stop_requested()) throw std::runtime_error("scan materialization cancelled");
      return std::move(state->decoded);
    }
    auto& batch      = state->batch;
    batch.table      = std::move(delivered.table);
    batch.contract   = program.contract;
    batch.generation = program.generation;
    batch.rows       = rows;
    batch.row_count_carrier =
      program.input_types && program.input_types->empty() && batch.table.n_columns() == 1;
    batch.logical_schema_complete = facts.logical_schema_complete;
    batch.projection_complete     = facts.projection_complete;
    batch.whole_static_complete   = facts.whole_static_complete;
    batch.predicates              = facts.predicates;
    batch.layout                  = facts.layout                ? facts.layout
                                    : facts.projection_complete ? program.output_layout
                                                                : program.input_layout;
    batch.visibility_applied      = facts.visibility_applied;
    batch.visibility_complete     = facts.visibility_complete;
    batch.lineage_record          = std::make_shared<row_lineage const>(lineage);
    step_identity const id{program.contract, split.attempt, split.logical_input};
    materialization_steps steps;
    steps.visibility_requires_positions = [&](std::span<applied_visibility const> pending) {
      return std::any_of(pending.begin(), pending.end(), [&](auto const& binding) {
        return std::any_of(
          inputs.visibility.begin(), inputs.visibility.end(), [&](auto const& input) {
            return input.binding.input == binding.input && !input.owner->positions.empty();
          });
      });
    };
    steps.mapping = [&](materialized_batch& b) {
      if (inputs.mapping.normalization) {
        auto mapped = inputs.mapping.normalization->apply(std::move(b.table), b.rows, stream, mr);
        b.table     = std::move(mapped.table);
      } else if (split.plan->virtual_columns.empty()) {
        throw std::invalid_argument("decoder needs mapping without certified normalization");
      }
      append_lineage_virtual_columns(
        b.table, lineage, *split.plan, split.virtual_file_indices, stream, mr);
      b.logical_schema_complete = true;
    };
    if (inputs.phase == materialization_phase::decoded) {
      steps.mapping(batch);
      validate_materialized_batch(batch, program);
      delivered.table               = std::move(batch.table);
      facts.logical_schema_complete = true;
      facts.layout                  = program.input_layout;
      facts.layout_width            = delivered.table.n_columns();
      delivered.execution           = std::move(facts);
      delivered.required_visibility = program.required_visibility;
      delivered.lineage_record      = batch.lineage_record;
      detail::retire_reads(state, stream, unjoined);
      if (split.stop.stop_requested()) throw std::runtime_error("scan materialization cancelled");
      return std::move(state->decoded);
    }
    steps.bound_visibility = [&](materialized_batch& b,
                                 std::span<applied_visibility const> pending) {
      if (!steps.visibility_requires_positions(pending)) {
        for (auto const& binding : pending)
          observe([&](auto& observer) { observer.on_visibility(id, binding); });
        return;
      }
      if (b.rows && !available(lineage))
        throw std::invalid_argument(inputs.diagnostic_context +
                                    ": row lineage required for visibility is missing or invalid");
      apply_lineage_visibility(
        b.table,
        lineage,
        [&](std::string const& file) -> std::span<int64_t const> {
          for (auto const& binding : pending)
            if (binding.file == file)
              for (auto const& input : inputs.visibility)
                if (input.binding.input == binding.input) return input.owner->positions;
          return {};
        },
        stream,
        mr);
      b.rows           = b.table.num_rows();
      b.lineage_record = std::make_shared<row_lineage const>(lineage);
      for (auto const& binding : pending)
        observe([&](auto& observer) { observer.on_visibility(id, binding); });
    };
    semantic_options options;
    options.policy             = inputs.expression_policy;
    options.like_swar_fastpath = inputs.like_swar_fastpath;
    options.like_cache         = inputs.like_cache;
    options.attempt            = split.attempt;
    options.logical_input      = split.logical_input;
    steps.residual             = [&](materialized_batch& b) {
      apply_materialization_residual(b, program, options, stream, mr);
      if (b.lineage_record) lineage = *b.lineage_record;
    };
    steps.projection = [&](materialized_batch& b) {
      apply_materialization_projection(
        b, program, *split.plan, split.partition_values, options, stream, mr);
    };
    finish_materialization_in_place(batch, program, steps);
    if (split.stop.stop_requested()) throw std::runtime_error("scan materialization cancelled");
    scan_execution_record completed;
    completed.producer                = execution_record_producer::own_decoder;
    completed.contract                = program.contract;
    completed.generation              = program.generation;
    completed.authoritative           = true;
    completed.logical_schema_complete = true;
    completed.visibility_complete     = true;
    completed.whole_static_complete   = true;
    completed.projection_complete     = true;
    completed.layout                  = program.output_layout;
    completed.layout_width            = state->batch.table.n_columns();
    completed.visibility_applied      = state->batch.visibility_applied;
    completed.positions               = lineage.state;
    delivered.table                   = std::move(state->batch.table);
    delivered.state                   = filter_state::ROW_FILTERED_AND_PROJECTED;
    delivered.predicate_columns.clear();
    delivered.predicates_enforced = false;
    delivered.execution           = std::move(completed);
    delivered.required_visibility = program.required_visibility;
    delivered.lineage_record      = batch.lineage_record;
    delivered.attempt             = split.attempt;
    delivered.logical_input       = split.logical_input;
    detail::check_read_checkpoint(detail::read_checkpoint::BATCH);
    detail::retire_reads(state, stream, unjoined);
    return std::move(state->decoded);
  } catch (...) {
    if (state) detail::retire_reads(state, stream, unjoined);
    throw;
  }
}
}  // namespace sirius::op::scan
