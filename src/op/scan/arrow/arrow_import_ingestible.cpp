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

#include "op/scan/arrow/arrow_import_ingestible.hpp"

#include "expression_evaluator/expression_evaluator.hpp"
#include "op/scan/detail/read_retirement.hpp"
#include "op/scan/scan_plan.hpp"
#include "op/scan/scan_utils.hpp"

#include <cudf/column/column_factories.hpp>

#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/exception.hpp>

#include <limits>

namespace sirius::op::scan {
namespace scan = sirius::op::scan;
namespace sm   = scan_manager;
namespace {
struct arrow_table_info final : ingestible_table_info {
  std::vector<std::string> names;
  std::string name;
  std::span<std::string const> column_names() const override { return names; }
  std::span<std::string const> file_paths() const override { return {}; }
  std::string display_name() const override { return name; }
  bool allows_cached_rows() const override { return false; }
};
struct input_claim {
  std::shared_ptr<sm::preparation_ledger> ledger;
  std::shared_ptr<arrow_input_gate> gate;
  sm::unit_key key;
  sm::w_permit permit;
  bool acquired = false;
  ~input_claim()
  {
    if (acquired) {
      ledger->retire_unit(key);
      permit.release();
      gate->active = false;
      gate->wake();
    }
  }
};
struct source_state {
  std::atomic<bool> eos{false}, stopped{false}, accepted{false};
  uint64_t eos_sequence = 0;
  bool flushed          = false;
};
class import_validator final : public split_validator {
 public:
  explicit import_validator(arrow_import_binding const& binding,
                            std::shared_ptr<source_state> state)
    : arrow_(owned_arrow_split_validator(binding.program)),
      identity_(binding.read_identity),
      inspect_(binding.validate_provenance),
      state_(std::move(state))
  {
  }
  bool accepts(scan_info const& input) const noexcept override { return arrow_->accepts(input); }
  split_validation_evidence inspect(split_validation_context const& context,
                                    scan_info const& input) const override
  {
    auto result          = arrow_->inspect(context, input);
    auto const& arrow    = dynamic_cast<owned_arrow_scan_info const&>(input);
    auto const& identity = arrow.read_identity();
    if (identity != identity_ || identity.owner_before(identity_) ||
        identity_.owner_before(identity))
      throw certificate_incomplete(context.contract, {}, "Arrow input belongs to another read");
    if (arrow.rootless() && (!state_->eos || state_->accepted || state_->stopped ||
                             arrow.sequence() != state_->eos_sequence))
      throw certificate_incomplete(context.contract, {}, "Arrow empty source has no normal EOS");
    if (inspect_) inspect_(arrow);
    return result;
  }

 private:
  std::shared_ptr<split_validator const> arrow_;
  std::shared_ptr<void const> identity_;
  std::function<void(owned_arrow_scan_info const&)> inspect_;
  std::shared_ptr<source_state> state_;
};
class import_coalescer final : public batch_coalescer {
 public:
  import_coalescer(arrow_import_binding binding, std::shared_ptr<source_state> state, size_t bytes)
    : binding_(std::move(binding)), state_(std::move(state)), bytes_(bytes)
  {
  }
  size_t target_batch_bytes() const noexcept override { return bytes_; }
  cursor_step advance(scan_info& input, size_t&, size_t) override
  {
    return {dynamic_cast<owned_arrow_scan_info&>(input).coalesced_copy(), true};
  }
  std::vector<std::unique_ptr<scan_info>> push(std::unique_ptr<scan_info>) override
  {
    throw std::logic_error("Arrow import requires bounded coalescing");
  }
  std::vector<std::unique_ptr<scan_info>> flush() override
  {
    std::vector<std::unique_ptr<scan_info>> result;
    if (!state_->flushed && state_->eos && !state_->stopped && !state_->accepted &&
        binding_.emit_empty_source) {
      result.push_back(std::make_unique<owned_arrow_scan_info>(
        binding_.program, state_->eos_sequence, binding_.profiles, binding_.read_identity));
    }
    state_->flushed = true;
    return result;
  }

 private:
  arrow_import_binding binding_;
  std::shared_ptr<source_state> state_;
  size_t bytes_;
};
}  // namespace
struct arrow_import_ingestible::impl {
  arrow_import_binding binding;
  std::shared_ptr<arrow_batch_producer> producer;
  arrow_producer_limits limits;
  arrow_table_info info;
  std::shared_ptr<split_validator const> validator;
  std::shared_ptr<sm::preparation_ledger> ledger;
  std::shared_ptr<source_state> state = std::make_shared<source_state>();
  std::optional<uint64_t> pending_identity;
  uint64_t sequence = 0, bytes = 0, temporary_limit = 0;
  impl(arrow_import_binding b, std::shared_ptr<arrow_batch_producer> p)
    : binding(std::move(b)), producer(std::move(p))
  {
    if (!producer || !binding.read_identity || !binding.profiles || !binding.admission ||
        !binding.admission->decision.deferred || !binding.program.input_types ||
        !binding.program.contract || !binding.program.generation ||
        binding.program.needs_visibility)
      throw std::invalid_argument("Arrow driver requires a qualified bound read and admission");
    if (binding.program.predicates != (binding.predicate ? 1 : 0) ||
        binding.program.output_types.size() != binding.output_columns.size() ||
        binding.column_names.size() != binding.program.input_types->size())
      throw std::invalid_argument("Arrow driver program/layout mismatch");
    for (size_t i = 0; i < binding.output_columns.size(); ++i)
      if (binding.output_columns[i] >= binding.program.input_types->size() ||
          binding.program.output_types[i] !=
            binding.program.input_types->at(binding.output_columns[i]))
        throw std::invalid_argument("Arrow driver output type mismatch");
    limits = producer->limits();
    if (limits.max_rows < 0 || limits.max_rows > std::numeric_limits<cudf::size_type>::max() ||
        limits.envelope.temporary.empty())
      throw std::invalid_argument("Arrow producer requires finite input bounds");
    for (auto n : limits.envelope.temporary) {
      if (n > UINT64_MAX - bytes) throw std::overflow_error("Arrow envelope overflow");
      bytes += n;
    }
    temporary_limit =
      limits.envelope.temporary_charge(binding.admission->decision.allocation_granularity);
    if (!bytes || temporary_limit > binding.admission->decision.w ||
        limits.envelope.retained > binding.admission->decision.sigma_retained)
      throw std::invalid_argument("Arrow producer exceeds sealed admission");
    if (!binding.gate) binding.gate = std::make_shared<arrow_input_gate>();
    info.contract_id = binding.program.contract;
    info.profiles    = binding.profiles;
    info.names       = binding.column_names;
    info.name        = binding.display_name;
    validator        = std::make_shared<import_validator>(binding, state);
  }
  void check_stop() const
  {
    if (state->stopped) throw duckdb::InterruptException();
  }
  std::unique_ptr<scan_info> produce(std::shared_ptr<input_claim> const& claim,
                                     sm::preparation_stop_token stop)
  {
    check_stop();
    if (stop.stop_requested()) throw duckdb::InterruptException();
    auto result =
      producer->next({claim->key, claim->ledger->allocator(claim->permit), claim}, stop);
    check_stop();
    if (stop.stop_requested()) throw duckdb::InterruptException();
    if (std::holds_alternative<arrow_end_of_stream>(result)) {
      state->eos_sequence = claim->key.unit_id;
      state->eos          = true;
      return {};
    }
    auto input = std::get<produced_arrow_input>(std::move(result));
    try {
      if (!input.batch || input.batch->row_count() > limits.max_rows)
        throw std::invalid_argument("Arrow producer exceeded its declared row bound");
      input.batch->validate_for(binding.program);
    } catch (std::invalid_argument const& error) {
      throw transparent::classified_execution_error(transparent::late_failure_cause::physical_input,
                                                    error.what());
    }
    auto result_info =
      std::make_unique<owned_arrow_scan_info>(std::move(input.batch),
                                              info.contract_id,
                                              binding.program.generation,
                                              claim->key.unit_id,
                                              info.profiles,
                                              claim->ledger->charged_bytes(claim->permit),
                                              binding.read_identity,
                                              claim);
    auto deps = std::vector(result_info->dependencies().begin(), result_info->dependencies().end());
    deps[0].visibility = std::move(input.provenance);
    result_info->set_contract_payload(
      info.contract_id,
      {result_info->certificates().begin(), result_info->certificates().end()},
      std::move(deps));
    validate_split_for_gpu(info.contract_id,
                           check_bit(later_check::host_staged),
                           {},
                           *result_info,
                           *validator,
                           binding.program.generation,
                           info.profiles);
    state->accepted = true;
    return result_info;
  }
};
arrow_import_ingestible::arrow_import_ingestible(arrow_import_binding b,
                                                 std::shared_ptr<arrow_batch_producer> p)
  : impl_(std::make_shared<impl>(std::move(b), std::move(p)))
{
}
arrow_import_ingestible::~arrow_import_ingestible() = default;
void arrow_import_ingestible::preflight_for_query() { impl_->producer->preflight(); }
arrow_import_ingestible::metadata_scan_task_t arrow_import_ingestible::next_split_provider(
  io::ioctx_resolver)
{
  throw std::logic_error(
    "Arrow import does not support synchronous pinning; admitted claims required");
}
arrow_import_ingestible::metadata_claim arrow_import_ingestible::try_claim_split(io::ioctx_resolver)
{
  impl_->check_stop();
  if (!impl_->ledger) throw std::logic_error("Arrow ledger is not attached");
  if (impl_->state->eos) return sm::preparation_exhausted{};
  if (impl_->binding.gate->active || impl_->producer->readiness() == arrow_readiness::pending)
    return sm::preparation_pending{};
  if (impl_->sequence == UINT64_MAX) throw std::overflow_error("Arrow sequence exhausted");
  auto claim    = std::make_shared<input_claim>();
  claim->ledger = impl_->ledger;
  claim->gate   = impl_->binding.gate;
  if (!impl_->pending_identity)
    impl_->pending_identity =
      impl_->binding.next_unit_id ? impl_->binding.next_unit_id() : impl_->sequence;
  claim->key    = {impl_->info.contract_id, *impl_->pending_identity};
  claim->permit = claim->ledger->try_acquire_unit(
    claim->key, impl_->limits.envelope.retained, impl_->temporary_limit);
  if (!claim->permit) return sm::preparation_pending{};
  claim->acquired     = true;
  claim->gate->active = true;
  impl_->pending_identity.reset();
  ++impl_->sequence;
  return metadata_job{[state = impl_, claim](sm::preparation_stop_token stop) {
    return state->produce(claim, stop);
  }};
}
arrow_import_ingestible::preparation_source arrow_import_ingestible::make_preparation_source(
  io::ioctx_resolver)
{
  preparation_source result;
  result.execution   = impl_->producer->execution_domain();
  result.try_claim   = [this] { return try_claim_split({}); };
  result.close       = [state = impl_] { state->producer->close(); };
  result.bind_wakeup = [this](sm::preparation_wakeup wake) {
    bind_preparation_wakeup(std::move(wake));
  };
  result.coalescer = create_batch_coalescer();
  return result;
}
std::unique_ptr<batch_coalescer> arrow_import_ingestible::create_batch_coalescer() const
{
  return std::make_unique<import_coalescer>(impl_->binding, impl_->state, impl_->bytes);
}
std::shared_ptr<sm::preparation_admission> arrow_import_ingestible::preparation_admission() const
{
  return impl_->binding.admission;
}
void arrow_import_ingestible::attach_preparation_ledger(
  std::shared_ptr<sm::preparation_ledger> ledger)
{
  if (!ledger || impl_->ledger) throw std::logic_error("Arrow ledger attachment invalid");
  impl_->ledger = std::move(ledger);
}
void arrow_import_ingestible::bind_preparation_wakeup(sm::preparation_wakeup wake)
{
  impl_->binding.gate->wake = wake;
  impl_->producer->bind_wakeup(wake);
  if (impl_->ledger) impl_->ledger->bind_wakeup(wake);
}
void arrow_import_ingestible::stop_preparation() noexcept
{
  impl_->state->stopped = true;
  impl_->producer->cancel();
  impl_->ledger.reset();
}
void arrow_import_ingestible::finish_preparation() noexcept
{
  stop_preparation();
  if (impl_->binding.owns_admission) impl_->binding.admission->finish();
}
ingestible_table_info const& arrow_import_ingestible::table_info() const noexcept
{
  return impl_->info;
}
std::vector<size_t> arrow_import_ingestible::materialized_column_order() const
{
  return impl_->binding.output_columns;
}
materializer_contract_identity arrow_import_ingestible::materializer() const
{
  return {materializer_kind::stream, "arrow_host_flat_v1", impl_->validator};
}
bool arrow_import_ingestible::has_row_filter() const noexcept
{
  return bool(impl_->binding.predicate);
}
scan::filtered_table arrow_import_ingestible::materialize_metadata_to_table(
  scan::scan_info const& input,
  cucascade::memory::memory_space const& space,
  ::cuda::stream_ref stream,
  bool like,
  std::shared_ptr<sirius::like_multiliteral_cache const> cache,
  expression_query_policy policy)
{
  impl_->check_stop();
  scan::validate_split_for_gpu(impl_->info.contract_id,
                               scan::check_bit(scan::later_check::host_staged),
                               {},
                               input,
                               *impl_->validator,
                               impl_->binding.program.generation,
                               impl_->info.profiles);
  auto const& arrow   = dynamic_cast<scan::owned_arrow_scan_info const&>(input);
  auto mr             = space.get_default_allocator();
  auto const unjoined = scan::detail::prepare_unjoined_failure();
  struct import_state {
    // Retain binding and provider proof as well as the root if completion is unconfirmed.
    std::unique_ptr<owned_arrow_scan_info> input;
    exec::retire_lane lane;
    scan::materialized_batch batch;
    explicit import_state(cudaStream_t stream) : lane(stream) {}
  };
  auto imported       = std::make_unique<import_state>(stream.get());
  imported->input     = arrow.coalesced_copy();
  auto quiesce_import = [&] {
    auto status = imported->lane.quiesce();
    if (status == cudaSuccess && !scan::detail::no_retirement_faults::fails_join(true)) return;
    // Test-owned retention is allowed only after a real successful barrier.
    if (status == cudaSuccess) scan::detail::no_retirement_faults::retain(imported);
    imported.release();
    std::rethrow_exception(unjoined);
  };
  try {
    if (arrow.rootless()) {
      std::vector<std::unique_ptr<cudf::column>> columns;
      for (auto type : *impl_->binding.program.input_types)
        columns.push_back(cudf::make_empty_column(type));
      imported->batch.table = owning_table_view(std::make_unique<cudf::table>(std::move(columns)));
      imported->batch.contract                = impl_->binding.program.contract;
      imported->batch.generation              = impl_->binding.program.generation;
      imported->batch.logical_schema_complete = true;
      ensure_row_count_carrier(imported->batch, stream, mr);
    } else {
      imported->batch =
        scan::import_arrow_host(arrow.batch(), impl_->binding.program, imported->lane, mr);
    }
  } catch (...) {
    // Import unwinding may enqueue output frees after commit recovery. Always
    // establish another barrier while the caller's output reservation is alive.
    quiesce_import();
    throw;
  }
  quiesce_import();
  if (auto counters = impl_->info.profiles->counters;
      counters && counters->track_units && !arrow.rootless()) {
    ++counters->arrow_batches_imported;
    counters->arrow_rows_imported += imported->batch.rows;
  }
  impl_->check_stop();
  auto batch = std::move(imported->batch);
  struct selection_state {
    scan::owning_table_view input;
    std::shared_ptr<sirius::ast::node const> predicate;
    std::unique_ptr<sirius::expression_evaluator> evaluator;
  };
  auto state       = std::make_unique<selection_state>();
  state->predicate = impl_->binding.predicate;
  scan::materialization_steps steps;
  steps.residual = [&](scan::materialized_batch& current) {
    state->input     = std::move(current.table);
    state->evaluator = std::make_unique<sirius::expression_evaluator>(
      state->predicate.get(),
      mr,
      stream,
      policy,
      sirius::expression_evaluator::default_min_ast_size,
      like,
      std::move(cache));
    auto filtered = state->evaluator->select(state->input.view());
    scan::detail::check_read_checkpoint(scan::detail::read_checkpoint::PAYLOAD_GATHER);
    scan::detail::check_read_checkpoint(scan::detail::read_checkpoint::READER_EVENT);
    state->input.record_reader_event(stream);
    current.table = scan::owning_table_view(std::move(filtered));
    current.rows  = current.table.num_rows();
  };
  steps.projection = [&](scan::materialized_batch& current) {
    if (impl_->binding.program.input_types->empty()) return;
    state->input = std::move(current.table);
    if (impl_->binding.output_columns.empty()) {
      current.table = scan::owning_table_view(
        std::make_unique<cudf::table>(std::vector<std::unique_ptr<cudf::column>>{}));
      scan::ensure_row_count_carrier(current, stream, mr);
    } else {
      current.table = scan::select_scan_columns(
        std::move(state->input), impl_->binding.output_columns, stream, mr);
    }
  };
  try {
    batch = scan::finish_materialization(std::move(batch), impl_->binding.program, steps);
    impl_->check_stop();
  } catch (...) {
    scan::detail::retire_reads(state, stream, unjoined);
    throw;
  }
  return {std::move(batch.table), scan::filter_state::ROW_FILTERED_AND_PROJECTED};
}
std::unique_ptr<cudf::table> arrow_import_ingestible::post_filter_and_project(
  scan::filtered_table&& input,
  cucascade::memory::memory_space const& space,
  ::cuda::stream_ref stream,
  bool,
  std::shared_ptr<sirius::like_multiliteral_cache const>,
  std::unique_ptr<cudf::column>*,
  std::span<size_t const> elided,
  expression_query_policy)
{
  if (!elided.empty()) throw std::invalid_argument("Arrow import elision is not qualified");
  input.table.materialize(stream, space.get_default_allocator());
  return input.table.release(stream, space.get_default_allocator());
}
}  // namespace sirius::op::scan
