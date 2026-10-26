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

#include "ducklake/scan_ingestible.hpp"

#include "ducklake/binding_bridge.hpp"
#include "ducklake/dv_visibility.hpp"
#include "ducklake/field_mapping.hpp"
#include "ducklake/inline_unit_ingestible.hpp"
#include "ducklake/parquet_unit_ingestible.hpp"
#include "ducklake/preparation_gate.hpp"
#include "op/scan/table_scan/split_validation.hpp"

#include <cudf/column/column_factories.hpp>

#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/exception.hpp>

#include <atomic>
#include <numeric>
namespace sirius::ducklake {
namespace {
namespace scan = op::scan;
namespace sm   = scan_manager;
struct scan_owner {
  uint64_t contract, generation;
};
struct scan_info final : scan::ingestible_table_info {
  std::vector<std::string> names;
  std::span<std::string const> column_names() const override { return names; }
  std::span<std::string const> file_paths() const override { return {}; }
  std::string display_name() const override { return "DuckLake bound scan"; }
  bool allows_cached_rows() const override { return false; }
};
struct unit_state {
  ~unit_state()
  {
    if (source) source->finish_preparation();
  }
  struct lane {
    scan::gpu_ingestible::preparation_source source;
    std::atomic<bool> exhausted{false};
    std::atomic<bool> closed{false};
    explicit lane(scan::gpu_ingestible::preparation_source s) : source(std::move(s)) {}
  };
  std::shared_ptr<scan::gpu_ingestible> source;
  std::shared_ptr<scan::split_validator const> validator;
  scan::later_check_set checks;
  std::vector<std::unique_ptr<lane>> lanes;
  std::atomic<size_t> jobs{0};
  std::atomic<bool> exhausted{false};
  std::atomic<bool> metadata_closed{false};
};
// Only this translation unit can construct the route and its owner witness.
struct unit_split final : scan::scan_info {
  std::shared_ptr<scan_owner const> owner;
  std::shared_ptr<unit_state> unit;
  std::unique_ptr<scan::scan_info> input;
  scan::batch_coalescer* coalescer = nullptr;
  bool coalesced = false, input_finished = false;
  unit_split(std::shared_ptr<scan_owner const> o,
             std::shared_ptr<unit_state> u,
             std::unique_ptr<scan::scan_info> i,
             scan::batch_coalescer* c,
             bool ready)
    : scan::scan_info(
        i ? std::vector<fadvise_entry>(i->fadvise_hints().begin(), i->fadvise_hints().end())
          : std::vector<fadvise_entry>{}),
      owner(std::move(o)),
      unit(std::move(u)),
      input(std::move(i)),
      coalescer(c),
      coalesced(ready)
  {
    set_contract_payload(
      owner->contract,
      input ? std::vector(input->certificates().begin(), input->certificates().end())
            : std::vector<scan::split_materializer_certificate>{},
      input ? std::vector(input->dependencies().begin(), input->dependencies().end())
            : std::vector<scan::split_dependencies>{});
  }
  std::span<scan::split_materializer_certificate const> certificates() const override
  {
    return input ? input->certificates() : std::span<scan::split_materializer_certificate const>{};
  }
  std::span<scan::split_dependencies const> dependencies() const override
  {
    return input ? input->dependencies() : std::span<scan::split_dependencies const>{};
  }
  scan::scan_contract_id contract_id() const override { return owner->contract; }
  size_t estimated_bytes() const noexcept override { return input ? input->estimated_bytes() : 0; }
  size_t estimated_working_set_bytes() const noexcept override
  {
    return input ? input->estimated_working_set_bytes() : 0;
  }
  size_t preparation_bytes() const noexcept override
  {
    return input ? input->preparation_bytes() : 0;
  }
};
class scan_validator final : public scan::split_validator {
  std::shared_ptr<scan_owner const> owner_;

 public:
  explicit scan_validator(std::shared_ptr<scan_owner const> o) : owner_(std::move(o)) {}
  bool accepts(scan::scan_info const& input) const noexcept override
  {
    auto split = dynamic_cast<unit_split const*>(&input);
    return split && split->owner == owner_ && split->coalesced;
  }
  scan::split_validation_evidence inspect(scan::split_validation_context const& context,
                                          scan::scan_info const& input) const override
  {
    if (!accepts(input))
      throw scan::certificate_incomplete(context.contract, {}, "DuckLake split owner mismatch");
    auto supported = scan::check_bit(scan::later_check::footer_per_file) |
                     scan::check_bit(scan::later_check::profile_per_file) |
                     scan::check_bit(scan::later_check::schema_per_file) |
                     scan::check_bit(scan::later_check::host_staged);
    if ((context.required & ~supported).any())
      throw scan::certificate_incomplete(context.contract,
                                         context.required & ~supported,
                                         "DuckLake split cannot establish requested evidence");
    auto const& split = static_cast<unit_split const&>(input);
    if (!split.input) return {{}, owner_->generation};
    auto evidence = split.unit->validator->inspect(
      {context.contract, split.unit->checks, context.key}, *split.input);
    if (evidence.generation != owner_->generation)
      throw scan::certificate_incomplete(context.contract, {}, "DuckLake unit generation mismatch");
    return evidence;
  }
};
class unit_coalescer final : public scan::batch_coalescer {
  std::shared_ptr<scan_owner const> owner_;
  bool owner_lane_;
  bool produced_ = false;
  std::shared_ptr<unit_state> active_;
  scan::batch_coalescer* inner_ = nullptr;

 public:
  unit_coalescer(std::shared_ptr<scan_owner const> o, bool owner_lane)
    : owner_(std::move(o)), owner_lane_(owner_lane)
  {
  }
  size_t target_batch_bytes() const noexcept override
  {
    return sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE;
  }
  size_t retained_bytes() const noexcept override { return inner_ ? inner_->retained_bytes() : 0; }
  std::unique_ptr<scan::scan_info> partial_emit() override
  {
    if (!inner_) return {};
    auto output = inner_->partial_emit();
    if (!inner_->retained_bytes()) clear_retained();
    if (!output) return {};
    produced_ = true;
    return std::make_unique<unit_split>(owner_, active_, std::move(output), nullptr, true);
  }
  cursor_step advance(scan::scan_info& input, size_t& cursor, size_t quantum) override
  {
    auto& split = dynamic_cast<unit_split&>(input);
    if (split.owner != owner_ || split.coalesced || !split.coalescer)
      throw std::logic_error("DuckLake coalescer input ownership mismatch");
    active_ = split.unit;
    inner_  = split.coalescer;
    std::unique_ptr<scan::scan_info> output;
    if (!split.input_finished) {
      auto step            = split.coalescer->advance(*split.input, cursor, quantum);
      split.input_finished = step.finished;
      if (inner_->retained_bytes())
        note_retained(clock::now());
      else
        clear_retained();
      output = std::move(step.batch);
      if (!output && !step.finished) return {nullptr, false};
    }
    if (!output && split.input_finished) {
      auto tail = split.coalescer->flush();
      active_.reset();
      inner_ = nullptr;
      clear_retained();
      if (tail.size() > 1) throw std::logic_error("DuckLake coalescer exceeded bounded tail");
      if (!tail.empty()) output = std::move(tail.front());
      if (output) {
        produced_ = true;
        return {std::make_unique<unit_split>(owner_, split.unit, std::move(output), nullptr, true),
                true};
      }
      return {nullptr, true};
    }
    produced_ = true;
    // A direct batch must retire its input in the same step. Retaining that
    // input until another lookahead slot opens can pin the last shared permit
    // behind a join whose build scan is waiting for that permit.
    bool finished = split.input_finished && !inner_->retained_bytes();
    if (finished) {
      active_.reset();
      inner_ = nullptr;
      clear_retained();
    }
    return {std::make_unique<unit_split>(owner_, split.unit, std::move(output), nullptr, true),
            finished};
  }
  std::vector<std::unique_ptr<scan::scan_info>> push(std::unique_ptr<scan::scan_info>) override
  {
    throw std::logic_error("DuckLake scan requires bounded coalescing");
  }
  std::vector<std::unique_ptr<scan::scan_info>> flush() override
  {
    std::vector<std::unique_ptr<scan::scan_info>> result;
    if (owner_lane_ && !std::exchange(produced_, true))
      result.push_back(std::make_unique<unit_split>(owner_, nullptr, nullptr, nullptr, true));
    return result;
  }
};
}  // namespace
struct scan_ingestible::impl {
  std::shared_ptr<binding_handle> binding;
  scan_info info;
  std::shared_ptr<scan_owner const> owner;
  std::shared_ptr<scan::split_validator const> validator;
  std::shared_ptr<sm::preparation_admission> admission;
  std::shared_ptr<sm::preparation_ledger> ledger;
  std::vector<size_t> selected, outputs;
  std::vector<cudf::data_type> output_types;
  unit_scan_options options;
  bool count_only;
  uint64_t next_file = 0;
  std::atomic<std::shared_ptr<unit_state>> current;
  std::atomic<bool> done{false}, owner_busy{false};
  sm::preparation_wakeup wake;
  impl(std::shared_ptr<binding_handle> b,
       uint64_t contract,
       uint64_t generation,
       std::shared_ptr<sm::preparation_admission> a,
       std::vector<size_t> columns,
       bool count,
       unit_scan_options opts)
    : binding(std::move(b)),
      owner(std::make_shared<scan_owner const>(scan_owner{contract, generation})),
      validator(std::make_shared<scan_validator>(owner)),
      admission(std::move(a)),
      options(std::move(opts)),
      count_only(count)
  {
    if (!binding || !contract || !generation || !admission || !admission->decision.deferred)
      throw std::invalid_argument("DuckLake scan requires original binding and admitted resources");
    auto const& identity = *binding->view().identity;
    if (count_only && (!columns.empty() || options.filters || options.output_columns))
      throw std::invalid_argument("DuckLake count-only cannot request filter columns");
    if (!count_only)
      selected = required_field_indices(identity.bound_types.size(), std::move(columns));
    validate_scan_options(options, selected.size());
    info.contract_id = contract;
    if (options.profiles) info.profiles = options.profiles;
    options.profiles    = info.profiles;
    options.identities  = std::make_shared<preparation_identity>();
    options.inline_gate = std::make_shared<preparation_gate>();
    outputs.resize(selected.size());
    std::iota(outputs.begin(), outputs.end(), 0);
    if (options.output_columns) outputs = *options.output_columns;
    for (auto column : selected)
      info.names.push_back(identity.bound_names.at(column));
    for (auto column : outputs) {
      auto type = scan::field_mapping_type(identity.bound_types.at(selected.at(column)));
      if (!type) throw std::invalid_argument("DuckLake scan output type is not qualified");
      output_types.push_back(*type);
    }
  }
  void close_lanes(std::shared_ptr<unit_state> const& state,
                   sm::preparation_execution_domain domain)
  {
    for (auto& lane : state->lanes)
      if (lane->source.execution == domain && !lane->closed.exchange(true)) lane->source.close();
  }
  std::shared_ptr<unit_state> discover(io::ioctx_resolver resolve)
  {
    auto descriptor = binding->unit_at(next_file);
    if (!descriptor) {
      done = true;
      return {};
    }
    auto state = std::make_shared<unit_state>();
    unit_scan_options opts;
    if (options.filters) opts.filters = options.filters->Copy();
    opts.output_columns = options.output_columns;
    opts.profiles       = options.profiles;
    opts.identities     = options.identities;
    opts.inline_gate    = options.inline_gate;
    opts.owns_admission = false;
    if (descriptor->kind == unit_kind::parquet) {
      auto source      = binding->make_parquet_unit(next_file,
                                               owner->contract,
                                               owner->generation,
                                               admission,
                                               selected,
                                               count_only,
                                               std::move(opts));
      state->validator = source->materializer().validator;
      state->checks    = scan::check_bit(scan::later_check::footer_per_file) |
                      scan::check_bit(scan::later_check::profile_per_file) |
                      scan::check_bit(scan::later_check::schema_per_file);
      state->source = std::move(source);
    } else {
      auto source      = binding->make_inline_unit(next_file,
                                              owner->contract,
                                              owner->generation,
                                              admission,
                                              count_only,
                                              selected,
                                              std::move(opts));
      state->validator = source->materializer().validator;
      state->checks    = scan::check_bit(scan::later_check::host_staged);
      state->source    = std::move(source);
    }
    state->source->preflight_for_query();
    state->source->attach_preparation_ledger(ledger);
    state->source->set_execution_completion(completion);
    bool has_metadata = false;
    for (auto& source : state->source->make_preparation_sources(resolve)) {
      if (source.execution == sm::preparation_execution_domain::native_reader)
        throw std::logic_error("DuckLake unit requested an unbound native lane");
      has_metadata |= source.execution == sm::preparation_execution_domain::metadata_worker;
      if (source.bind_wakeup) source.bind_wakeup(wake);
      state->lanes.push_back(std::make_unique<unit_state::lane>(std::move(source)));
    }
    state->metadata_closed = !has_metadata;
    if (next_file == std::numeric_limits<uint64_t>::max())
      throw std::overflow_error("DuckLake original file index exhausted");
    ++next_file;
    return state;
  }
  std::shared_ptr<pipeline::completion_handler> completion;
  scan::gpu_ingestible::metadata_claim claim(sm::preparation_execution_domain domain,
                                             io::ioctx_resolver resolve)
  {
    using job = scan::gpu_ingestible::metadata_job;
    if (done) return sm::preparation_exhausted{};
    auto state = current.load();
    if (domain == sm::preparation_execution_domain::query_owner && !owner_busy &&
        (!state || (state->exhausted && !state->jobs && state->metadata_closed))) {
      owner_busy = true;
      return job{[this, state, resolve](
                   sm::preparation_stop_token stop) -> std::unique_ptr<scan::scan_info> {
        struct release {
          impl* self;
          ~release()
          {
            self->owner_busy = false;
            self->wake();
          }
        } guard{this};
        if (stop.stop_requested()) throw duckdb::InterruptException();
        if (state) close_lanes(state, sm::preparation_execution_domain::query_owner);
        current.store(discover(resolve));
        return {};
      }};
    }
    if (!state || owner_busy) return sm::preparation_pending{};
    for (auto& lane : state->lanes) {
      if (lane->source.execution != domain || lane->exhausted) continue;
      auto claim = lane->source.try_claim();
      if (std::holds_alternative<sm::preparation_exhausted>(claim)) {
        lane->exhausted = true;
        continue;
      }
      if (std::holds_alternative<sm::preparation_pending>(claim)) continue;
      auto work        = std::move(std::get<job>(claim));
      auto readiness   = work.unit;
      auto requirement = work.requirement;
      ++state->jobs;
      struct pending_job {
        std::shared_ptr<unit_state> state;
        sm::preparation_wakeup wake;
        pending_job(std::shared_ptr<unit_state> s, sm::preparation_wakeup w)
          : state(std::move(s)), wake(w)
        {
        }
        ~pending_job()
        {
          --state->jobs;
          wake();
        }
      };
      auto pending    = std::make_shared<pending_job>(state, wake);
      auto* coalescer = lane->source.coalescer.get();
      job result{[owner = owner, state, coalescer, pending, work = std::move(work)](
                   sm::preparation_stop_token stop) mutable {
                   auto input = work.run(stop);
                   return input ? std::make_unique<unit_split>(
                                    owner, state, std::move(input), coalescer, false)
                                : nullptr;
                 },
                 readiness};
      result.requirement = requirement;
      return result;
    }
    if (std::all_of(state->lanes.begin(), state->lanes.end(), [](auto const& lane) {
          return lane->exhausted.load();
        }))
      if (!state->exhausted.exchange(true)) wake();
    if (domain == sm::preparation_execution_domain::metadata_worker && state->exhausted &&
        !state->jobs && !state->metadata_closed) {
      close_lanes(state, domain);
      state->metadata_closed = true;
      wake();
    }
    return sm::preparation_pending{};
  }
};
scan_ingestible::scan_ingestible(std::shared_ptr<binding_handle> binding,
                                 uint64_t contract,
                                 uint64_t generation,
                                 std::shared_ptr<sm::preparation_admission> admission,
                                 std::vector<size_t> columns,
                                 bool count,
                                 unit_scan_options options)
  : impl_(std::make_shared<impl>(std::move(binding),
                                 contract,
                                 generation,
                                 std::move(admission),
                                 std::move(columns),
                                 count,
                                 std::move(options)))
{
}
scan_ingestible::~scan_ingestible() = default;
sm::unit_envelope scan_ingestible::envelope()
{
  auto result              = inline_unit_ingestible::envelope(inline_unit_ingestible::max_columns,
                                                 inline_unit_ingestible::max_columns);
  result.temporary.front() = std::max<uint64_t>(dv_max_file_bytes, result.temporary.front());
  result.retained = 14 * dv_max_footer_bytes + 512 + 4 * dv_max_positions * sizeof(int64_t);
  return result;
}

void scan_ingestible::preflight_for_query() { impl_->binding->bind_execution_owner(); }
bool scan_ingestible::has_processed_all_metadata() const { return impl_->done; }
scan_ingestible::metadata_scan_task_t scan_ingestible::next_split_provider(io::ioctx_resolver)
{
  throw std::logic_error("DuckLake scan requires admitted multi-lane preparation");
}
std::vector<scan_ingestible::preparation_source> scan_ingestible::make_preparation_sources(
  io::ioctx_resolver resolve)
{
  impl_->completion = _execution_completion;
  std::vector<preparation_source> result;
  for (auto domain : {sm::preparation_execution_domain::query_owner,
                      sm::preparation_execution_domain::metadata_worker}) {
    preparation_source source;
    source.execution   = domain;
    source.try_claim   = [state = impl_, domain, resolve] { return state->claim(domain, resolve); };
    source.bind_wakeup = [state = impl_](sm::preparation_wakeup wake) {
      state->wake                      = wake;
      state->options.inline_gate->wake = wake;
      state->ledger->bind_wakeup(wake);
    };
    source.close = [state = impl_, domain] {
      if (auto unit = state->current.load()) state->close_lanes(unit, domain);
    };
    source.coalescer = std::make_unique<unit_coalescer>(
      impl_->owner, domain == sm::preparation_execution_domain::query_owner);
    result.push_back(std::move(source));
  }
  return result;
}
std::unique_ptr<scan::batch_coalescer> scan_ingestible::create_batch_coalescer() const
{
  return std::make_unique<unit_coalescer>(impl_->owner, true);
}
std::shared_ptr<sm::preparation_admission> scan_ingestible::preparation_admission() const
{
  return impl_->admission;
}
void scan_ingestible::attach_preparation_ledger(std::shared_ptr<sm::preparation_ledger> ledger)
{
  if (!ledger || impl_->ledger) throw std::logic_error("DuckLake scan ledger attachment invalid");
  impl_->ledger = std::move(ledger);
}
void scan_ingestible::stop_preparation() noexcept
{
  if (auto state = impl_->current.load()) state->source->stop_preparation();
  impl_->ledger.reset();
}
void scan_ingestible::finish_preparation() noexcept
{
  if (auto state = impl_->current.load()) state->source->finish_preparation();
  stop_preparation();
  impl_->current.store({});
  impl_->admission->finish();
}
scan::ingestible_table_info const& scan_ingestible::table_info() const noexcept
{
  return impl_->info;
}
std::vector<size_t> scan_ingestible::materialized_column_order() const { return impl_->outputs; }
bool scan_ingestible::has_row_filter() const noexcept { return bool(impl_->options.filters); }
scan::materializer_contract_identity scan_ingestible::materializer() const
{
  return {scan::materializer_kind::stream, "ducklake.bound-scan.v1", impl_->validator};
}
scan::filtered_table scan_ingestible::materialize_metadata_to_table(
  scan::scan_info const& input,
  cucascade::memory::memory_space const& space,
  ::cuda::stream_ref stream,
  bool like,
  std::shared_ptr<sirius::like_multiliteral_cache const> cache,
  expression_query_policy policy)
{
  scan::validate_split_for_gpu(impl_->owner->contract,
                               {},
                               {},
                               input,
                               *impl_->validator,
                               impl_->owner->generation,
                               impl_->info.profiles);
  auto const& split = static_cast<unit_split const&>(input);
  if (!split.input) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    for (auto type : impl_->output_types)
      columns.push_back(cudf::make_empty_column(type));
    if (columns.empty())
      columns.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::INT8}));
    return {scan::owning_table_view(std::make_unique<cudf::table>(std::move(columns))),
            scan::filter_state::ROW_FILTERED_AND_PROJECTED};
  }
  auto batch = split.unit->source->materialize_metadata_to_table(
    *split.input, space, stream, like, cache, policy);
  if (batch.state != scan::filter_state::ROW_FILTERED_AND_PROJECTED) {
    auto table = split.unit->source->post_filter_and_project(
      std::move(batch), space, stream, like, std::move(cache), nullptr, {}, policy);
    return {scan::owning_table_view(std::move(table)),
            scan::filter_state::ROW_FILTERED_AND_PROJECTED};
  }
  return batch;
}
std::unique_ptr<cudf::table> scan_ingestible::post_filter_and_project(
  scan::filtered_table&& input,
  cucascade::memory::memory_space const& space,
  ::cuda::stream_ref stream,
  bool,
  std::shared_ptr<sirius::like_multiliteral_cache const>,
  std::unique_ptr<cudf::column>*,
  std::span<size_t const> elided,
  expression_query_policy)
{
  if (!elided.empty()) throw std::invalid_argument("DuckLake scan elision is not qualified");
  input.table.materialize(stream, space.get_default_allocator());
  return input.table.release(stream, space.get_default_allocator());
}
}  // namespace sirius::ducklake
