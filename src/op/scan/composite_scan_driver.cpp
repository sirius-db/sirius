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

#include "op/scan/composite_scan_driver.hpp"

#include "scan_manager/preparation_ledger.hpp"

#include <duckdb/common/exception.hpp>

#include <algorithm>
#include <atomic>
#include <utility>

namespace sirius::op::scan {
namespace scan = sirius::op::scan;
namespace sm   = scan_manager;
namespace {
struct scan_owner {
  uint64_t contract, generation;
  scan::later_check_set supported;
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
      throw scan::certificate_incomplete(
        context.contract, {}, "Composite scan split owner mismatch");
    auto supported = owner_->supported;
    if ((context.required & ~supported).any())
      throw scan::certificate_incomplete(
        context.contract,
        context.required & ~supported,
        "Composite scan split cannot establish requested evidence");
    auto const& split = static_cast<unit_split const&>(input);
    if (!split.input) return {{}, owner_->generation};
    scan::validate_split_contract(context.contract, *split.input);
    if (!split.unit->validator->accepts(*split.input))
      throw scan::certificate_incomplete(context.contract, {}, "Composite child split rejected");
    auto evidence = split.unit->validator->inspect(
      {context.contract, split.unit->checks, context.key}, *split.input);
    if (evidence.generation != owner_->generation)
      throw scan::certificate_incomplete(
        context.contract, {}, "Composite scan unit generation mismatch");
    return evidence;
  }
};
struct scan_emission_state {
  std::atomic<size_t> remaining;
  std::atomic<bool> produced{false};
  explicit scan_emission_state(size_t lanes) : remaining(lanes) {}
};
class unit_coalescer final : public scan::batch_coalescer {
  std::shared_ptr<scan_owner const> owner_;
  std::shared_ptr<scan_emission_state> emission_;
  size_t target_batch_bytes_;
  bool flushed_ = false;
  std::shared_ptr<unit_state> active_;
  scan::batch_coalescer* inner_ = nullptr;

 public:
  unit_coalescer(std::shared_ptr<scan_owner const> o,
                 std::shared_ptr<scan_emission_state> emission,
                 size_t target_batch_bytes)
    : owner_(std::move(o)), emission_(std::move(emission)), target_batch_bytes_(target_batch_bytes)
  {
  }
  size_t target_batch_bytes() const noexcept override { return target_batch_bytes_; }
  size_t retained_bytes() const noexcept override { return inner_ ? inner_->retained_bytes() : 0; }
  std::unique_ptr<scan::scan_info> partial_emit() override
  {
    if (!inner_) return {};
    auto output = inner_->partial_emit();
    if (!inner_->retained_bytes()) clear_retained();
    if (!output) return {};
    emission_->produced = true;
    return std::make_unique<unit_split>(owner_, active_, std::move(output), nullptr, true);
  }
  cursor_step advance(scan::scan_info& input, size_t& cursor, size_t quantum) override
  {
    auto& split = dynamic_cast<unit_split&>(input);
    if (split.owner != owner_ || split.coalesced || !split.coalescer)
      throw std::logic_error("Composite scan coalescer input ownership mismatch");
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
      if (tail.size() > 1) throw std::logic_error("Composite scan coalescer exceeded bounded tail");
      if (!tail.empty()) output = std::move(tail.front());
      if (output) {
        emission_->produced = true;
        return {std::make_unique<unit_split>(owner_, split.unit, std::move(output), nullptr, true),
                true};
      }
      return {nullptr, true};
    }
    emission_->produced = true;
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
    throw std::logic_error("Composite scan requires bounded coalescing");
  }
  std::vector<std::unique_ptr<scan::scan_info>> flush() override
  {
    std::vector<std::unique_ptr<scan::scan_info>> result;
    // Only the final lane can establish that the entire scan had no output.
    if (!std::exchange(flushed_, true) && emission_->remaining.fetch_sub(1) == 1 &&
        !emission_->produced.load())
      result.push_back(std::make_unique<unit_split>(owner_, nullptr, nullptr, nullptr, true));
    return result;
  }
};
}  // namespace
struct composite_scan_driver::impl : std::enable_shared_from_this<impl> {
  std::shared_ptr<scan_owner const> owner;
  std::shared_ptr<scan::split_validator const> validator;
  discover_child discover_next;
  size_t target;
  std::shared_ptr<sm::preparation_ledger> ledger;
  std::shared_ptr<pipeline::completion_handler> completion;
  std::atomic<std::shared_ptr<unit_state>> current;
  std::atomic<bool> done{false}, owner_busy{false};
  bool started = false;
  sm::preparation_wakeup wake;
  impl(scan_contract_id contract,
       uint64_t generation,
       later_check_set supported,
       size_t bytes,
       discover_child discover)
    : owner(std::make_shared<scan_owner const>(scan_owner{contract, generation, supported})),
      validator(std::make_shared<scan_validator>(owner)),
      discover_next(std::move(discover)),
      target(bytes)
  {
    if (!contract || !generation || !target || !discover_next)
      throw std::invalid_argument("Composite scan requires identity, batch bound and discovery");
  }
  void close_lanes(std::shared_ptr<unit_state> const& state,
                   sm::preparation_execution_domain domain)
  {
    std::exception_ptr error;
    for (auto& lane : state->lanes) {
      if (lane->source.execution != domain || lane->closed.exchange(true)) continue;
      try {
        lane->source.close();
      } catch (...) {
        if (!error) error = std::current_exception();
      }
    }
    if (error) std::rethrow_exception(error);
  }
  std::shared_ptr<unit_state> discover(io::ioctx_resolver resolve)
  {
    auto child = discover_next();
    if (!child) {
      done = true;
      return {};
    }
    auto state       = std::make_shared<unit_state>();
    state->source    = std::move(child->source);
    state->validator = std::move(child->validator);
    state->checks    = child->checks;
    // Publish before initialization so failed setup is cleaned up through source close/drain.
    current.store(state);
    if (!state->source || !state->validator || (state->checks & ~owner->supported).any())
      throw std::invalid_argument("Composite child requires a source and supported validator");
    state->source->preflight_for_query();
    state->source->attach_preparation_ledger(ledger);
    state->source->set_execution_completion(completion);
    auto sources      = state->source->make_preparation_sources(resolve);
    bool has_metadata = false;
    for (auto& source : sources) {
      has_metadata |= source.execution == sm::preparation_execution_domain::metadata_worker;
      state->lanes.push_back(std::make_unique<unit_state::lane>(std::move(source)));
    }
    state->metadata_closed = !has_metadata;
    if (state->lanes.empty()) throw std::logic_error("Composite child has no preparation lanes");
    for (auto& lane : state->lanes) {
      if (lane->source.execution != sm::preparation_execution_domain::query_owner &&
          lane->source.execution != sm::preparation_execution_domain::metadata_worker)
        throw std::logic_error("Composite child requested an unsupported execution domain");
      if (!lane->source.try_claim || !lane->source.coalescer)
        throw std::logic_error("Composite child requires a claim callback and coalescer");
      if (lane->source.bind_wakeup) lane->source.bind_wakeup(wake);
    }
    return state;
  }
  scan::gpu_ingestible::metadata_claim claim(sm::preparation_execution_domain domain,
                                             io::ioctx_resolver resolve)
  {
    using job = scan::gpu_ingestible::metadata_job;
    if (done) return sm::preparation_exhausted{};
    auto state = current.load();
    if (domain == sm::preparation_execution_domain::query_owner && !owner_busy &&
        (!state || (state->exhausted && !state->jobs && state->metadata_closed))) {
      owner_busy = true;
      struct release_owner {
        std::shared_ptr<impl> self;
        explicit release_owner(std::shared_ptr<impl> s) : self(std::move(s)) {}
        ~release_owner()
        {
          self->owner_busy = false;
          self->wake();
        }
      };
      auto pending = std::make_shared<release_owner>(shared_from_this());
      return job{[this, state, resolve, pending](
                   sm::preparation_stop_token stop) -> std::unique_ptr<scan::scan_info> {
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
      auto work = std::move(std::get<job>(claim));
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
      return std::move(work).map_output(
        [owner = owner, state, coalescer, pending](
          std::unique_ptr<scan::scan_info> input) -> std::unique_ptr<scan::scan_info> {
          return input
                   ? std::make_unique<unit_split>(owner, state, std::move(input), coalescer, false)
                   : nullptr;
        });
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
composite_scan_driver::composite_scan_driver(scan_contract_id contract,
                                             uint64_t generation,
                                             later_check_set supported,
                                             size_t target,
                                             discover_child discover)
  : impl_(std::make_shared<impl>(contract, generation, supported, target, std::move(discover)))
{
}
composite_scan_driver::~composite_scan_driver() = default;
std::vector<gpu_ingestible::preparation_source> composite_scan_driver::make_preparation_sources(
  io::ioctx_resolver resolve,
  std::shared_ptr<sm::preparation_ledger> ledger,
  std::shared_ptr<pipeline::completion_handler> completion,
  std::function<void(sm::preparation_wakeup)> bind_wakeup)
{
  if (impl_->started || !ledger) throw std::logic_error("Composite scan registration invalid");
  impl_->started    = true;
  impl_->ledger     = std::move(ledger);
  impl_->completion = std::move(completion);
  std::vector<gpu_ingestible::preparation_source> result;
  auto emission = std::make_shared<scan_emission_state>(2);
  for (auto domain : {sm::preparation_execution_domain::query_owner,
                      sm::preparation_execution_domain::metadata_worker}) {
    gpu_ingestible::preparation_source source;
    source.execution   = domain;
    source.try_claim   = [state = impl_, domain, resolve] { return state->claim(domain, resolve); };
    source.bind_wakeup = [state = impl_, bind_wakeup](sm::preparation_wakeup wake) {
      state->wake = wake;
      state->ledger->bind_wakeup(wake);
      if (bind_wakeup) bind_wakeup(wake);
    };
    source.close = [state = impl_, domain] {
      if (auto unit = state->current.load()) state->close_lanes(unit, domain);
    };
    source.coalescer = std::make_unique<unit_coalescer>(impl_->owner, emission, impl_->target);
    result.push_back(std::move(source));
  }
  return result;
}
std::unique_ptr<batch_coalescer> composite_scan_driver::create_batch_coalescer() const
{
  return std::make_unique<unit_coalescer>(
    impl_->owner, std::make_shared<scan_emission_state>(1), impl_->target);
}
std::shared_ptr<split_validator const> composite_scan_driver::validator() const
{
  return impl_->validator;
}
bool composite_scan_driver::exhausted() const { return impl_->done; }
void composite_scan_driver::stop_preparation() noexcept
{
  if (auto state = impl_->current.load(); state && state->source) state->source->stop_preparation();
  impl_->ledger.reset();
}
void composite_scan_driver::finish_preparation() noexcept
{
  stop_preparation();
  impl_->current.store({});
}
std::optional<filtered_table> composite_scan_driver::materialize(
  scan_info const& input,
  cucascade::memory::memory_space const& space,
  ::cuda::stream_ref stream,
  bool like,
  std::shared_ptr<sirius::like_multiliteral_cache const> cache,
  expression_query_policy policy)
{
  if (!impl_->validator->accepts(input))
    throw std::invalid_argument("Composite split owner mismatch");
  auto const& split = static_cast<unit_split const&>(input);
  if (!split.input) return std::nullopt;
  auto batch = split.unit->source->materialize_metadata_to_table(
    *split.input, space, stream, like, cache, policy);
  if (batch.state == filter_state::ROW_FILTERED_AND_PROJECTED) return batch;
  return filtered_table{
    owning_table_view(split.unit->source->post_filter_and_project(
      std::move(batch), space, stream, like, std::move(cache), nullptr, {}, policy)),
    filter_state::ROW_FILTERED_AND_PROJECTED};
}
}  // namespace sirius::op::scan
