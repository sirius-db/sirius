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

#include "scan_manager/preparation_coordinator.hpp"

#include "duckdb/common/exception.hpp"

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <system_error>

namespace sirius::scan_manager {
std::unique_ptr<op::scan::scan_info> preparation_job::run(preparation_stop_token stop)
{
  if (!operation_) throw std::logic_error("preparation job already consumed");
  auto operation = std::move(operation_);
  operation_     = nullptr;
  try {
    auto input = operation(stop);
    if (!requirement || requirement->required.none()) return input;
    if (!unit) throw std::logic_error("preparation job has no admitted unit");
    if (!input || input->dependencies().size() != 1)
      throw std::invalid_argument("preparation unit requires one dependency record");
    auto const& deps = input->dependencies().front();
    unit->complete_input(footer_input{deps.footer, deps.parquet_approval, deps.datasource});
    unit->complete_input(visibility_input{deps.visibility});
    if (requirement->required.test(static_cast<size_t>(required_input::segments)))
      unit->complete_input(segments_input{deps.profiles});
    if (requirement->required.test(static_cast<size_t>(required_input::checkpoint_iteration)))
      unit->complete_input(checkpoint_input{deps.checkpoint_iteration.value()});
    return input;
  } catch (...) {
    if (unit) {
      auto error = std::current_exception();
      auto cause = transparent::classify_failure(error);
      std::optional<op::scan::verdict_reason> reason;
      try {
        std::rethrow_exception(error);
      } catch (op::scan::unsupported_physical_input const& e) {
        reason = e.reason;
      } catch (...) {
      }
      unit->fail(preparation_failure{cause.cause, reason, cause.detail, error});
    }
    throw;
  }
}

struct preparation_coordinator::state {
  static thread_local state const* native_caller;
  struct native_call_guard {
    state const* previous;
    explicit native_call_guard(state const* current)
      : previous(std::exchange(native_caller, current))
    {
    }
    ~native_call_guard() { native_caller = previous; }
  };
  struct slot {
    enum class phase { free, running, complete } status = phase::free;
    size_t source_id = 0, cursor = 0;
    std::optional<size_t> next_free;
    std::unique_ptr<op::scan::scan_info> input;
    std::shared_ptr<preparation_unit> unit;
  };
  struct job_ticket;
  // Captures always die before the ticket returns the completion slot.
  struct accepted_job {
    std::shared_ptr<job_ticket> ticket;
    job work;
    void dispose()
    {
      work = {};
      ticket.reset();
    }
  };
  class provider_execution {
   public:
    bool can_claim() const { return phase_ == phase::idle && !close_requested_; }
    void begin_claim() { phase_ = phase::claiming; }
    void settle_claim()
    {
      if (phase_ == phase::claiming) phase_ = phase::idle;
    }
    job& stage(accepted_job request)
    {
      request_.emplace(std::move(request));
      phase_ = phase::staged;
      return request_->work;
    }
    void dispatch()
    {
      if (phase_ == phase::staged) phase_ = phase::ready;
    }
    struct operation {
      accepted_job accepted;
      bool closing = false;
    };
    std::optional<operation> take(bool stopping)
    {
      if (phase_ == phase::ready || (phase_ == phase::staged && stopping)) {
        operation next{std::move(*request_)};
        request_.reset();
        phase_ = phase::executing;
        return next;
      }
      if (phase_ == phase::idle && (close_requested_ || stopping)) {
        phase_ = phase::closing;
        return operation{{}, true};
      }
      return {};
    }
    void complete() { phase_ = phase_ == phase::closing ? phase::closed : phase::idle; }
    bool request_close() { return !std::exchange(close_requested_, true); }
    bool closed() const { return phase_ == phase::closed; }

   private:
    enum class phase { idle, claiming, staged, ready, executing, closing, closed };
    phase phase_ = phase::idle;
    std::optional<accepted_job> request_;
    bool close_requested_ = false;
  };
  struct native_job;
  class native_execution {
   public:
    void prepare_close(std::unique_ptr<native_reader_executor::operation> close)
    {
      close_ = std::move(close);
    }
    void attach(std::unique_ptr<native_reader_executor::lease> location)
    {
      location_ = std::move(location);
    }
    bool can_claim() const { return location_ && location_->idle(); }
    void begin_claim(std::shared_ptr<job_ticket>);
    job& stage(job) noexcept;
    void submit(std::function<void()> const& before_submit);
    std::shared_ptr<job_ticket> abandon_claim();
    bool retire(std::unique_lock<std::mutex>&);
    void complete_close() { closed_ = true; }
    bool closed() const { return closed_ && !location_; }

   private:
    std::unique_ptr<native_reader_executor::lease> location_;
    std::unique_ptr<native_job> pending_;
    std::unique_ptr<native_reader_executor::operation> close_;
    bool close_submitted_ = false, closed_ = false;
  };
  struct output_group {
    std::shared_ptr<split_connector> destination;
    std::function<void()> close;
    size_t remaining;
  };
  struct source_state {
    source hooks;
    std::shared_ptr<output_group> output;
    bool output_done = false;
    enum class phase {
      enumerating,
      enumeration_done,
      tail_flushed,
      closed
    } status           = phase::enumerating;
    size_t outstanding = 0, pending_batches = 0;
    // Completed work remains charged through coalescing and publication until downstream pop.
    size_t work_bytes = 0, output_bytes = 0, window_bytes = 0;
    std::optional<size_t> active_result;
    provider_execution provider;
    native_execution native;
    bool on_native_reader() const { return hooks.execution == execution_domain::native_reader; }
    bool on_query_owner() const { return hooks.execution == execution_domain::query_owner; }
    bool reader_closed() const
    {
      return (!on_query_owner() || provider.closed()) && (!on_native_reader() || native.closed());
    }
  };
  struct pending_batch {
    size_t source_id;
    std::unique_ptr<op::scan::scan_info> batch;
    size_t preparation_bytes;
    std::optional<op::scan::batch_coalescer::clock::time_point> deadline_retained;
    bool size_triggered = false;
  };
  std::shared_ptr<preparation_gate> gate = std::make_shared<preparation_gate>();
  preparation_options options;
  pipeline::completion_handler* completion;
  exec::scoped_dispatcher* dispatcher;
  native_reader_executor* native_executor;
  std::shared_ptr<preparation_wakeup> native_wake;
  std::deque<slot> slots;
  std::optional<size_t> free_slot;
  std::deque<size_t> ready_results;
  std::vector<source_state> sources;
  std::map<unit_key, std::shared_ptr<preparation_unit>> units;
  std::deque<pending_batch> batches;
  lifecycle phase = lifecycle::constructed;
  size_t jobs = 0, results = 0, output = 0, external = 0;
  size_t reader_jobs = 0, next_provider_source = 0;
  std::thread::id provider_owner;
  bool provider_pumping = false, independent_driver = false, control_active = false;
  std::stop_source reader_stop;
  std::shared_ptr<preparation_stop_token::reason_state> reader_stop_reason =
    std::make_shared<preparation_stop_token::reason_state>();
  size_t next_deadline_source = 0, next_claim_source = 0;
  statistics stats;
  std::function<void()> before_thread, before_submit, before_publish;
  std::function<void(size_t, std::shared_ptr<op::scan::scan_info> const&, size_t)> after_publish;
  std::function<op::scan::batch_coalescer::clock::time_point()> now =
    op::scan::batch_coalescer::clock::now;
  bool manual_clock = false, start_called = false, fail_batch_enqueue = false;
  state(preparation_options o,
        pipeline::completion_handler& c,
        exec::scoped_dispatcher& d,
        native_reader_executor* native)
    : options(o), completion(&c), dispatcher(&d), native_executor(native)
  {
  }

  bool can_emit(size_t source_id) const
  {
    auto const& source = sources[source_id];
    return source.output_bytes < source.window_bytes;
  }

  struct job_ticket {
    std::shared_ptr<state> st;
    size_t index = 0;
    bool active = false, completed = false;
    ~job_ticket()
    {
      if (!active || completed) return;
      bool cancelled;
      {
        std::lock_guard lock(st->gate->mutex);
        cancelled = st->gate->closed;
      }
      if (!cancelled) {
        // An exception leaving claim/enqueue is reported by the runner; preserve that first cause.
        if (!std::uncaught_exceptions()) {
          try {
            throw std::runtime_error("accepted preparation job was discarded");
          } catch (...) {
            st->completion->report_error(std::current_exception());
          }
        }
        st->stop();
      }
      st->settle(index, {});
    }
  };
  void stop(stop_reason reason = stop_reason::failure_induced) noexcept
  {
    {
      std::lock_guard lock(gate->mutex);
      // The gate may already be closed by interrupt detection. Keep intent independent
      // of that gate and never downgrade it during later cleanup.
      reader_stop_reason->request(reason);
      gate->closed = true;
      for (auto& [key, u] : units)
        u->cancel_under_gate();
      gate->notify_under_lock();
    }
    // Stop callbacks can reenter; invoke them outside the publication gate.
    reader_stop.request_stop();
  }
  bool normal_eos_requested() const noexcept
  {
    return reader_stop_reason->get() == stop_reason::normal_eos;
  }
  void reserve_slot(size_t index, size_t source_id)
  {
    auto& s     = slots[index];
    s.source_id = source_id;
    s.cursor    = 0;
    s.status    = slot::phase::running;
    ++jobs;
    ++sources[source_id].outstanding;
    if (sources[source_id].on_query_owner() || sources[source_id].on_native_reader()) ++reader_jobs;
    if (sources[source_id].on_query_owner()) sources[source_id].provider.begin_claim();
    stats.jobs_peak = std::max(stats.jobs_peak, jobs);
  }
  struct retired_slot {
    std::unique_ptr<op::scan::scan_info> input;
    std::shared_ptr<preparation_unit> unit;
  };
  // Detach under the gate; the caller destroys the returned owners after unlocking.
  retired_slot release_slot(size_t index)
  {
    auto& s = slots[index];
    if (s.status == slot::phase::free) return {};
    if (s.status == slot::phase::running) {
      --jobs;
      if (sources[s.source_id].on_query_owner() || sources[s.source_id].on_native_reader())
        --reader_jobs;
      if (sources[s.source_id].on_query_owner()) sources[s.source_id].provider.settle_claim();
    } else
      --results;
    auto& source = sources[s.source_id];
    --source.outstanding;
    if (source.active_result == index) source.active_result.reset();
    retired_slot retired{std::move(s.input), s.unit};
    if (s.unit) units.erase(s.unit->key());
    s.unit.reset();
    s.status    = slot::phase::free;
    s.next_free = free_slot;
    free_slot   = index;
    return retired;
  }
  // Runs only after work and its input captures have been physically released.
  void settle(size_t index, std::unique_ptr<op::scan::scan_info> value) noexcept
  {
    auto const bytes = value ? value->preparation_bytes() : 0;
    retired_slot retired;
    try {
      std::lock_guard lock(gate->mutex);
      auto& s = slots[index];
      if (gate->closed) {
        retired = release_slot(index);
      } else {
        auto& source = sources[s.source_id];
        if (bytes > std::numeric_limits<size_t>::max() - source.work_bytes)
          throw std::overflow_error("preparation lookahead work size overflow");
        ready_results.push_back(index);
        source.work_bytes += bytes;
        --jobs;
        if (source.on_query_owner() || source.on_native_reader()) --reader_jobs;
        if (source.on_query_owner()) source.provider.settle_claim();
        s.input  = std::move(value);
        s.status = slot::phase::complete;
        ++results;
        stats.results_peak = std::max(stats.results_peak, results);
      }
      gate->notify_under_lock();
    } catch (...) {
      completion->report_error(std::current_exception());
      stop();
      std::lock_guard lock(gate->mutex);
      retired = release_slot(index);
      gate->notify_under_lock();
    }
  }

  using lock_type = std::unique_lock<std::mutex>;
  // These steps enter and return with the gate locked; callbacks run unlocked,
  // except the final visible push. On exception the lock may be released.
  std::optional<op::scan::batch_coalescer::clock::time_point> size_deadline(size_t id) const
  {
    auto const& source = sources[id];
    if (!options.partial_batch || source.pending_batches || !can_emit(id)) return {};
    auto const& coalescer = *source.hooks.coalescer;
    auto first            = coalescer.first_retained_time();
    if (!first) return {};
    auto cap     = coalescer.target_batch_bytes();
    auto percent = options.partial_batch->min_fill_percent;
    // Round up without overflowing when the target is near SIZE_MAX.
    auto threshold = (cap / 100) * percent + ((cap % 100) * percent + 99) / 100;
    if (coalescer.retained_bytes() < threshold) return {};
    return *first + options.partial_batch->min_residence;
  }
  bool emit_partial_batch(lock_type&, bool size_triggered = false);
  bool publish_batch(lock_type&);
  bool advance_result(lock_type&);
  bool submit_job(lock_type&, std::shared_ptr<state> const&);
  bool close_finished_source(lock_type&);
  bool emit_full_window(lock_type&);
  void queue_batch(lock_type&,
                   size_t,
                   std::unique_ptr<op::scan::scan_info>,
                   std::optional<op::scan::batch_coalescer::clock::time_point> = {},
                   bool size_triggered                                         = false);
  bool inputs_settled() const;
  std::optional<op::scan::batch_coalescer::clock::time_point> nearest_deadline() const;
  void execute_job(job&, std::shared_ptr<job_ticket> const&, std::stop_token);
  bool pump_provider();
  void arm_native();
  bool retire_reader(size_t, lock_type&);
  void close_source(size_t) noexcept;
  void close_source_output(size_t) noexcept;
  void dispatch_job(source_state&, size_t, job, std::shared_ptr<job_ticket>, lock_type&);
  bool advance_control(std::shared_ptr<state> const&, std::optional<std::chrono::milliseconds>);
  void run(std::shared_ptr<state> const&);
  void shutdown();
  void finish();

  // Destroyed without the gate held; unpublished construction returns its output credit.
  struct publication_ticket {
    state& owner;
    bool published = false;
    ~publication_ticket()
    {
      std::lock_guard lock(owner.gate->mutex);
      if (!published) --owner.output;
      owner.gate->notify_under_lock();
    }
  };
};
thread_local preparation_coordinator::state const* preparation_coordinator::state::native_caller =
  nullptr;

struct preparation_coordinator::state::native_job : native_reader_executor::operation {
  accepted_job accepted;
  void run() noexcept override
  {
    auto& owner = *accepted.ticket->st;
    native_call_guard call(&owner);
    bool cancelled;
    {
      std::lock_guard lock(owner.gate->mutex);
      cancelled = owner.gate->closed;
    }
    if (!cancelled)
      owner.execute_job(accepted.work, accepted.ticket, owner.reader_stop.get_token());
    accepted.dispose();  // Abandoned work settles on the assigned native thread.
  }
};

void preparation_coordinator::state::arm_native()
{
  auto count = std::ranges::count_if(sources, [](auto const& s) { return s.on_native_reader(); });
  if (!count) return;
  struct close_operation : native_reader_executor::operation {
    state& owner;
    size_t id;
    close_operation(state& o, size_t i) : owner(o), id(i) {}
    void run() noexcept override
    {
      native_call_guard call(&owner);
      owner.close_source(id);
      std::lock_guard lock(owner.gate->mutex);
      owner.sources[id].native.complete_close();
      owner.gate->notify_under_lock();
    }
  };
  // All allocations, including close controls, precede the first native operation.
  for (size_t i = 0; i < sources.size(); ++i)
    if (sources[i].on_native_reader())
      sources[i].native.prepare_close(std::make_unique<close_operation>(*this, i));
  native_wake    = native_executor->observe(preparation_wakeup{gate});
  auto locations = native_executor->reserve(static_cast<size_t>(count));
  size_t next    = 0;
  for (auto& source : sources)
    if (source.on_native_reader()) source.native.attach(std::move(locations[next++]));
}

void preparation_coordinator::state::native_execution::begin_claim(
  std::shared_ptr<job_ticket> ticket)
{
  pending_                  = std::make_unique<native_job>();
  pending_->accepted.ticket = std::move(ticket);
}
preparation_coordinator::job& preparation_coordinator::state::native_execution::stage(
  job work) noexcept
{
  pending_->accepted.work = std::move(work);
  return pending_->accepted.work;
}
void preparation_coordinator::state::native_execution::submit(
  std::function<void()> const& before_submit)
{
  if (before_submit) before_submit();
  location_->submit(std::move(pending_));
}
std::shared_ptr<preparation_coordinator::state::job_ticket>
preparation_coordinator::state::native_execution::abandon_claim()
{
  auto ticket = std::move(pending_->accepted.ticket);
  pending_.reset();  // Pending/exhausted claims contain no accepted work.
  return ticket;
}

// The coordinator retires the lease only after the worker has disposed work and closed.
bool preparation_coordinator::state::native_execution::retire(std::unique_lock<std::mutex>& lock)
{
  if (!location_) {
    closed_ = true;  // Group admission failed; no reader could have been initialized.
    return false;
  }
  if (!location_->idle()) return false;
  if (pending_) {
    // Accepted captures from a failed submission still belong to the worker.
    location_->submit(std::move(pending_));
    return true;
  }
  if (!close_submitted_) {
    close_submitted_ = true;
    location_->submit(std::move(close_));
    return true;
  }
  if (!closed_) return false;
  auto retired = std::move(location_);
  lock.unlock();
  retired.reset();  // Release may wake this coordinator.
  lock.lock();
  return true;
}

bool preparation_coordinator::state::retire_reader(size_t id, lock_type& lock)
{
  auto& source = sources[id];
  if (source.on_native_reader()) return source.native.retire(lock);
  if (source.on_query_owner() && source.provider.request_close()) {
    gate->notify_under_lock();
    return true;
  }
  return false;
}

void preparation_coordinator::state::close_source(size_t id) noexcept
{
  close_source_output(id);
  try {
    sources[id].hooks.close();
  } catch (...) {
    completion->report_preparation_cleanup_error(std::current_exception());
    stop();
  }
}

void preparation_coordinator::state::close_source_output(size_t id) noexcept
{
  auto& source = sources[id];
  auto& output = *source.output;
  {
    std::lock_guard lock(gate->mutex);
    if (std::exchange(source.output_done, true) || --output.remaining != 0) return;
  }
  try {
    output.destination->close();
    if (output.close) output.close();
  } catch (...) {
    completion->report_preparation_cleanup_error(std::current_exception());
    stop();
  }
}

preparation_coordinator::preparation_coordinator(pipeline::completion_handler& completion,
                                                 exec::scoped_dispatcher& dispatcher,
                                                 preparation_options options,
                                                 native_reader_executor* native)
  : state_(std::make_shared<state>(options, completion, dispatcher, native))
{
  if (options.partial_batch) options.partial_batch->validate();
  if (!options.lookahead_batches || !options.max_control_work || !options.drain_quantum ||
      options.interrupt_check_interval.count() <= 0)
    throw std::invalid_argument("query driver needs positive limits");
}
preparation_coordinator::~preparation_coordinator()
{
  request_stop(stop_reason::normal_eos);
  drain();
}
void preparation_coordinator::add_source(source s)
{
  std::vector<source> sources;
  sources.push_back(std::move(s));
  add_sources(std::move(sources));
}
void preparation_coordinator::add_sources(std::vector<source> sources,
                                          std::function<void()> close_output)
{
  std::unique_lock lock(state_->gate->mutex);
  if (state_->phase != lifecycle::constructed || state_->gate->closed)
    throw std::logic_error("source registration is closed");
  if (sources.empty()) throw std::invalid_argument("scan requires at least one preparation source");
  auto destination  = sources.front().destination;
  bool has_provider = false;
  for (size_t i = 0; i < sources.size(); ++i) {
    auto const& s = sources[i];
    if (!s.coalescer || !s.try_claim || !s.construct || !s.destination || !s.close)
      throw std::invalid_argument("incomplete coordinator source");
    if (s.destination != destination)
      throw std::invalid_argument("preparation lanes must share one scan output");
    if (s.execution == execution_domain::native_reader && !state_->native_executor)
      throw std::invalid_argument("native reader source requires a shared native executor");
    auto cap = s.coalescer->target_batch_bytes();
    if (!cap || cap > std::numeric_limits<size_t>::max() / state_->options.lookahead_batches)
      throw std::invalid_argument("preparation lookahead byte target is invalid");
    for (size_t j = 0; j < i; ++j)
      if (sources[j].coalescer == s.coalescer)
        throw std::invalid_argument("preparation lanes require independent coalescers");
    has_provider |= s.execution == execution_domain::query_owner;
  }
  for (auto const& registered : state_->sources)
    if (registered.hooks.destination == destination)
      throw std::logic_error("scan output already has a preparation group");
  auto current = std::this_thread::get_id();
  if (has_provider && state_->provider_owner != std::thread::id{} &&
      state_->provider_owner != current)
    throw std::logic_error("provider sources must share their query owner");
  auto first = state_->sources.size();
  auto output =
    std::make_shared<state::output_group>(destination, std::move(close_output), sources.size());
  state_->sources.reserve(first + sources.size());
  std::weak_ptr<state> weak = state_;
  destination->set_consumption_callback([weak](size_t bytes, size_t id) {
    if (auto st = weak.lock()) {
      std::lock_guard lock(st->gate->mutex);
      auto& source = st->sources[id];
      --source.pending_batches;
      source.work_bytes -= bytes;
      source.output_bytes -= bytes;
      if (st->output) --st->output;
      st->gate->notify_under_lock();
    }
  });
  if (has_provider) state_->provider_owner = current;
  for (auto& s : sources) {
    auto cap = s.coalescer->target_batch_bytes();
    state_->sources.push_back({std::move(s), output});
    state_->sources.back().window_bytes = cap * state_->options.lookahead_batches;
  }
  lock.unlock();
  for (size_t id = first; id < state_->sources.size(); ++id) {
    auto& registered = state_->sources[id];
    if (registered.hooks.bind_wakeup)
      registered.hooks.bind_wakeup(preparation_wakeup{state_->gate});
  }
}
std::shared_ptr<preparation_gate> preparation_coordinator::publication_gate() const
{
  return state_->gate;
}
std::shared_ptr<void> preparation_coordinator::external_use()
{
  std::unique_lock lock(state_->gate->mutex);
  if (state_->gate->closed) return {};
  // Allocate the guard before changing the count, so allocation failure cannot strand drain.
  struct use {
    std::shared_ptr<state> st;
    bool active = false;
    ~use()
    {
      if (active) {
        std::lock_guard l(st->gate->mutex);
        --st->external;
        st->gate->notify_under_lock();
      }
    }
  };
  auto guard = std::make_shared<use>();
  guard->st  = state_;
  ++state_->external;
  guard->active = true;
  return guard;
}
void preparation_coordinator::set_interrupt_check(std::function<bool()> interrupted)
{
  std::lock_guard lock(state_->gate->mutex);
  if (state_->phase != lifecycle::constructed)
    throw std::logic_error("interrupt check must be bound before preparation arms");
  state_->gate->interrupted = std::move(interrupted);
}
void preparation_coordinator::arm()
{
  std::unique_lock execution_lock(execution_mutex_);
  {
    std::lock_guard lock(state_->gate->mutex);
    if (state_->phase != lifecycle::constructed) throw std::logic_error("preparation arms once");
    if (state_->provider_owner != std::thread::id{} &&
        state_->provider_owner != std::this_thread::get_id())
      throw std::logic_error("provider preparation must arm on its registering owner");
    state_->stats.owner        = std::this_thread::get_id();
    state_->independent_driver = std::ranges::any_of(state_->sources, [](auto const& source) {
      return source.hooks.execution != execution_domain::metadata_worker;
    });
  }
  try {
    state_->completion->begin_preparation();
    {
      std::lock_guard lock(state_->gate->mutex);
      state_->phase = lifecycle::armed;
    }
    std::weak_ptr<state> weak = state_;
    state_->completion->set_preparation_stop_callback([weak](bool error) {
      if (auto st = weak.lock()) {
        auto reason = error ? stop_reason::failure_induced : stop_reason::normal_eos;
        if (error) {
          // Completion commits the original error before invoking this observer. In
          // particular, user cancellation must not first notify readers as a failure.
          try {
            if (auto original = st->completion->terminal_error()) std::rethrow_exception(original);
          } catch (duckdb::InterruptException const&) {
            reason = stop_reason::user_cancel;
          } catch (...) {
          }
        }
        st->stop(reason);
      }
    });
    std::optional<preparation_failure> existing;
    {
      std::lock_guard lock(state_->gate->mutex);
      state_->gate->report_error = [weak](std::exception_ptr error) {
        if (auto st = weak.lock()) st->completion->report_error(error);
      };
      state_->gate->report_failure = [weak](preparation_failure const& error) {
        if (auto st = weak.lock()) st->completion->report_error(error);
      };
      for (auto const& [key, unit] : state_->units) {
        auto record = unit->record();
        if (record.failure) {
          existing = record.failure;
          break;
        }
      }
    }
    if (existing) state_->completion->report_error(*existing);
    state_->arm_native();
    if (state_->independent_driver) {
      if (state_->before_thread) state_->before_thread();
      // Readers may block Q or their native worker; control work must remain independent.
      runner_ = std::thread([st = state_] { st->run(st); });
    }
  } catch (...) {
    execution_lock.unlock();
    state_->completion->report_error(std::current_exception());
    request_stop(stop_reason::failure_induced);
    drain();
    throw;
  }
}
std::shared_ptr<preparation_unit> preparation_coordinator::admit_unit(unit_key k,
                                                                      required_input_set required)
{
  std::lock_guard lock(state_->gate->mutex);
  if (state_->gate->closed) return {};
  state_->gate->check_interrupted();
  if (auto it = state_->units.find(k); it != state_->units.end()) return it->second;
  auto u = std::make_shared<preparation_unit>(k, required, state_->gate);
  state_->units.emplace(k, u);
  return u;
}
void preparation_coordinator::retire_unit(unit_key k)
{
  decltype(state_->units)::node_type retired;
  {
    std::lock_guard lock(state_->gate->mutex);
    retired = state_->units.extract(k);
    state_->gate->notify_under_lock();
  }
}
std::optional<unit_record> preparation_coordinator::unit_state_snapshot(unit_key k) const
{
  std::lock_guard lock(state_->gate->mutex);
  auto it = state_->units.find(k);
  if (it == state_->units.end()) return {};
  return it->second->record();
}
void preparation_coordinator::request_stop(stop_reason reason) noexcept
{
  // Commit cancellation before waking the owner, which may immediately become quiescent.
  if (reason == stop_reason::user_cancel) {
    try {
      state_->completion->report_error(std::make_exception_ptr(duckdb::InterruptException()));
    } catch (...) {
      state_->completion->report_error(std::current_exception());
    }
  }
  state_->stop(reason);
  state_->dispatcher->request_stop();
}
void preparation_coordinator::start()
{
  std::lock_guard execution_lock(execution_mutex_);
  std::lock_guard lock(state_->gate->mutex);
  if (state_->start_called || state_->phase == lifecycle::constructed ||
      state_->stats.owner != std::this_thread::get_id())
    throw std::logic_error("preparation starts once on the query owner after consumers start");
  state_->start_called = true;
  // Consumers may finish an empty query before start opens the producer gate.
  if (state_->gate->closed) return;
  state_->phase = lifecycle::running;
  state_->gate->notify_under_lock();
}

bool preparation_coordinator::advance(std::chrono::milliseconds idle_wait)
{
  bool owner_driven;
  {
    std::unique_lock lock(state_->gate->mutex);
    if (state_->phase == lifecycle::quiescent) {
      auto observed = state_->gate->wake_generation;
      state_->gate->cv.wait_for(
        lock, idle_wait, [&] { return state_->gate->wake_generation != observed; });
      return false;
    }
    if (state_->stats.owner != std::this_thread::get_id())
      throw std::logic_error("preparation advance requires its query owner");
    if (state_->phase != lifecycle::running && !state_->gate->closed) return false;
    owner_driven = !state_->independent_driver;
  }
  if (owner_driven) return state_->advance_control(state_, idle_wait);
  if (idle_wait.count() <= 0) return state_->pump_provider();
  uint64_t observed;
  {
    std::lock_guard lock(state_->gate->mutex);
    observed = state_->gate->wake_generation;
  }
  if (state_->pump_provider()) return true;
  std::unique_lock lock(state_->gate->mutex);
  state_->gate->cv.wait_for(
    lock, idle_wait, [&] { return state_->gate->wake_generation != observed; });
  return false;  // The caller rechecks completion/cancellation before the next operation.
}

void preparation_coordinator::drain()
{
  if (state::native_caller == state_.get())
    throw std::logic_error("native reader operation cannot drain its own preparation");
  bool owns_provider, owner_driven, owns_control, wait_for_owner, stop_needed;
  {
    std::lock_guard lock(state_->gate->mutex);
    if (state_->control_active && state_->stats.runner == std::this_thread::get_id())
      throw std::logic_error("preparation runner cannot join itself");
    owns_provider = state_->provider_owner == std::this_thread::get_id();
    if (owns_provider && state_->provider_pumping)
      throw std::logic_error("provider operation cannot recursively drain preparation");
  }
  {
    // Serialize with arm, but never hold this lock while another thread must advance.
    std::unique_lock execution_lock(execution_mutex_);
    {
      std::lock_guard lock(state_->gate->mutex);
      owner_driven = !state_->independent_driver && state_->stats.owner != std::thread::id{} &&
                     state_->phase != lifecycle::constructed;
      owns_control   = owner_driven && state_->stats.owner == std::this_thread::get_id();
      wait_for_owner = (owner_driven && !owns_control) ||
                       (state_->provider_owner != std::thread::id{} && !owns_provider);
      stop_needed = state_->phase == lifecycle::armed || wait_for_owner;
    }
    if (!runner_.joinable() && !owner_driven && !wait_for_owner) {
      // Failed startup or an unarmed attempt: there is no driver to wait for.
      request_stop(stop_reason::normal_eos);
      if (owns_provider)
        while (state_->pump_provider()) {}
      state_->advance_control(state_, std::chrono::milliseconds{0});
      return;
    }
  }
  if (stop_needed) request_stop(stop_reason::normal_eos);
  for (;;) {
    uint64_t observed;
    {
      std::lock_guard lock(state_->gate->mutex);
      if (state_->phase == lifecycle::quiescent) break;
      observed = state_->gate->wake_generation;
    }
    if (owns_control) {
      state_->advance_control(state_, std::nullopt);
      continue;
    }
    if (owns_provider && state_->pump_provider()) continue;
    std::unique_lock lock(state_->gate->mutex);
    state_->gate->cv.wait(lock, [&] {
      return state_->phase == lifecycle::quiescent || state_->gate->wake_generation != observed;
    });
  }
  std::lock_guard execution_lock(execution_mutex_);
  if (runner_.joinable()) runner_.join();
}
preparation_coordinator::statistics preparation_coordinator::snapshot() const
{
  std::lock_guard lock(state_->gate->mutex);
  auto stats  = state_->stats;
  stats.phase = state_->phase;
  return stats;
}
void preparation_coordinator::before_thread_for_testing(std::function<void()> hook)
{
  std::lock_guard lock(state_->gate->mutex);
  if (state_->phase != lifecycle::constructed)
    throw std::logic_error("thread creation seam already frozen");
  state_->before_thread = std::move(hook);
}

void preparation_coordinator::before_submission_for_testing(std::function<void()> f)
{
  if (state_->phase != lifecycle::constructed)
    throw std::logic_error("submission seam already frozen");
  state_->before_submit = std::move(f);
}

void preparation_coordinator::before_publication_for_testing(std::function<void()> hook)
{
  if (state_->phase != lifecycle::constructed)
    throw std::logic_error("publication seam already frozen");
  state_->before_publish = std::move(hook);
}
void preparation_coordinator::after_publication_for_testing(
  std::function<void(size_t, std::shared_ptr<op::scan::scan_info> const&, size_t)> hook)
{
  if (state_->phase != lifecycle::constructed)
    throw std::logic_error("publication seam already frozen");
  state_->after_publish = std::move(hook);
}

void preparation_coordinator::clock_for_testing(
  std::function<op::scan::batch_coalescer::clock::time_point()> clock)
{
  std::lock_guard lock(state_->gate->mutex);
  if (state_->phase != lifecycle::constructed) throw std::logic_error("clock already frozen");
  if (!clock) throw std::invalid_argument("empty test clock");
  state_->now          = std::move(clock);
  state_->manual_clock = true;
}
void preparation_coordinator::wake_for_testing()
{
  std::lock_guard lock(state_->gate->mutex);
  state_->gate->notify_under_lock();
}

void preparation_coordinator::fail_next_batch_enqueue_for_testing()
{
  std::lock_guard lock(state_->gate->mutex);
  state_->fail_batch_enqueue = true;
}

void preparation_coordinator::state::queue_batch(
  lock_type& lock,
  size_t source_id,
  std::unique_ptr<op::scan::scan_info> batch,
  std::optional<op::scan::batch_coalescer::clock::time_point> first,
  bool size_triggered)
{
  if (!batch) return;
  auto const bytes = batch->preparation_bytes();
  try {
    if (std::exchange(fail_batch_enqueue, false)) throw std::bad_alloc();
    batches.emplace_back();  // Allocate before transferring batch ownership.
  } catch (...) {
    lock.unlock();  // Releasing the last allocation can wake this same gate.
    throw;
  }
  batches.back() = {source_id, std::move(batch), bytes, first, size_triggered};
  ++sources[source_id].pending_batches;
  sources[source_id].output_bytes += bytes;
  ++output;
  stats.output_peak = std::max(stats.output_peak, output);
}

bool preparation_coordinator::state::emit_partial_batch(lock_type& lock, bool size_triggered)
{
  if (!options.partial_batch) return false;
  auto time = now();
  for (size_t n = 0; n < sources.size(); ++n) {
    auto id         = (next_deadline_source + n) % sources.size();
    auto& coalescer = *sources[id].hooks.coalescer;
    auto first      = coalescer.first_retained_time();
    if (!can_emit(id) || !first) continue;
    auto deadline =
      size_triggered ? size_deadline(id) : *first + options.partial_batch->max_residence;
    if (!deadline || time < *deadline) continue;
    lock.unlock();
    auto batch = coalescer.partial_emit();
    lock.lock();
    queue_batch(lock, id, std::move(batch), first, size_triggered);
    next_deadline_source = id + 1;
    return true;
  }
  return false;
}

bool preparation_coordinator::state::publish_batch(lock_type& lock)
{
  if (batches.empty()) return false;
  auto batch = std::move(batches.front());
  batches.pop_front();
  auto& source = sources[batch.source_id];
  lock.unlock();
  {
    publication_ticket ticket{*this};
    auto publication              = source.hooks.construct(std::move(batch.batch));
    publication.preparation_bytes = batch.preparation_bytes;
    if (!publication.input) throw std::logic_error("publication requires a scan input");
    bool allowed;
    {
      std::lock_guard check(gate->mutex);
      gate->check_interrupted();
      allowed = !gate->closed && !completion->has_error();
    }
    if (allowed) publication.input->prepare_prefetch();
    if (allowed && before_publish) before_publish();
    auto observed =
      after_publish && publication.input->has_scan_metadata()
        ? std::get<std::shared_ptr<op::scan::scan_info>>(publication.input->materialization_info)
        : nullptr;
    {
      std::lock_guard commit(gate->mutex);
      gate->check_interrupted();
      if (!gate->closed && !completion->has_error()) {
        publication.input->publish_prefetch();
        source.hooks.destination->push_prepared_scan(
          publication.input, publication.bytes, publication.preparation_bytes, batch.source_id);
        stats.publisher  = std::this_thread::get_id();
        ticket.published = true;
        if (batch.deadline_retained) {
          auto residence =
            std::chrono::duration_cast<std::chrono::microseconds>(now() - *batch.deadline_retained);
          ++stats.partial_emissions;
          stats.max_residence = std::max(stats.max_residence, residence);
          if (batch.size_triggered)
            ++stats.size_emissions;
          else
            stats.max_deadline_lateness = std::max(
              stats.max_deadline_lateness, residence - options.partial_batch->max_residence);
        } else {
          ++stats.normal_emissions;
        }
      }
    }
    if (ticket.published && after_publish)
      after_publish(batch.source_id, observed, batch.preparation_bytes);
    // Losing construction results (including their disposed callbacks) die outside the gate.
  }
  lock.lock();
  return true;
}

bool preparation_coordinator::state::advance_result(lock_type& lock)
{
  for (size_t remaining = ready_results.size(); remaining > 0; --remaining) {
    auto index = ready_results.front();
    ready_results.pop_front();
    ready_results.push_back(index);
    auto& slot = slots[index];
    if (!can_emit(slot.source_id)) continue;
    auto& source = sources[slot.source_id];
    if (source.active_result && *source.active_result != index) continue;
    if (slot.unit) {
      auto record = slot.unit->record();
      if (record.state == unit_state::failed) {
        lock.unlock();
        completion->report_error(*record.failure);
        stop();
        lock.lock();
        return true;
      }
      if (record.state == unit_state::pending) continue;
      if (record.state == unit_state::cancelled && slot.input) {
        source.work_bytes -= slot.input->preparation_bytes();
        auto discarded = std::move(slot.input);
        lock.unlock();
        discarded.reset();
        lock.lock();
      }
    }
    ready_results.pop_back();
    source.active_result = index;
    lock.unlock();
    auto step = slot.input
                  ? source.hooks.coalescer->advance(*slot.input, slot.cursor, options.drain_quantum)
                  : op::scan::batch_coalescer::cursor_step{nullptr, true};
    lock.lock();
    bool const emitted = static_cast<bool>(step.batch);
    queue_batch(lock, slot.source_id, std::move(step.batch));
    if (step.finished) {
      auto retired = release_slot(index);
      lock.unlock();
      retired = {};
      lock.lock();
    } else
      ready_results.push_back(index);
    // Give ready metadata a bounded advance before considering an early partial batch.
    if (!emitted) emit_partial_batch(lock, /*size_triggered=*/true);
    return true;
  }
  return false;
}

void preparation_coordinator::state::execute_job(job& work,
                                                 std::shared_ptr<job_ticket> const& ticket,
                                                 std::stop_token stop_token)
{
  std::unique_ptr<op::scan::scan_info> value;
  try {
    value = work.run(preparation_stop_token{stop_token, reader_stop_reason});
  } catch (...) {
    auto error               = std::current_exception();
    bool expected_normal_eos = false;
    try {
      std::rethrow_exception(error);
    } catch (std::system_error const& e) {
      expected_normal_eos = e.code() == std::errc::operation_canceled && normal_eos_requested();
    } catch (...) {
    }
    if (!expected_normal_eos) {
      completion->report_error(error);
      stop();
    }
  }
  work              = {};  // Captures die before settlement returns the slot.
  ticket->completed = true;
  settle(ticket->index, std::move(value));
}

bool preparation_coordinator::state::pump_provider()
{
  size_t selected = sources.size();
  bool discard    = false;
  std::optional<provider_execution::operation> operation;
  {
    std::lock_guard lock(gate->mutex);
    if (provider_owner == std::thread::id{}) return false;
    if (provider_owner != std::this_thread::get_id())
      throw std::logic_error("provider pump requires its query owner");
    if (provider_pumping) throw std::logic_error("provider pump cannot be reentered");
    for (size_t n = 0; n < sources.size(); ++n) {
      auto id = (next_provider_source + n) % sources.size();
      if (!sources[id].on_query_owner()) continue;
      operation = sources[id].provider.take(gate->closed);
      if (!operation) continue;
      discard              = gate->closed;
      provider_pumping     = true;
      selected             = id;
      next_provider_source = id + 1;
      break;
    }
    if (!operation) return false;
  }
  if (operation->closing)
    close_source(selected);
  else if (!discard)
    execute_job(operation->accepted.work, operation->accepted.ticket, reader_stop.get_token());
  operation->accepted
    .dispose();  // Captures and ticket retire on the query owner, outside the gate.
  {
    std::lock_guard lock(gate->mutex);
    sources[selected].provider.complete();
    provider_pumping = false;
    gate->notify_under_lock();
  }
  return true;
}

// Once accepted, each domain retains the job even if the submission hook throws.
void preparation_coordinator::state::dispatch_job(
  source_state& source, size_t index, job work, std::shared_ptr<job_ticket> ticket, lock_type& lock)
{
  auto prepare = [&](job& owned) {
    if (owned.requirement && !gate->closed) {
      gate->check_interrupted();
      auto const& spec = *owned.requirement;
      auto it          = units.find(spec.key);
      if (it == units.end())
        it =
          units.emplace(spec.key, std::make_shared<preparation_unit>(spec.key, spec.required, gate))
            .first;
      owned.unit = it->second;
    }
    slots[index].unit = owned.unit;
  };
  if (source.on_native_reader()) {
    auto& owned = source.native.stage(std::move(work));
    lock.lock();
    prepare(owned);
    lock.unlock();
    source.native.submit(before_submit);
    lock.lock();
  } else if (source.on_query_owner()) {
    lock.lock();
    auto& owned = source.provider.stage({std::move(ticket), std::move(work)});
    prepare(owned);
    lock.unlock();
    if (before_submit) before_submit();
    lock.lock();
    source.provider.dispatch();
    gate->notify_under_lock();
  } else {
    accepted_job submitted{std::move(ticket), std::move(work)};
    {
      std::lock_guard prepare_lock(gate->mutex);
      prepare(submitted.work);
    }
    if (before_submit) before_submit();
    {
      std::lock_guard check(gate->mutex);
      gate->check_interrupted();
    }
    dispatcher->enqueue([submitted = std::move(submitted)](std::stop_token stop) mutable {
      submitted.ticket->st->execute_job(submitted.work, submitted.ticket, stop);
    });
    lock.lock();
  }
}

bool preparation_coordinator::state::submit_job(lock_type& lock, std::shared_ptr<state> const& self)
{
  auto const start = next_claim_source;
  for (size_t n = 0; n < sources.size(); ++n) {
    auto id      = (start + n) % sources.size();
    auto& source = sources[id];
    if (source.on_query_owner() || source.on_native_reader()) {
      if (source.outstanding || source.pending_batches >= options.lookahead_batches) continue;
      if (source.on_query_owner() ? !source.provider.can_claim() : !source.native.can_claim())
        continue;
    } else if (options.max_inflight_metadata_jobs &&
               jobs - reader_jobs >= options.max_inflight_metadata_jobs)
      continue;
    if (source.status != source_state::phase::enumerating ||
        source.work_bytes >= source.window_bytes)
      continue;
    gate->check_interrupted();
    auto ticket = std::make_shared<job_ticket>();
    auto index  = free_slot.value_or(slots.size());
    if (free_slot)
      free_slot = slots[index].next_free;
    else
      slots.emplace_back();
    ticket->st    = self;
    ticket->index = index;
    reserve_slot(index, id);
    ticket->active = true;
    lock.unlock();
    if (source.on_native_reader()) source.native.begin_claim(std::move(ticket));
    auto claimed = source.hooks.try_claim();
    if (auto* ready = std::get_if<job>(&claimed)) {
      dispatch_job(source, index, std::move(*ready), std::move(ticket), lock);
      next_claim_source = id + 1;
      return true;
    }
    if (source.on_native_reader()) ticket = source.native.abandon_claim();
    lock.lock();
    ticket->completed = true;
    auto retired      = release_slot(index);
    lock.unlock();
    retired = {};
    lock.lock();
    if (std::holds_alternative<preparation_exhausted>(claimed)) {
      source.status     = source_state::phase::enumeration_done;
      next_claim_source = id + 1;
      return true;
    }
    if (gate->closed) return false;
    // Pending consumes no cursor or slot and cannot prevent later sources from claiming.
  }
  next_claim_source = sources.empty() ? 0 : (start + 1) % sources.size();
  return false;
}

bool preparation_coordinator::state::emit_full_window(lock_type& lock)
{
  // A full window can prevent the next row group from triggering the coalescer's normal flush.
  for (size_t id = 0; id < sources.size(); ++id) {
    auto& source = sources[id];
    if (source.outstanding || source.work_bytes < source.window_bytes || !can_emit(id) ||
        !source.hooks.coalescer->first_retained_time())
      continue;
    lock.unlock();
    auto batch = source.hooks.coalescer->partial_emit();
    lock.lock();
    if (batch) {
      queue_batch(lock, id, std::move(batch));
      return true;
    }
  }
  return false;
}

bool preparation_coordinator::state::close_finished_source(lock_type& lock)
{
  if (!batches.empty()) return false;
  for (size_t id = 0; id < sources.size(); ++id) {
    auto& source = sources[id];
    if (source.status == source_state::phase::closed ||
        source.status == source_state::phase::enumerating || source.outstanding ||
        (source.status == source_state::phase::enumeration_done && !can_emit(id)))
      continue;
    if (source.status == source_state::phase::enumeration_done) {
      lock.unlock();
      auto final = source.hooks.coalescer->flush();
      lock.lock();
      source.status = source_state::phase::tail_flushed;
      if (final.size() > 1) {
        lock.unlock();
        throw std::logic_error("final flush exceeded its reserved output slot");
      }
      if (!final.empty()) {
        queue_batch(lock, id, std::move(final.front()));
        return true;  // Publish the final batch before closing its connector.
      }
    }
    if (source.on_native_reader() || source.on_query_owner()) {
      auto progressed = retire_reader(id, lock);
      if (source.reader_closed()) {
        source.status = source_state::phase::closed;
        return true;
      }
      if (progressed) return true;
      continue;
    }
    source.status = source_state::phase::closed;
    lock.unlock();
    close_source(id);
    lock.lock();
    return true;
  }
  return false;
}

bool preparation_coordinator::state::inputs_settled() const
{
  return !jobs && !external && !gate->callbacks && !results && batches.empty() &&
         std::ranges::all_of(
           sources, [](auto const& s) { return s.status == source_state::phase::closed; }) &&
         std::ranges::none_of(
           units, [](auto const& p) { return p.second->record().state == unit_state::pending; });
}

std::optional<op::scan::batch_coalescer::clock::time_point>
preparation_coordinator::state::nearest_deadline() const
{
  std::optional<op::scan::batch_coalescer::clock::time_point> deadline;
  if (!options.partial_batch) return {};
  auto limit = options.partial_batch->max_residence;
  for (size_t id = 0; id < sources.size(); ++id) {
    if (!can_emit(id)) continue;
    auto first = sources[id].hooks.coalescer->first_retained_time();
    if (!first) continue;
    auto next = size_deadline(id);
    if (!next || *first + limit < *next) next = *first + limit;
    if (next && (!deadline || *next < *deadline)) deadline = next;
  }
  return deadline;
}

void preparation_coordinator::state::shutdown()
{
  // Errors have already notified the observer. Natural exhaustion is not a failure,
  // and this final retirement request cannot downgrade an earlier error/cancellation.
  stop(stop_reason::normal_eos);
  dispatcher->request_stop();
  dispatcher->wait_for_all();
  {
    std::unique_lock lock(gate->mutex);
    for (;;) {
      auto observed   = gate->wake_generation;
      bool progressed = false;
      for (size_t id = 0; id < sources.size(); ++id)
        progressed |= retire_reader(id, lock);
      if (!jobs && !external && !gate->callbacks &&
          std::ranges::all_of(sources, [](auto const& s) { return s.reader_closed(); }))
        break;
      if (progressed) continue;
      gate->cv.wait(lock, [&] { return gate->wake_generation != observed; });
    }
  }
  finish();
}

void preparation_coordinator::state::finish()
{
  // Only the execution runner closes sources after their workers and callbacks have settled.
  for (size_t id = 0; id < sources.size(); ++id) {
    auto& source = sources[id];
    if (source.status == source_state::phase::closed) continue;
    if (!source.on_query_owner() && !source.on_native_reader())
      close_source(id);
    else
      close_source_output(id);  // Also settles readers rejected before native lease admission.
    source.status = source_state::phase::closed;
  }
  std::deque<pending_batch> discarded_batches;
  decltype(units) discarded_units;
  {
    std::unique_lock lock(gate->mutex);
    discarded_batches.swap(batches);
    ready_results.clear();
    for (size_t index = 0; index < slots.size(); ++index) {
      auto retired = release_slot(index);
      lock.unlock();
      retired = {};
      lock.lock();
    }
    discarded_units.swap(units);
    gate->closed = true;
  }
  discarded_batches.clear();
  discarded_units.clear();
  {
    std::lock_guard lock(gate->mutex);
    phase = lifecycle::quiescent;
    gate->notify_under_lock();
  }
  if (stats.runs || stats.owner != std::thread::id{}) {
    completion->close_preparation_inputs();
    completion->preparation_quiescent();
  }
}

bool preparation_coordinator::state::advance_control(
  std::shared_ptr<state> const& st, std::optional<std::chrono::milliseconds> idle_wait)
{
  {
    std::lock_guard lock(gate->mutex);
    if (phase == lifecycle::quiescent) return false;
    if (control_active) throw std::logic_error("preparation control cannot reenter");
    control_active = true;
    stats.runner   = std::this_thread::get_id();
    if (!gate->closed && !stats.runs) ++stats.runs;
  }
  struct control_guard {
    state& st;
    ~control_guard()
    {
      std::lock_guard lock(st.gate->mutex);
      st.control_active = false;
    }
  } active{*this};
  try {
    std::unique_lock lock(gate->mutex);
    ++stats.wakes;
    if (!gate->closed) {
      gate->check_interrupted();
      auto const observed_wake = gate->wake_generation;
      // Both drivers use the same bounded turn, deadlines and wake generation.
      size_t actions = emit_partial_batch(lock) ? 1 : 0;
      auto available = [&] {
        if (gate->closed || actions >= options.max_control_work) return false;
        gate->check_interrupted();
        return true;
      };
      if (available() && publish_batch(lock)) ++actions;
      if (available() && (advance_result(lock) || emit_partial_batch(lock, true))) ++actions;
      if (available() && submit_job(lock, st)) ++actions;
      if (available() && close_finished_source(lock)) ++actions;
      if (available() && emit_full_window(lock)) ++actions;
      if (!gate->closed && !inputs_settled()) {
        if (actions || gate->wake_generation != observed_wake) return true;
        auto deadline = nearest_deadline();
        if (manual_clock) deadline.reset();
        auto bound_wait = [&](std::chrono::milliseconds interval) {
          auto end = op::scan::batch_coalescer::clock::now() + interval;
          deadline = deadline ? std::min(*deadline, end) : end;
        };
        if (gate->interrupted) bound_wait(options.interrupt_check_interval);
        if (idle_wait) bound_wait(*idle_wait);
        if (deadline)
          gate->cv.wait_until(lock, *deadline);
        else
          gate->cv.wait(lock);
        return false;
      }
    }
  } catch (...) {
    completion->report_error(std::current_exception());
  }
  shutdown();
  return false;
}

void preparation_coordinator::state::run(std::shared_ptr<state> const& st)
{
  {
    std::unique_lock lock(gate->mutex);
    gate->cv.wait(lock, [&] { return gate->closed || phase == lifecycle::running; });
  }
  for (;;) {
    advance_control(st, std::nullopt);
    std::lock_guard lock(gate->mutex);
    if (phase == lifecycle::quiescent) return;
  }
}
}  // namespace sirius::scan_manager
