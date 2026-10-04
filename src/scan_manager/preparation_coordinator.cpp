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
#include <map>

namespace sirius::scan_manager {
struct preparation_coordinator::state {
  struct slot {
    enum class phase { free, running, complete } status = phase::free;
    size_t source_id = 0, cursor = 0;
    std::unique_ptr<op::scan::scan_info> input;
    std::shared_ptr<preparation_unit> unit;
  };
  struct source_state {
    source hooks;
    enum class phase {
      enumerating,
      enumeration_done,
      tail_flushed,
      closed
    } status           = phase::enumerating;
    size_t outstanding = 0;
    std::optional<size_t> active_result;
  };
  struct pending_batch {
    size_t source_id;
    std::unique_ptr<op::scan::scan_info> batch;
    std::optional<op::scan::batch_coalescer::clock::time_point> deadline_retained;
  };
  std::shared_ptr<preparation_gate> gate = std::make_shared<preparation_gate>();
  preparation_options options;
  pipeline::completion_handler* completion;
  exec::scoped_dispatcher* dispatcher;
  std::vector<slot> slots;
  std::vector<source_state> sources;
  std::map<unit_key, std::shared_ptr<preparation_unit>> units;
  std::deque<pending_batch> batches;
  lifecycle phase = lifecycle::constructed;
  size_t jobs = 0, results = 0, output = 0, external = 0;
  size_t next_result = 0, next_deadline_source = 0;
  statistics stats;
  std::function<void()> before_submit;
  std::function<op::scan::batch_coalescer::clock::time_point()> now =
    op::scan::batch_coalescer::clock::now;
  bool manual_clock = false;
  state(preparation_options o, pipeline::completion_handler& c, exec::scoped_dispatcher& d)
    : options(o), completion(&c), dispatcher(&d), slots(o.max_pending_results)
  {
  }

  bool can_emit(size_t source_id) const
  {
    if (output >= options.max_pending_results) return false;
    // Task creation consumes scans in registration order. Later pipelines must not
    // occupy the last credit while the first open source still needs to publish.
    auto first = std::find_if(sources.begin(), sources.end(), [](auto const& s) {
      return s.status != source_state::phase::closed;
    });
    return first == sources.end() || source_id == static_cast<size_t>(first - sources.begin()) ||
           output + 1 < options.max_pending_results;
  }

  struct job_ticket {
    std::shared_ptr<state> st;
    size_t index = 0;
    bool active = false, completed = false;
    ~job_ticket()
    {
      if (active && !completed) st->settle(index, {});
    }
  };
  void stop() noexcept
  {
    std::lock_guard lock(gate->mutex);
    gate->closed = true;
    for (auto& [key, u] : units)
      u->cancel_under_gate();
    gate->cv.notify_all();
  }
  void reserve_slot(size_t index, size_t source_id)
  {
    auto& s     = slots[index];
    s.source_id = source_id;
    s.cursor    = 0;
    s.status    = slot::phase::running;
    ++jobs;
    ++sources[source_id].outstanding;
    stats.jobs_peak = std::max(stats.jobs_peak, jobs);
  }
  void release_slot(size_t index)
  {
    auto& s = slots[index];
    if (s.status == slot::phase::free) return;
    if (s.status == slot::phase::running)
      --jobs;
    else
      --results;
    auto& source = sources[s.source_id];
    --source.outstanding;
    if (source.active_result == index) source.active_result.reset();
    s.input.reset();
    if (s.unit) units.erase(s.unit->key());
    s.unit.reset();
    s.status = slot::phase::free;
  }
  // Runs only after work and its input captures have been physically released.
  void settle(size_t index, std::unique_ptr<op::scan::scan_info> value) noexcept
  {
    std::lock_guard lock(gate->mutex);
    auto& s = slots[index];
    if (gate->closed) value.reset();
    if (gate->closed) {
      release_slot(index);
    } else {
      --jobs;
      s.input  = std::move(value);
      s.status = slot::phase::complete;
      ++results;
      stats.results_peak = std::max(stats.results_peak, results);
    }
    gate->cv.notify_all();
  }
  using lock_type = std::unique_lock<std::mutex>;
  // These steps enter and return with the gate locked; callbacks run unlocked,
  // except the final visible push. On exception the lock may be released.
  bool emit_due_batch(lock_type&);
  bool publish_batch(lock_type&);
  bool advance_result(lock_type&);
  bool submit_job(lock_type&, std::shared_ptr<state> const&);
  bool close_finished_source(lock_type&);
  void queue_batch(size_t,
                   std::unique_ptr<op::scan::scan_info>,
                   std::optional<op::scan::batch_coalescer::clock::time_point> = {});
  bool inputs_settled() const;
  std::optional<op::scan::batch_coalescer::clock::time_point> nearest_deadline() const;
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
      owner.gate->cv.notify_all();
    }
  };
};
preparation_coordinator::preparation_coordinator(pipeline::completion_handler& completion,
                                                 exec::scoped_dispatcher& dispatcher,
                                                 preparation_options options)
  : state_(std::make_shared<state>(options, completion, dispatcher))
{
  if (!options.max_inflight_jobs || !options.max_pending_results || !options.max_active_units ||
      !options.max_control_work || !options.drain_quantum ||
      (options.underfilled_batch_residence && options.underfilled_batch_residence->count() <= 0))
    throw std::invalid_argument("query driver needs positive limits");
}
preparation_coordinator::~preparation_coordinator() { drain(); }
void preparation_coordinator::add_source(source s)
{
  std::lock_guard lock(state_->gate->mutex);
  if (state_->phase != lifecycle::constructed || state_->gate->closed)
    throw std::logic_error("source registration is closed");
  if (!s.coalescer || !s.claim || !s.construct || !s.publish || !s.close || !s.bind_consumption)
    throw std::invalid_argument("incomplete coordinator source");
  std::weak_ptr<state> weak = state_;
  s.bind_consumption([weak] {
    if (auto st = weak.lock()) {
      std::lock_guard lock(st->gate->mutex);
      if (st->output) --st->output;
      st->gate->cv.notify_all();
    }
  });
  state_->sources.push_back({std::move(s)});
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
        st->gate->cv.notify_all();
      }
    }
  };
  auto guard = std::make_shared<use>();
  guard->st  = state_;
  ++state_->external;
  guard->active = true;
  return guard;
}
void preparation_coordinator::arm()
{
  {
    std::lock_guard lock(state_->gate->mutex);
    if (state_->phase != lifecycle::constructed) throw std::logic_error("preparation arms once");
    state_->stats.owner = std::this_thread::get_id();
  }
  try {
    state_->completion->begin_preparation();
    {
      std::lock_guard lock(state_->gate->mutex);
      state_->phase = lifecycle::armed;
    }
    std::weak_ptr<state> weak = state_;
    state_->completion->set_preparation_stop_callback([weak](bool) {
      if (auto st = weak.lock()) st->stop();
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
  } catch (...) {
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
  if (auto it = state_->units.find(k); it != state_->units.end()) return it->second;
  if (state_->units.size() >= state_->options.max_active_units) return {};
  auto u = std::make_shared<preparation_unit>(k, required, state_->gate);
  state_->units.emplace(k, u);
  return u;
}
void preparation_coordinator::retire_unit(unit_key k)
{
  std::lock_guard lock(state_->gate->mutex);
  state_->units.erase(k);
  state_->gate->cv.notify_all();
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
  state_->stop();
  state_->dispatcher->request_stop();
}
void preparation_coordinator::drain()
{
  {
    std::lock_guard lock(state_->gate->mutex);
    if (state_->phase == lifecycle::quiescent) return;
    if (state_->phase == lifecycle::running ||
        (state_->phase == lifecycle::armed && state_->stats.owner != std::this_thread::get_id()))
      throw std::logic_error("only query owner can drain preparation after run");
  }
  state_->shutdown();
}
preparation_coordinator::statistics preparation_coordinator::snapshot() const
{
  std::lock_guard lock(state_->gate->mutex);
  auto stats  = state_->stats;
  stats.phase = state_->phase;
  return stats;
}
void preparation_coordinator::before_submission_for_testing(std::function<void()> f)
{
  if (state_->phase != lifecycle::constructed)
    throw std::logic_error("submission seam already frozen");
  state_->before_submit = std::move(f);
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
  state_->gate->cv.notify_all();
}

void preparation_coordinator::state::queue_batch(
  size_t source_id,
  std::unique_ptr<op::scan::scan_info> batch,
  std::optional<op::scan::batch_coalescer::clock::time_point> first)
{
  if (!batch) return;
  // Only the coordinator takes output credits; consumers only return them.
  // Capacity checked before producing a batch remains available until this charge.
  batches.push_back({source_id, std::move(batch), first});
  ++output;
  stats.output_peak = std::max(stats.output_peak, output);
}

bool preparation_coordinator::state::emit_due_batch(lock_type& lock)
{
  if (!options.underfilled_batch_residence) return false;
  auto time = now();
  for (size_t n = 0; n < sources.size(); ++n) {
    auto id      = (next_deadline_source + n) % sources.size();
    auto& source = sources[id];
    auto first   = source.hooks.coalescer->first_retained_time();
    if (!can_emit(id) || !first || time < *first + *options.underfilled_batch_residence) continue;
    lock.unlock();
    auto batch = source.hooks.coalescer->partial_emit();
    lock.lock();
    queue_batch(id, std::move(batch), first);
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
    auto publication = source.hooks.construct(std::move(batch.batch));
    {
      std::lock_guard commit(gate->mutex);
      if (!gate->closed && !completion->has_error()) {
        source.hooks.publish(std::move(publication));
        stats.publisher  = std::this_thread::get_id();
        ticket.published = true;
        if (batch.deadline_retained) {
          auto residence =
            std::chrono::duration_cast<std::chrono::microseconds>(now() - *batch.deadline_retained);
          ++stats.partial_emissions;
          stats.max_residence = std::max(stats.max_residence, residence);
          stats.max_deadline_lateness =
            std::max(stats.max_deadline_lateness, residence - *options.underfilled_batch_residence);
        }
      }
    }
    // Losing construction results (including their disposed callbacks) die outside the gate.
  }
  lock.lock();
  return true;
}

bool preparation_coordinator::state::advance_result(lock_type& lock)
{
  for (size_t n = 0; n < slots.size(); ++n) {
    auto index = (next_result + n) % slots.size();
    auto& slot = slots[index];
    if (slot.status != slot::phase::complete || !can_emit(slot.source_id)) continue;
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
      if (record.state == unit_state::cancelled) slot.input.reset();
    }
    source.active_result = index;
    lock.unlock();
    auto step = slot.input
                  ? source.hooks.coalescer->advance(*slot.input, slot.cursor, options.drain_quantum)
                  : op::scan::batch_coalescer::cursor_step{nullptr, true};
    lock.lock();
    queue_batch(slot.source_id, std::move(step.batch));
    if (step.finished) release_slot(index);
    next_result = index + 1;
    return true;
  }
  return false;
}

bool preparation_coordinator::state::submit_job(lock_type& lock, std::shared_ptr<state> const& self)
{
  if (jobs >= options.max_inflight_jobs || results + jobs >= options.max_active_units ||
      output >= options.max_pending_results)
    return false;
  auto free = std::find_if(
    slots.begin(), slots.end(), [](auto const& s) { return s.status == slot::phase::free; });
  if (free == slots.end()) return false;
  // Earlier scans must get completion slots before later scans can fill the window.
  auto source = std::find_if(sources.begin(), sources.end(), [](auto const& s) {
    return s.status == source_state::phase::enumerating;
  });
  if (source == sources.end()) return false;
  // The nonblocking W check shares the wait lock: settlement cannot be lost between
  // observing exhausted credit and sleeping. It allocates no payload and performs no I/O.
  if (source->hooks.can_claim && !source->hooks.can_claim()) return false;
  auto ticket   = std::make_shared<job_ticket>();
  auto index    = static_cast<size_t>(free - slots.begin());
  ticket->st    = self;
  ticket->index = index;
  reserve_slot(index, static_cast<size_t>(source - sources.begin()));
  ticket->active = true;
  lock.unlock();
  auto work = source->hooks.claim();
  if (work) {
    // Workers cannot see this slot until enqueue; source registration is already frozen.
    free->unit = work->unit;
    if (before_submit) before_submit();
    dispatcher->enqueue([ticket = std::move(ticket), work = std::move(work->work)]() mutable {
      auto& owner = *ticket->st;
      std::unique_ptr<op::scan::scan_info> value;
      try {
        value = work();
      } catch (...) {
        owner.completion->report_error(std::current_exception());
        owner.stop();
      }
      work              = {};  // Release physical-input captures before the job ticket is returned.
      ticket->completed = true;
      owner.settle(ticket->index, std::move(value));
    });
    lock.lock();
  } else {
    lock.lock();
    ticket->completed = true;
    release_slot(index);
    source->status = source_state::phase::enumeration_done;
  }
  return true;
}

bool preparation_coordinator::state::close_finished_source(lock_type& lock)
{
  if (!batches.empty()) return false;
  for (size_t id = 0; id < sources.size(); ++id) {
    auto& source = sources[id];
    if (source.status == source_state::phase::closed ||
        source.status == source_state::phase::enumerating || source.outstanding || !can_emit(id))
      continue;
    if (source.status == source_state::phase::enumeration_done) {
      lock.unlock();
      auto final = source.hooks.coalescer->flush();
      lock.lock();
      source.status = source_state::phase::tail_flushed;
      if (final.size() > 1) throw std::logic_error("final flush exceeded its reserved output slot");
      if (!final.empty()) {
        queue_batch(id, std::move(final.front()));
        return true;  // Publish the final batch before closing its connector.
      }
    }
    // Do not retry a throwing close hook; query cleanup still closes the connectors.
    source.status = source_state::phase::closed;
    lock.unlock();
    source.hooks.close();
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
  if (options.underfilled_batch_residence)
    for (size_t id = 0; id < sources.size(); ++id) {
      if (!can_emit(id)) continue;
      if (auto first = sources[id].hooks.coalescer->first_retained_time()) {
        auto next = *first + *options.underfilled_batch_residence;
        if (!deadline || next < *deadline) deadline = next;
      }
    }
  return deadline;
}

void preparation_coordinator::state::shutdown()
{
  stop();
  dispatcher->request_stop();
  dispatcher->wait_for_all();
  {
    std::unique_lock lock(gate->mutex);
    gate->cv.wait(lock, [&] { return !jobs && !external && !gate->callbacks; });
  }
  finish();
}

void preparation_coordinator::state::finish()
{
  // Only the query owner closes sources after their workers and callbacks have settled.
  for (auto& source : sources)
    if (source.status != source_state::phase::closed) {
      try {
        source.hooks.close();
      } catch (...) {
        completion->report_error(std::current_exception());
      }
      source.status = source_state::phase::closed;
    }
  {
    std::lock_guard lock(gate->mutex);
    batches.clear();
    for (size_t index = 0; index < slots.size(); ++index)
      release_slot(index);
    gate->closed = true;
    phase        = lifecycle::quiescent;
    gate->cv.notify_all();
  }
  if (stats.runs || stats.owner != std::thread::id{}) {
    completion->close_preparation_inputs();
    completion->preparation_quiescent();
  }
}

void preparation_coordinator::run_on_query_thread()
{
  auto st = state_;
  auto& g = *st->gate;
  {
    std::lock_guard lock(g.mutex);
    if (st->phase != lifecycle::armed || st->stats.owner != std::this_thread::get_id())
      throw std::logic_error("preparation runs once on the query owner after consumers start");
    st->phase = lifecycle::running;
    ++st->stats.runs;
    st->stats.runner = std::this_thread::get_id();
  }

  try {
    for (;;) {
      std::unique_lock lock(g.mutex);
      ++st->stats.wakes;
      if (g.closed) break;
      // Deadlines run first. Every step is bounded and leaves the gate locked on return.
      size_t actions = st->emit_due_batch(lock) ? 1 : 0;
      auto available = [&] { return !g.closed && actions < st->options.max_control_work; };
      if (available() && st->publish_batch(lock)) ++actions;
      if (available() && st->advance_result(lock)) ++actions;
      if (available() && st->submit_job(lock, st)) ++actions;
      if (available() && st->close_finished_source(lock)) ++actions;
      if (st->inputs_settled()) break;
      if (actions) continue;
      auto deadline = st->nearest_deadline();
      if (deadline && !st->manual_clock)
        g.cv.wait_until(lock, *deadline);
      else
        g.cv.wait(lock);
    }
  } catch (...) {
    st->completion->report_error(std::current_exception());
  }
  st->shutdown();
}
}  // namespace sirius::scan_manager
