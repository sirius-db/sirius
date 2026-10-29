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

#include "preload/preload_executor.hpp"

#include "exec/query_lifecycle_registry.hpp"
#include "exec/thread_util.hpp"
#include "log/logging.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "memory/topology_index.hpp"
#include "preload/preload_streams.hpp"

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <cassert>
#include <format>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace sirius::preload {

namespace {

using cucascade::memory::Tier;

/// Popped tasks remembered to recognize a task_downgraded delivered after its task's pop.
constexpr std::size_t recent_pops_cap = 4096;

std::vector<cucascade::memory::memory_space*> spaces_of(
  memory::sirius_memory_reservation_manager& memory, Tier tier)
{
  std::vector<cucascade::memory::memory_space*> out;
  for (auto const* space : memory.get_memory_spaces_for_tier(tier)) {
    if (space == nullptr) { continue; }
    if (auto* mutable_space = memory.get_memory_space(tier, space->get_device_id())) {
      out.push_back(mutable_space);
    }
  }
  return out;
}

[[nodiscard]] std::size_t bytes_of(double fraction, std::size_t capacity) noexcept
{
  return static_cast<std::size_t>(fraction * static_cast<double>(capacity));
}

[[nodiscard]] int numa_of(memory::topology_index const* topology, int device) noexcept
{
  try {
    return topology != nullptr ? topology->numa_node_of(device) : -1;
  } catch (...) {
    return -1;
  }
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Construction and lifecycle
// ---------------------------------------------------------------------------------------------

preload_executor::preload_executor(event::query_event_publisher& publisher,
                                   preload_executor_config config,
                                   preload_dependencies deps)
  : event::query_event_subscriber(
      publisher, {event::event_type::task_downgraded, event::event_type::task_popped}),
    _cfg(config),
    _lifecycle(deps.lifecycle),
    _streams(deps.streams),
    _pipeline_threads(static_cast<std::size_t>(std::max(deps.pipeline_threads_per_gpu, 1))),
    _env{deps.lifecycle, deps.streams.get(), {}}
{
  if (!deps.space_capacity) {
    throw std::invalid_argument("preload_executor: space_capacity is required");
  }
  if (_cfg.device_budget > 0.0 && !_streams) {
    throw std::invalid_argument("preload_executor: device promotion needs stream pools");
  }
  build_spaces(deps);
  _env.test_hook = [this](promotion_job const& job) {
    std::shared_ptr<std::function<void(promotion_job const&)>> hook;
    {
      std::lock_guard const lk{_mu};
      hook = _test_hook;
    }
    if (hook && *hook) { (*hook)(job); }
  };
}

preload_executor::~preload_executor() { stop(); }

void preload_executor::build_spaces(preload_dependencies const& deps)
{
  auto const limits =
    [&](
      cucascade::memory::memory_space& space, double budget, double ceiling, std::string_view key) {
      auto const info = deps.space_capacity(space.get_id());
      // So that preload alone never starts a downgrade.
      if (budget > 0.0 && ceiling > info.downgrade_stop_fraction) {
        SIRIUS_LOG_WARN(
          "[preload] {} ({:.3f}) exceeds the downgrade_stop_fraction ({:.3f}) of {} space {}; "
          "clamping to it",
          key,
          ceiling,
          info.downgrade_stop_fraction,
          space.get_tier() == Tier::GPU ? "GPU" : "HOST",
          space.get_device_id());
        ceiling = info.downgrade_stop_fraction;
      }
      return space_state{
        &space, bytes_of(budget, info.capacity), bytes_of(ceiling, info.capacity), 0};
    };

  for (auto* space : spaces_of(deps.memory, Tier::HOST)) {
    _hosts.push_back(limits(*space, _cfg.host_budget, _cfg.host_ceiling, "host_ceiling"));
  }
  for (auto* space : spaces_of(deps.memory, Tier::GPU)) {
    gpu_state g{limits(*space, _cfg.device_budget, _cfg.device_ceiling, "device_ceiling")};
    g.device        = space->get_device_id();
    auto const numa = numa_of(deps.topology.get(), g.device);
    for (std::size_t h = 0; numa >= 0 && h < _hosts.size(); ++h) {
      if (_hosts[h].space->get_device_id() == numa) {
        g.local_host = h;
        break;
      }
    }
    if (!g.local_host && !_hosts.empty()) { g.local_host = 0; }
    _gpus.push_back(std::move(g));
  }
}

void preload_executor::start()
{
  std::lock_guard const lifecycle{_lifecycle_mu};
  if (_state != run_state::idle) { return; }

  log::log_noexcept([&] {
    std::string line;
    for (auto const& g : _gpus) {
      line +=
        std::format(" GPU:{} budget={}B ceiling={}B;", g.device, g.budget_bytes, g.ceiling_bytes);
    }
    for (auto const& h : _hosts) {
      line += std::format(" HOST:{} budget={}B ceiling={}B;",
                          h.space->get_device_id(),
                          h.budget_bytes,
                          h.ceiling_bytes);
    }
    SIRIUS_LOG_INFO("[preload] started:{}", line);
  });

  try {
    // One worker per GPU even without a device budget: GPU pops trigger DISK->HOST staging too.
    for (std::size_t g = 0; g < _gpus.size(); ++g) {
      _workers.emplace_back([this, g](std::stop_token const& st) { worker_loop(g, st); });
      std::ignore = exec::thread_util::set_thread_name(_workers.back(),
                                                       std::format("preload-{}", _gpus[g].device));
    }
  } catch (...) {
    join_workers();
    _state = run_state::stopped;
    throw;
  }
  _state = run_state::running;
  // Under the lifecycle lock, so a concurrent stop() cannot run between the workers starting and
  // event delivery starting.
  event::query_event_subscriber::start();
}

void preload_executor::stop() noexcept
{
  std::lock_guard const lifecycle{_lifecycle_mu};
  // First, so no hook runs, and no credit arrives, while the workers are joined.
  event::query_event_subscriber::stop();
  if (_state == run_state::stopped) { return; }
  bool const was_running = _state == run_state::running;
  _state                 = run_state::stopped;
  join_workers();
  if (!was_running) { return; }
  log::log_noexcept([&] {
    auto const c = counters();
    SIRIUS_LOG_INFO(
      "[preload] stopped: pops={} to_gpu={} ({}B) to_host={} ({}B) gated={} no_candidate={} "
      "no_room={} not_needed={} failed={} stale_event={}",
      c.pops,
      c.promoted_to_gpu,
      c.bytes_to_gpu,
      c.promoted_to_host,
      c.bytes_to_host,
      c.gated,
      c.no_candidate,
      c.no_room,
      c.not_needed,
      c.failed,
      c.stale_event);
  });
}

void preload_executor::join_workers() noexcept
{
  for (auto& w : _workers) {
    w.request_stop();
  }
  for (auto& w : _workers) {
    if (w.joinable()) { w.join(); }
  }
  _workers.clear();
}

void preload_executor::retire_query(query_id_t q) noexcept
{
  std::lock_guard const lk{_mu};
  for (auto it = _candidates.begin(); it != _candidates.end();) {
    if (it->second.query == q) {
      release_all(it->second);
      it = _candidates.erase(it);
    } else {
      ++it;
    }
  }
}

preload_counters preload_executor::counters() const
{
  std::lock_guard const lk{_mu};
  return _counters;
}

// ---------------------------------------------------------------------------------------------
// Event hooks
// ---------------------------------------------------------------------------------------------

void preload_executor::on_task_downgraded(event::event_id_t event_id,
                                          event::timestamp_t,
                                          query_id_t query_id,
                                          event::downgraded_task const& task) noexcept
{
  std::lock_guard const lk{_mu};
  ++_events_handled;
  try {
    // The downgrade executor and the scheduler publish into one mailbox in no guaranteed order,
    // so this event can arrive after its task was popped. It was published before the task was
    // re-pushed, so its id is then smaller than the pop's.
    if (auto const pop = _recent_pops.find(task.task_id);
        pop != _recent_pops.end() && pop->second > event_id) {
      ++_counters.stale_event;
      return;
    }
    // A query that grants no lease will never pop the task. Checked under _mu: retire_query runs
    // once the query refuses leases, so either it sees this candidate or this sees the refusal.
    if (!_lifecycle.try_acquire_work(query_id)) { return; }

    auto& c            = _candidates[task.task_id];
    c.query            = query_id;
    c.pipeline_id      = task.pipeline_id;
    c.preferred_device = task.preferred_device;
    c.priority         = task.priority;
    for (auto const& moved : task.inputs) {
      auto in = std::find_if(c.inputs.begin(), c.inputs.end(), [&](input_state const& i) {
        return i.batch_id == moved.batch_id;
      });
      if (in == c.inputs.end()) {
        c.inputs.push_back({.batch     = moved.batch,
                            .batch_id  = moved.batch_id,
                            .location  = moved.to.tier,
                            .gpu_bytes = moved.gpu_bytes});
        continue;
      }
      // Moved back down: a promotion it held budget for is undone.
      release(in->held);
      in->location  = moved.to.tier;
      in->gpu_bytes = std::max(in->gpu_bytes, moved.gpu_bytes);
      in->failed    = false;
    }
  } catch (...) {
    log_unexpected("task_downgraded");
  }
}

void preload_executor::on_task_popped(event::event_id_t event_id,
                                      event::timestamp_t,
                                      query_id_t,
                                      std::uint64_t task_id,
                                      std::size_t pipeline_id,
                                      int gpu,
                                      std::size_t tasks_left) noexcept
{
  std::lock_guard const lk{_mu};
  ++_events_handled;
  ++_counters.pops;
  try {
    remember_pop(task_id, event_id);
    // The task runs now: its inputs leave preload's budget.
    if (auto const it = _candidates.find(task_id); it != _candidates.end()) {
      release_all(it->second);
      _candidates.erase(it);
    }
    auto const g = std::find_if(
      _gpus.begin(), _gpus.end(), [gpu](gpu_state const& s) { return s.device == gpu; });
    if (g == _gpus.end()) { return; }
    // Pops that arrive while the worker is busy wait as credits; past the cap the oldest go.
    if (g->credits.size() >= _pipeline_threads) { g->credits.pop_front(); }
    g->credits.push_back({tasks_left, pipeline_id});
    _work_cv.notify_all();
  } catch (...) {
    log_unexpected("task_popped");
  }
}

void preload_executor::remember_pop(std::uint64_t task_id, event::event_id_t event_id)
{
  if (auto const it = _recent_pops.find(task_id); it != _recent_pops.end()) {
    it->second = std::max(it->second, event_id);
    return;
  }
  _recent_pop_order.push_back(task_id);
  _recent_pops.emplace(task_id, event_id);
  if (_recent_pop_order.size() > recent_pops_cap) {
    _recent_pops.erase(_recent_pop_order.front());
    _recent_pop_order.pop_front();
  }
}

// ---------------------------------------------------------------------------------------------
// Workers
// ---------------------------------------------------------------------------------------------

void preload_executor::worker_loop(std::size_t g, std::stop_token const& st) noexcept
{
  auto& gpu = _gpus[g];
  std::unique_lock lk{_mu};
  while (true) {
    _work_cv.wait(lk, st, [&] { return !gpu.credits.empty(); });
    // The wait also returns with credits left once stop is requested.
    if (st.stop_requested()) { return; }
    auto const credit = gpu.credits.front();
    gpu.credits.pop_front();
    std::optional<promotion_job> job;
    try {
      job = select(g, credit);
    } catch (...) {
      log_unexpected("select");
    }
    if (job) {
      gpu.busy = true;
      lk.unlock();
      auto const result = promote_one(*job, _env);
      lk.lock();
      apply(*job, result);
      gpu.busy = false;
    }
    _idle_cv.notify_all();
  }
}

std::optional<promotion_job> preload_executor::select(std::size_t g, pop_credit const& credit)
{
  auto& gpu              = _gpus[g];
  bool const short_queue = credit.tasks_left <= _pipeline_threads;
  bool any_for_gpu       = false;
  std::vector<std::pair<std::pair<exec::queue_priority, std::uint64_t>, candidate_task*>> eligible;
  for (auto it = _candidates.begin(); it != _candidates.end();) {
    auto& c = it->second;
    for (auto in = c.inputs.begin(); in != c.inputs.end();) {
      if (in->batch.expired()) {
        release(in->held);
        in = c.inputs.erase(in);
      } else {
        ++in;
      }
    }
    if (c.inputs.empty()) {
      it = _candidates.erase(it);
      continue;
    }
    if (!c.preferred_device || *c.preferred_device == gpu.device) {
      any_for_gpu = true;
      if (short_queue || c.pipeline_id == credit.pipeline_id) {
        eligible.push_back({{c.priority, it->first}, &c});
      }
    }
    ++it;
  }
  if (eligible.empty()) {
    ++(any_for_gpu ? _counters.gated : _counters.no_candidate);
    return std::nullopt;
  }

  // Pop order: the inputs of the task that runs first are needed first.
  std::sort(eligible.begin(), eligible.end(), [](auto const& a, auto const& b) {
    return a.first < b.first;
  });
  for (auto const& [key, c] : eligible) {
    for (auto& in : c->inputs) {
      if (in.in_flight || in.failed || in.location == Tier::GPU) { continue; }
      auto const est = with_slack(std::max<std::size_t>(in.gpu_bytes, 1));
      std::optional<charge> target;
      if (_cfg.device_budget > 0.0 && fits(gpu, est)) {
        target = charge{Tier::GPU, g, est};
      } else if (in.location == Tier::DISK && _cfg.host_budget > 0.0 && gpu.local_host &&
                 fits(_hosts[*gpu.local_host], est)) {
        target = charge{Tier::HOST, *gpu.local_host, est};
      }
      if (!target) { continue; }
      auto const& s = state_of(*target);
      promotion_job job{.batch         = in.batch,
                        .batch_id      = in.batch_id,
                        .task_id       = key.second,
                        .query         = c->query,
                        .tier          = target->tier,
                        .space         = target->space,
                        .target        = s.space,
                        .est_bytes     = est,
                        .ceiling_bytes = s.ceiling_bytes};
      add(*target);
      in.in_flight = true;
      return job;
    }
  }
  ++_counters.no_room;
  return std::nullopt;
}

void preload_executor::apply(promotion_job const& job, promotion_result const& result) noexcept
{
  // The job's charge always ends here; a promotion re-charges the exact bytes to its input.
  charge spent{job.tier, job.space, job.est_bytes};
  release(spent);
  switch (result.outcome) {
    case promotion_outcome::promoted:
      if (job.tier == Tier::GPU) {
        ++_counters.promoted_to_gpu;
        _counters.bytes_to_gpu += result.bytes;
      } else {
        ++_counters.promoted_to_host;
        _counters.bytes_to_host += result.bytes;
      }
      break;
    case promotion_outcome::not_needed: ++_counters.not_needed; break;
    case promotion_outcome::no_room: ++_counters.no_room; break;
    case promotion_outcome::failed: ++_counters.failed; break;
  }

  // Popped or retired while the job ran: nothing to record.
  auto const task = _candidates.find(job.task_id);
  if (task == _candidates.end()) { return; }
  auto& inputs  = task->second.inputs;
  auto const in = std::find_if(
    inputs.begin(), inputs.end(), [&](input_state const& i) { return i.batch_id == job.batch_id; });
  if (in == inputs.end()) { return; }
  in->in_flight = false;
  // A re-downgrade of this input may be applied before or after this, whichever happened first:
  // either way, what it holds is released at the latest when its task is popped or its query
  // retires.
  switch (result.outcome) {
    case promotion_outcome::promoted:
      release(in->held);
      in->held = {job.tier, job.space, result.bytes};
      add(in->held);
      in->location = job.tier;
      break;
    case promotion_outcome::not_needed:
      if (result.found != Tier::SIZE) { in->location = result.found; }
      break;
    case promotion_outcome::failed: in->failed = true; break;
    case promotion_outcome::no_room: break;
  }
  if (in->batch.expired()) {
    release(in->held);
    inputs.erase(in);
    if (inputs.empty()) { _candidates.erase(task); }
  }
}

// ---------------------------------------------------------------------------------------------
// Budget
// ---------------------------------------------------------------------------------------------

bool preload_executor::fits(space_state const& s, std::size_t bytes) const
{
  auto const consumed = consumed_bytes(s.space->get_max_memory(), s.space->get_available_memory());
  return sat_add(s.outstanding, bytes) <= s.budget_bytes &&
         sat_add(consumed, bytes) <= s.ceiling_bytes;
}

preload_executor::space_state& preload_executor::state_of(charge const& c) noexcept
{
  return c.tier == Tier::GPU ? static_cast<space_state&>(_gpus[c.space]) : _hosts[c.space];
}

void preload_executor::add(charge const& c) noexcept
{
  if (c.bytes == 0) { return; }
  auto& s       = state_of(c);
  s.outstanding = sat_add(s.outstanding, c.bytes);
}

void preload_executor::release(charge& c) noexcept
{
  if (c.bytes == 0) { return; }
  auto& s = state_of(c);
  assert(s.outstanding >= c.bytes);
  s.outstanding -= std::min(s.outstanding, c.bytes);
  c = {};
}

void preload_executor::release_all(candidate_task& c) noexcept
{
  for (auto& in : c.inputs) {
    release(in.held);
  }
}

bool preload_executor::idle() const noexcept
{
  return std::all_of(
    _gpus.begin(), _gpus.end(), [](gpu_state const& g) { return g.credits.empty() && !g.busy; });
}

void preload_executor::log_unexpected(std::string_view where) noexcept
{
  // Warn once: a failure that repeats on every event must not flood the log.
  bool const first = !std::exchange(_unexpected_logged, true);
  log::log_noexcept([&] {
    if (first) {
      SIRIUS_LOG_WARN("[preload] unexpected error in {}; preload continues", where);
    } else {
      SIRIUS_LOG_DEBUG("[preload] unexpected error in {}", where);
    }
  });
}

// ---------------------------------------------------------------------------------------------
// Tests only
// ---------------------------------------------------------------------------------------------

std::uint64_t preload_executor::events_handled_for_testing() const
{
  std::lock_guard const lk{_mu};
  return _events_handled;
}

bool preload_executor::wait_idle_for_testing(std::chrono::milliseconds timeout) const
{
  std::unique_lock lk{_mu};
  return _idle_cv.wait_for(lk, timeout, [&] { return idle(); });
}

std::size_t preload_executor::outstanding_for_testing(cucascade::memory::memory_space_id id) const
{
  std::lock_guard const lk{_mu};
  auto const matches = [&](space_state const& s) { return s.space->get_id() == id; };
  if (auto const g = std::find_if(_gpus.begin(), _gpus.end(), matches); g != _gpus.end()) {
    return g->outstanding;
  }
  auto const h = std::find_if(_hosts.begin(), _hosts.end(), matches);
  return h == _hosts.end() ? 0 : h->outstanding;
}

std::size_t preload_executor::candidate_count_for_testing() const
{
  std::lock_guard const lk{_mu};
  return _candidates.size();
}

void preload_executor::set_promotion_hook_for_testing(
  std::function<void(promotion_job const&)> hook)
{
  auto shared = std::make_shared<std::function<void(promotion_job const&)>>(std::move(hook));
  std::lock_guard const lk{_mu};
  _test_hook = std::move(shared);
}

}  // namespace sirius::preload
