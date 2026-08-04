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

#include "scan_manager/prefetching_state_manager.hpp"

#include "log/logging.hpp"
#include "planner/query.hpp"

#include <atomic>
#include <format>

namespace sirius::scan_manager {

namespace {

/// Every counter access in this file uses relaxed ordering, mirroring
/// io::cache::prefetching_cache::counters. The counters are diagnostics and a scheduling
/// heuristic; nothing is published through them, so no counter ever needs to order any other
/// memory. That is what keeps the mutators usable from a destructor, from a GPU executor thread
/// mid-pipeline, and from a task_creator worker that is holding sirius_pipeline::_status_mutex.
constexpr auto kOrder = std::memory_order_relaxed;

}  // namespace

prefetching_state_manager::prefetching_state_manager(config cfg) noexcept : _cfg(cfg) {}

void prefetching_state_manager::prepare_for_query(const sirius::planner::query& query) noexcept
{
  // The id is all that is kept: planner::query is destroyed before the scan manager is reset, so
  // a retained reference would dangle by the time clean_up() runs.
  _query_id.store(query.query_id(), kOrder);

  _counters.n_inputs_created.store(0, kOrder);
  _counters.n_inputs_disposed.store(0, kOrder);
  _counters.n_metadata_created.store(0, kOrder);
  _counters.n_task_queued.store(0, kOrder);
  _counters.n_task_prepared.store(0, kOrder);
  _counters.n_task_completed.store(0, kOrder);
  _counters.n_live.store(0, kOrder);

  // Re-attach: a manager that was cleaned up and then bound to a new query must serve hooks again.
  // Cleared after the counters are zeroed, so the statement order reads "reset, then re-arm".
  // Relaxed like every other access here; nothing is published through these.
  _detached.store(false, kOrder);
}

void prefetching_state_manager::clean_up() noexcept
{
  // summary() builds a std::string and the log call formats it, either of which can throw. This
  // method is noexcept and runs on the query teardown path, so an escaping exception would call
  // std::terminate: losing one diagnostic line is always the better trade.
  try {
    SIRIUS_LOG_TRACE("{}", summary());
  } catch (...) {  // deliberately swallowed: the log line is the only casualty
  }

  // Detach. Be precise about *when* this lands, because the ordering is the opposite of what
  // "detach first" would suggest:
  //
  //   SiriusContext::run_mandatory_cleanup
  //     query_.reset()                 <- destroys the query's pipelines and, with them, every
  //                                       split_connector. The gate is still OPEN here.
  //     ... drain, telemetry, repositories ...
  //     scan_manager_->reset()         -> sirius_scan_manager::reset -> clean_up() -> this store.
  //
  // So the flag is latched *after* the connectors are already gone, and the gate stands open
  // across that entire interval with nothing behind it. Latching it here closes only the tail of
  // the window -- from this store until the manager itself dies -- which is still worth having,
  // because that tail is unbounded in principle (the manager outlives the scan manager for as long
  // as any straggler split holds a shared_ptr to it).
  //
  // The weak_ptr the hooks capture does not cover the interval either, which is why this flag
  // exists at all: a straggler split still holds a shared_ptr to this manager, so the weak_ptr
  // does NOT expire and lock() succeeds -- precisely when the connectors have already been
  // destroyed.
  //
  // WARNING (mirrored as an @warning on clean_up's declaration): whoever wires the TODO'd
  // split_connector::prefetch_if walk into on_task_queue_depleted / on_task_not_created must NOT
  // read `_detached == false` as "this query's connectors are alive". It does not mean that in
  // either direction: it is false throughout the post-query_.reset() interval above, and a hook
  // that had already passed the check when this store lands is still running. That walk needs the
  // connectors' own lifetime guarantee -- shared ownership of each connector it touches, or a
  // driver whose lifetime is tied to them -- and this gate on top, never this gate alone.
  _detached.store(true, kOrder);

  // The counters are deliberately left alone: a straggler split can still be destroyed after this
  // point (~split_connector runs when the query's pipelines die, before the scan manager is reset,
  // and a task can be in flight on a GPU executor thread), and its decrement must land somewhere
  // harmless rather than be raced against a reset.
  _query_id.store(sirius::query_id_t{}, kOrder);
}

sirius::query_id_t prefetching_state_manager::query_id() const noexcept
{
  return _query_id.load(kOrder);
}

void prefetching_state_manager::update(io::cache::prefetching_stage site) noexcept
{
  switch (site) {
    case io::cache::prefetching_stage::metadata_created:
      _counters.n_metadata_created.fetch_add(1, kOrder);
      break;
    case io::cache::prefetching_stage::task_queued:
      _counters.n_task_queued.fetch_add(1, kOrder);
      break;
    case io::cache::prefetching_stage::task_preprocessing:
      _counters.n_task_prepared.fetch_add(1, kOrder);
      break;
    case io::cache::prefetching_stage::disposable:
      _counters.n_task_completed.fetch_add(1, kOrder);
      break;
    // `none` is not a rung: io_context uses it to mean "this backend never wants prefetch
    // activated", so counting it would report progress that never happened.
    case io::cache::prefetching_stage::none: break;
  }
}

void prefetching_state_manager::on_input_created() noexcept
{
  _counters.n_inputs_created.fetch_add(1, kOrder);
  _counters.n_live.fetch_add(1, kOrder);
}

void prefetching_state_manager::on_input_disposed() noexcept
{
  _counters.n_inputs_disposed.fetch_add(1, kOrder);
  _counters.n_live.fetch_sub(1, kOrder);
}

void prefetching_state_manager::on_task_queue_depleted() noexcept
{
  // The detach gate, and the first statement on purpose. Once clean_up() has run, this query's
  // split_connectors have certainly been destroyed with its pipelines, so anything this hook would
  // walk is gone. The weak_ptr the hook captures does not protect that window: a straggler split
  // still holds a shared_ptr here, so the lock() succeeds.
  //
  // A gate, not a barrier, and it narrows the window from one side only. `false` does NOT mean the
  // connectors are alive -- clean_up() runs well after query_.reset() has destroyed them (see the
  // teardown sequence spelled out there) -- and a hook that had already passed this line when
  // clean_up() ran is still inside. Whatever goes below must therefore hold its own guarantee that
  // what it walks outlives the walk.
  if (_detached.load(kOrder)) { return; }

  // TODO: drive the bounded split_connector::prefetch_if walk from here, once the query's
  // connectors reach this object with the scan-manager wiring. Until then the hook is installable
  // and inert rather than absent. Three constraints on that walk, all load-bearing:
  //   - budget: one relaxed counter read plus one prefetch_if call bounded by
  //     _cfg.prefetch_lookahead_window. This runs on the task_scheduler management thread, which
  //     is also the thread that matches every ready device to a task.
  //   - try/catch(...): prefetch_if acquires split_connector::_mutex and runs a caller-supplied
  //     predicate, either of which can throw, and this method is noexcept -- an escaping exception
  //     calls std::terminate. Same for on_task_not_created below, where it would instead take down
  //     the engine's single task-creation thread.
  //   - lock rank: the predicate passed to prefetch_if runs at L2 and must acquire nothing of rank
  //     L1 or lower. See split_connector::prefetch_if's @warning.
}

void prefetching_state_manager::on_task_not_created(
  const op::sirius_physical_operator* /*requested*/, creator::request_type /*kind*/) noexcept
{
  // See on_task_queue_depleted: the same detach gate, and the same TODO and its three constraints
  // apply to the walk that will go here.
  if (_detached.load(kOrder)) { return; }
}

bool prefetching_state_manager::is_detached() const noexcept { return _detached.load(kOrder); }

prefetching_state_manager::counters_snapshot prefetching_state_manager::snapshot() const noexcept
{
  // Seven independent relaxed loads, so the result need not correspond to any single instant.
  // Documented and accepted: this feeds a log line and a look-ahead heuristic, never a decision
  // that has to be consistent.
  return counters_snapshot{
    .n_inputs_created   = _counters.n_inputs_created.load(kOrder),
    .n_inputs_disposed  = _counters.n_inputs_disposed.load(kOrder),
    .n_metadata_created = _counters.n_metadata_created.load(kOrder),
    .n_task_queued      = _counters.n_task_queued.load(kOrder),
    .n_task_prepared    = _counters.n_task_prepared.load(kOrder),
    .n_task_completed   = _counters.n_task_completed.load(kOrder),
    .n_live             = _counters.n_live.load(kOrder),
  };
}

std::string prefetching_state_manager::summary() const
{
  auto const counters = snapshot();
  return std::format(
    "prefetching_state_manager: query={} "
    "inputs[created={} disposed={} live={}] "
    "ladder[metadata_created={} task_queued={} task_prepared={} task_completed={}]",
    _query_id.load(kOrder),
    counters.n_inputs_created,
    counters.n_inputs_disposed,
    counters.n_live,
    counters.n_metadata_created,
    counters.n_task_queued,
    counters.n_task_prepared,
    counters.n_task_completed);
}

const prefetching_state_manager::config& prefetching_state_manager::get_config() const noexcept
{
  return _cfg;
}

}  // namespace sirius::scan_manager
