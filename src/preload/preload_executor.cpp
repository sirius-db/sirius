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

#include "data/convertible_gpu_pipeline_task.hpp"
#include "exec/multi_index_priority_queue.hpp"
#include "exec/query_lifecycle_registry.hpp"
#include "exec/thread_util.hpp"
#include "log/logging.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/partition_placement.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"
#include "parallel/task.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "preload/preload_streams.hpp"
#include "telemetry/nvtx.hpp"

#include <cuda_runtime_api.h>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <exception>
#include <format>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace sirius::preload {

namespace {

using cucascade::memory::Tier;

/// Why the planner must run another pass.
namespace dirty {
constexpr std::uint32_t candidates = 1U << 0;
constexpr std::uint32_t memory     = 1U << 1;
constexpr std::uint32_t queue      = 1U << 2;
constexpr std::uint32_t ledger     = 1U << 3;
constexpr std::uint32_t readiness  = 1U << 4;
constexpr std::uint32_t idle       = 1U << 5;
constexpr std::uint32_t lane_done  = 1U << 6;
}  // namespace dirty

/// Repository scans per planner pass: bounds the pass whatever the number of repositories.
constexpr std::size_t max_repository_scans = 64;

/// Expired candidates checked per pass.
constexpr std::size_t prune_budget = 256;

/// The executor whose event hook is running on this thread, if any.
thread_local void const* t_hook_owner = nullptr;

/// Marks the current thread as running an event hook of @p owner for the guard's lifetime.
class hook_scope {
 public:
  explicit hook_scope(void const* owner) noexcept : _previous(t_hook_owner)
  {
    t_hook_owner = owner;
  }
  ~hook_scope() { t_hook_owner = _previous; }

  hook_scope(hook_scope const&)            = delete;
  hook_scope& operator=(hook_scope const&) = delete;
  hook_scope(hook_scope&&)                 = delete;
  hook_scope& operator=(hook_scope&&)      = delete;

 private:
  void const* _previous;
};

template <typename T>
std::shared_ptr<T> required(std::shared_ptr<T> p, char const* what)
{
  if (!p) { throw std::invalid_argument(std::string{"preload_executor: missing "} + what); }
  return p;
}

std::vector<cucascade::memory::memory_space*> spaces_of(
  cucascade::memory::memory_reservation_manager& memory, Tier tier)
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

std::vector<int> device_ids(std::vector<cucascade::memory::memory_space*> const& spaces)
{
  std::vector<int> ids;
  ids.reserve(spaces.size());
  for (auto const* space : spaces) {
    ids.push_back(space->get_device_id());
  }
  return ids;
}

counter refusal_counter(refusal r) noexcept
{
  switch (r) {
    case refusal::held_off: return counter::refused_held_off;
    case refusal::epoch: return counter::refused_epoch;
    case refusal::trigger: return counter::refused_trigger;
    case refusal::ceiling: return counter::refused_ceiling;
    case refusal::window: return counter::refused_window;
    case refusal::speculative_window: return counter::refused_speculative_window;
    case refusal::inflight: return counter::refused_inflight;
    case refusal::batch_too_big: return counter::refused_batch_too_big;
    case refusal::none:
    case refusal::no_target: break;
  }
  return counter::refused_no_target;
}

/// A budget for the per-pass log: the bytes it allows, or the refusal that closed it.
std::string budget_text(std::size_t bytes, refusal closed)
{
  if (closed != refusal::none) { return std::string{counter_name(refusal_counter(closed))}; }
  return std::format("{}MiB", bytes >> 20);
}

[[maybe_unused]] std::string_view outcome_name(promotion_outcome o) noexcept
{
  switch (o) {
    case promotion_outcome::promoted: return "promoted";
    case promotion_outcome::retired: return "retired";
    case promotion_outcome::expired: return "expired";
    case promotion_outcome::contended: return "contended";
    case promotion_outcome::stale: return "stale";
    case promotion_outcome::already_resident: return "already_resident";
    case promotion_outcome::unsupported: return "unsupported";
    case promotion_outcome::too_big: return "too_big";
    case promotion_outcome::no_memory: return "no_memory";
    case promotion_outcome::gated: return "gated";
    case promotion_outcome::held_off: return "held_off";
    case promotion_outcome::oom: return "oom";
    case promotion_outcome::failed: break;
  }
  return "failed";
}

[[maybe_unused]] std::string_view tier_name(Tier tier) noexcept
{
  switch (tier) {
    case Tier::GPU: return "GPU";
    case Tier::HOST: return "HOST";
    case Tier::DISK: return "DISK";
    default: break;
  }
  return "?";
}

/// Time and duration counters of a promotion kind.
std::pair<counter, counter> timing_counters(promotion_kind kind) noexcept
{
  switch (kind) {
    case promotion_kind::d2g: return {counter::promote_us_sum_d2g, counter::promote_us_max_d2g};
    case promotion_kind::d2h: return {counter::promote_us_sum_d2h, counter::promote_us_max_d2h};
    case promotion_kind::relocate_d2h:
      return {counter::promote_us_sum_relocate_d2h, counter::promote_us_max_relocate_d2h};
    case promotion_kind::h2g:
    case promotion_kind::none: break;
  }
  return {counter::promote_us_sum_h2g, counter::promote_us_max_h2g};
}

/// Device of @p partition under @p consumer's placement; empty when unpinned or unknown.
std::optional<int> placement_device(op::sirius_physical_partition_consumer_operator const& consumer,
                                    std::size_t partition) noexcept
{
  try {
    auto const placement = consumer.placement();
    return placement ? placement->device_for(partition) : std::nullopt;
  } catch (...) {
    return std::nullopt;
  }
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Repository discovery
// ---------------------------------------------------------------------------------------------

std::vector<repo_info> repositories_of(op::sirius_physical_operator const& op)
{
  std::vector<std::weak_ptr<pipeline::sirius_pipeline>> full_sources;
  for (auto const& port : op.get_ports()) {
    if (port && port->type == op::MemoryBarrierType::FULL && port->src_pipeline) {
      full_sources.push_back(port->src_pipeline);
    }
  }
  auto const* consumer = dynamic_cast<op::sirius_physical_partition_consumer_operator const*>(&op);
  auto const own       = op.get_pipeline();

  std::vector<repo_info> out;
  for (auto const& port : op.get_ports()) {
    if (!port || port->repo == nullptr) { continue; }
    repo_info info;
    info.token = reinterpret_cast<std::uintptr_t>(port->repo);
    info.repo  = port->repo;
    if (port->dest_pipeline) {
      info.dest_priority = port->dest_pipeline->get_priority();
    } else if (own) {
      info.dest_priority = own->get_priority();
    }
    info.partition_consumer = consumer;
    info.full_sources       = full_sources;
    out.push_back(std::move(info));
  }
  return out;
}

std::vector<repo_info> collect_repositories(
  std::span<std::shared_ptr<pipeline::sirius_pipeline> const> pipelines)
{
  std::vector<repo_info> out;
  for (auto const& pipeline : pipelines) {
    if (!pipeline) { continue; }
    for (auto const& op : std::as_const(*pipeline).get_operators()) {
      auto repos = repositories_of(op.get());
      std::move(repos.begin(), repos.end(), std::back_inserter(out));
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Lane slot
// ---------------------------------------------------------------------------------------------

/**
 * Holds a lane's concurrency slot, the target GPU's HOST->GPU token when the job needs one, and
 * the job's bytes in the target's in-flight count (and the GPU's class-B count for class B).
 * Construct and release with the executor's mutex held; release is idempotent and runs before
 * any bookkeeping that can throw. Construction also adds the bytes to the target's unreserved
 * count, which @ref promote_one takes back.
 */
class preload_executor::lane_slot {
 public:
  lane_slot(preload_executor& ex, lane& l, taken_job const& taken) noexcept
    : _ex(ex), _lane(l), _t(taken.job.t), _bytes(taken.job.est_reserve), _token(taken.token)
  {
    ++_lane.active;
    if (_token) { _ex._store.acquire_h2g_token(_t.index); }
    _ex._store.add_inflight(_t, _bytes);
    if (auto* const unreserved = _ex.unreserved_of(_t)) {
      unreserved->fetch_add(_bytes, std::memory_order_relaxed);
    }
    _ex._counters.raise_lane_concurrency(_lane.idx, static_cast<std::uint64_t>(_lane.active));
    if (_t.tier == Tier::GPU) {
      _ex._counters.raise_gpu_inflight_bytes(_t.index, _ex._store.inflight_bytes(_t));
    }
  }

  ~lane_slot() { release(); }

  lane_slot(lane_slot const&)            = delete;
  lane_slot& operator=(lane_slot const&) = delete;
  lane_slot(lane_slot&&)                 = delete;
  lane_slot& operator=(lane_slot&&)      = delete;

  void release() noexcept
  {
    if (!_held) { return; }
    _held = false;
    --_lane.active;
    if (_token) { _ex._store.release_h2g_token(_t.index); }
    _ex._store.remove_inflight(_t, _bytes);
  }

 private:
  preload_executor& _ex;
  lane& _lane;
  target _t;
  std::size_t _bytes;
  bool _token;
  bool _held{true};
};

// ---------------------------------------------------------------------------------------------
// Construction and lifecycle
// ---------------------------------------------------------------------------------------------

preload_executor::preload_executor(event::query_event_publisher& publisher,
                                   preload_executor_config config,
                                   preload_dependencies deps)
  : event::query_event_subscriber(publisher,
                                  {event::event_type::batch_downgraded,
                                   event::event_type::task_completed,
                                   event::event_type::memory_wait_ended_for_task,
                                   event::event_type::task_created,
                                   event::event_type::task_deployed,
                                   event::event_type::pipeline_closed,
                                   event::event_type::task_queue_empty,
                                   event::event_type::executor_awaiting_task},
                                  /*drain_on_stop=*/false),
    _cfg(std::move(config)),
    _lifecycle(deps.lifecycle),
    _memory(deps.memory),
    _topology(std::move(deps.topology)),
    _queue(deps.task_queue),
    _board(required(std::move(deps.wait_board), "wait board")),
    _streams(std::move(deps.streams)),
    _budget(std::move(deps.device_budget)),
    _device_preloading(_budget != nullptr && _cfg.device_preloading_budget > 0.0),
    _space_capacity(deps.space_capacity ? std::move(deps.space_capacity)
                                        : throw std::invalid_argument(
                                            "preload_executor: space_capacity is required")),
    _caps(deps.caps),
    _head_margin(resolve_head_margin(_cfg, deps.pipeline_threads_per_gpu)),
    _disks(spaces_of(_memory, Tier::DISK)),
    _counters(spaces_of(_memory, Tier::GPU).size() + _disks.size(),
              spaces_of(_memory, Tier::GPU).size()),
    _env{_lifecycle, *_board, {}, {}, {}},
    _store(_cfg,
           device_ids(spaces_of(_memory, Tier::GPU)),
           device_ids(spaces_of(_memory, Tier::HOST)),
           _counters,
           _budget.get())
{
  // A device preloading budget board requires preload's exclusive streams: both or neither.
  if (_device_preloading && !_streams) {
    throw std::invalid_argument("preload_executor: device preloading needs stream pools");
  }
  build_spaces();
  build_lanes();
  _unreserved    = std::make_unique<std::atomic<std::size_t>[]>(_gpus.size() + _hosts.size());
  _env.gpus      = _promotion_gpus;
  _env.hosts     = _promotion_hosts;
  _env.test_hook = [this](promotion_job const& job) {
    std::shared_ptr<std::function<void(promotion_job const&)>> hook;
    {
      std::lock_guard const g{_hook_mu};
      hook = _hook;
    }
    if (hook && *hook) { (*hook)(job); }
  };
  _env.post_convert_hook = [this](promotion_job const& job) {
    std::shared_ptr<std::function<void(promotion_job const&)>> hook;
    {
      std::lock_guard const g{_hook_mu};
      hook = _post_convert_hook;
    }
    if (hook && *hook) { (*hook)(job); }
  };
}

preload_executor::~preload_executor() { stop(); }

space_limits preload_executor::limits_of(
  cucascade::memory::memory_space const& space) const noexcept
{
  auto const info = _space_capacity(space.get_id());
  auto const stop =
    static_cast<std::size_t>(static_cast<double>(info.capacity) * info.downgrade_stop_fraction);
  return {info.capacity, space.get_max_memory(), stop};
}

void preload_executor::build_spaces()
{
  for (auto* space : spaces_of(_memory, Tier::GPU)) {
    gpu_state g;
    g.space             = space;
    g.device            = space->get_device_id();
    g.numa              = numa_of(g.device);
    g.c2c               = _caps.c2c ? *_caps.c2c : (_topology && gpu_has_c2c(*_topology, g.device));
    g.h2g_cap           = std::max(h2g_cap(_cfg, g.c2c), 1);
    g.capacity          = limits_of(*space).capacity;
    g.ceilings          = gpu_ceilings_bytes(_cfg, limits_of(*space));
    g.lookahead_bytes   = fraction_of(_cfg.device_preloading_budget, g.capacity);
    g.speculative_bytes = fraction_of(_cfg.speculative_fraction, g.capacity);
    g.max_inflight_bytes = fraction_of(_cfg.max_inflight_fraction, g.capacity);
    // A batch above the in-flight cap could never be dispatched.
    g.max_batch_bytes =
      std::min(fraction_of(_cfg.max_batch_fraction, g.capacity), g.max_inflight_bytes);
    _gpus.push_back(g);
  }
  for (auto* space : spaces_of(_memory, Tier::HOST)) {
    host_state h;
    h.space    = space;
    h.device   = space->get_device_id();
    h.capacity = limits_of(*space).capacity;
    h.ceiling  = _cfg.stage_to_host ? host_ceiling_bytes(_cfg, limits_of(*space)) : 0;
    _hosts.push_back(std::move(h));
  }
  // A GPU's local HOST is the one on its NUMA node; with a single HOST space, that one.
  for (std::size_t i = 0; i < _gpus.size(); ++i) {
    auto& g = _gpus[i];
    if (g.numa >= 0) {
      for (std::size_t j = 0; j < _hosts.size(); ++j) {
        if (_hosts[j].device == g.numa) {
          g.local_host = j;
          break;
        }
      }
    }
    if (!g.local_host && _hosts.size() == 1) { g.local_host = 0; }
    if (g.local_host) { _hosts[*g.local_host].local_gpus.push_back(i); }
  }

  for (auto const& g : _gpus) {
    // A GPU promotion needs preload's exclusive stream pool, which exists only with device
    // preloading; staging-only mode never takes a GPU job, so a null pool here is never used.
    auto* const streams = _device_preloading ? _streams->for_device(g.device) : nullptr;
    _promotion_gpus.push_back({g.space, g.max_batch_bytes, streams});
  }
  for (auto const& h : _hosts) {
    // DISK->HOST copies do not run on the stream; it only orders the conversion. With device
    // preloading, borrow a local GPU's pool, else the first GPU's; otherwise use the calling
    // thread's default stream (null).
    cucascade::memory::exclusive_stream_pool* ordering = nullptr;
    if (_device_preloading && !h.local_gpus.empty()) {
      ordering = _streams->for_device(_gpus[h.local_gpus.front()].device);
    } else if (_device_preloading && !_gpus.empty()) {
      ordering = _streams->for_device(_gpus.front().device);
    }
    _promotion_hosts.push_back({h.space, h.local_gpus, ordering});
  }
}

void preload_executor::build_lanes()
{
  _gpu_lane.assign(_gpus.size(), std::nullopt);
  _disk_lane.assign(_disks.size(), 0);
  // GPU HOST->GPU lanes exist only with device preloading; staging-only mode has DISK lanes only,
  // so DISK lanes do not always start at index _gpus.size(). Record the exact lane index of each
  // space (see @ref lane_of).
  if (_device_preloading) {
    for (std::size_t i = 0; i < _gpus.size(); ++i) {
      auto l           = std::make_unique<lane>();
      l->idx           = _lanes.size();
      l->kind          = lane_kind::h2g;
      l->space_index   = i;
      l->space         = _gpus[i].space;
      l->threads       = _gpus[i].h2g_cap;
      l->effective_cap = _gpus[i].h2g_cap;
      l->refined       = true;
      _gpu_lane[i]     = l->idx;
      _lanes.push_back(std::move(l));
    }
  }
  for (std::size_t k = 0; k < _disks.size(); ++k) {
    auto l         = std::make_unique<lane>();
    l->idx         = _lanes.size();
    l->kind        = lane_kind::disk;
    l->space_index = k;
    l->space       = _disks[k];
    _disk_lane[k]  = l->idx;
    _lanes.push_back(std::move(l));
  }
}

void preload_executor::start()
{
  {
    std::lock_guard const lifecycle{_lifecycle_mu};
    if (_state != run_state::idle) { return; }

    // Probing GPUDirect Storage may open the cuFile driver: do it once, here, and only for DISK
    // spaces that exist.
    std::vector<bool> gds(_lanes.size(), false);
    for (auto const& l : _lanes) {
      if (l->kind != lane_kind::disk) { continue; }
      if (_caps.gds) {
        gds[l->idx] = *_caps.gds;
        continue;
      }
      try {
        gds[l->idx] = disk_uses_gds(*l->space);
      } catch (std::exception const& e) {
        // The probe is advisory: without it, the lane runs as if there were no GDS.
        SIRIUS_LOG_WARN("[preload] DISK:{} GPUDirect Storage probe failed ({}); assuming no GDS",
                        l->space->get_device_id(),
                        e.what());
      } catch (...) {
        SIRIUS_LOG_WARN("[preload] DISK:{} GPUDirect Storage probe failed; assuming no GDS",
                        l->space->get_device_id());
      }
    }
    {
      std::lock_guard const lk{_mu};
      for (auto const& l : _lanes) {
        if (l->kind != lane_kind::disk) { continue; }
        l->gds           = gds[l->idx];
        l->threads       = std::max(disk_cap(_cfg, l->gds), 1);
        l->effective_cap = l->threads;
        // Auto mode with GDS: the cap still depends on the io context the space adopts.
        l->refined = _caps.gds.has_value() || _cfg.concurrent_preloads.has_value() || !l->gds;
      }
    }
    log_start();

    std::vector<int> cpus;
    for (int const cpu : _cfg.cpu_affinity_list) {
      if (cpu >= 0) { cpus.push_back(cpu); }
    }
    if (cpus.size() != _cfg.cpu_affinity_list.size()) {
      SIRIUS_LOG_WARN("[preload] ignoring negative core ids in cpu_affinity");
    }
    auto const pin = [&](std::jthread& t) {
      if (!cpus.empty() && !exec::thread_util::set_thread_affinity(t, cpus)) {
        SIRIUS_LOG_WARN("[preload] could not pin a preload thread to cpu_affinity");
      }
    };

    try {
      _planner    = std::jthread([this](std::stop_token const& st) { planner_loop(st); });
      std::ignore = exec::thread_util::set_thread_name(_planner, "preload-plan");
      pin(_planner);
      for (auto const& l : _lanes) {
        auto* const lp = l.get();
        for (int i = 0; i < lp->threads; ++i) {
          lp->workers.emplace_back([this, lp](std::stop_token const& st) { lane_loop(*lp, st); });
          auto const name =
            std::format("preload-{}-{}-{}",
                        lp->kind == lane_kind::h2g ? "h2g" : "dsk",
                        lp->kind == lane_kind::h2g ? _gpus[lp->space_index].device
                                                   : static_cast<int>(lp->space_index),
                        i);
          std::ignore = exec::thread_util::set_thread_name(lp->workers.back(), name);
          pin(lp->workers.back());
        }
      }
    } catch (...) {
      join_threads();
      _state = run_state::stopped;
      throw;
    }
    _state = run_state::running;
    consumption_tracker::instance().enable();
    // Under the lifecycle lock, so a concurrent stop() cannot run between the threads starting
    // and event delivery starting. Events delivered from here on find the planner and the lanes
    // running.
    event::query_event_subscriber::start();
  }
}

void preload_executor::join_threads() noexcept
{
  _planner.request_stop();
  for (auto const& l : _lanes) {
    for (auto& w : l->workers) {
      w.request_stop();
    }
  }
  if (_planner.joinable()) { _planner.join(); }
  for (auto const& l : _lanes) {
    for (auto& w : l->workers) {
      if (w.joinable()) { w.join(); }
    }
  }
}

void preload_executor::stop() noexcept
{
  // Joining the planner and the lanes is fine from anywhere but an event hook, which would
  // otherwise stop its own worker without joining it.
  assert(t_hook_owner != this);
  std::lock_guard const lifecycle{_lifecycle_mu};
  // Hooks never take the lifecycle lock, so stopping delivery under it cannot deadlock.
  event::query_event_subscriber::stop();
  if (_state == run_state::stopped) { return; }
  bool const was_running = _state == run_state::running;
  _state                 = run_state::stopped;
  join_threads();
  if (was_running) { consumption_tracker::instance().disable(); }
  std::lock_guard const lk{_mu};
  clear_ready_lists();
  _store.clear();
  _dirty        = 0;
  _planner_busy = false;
  _quiescent_cv.notify_all();
}

void preload_executor::log_start() const noexcept
{
  log::log_noexcept([&] {
    std::string line;
    for (auto const& g : _gpus) {
      auto const cap = static_cast<double>(std::max<std::size_t>(g.capacity, 1));
      line += std::format("GPU:{} h2g cap {} ({}) ceilings A {:.2f} B {:.2f} | ",
                          g.device,
                          g.h2g_cap,
                          g.c2c ? "c2c" : "no c2c",
                          static_cast<double>(g.ceilings.committed) / cap,
                          static_cast<double>(g.ceilings.speculative) / cap);
    }
    for (auto const& l : _lanes) {
      if (l->kind != lane_kind::disk) { continue; }
      line += std::format(
        "DISK:{} cap {} ({}) | ", l->space->get_device_id(), l->threads, l->gds ? "GDS" : "no GDS");
    }
    for (auto const& h : _hosts) {
      auto const cap = static_cast<double>(std::max<std::size_t>(h.capacity, 1));
      line +=
        std::format("HOST:{} ceiling {:.2f} | ", h.device, static_cast<double>(h.ceiling) / cap);
    }
    if (line.size() >= 3) { line.resize(line.size() - 3); }
    SIRIUS_LOG_INFO("[preload] device preloading {}; {}",
                    _device_preloading ? "on" : "off (DISK->HOST staging only)",
                    line);

    if (_device_preloading) {
      for (auto const& g : _gpus) {
        if (g.ceilings.committed == 0) {
          SIRIUS_LOG_WARN("[preload] GPU:{} committed ceiling is 0: no GPU promotion there",
                          g.device);
        } else if (g.ceilings.speculative == 0) {
          SIRIUS_LOG_WARN("[preload] GPU:{} speculative ceiling is 0: class B disabled there",
                          g.device);
        }
      }
    }
    for (auto const& h : _hosts) {
      if (_cfg.stage_to_host && h.ceiling == 0) {
        SIRIUS_LOG_WARN("[preload] HOST:{} staging ceiling is 0: no DISK->HOST staging there",
                        h.device);
      }
    }
    if (_topology && _hosts.size() > 1) {
      for (auto const& g : _gpus) {
        if (g.numa < 0) {
          SIRIUS_LOG_WARN(
            "[preload] GPU:{} has no known NUMA node: staging for it uses the batch's origin, or "
            "the first HOST space when that is unknown too",
            g.device);
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------------------------

void preload_executor::prepare_for_query(query_id_t q,
                                         planner::query const& query,
                                         std::vector<int> active_gpus)
{
  std::vector<repo_info> repos;
  try {
    repos = collect_repositories(query.get_pipelines());
  } catch (std::exception const& e) {
    // Advisory: the query still runs, preload only knows fewer of its repositories.
    SIRIUS_LOG_WARN("[preload] query {}: repositories not collected: {}", q, e.what());
    repos.clear();
  }
  prepare_for_query(q, std::move(repos), std::move(active_gpus));
}

void preload_executor::prepare_for_query(query_id_t q,
                                         std::vector<repo_info> repos,
                                         std::vector<int> active_gpus)
{
  std::lock_guard const lk{_mu};
  _store.add_query(q, std::move(active_gpus), std::move(repos));
}

void preload_executor::retire_query(query_id_t q) noexcept
{
  std::lock_guard const lk{_mu};
  purge_query_items(q);
  _store.retire_query(q);
  _quiescent_cv.notify_all();
}

void preload_executor::forget_query(query_id_t q) noexcept
{
  std::optional<per_query_stats> stats;
  {
    std::lock_guard const lk{_mu};
    purge_query_items(q);
    stats = _store.forget_query(q);
    // Its repositories' memory is free: anything refused for lack of room may fit now.
    if (stats) { mark_dirty(dirty::memory); }
    _quiescent_cv.notify_all();
  }
  auto const inputs = consumption_tracker::instance().take_query(q);
  if (!stats) { return; }
  add_input_readiness(*stats, inputs);
  auto const& s = *stats;
  log::log_noexcept([&] {
    SIRIUS_LOG_INFO(
      "[preload] query {}: promoted h2g={} d2g={} d2h={} relocate_d2h={} bytes={} hit={} "
      "bounce={} mispredicted={} wasted={} no_memory={} gated={} held_off={} contended={} "
      "stale={} oom={} gpu_banned={} failed={} dropped(lock={} cap={} overflow={}) "
      "inputs(ready={} waited={}/{}us upgraded={}/{}us)",
      q,
      s[counter::promoted_h2g],
      s[counter::promoted_d2g],
      s[counter::promoted_d2h],
      s[counter::promoted_relocate_d2h],
      s[counter::bytes_h2g] + s[counter::bytes_d2g] + s[counter::bytes_d2h] +
        s[counter::bytes_relocate_d2h],
      s[counter::hit],
      s[counter::bounce],
      s[counter::mispredicted_device],
      s[counter::wasted_unconsumed],
      s[counter::no_memory],
      s[counter::gated],
      s[counter::held_off],
      s[counter::contended],
      s[counter::stale],
      s[counter::oom],
      s[counter::gpu_banned],
      s[counter::failed],
      s[counter::dropped_lock_attempts],
      s[counter::dropped_promotion_cap],
      s[counter::dropped_overflow],
      s[counter::input_ready],
      s[counter::input_waited],
      s[counter::input_waited_us],
      s[counter::input_upgraded],
      s[counter::input_upgraded_us]);
  });
}

void preload_executor::add_input_readiness(per_query_stats& stats,
                                           input_readiness_counts const& inputs) noexcept
{
  auto const add = [&](counter c, std::uint64_t n) {
    stats.add(c, n);
    _counters.add(c, n);
  };
  add(counter::input_ready, inputs.ready);
  add(counter::input_waited, inputs.waited);
  add(counter::input_waited_us, inputs.waited_ns / 1000);
  add(counter::input_upgraded, inputs.upgraded);
  add(counter::input_upgraded_us, inputs.upgraded_ns / 1000);
}

preload_stats preload_executor::stats() const { return _counters.snapshot(); }

// ---------------------------------------------------------------------------------------------
// Event hooks: O(1) under the leaf lock; never call into the queue, batches or memory spaces.
// ---------------------------------------------------------------------------------------------

void preload_executor::on_batch_downgraded(event::event_id_t,
                                           event::timestamp_t,
                                           query_id_t query_id,
                                           event::downgraded_batch const& batch) noexcept
{
  hook_scope const in_hook{this};
  consumption_tracker::instance().note_downgraded(query_id, batch.batch_id);
  std::lock_guard const lk{_mu};
  try {
    auto const status = _store.intake(query_id, batch);
    if (status == intake_status::inserted || status == intake_status::bounced ||
        status == intake_status::merged) {
      mark_dirty(dirty::candidates);
    } else if (batch.from.tier == Tier::GPU && _store.has_candidates()) {
      // A spill freed GPU memory: staging held off by that GPU's pressure may proceed. It opens
      // no memory gate, since a downgrade frees only down to its stop threshold.
      mark_dirty(dirty::memory);
    }
  } catch (...) {
    log_unexpected("intake");
  }
  ++_events_handled;
  _quiescent_cv.notify_all();
}

void preload_executor::on_task_completed(event::event_id_t,
                                         event::timestamp_t,
                                         query_id_t,
                                         std::size_t,
                                         int gpu_id,
                                         event::task_outcome outcome) noexcept
{
  hook_scope const in_hook{this};
  std::lock_guard const lk{_mu};
  _store.advance_queue_epoch();
  if (outcome == event::task_outcome::succeeded) {
    // The task's reservation, inputs and outputs on its GPU were released, and its uploads
    // freed host memory: reopen those spaces.
    if (auto const g = _store.gpu_index_of(gpu_id)) { _store.advance_mem_epoch({Tier::GPU, *g}); }
    _store.advance_host_mem_epochs();
    if (_store.has_candidates() || _store.ledger_size() > 0) {
      mark_dirty(dirty::memory | dirty::queue | dirty::ledger);
    }
  } else if (_store.has_candidates()) {
    // A rescheduled task still holds its reservation: the queue moved, memory did not free.
    mark_dirty(dirty::queue);
  }
  ++_events_handled;
  _quiescent_cv.notify_all();
}

void preload_executor::on_memory_wait_ended_for_task(
  event::event_id_t, event::timestamp_t, query_id_t, std::size_t, int gpu_id, std::size_t) noexcept
{
  hook_scope const in_hook{this};
  std::lock_guard const lk{_mu};
  if (auto const g = _store.gpu_index_of(gpu_id)) { _store.advance_mem_epoch({Tier::GPU, *g}); }
  if (_store.has_candidates()) { mark_dirty(dirty::memory); }
  ++_events_handled;
  _quiescent_cv.notify_all();
}

void preload_executor::on_task_created(event::event_id_t,
                                       event::timestamp_t,
                                       query_id_t,
                                       std::size_t,
                                       op::SiriusPhysicalOperatorType,
                                       exec::queue_priority) noexcept
{
  hook_scope const in_hook{this};
  std::lock_guard const lk{_mu};
  _store.advance_queue_epoch();
  if (_store.has_candidates()) { mark_dirty(dirty::queue); }
  ++_events_handled;
  _quiescent_cv.notify_all();
}

void preload_executor::on_task_deployed(event::event_id_t,
                                        event::timestamp_t,
                                        query_id_t,
                                        std::size_t,
                                        op::SiriusPhysicalOperatorType,
                                        int) noexcept
{
  hook_scope const in_hook{this};
  std::lock_guard const lk{_mu};
  _store.advance_queue_epoch();
  if (_store.has_candidates()) { mark_dirty(dirty::queue); }
  ++_events_handled;
  _quiescent_cv.notify_all();
}

void preload_executor::on_pipeline_closed(
  event::event_id_t, event::timestamp_t, query_id_t query_id, std::size_t, std::size_t) noexcept
{
  hook_scope const in_hook{this};
  std::lock_guard const lk{_mu};
  // A FULL barrier may have cleared, and a finalized operator freed its memory.
  _store.mark_readiness_dirty(query_id);
  _store.advance_all_mem_epochs();
  if (_store.has_candidates()) { mark_dirty(dirty::readiness | dirty::memory); }
  ++_events_handled;
  _quiescent_cv.notify_all();
}

void preload_executor::on_task_queue_empty(event::event_id_t, event::timestamp_t) noexcept
{
  hook_scope const in_hook{this};
  std::lock_guard const lk{_mu};
  mark_idle();
  ++_events_handled;
  _quiescent_cv.notify_all();
}

void preload_executor::on_executor_awaiting_task(event::event_id_t,
                                                 event::timestamp_t,
                                                 int) noexcept
{
  hook_scope const in_hook{this};
  std::lock_guard const lk{_mu};
  mark_idle();
  ++_events_handled;
  _quiescent_cv.notify_all();
}

void preload_executor::mark_idle() noexcept
{
  // While a device waits for a task it cannot take, the scheduler publishes these on every turn
  // of its loop: only the first one after another event can change a plan.
  if (_idle_wake_pending || !_store.has_candidates()) { return; }
  mark_dirty(dirty::idle);
  _idle_wake_pending = true;
}

void preload_executor::mark_dirty(std::uint32_t reasons) noexcept
{
  if (_dirty == 0) { _dirty_since = std::chrono::steady_clock::now(); }
  _dirty |= reasons;
  _idle_wake_pending = false;
  _planner_cv.notify_one();
}

// ---------------------------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------------------------

void preload_executor::planner_loop(std::stop_token const& st) noexcept
{
  std::unique_lock lk{_mu};
  for (;;) {
    _planner_cv.wait(lk, st, [&] { return _dirty != 0; });
    if (st.stop_requested()) { return; }
    auto const woke = _dirty_since;
    _dirty          = 0;

    // Phase 0, under the lock: what the scans need.
    std::optional<pass_snapshot> snap;
    try {
      snap.emplace(_store.snapshot_phase0());
    } catch (...) {
      log_unexpected("snapshot");
      _quiescent_cv.notify_all();
      continue;  // the next event retries
    }
    _planner_busy = true;
    lk.unlock();

    // Phases 1 and 2, without the lock: the queue (under its own lock only) and the
    // repositories (under each query's lease).
    plan_input in;
    try {
      sirius::nvtx_scoped_range const range{"sirius::preload::plan"};
      // While preloaded batches wait for their consumer, the visit also counts misplaced ones.
      if (snap->has_candidates || !snap->ledger.empty()) {
        sirius::nvtx_scoped_range const visit{"sirius::preload::visit_queue"};
        in.class_a = visit_queues();
      }
      scan_repositories(*snap, in);
      in.log_budgets = log::get_sink()->should_log(log::level::debug);
      in.scanned     = true;
    } catch (...) {
      in = plan_input{};
      log_unexpected("scan");
    }
    snap.reset();

    // Phase 3, under the lock: install the plan.
    lk.lock();
    _planner_busy = false;
    std::optional<pass_summary> summary;
    try {
      if (in.scanned) { read_spaces(in); }
      summary = install_plan(in);
      bool const dispatched =
        std::any_of(_lanes.begin(), _lanes.end(), [](auto const& l) { return !l->ready.empty(); });
      if (dispatched) {
        auto const us = std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now() - woke)
                          .count();
        _counters.raise(counter::wake_to_dispatch_us_max, static_cast<std::uint64_t>(us));
      }
    } catch (...) {
      clear_ready_lists();
      _room_refused = true;  // unknown: let the next finished job re-plan
      log_unexpected("install");
    }
    _quiescent_cv.notify_all();
    if (summary) {
      // Formatted outside the lock: event hooks wait on it.
      lk.unlock();
      log::log_noexcept([&] {
        std::string budgets;
        for (std::size_t i = 0; i < summary->gpus.size(); ++i) {
          auto const& [reading, b] = summary->gpus[i];
          auto const a             = b.available(klass::A);
          auto const s             = b.available(klass::B);
          budgets += std::format(" | GPU:{} consumed {}MiB unreserved {}MiB A {} B {}",
                                 _gpus[i].device,
                                 reading.consumed >> 20,
                                 reading.unreserved >> 20,
                                 budget_text(a.bytes, a.bytes == 0 ? a.reason : refusal::none),
                                 budget_text(s.bytes, s.bytes == 0 ? s.reason : refusal::none));
        }
        for (std::size_t j = 0; j < summary->hosts.size(); ++j) {
          auto const& [reading, b] = summary->hosts[j];
          budgets += std::format(" | HOST:{} consumed {}MiB unreserved {}MiB staging {}",
                                 _hosts[j].device,
                                 reading.consumed >> 20,
                                 reading.unreserved >> 20,
                                 budget_text(b.room, b.room == 0 ? b.closed : refusal::none));
        }
        SIRIUS_LOG_DEBUG(
          "[preload] pass: {} class A task(s), {} class B item(s), {} planned, {} assigned{}",
          summary->class_a_tasks,
          summary->class_b_items,
          summary->planned,
          summary->assigned,
          budgets);
      });
      lk.lock();
    }
  }
}

std::vector<preload_executor::visited_task> preload_executor::visit_queues() const
{
  std::vector<visited_task> out;
  if (_queue == nullptr || _gpus.empty()) { return out; }
  auto const lookahead = static_cast<std::size_t>(std::max(_cfg.lookahead_tasks, 0));
  auto const n_gpus    = _gpus.size();

  // Only ids are copied under the queue lock: no batch lock, no executor lock.
  auto const record = [&](parallel::itask const& task,
                          std::size_t position,
                          std::optional<std::size_t> gpu,
                          bool in_window) {
    auto const* gpt = dynamic_cast<pipeline::gpu_pipeline_task const*>(&task);
    if (gpt == nullptr) { return; }
    auto const* data = sirius::convertible_gpu_pipeline_task::get_pipelineable_data(task);
    if (data == nullptr) { return; }
    visited_task v;
    v.priority = gpt->get_priority();
    v.position = static_cast<std::uint32_t>(
      std::min<std::size_t>(position, std::numeric_limits<std::uint32_t>::max()));
    v.gpu       = gpu;
    v.in_window = in_window;
    for (auto const& batch : data->get_data_batches()) {
      if (batch) { v.inputs.push_back(batch->get_batch_id()); }
    }
    if (!v.inputs.empty()) { out.push_back(std::move(v)); }
  };

  auto const exact                     = exact_bucket_window(_head_margin, lookahead);
  std::size_t exact_size_of_single_gpu = 0;
  for (std::size_t i = 0; i < n_gpus; ++i) {
    std::size_t position = 0;
    _queue->visit_in_pop_order(exec::gpu_index{_gpus[i].device}, [&](parallel::itask const& task) {
      auto const p = position++;
      // Head tasks are recorded too, so a misplaced preload is
      // counted before the task runs; they are never planned.
      record(task, p, i, exact.contains(p));
      return position < exact.end;
    });
    if (n_gpus == 1) { exact_size_of_single_gpu = position; }
  }

  auto const shared = shared_bucket_window(_head_margin, lookahead, n_gpus);
  // With one GPU, the shared bucket's positions continue after its own bucket; when that bucket
  // alone fills the window, nothing in the shared one is inside it.
  if (n_gpus == 1 && exact_size_of_single_gpu >= shared.end) { return out; }
  std::size_t index = 0;
  _queue->visit_in_pop_order(
    exec::gpu_index{exec::no_preferred_device}, [&](parallel::itask const& task) {
      auto const p = shared_bucket_position(index++, exact_size_of_single_gpu, n_gpus);
      if (shared.contains(p)) { record(task, p, std::nullopt, true); }
      return p + 1 < shared.end;
    });
  return out;
}

void preload_executor::scan_repositories(pass_snapshot const& snap, plan_input& in)
{
  // Live ledger entries: lock() creates a strong reference, so each query is probed under its
  // own lease.
  {
    std::map<query_id_t, std::vector<ledger_probe const*>> probes;
    for (auto const& probe : snap.ledger) {
      probes[probe.query].push_back(&probe);
    }
    for (auto const& [q, list] : probes) {
      auto const lease = _lifecycle.try_acquire_work(q);
      if (!lease) { continue; }
      for (auto const* probe : list) {
        auto const batch = probe->batch.lock();
        if (!batch) { continue; }  // swept without a lease in phase 3
        auto const subscribers = batch->get_subscriber_count();
        if (probe->k == klass::A && subscribers == 0) {
          // Its task ran and unsubscribed.
          in.verdicts.emplace_back(probe->batch_id, ledger_verdict::hit);
        } else if (probe->k == klass::B && subscribers > 0) {
          in.verdicts.emplace_back(probe->batch_id, ledger_verdict::reclassify_a);
        }
      }
    }
  }

  auto const window    = static_cast<std::size_t>(std::max(_cfg.repository_window, 0));
  auto const lookahead = static_cast<std::size_t>(std::max(_cfg.repository_partition_lookahead, 0));
  if (!snap.has_candidates || window == 0) { return; }

  // Repositories with candidates, oldest query first. The scan budget bounds a pass, and a pass
  // resumes after the repository the previous one scanned last, so none is starved.
  std::vector<std::pair<query_view const*, repo_view const*>> order;
  for (auto const& qv : snap.queries) {
    for (auto const& rv : qv.repos) {
      if (rv.info.repo != nullptr && !rv.candidate_ids.empty()) { order.emplace_back(&qv, &rv); }
    }
  }
  if (order.empty()) { return; }
  std::size_t first = 0;
  if (_scan_cursor) {
    auto const last = std::find_if(order.begin(), order.end(), [&](auto const& e) {
      return std::pair{e.first->query, e.second->info.token} == *_scan_cursor;
    });
    if (last != order.end()) {
      first = static_cast<std::size_t>(std::distance(order.begin(), last)) + 1;
    } else {
      // That repository has no candidates left: resume at its query, or the next younger one.
      auto const next = std::find_if(order.begin(), order.end(), [&](auto const& e) {
        return e.first->query >= _scan_cursor->first;
      });
      first           = static_cast<std::size_t>(std::distance(order.begin(), next));
    }
    first %= order.size();
  }

  std::size_t scans = 0;
  std::optional<query_id_t> leased;
  exec::query_lifecycle_registry::work_lease lease;
  for (std::size_t n = 0; n < order.size() && scans < max_repository_scans; ++n) {
    auto const& [qv, rv] = order[(first + n) % order.size()];
    if (leased != qv->query) {
      // Strong references and repository pointers are used only while this lease is held.
      leased = qv->query;
      lease  = _lifecycle.try_acquire_work(qv->query);
    }
    if (!lease) { continue; }
    auto const& info = rv->info;
    bool const ready =
      std::all_of(info.full_sources.begin(), info.full_sources.end(), [](auto const& w) {
        auto const source = w.lock();
        return source && source->is_pipeline_finished();
      });
    bool const partitioned = info.partition_consumer != nullptr;
    // A ready consumer that is not partition-draining is the creator's next pop: left alone.
    if (ready && !partitioned) { continue; }
    ++scans;
    _scan_cursor = std::pair{qv->query, info.token};
    try {
      auto const parts  = info.repo->num_partitions();
      std::size_t rank  = 0;
      std::size_t found = 0;
      for (std::size_t p = 0; p < parts && found < rv->candidate_ids.size(); ++p) {
        if (ready && rank > lookahead) { break; }
        auto const ids = info.repo->get_batch_ids(p);
        if (ids.empty()) { continue; }
        std::optional<int> device;
        if (ready && rank > 0) { device = placement_device(*info.partition_consumer, p); }
        for (std::size_t i = 0; i < std::min(window, ids.size()); ++i) {
          if (!std::binary_search(rv->candidate_ids.begin(), rv->candidate_ids.end(), ids[i])) {
            continue;
          }
          ++found;
          repo_item item;
          item.query              = qv->query;
          item.batch_id           = ids[i];
          item.repo               = info.token;
          item.priority           = info.dest_priority;
          item.partition          = static_cast<std::uint32_t>(p);
          item.position           = static_cast<std::uint32_t>(i);
          item.ready              = ready;
          item.partition_consumer = partitioned;
          item.rank               = rank;
          item.placement_device   = device;
          in.class_b.push_back(item);
        }
        ++rank;
      }
    } catch (std::exception const&) {
      // The repository changed shape under the scan; the next pass sees it settled.
    }
  }
}

void preload_executor::read_spaces(plan_input& in) const
{
  auto const read = [](cucascade::memory::memory_space& space,
                       std::atomic<std::size_t> const* unreserved) {
    // Unreserved bytes first: a job no longer in them has made its reservation, which the
    // consumed bytes read next then count. A job taken later would need `_mu`.
    space_reading r;
    r.unreserved    = unreserved ? unreserved->load(std::memory_order_acquire) : 0;
    r.consumed      = consumed_bytes(space.get_max_memory(), space.get_available_memory());
    r.above_trigger = space.should_downgrade_memory();
    return r;
  };
  in.gpus.reserve(_gpus.size());
  for (std::size_t i = 0; i < _gpus.size(); ++i) {
    in.gpus.push_back(read(*_gpus[i].space, unreserved_of({Tier::GPU, i})));
  }
  in.hosts.reserve(_hosts.size());
  for (std::size_t j = 0; j < _hosts.size(); ++j) {
    in.hosts.push_back(read(*_hosts[j].space, unreserved_of({Tier::HOST, j})));
  }
}

preload_executor::pass_summary preload_executor::install_plan(plan_input const& in)
{
  _counters.add(counter::passes);
  _store.sweep_expired_ledger();
  _store.apply_ledger_verdicts(in.verdicts);
  _store.prune_expired_candidates(prune_budget);
  if (!in.scanned || in.gpus.size() != _gpus.size() || in.hosts.size() != _hosts.size()) {
    clear_ready_lists();
    _room_refused = true;  // nothing was judged: let the next finished job re-plan
    _counters.add(counter::passes_noop);
    return {};
  }

  std::vector<plan_entry> entries;

  // Class A: inputs of queued tasks past the head margin.
  for (auto const& task : in.class_a) {
    for (auto const id : task.inputs) {
      if (task.gpu) { _store.note_consumer_device(id, *task.gpu); }
      if (!task.in_window) { continue; }
      auto const* c = _store.find(id);
      if (c == nullptr || !_store.plannable(*c)) { continue; }
      target_request r;
      r.k        = klass::A;
      r.location = c->location.tier;
      // Without device preloading, a GPU is never a target: a DISK candidate stages on HOST and a
      // HOST candidate is left alone, exactly as if it were banned from GPUs.
      r.gpu_banned         = c->gpu_banned || !_device_preloading;
      r.speculative_banned = c->speculative_banned;
      std::array<int, 1> single{};
      if (task.gpu) {
        r.bucket        = bucket_kind::exact;
        r.bucket_device = _gpus[*task.gpu].device;
      } else {
        // The no-preference bucket is popped by every GPU: its consumer's device is known only
        // when there is a single GPU.
        r.bucket = bucket_kind::shared;
        if (_gpus.size() == 1) {
          single[0]     = _gpus.front().device;
          r.active_gpus = single;
        }
      }
      add_entry(entries, *c, r, plan_key{band_gpu, task.priority, klass::A, 0, task.position, id});
    }
  }

  // Class B: repository batches of consumers behind a FULL barrier, or of partition-draining
  // consumers past the partition the creator drains next.
  auto const window = std::max(_cfg.repository_window, 0);
  std::vector<int> all_gpus;
  for (auto const& g : _gpus) {
    all_gpus.push_back(g.device);
  }
  // Behind a barrier only DISK batches move (to HOST), and a repository holds at most a window of
  // staged batches its consumer has not taken yet: those staged or being staged count first.
  std::map<std::pair<query_id_t, std::uintptr_t>, int> staged_per_repo;
  for (auto const& b : in.class_b) {
    if (b.ready) { continue; }
    auto const* c = _store.find(b.batch_id);
    if (c == nullptr || c->query != b.query) { continue; }
    auto const* e      = _store.find_ledger(b.batch_id);
    bool const staged  = c->location.tier == Tier::HOST && e != nullptr && e->gpu < 0;
    bool const staging = c->in_flight && c->location.tier == Tier::DISK;
    if (staged || staging) { ++staged_per_repo[{b.query, b.repo}]; }
  }
  for (auto const& b : in.class_b) {
    auto const* c = _store.find(b.batch_id);
    if (c == nullptr || !_store.plannable(*c) || c->query != b.query) { continue; }
    auto const* qs = _store.find_query(b.query);
    if (qs == nullptr) { continue; }
    int* staged = nullptr;
    if (!b.ready) {
      if (c->location.tier != Tier::DISK) { continue; }
      staged = &staged_per_repo[{b.query, b.repo}];
      if (*staged >= window) { continue; }
    }
    target_request r;
    r.k                  = klass::B;
    r.location           = c->location.tier;
    r.gpu_banned         = c->gpu_banned || !_device_preloading;
    r.speculative_banned = c->speculative_banned;
    r.consumer_ready     = b.ready;
    r.partition_consumer = b.partition_consumer;
    r.partition_rank     = b.rank;
    r.placement_device   = b.placement_device;
    r.active_gpus        = qs->active_gpus.empty() ? std::span<int const>{all_gpus}
                                                   : std::span<int const>{qs->active_gpus};
    if (add_entry(entries,
                  *c,
                  r,
                  plan_key{band_gpu, b.priority, klass::B, b.partition, b.position, b.batch_id}) &&
        staged != nullptr) {
      ++*staged;
    }
  }

  std::stable_sort(entries.begin(), entries.end(), [](plan_entry const& a, plan_entry const& b) {
    return a.key < b.key;
  });
  {
    std::unordered_set<std::uint64_t> seen;
    std::erase_if(entries, [&](plan_entry const& e) { return !seen.insert(e.batch_id).second; });
  }

  std::vector<assign_item> items;
  items.reserve(entries.size());
  for (auto const& e : entries) {
    items.push_back(e.a);
  }

  std::vector<gpu_budget_state> gpu_budgets;
  std::vector<gpu_pressure> pressures;
  gpu_budgets.reserve(_gpus.size());
  pressures.reserve(_gpus.size());
  for (std::size_t i = 0; i < _gpus.size(); ++i) {
    auto const& g = _gpus[i];
    target const t{Tier::GPU, i};
    gpu_budget_inputs gin;
    gin.device           = g.device;
    gin.consumed         = in.gpus[i].consumed;
    gin.above_trigger    = in.gpus[i].above_trigger;
    gin.mem_epoch        = _store.mem_epoch(t);
    gin.gate_epoch       = _store.gate_epoch(t);
    gin.unreserved_bytes = in.gpus[i].unreserved;
    gin.ceilings         = g.ceilings;
    gin.lookahead_bytes  = g.lookahead_bytes;
    // In-flight copies are charged against the budget board when their job is taken, so
    // outstanding already includes them: the planner window is budget - outstanding, with no
    // separate in-flight term.
    gin.outstanding       = _store.outstanding_bytes(i);
    gin.speculative_bytes = g.speculative_bytes;
    gin.outstanding_b     = _store.outstanding_b_bytes(i);
    gin.max_batch_bytes   = g.max_batch_bytes;
    gpu_budgets.push_back(make_gpu_budget_state(gin, *_board));
    pressures.push_back({g.device, in.gpus[i].above_trigger});
  }
  std::vector<host_budget_state> host_budgets;
  host_budgets.reserve(_hosts.size());
  for (std::size_t j = 0; j < _hosts.size(); ++j) {
    auto const& h = _hosts[j];
    target const t{Tier::HOST, j};
    std::vector<gpu_pressure> local;
    for (auto const gi : h.local_gpus) {
      local.push_back(pressures[gi]);
    }
    host_budget_inputs hin;
    hin.consumed         = in.hosts[j].consumed;
    hin.above_trigger    = in.hosts[j].above_trigger;
    hin.mem_epoch        = _store.mem_epoch(t);
    hin.gate_epoch       = _store.gate_epoch(t);
    hin.unreserved_bytes = in.hosts[j].unreserved;
    hin.ceiling          = h.ceiling;
    hin.local_gpus       = local;
    host_budgets.push_back(make_host_budget_state(hin, *_board));
  }

  pass_summary summary{in.class_a.size(), in.class_b.size(), entries.size(), 0, {}, {}};
  if (in.log_budgets) {
    summary.gpus.reserve(gpu_budgets.size());
    for (std::size_t i = 0; i < gpu_budgets.size(); ++i) {
      summary.gpus.emplace_back(in.gpus[i], gpu_budgets[i]);
    }
    summary.hosts.reserve(host_budgets.size());
    for (std::size_t j = 0; j < host_budgets.size(); ++j) {
      summary.hosts.emplace_back(in.hosts[j], host_budgets[j]);
    }
  }

  auto const result = assign_head_of_line(items, gpu_budgets, host_budgets);
  for (std::size_t r = 0; r < result.refusals.size(); ++r) {
    auto const reason = static_cast<refusal>(r);
    if (reason == refusal::none || result.refusals[r] == 0) { continue; }
    _counters.add(refusal_counter(reason), result.refusals[r]);
  }
  // Room counts running jobs' unreserved bytes and the windows their in-flight bytes: a job
  // finishing may let a refused item fit.
  auto const refused = [&](refusal r) { return result.refusals[static_cast<std::size_t>(r)] > 0; };
  _room_refused =
    refused(refusal::ceiling) || refused(refusal::window) || refused(refusal::speculative_window);
  rebuild_ready_lists(entries, result);
  summary.assigned = result.assigned.size();
  return summary;
}

bool preload_executor::add_entry(std::vector<plan_entry>& entries,
                                 candidate const& c,
                                 target_request const& request,
                                 plan_key key)
{
  auto const choice = choose_target(request, _cfg);
  if (choice.decision == target_decision::not_planned) {
    if (choice.reason == skip_reason::creator_next) {
      _counters.add(counter::skipped_creator_next);
    }
    return false;
  }
  plan_entry e;
  e.batch_id     = c.batch_id;
  e.query        = c.query;
  e.k            = request.k;
  e.source       = c.location;
  e.a.k          = request.k;
  e.a.gpu_bytes  = reserve_estimate(c.est_bytes);
  e.a.host_bytes = reserve_estimate(c.est_bytes);
  if (choice.decision == target_decision::gpu) { e.a.gpu = _store.gpu_index_of(choice.device); }
  // HOST is the target of a staging item and the fallback of a DISK-resident GPU item, and never
  // a target at all when staging is off.
  if (c.location.tier == Tier::DISK && _cfg.stage_to_host) {
    e.a.host = staging_host(choice.device, c);
  }
  if (!e.a.gpu && !e.a.host) { return false; }
  key.band = e.a.gpu ? band_gpu : band_host_staging;
  e.key    = key;
  entries.push_back(e);
  return true;
}

void preload_executor::rebuild_ready_lists(std::vector<plan_entry> const& entries,
                                           assignment_result const& result)
{
  // An item stays counted as refused by the in-flight cap across plans, so each counts once.
  std::unordered_set<std::uint64_t> counted;
  for (auto const& l : _lanes) {
    for (auto const& item : l->ready) {
      if (item.counted_inflight) { counted.insert(item.r.batch_id); }
    }
  }
  clear_ready_lists();
  for (auto const& a : result.assigned) {
    auto const& e    = entries[a.item];
    auto const index = lane_of(e.source, a.t);
    if (!index) { continue; }
    lane_item item;
    item.query            = e.query;
    item.source           = e.source.tier;
    item.r                = {e.batch_id, a.t, e.k, 0};
    item.counted_inflight = counted.contains(e.batch_id);
    if (a.t.tier == Tier::GPU) {
      auto const& g  = _gpus[a.t.index];
      item.r.ceiling = e.k == klass::A ? g.ceilings.committed : g.ceilings.speculative;
      item.est       = e.a.gpu_bytes;
    } else {
      item.r.ceiling = _hosts[a.t.index].ceiling;
      item.est       = e.a.host_bytes;
    }
    _lanes[*index]->ready.push_back(item);
  }
  if (result.assigned.empty()) { _counters.add(counter::passes_noop); }
  notify_lanes_with_work();
}

std::optional<std::size_t> preload_executor::lane_of(cucascade::memory::memory_space_id source,
                                                     target t) const noexcept
{
  // A HOST->GPU item uses its GPU's lane, which exists only with device preloading.
  if (source.tier == Tier::HOST && t.tier == Tier::GPU && t.index < _gpu_lane.size()) {
    return _gpu_lane[t.index];
  }
  if (source.tier == Tier::DISK) {
    for (std::size_t k = 0; k < _disks.size(); ++k) {
      if (_disks[k]->get_device_id() == source.device_id) { return _disk_lane[k]; }
    }
  }
  return std::nullopt;
}

std::optional<std::size_t> preload_executor::staging_host(int gpu_device,
                                                          candidate const& c) const noexcept
{
  if (gpu_device >= 0) {
    if (auto const g = _store.gpu_index_of(gpu_device); g && _gpus[*g].local_host) {
      return _gpus[*g].local_host;
    }
  }
  auto const numa_target = gpu_device >= 0 ? numa_of(gpu_device) : -1;
  auto const numa_origin = c.origin_gpu ? numa_of(*c.origin_gpu) : -1;
  auto const device      = choose_staging_host(numa_target, c.origin_host, numa_origin, [&](int d) {
    return _store.host_index_of(d).has_value();
  });
  if (device >= 0) { return _store.host_index_of(device); }
  bool const numa_known = numa_target >= 0 || numa_origin >= 0 || c.origin_host.value_or(-1) >= 0;
  // Nothing known at all: any HOST beats leaving the batch on DISK.
  if (!numa_known && !_hosts.empty()) { return 0; }
  return std::nullopt;
}

int preload_executor::numa_of(int gpu_device) const noexcept
{
  try {
    return _topology ? _topology->numa_node_of(gpu_device) : -1;
  } catch (...) {
    return -1;
  }
}

std::atomic<std::size_t>* preload_executor::unreserved_of(target t) const noexcept
{
  if (t.tier == Tier::GPU && t.index < _gpus.size()) { return &_unreserved[t.index]; }
  if (t.tier == Tier::HOST && t.index < _hosts.size()) {
    return &_unreserved[_gpus.size() + t.index];
  }
  return nullptr;
}

// ---------------------------------------------------------------------------------------------
// Lanes
// ---------------------------------------------------------------------------------------------

bool preload_executor::needs_token(lane const& l, lane_item const& item) const noexcept
{
  return takes_h2g_token(item.source, item.r.t.tier, l.kind == lane_kind::disk && l.gds);
}

preload_executor::take_check preload_executor::check_take(lane const& l,
                                                          lane_item const& item) const noexcept
{
  auto const* c = _store.find(item.r.batch_id);
  if (c == nullptr || !_store.plannable(*c)) { return take_check::blocked; }
  if (_store.gated(item.r.t)) { return take_check::blocked; }
  if (item.r.t.tier == Tier::GPU) {
    if (item.r.t.index >= _gpus.size()) { return take_check::blocked; }
    auto const& g = _gpus[item.r.t.index];
    if (held_off(*_board, g.device)) { return take_check::blocked; }
    if (sat_add(_store.inflight_bytes(item.r.t), item.est) > g.max_inflight_bytes) {
      return take_check::inflight;
    }
    if (needs_token(l, item) &&
        _store.h2g_tokens(item.r.t.index) >= static_cast<std::uint32_t>(g.h2g_cap)) {
      return take_check::blocked;
    }
    return take_check::ok;
  }
  if (item.r.t.index >= _hosts.size()) { return take_check::blocked; }
  auto const& local  = _hosts[item.r.t.index].local_gpus;
  bool const pressed = std::any_of(local.begin(), local.end(), [&](std::size_t gi) {
    return held_off(*_board, _gpus[gi].device);
  });
  return pressed ? take_check::blocked : take_check::ok;
}

bool preload_executor::has_takeable(lane const& l) const noexcept
{
  return std::any_of(l.ready.begin(), l.ready.end(), [&](lane_item const& item) {
    return check_take(l, item) == take_check::ok;
  });
}

std::optional<preload_executor::taken_job> preload_executor::take_first_takeable(lane& l)
{
  for (auto it = l.ready.begin(); it != l.ready.end();) {
    if (_store.find(it->r.batch_id) == nullptr) {
      it = l.ready.erase(it);
      continue;
    }
    auto const check = check_take(l, *it);
    if (check != take_check::ok) {
      if (check == take_check::inflight && !it->counted_inflight) {
        it->counted_inflight = true;
        _counters.add(counter::refused_inflight);
      }
      ++it;
      continue;
    }
    auto job        = _store.take(it->r);
    auto const item = *it;
    it              = l.ready.erase(it);
    if (!job) { continue; }
    // Counted per job handed to a worker, not per plan: passes re-plan the same items.
    _counters.add(counter::items_planned);
    _counters.add(job->k == klass::A ? counter::class_a_items : counter::class_b_items);
    if (job->t.tier == Tier::HOST) { _counters.add(counter::staging_items); }
    return taken_job{std::move(*job), item, needs_token(l, item)};
  }
  return std::nullopt;
}

void preload_executor::lane_loop(lane& l, std::stop_token const& st) noexcept
{
  if (l.kind == lane_kind::h2g) {
    // Every job of this lane targets one GPU.
    if (auto const err = cudaSetDevice(_gpus[l.space_index].device); err != cudaSuccess) {
      log::log_noexcept([&] {
        SIRIUS_LOG_ERROR("[preload] cudaSetDevice({}) failed: {}",
                         _gpus[l.space_index].device,
                         cudaGetErrorString(err));
      });
    }
  }
  std::unique_lock lk{_mu};
  while (!st.stop_requested()) {
    // Looking for work also counts the items it finds over the in-flight cap, once each; the
    // wait predicate itself has no side effects.
    std::optional<taken_job> taken;
    if (l.active < l.effective_cap) {
      try {
        taken = take_first_takeable(l);
      } catch (...) {
        l.ready.clear();
        log_unexpected("take");
        _quiescent_cv.notify_all();
      }
    }
    if (!taken) {
      // Also wake for an item newly over the in-flight cap, so that the next look counts it.
      l.cv.wait(lk, st, [&] {
        return l.active < l.effective_cap &&
               std::any_of(
                 l.ready.begin(),
                 l.ready.end(),
                 [&](lane_item const& item) {
                   auto const check = check_take(l, item);
                   return check == take_check::ok ||
                          (check == take_check::inflight && !item.counted_inflight);
                 });
      });
      continue;
    }

    lane_slot slot{*this, l, *taken};
    lk.unlock();
    bool const to_gpu = taken->job.t.tier == Tier::GPU;
    if (to_gpu) { consumption_tracker::instance().note_promotion_started(taken->job.batch_id); }
    auto const started = std::chrono::steady_clock::now();
    auto const result  = promote_one(taken->job, _env, unreserved_of(taken->job.t));
    if (to_gpu) {
      consumption_tracker::instance().note_promotion_finished(
        taken->job.batch_id, result.outcome == promotion_outcome::promoted);
    }
    auto const us = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - started)
                      .count();
    // Logged before taking the lock again: event hooks wait on it.
    log::log_noexcept([&] {
      SIRIUS_LOG_TRACE("[preload][{}:{}] batch={} q={} klass={} {}->{} bytes={} us={} outcome={}",
                       tier_name(taken->job.t.tier),
                       taken->job.t.index,
                       taken->job.batch_id,
                       taken->job.query,
                       taken->job.k == klass::A ? "A" : "B",
                       tier_name(taken->job.expected_source.tier),
                       tier_name(taken->job.t.tier),
                       result.bytes,
                       us,
                       outcome_name(result.outcome));
    });
    lk.lock();
    slot.release();

    bool const first_failure = apply_result(*taken, result);
    if (result.outcome == promotion_outcome::promoted) {
      auto const [sum, max] = timing_counters(result.kind);
      _counters.add(sum, static_cast<std::uint64_t>(us));
      _counters.raise(max, static_cast<std::uint64_t>(us));
    }
    if (replan_after(taken->job, result.outcome)) { mark_dirty(dirty::lane_done); }
    // A freed slot, token or in-flight budget may let any lane take more.
    notify_lanes_with_work();
    _quiescent_cv.notify_all();
    if (first_failure) {
      lk.unlock();
      log::log_noexcept([&] {
        SIRIUS_LOG_WARN(
          "[preload] query {}: promotion of batch {} failed; further failures of this query are "
          "logged at DEBUG",
          taken->job.query,
          taken->job.batch_id);
      });
      lk.lock();
    }
  }
}

bool preload_executor::replan_after(promotion_job const& job,
                                    promotion_outcome outcome) const noexcept
{
  switch (outcome) {
    // A lock refusal waits for the next queue event, unless one came during the attempt: the
    // pass it ran could not plan an in-flight candidate.
    case promotion_outcome::contended:
      return _room_refused || _store.queue_epoch() != job.queue_epoch_at_take;
    // These only dropped the candidate (or, unsupported, parked it for good). Nothing a pass
    // reads changed but the job's released bytes, which matter only if the last pass refused an
    // item for room.
    case promotion_outcome::unsupported:
    case promotion_outcome::retired:
    case promotion_outcome::expired:
    case promotion_outcome::already_resident:
    case promotion_outcome::failed: return _room_refused;
    // A staged batch can go on to its GPU; a new location or a ban changes the candidate's
    // target; a gate may send DISK items to HOST instead; and the event that ends a hold-off may
    // have come during the attempt, while the candidate could not be planned.
    case promotion_outcome::held_off:
    case promotion_outcome::promoted:
    case promotion_outcome::stale:
    case promotion_outcome::too_big:
    case promotion_outcome::oom:
    case promotion_outcome::no_memory:
    case promotion_outcome::gated: break;
  }
  return true;
}

bool preload_executor::apply_result(taken_job const& taken, promotion_result const& result) noexcept
{
  bool first_failure = false;
  try {
    auto const effect = _store.apply_outcome(taken.job, result);
    purge_ready_items(taken.job.t, effect.purge);
    first_failure = effect.first_failure_in_query;
  } catch (...) {
    _store.drop_candidate_noexcept(taken.job.batch_id);
    log_unexpected("outcome");
  }
  if (_log_promotions) {
    try {
      _promotion_log.push_back({taken.job.batch_id,
                                taken.job.query,
                                taken.job.k,
                                taken.job.t,
                                taken.job.expected_source,
                                result.outcome,
                                result.kind,
                                result.bytes});
    } catch (...) {
      // The test log is best effort.
    }
  }
  return first_failure;
}

void preload_executor::purge_ready_items(target t, purge_scope scope) noexcept
{
  if (scope == purge_scope::none) { return; }
  std::optional<target> also;
  if (scope == purge_scope::held_off && t.tier == Tier::GPU && t.index < _gpus.size() &&
      _gpus[t.index].local_host) {
    // A held-off GPU also closes staging into its NUMA-local host.
    also = target{Tier::HOST, *_gpus[t.index].local_host};
  }
  for (auto const& l : _lanes) {
    std::erase_if(l->ready, [&](lane_item const& item) {
      return item.r.t == t || (also && item.r.t == *also);
    });
  }
}

void preload_executor::purge_query_items(query_id_t q) noexcept
{
  for (auto const& l : _lanes) {
    std::erase_if(l->ready, [&](lane_item const& item) { return item.query == q; });
  }
}

void preload_executor::clear_ready_lists() noexcept
{
  for (auto const& l : _lanes) {
    l->ready.clear();
  }
}

void preload_executor::notify_lanes_with_work() noexcept
{
  for (auto const& l : _lanes) {
    if (!l->ready.empty()) { l->cv.notify_all(); }
  }
}

bool preload_executor::quiescent() const noexcept
{
  if (_dirty != 0 || _planner_busy) { return false; }
  return std::none_of(_lanes.begin(), _lanes.end(), [&](auto const& l) {
    return l->active > 0 || (l->active < l->effective_cap && has_takeable(*l));
  });
}

void preload_executor::log_unexpected(std::string_view where) noexcept
{
  // Warn once per executor: a failure that repeats on every pass must not flood the log.
  if (!_unexpected_logged.exchange(true, std::memory_order_relaxed)) {
    log::log_noexcept(
      [&] { SIRIUS_LOG_WARN("[preload] unexpected error in {}; preload continues", where); });
  } else {
    log::log_noexcept([&] { SIRIUS_LOG_DEBUG("[preload] unexpected error in {}", where); });
  }
}

// ---------------------------------------------------------------------------------------------
// Tests only
// ---------------------------------------------------------------------------------------------

bool preload_executor::wait_until_quiescent_for_testing(std::chrono::milliseconds limit,
                                                        std::uint64_t min_events)
{
  std::unique_lock lk{_mu};
  return _quiescent_cv.wait_for(
    lk, limit, [&] { return _events_handled >= min_events && quiescent(); });
}

std::uint64_t preload_executor::events_handled_for_testing() const
{
  std::lock_guard const lk{_mu};
  return _events_handled;
}

void preload_executor::enable_promotion_log_for_testing()
{
  std::lock_guard const lk{_mu};
  _log_promotions = true;
}

std::vector<promotion_record> preload_executor::promotion_log_for_testing() const
{
  std::lock_guard const lk{_mu};
  return _promotion_log;
}

void preload_executor::set_promotion_hook_for_testing(
  std::function<void(promotion_job const&)> hook)
{
  auto shared = std::make_shared<std::function<void(promotion_job const&)>>(std::move(hook));
  std::lock_guard const g{_hook_mu};
  _hook = std::move(shared);
}

void preload_executor::set_post_convert_hook_for_testing(
  std::function<void(promotion_job const&)> hook)
{
  auto shared = std::make_shared<std::function<void(promotion_job const&)>>(std::move(hook));
  std::lock_guard const g{_hook_mu};
  _post_convert_hook = std::move(shared);
}

std::size_t preload_executor::outstanding_bytes_for_testing(std::size_t gpu) const
{
  std::lock_guard const lk{_mu};
  return _store.outstanding_bytes(gpu);
}

std::size_t preload_executor::outstanding_speculative_bytes_for_testing(std::size_t gpu) const
{
  std::lock_guard const lk{_mu};
  return _store.outstanding_b_bytes(gpu);
}

std::vector<int> preload_executor::lane_threads_for_testing() const
{
  std::lock_guard const lk{_mu};
  std::vector<int> out;
  for (auto const& l : _lanes) {
    out.push_back(l->threads);
  }
  return out;
}

}  // namespace sirius::preload
