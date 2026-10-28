/*
 * Copyright 2025, Sirius Contributors.
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

#include "downgrade/downgrade_executor.hpp"

#include "data/convertible_data.hpp"
#include "data/convertible_data_batch.hpp"
#include "data/convertible_gpu_pipeline_task.hpp"
#include "downgrade/spill_policy.hpp"
#include "log/logging.hpp"

#include <nvtx3/nvtx3.hpp>

#include <absl/cleanup/cleanup.h>
#include <cucascade/io/cache/fs_cache.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <limits>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace sirius {
namespace parallel {

static std::string tier_to_string(cucascade::memory::Tier tier)
{
  switch (tier) {
    case cucascade::memory::Tier::GPU: return "GPU";
    case cucascade::memory::Tier::HOST: return "HOST";
    case cucascade::memory::Tier::DISK: return "DISK";
    case cucascade::memory::Tier::REMOTE: return "REMOTE";
    default: return "UNKNOWN";
  }
}

downgrade_executor::downgrade_executor(
  sirius::exec::query_lifecycle_registry& lifecycle,
  exec::downgrade_executor_config config,
  sirius::data::data_repository_manager_registry& data_repo_registry,
  cucascade::memory::memory_space_id space_id,
  cucascade::memory::memory_space* memory_space,
  sirius::memory::sirius_memory_reservation_manager& reservation_manager,
  sirius::exec::multi_index_priority_queue<sirius::parallel::itask>* pipeline_task_queue)
  : _query_lifecycle(lifecycle),
    _config(std::move(config)),
    _data_repo_registry(data_repo_registry),
    _space_id(space_id),
    _memory_space(memory_space),
    _source_label(tier_to_string(space_id.tier) + ":" + std::to_string(space_id.device_id)),
    _reservation_manager(reservation_manager),
    _pipeline_task_queue(pipeline_task_queue)
{
}

downgrade_executor::~downgrade_executor() { stop(); }

void downgrade_executor::start()
{
  std::lock_guard lock(_lifecycle_mutex);
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  _paused.store(false);

  absl::Cleanup rollback = [this]() {
    {
      std::lock_guard monitor_lock(_monitor_cv_mutex);
      _running.store(false);
    }
    _request_queue.interrupt();
    _monitor_cv.notify_all();
    if (_pool) { _pool->interrupt(); }
    if (_monitor_thread.joinable()) { _monitor_thread.join(); }
    if (_processing_thread.joinable()) { _processing_thread.join(); }
    if (_pool) {
      _pool->wait_all();
      _pool->stop();
      _pool.reset();
    }
    cancel_pending_requests();
    _stream_pool.reset();
  };

  // HOST/DISK tier memory_spaces return device_id == -1; passing that to
  // rmm::cuda_device_id or cudaSetDevice fails with cudaErrorInvalidDevice.
  // Default the stream pool to GPU 0 for non-GPU tiers and skip per-thread
  // CUDA binding entirely (the stream is ordering metadata; host/disk work
  // is CPU-side).
  {
    int device_id = 0;
    if (_space_id.tier == cucascade::memory::Tier::GPU && _memory_space) {
      device_id = _memory_space->get_device_id();
    }
    _stream_pool = std::make_unique<cucascade::memory::exclusive_stream_pool>(
      rmm::cuda_device_id{device_id}, _config.thread_pool.num_threads);
  }

  _request_queue.reactivate();

  sirius::exec::invocable<void() noexcept> per_thread_init = nullptr;
  if (_memory_space && _space_id.tier == cucascade::memory::Tier::GPU) {
    auto device_id  = _memory_space->get_device_id();
    per_thread_init = [device_id]() noexcept {
      // Pin each worker to its GPU; silent failure leaks downgrade memcpys
      // across contexts. Lambda is noexcept, so check inline.
      cudaError_t err = cudaSetDevice(device_id);
      if (err != cudaSuccess) {
        SIRIUS_LOG_ERROR("downgrade_executor per-thread init: cudaSetDevice({}) failed: {}",
                         device_id,
                         cudaGetErrorString(err));
      }
    };
  }

  _pool = std::make_unique<exec::bounded_thread_pool>(_config.thread_pool.num_threads,
                                                      _config.thread_pool.thread_name_prefix,
                                                      _config.thread_pool.cpu_affinity_list,
                                                      std::move(per_thread_init));

  _processing_thread = std::thread(&downgrade_executor::processing_loop, this);

  if (_memory_space && _config.monitor_period > std::chrono::milliseconds::zero()) {
    _monitor_thread = std::thread(&downgrade_executor::monitor_loop, this);
  }
  std::move(rollback).Cancel();
}

void downgrade_executor::stop()
{
  std::lock_guard lock(_lifecycle_mutex);
  {
    std::lock_guard monitor_lock(_monitor_cv_mutex);
    if (!_running.exchange(false)) { return; }
  }

  _pool->interrupt();
  _request_queue.interrupt();
  _monitor_cv.notify_one();

  if (_monitor_thread.joinable()) { _monitor_thread.join(); }
  if (_processing_thread.joinable()) { _processing_thread.join(); }

  _pool->wait_all();
  cancel_pending_requests();
  _pool->stop();
  _pool.reset();
  _stream_pool.reset();
  _paused.store(false);
}

void downgrade_executor::pause()
{
  std::lock_guard lock(_lifecycle_mutex);
  if (!_running.load() || _paused.load()) { return; }
  {
    // Change the wait predicate under its mutex so a long monitor sleep cannot miss the wakeup.
    std::lock_guard monitor_lock(_monitor_cv_mutex);
    _paused.store(true);
  }

  _pool->interrupt();
  _request_queue.interrupt();
  _monitor_cv.notify_all();

  // The monitor also borrows the cache for its viability check. Join it as well as the
  // processing thread before permitting a rebind, and stop it from publishing while paused.
  if (_monitor_thread.joinable()) { _monitor_thread.join(); }
  if (_processing_thread.joinable()) { _processing_thread.join(); }

  _pool->wait_all();
  cancel_pending_requests();
}

void downgrade_executor::resume()
{
  std::lock_guard lock(_lifecycle_mutex);
  if (!_running.load() || !_paused.load()) { return; }

  _pool->resume();
  _request_queue.reactivate();
  _paused.store(false);
  try {
    _processing_thread = std::thread(&downgrade_executor::processing_loop, this);
    if (_memory_space && _config.monitor_period > std::chrono::milliseconds::zero()) {
      _monitor_thread = std::thread(&downgrade_executor::monitor_loop, this);
    }
  } catch (...) {
    // A thread-creation failure leaves a quiescent, rebindable executor, not a half-live
    // borrower of a cache its owner may replace while unwinding.
    {
      std::lock_guard monitor_lock(_monitor_cv_mutex);
      _paused.store(true);
    }
    _pool->interrupt();
    _request_queue.interrupt();
    _monitor_cv.notify_all();
    if (_monitor_thread.joinable()) { _monitor_thread.join(); }
    if (_processing_thread.joinable()) { _processing_thread.join(); }
    _pool->wait_all();
    cancel_pending_requests();
    throw;
  }
}

void downgrade_executor::drain()
{
  pause();
  resume();
}

void downgrade_executor::set_fs_cache(cucascade::io::cache::fs_cache* cache)
{
  std::lock_guard lock(_lifecycle_mutex);
  if (_running.load() && !_paused.load()) {
    throw std::logic_error("fs_cache must be rebound before start() or while paused");
  }
  _fs_cache = cache;
}

void downgrade_executor::processing_loop()
{
  while (_running.load()) {
    auto request = _request_queue.pop();
    if (!request) break;  // interrupted

    auto& req = request;

    auto t_start = std::chrono::steady_clock::now();

    // Per-source tracking (repos vs pipeline_queue)
    struct source_stats {
      std::atomic<size_t> batches{0};
      std::atomic<size_t> bytes{0};
    };
    source_stats repo_stats, pipeline_queue_stats;

    // Index by the actual target tier: fallbacks after HOST are not necessarily DISK.
    struct target_tier_stats {
      std::atomic<size_t> batches{0};
      std::atomic<size_t> bytes{0};
    };
    std::array<target_tier_stats, static_cast<size_t>(cucascade::memory::Tier::SIZE)>
      target_stats{};
    auto& host_target_stats   = target_stats[static_cast<size_t>(cucascade::memory::Tier::HOST)];
    auto& disk_target_stats   = target_stats[static_cast<size_t>(cucascade::memory::Tier::DISK)];
    auto& remote_target_stats = target_stats[static_cast<size_t>(cucascade::memory::Tier::REMOTE)];

    cucascade::memory::memory_space* source_space = nullptr;
    std::vector<const cucascade::memory::memory_space*> target_spaces;
    bool spill_not_configured = true;
    std::mutex failed_pipeline_candidates_mutex;
    std::vector<std::unique_ptr<convertible_data>> failed_pipeline_candidates;
    std::exception_ptr dispatch_error;

    // Workers evaluate the predicate after each conversion; evaluating it before each dispatch
    // too means a request already satisfied by other means (the caller's reservation landing,
    // the running query freeing memory) spills nothing at all instead of at least one batch.
    auto predicate_satisfied = [](downgrade_request* req_ptr) noexcept {
      if (req_ptr->satisfied.load(std::memory_order_acquire)) { return true; }
      try {
        if (req_ptr->predicate && req_ptr->predicate()) {
          req_ptr->satisfied.store(true, std::memory_order_release);
          return true;
        }
      } catch (const std::exception& e) {
        try {
          SIRIUS_LOG_ERROR("[downgrade] predicate evaluation failed: {}", e.what());
        } catch (...) {
        }
      } catch (...) {
        try {
          SIRIUS_LOG_ERROR("[downgrade] predicate evaluation failed");
        } catch (...) {
        }
      }
      return false;
    };

    // Bytes freed or still in flight toward the request's byte target. Failed conversions remove
    // their contribution so another candidate can be attempted.
    std::atomic<std::size_t> planned_bytes{0};
    auto target_reached = [&req, &planned_bytes]() {
      return req->target_bytes.has_value() &&
             planned_bytes.load(std::memory_order_relaxed) >= *req->target_bytes;
    };
    auto target_completed = [&req]() {
      return req->target_bytes.has_value() &&
             req->bytes_freed.load(std::memory_order_relaxed) >= *req->target_bytes;
    };
    auto wait_if_target_reached =
      [this, &req, &predicate_satisfied, &target_reached, &target_completed]() {
        if (!target_reached()) { return false; }
        _pool->wait_all();
        return predicate_satisfied(req.get()) || target_completed();
      };

    try {
      // Resolve the source memory space for filtering candidates
      source_space = _reservation_manager.get_memory_space(_space_id.tier, _space_id.device_id);

      // GPU targets prefer HOST, then DISK, then usable REMOTE; HOST sources skip HOST.
      // NUMA preference (from downgrade_executor_config, v1.0 dd86dd0 intent re-authored on
      // the post-#637 architecture): if preferred_numa_node is set, stable_partition the matching
      // HOST space(s) to the front of target_spaces so cand->convert() tries the NUMA-local space
      // first.
      if (_space_id.tier == cucascade::memory::Tier::GPU) {
        auto host_span =
          _reservation_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
        // Copy span -> vector before reordering: the span is a view into the manager's
        // internal storage, and stable_partition would otherwise mutate it in place.
        std::vector<const cucascade::memory::memory_space*> host_spaces(host_span.begin(),
                                                                        host_span.end());
        if (auto pref = _config.preferred_numa_node; pref.has_value()) {
          std::stable_partition(host_spaces.begin(),
                                host_spaces.end(),
                                [pref_numa = *pref](const cucascade::memory::memory_space* s) {
                                  return s != nullptr &&
                                         static_cast<int>(s->get_device_id()) == pref_numa;
                                });
        }
        for (auto* hs : host_spaces) {
          target_spaces.push_back(hs);
        }
      }
      if (_space_id.tier == cucascade::memory::Tier::GPU ||
          _space_id.tier == cucascade::memory::Tier::HOST) {
        for (auto* ds :
             _reservation_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::DISK)) {
          target_spaces.push_back(ds);
          spill_not_configured = false;
        }
        for (auto* rs :
             _reservation_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::REMOTE)) {
          // Unlike DISK, REMOTE has no lazy default: credentials/context attachment may have
          // been unavailable at startup. Never reserve or dispatch a conversion to that space.
          if (rs && rs->get_io_context()) {
            target_spaces.push_back(rs);
            spill_not_configured = false;
          }
        }
      }

      // === TIER 0: default-local fs_cache (HOST only, idle chunks first) ===
      // Return unused staging buffers before spilling repository data. Credit both the
      // completed and planned totals so byte-target requests do not over-dispatch batches.
      if (_space_id.tier == cucascade::memory::Tier::HOST && _fs_cache &&
          !predicate_satisfied(req.get()) && !target_completed()) {
        auto const target =
          req->target_bytes.has_value()
            ? *req->target_bytes - req->bytes_freed.load(std::memory_order_relaxed)
            : std::numeric_limits<std::size_t>::max();
        auto const freed = reclaim_fs_cache(cucascade::io::cache::reclaim_mode::idle, target);
        req->bytes_freed.fetch_add(freed, std::memory_order_relaxed);
        planned_bytes.fetch_add(freed, std::memory_order_relaxed);
        predicate_satisfied(req.get());
      }

      // === TIER 1: Data repositories ===
      // Memory pressure is a global condition, so candidates are drawn from EVERY in-flight
      // query: outer loop DESCENDING by query id (newest query first), inner loop in
      // get_repositories() order — ascending {operator_id, port_id} — with early stop once the
      // reservation is satisfied. Operator ids restart at 0 per query, so ordering is
      // (query id, operator id) rather than a single global counter.
      //
      // Newest-first is the fairness policy: query ids are monotonic, so the newest query is the
      // one that has made the least progress, and spilling it costs the least re-materialization
      // work. It also keeps the oldest query's working set resident so it can finish and release
      // its memory, which is what actually relieves the pressure — spilling the oldest query
      // instead would slow down the query closest to completing while newer arrivals keep
      // allocating. This makes the pressure response FIFO: earliest queries keep execution
      // priority, latest queries pay for the memory.
      //
      // get_all() returns ascending by query id, so this iterates the snapshot in reverse.
      // get_all_convertible() snapshots eligible batches once per repo so a batch isn't
      // re-scanned before leaving idle. Managers are held by shared_ptr for the duration of the
      // sweep, so a query ending concurrently cannot pull one out from under this loop.
      bool pool_interrupted = false;
      auto const managers   = _data_repo_registry.candidates();
      for (auto const& [victim_id, manager] : std::views::reverse(managers)) {
        auto borrow = _query_lifecycle.try_acquire_work(victim_id);
        if (!borrow) { continue; }
        if (req->satisfied.load() || pool_interrupted || target_completed()) break;
        auto repos = manager->get_repositories();
        for (auto* repo : repos) {
          if (req->satisfied.load() || target_completed()) break;

          convertible_data_batch_provider provider(repo);
          auto candidates = provider.get_all_convertible(
            source_space, /*front_to_back=*/false, /*ignore_subscribed=*/true);

          // Snapshot candidate sizes so target-carrying requests can right-size their final pick
          // (spill_policy.hpp) instead of taking the next whole batch in policy order. Requests
          // without a target keep plain policy-order iteration.
          std::vector<std::size_t> candidate_sizes(candidates.size());
          for (std::size_t i = 0; i < candidates.size(); ++i) {
            candidate_sizes[i] = candidates[i]->bytes_in_space(source_space);
          }
          std::vector<bool> already_dispatched(candidates.size(), false);

          for (std::size_t n = 0; n < candidates.size(); ++n) {
            if (predicate_satisfied(req.get()) || wait_if_target_reached()) break;

            std::optional<std::size_t> remaining;
            if (req->target_bytes.has_value()) {
              remaining = *req->target_bytes - planned_bytes.load(std::memory_order_relaxed);
            }
            auto const pick =
              select_next_spill_candidate(candidate_sizes, already_dispatched, remaining);
            if (!pick.has_value()) break;
            already_dispatched[*pick] = true;
            auto candidate            = std::move(candidates[*pick]);
            auto candidate_bytes      = candidate_sizes[*pick];
            // The manager-local borrow protects candidate discovery. Transfer a separate
            // victim claim into each candidate before handing it to another thread.

            auto work = _query_lifecycle.try_acquire_work(victim_id);
            if (!work) { break; }
            candidate->retain_work(std::move(work));

            auto slot = _pool->reserve();
            if (!slot) {
              pool_interrupted = true;
              break;
            }

            // Re-check after reserve() returns -- the previous candidate's worker may
            // have set satisfied while we were blocked waiting for a thread slot.
            if (req->satisfied.load()) break;

            maybe_fail_stream_acquisition(stream_acquisition_site::repository);
            auto exc_stream = _stream_pool->acquire_stream(
              cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);
            planned_bytes.fetch_add(candidate_bytes, std::memory_order_relaxed);

            _pool->dispatch(
              std::move(slot),
              [cand       = std::move(candidate),
               req_ptr    = req.get(),
               &res_mgr   = _reservation_manager,
               &targets   = target_spaces,
               exc_stream = std::move(exc_stream),
               candidate_bytes,
               &planned_bytes,
               &repo_stats,
               &target_stats,
               predicate_satisfied]() mutable {
                absl::Cleanup rollback = [&planned_bytes, candidate_bytes]() {
                  planned_bytes.fetch_sub(candidate_bytes, std::memory_order_relaxed);
                };
                try {
                  nvtx3::scoped_range nvtx_range{"sirius::downgrade::convert_batch"};
                  auto result =
                    cand->convert(targets, ::cuda::stream_ref{exc_stream.get()}, res_mgr, false);
                  if (result) {
                    std::move(rollback).Cancel();
                    req_ptr->bytes_freed.fetch_add(candidate_bytes, std::memory_order_relaxed);
                    req_ptr->batches_downgraded.fetch_add(1, std::memory_order_relaxed);
                    repo_stats.batches.fetch_add(1, std::memory_order_relaxed);
                    repo_stats.bytes.fetch_add(candidate_bytes, std::memory_order_relaxed);
                    for (size_t i = 0; i < result->size(); ++i) {
                      if ((*result)[i] == 0) continue;
                      auto& stats = target_stats[static_cast<size_t>(targets[i]->get_tier())];
                      stats.batches.fetch_add(1, std::memory_order_relaxed);
                      stats.bytes.fetch_add((*result)[i], std::memory_order_relaxed);
                    }
                    predicate_satisfied(req_ptr);
                  }
                } catch (const std::exception& e) {
                  SIRIUS_LOG_ERROR("[downgrade] convert failed from data repository: {}", e.what());
                } catch (...) {
                  SIRIUS_LOG_ERROR("[downgrade] convert failed from data repository");
                }
              });
          }
          if (pool_interrupted) break;
        }
      }

      // === TIER 2: task_scheduler task queue ===
      if (!req->satisfied.load() && !target_completed() && _pipeline_task_queue) {
        size_t max_tasks_to_convert = _pipeline_task_queue->size();
        size_t tasks_converted      = 0;
        failed_pipeline_candidates.reserve(max_tasks_to_convert);
        // The gate travels with the provider: each wrapper it hands out consults it before its RAII
        // re-push, so a task extracted here cannot land back in the queue after its query's drain.
        convertible_gpu_pipeline_task_provider pipeline_provider(*_pipeline_task_queue,
                                                                 _query_lifecycle);
        while (!req->satisfied.load() && tasks_converted < max_tasks_to_convert) {
          if (predicate_satisfied(req.get()) || wait_if_target_reached()) break;
          auto candidate =
            pipeline_provider.get_next_convertible(source_space, /*front_to_back=*/false);
          if (!candidate) break;
          tasks_converted++;

          auto candidate_bytes = candidate->bytes_in_space(source_space);

          auto slot = _pool->reserve();
          if (!slot) break;  // interrupted

          if (req->satisfied.load()) break;

          maybe_fail_stream_acquisition(stream_acquisition_site::pipeline_queue);
          auto exc_stream = _stream_pool->acquire_stream(
            cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);
          planned_bytes.fetch_add(candidate_bytes, std::memory_order_relaxed);

          _pool->dispatch(
            std::move(slot),
            [cand       = std::move(candidate),
             req_ptr    = req.get(),
             &res_mgr   = _reservation_manager,
             &targets   = target_spaces,
             exc_stream = std::move(exc_stream),
             candidate_bytes,
             &planned_bytes,
             &failed_pipeline_candidates_mutex,
             &failed_pipeline_candidates,
             &pipeline_queue_stats,
             &target_stats,
             predicate_satisfied]() mutable {
              absl::Cleanup rollback = [&]() {
                planned_bytes.fetch_sub(candidate_bytes, std::memory_order_relaxed);
                std::lock_guard lock(failed_pipeline_candidates_mutex);
                failed_pipeline_candidates.push_back(std::move(cand));
              };
              try {
                nvtx3::scoped_range nvtx_range{"sirius::downgrade::convert_task_batches"};
                auto result =
                  cand->convert(targets, ::cuda::stream_ref{exc_stream.get()}, res_mgr, false);
                if (result) {
                  std::move(rollback).Cancel();
                  req_ptr->bytes_freed.fetch_add(candidate_bytes, std::memory_order_relaxed);
                  req_ptr->batches_downgraded.fetch_add(1, std::memory_order_relaxed);
                  pipeline_queue_stats.batches.fetch_add(1, std::memory_order_relaxed);
                  pipeline_queue_stats.bytes.fetch_add(candidate_bytes, std::memory_order_relaxed);
                  for (size_t i = 0; i < result->size(); ++i) {
                    if ((*result)[i] == 0) continue;
                    auto& stats = target_stats[static_cast<size_t>(targets[i]->get_tier())];
                    stats.batches.fetch_add(1, std::memory_order_relaxed);
                    stats.bytes.fetch_add((*result)[i], std::memory_order_relaxed);
                  }
                  predicate_satisfied(req_ptr);
                }
              } catch (const std::exception& e) {
                SIRIUS_LOG_ERROR("[downgrade] convert failed from task queue: {}", e.what());
              } catch (...) {
                SIRIUS_LOG_ERROR("[downgrade] convert failed from task queue");
              }
            });
        }
      }

    } catch (...) {
      dispatch_error = std::current_exception();
    }

    // Dispatch-local slots must unwind before this wait; all worker-borrowed state stays alive
    // until it returns.
    _pool->wait_all();
    failed_pipeline_candidates.clear();

    // === TIER 3: default-local fs_cache (HOST only, forced final pass) ===
    // Only a known, still-unmet byte target justifies dropping subscribed cache warmth.
    // All workers have settled, so the deficit reflects actual rather than planned releases.
    if (!dispatch_error && _space_id.tier == cucascade::memory::Tier::HOST && _fs_cache &&
        req->target_bytes.has_value() && !predicate_satisfied(req.get()) && !target_completed()) {
      try {
        auto const remaining =
          *req->target_bytes - req->bytes_freed.load(std::memory_order_relaxed);
        auto const freed = reclaim_fs_cache(cucascade::io::cache::reclaim_mode::forced, remaining);
        req->bytes_freed.fetch_add(freed, std::memory_order_relaxed);
        planned_bytes.fetch_add(freed, std::memory_order_relaxed);
        predicate_satisfied(req.get());
      } catch (...) {
        dispatch_error = std::current_exception();
      }
    }

    if (req->is_monitor_request) {
      _monitor_request_enqueued.store(false, std::memory_order_relaxed);
    }

    if (dispatch_error) {
      if (req->is_monitor_request) {
        log::log_noexcept([&] {
          try {
            std::rethrow_exception(dispatch_error);
          } catch (const std::exception& e) {
            SIRIUS_LOG_ERROR(
              "[downgrade][monitor] [{}] request failed: {}", _source_label, e.what());
          } catch (...) {
            SIRIUS_LOG_ERROR("[downgrade][monitor] [{}] request failed", _source_label);
          }
        });
      }
    } else {
      log::log_noexcept([&] {
        // Monitor requests are gated by has_viable_downgrade_target() and warn once per stall
        // episode in monitor_loop(); only warn here for one-shot (external) requests to avoid log
        // spam.
        if (spill_not_configured && !req->satisfied.load() && !req->is_monitor_request) {
          SIRIUS_LOG_WARN(
            "[downgrade] [{}] downgrade request not satisfied and no disk or usable remote "
            "memory space is configured. Configure a disk or remote memory space to enable "
            "spilling.",
            _source_label);
        }

        // === Logging ===
        auto total_bytes   = req->bytes_freed.load(std::memory_order_relaxed);
        auto total_batches = req->batches_downgraded.load(std::memory_order_relaxed);
        auto duration_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start)
            .count();
        double throughput_mbs =
          (duration_ms > 0.0) ? (total_bytes / (1024.0 * 1024.0)) / (duration_ms / 1000.0) : 0.0;
        std::string request_label = req->is_monitor_request ? "monitor " : "";
        if (req->is_monitor_request) {
          // Outcome of the periodic check, at INFO so a normal run shows what the
          // monitor path actually accomplished. `satisfied` is the honest signal:
          // bytes freed can be non-zero while the request still failed to reach its
          // target, which is exactly the silent-failure case that hid the broken
          // spill directory (every convert threw, nothing was freed, no warning).
          SIRIUS_LOG_INFO(
            "[downgrade][monitor] [{}] request done: {} batches, {} bytes in {:.2f} ms ({:.1f} "
            "MB/s), "
            "satisfied={} | to_host: {}/{} batches/bytes, to_disk: {}/{} batches/bytes, "
            "to_remote: {}/{} batches/bytes",
            _source_label,
            total_batches,
            total_bytes,
            duration_ms,
            throughput_mbs,
            req->satisfied.load(),
            host_target_stats.batches.load(std::memory_order_relaxed),
            host_target_stats.bytes.load(std::memory_order_relaxed),
            disk_target_stats.batches.load(std::memory_order_relaxed),
            disk_target_stats.bytes.load(std::memory_order_relaxed),
            remote_target_stats.batches.load(std::memory_order_relaxed),
            remote_target_stats.bytes.load(std::memory_order_relaxed));
        }

        SIRIUS_LOG_DEBUG(
          "[downgrade] [{}] request {}done: {} batches, {} bytes (target {}, planned {}) in "
          "{:.2f} ms ({:.1f} MB/s) | "
          "repos: {}/{} batches/bytes, pipeline_queue: {}/{} | "
          "to_host: {}/{} batches/bytes, to_disk: {}/{} batches/bytes, "
          "to_remote: {}/{} batches/bytes",
          _source_label,
          request_label,
          total_batches,
          total_bytes,
          req->target_bytes.has_value() ? std::to_string(*req->target_bytes) : std::string("none"),
          planned_bytes.load(std::memory_order_relaxed),
          duration_ms,
          throughput_mbs,
          repo_stats.batches.load(std::memory_order_relaxed),
          repo_stats.bytes.load(std::memory_order_relaxed),
          pipeline_queue_stats.batches.load(std::memory_order_relaxed),
          pipeline_queue_stats.bytes.load(std::memory_order_relaxed),
          host_target_stats.batches.load(std::memory_order_relaxed),
          host_target_stats.bytes.load(std::memory_order_relaxed),
          disk_target_stats.batches.load(std::memory_order_relaxed),
          disk_target_stats.bytes.load(std::memory_order_relaxed),
          remote_target_stats.batches.load(std::memory_order_relaxed),
          remote_target_stats.bytes.load(std::memory_order_relaxed));
      });
    }

    // Fulfill the promise
    if (dispatch_error) {
      req->result.set_exception(dispatch_error);
    } else {
      req->result.set_value(req->bytes_freed.load(std::memory_order_relaxed));
    }
  }
}

void downgrade_executor::maybe_fail_stream_acquisition(stream_acquisition_site site)
{
  if (!_stream_acquisition_fault || _stream_acquisition_fault->site != site) { return; }
  if (_stream_acquisition_fault->skip != 0) {
    --_stream_acquisition_fault->skip;
    return;
  }
  auto fault = std::move(*_stream_acquisition_fault);
  _stream_acquisition_fault.reset();
  if (fault.before_failure) { fault.before_failure(); }
  std::rethrow_exception(fault.error);
}

std::size_t downgrade_executor::reclaim_fs_cache(cucascade::io::cache::reclaim_mode mode,
                                                 std::size_t deficit)
{
  auto const numa       = _space_id.device_id;
  auto const releasable = _fs_cache->releasable_bytes(numa);
  if (releasable == 0) { return 0; }
  auto const freed = _fs_cache->reclaim(mode, std::min(deficit, releasable), numa);
  // Only the overflow above the reservation returns to this HOST space.
  return std::min(freed, releasable);
}

bool downgrade_executor::has_disk_tier() const
{
  return !_reservation_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::DISK).empty();
}

bool downgrade_executor::has_viable_downgrade_target() const
{
  using cucascade::memory::Tier;

  // Only GPU/HOST sources have supported lower-tier converters. DISK->REMOTE is deferred.
  if (_space_id.tier != Tier::GPU && _space_id.tier != Tier::HOST) { return false; }

  // DISK adopts its default ioctx lazily. REMOTE is viable only with an attached context.
  if (has_disk_tier()) { return true; }
  for (auto* remote : _reservation_manager.get_memory_spaces_for_tier(Tier::REMOTE)) {
    if (remote && remote->get_io_context()) { return true; }
  }

  // HOST cache reclaim needs no DISK target, but chunks inside the prefetch reservation
  // do not relieve pressure. Only overflow on this executor's NUMA space counts.
  if (_space_id.tier == Tier::HOST) {
    return _fs_cache && _fs_cache->releasable_bytes(_space_id.device_id) != 0;
  }

  // Without a usable spill sink, only a GPU source can downgrade to HOST.
  if (_space_id.tier != Tier::GPU) { return false; }

  // GPU source, no spill sink: viable iff some HOST space can currently accept a reservation.
  // Probe with exactly the operation a downgrade performs -- make_reservation_or_null of one chunk,
  // released immediately (RAII). This is the ground truth: HOST reserve() is bounded by
  // _allocated_bytes, which reflects BOTH live reservations AND already-stored downgraded data, so
  // neither get_total_reserved_memory nor get_available_memory alone captures whether a downgrade
  // can land. (The chunk-sized probe matches the chunked allocator's rounding; a sub-chunk request
  // would succeed against a remainder that no real batch could use.)
  for (const auto* hs : _reservation_manager.get_memory_spaces_for_tier(Tier::HOST)) {
    if (!hs) { continue; }
    // The manager owns these spaces mutably; the span just exposes them as const. make_reservation
    // mutates accounting, so we need a mutable handle.
    auto* host         = const_cast<cucascade::memory::memory_space*>(hs);
    size_t probe_bytes = 1;
    if (const auto* chunked = host->get_chunked_resource_info()) {
      probe_bytes = chunked->max_chunk_bytes();
    }
    // A chunk larger than this space's reservation limit can never be reserved (the HOST allocator
    // throws on an over-limit request), so such a space cannot accept a downgrade -- skip it. The
    // try/catch is belt-and-suspenders: this runs on the monitor thread, which must never throw.
    if (probe_bytes > host->get_max_memory()) { continue; }
    try {
      if (auto probe = host->make_reservation_or_null(probe_bytes)) { return true; }
    } catch (const std::exception& e) {
      SIRIUS_LOG_DEBUG("[downgrade] [{}] host viability probe failed: {}", _source_label, e.what());
    }
  }
  return false;
}

void downgrade_executor::monitor_loop()
{
  using namespace std::chrono_literals;

  // Monitor-thread-local; throttles the back-off warning to once per stall episode.
  bool backed_off = false;

  while (_running.load() && !_paused.load()) {
    if (_memory_space && _memory_space->should_downgrade_memory() &&
        !_monitor_request_enqueued.load(std::memory_order_relaxed)) {
      // Stateless viability gate: only issue a downgrade request when one could plausibly free
      // memory. When idle GPU batches' only lower tier is a full HOST and no spill sink is usable,
      // re-firing would just re-scan every repository and the task queue, free nothing, and spam
      // the log every monitor_period (~100x/s by default) forever. Skipping the cycle backs
      // off cleanly; because this is re-checked every cycle the monitor resumes the instant host
      // frees or pressure drops -- there is no latched state to get wedged on.
      if (has_viable_downgrade_target()) {
        backed_off    = false;
        size_t amount = _memory_space->get_amount_to_downgrade();
        if (amount > 0) {
          auto req                = std::make_unique<downgrade_request>();
          req->is_monitor_request = true;
          req->target_bytes       = amount;
          // Count concurrent memory releases and stop once usage reaches the stop threshold.
          req->predicate = [space = _memory_space, &freed = req->bytes_freed, amount]() {
            return freed.load(std::memory_order_relaxed) >= amount ||
                   space->should_stop_downgrading_memory();
          };
          auto const issued = _monitor_requests_issued.fetch_add(1, std::memory_order_relaxed) + 1;
          _monitor_request_enqueued.store(true, std::memory_order_relaxed);
          // Periodic-path attribution, counterpart to the "[downgrade][reserve]" line
          // in gpu_pipeline_executor. Throttled in practice by _monitor_request_enqueued
          // (one monitor request in flight at a time), so this is one line per
          // request round-trip -- not one per candidate batch like the convert errors.
          SIRIUS_LOG_INFO(
            "[downgrade][monitor] [{}] memory pressure: requesting downgrade of {} bytes "
            "(monitor request #{})",
            _source_label,
            amount,
            issued);
          // Fire-and-forget: monitor does not wait for the result
          _request_queue.push(std::move(req));
        }
      } else if (!backed_off) {
        SIRIUS_LOG_WARN(
          "[downgrade] [{}] memory pressure but no viable downgrade target (host full, no disk "
          "or usable remote configured); backing off until memory is released. Consider "
          "configuring a disk or remote memory space to enable spilling.",
          _source_label);
        backed_off = true;
      }
    } else {
      // Pressure gone -- reset so the next stall episode warns again.
      backed_off = false;
    }
    // Wait for the monitor period, but wake immediately on shutdown.
    std::unique_lock<std::mutex> lock(_monitor_cv_mutex);
    _monitor_cv.wait_for(lock, _config.monitor_period, [this]() {
      return !_running.load(std::memory_order_relaxed) || _paused.load(std::memory_order_relaxed);
    });
  }
}

void downgrade_executor::cancel_pending_requests()
{
  while (auto req = _request_queue.try_pop()) {
    try {
      req->result.set_exception(
        std::make_exception_ptr(std::runtime_error("downgrade executor shutting down")));
    } catch (...) {
      // Promise may already be fulfilled — safe to ignore
    }
  }
  // All producers/consumers are joined by callers. A cancelled queued monitor request must not
  // leave the monitor permanently believing that a request is still in flight after resume().
  _monitor_request_enqueued.store(false, std::memory_order_relaxed);
}

void downgrade_executor::set_pipeline_task_queue(
  sirius::exec::multi_index_priority_queue<sirius::parallel::itask>* pipeline_task_queue)
{
  _pipeline_task_queue = pipeline_task_queue;
}

// --- Public request API ---

std::future<size_t> downgrade_executor::request_free_memory(size_t bytes)
{
  auto req          = std::make_unique<downgrade_request>();
  req->target_bytes = bytes;
  req->predicate    = [&freed = req->bytes_freed, bytes]() {
    return freed.load(std::memory_order_relaxed) >= bytes;
  };
  auto future = req->result.get_future();
  if (!_request_queue.push(std::move(req))) {
    SIRIUS_LOG_WARN(
      "[downgrade] request_free_memory: queue inactive, dropping request for {} bytes", bytes);
  }
  return future;
}

size_t downgrade_executor::request_free_memory_and_wait(size_t bytes)
{
  return request_free_memory(bytes).get();
}

std::future<size_t> downgrade_executor::request_downgrade(std::function<bool()> predicate)
{
  auto req       = std::make_unique<downgrade_request>();
  req->predicate = std::move(predicate);
  auto future    = req->result.get_future();
  if (!_request_queue.push(std::move(req))) {
    SIRIUS_LOG_WARN("[downgrade] request_downgrade: queue inactive, dropping request");
    return future;
  }
  return future;
}

}  // namespace parallel
}  // namespace sirius
