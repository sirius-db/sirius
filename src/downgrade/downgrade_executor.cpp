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
#include "log/logging.hpp"

#include <algorithm>
#include <chrono>
#include <ranges>
#include <thread>
#include <vector>

namespace sirius {
namespace parallel {

static std::string tier_to_string(cucascade::memory::Tier tier)
{
  switch (tier) {
    case cucascade::memory::Tier::GPU: return "GPU";
    case cucascade::memory::Tier::HOST: return "HOST";
    case cucascade::memory::Tier::DISK: return "DISK";
    default: return "UNKNOWN";
  }
}

downgrade_executor::downgrade_executor(
  exec::downgrade_executor_config config,
  sirius::data::data_repository_manager_registry& data_repo_registry,
  cucascade::memory::memory_space_id space_id,
  cucascade::memory::memory_space* memory_space,
  sirius::memory::sirius_memory_reservation_manager& reservation_manager,
  sirius::exec::multi_index_priority_queue<sirius::parallel::itask>* pipeline_task_queue)
  : _config(std::move(config)),
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
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }

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
}

void downgrade_executor::stop()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }

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
}

void downgrade_executor::drain()
{
  _pool->interrupt();
  _request_queue.interrupt();

  if (_processing_thread.joinable()) { _processing_thread.join(); }

  _pool->wait_all();
  cancel_pending_requests();
  _pool->resume();
  _request_queue.reactivate();

  _processing_thread = std::thread(&downgrade_executor::processing_loop, this);
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

    // Per-target-tier tracking (host vs disk)
    struct target_tier_stats {
      std::atomic<size_t> batches{0};
      std::atomic<size_t> bytes{0};
    };
    target_tier_stats host_target_stats, disk_target_stats;

    // Resolve the source memory space for filtering candidates
    auto* source_space = _reservation_manager.get_memory_space(_space_id.tier, _space_id.device_id);

    // Build target spaces list: for GPU->HOST downgrade, target is HOST tier followed by DISK tier.
    // NUMA preference (from downgrade_executor_config, v1.0 dd86dd0 intent re-authored on the
    // post-#637 architecture): if preferred_numa_node is set, stable_partition the matching HOST
    // space(s) to the front of target_spaces so cand->convert() tries the NUMA-local space first.
    std::vector<const cucascade::memory::memory_space*> target_spaces;
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
    size_t host_end_idx = target_spaces.size();
    auto disk_spaces =
      _reservation_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::DISK);
    bool disk_not_configured = disk_spaces.empty();
    for (auto* ds : disk_spaces) {
      target_spaces.push_back(ds);
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
    auto const managers   = _data_repo_registry.get_all();
    for (auto const& manager : std::views::reverse(managers)) {
      if (req->satisfied.load() || pool_interrupted) break;
      auto repos = manager->get_repositories();
      for (auto* repo : repos) {
        if (req->satisfied.load()) break;

        convertible_data_batch_provider provider(repo);
        auto candidates = provider.get_all_convertible(
          source_space, /*front_to_back=*/false, /*ignore_subscribed=*/true);
        for (auto& candidate : candidates) {
          if (req->satisfied.load()) break;

          auto candidate_bytes = candidate->bytes_in_space(source_space);

          auto slot = _pool->reserve();
          if (!slot) {
            pool_interrupted = true;
            break;
          }

          // Re-check after reserve() returns -- the previous candidate's worker may
          // have set satisfied while we were blocked waiting for a thread slot.
          if (req->satisfied.load()) break;

          auto exc_stream = _stream_pool->acquire_stream(
            cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);

          _pool->dispatch(
            std::move(slot),
            [cand       = std::move(candidate),
             req_ptr    = req.get(),
             &res_mgr   = _reservation_manager,
             &targets   = target_spaces,
             exc_stream = std::move(exc_stream),
             candidate_bytes,
             host_end_idx,
             &repo_stats,
             &host_target_stats,
             &disk_target_stats]() mutable {
              try {
                auto result = cand->convert(targets, exc_stream, res_mgr, false);
                if (result) {
                  req_ptr->bytes_freed.fetch_add(candidate_bytes, std::memory_order_relaxed);
                  req_ptr->batches_downgraded.fetch_add(1, std::memory_order_relaxed);
                  repo_stats.batches.fetch_add(1, std::memory_order_relaxed);
                  repo_stats.bytes.fetch_add(candidate_bytes, std::memory_order_relaxed);
                  for (size_t i = 0; i < result->size(); ++i) {
                    if ((*result)[i] == 0) continue;
                    if (i < host_end_idx) {
                      host_target_stats.batches.fetch_add(1, std::memory_order_relaxed);
                      host_target_stats.bytes.fetch_add((*result)[i], std::memory_order_relaxed);
                    } else {
                      disk_target_stats.batches.fetch_add(1, std::memory_order_relaxed);
                      disk_target_stats.bytes.fetch_add((*result)[i], std::memory_order_relaxed);
                    }
                  }
                  if (req_ptr->predicate && req_ptr->predicate()) {
                    req_ptr->satisfied.store(true);
                  }
                }
              } catch (const std::exception& e) {
                SIRIUS_LOG_ERROR("[downgrade] convert failed from data repository: {}", e.what());
              }
            });
        }
        if (pool_interrupted) break;
      }
    }

    // === TIER 2: task_scheduler task queue ===
    if (!req->satisfied.load() && _pipeline_task_queue) {
      size_t max_tasks_to_convert = _pipeline_task_queue->size();
      size_t tasks_converted      = 0;
      convertible_gpu_pipeline_task_provider pipeline_provider(*_pipeline_task_queue);
      while (!req->satisfied.load() && tasks_converted < max_tasks_to_convert) {
        auto candidate =
          pipeline_provider.get_next_convertible(source_space, /*front_to_back=*/false);
        if (!candidate) break;
        tasks_converted++;

        auto candidate_bytes = candidate->bytes_in_space(source_space);

        auto slot = _pool->reserve();
        if (!slot) break;  // interrupted

        if (req->satisfied.load()) break;

        auto exc_stream = _stream_pool->acquire_stream(
          cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);

        _pool->dispatch(
          std::move(slot),
          [cand       = std::move(candidate),
           req_ptr    = req.get(),
           &res_mgr   = _reservation_manager,
           &targets   = target_spaces,
           exc_stream = std::move(exc_stream),
           candidate_bytes,
           host_end_idx,
           &pipeline_queue_stats,
           &host_target_stats,
           &disk_target_stats]() mutable {
            try {
              auto result = cand->convert(targets, exc_stream, res_mgr, false);
              if (result) {
                req_ptr->bytes_freed.fetch_add(candidate_bytes, std::memory_order_relaxed);
                req_ptr->batches_downgraded.fetch_add(1, std::memory_order_relaxed);
                pipeline_queue_stats.batches.fetch_add(1, std::memory_order_relaxed);
                pipeline_queue_stats.bytes.fetch_add(candidate_bytes, std::memory_order_relaxed);
                for (size_t i = 0; i < result->size(); ++i) {
                  if ((*result)[i] == 0) continue;
                  if (i < host_end_idx) {
                    host_target_stats.batches.fetch_add(1, std::memory_order_relaxed);
                    host_target_stats.bytes.fetch_add((*result)[i], std::memory_order_relaxed);
                  } else {
                    disk_target_stats.batches.fetch_add(1, std::memory_order_relaxed);
                    disk_target_stats.bytes.fetch_add((*result)[i], std::memory_order_relaxed);
                  }
                }
                if (req_ptr->predicate && req_ptr->predicate()) { req_ptr->satisfied.store(true); }
              }
            } catch (const std::exception& e) {
              SIRIUS_LOG_ERROR("[downgrade] convert failed from task queue: {}", e.what());
            }
          });
      }
    }

    // Wait for all in-flight work to finish (predicate also checked in workers)
    _pool->wait_all();

    // Monitor requests are gated by has_viable_downgrade_target() and warn once per stall episode
    // in monitor_loop(); only warn here for one-shot (external) requests to avoid log spam.
    if (disk_not_configured && !req->satisfied.load() && !req->is_monitor_request) {
      SIRIUS_LOG_WARN(
        "[downgrade] [{}] downgrade request not satisfied and disk memory space is not configured; "
        "data cannot be spilled to disk. Consider configuring a disk memory space to enable "
        "spilling.",
        _source_label);
    }

    // === Logging ===
    auto total_bytes   = req->bytes_freed.load(std::memory_order_relaxed);
    auto total_batches = req->batches_downgraded.load(std::memory_order_relaxed);
    auto duration_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    double throughput_mbs =
      (duration_ms > 0.0) ? (total_bytes / (1024.0 * 1024.0)) / (duration_ms / 1000.0) : 0.0;
    std::string request_label = req->is_monitor_request ? "monitor " : "";
    if (req->is_monitor_request) {
      _monitor_request_enqueued.store(false, std::memory_order_relaxed);
      // Outcome of the periodic check, at INFO so a normal run shows what the
      // monitor path actually accomplished. `satisfied` is the honest signal:
      // bytes freed can be non-zero while the request still failed to reach its
      // target, which is exactly the silent-failure case that hid the broken
      // spill directory (every convert threw, nothing was freed, no warning).
      SIRIUS_LOG_INFO(
        "[downgrade][monitor] [{}] request done: {} batches, {} bytes in {:.2f} ms ({:.1f} MB/s), "
        "satisfied={} | to_host: {}/{} batches/bytes, to_disk: {}/{} batches/bytes",
        _source_label,
        total_batches,
        total_bytes,
        duration_ms,
        throughput_mbs,
        req->satisfied.load(),
        host_target_stats.batches.load(std::memory_order_relaxed),
        host_target_stats.bytes.load(std::memory_order_relaxed),
        disk_target_stats.batches.load(std::memory_order_relaxed),
        disk_target_stats.bytes.load(std::memory_order_relaxed));
    }

    SIRIUS_LOG_DEBUG(
      "[downgrade] [{}] request {}done: {} batches, {} bytes in {:.2f} ms ({:.1f} MB/s) | "
      "repos: {}/{} batches/bytes, pipeline_queue: {}/{} | "
      "to_host: {}/{} batches/bytes, to_disk: {}/{} batches/bytes",
      _source_label,
      request_label,
      total_batches,
      total_bytes,
      duration_ms,
      throughput_mbs,
      repo_stats.batches.load(std::memory_order_relaxed),
      repo_stats.bytes.load(std::memory_order_relaxed),
      pipeline_queue_stats.batches.load(std::memory_order_relaxed),
      pipeline_queue_stats.bytes.load(std::memory_order_relaxed),
      host_target_stats.batches.load(std::memory_order_relaxed),
      host_target_stats.bytes.load(std::memory_order_relaxed),
      disk_target_stats.batches.load(std::memory_order_relaxed),
      disk_target_stats.bytes.load(std::memory_order_relaxed));

    // Fulfill the promise
    req->result.set_value(total_bytes);
  }
}

bool downgrade_executor::has_disk_tier() const
{
  return !_reservation_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::DISK).empty();
}

bool downgrade_executor::has_viable_downgrade_target() const
{
  using cucascade::memory::Tier;

  // DISK is an effectively unbounded sink and a valid target for any source tier.
  if (has_disk_tier()) { return true; }

  // Without a disk tier, only a GPU source has a lower tier to downgrade to (HOST). processing_loop
  // only adds HOST target spaces when the source is GPU, so a HOST (or other) source has no target
  // at all and can never free anything -- it must back off rather than re-fire.
  if (_space_id.tier != Tier::GPU) { return false; }

  // GPU source, no disk: viable iff some HOST space can currently accept a reservation. Probe with
  // exactly the operation a downgrade performs -- make_reservation_or_null of one chunk, released
  // immediately (RAII). This is the ground truth: HOST reserve() is bounded by _allocated_bytes,
  // which reflects BOTH live reservations AND already-stored downgraded data, so neither
  // get_total_reserved_memory nor get_available_memory alone captures whether a downgrade can land.
  // (The chunk-sized probe matches the chunked allocator's rounding; a sub-chunk request would
  // succeed against a remainder that no real batch could use.)
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

  while (_running.load()) {
    if (_memory_space && _memory_space->should_downgrade_memory() &&
        !_monitor_request_enqueued.load(std::memory_order_relaxed)) {
      // Stateless viability gate: only issue a downgrade request when one could plausibly free
      // memory. When idle GPU batches' only lower tier is a full HOST and no DISK is configured,
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
          req->predicate          = [&freed = req->bytes_freed, amount]() {
            return freed.load(std::memory_order_relaxed) >= amount;
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
          "configured); backing off until memory is released. Consider configuring a disk memory "
          "space to enable spilling.",
          _source_label);
        backed_off = true;
      }
    } else {
      // Pressure gone -- reset so the next stall episode warns again.
      backed_off = false;
    }
    // Wait for the monitor period, but wake immediately on shutdown.
    std::unique_lock<std::mutex> lock(_monitor_cv_mutex);
    _monitor_cv.wait_for(
      lock, _config.monitor_period, [this]() { return !_running.load(std::memory_order_relaxed); });
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
}

void downgrade_executor::set_pipeline_task_queue(
  sirius::exec::multi_index_priority_queue<sirius::parallel::itask>* pipeline_task_queue)
{
  _pipeline_task_queue = pipeline_task_queue;
}

// --- Public request API ---

std::future<size_t> downgrade_executor::request_free_memory(size_t bytes)
{
  auto req       = std::make_unique<downgrade_request>();
  req->predicate = [&freed = req->bytes_freed, bytes]() {
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
