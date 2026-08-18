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

#include "compression/compression_alloc_stats.hpp"
#include "compression/output_compression.hpp"
#include "compression/spill_context.hpp"
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

  absl::AnyInvocable<void() noexcept> per_thread_init = nullptr;
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
  // Bind this thread to the GPU, exactly as the worker pool's per_thread_init does.
  //
  // The in-place compression pass below runs HERE, on the processing thread, not on a
  // pool worker — so without this the thread has no current CUDA context. That is not
  // merely untidy: cudaSetDevice is what makes the device's primary context current,
  // and simpatico derives its JIT CUfunction lazily on whichever thread first asks
  // (CompiledKernel::func_for_current_device, keyed by device id, not by context). If
  // this thread got there first, cuKernelGetFunction handed back a function that
  // cuLaunchKernel then rejected with CUDA_ERROR_INVALID_HANDLE — "invalid resource
  // handle" — and every in-place compression attempt declined.
  //
  // That failure was previously read as memory pressure, because it only ever showed
  // up during a downgrade. It is not: with task-output compression also enabled, some
  // task-executor thread populates the CUfunction cache first, this thread hits the
  // warm entry, and the identical q3/SF100 run goes from 0/78 batches compressed to
  // 76/76. The trigger was cache-warm order, not free memory.
  if (_memory_space && _space_id.tier == cucascade::memory::Tier::GPU) {
    const int device_id   = _memory_space->get_device_id();
    const cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
      SIRIUS_LOG_ERROR("downgrade_executor processing_loop: cudaSetDevice({}) failed: {}",
                       device_id,
                       cudaGetErrorString(err));
    }
  }

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

    // === TIER 0: Compress in place on the device ===
    //
    // Cheapest option when it works: the batch stays on the GPU and stays usable,
    // so there is no D2H copy now and no readback later — only a decode when a
    // consumer materializes it.
    //
    // Runs for every request, with or without an explicit byte target. Almost all
    // downgrade traffic is the monitor's predicate-only request (measured: 24414
    // monitor vs ~0 explicit on an SF100 sweep), so gating this on
    // `requested_bytes > 0` made it dead code — it never executed once across a
    // full 4-arm sweep despite 18137 requests.
    //
    // Where a target exists, the set is still priced against it first:
    // compressing frees size*(1 - 1/ratio), so candidate bytes C against request R
    // need a ratio of at least C/(C-R), and C <= R cannot be satisfied at any
    // ratio. That check is worth keeping because a compression that under-delivers
    // is worse than a spill that under-delivers — it spent GPU time, freed too
    // little, AND left a batch that must be decoded before use. With no target
    // there is nothing to price against, so each candidate is judged on its own
    // predicted saving and the loop stops as soon as the request's predicate is
    // satisfied.
    std::size_t inplace_batches = 0;
    std::size_t inplace_freed   = 0;
    if (compression::device_compression_downgrade_enabled() && !req->satisfied.load()) {
      const std::size_t requested = req->requested_bytes;  // 0 = predicate-only
      std::vector<std::unique_ptr<convertible_data>> picks;
      std::size_t predicted_total = 0;

      // Same traversal as TIER 1 below: memory pressure is global, so candidates come
      // from every in-flight query, newest first (get_all() is ascending by query id).
      auto const managers = _data_repo_registry.get_all();
      for (auto const& manager : std::views::reverse(managers)) {
        if (requested > 0 && predicted_total >= requested) break;
        for (auto* repo : manager->get_repositories()) {
          if (requested > 0 && predicted_total >= requested) break;
          convertible_data_batch_provider provider(repo);
          for (auto& cand : provider.get_all_convertible(source_space,
                                                         /*front_to_back=*/false,
                                                         /*ignore_subscribed=*/true)) {
            if (!cand) continue;
            const std::size_t saving = cand->predicted_compression_saving();
            if (saving == 0) continue;
            predicted_total += saving;
            picks.push_back(std::move(cand));
            if (requested > 0 && predicted_total >= requested) break;
          }
        }
      }

      const bool set_is_sufficient = requested == 0 || predicted_total >= requested;
      if (set_is_sufficient && !picks.empty()) {
        auto exc_stream = _stream_pool->acquire_stream(
          cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);
        for (auto& cand : picks) {
          const std::size_t freed = cand->compress_in_place(exc_stream);
          if (freed > 0) {
            ++inplace_batches;
            inplace_freed += freed;
            req->bytes_freed.fetch_add(freed, std::memory_order_relaxed);
            req->batches_downgraded.fetch_add(1, std::memory_order_relaxed);
          }
          // With no byte target the predicate is the only stopping condition, so
          // check it per batch rather than compressing every candidate we found.
          if (req->predicate && req->predicate()) {
            req->satisfied.store(true);
            break;
          }
        }
        SIRIUS_LOG_DEBUG(
          "[downgrade] [{}] in-place compression: {}/{} batches compressed, predicted {} bytes, "
          "freed {} bytes (request target {} bytes)",
          _source_label,
          inplace_batches,
          picks.size(),
          predicted_total,
          inplace_freed,
          requested);
      } else if (!picks.empty()) {
        // Targeted request the set cannot meet: deliberately all-or-nothing.
        // Compressing a subset would spend GPU time, still leave the request
        // unmet, and leave every batch it touched needing a decode — strictly
        // worse than spilling those same batches.
        SIRIUS_LOG_DEBUG(
          "[downgrade] [{}] in-place compression declined: {} candidates predict only {} of {} "
          "bytes; spilling instead",
          _source_label,
          picks.size(),
          predicted_total,
          requested);
      }
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

        // Spill uncompressed batches first; already-compressed ones last.
        //
        // A device-compressed batch has been downgraded once already. It is small,
        // so evicting it frees less than an uncompressed batch of the same logical
        // size, and it cost GPU time to produce which spilling discards. Preferring
        // the uncompressed candidates therefore frees more memory per unit of work
        // and keeps the compression already paid for.
        //
        // stable_partition, not sort: the existing order is meaningful (repos are
        // visited in ascending {operator_id, port_id} and batches back-to-front),
        // and this must reorder only across the compressed/uncompressed boundary.
        if (compression::device_compression_downgrade_enabled()) {
          std::stable_partition(
            candidates.begin(), candidates.end(), [](const std::unique_ptr<convertible_data>& c) {
              return c && !c->is_device_compressed();
            });
        }

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

  // Periodic tier-occupancy sample. The eviction logs report how many bytes a
  // downgrade moved, but not the level the tier was at when it moved them, so a
  // tier that fills and drains within a query is indistinguishable from one that
  // never gives memory back. Sampled on a wall-clock interval rather than per
  // cycle because the monitor runs ~100x/s.
  auto last_occupancy_log      = std::chrono::steady_clock::now();
  constexpr auto kOccupancyLog = std::chrono::seconds(1);

  // Cycles observing pressure, split by whether the monitor could act on it. Only
  // one monitor request may be outstanding at a time, and the flag clears when
  // that request completes, so a slow request leaves the monitor unable to
  // respond to pressure that develops while it runs. The suppressed count says
  // how much of an episode was spent in that state.
  std::uint64_t pressure_cycles   = 0;
  std::uint64_t suppressed_cycles = 0;

  while (_running.load()) {
    const bool pressure  = _memory_space && _memory_space->should_downgrade_memory();
    const bool in_flight = _monitor_request_enqueued.load(std::memory_order_relaxed);
    if (pressure) {
      ++pressure_cycles;
      if (in_flight) { ++suppressed_cycles; }
    }

    if (_memory_space) {
      const auto now = std::chrono::steady_clock::now();
      if (now - last_occupancy_log >= kOccupancyLog) {
        last_occupancy_log     = now;
        const std::size_t cap  = _memory_space->get_max_memory();
        const std::size_t free = _memory_space->get_available_memory();
        const std::size_t used = cap > free ? cap - free : 0;
        SIRIUS_LOG_DEBUG(
          "[occupancy] [{}] used={}B free={}B capacity={}B ({:.1f}%) | pressure_cycles={} "
          "suppressed={} ({:.0f}%)",
          _source_label,
          used,
          free,
          cap,
          cap > 0 ? 100.0 * static_cast<double>(used) / static_cast<double>(cap) : 0.0,
          pressure_cycles,
          suppressed_cycles,
          pressure_cycles > 0 ? 100.0 * static_cast<double>(suppressed_cycles) /
                                  static_cast<double>(pressure_cycles)
                              : 0.0);
        // Same cadence as occupancy so the two can be read together: what the
        // encode is asking the allocator for, against how much room there was.
        if (compression::alloc_stats_enabled()) {
          SIRIUS_LOG_DEBUG("[compression_alloc] [{}] {}",
                           _source_label,
                           compression::alloc_stats_format());
        }
      }
    }

    if (pressure && !in_flight) {
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
          _monitor_requests_issued.fetch_add(1, std::memory_order_relaxed);
          _monitor_request_enqueued.store(true, std::memory_order_relaxed);
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
      // Release the OOM policy's compression suppression here rather than on a
      // successful retry: should_downgrade_memory() is debounced by the
      // trigger/stop fractions, so it reports recovery once the space has real
      // headroom again instead of after a single allocation squeaks through.
      if (compression::spill_compression_suppressed()) {
        compression::set_spill_compression_suppressed(false);
        SIRIUS_LOG_DEBUG("[downgrade] [{}] memory pressure resolved; re-enabling spill compression",
                         _source_label);
      }
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
  // The in-place compression pass needs the target as a number, not just as a
  // predicate: it has to price a whole candidate set against it before it will
  // compress any of it.
  req->requested_bytes = bytes;
  auto future          = req->result.get_future();
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
