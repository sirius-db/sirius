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
  // A monitor request eaten by a previous stop() must not keep the monitor dead across a
  // restart. cancel_pending_requests() re-arms too; this is the belt to that suspender.
  _monitor_request_enqueued.store(false, std::memory_order_relaxed);

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
  // The _running CAS is the lifecycle serialization: exactly one caller wins the
  // true->false transition and tears down; a concurrent second stop() returns immediately
  // (as it always has). The mutex that used to sit here only ordered stop() against the
  // deleted global drain()'s stop-join-restart.
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

void downgrade_executor::drain(sirius::query_id_t query_id)
{
  // Fail only THIS query's queued promises. Its waiters (GPU manager threads blocked in
  // request_downgrade(...).get() on the query's behalf) unblock into the caller's
  // cancelled-downgrade handling; every other query's requests stay queued and the
  // processing/monitor threads keep running.
  cancel_pending_requests_for_query(query_id);

  // Wait out THIS query's in-flight requests (plural — requests process concurrently). Each
  // fulfils its promise before its entry clears, so returning implies this query has no
  // waiter left inside this executor. Bounded: the caller has quiesced the query, so no new
  // requests of this query can be published.
  //
  // A request popped but not yet published can slip past this wait. That is benign: the
  // caller has already quiesced the query (no new waiters can appear), the slipped request
  // still fulfils its promise normally, and repository teardown needs no fence at all — its
  // sweep co-owns whatever it borrows (step 6).
  std::unique_lock<std::mutex> lock(_in_flight_mutex);
  _in_flight_cv.wait(lock, [&] {
    for (auto const& [seq, query] : _in_flight_requests) {
      if (query == query_id) { return false; }
    }
    return true;
  });
}

void downgrade_executor::wait_inflight_request()
{
  // See the header: an in-flight request's TIER-2 sweep may hold ANOTHER query's task in a
  // convertible wrapper; the caller must not destroy that query's plan until the request (and
  // with it every wrapper it created) has completed. BARRIER semantics: wait only for the
  // requests published before entry — waiting for idleness would never return under a steady
  // stream of monitor/peer requests, and later requests cannot extract the caller's tasks
  // (TIER-2 extraction consults the lifecycle gate; the caller has already quiesced).
  std::unique_lock<std::mutex> lock(_in_flight_mutex);
  const std::uint64_t barrier = _next_request_seq;
  _in_flight_cv.wait(lock, [&] {
    return _in_flight_requests.empty() || _in_flight_requests.begin()->first >= barrier;
  });
}

/// Per-request processing state, shared by the processing thread and the request's dispatched
/// conversion workers. The processing thread holds one "dispatch token" while it collects and
/// dispatches candidates; every dispatched worker holds one more. Whoever drops the count to
/// zero runs complete_request() — that is what lets the processing thread move on to the NEXT
/// request while this one's conversions are still running (F8).
struct downgrade_executor::request_context {
  request_context(downgrade_executor& owner, std::unique_ptr<downgrade_request> request)
    : self(owner), req(std::move(request))
  {
  }

  downgrade_executor& self;
  std::unique_ptr<downgrade_request> req;
  /// Key of this request's entry in the executor's in-flight map.
  std::uint64_t seq{0};

  struct running_stats {
    std::atomic<size_t> batches{0};
    std::atomic<size_t> bytes{0};
  };
  // Per-source tracking (repos vs pipeline_queue) and per-target-tier tracking (host vs
  // disk). Owned here — not on the loop's stack — because workers update them after the
  // processing thread has moved on.
  running_stats repo_stats;
  running_stats pipeline_queue_stats;
  running_stats host_target_stats;
  running_stats disk_target_stats;

  /// Candidate conversion targets, in preference order; referenced by every worker.
  std::vector<const cucascade::memory::memory_space*> target_spaces;
  size_t host_end_idx{0};
  bool disk_not_configured{false};
  std::chrono::steady_clock::time_point t_start{};

  std::atomic<size_t> outstanding{1};

  void add_worker() { outstanding.fetch_add(1, std::memory_order_relaxed); }
  void release()
  {
    const size_t remaining = outstanding.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (remaining == 0) { self.complete_request(*this); }
    outstanding.notify_all();
  }

  /// Processing-thread only: block until every dispatched worker of THIS request has finished
  /// (outstanding back to the loop's own token). Used between the TIER-2 victim-preference
  /// passes — "own-query victims as the last resort" is only knowable once the peers-first
  /// pass's conversions have actually landed, because `satisfied` lags the dispatches.
  /// Bounded: waits only for this request's workers, never the pool or other requests.
  void wait_for_dispatched_workers()
  {
    for (auto value = outstanding.load(std::memory_order_acquire); value > 1;
         value      = outstanding.load(std::memory_order_acquire)) {
      outstanding.wait(value, std::memory_order_acquire);
    }
  }
};

void downgrade_executor::complete_request(request_context& ctx)
{
  auto& req = *ctx.req;

  // Monitor re-arm (D6): the completion side owns clearing the flag for requests the loop
  // consumed; fail_request() owns it for requests destroyed unprocessed.
  if (req.is_monitor_request) { _monitor_request_enqueued.store(false, std::memory_order_relaxed); }

  // Monitor requests are gated by has_viable_downgrade_target() and warn once per stall
  // episode in monitor_loop(); only warn here for one-shot (external) requests to avoid spam.
  if (ctx.disk_not_configured && !req.satisfied.load() && !req.is_monitor_request) {
    SIRIUS_LOG_WARN(
      "[downgrade] [{}] downgrade request not satisfied and disk memory space is not configured; "
      "data cannot be spilled to disk. Consider configuring a disk memory space to enable "
      "spilling.",
      _source_label);
  }

  auto total_bytes   = req.bytes_freed.load(std::memory_order_relaxed);
  auto total_batches = req.batches_downgraded.load(std::memory_order_relaxed);
  auto duration_ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ctx.t_start)
      .count();
  double throughput_mbs =
    (duration_ms > 0.0) ? (total_bytes / (1024.0 * 1024.0)) / (duration_ms / 1000.0) : 0.0;
  std::string request_label = req.is_monitor_request ? "monitor " : "";

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
    ctx.repo_stats.batches.load(std::memory_order_relaxed),
    ctx.repo_stats.bytes.load(std::memory_order_relaxed),
    ctx.pipeline_queue_stats.batches.load(std::memory_order_relaxed),
    ctx.pipeline_queue_stats.bytes.load(std::memory_order_relaxed),
    ctx.host_target_stats.batches.load(std::memory_order_relaxed),
    ctx.host_target_stats.bytes.load(std::memory_order_relaxed),
    ctx.disk_target_stats.batches.load(std::memory_order_relaxed),
    ctx.disk_target_stats.bytes.load(std::memory_order_relaxed));

  // Fulfill the promise BEFORE clearing the in-flight entry: drain(query_id) returning must
  // imply this query's waiters have unblocked.
  req.result.set_value(total_bytes);

  {
    std::lock_guard<std::mutex> in_flight_lock(_in_flight_mutex);
    _in_flight_requests.erase(ctx.seq);
  }
  _in_flight_cv.notify_all();
}

void downgrade_executor::processing_loop()
{
  while (_running.load()) {
    auto request = _request_queue.pop();
    if (!request) break;  // interrupted

    auto ctx = std::make_shared<request_context>(*this, std::move(request));

    // Publish {seq -> query} before doing any work; complete_request() erases the entry after
    // the promise is fulfilled. drain(query_id)/wait_inflight_request() wait on these entries.
    // (A request popped but not yet published can slip past those waits — benign, closed from
    // the other side by the TIER-2 extraction gate; see the header.)
    {
      std::lock_guard<std::mutex> in_flight_lock(_in_flight_mutex);
      ctx->seq = _next_request_seq++;
      _in_flight_requests.emplace(ctx->seq, ctx->req->query_id);
    }

    ctx->t_start = std::chrono::steady_clock::now();

    // Resolve the source memory space for filtering candidates
    auto* source_space = _reservation_manager.get_memory_space(_space_id.tier, _space_id.device_id);

    // Build target spaces list: for GPU->HOST downgrade, target is HOST tier followed by DISK tier.
    // NUMA preference (from downgrade_executor_config, v1.0 dd86dd0 intent re-authored on the
    // post-#637 architecture): if preferred_numa_node is set, stable_partition the matching HOST
    // space(s) to the front of target_spaces so cand->convert() tries the NUMA-local space first.
    // Owned by the context: this request's workers read it after the loop moves on.
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
        ctx->target_spaces.push_back(hs);
      }
    }
    ctx->host_end_idx = ctx->target_spaces.size();
    auto disk_spaces =
      _reservation_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::DISK);
    ctx->disk_not_configured = disk_spaces.empty();
    for (auto* ds : disk_spaces) {
      ctx->target_spaces.push_back(ds);
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
    // re-scanned before leaving idle. The sweep OWNS everything it borrows: managers and
    // repositories are held by shared_ptr for the duration of the sweep (and batches by
    // shared_ptr inside each candidate), so a query ending concurrently — its cleanup no
    // longer quiesces this executor — cannot pull any of them out from under this loop; its
    // erase() just drops the registry's map entry and the last holder does the destruction.
    // (This ownership is what retired the registry's interim sweep gate.)
    bool pool_interrupted = false;
    auto const managers   = _data_repo_registry.get_all();
    for (auto const& manager : std::views::reverse(managers)) {
      if (ctx->req->satisfied.load() || pool_interrupted) break;
      auto repos = manager->get_repositories();
      for (auto const& repo : repos) {
        if (ctx->req->satisfied.load()) break;

        convertible_data_batch_provider provider(repo);
        auto candidates = provider.get_all_convertible(
          source_space, /*front_to_back=*/false, /*ignore_subscribed=*/true);
        for (auto& candidate : candidates) {
          if (ctx->req->satisfied.load()) break;

          auto candidate_bytes = candidate->bytes_in_space(source_space);

          auto slot = _pool->reserve();
          if (!slot) {
            pool_interrupted = true;
            break;
          }

          // Re-check after reserve() returns -- the previous candidate's worker may
          // have set satisfied while we were blocked waiting for a thread slot.
          if (ctx->req->satisfied.load()) break;

          auto exc_stream = _stream_pool->acquire_stream(
            cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);

          ctx->add_worker();
          _pool->dispatch(
            std::move(slot),
            [cand = std::move(candidate),
             ctx,
             exc_stream = std::move(exc_stream),
             candidate_bytes]() mutable {
              try {
                auto* req_ptr = ctx->req.get();
                auto result   = cand->convert(
                  ctx->target_spaces, exc_stream, ctx->self._reservation_manager, false);
                if (result) {
                  req_ptr->bytes_freed.fetch_add(candidate_bytes, std::memory_order_relaxed);
                  req_ptr->batches_downgraded.fetch_add(1, std::memory_order_relaxed);
                  ctx->repo_stats.batches.fetch_add(1, std::memory_order_relaxed);
                  ctx->repo_stats.bytes.fetch_add(candidate_bytes, std::memory_order_relaxed);
                  for (size_t i = 0; i < result->size(); ++i) {
                    if ((*result)[i] == 0) continue;
                    if (i < ctx->host_end_idx) {
                      ctx->host_target_stats.batches.fetch_add(1, std::memory_order_relaxed);
                      ctx->host_target_stats.bytes.fetch_add((*result)[i],
                                                             std::memory_order_relaxed);
                    } else {
                      ctx->disk_target_stats.batches.fetch_add(1, std::memory_order_relaxed);
                      ctx->disk_target_stats.bytes.fetch_add((*result)[i],
                                                             std::memory_order_relaxed);
                    }
                  }
                  if (req_ptr->predicate && req_ptr->predicate()) {
                    req_ptr->satisfied.store(true);
                  }
                }
              } catch (const std::exception& e) {
                SIRIUS_LOG_ERROR("[downgrade] convert failed from data repository: {}", e.what());
              }
              // The candidate wrapper is destroyed before the token drops, so a request's
              // completion implies every wrapper it created is gone (the plan-lifetime fence
              // relies on this ordering).
              cand.reset();
              ctx->release();
            });
        }
        if (pool_interrupted) break;
      }
    }

    // === TIER 2: task_scheduler task queue ===
    // Victim preference (F8): first take tasks that do NOT belong to the requesting query —
    // extracting the requester's own queued work to satisfy its request is self-defeating —
    // then, only if still unsatisfied, take own-query victims as the last resort.
    // Unattributed requests (query 0: the monitor's, external byte targets) own nothing, so
    // one unfiltered pass suffices.
    if (!ctx->req->satisfied.load() && _pipeline_task_queue) {
      convertible_gpu_pipeline_task_provider pipeline_provider(*_pipeline_task_queue,
                                                               _query_lifecycle);
      const bool attributed = sirius::value_of(ctx->req->query_id) != 0;
      if (attributed) {
        run_tier2_pass(ctx, pipeline_provider, source_space, ctx->req->query_id);
        // Let the peers-first pass's conversions LAND before deciding that the requester's
        // own queued work is truly the last resort — `satisfied` lags the dispatches, and
        // cannibalizing the requester because peers' results were still in flight would
        // defeat the preference in the common case. Bounded: this request's workers only.
        ctx->wait_for_dispatched_workers();
      }
      if (!ctx->req->satisfied.load()) {
        run_tier2_pass(ctx, pipeline_provider, source_space, std::nullopt);
      }
    }

    // Dispatch phase over: drop the loop's token. If no worker is still out, this completes
    // the request right here; otherwise the LAST worker completes it — and either way the
    // loop is already free to pop the next request, so requests process concurrently,
    // bounded by the pool's capacity (F8).
    ctx->release();
  }
}

void downgrade_executor::run_tier2_pass(const std::shared_ptr<request_context>& ctx,
                                        sirius::convertible_gpu_pipeline_task_provider& provider,
                                        cucascade::memory::memory_space* source_space,
                                        std::optional<sirius::query_id_t> exclude_query)
{
  // Budget: the queue's size at pass start — the candidates this pass could actually take.
  // Each wrapper the provider hands out consults the lifecycle gate before its RAII re-push,
  // so a task extracted here cannot land back in the queue after its query's drain.
  size_t max_tasks_to_convert = _pipeline_task_queue->size();
  size_t tasks_converted      = 0;
  while (!ctx->req->satisfied.load() && tasks_converted < max_tasks_to_convert) {
    auto candidate =
      provider.get_next_convertible(source_space, /*front_to_back=*/false, exclude_query);
    if (!candidate) break;
    tasks_converted++;

    auto candidate_bytes = candidate->bytes_in_space(source_space);

    auto slot = _pool->reserve();
    if (!slot) break;  // interrupted

    if (ctx->req->satisfied.load()) break;

    auto exc_stream = _stream_pool->acquire_stream(
      cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);

    ctx->add_worker();
    _pool->dispatch(
      std::move(slot),
      [cand = std::move(candidate),
       ctx,
       exc_stream = std::move(exc_stream),
       candidate_bytes]() mutable {
        try {
          auto* req_ptr = ctx->req.get();
          auto result =
            cand->convert(ctx->target_spaces, exc_stream, ctx->self._reservation_manager, false);
          if (result) {
            req_ptr->bytes_freed.fetch_add(candidate_bytes, std::memory_order_relaxed);
            req_ptr->batches_downgraded.fetch_add(1, std::memory_order_relaxed);
            ctx->pipeline_queue_stats.batches.fetch_add(1, std::memory_order_relaxed);
            ctx->pipeline_queue_stats.bytes.fetch_add(candidate_bytes, std::memory_order_relaxed);
            for (size_t i = 0; i < result->size(); ++i) {
              if ((*result)[i] == 0) continue;
              if (i < ctx->host_end_idx) {
                ctx->host_target_stats.batches.fetch_add(1, std::memory_order_relaxed);
                ctx->host_target_stats.bytes.fetch_add((*result)[i], std::memory_order_relaxed);
              } else {
                ctx->disk_target_stats.batches.fetch_add(1, std::memory_order_relaxed);
                ctx->disk_target_stats.bytes.fetch_add((*result)[i], std::memory_order_relaxed);
              }
            }
            if (req_ptr->predicate && req_ptr->predicate()) { req_ptr->satisfied.store(true); }
          }
        } catch (const std::exception& e) {
          SIRIUS_LOG_ERROR("[downgrade] convert failed from task queue: {}", e.what());
        }
        // Wrapper destroyed (task re-pushed or gate-dropped) BEFORE the token drops — a
        // request's completion must imply its wrappers are gone (plan-lifetime fence).
        cand.reset();
        ctx->release();
      });
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
          _monitor_requests_issued.fetch_add(1, std::memory_order_relaxed);
          _monitor_request_enqueued.store(true, std::memory_order_relaxed);
          // Fire-and-forget: monitor does not wait for the result. A refused push (queue
          // interrupted by a racing stop()) must re-arm immediately — otherwise the
          // flag latches true with no request in flight and pressure-driven downgrade for
          // this space is dead for the rest of the process.
          if (!_request_queue.push(std::move(req))) {
            _monitor_request_enqueued.store(false, std::memory_order_relaxed);
          }
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
    fail_request(std::move(req));
  }
}

void downgrade_executor::cancel_pending_requests_for_query(sirius::query_id_t query_id)
{
  // Selective cancel: pop everything reachable, fail the ending query's requests, put the
  // rest back. The processing thread may pop concurrently; anything it takes is covered by
  // drain(query_id)'s in-flight wait. Bounded even against concurrent producers: the only
  // repeating unattributed producer (the monitor) keeps at most one request outstanding.
  std::vector<std::unique_ptr<downgrade_request>> kept;
  while (auto req = _request_queue.try_pop()) {
    if (req->query_id == query_id) {
      fail_request(std::move(req));
    } else {
      kept.push_back(std::move(req));
    }
  }
  for (auto& req : kept) {
    if (!_request_queue.is_open()) {
      // Racing stop(): a push would be refused and silently destroy the
      // promise. Fail it loudly instead — the waiter unblocks and, for a monitor request,
      // the re-arm flag is reset. This is exactly what the racing global cancel would do.
      fail_request(std::move(req));
      continue;
    }
    // A refusal here means the interrupt landed between the check above and the push; the
    // destroyed promise still unblocks its waiter via std::future_error(broken_promise).
    (void)_request_queue.push(std::move(req));
  }
}

void downgrade_executor::fail_request(std::unique_ptr<downgrade_request> request)
{
  // This request dies without reaching the processing loop — the only other place that
  // clears _monitor_request_enqueued. Re-arm here or the monitor latches enqueued-forever
  // and memory-pressure downgrade for this space is dead for the rest of the process.
  if (request->is_monitor_request) {
    _monitor_request_enqueued.store(false, std::memory_order_relaxed);
  }
  try {
    request->result.set_exception(
      std::make_exception_ptr(std::runtime_error("downgrade request cancelled")));
  } catch (...) {
    // Promise may already be fulfilled — safe to ignore
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
  return request_downgrade(sirius::make_query_id(0), std::move(predicate));
}

std::future<size_t> downgrade_executor::request_downgrade(sirius::query_id_t query_id,
                                                          std::function<bool()> predicate)
{
  auto req       = std::make_unique<downgrade_request>();
  req->query_id  = query_id;
  req->predicate = std::move(predicate);
  auto future    = req->result.get_future();
  if (!_request_queue.push(std::move(req))) {
    SIRIUS_LOG_WARN("[downgrade] request_downgrade: queue inactive, dropping request for query {}",
                    query_id);
    return future;
  }
  return future;
}

}  // namespace parallel
}  // namespace sirius
