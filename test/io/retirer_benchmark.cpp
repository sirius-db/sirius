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

// retirer_benchmark — stream_ordered_retirer vs. event + retirement thread pool.
//
// The question: the retirer has no thread of its own.  State only advances when
// somebody calls drain(), which happens on the submitter as it comes back round
// for the next batch.  Does dropping the retirement thread slow down how fast
// chunks reach `cached`?
//
// Both arms run the IDENTICAL workload and the IDENTICAL retirement function.
// The only difference is how that function gets to run:
//
//   retirer   lane.begin() -> 8 memcpys -> on_retire(fn) -> commit(), and the
//             submitter calls drain_all() between batches.  No extra threads.
//
//   events    the same 8 memcpys -> cudaEventRecord -> push (event, fn) onto a
//             queue served by 2 threads, each doing cudaEventSynchronize then
//             running fn.  One shared queue, so a batch is retired by whichever
//             worker is free -- the arrangement the retirer replaced.
//
// Workload: 4 streams, one submitter thread per stream (the retirer's intended
// topology, and the only one where its submit lock is uncontended).  1000 x
// 1 MiB device chunks, filled in batches of 8 from pinned host blocks, so 125
// batches round-robined across the streams.  Retirement flips all 8 chunks from
// `loading` to `cached` and hands the pinned blocks back -- the real cache
// state advance, not an empty lambda.
//
// Two staging configurations, because they answer different questions:
//
//   full      1 GiB of pinned staging: every batch has a free block waiting, so
//             nobody ever blocks on retirement.  Measures pure per-batch
//             overhead.
//
//   bounded   64 MiB of pinned staging, recycled by retirement.  The submitter
//             cannot start a batch until earlier ones have retired, which puts
//             retirement on the critical path -- what the real cache does, and
//             the only configuration where a missing retirement thread can
//             actually cost anything.
//
// Reported per arm: wall time and H2D bandwidth, time for the chunk population
// to reach 50/90/100% cached, retirement lag, and how long submitters spent
// waiting for a staging block.
//
// NOTE on retirement lag: it is measured from commit() to the retire fn
// running, so it includes the copy itself and is NOT a latency figure in its
// own right.  Both arms copy the same bytes on the same streams, so only the
// DIFFERENCE between the arms is meaningful.

#include "exec/stream_ordered_retirer.hpp"

#include <cuda_runtime.h>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;
using seconds_d  = std::chrono::duration<double>;

constexpr std::size_t BLOCK_BYTES    = std::size_t{1} << 20;  // 1 MiB
constexpr std::size_t N_CHUNKS       = 1000;
constexpr std::size_t BATCH_CHUNKS   = 8;
constexpr std::size_t MAX_STREAMS    = 8;
constexpr std::size_t N_BATCHES      = N_CHUNKS / BATCH_CHUNKS;  // 125
constexpr std::size_t RETIRE_THREADS = 2;

constexpr std::size_t FULL_STAGING_BLOCKS    = N_CHUNKS;  // 1 GiB: never blocks
constexpr std::size_t BOUNDED_STAGING_BLOCKS = 64;        // 64 MiB: recycled

static_assert(N_CHUNKS % BATCH_CHUNKS == 0, "batches must divide the chunk count");
static_assert(BOUNDED_STAGING_BLOCKS >= MAX_STREAMS * BATCH_CHUNKS,
              "each stream must be able to hold one batch, or the run deadlocks");

/// Streams (and submitter threads) this run uses. Settable so the H2D copy
/// engine's actual concurrency can be measured rather than assumed: on a GPU
/// with one H2D engine, extra streams buy no bandwidth and the run is
/// link-bound no matter what the retirement side does.
std::size_t g_n_streams = 4;

// ---------------------------------------------------------------------------
// chunk state — what retirement advances
// ---------------------------------------------------------------------------

enum class chunk_state : int { idle = 0, loading = 1, cached = 2, failed = 3 };

struct chunk_table {
  explicit chunk_table(std::size_t n) : states(n) {}

  void reset()
  {
    for (auto& s : states) {
      s.store(chunk_state::idle, std::memory_order_relaxed);
    }
    n_cached.store(0, std::memory_order_release);
  }

  std::vector<std::atomic<chunk_state>> states;
  std::atomic<std::size_t> n_cached{0};
};

// ---------------------------------------------------------------------------
// staging pool — pinned blocks, returned by retirement
// ---------------------------------------------------------------------------

class block_pool {
 public:
  explicit block_pool(std::vector<std::byte*> blocks) : _free(blocks.begin(), blocks.end()) {}

  /// Non-blocking. Falsy when empty, which is what retirer::acquire() retries on.
  std::byte* try_pop() noexcept
  {
    std::lock_guard g(_m);
    if (_free.empty()) { return nullptr; }
    auto* b = _free.back();
    _free.pop_back();
    return b;
  }

  /// Blocking, for the arm that has retirement threads to wait on.
  std::byte* pop_blocking()
  {
    std::unique_lock lk(_m);
    _cv.wait(lk, [this] { return !_free.empty(); });
    auto* b = _free.back();
    _free.pop_back();
    return b;
  }

  void push(std::byte* b)
  {
    {
      std::lock_guard g(_m);
      _free.push_back(b);
    }
    _cv.notify_one();
  }

 private:
  std::mutex _m;
  std::condition_variable _cv;
  std::vector<std::byte*> _free;
};

// ---------------------------------------------------------------------------
// per-run measurements
// ---------------------------------------------------------------------------

struct run_stats {
  double wall_s{0};
  /// commit() -> retire fn ran, one per batch. Includes the copy; see header.
  std::vector<double> retire_lag_ms;
  /// Total time submitters spent waiting for a staging block, summed over threads.
  double staging_wait_s{0};
  /// Fraction-cached timeline, sampled on a 1 ms tick.
  double t50_s{0};
  double t90_s{0};
  double t100_s{0};
  bool all_cached{false};
};

double percentile(std::vector<double> v, double p)
{
  if (v.empty()) { return 0.0; }
  std::sort(v.begin(), v.end());
  const auto idx = static_cast<std::size_t>(p * static_cast<double>(v.size() - 1));
  return v[idx];
}

double median_of(std::vector<double> v) { return percentile(std::move(v), 0.5); }

/// Samples the cached count on a 1 ms tick so the two arms can be compared on
/// how fast state actually advances, not just on when the last batch lands.
class progress_sampler {
 public:
  progress_sampler(chunk_table& chunks, clock_type::time_point start)
    : _chunks(chunks), _start(start)
  {
    _thread = std::thread([this] {
      while (!_stop.load(std::memory_order_acquire)) {
        _samples.push_back({clock_type::now(), _chunks.n_cached.load(std::memory_order_acquire)});
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
      }
      _samples.push_back({clock_type::now(), _chunks.n_cached.load(std::memory_order_acquire)});
    });
  }

  ~progress_sampler()
  {
    _stop.store(true, std::memory_order_release);
    if (_thread.joinable()) { _thread.join(); }
  }

  progress_sampler(progress_sampler const&)            = delete;
  progress_sampler& operator=(progress_sampler const&) = delete;

  /// First sample time at which at least @p fraction of chunks were cached.
  [[nodiscard]] double time_to(double fraction) const
  {
    const auto target = static_cast<std::size_t>(fraction * static_cast<double>(N_CHUNKS));
    for (auto const& s : _samples) {
      if (s.n >= target) { return seconds_d{s.at - _start}.count(); }
    }
    return -1.0;
  }

  void stop()
  {
    _stop.store(true, std::memory_order_release);
    if (_thread.joinable()) { _thread.join(); }
  }

 private:
  struct sample {
    clock_type::time_point at;
    std::size_t n;
  };

  chunk_table& _chunks;
  clock_type::time_point _start;
  std::atomic<bool> _stop{false};
  std::vector<sample> _samples;
  std::thread _thread;
};

// ---------------------------------------------------------------------------
// shared fixture: streams, device chunks, pinned blocks
// ---------------------------------------------------------------------------

#define CUDA_OK(expr)                                                                  \
  do {                                                                                 \
    const cudaError_t _e = (expr);                                                     \
    if (_e != cudaSuccess) {                                                           \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
      std::abort();                                                                    \
    }                                                                                  \
  } while (0)

struct fixture {
  fixture()
  {
    for (auto& s : streams) {
      CUDA_OK(cudaStreamCreate(&s));
    }
    for (auto& d : device_chunks) {
      CUDA_OK(cudaMalloc(&d, BLOCK_BYTES));
    }

    // Pinned staging, from the same pool the cache bounces through.
    constexpr std::size_t blocks_per_slab = 128;
    constexpr std::size_t slabs    = (FULL_STAGING_BLOCKS + blocks_per_slab - 1) / blocks_per_slab;
    constexpr std::size_t capacity = slabs * blocks_per_slab * BLOCK_BYTES;

    upstream =
      std::make_unique<cucascade::memory::numa_region_pinned_host_memory_resource>(0, true);
    pinned_mr = std::make_unique<cucascade::memory::fixed_size_host_memory_resource>(
      0, *upstream, capacity, capacity, BLOCK_BYTES, blocks_per_slab, slabs);

    // One multi-block allocation, held for the program's lifetime; the pool
    // hands out whole blocks, so this is exactly FULL_STAGING_BLOCKS of them.
    staging_alloc = pinned_mr->allocate_multiple_blocks(FULL_STAGING_BLOCKS * BLOCK_BYTES);
    auto blocks   = staging_alloc->get_blocks();
    pinned.assign(blocks.begin(), blocks.end());
    for (std::size_t i = 0; i < pinned.size(); ++i) {
      // Fault the pages in once, so no run pays for first touch.
      std::memset(pinned[i], static_cast<int>(i & 0xff), BLOCK_BYTES);
    }
  }

  ~fixture()
  {
    for (auto* s : streams) {
      cudaStreamSynchronize(s);
      cudaStreamDestroy(s);
    }
    for (auto* d : device_chunks) {
      cudaFree(d);
    }
  }

  fixture(fixture const&)            = delete;
  fixture& operator=(fixture const&) = delete;

  /// The staging blocks visible to a run, so `bounded` can hand out fewer.
  [[nodiscard]] std::vector<std::byte*> staging(std::size_t n) const
  {
    return {pinned.begin(), pinned.begin() + static_cast<std::ptrdiff_t>(n)};
  }

  std::array<cudaStream_t, MAX_STREAMS> streams{};
  std::array<void*, N_CHUNKS> device_chunks{};
  std::unique_ptr<cucascade::memory::numa_region_pinned_host_memory_resource> upstream;
  std::unique_ptr<cucascade::memory::fixed_size_host_memory_resource> pinned_mr;
  cucascade::memory::fixed_multiple_blocks_allocation staging_alloc;
  std::vector<std::byte*> pinned;
};

/// The batches one stream owns, round-robined so every stream stays busy.
std::vector<std::size_t> batches_for(std::size_t stream_idx)
{
  std::vector<std::size_t> out;
  for (std::size_t b = stream_idx; b < N_BATCHES; b += g_n_streams) {
    out.push_back(b);
  }
  return out;
}

/// Advance the 8 chunks of `batch` and hand their staging blocks back. This is
/// the retirement work; it is byte-identical between the two arms.
void retire_batch(chunk_table& chunks,
                  block_pool& pool,
                  std::size_t batch,
                  std::vector<std::byte*> const& blocks,
                  cudaError_t status) noexcept
{
  const auto state = status == cudaSuccess ? chunk_state::cached : chunk_state::failed;
  for (std::size_t i = 0; i < BATCH_CHUNKS; ++i) {
    chunks.states[batch * BATCH_CHUNKS + i].store(state, std::memory_order_release);
  }
  for (auto* b : blocks) {
    pool.push(b);
  }
  chunks.n_cached.fetch_add(BATCH_CHUNKS, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// arm A — stream_ordered_retirer
// ---------------------------------------------------------------------------

run_stats run_retirer(fixture& fx, chunk_table& chunks, std::size_t staging_blocks)
{
  chunks.reset();
  block_pool pool{fx.staging(staging_blocks)};
  sirius::exec::stream_ordered_retirer retirer;

  std::vector<double> lag_ms(N_BATCHES, 0.0);
  std::vector<clock_type::time_point> committed_at(N_BATCHES);
  std::atomic<double> staging_wait_s{0.0};

  const auto start = clock_type::now();
  progress_sampler sampler{chunks, start};

  std::vector<std::thread> submitters;
  submitters.reserve(g_n_streams);
  for (std::size_t s = 0; s < g_n_streams; ++s) {
    submitters.emplace_back([&, s] {
      auto& lane    = retirer.lane_for(fx.streams[s]);
      double waited = 0.0;

      for (std::size_t batch : batches_for(s)) {
        std::vector<std::byte*> blocks;
        blocks.reserve(BATCH_CHUNKS);
        const auto wait_start = clock_type::now();
        for (std::size_t i = 0; i < BATCH_CHUNKS; ++i) {
          // Drain-on-demand: the allocation path is what makes state advance.
          blocks.push_back(retirer.acquire([&pool] { return pool.try_pop(); }));
        }
        waited += seconds_d{clock_type::now() - wait_start}.count();

        for (std::size_t i = 0; i < BATCH_CHUNKS; ++i) {
          chunks.states[batch * BATCH_CHUNKS + i].store(chunk_state::loading,
                                                        std::memory_order_release);
        }

        {
          auto sub = lane.begin();
          for (std::size_t i = 0; i < BATCH_CHUNKS; ++i) {
            CUDA_OK(cudaMemcpyAsync(fx.device_chunks[batch * BATCH_CHUNKS + i],
                                    blocks[i],
                                    BLOCK_BYTES,
                                    cudaMemcpyHostToDevice,
                                    sub.stream()));
          }
          sub.on_retire([&chunks, &pool, &lag_ms, &committed_at, batch, blocks = std::move(blocks)](
                          cudaError_t status) mutable noexcept {
            lag_ms[batch] = seconds_d{clock_type::now() - committed_at[batch]}.count() * 1e3;
            retire_batch(chunks, pool, batch, blocks, status);
          });
          committed_at[batch] = clock_type::now();
          CUDA_OK(sub.commit());
        }

        // Nothing polls in steady state; this is the drain that advances state.
        retirer.drain_all();
      }

      // Whatever is still outstanding on this lane.
      while (!lane.idle()) {
        if (lane.drain() == 0) { std::this_thread::yield(); }
      }

      double expected = staging_wait_s.load(std::memory_order_relaxed);
      while (!staging_wait_s.compare_exchange_weak(expected, expected + waited)) {}
    });
  }
  for (auto& t : submitters) {
    t.join();
  }
  retirer.quiesce();

  const auto wall = seconds_d{clock_type::now() - start}.count();
  sampler.stop();

  run_stats st;
  st.wall_s         = wall;
  st.retire_lag_ms  = std::move(lag_ms);
  st.staging_wait_s = staging_wait_s.load();
  st.t50_s          = sampler.time_to(0.5);
  st.t90_s          = sampler.time_to(0.9);
  st.t100_s         = sampler.time_to(1.0);
  st.all_cached     = chunks.n_cached.load() == N_CHUNKS;
  return st;
}

// ---------------------------------------------------------------------------
// control — copies only, no retirement of any kind
// ---------------------------------------------------------------------------
//
// The DMA floor.  Without it the two arms are only comparable to each other,
// and a cost they BOTH pay is invisible -- which matters here because
// cudaStreamAddCallback blocks later work in its stream until the callback
// returns, so the retirer's per-batch callback is a potential stream stall that
// an arm-vs-arm comparison can never surface.
//
// Runs with full staging and no state advance, so nothing throttles the
// submitters: whatever this costs is the copies and nothing else.
run_stats run_no_retirement(fixture& fx, chunk_table& chunks)
{
  chunks.reset();
  block_pool pool{fx.staging(FULL_STAGING_BLOCKS)};

  const auto start = clock_type::now();

  std::vector<std::thread> submitters;
  submitters.reserve(g_n_streams);
  for (std::size_t s = 0; s < g_n_streams; ++s) {
    submitters.emplace_back([&, s] {
      for (std::size_t batch : batches_for(s)) {
        for (std::size_t i = 0; i < BATCH_CHUNKS; ++i) {
          auto* block = pool.try_pop();
          CUDA_OK(cudaMemcpyAsync(fx.device_chunks[batch * BATCH_CHUNKS + i],
                                  block,
                                  BLOCK_BYTES,
                                  cudaMemcpyHostToDevice,
                                  fx.streams[s]));
        }
      }
      CUDA_OK(cudaStreamSynchronize(fx.streams[s]));
    });
  }
  for (auto& t : submitters) {
    t.join();
  }

  run_stats st;
  st.wall_s     = seconds_d{clock_type::now() - start}.count();
  st.all_cached = true;  // no state to advance
  return st;
}

// ---------------------------------------------------------------------------
// arm B — cudaEvent + a 2-thread retirement pool
// ---------------------------------------------------------------------------

/// One shared queue served by RETIRE_THREADS workers, each blocking on the
/// batch's event before running its retirement fn.
class event_retire_pool {
 public:
  using fn_type = sirius::exec::invocable<void(cudaError_t) noexcept>;

  explicit event_retire_pool(std::size_t n_threads)
  {
    _workers.reserve(n_threads);
    for (std::size_t i = 0; i < n_threads; ++i) {
      _workers.emplace_back([this] { work_loop(); });
    }
  }

  ~event_retire_pool()
  {
    {
      std::lock_guard g(_m);
      _stop = true;
    }
    _cv.notify_all();
    for (auto& t : _workers) {
      t.join();
    }
    for (auto ev : _events) {
      cudaEventDestroy(ev);
    }
  }

  event_retire_pool(event_retire_pool const&)            = delete;
  event_retire_pool& operator=(event_retire_pool const&) = delete;

  /// Record on `stream` and hand the wait to a worker.
  void submit(cudaStream_t stream, fn_type fn)
  {
    cudaEvent_t ev = acquire_event();
    CUDA_OK(cudaEventRecord(ev, stream));
    {
      std::lock_guard g(_m);
      _queue.push_back({ev, std::move(fn)});
    }
    _cv.notify_one();
  }

  /// Block until every submitted batch has been retired.
  void wait_idle()
  {
    std::unique_lock lk(_m);
    _idle_cv.wait(lk, [this] { return _queue.empty() && _in_flight == 0; });
  }

 private:
  struct item {
    cudaEvent_t ev;
    fn_type fn;
  };

  cudaEvent_t acquire_event()
  {
    {
      std::lock_guard g(_free_m);
      if (!_free_events.empty()) {
        auto ev = _free_events.back();
        _free_events.pop_back();
        return ev;
      }
    }
    cudaEvent_t ev{};
    // Timing disabled: we only need the completion signal, and the timing
    // variant costs more to record.
    CUDA_OK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    std::lock_guard g(_free_m);
    _events.push_back(ev);
    return ev;
  }

  void release_event(cudaEvent_t ev)
  {
    std::lock_guard g(_free_m);
    _free_events.push_back(ev);
  }

  void work_loop()
  {
    for (;;) {
      item it;
      {
        std::unique_lock lk(_m);
        _cv.wait(lk, [this] { return _stop || !_queue.empty(); });
        if (_queue.empty()) {
          if (_stop) { return; }
          continue;
        }
        it = std::move(_queue.front());
        _queue.pop_front();
        ++_in_flight;
      }

      const cudaError_t status = cudaEventSynchronize(it.ev);
      it.fn(status);
      release_event(it.ev);

      {
        std::lock_guard g(_m);
        --_in_flight;
      }
      _idle_cv.notify_all();
    }
  }

  std::mutex _m;
  std::condition_variable _cv;
  std::condition_variable _idle_cv;
  std::deque<item> _queue;
  std::size_t _in_flight{0};
  bool _stop{false};
  std::vector<std::thread> _workers;

  std::mutex _free_m;
  std::vector<cudaEvent_t> _events;       // owned, destroyed at teardown
  std::vector<cudaEvent_t> _free_events;  // recycled
};

run_stats run_events(fixture& fx, chunk_table& chunks, std::size_t staging_blocks)
{
  chunks.reset();
  block_pool pool{fx.staging(staging_blocks)};
  event_retire_pool retire_pool{RETIRE_THREADS};

  std::vector<double> lag_ms(N_BATCHES, 0.0);
  std::vector<clock_type::time_point> committed_at(N_BATCHES);
  std::atomic<double> staging_wait_s{0.0};

  const auto start = clock_type::now();
  progress_sampler sampler{chunks, start};

  std::vector<std::thread> submitters;
  submitters.reserve(g_n_streams);
  for (std::size_t s = 0; s < g_n_streams; ++s) {
    submitters.emplace_back([&, s] {
      double waited = 0.0;

      for (std::size_t batch : batches_for(s)) {
        std::vector<std::byte*> blocks;
        blocks.reserve(BATCH_CHUNKS);
        const auto wait_start = clock_type::now();
        for (std::size_t i = 0; i < BATCH_CHUNKS; ++i) {
          // The retirement threads are what refill the pool, so this just waits.
          blocks.push_back(pool.pop_blocking());
        }
        waited += seconds_d{clock_type::now() - wait_start}.count();

        for (std::size_t i = 0; i < BATCH_CHUNKS; ++i) {
          chunks.states[batch * BATCH_CHUNKS + i].store(chunk_state::loading,
                                                        std::memory_order_release);
        }

        for (std::size_t i = 0; i < BATCH_CHUNKS; ++i) {
          CUDA_OK(cudaMemcpyAsync(fx.device_chunks[batch * BATCH_CHUNKS + i],
                                  blocks[i],
                                  BLOCK_BYTES,
                                  cudaMemcpyHostToDevice,
                                  fx.streams[s]));
        }

        committed_at[batch] = clock_type::now();
        retire_pool.submit(
          fx.streams[s],
          [&chunks, &pool, &lag_ms, &committed_at, batch, blocks = std::move(blocks)](
            cudaError_t status) mutable noexcept {
            lag_ms[batch] = seconds_d{clock_type::now() - committed_at[batch]}.count() * 1e3;
            retire_batch(chunks, pool, batch, blocks, status);
          });
      }

      double expected = staging_wait_s.load(std::memory_order_relaxed);
      while (!staging_wait_s.compare_exchange_weak(expected, expected + waited)) {}
    });
  }
  for (auto& t : submitters) {
    t.join();
  }
  retire_pool.wait_idle();

  const auto wall = seconds_d{clock_type::now() - start}.count();
  sampler.stop();

  run_stats st;
  st.wall_s         = wall;
  st.retire_lag_ms  = std::move(lag_ms);
  st.staging_wait_s = staging_wait_s.load();
  st.t50_s          = sampler.time_to(0.5);
  st.t90_s          = sampler.time_to(0.9);
  st.t100_s         = sampler.time_to(1.0);
  st.all_cached     = chunks.n_cached.load() == N_CHUNKS;
  return st;
}

// ---------------------------------------------------------------------------
// reporting
// ---------------------------------------------------------------------------

struct aggregate {
  std::string arm;
  double wall_s{0};
  double gib_s{0};
  double t50_ms{0};
  double t90_ms{0};
  double t100_ms{0};
  double lag_p50_ms{0};
  double lag_p99_ms{0};
  double lag_max_ms{0};
  double staging_wait_ms{0};
  bool all_cached{true};
};

aggregate summarize(std::string arm, std::vector<run_stats> const& runs)
{
  aggregate a;
  a.arm = std::move(arm);

  std::vector<double> walls, t50, t90, t100, waits, lags;
  for (auto const& r : runs) {
    walls.push_back(r.wall_s);
    t50.push_back(r.t50_s);
    t90.push_back(r.t90_s);
    t100.push_back(r.t100_s);
    waits.push_back(r.staging_wait_s);
    lags.insert(lags.end(), r.retire_lag_ms.begin(), r.retire_lag_ms.end());
    a.all_cached = a.all_cached && r.all_cached;
  }

  a.wall_s  = median_of(walls);
  a.gib_s   = static_cast<double>(N_CHUNKS * BLOCK_BYTES) / (1024.0 * 1024.0 * 1024.0) / a.wall_s;
  a.t50_ms  = median_of(t50) * 1e3;
  a.t90_ms  = median_of(t90) * 1e3;
  a.t100_ms = median_of(t100) * 1e3;
  a.lag_p50_ms      = percentile(lags, 0.50);
  a.lag_p99_ms      = percentile(lags, 0.99);
  a.lag_max_ms      = percentile(lags, 1.0);
  a.staging_wait_ms = median_of(waits) * 1e3;
  return a;
}

void print_table(char const* title, aggregate const& retirer, aggregate const& events)
{
  std::printf("\n== %s ==\n", title);
  std::printf("%-26s %14s %14s %12s\n", "", "retirer", "event+pool(2)", "delta");
  auto row = [](char const* label, double a, double b, char const* unit) {
    const double delta = b == 0.0 ? 0.0 : (a - b) / b * 100.0;
    std::printf("%-26s %11.3f %-2s %11.3f %-2s %+11.1f%%\n", label, a, unit, b, unit, delta);
  };
  row("wall", retirer.wall_s * 1e3, events.wall_s * 1e3, "ms");
  row("H2D bandwidth", retirer.gib_s, events.gib_s, "GiB/s");
  row("time to 50% cached", retirer.t50_ms, events.t50_ms, "ms");
  row("time to 90% cached", retirer.t90_ms, events.t90_ms, "ms");
  row("time to 100% cached", retirer.t100_ms, events.t100_ms, "ms");
  row("retire lag p50", retirer.lag_p50_ms, events.lag_p50_ms, "ms");
  row("retire lag p99", retirer.lag_p99_ms, events.lag_p99_ms, "ms");
  row("retire lag max", retirer.lag_max_ms, events.lag_max_ms, "ms");
  row("staging wait (all thr)", retirer.staging_wait_ms, events.staging_wait_ms, "ms");
  if (!retirer.all_cached || !events.all_cached) {
    std::printf("  !! not every chunk reached `cached` -- results are invalid\n");
  }
}

}  // namespace

int main(int argc, char** argv)
{
  std::size_t reps = 5;
  bool sweep       = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.starts_with("--reps=")) { reps = std::stoul(arg.substr(7)); }
    if (arg.starts_with("--streams=")) {
      g_n_streams = std::min<std::size_t>(MAX_STREAMS, std::stoul(arg.substr(10)));
    }
    if (arg == "--stream-sweep") { sweep = true; }
  }

  fixture fx;
  chunk_table chunks{N_CHUNKS};

  // The DMA floor, so the arms below can be read against something other than
  // each other.  Anything above this is what retirement costs.
  {
    run_no_retirement(fx, chunks);  // warm
    std::vector<run_stats> runs;
    for (std::size_t r = 0; r < reps; ++r) {
      runs.push_back(run_no_retirement(fx, chunks));
    }
    const auto a = summarize("none", runs);
    std::printf(
      "DMA floor (copies only, no retirement): %.3f ms  %.2f GiB/s\n", a.wall_s * 1e3, a.gib_s);
  }

  // Does adding streams add H2D bandwidth?  If the GPU has a single host-to-
  // device copy engine they all serialize onto it, the run is link-bound at any
  // stream count, and no retirement scheme can change the wall time -- which
  // decides how the main tables below may be read.
  if (sweep) {
    std::printf("H2D bandwidth vs stream count (full staging, retirer arm):\n");
    for (std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
      g_n_streams = n;
      run_retirer(fx, chunks, FULL_STAGING_BLOCKS);  // warm
      std::vector<run_stats> runs;
      for (std::size_t r = 0; r < reps; ++r) {
        runs.push_back(run_retirer(fx, chunks, FULL_STAGING_BLOCKS));
      }
      const auto a = summarize("retirer", runs);
      std::printf(
        "  %zu stream%-2s %8.2f ms  %6.2f GiB/s\n", n, n == 1 ? " " : "s", a.wall_s * 1e3, a.gib_s);
    }
    return 0;
  }

  std::printf("retirer_benchmark: %zu chunks x %zu MiB, batches of %zu, %zu streams, %zu reps\n",
              N_CHUNKS,
              BLOCK_BYTES >> 20,
              BATCH_CHUNKS,
              g_n_streams,
              reps);

  struct config {
    char const* title;
    std::size_t staging_blocks;
  };
  const std::array<config, 2> configs{
    config{"full staging (1 GiB pinned, retirement never on the critical path)",
           FULL_STAGING_BLOCKS},
    config{"bounded staging (64 MiB pinned, recycled by retirement)", BOUNDED_STAGING_BLOCKS},
  };

  for (auto const& cfg : configs) {
    // One untimed pass per arm so neither pays for first-touch or pool warmup.
    run_retirer(fx, chunks, cfg.staging_blocks);
    run_events(fx, chunks, cfg.staging_blocks);

    std::vector<run_stats> retirer_runs, event_runs;
    for (std::size_t r = 0; r < reps; ++r) {
      retirer_runs.push_back(run_retirer(fx, chunks, cfg.staging_blocks));
      event_runs.push_back(run_events(fx, chunks, cfg.staging_blocks));
    }
    print_table(cfg.title, summarize("retirer", retirer_runs), summarize("events", event_runs));
  }

  std::printf(
    "\nnote: retire lag is measured commit()->fn and includes the copy itself;\n"
    "      only the difference between the arms is meaningful.\n");
  return 0;
}
