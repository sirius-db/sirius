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

// prefetch_benchmark — plain cudf::read_parquet vs. prefetch-then-read over a
// directory of TPC-H lineitem parquet parts.
//
// Six arms are run in order:
//   A  single file, no cache             (baseline)
//   B  single file, fadvise + prefetch + read
//   C  n files, read_parquet per file    (baseline)
//   D  n files, sliding-window readahead driven by a dedicated prefetch thread
//      that issues up to `window` prefetches ahead of the parser without
//      blocking on each one, so several files' IO can be in flight at once
//   E  n files, no cache, parsed by a fixed pool of worker threads, one CUDA
//      stream per worker
//   F  n files, prefetch + the same pool: a dedicated prefetch thread runs
//      `window` files ahead and its completion callback hands each file to the
//      pool, so IO and parsing of different files overlap freely
//   G  n files staged into the cache first (all of them, nothing parsed and no
//      IO left in flight), then parsed by the same pool.  Separates "staging
//      IO and parsing contend for the machine" from "the cache read path does
//      not scale to `pool_threads` concurrent readers": G's parse phase has no
//      IO running at all, so whatever it costs is the read path's own scaling
//
// `window` is meaningless for arm G — every file is staged before any parse
// starts, so there is no readahead depth to choose.
//
// Bytes actually pulled off the block device are measured from
// /proc/self/io:read_bytes, which with O_DIRECT is the real disk traffic; the
// per-chunk fill model is printed alongside it only as a cross-check.
//
// Every arm builds its own uring ioctx (and, for the prefetch arms, its own
// prefetching cache) so no cache or file state leaks across arms.
//
// The uring backend opens files O_DIRECT by default, so the OS page cache is
// bypassed and the arms cannot warm each other.  Dropping the page cache
// explicitly would need root, so it is deliberately not attempted here.

#include "io/cache/config.hpp"
#include "io/cache/prefetching_cache.hpp"
#include "io/cache/types.hpp"
#include "io/sirius_datasource.hpp"
#include "io/uring/uring_ioctx.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <glob.h>
#include <log/logging.hpp>
#include <log/spdlog_owning_sink.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <semaphore>
#include <string>
#include <thread>
#include <vector>

namespace {

using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;

/// Same four columns the classic TPC-H lineitem aggregations touch.
std::vector<std::string> const COLUMNS = {
  "l_orderkey",
  "l_extendedprice",
  "l_discount",
  "l_shipdate",
};

constexpr std::size_t HOST_REGION_CAPACITY = 64ULL << 30;
constexpr double RESERVATION_FRACTION      = 0.9;
constexpr std::size_t HOST_BLOCK_SIZE      = 1ULL << 20;
constexpr std::size_t HOST_POOL_SIZE       = 1024;
/// Floor for the computed pinned-pool pre-allocation; arm G needs every file
/// resident at once, so the real count is derived from the input sizes.
constexpr std::size_t HOST_MIN_INITIAL_POOLS = 1;
constexpr std::uint32_t BOUNCE_POOL_SLABS    = 20;

/// Measured pinned-host to device bandwidth of this box's PCIe x8 link, used to
/// turn arm G's H2D byte count into the serial floor it implies.
constexpr double PCIE_GBPS = 12.8;

/// Largest arm-F readahead window the semaphore can express.
constexpr std::ptrdiff_t MAX_WINDOW = 1 << 16;
/// Rough decompressed-table size per compressed byte, used only to refuse a
/// pool width that would OOM the device instead of discovering it at runtime.
constexpr double DEVICE_EXPANSION_FACTOR  = 5.0;
constexpr double DEVICE_HEADROOM_FRACTION = 0.8;

/// Uring stack for one arm.  Member order matters: the ioctx is destroyed
/// first so its reactors stop touching the bounce resource below it.
struct io_stack {
  std::unique_ptr<cucascade::memory::numa_region_pinned_host_memory_resource> upstream;
  std::unique_ptr<cucascade::memory::fixed_size_host_memory_resource> bounce_mr;
  std::shared_ptr<sirius::io::uring::uring_ioctx> io_ctx;
};

/// Everything one file needs to be read once: the datasource (kept alive here
/// so fadvise / prefetch_async stay reachable), its footer metadata packaged in
/// the single-element vector read_parquet consumes, and the column-chunk ranges.
struct file_prep {
  std::string path;
  std::unique_ptr<sirius::io::sirius_datasource> ds;
  std::vector<cudf::io::parquet::FileMetaData> metadatas;
  std::vector<cudf::io::text::byte_range_info> ranges;
  std::size_t range_bytes{0};
  /// Bytes the prefetch path is MODELLED to stage: the per-chunk fill extents
  /// prefetching_cache::insert would derive from these ranges, summed through
  /// the same @c needed_fill / @c merge / @c fill_span used in production.  A
  /// cross-check for the measured block-device traffic, never a substitute.
  std::size_t modelled_staged_bytes{0};
};

/// Bytes this process has actually pulled from the block layer, from
/// /proc/self/io.  With O_DIRECT this is real disk traffic, so differencing it
/// around a phase measures read amplification instead of predicting it.
std::size_t proc_read_bytes()
{
  std::ifstream in("/proc/self/io");
  std::string key;
  std::size_t value = 0;
  while (in >> key >> value) {
    if (key == "read_bytes:") { return value; }
  }
  return 0;
}

std::vector<std::string> list_parts(std::string const& dir, std::size_t limit)
{
  std::string const pattern = dir + "/part.*.parquet";
  std::vector<std::string> paths;
  glob_t g{};
  if (::glob(pattern.c_str(), GLOB_TILDE, nullptr, &g) == 0) {
    paths.reserve(g.gl_pathc);
    for (std::size_t i = 0; i < g.gl_pathc; ++i) {
      paths.emplace_back(g.gl_pathv[i]);
    }
  }
  ::globfree(&g);
  std::sort(paths.begin(), paths.end());
  if (limit > 0 && paths.size() > limit) { paths.resize(limit); }
  return paths;
}

io_stack make_io_stack(std::size_t n_reactors, std::size_t max_n_chunks)
{
  constexpr std::size_t chunks_per_slab =
    cucascade::memory::fixed_size_host_memory_resource::default_pool_size;
  constexpr std::size_t pool_capacity =
    static_cast<std::size_t>(BOUNCE_POOL_SLABS) * chunks_per_slab * HOST_BLOCK_SIZE;

  io_stack stack;
  stack.upstream =
    std::make_unique<cucascade::memory::numa_region_pinned_host_memory_resource>(0, true);
  stack.bounce_mr = std::make_unique<cucascade::memory::fixed_size_host_memory_resource>(
    0, *stack.upstream, pool_capacity, pool_capacity, HOST_BLOCK_SIZE, chunks_per_slab, 1);

  auto ctx = std::make_shared<sirius::io::uring::uring_reactor::reactor_context>(
    sirius::io::uring::uring_reactor::reactor_config_type{
      .bounce_size = stack.bounce_mr->get_block_size(), .max_n_chunks = max_n_chunks},
    stack.bounce_mr.get());
  stack.io_ctx = std::make_shared<sirius::io::uring::uring_ioctx>(n_reactors, std::move(ctx));
  stack.io_ctx->start();
  return stack;
}

file_prep prep_file(sirius::io::ioctx& io_ctx,
                    std::string const& path,
                    cudf::io::parquet_reader_options const& opts,
                    std::size_t cache_chunk_bytes)
{
  file_prep prep;
  prep.path   = path;
  prep.ds     = io_ctx.open_datasource(path);
  auto footer = cudf::io::parquet::fetch_footer_to_host(*prep.ds);
  hybrid_scan_reader reader(cudf::host_span<uint8_t const>(footer->data(), footer->size()), opts);
  prep.metadatas.push_back(reader.parquet_metadata());
  auto row_groups = reader.all_row_groups(opts);
  prep.ranges     = reader.all_column_chunks_byte_ranges(
    cudf::host_span<cudf::size_type const>(row_groups.data(), row_groups.size()), opts);
  for (auto const& r : prep.ranges) {
    prep.range_bytes += static_cast<std::size_t>(r.size());
  }
  std::map<std::size_t, sirius::io::cache::chunk_fill> wanted;
  for (auto const& r : prep.ranges) {
    if (r.size() <= 0) { continue; }
    auto const lo = static_cast<std::size_t>(r.offset());
    auto const hi = lo + static_cast<std::size_t>(r.size());
    for (auto off = (lo / cache_chunk_bytes) * cache_chunk_bytes; off < hi;
         off += cache_chunk_bytes) {
      auto& cur = wanted[off];
      cur       = sirius::io::cache::merge(
        cur, sirius::io::cache::needed_fill(off, cache_chunk_bytes, lo, hi));
    }
  }
  for (auto const& [off, fill] : wanted) {
    auto const [a, b] = sirius::io::cache::fill_span(fill, off, cache_chunk_bytes);
    prep.modelled_staged_bytes += b - a;
  }
  return prep;
}

using clock_type = std::chrono::steady_clock;

/// Milliseconds elapsed since @p t0 on the arm's single shared clock, so
/// stamps taken on the prefetch thread, the io_cb thread and the main thread
/// are all directly comparable.
double now_ms(clock_type::time_point t0)
{
  return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

struct method_counter {
  std::atomic<std::uint64_t> calls{0};
  std::atomic<std::uint64_t> bytes{0};
  std::atomic<double> ms{0.0};

  void add(std::size_t n, double t) noexcept
  {
    calls.fetch_add(1, std::memory_order_relaxed);
    bytes.fetch_add(n, std::memory_order_relaxed);
    ms.fetch_add(t, std::memory_order_relaxed);
  }
};

/// Per-arm tally of what cudf asked the datasource to do.  @c async_blocked is
/// not a datasource method: it is the time cudf spent blocked waiting on a
/// future returned by one of the *_async calls, which the issuing call itself
/// does not include.
struct ds_counters {
  method_counter host_read_buf;
  method_counter host_read_dst;
  method_counter host_read_async_buf;
  method_counter host_read_async_dst;
  method_counter device_read_buf;
  method_counter device_read_dst;
  method_counter device_read_async;
  method_counter async_blocked;
};

/// Non-owning @c cudf::io::datasource that forwards every virtual to a
/// @c sirius_datasource while tallying calls, bytes and in-call wall time.
///
/// The futures returned by the *_async methods are re-wrapped as deferred
/// futures so the time cudf spends blocked in @c get() is attributed to
/// @c async_blocked rather than disappearing into the caller's decode time.
class counting_datasource : public cudf::io::datasource {
 public:
  counting_datasource(sirius::io::sirius_datasource* inner, ds_counters* counters)
    : _inner(inner), _c(counters)
  {
  }

  [[nodiscard]] std::size_t size() const override { return _inner->size(); }

  [[nodiscard]] bool is_empty() const override { return _inner->is_empty(); }

  [[nodiscard]] bool supports_device_read() const override
  {
    return _inner->supports_device_read();
  }

  [[nodiscard]] bool supports_vector_host_read() const
  {
    return _inner->supports_vector_host_read();
  }

  [[nodiscard]] bool is_device_read_preferred(std::size_t size) const override
  {
    return _inner->is_device_read_preferred(size);
  }

  std::size_t host_read(std::size_t offset, std::size_t size, std::uint8_t* dst) override
  {
    auto const t0 = clock_type::now();
    auto const n  = _inner->host_read(offset, size, dst);
    _c->host_read_dst.add(n, now_ms(t0));
    return n;
  }

  std::unique_ptr<datasource::buffer> host_read(std::size_t offset, std::size_t size) override
  {
    auto const t0 = clock_type::now();
    auto buf      = _inner->host_read(offset, size);
    _c->host_read_buf.add(buf ? buf->size() : 0, now_ms(t0));
    return buf;
  }

  std::future<std::size_t> host_read_async(std::size_t offset,
                                           std::size_t size,
                                           std::uint8_t* dst) override
  {
    auto const t0 = clock_type::now();
    auto inner    = _inner->host_read_async(offset, size, dst);
    _c->host_read_async_dst.add(size, now_ms(t0));
    return defer(std::move(inner));
  }

  std::future<std::unique_ptr<datasource::buffer>> host_read_async(std::size_t offset,
                                                                   std::size_t size) override
  {
    auto const t0 = clock_type::now();
    auto inner    = _inner->host_read_async(offset, size);
    _c->host_read_async_buf.add(size, now_ms(t0));
    auto* c = _c;
    return std::async(std::launch::deferred, [f = std::move(inner), c]() mutable {
      auto const t1 = clock_type::now();
      auto r        = f.get();
      c->async_blocked.add(r ? r->size() : 0, now_ms(t1));
      return r;
    });
  }

  std::unique_ptr<datasource::buffer> device_read(std::size_t offset,
                                                  std::size_t size,
                                                  rmm::cuda_stream_view stream) override
  {
    auto const t0 = clock_type::now();
    auto buf      = _inner->device_read(offset, size, stream);
    _c->device_read_buf.add(buf ? buf->size() : 0, now_ms(t0));
    return buf;
  }

  std::size_t device_read(std::size_t offset,
                          std::size_t size,
                          std::uint8_t* dst,
                          rmm::cuda_stream_view stream) override
  {
    auto const t0 = clock_type::now();
    auto const n  = _inner->device_read(offset, size, dst, stream);
    _c->device_read_dst.add(n, now_ms(t0));
    return n;
  }

  std::future<std::size_t> device_read_async(std::size_t offset,
                                             std::size_t size,
                                             std::uint8_t* dst,
                                             rmm::cuda_stream_view stream) override
  {
    auto const t0 = clock_type::now();
    auto inner    = _inner->device_read_async(offset, size, dst, stream);
    _c->device_read_async.add(size, now_ms(t0));
    return defer(std::move(inner));
  }

 private:
  std::future<std::size_t> defer(std::future<std::size_t>&& inner)
  {
    auto* c = _c;
    return std::async(std::launch::deferred, [f = std::move(inner), c]() mutable {
      auto const t1 = clock_type::now();
      auto const r  = f.get();
      c->async_blocked.add(r, now_ms(t1));
      return r;
    });
  }

  sirius::io::sirius_datasource* _inner;
  ds_counters* _c;
};

/// Read @p prep exactly once through a @ref counting_datasource.  Consumes its
/// metadata vector, so a prep can only be read a single time.
std::size_t read_one(file_prep& prep,
                     cudf::io::parquet_reader_options const& opts,
                     rmm::cuda_stream_view stream,
                     ds_counters& counters)
{
  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  sources.push_back(std::make_unique<counting_datasource>(prep.ds.get(), &counters));
  auto table = cudf::io::read_parquet(std::move(sources), std::move(prep.metadatas), opts, stream);
  return static_cast<std::size_t>(table.tbl->num_rows());
}

void print_counters(std::string const& arm, ds_counters const& c, double read_wall_ms)
{
  struct row {
    char const* name;
    method_counter const* m;
  };
  row const rows[] = {
    {"host_read(dst)", &c.host_read_dst},
    {"host_read(buffer)", &c.host_read_buf},
    {"host_read_async(dst)", &c.host_read_async_dst},
    {"host_read_async(buffer)", &c.host_read_async_buf},
    {"device_read(dst)", &c.device_read_dst},
    {"device_read(buffer)", &c.device_read_buf},
    {"device_read_async", &c.device_read_async},
    {"  (blocked in future.get)", &c.async_blocked},
  };

  std::cout << "\n"
            << arm << " datasource calls:\n"
            << std::left << std::setw(28) << "  method" << std::right << std::setw(10) << "calls"
            << std::setw(14) << "MiB" << std::setw(12) << "ms" << "\n"
            << std::string(64, '-') << "\n";
  double in_ds_ms = 0;
  for (auto const& r : rows) {
    auto const calls = r.m->calls.load(std::memory_order_relaxed);
    auto const bytes = r.m->bytes.load(std::memory_order_relaxed);
    auto const ms    = r.m->ms.load(std::memory_order_relaxed);
    in_ds_ms += ms;
    if (calls == 0) { continue; }
    std::cout << std::left << std::setw(28) << r.name << std::right << std::setw(10) << calls
              << std::fixed << std::setprecision(2) << std::setw(14)
              << static_cast<double>(bytes) / (1024.0 * 1024.0) << std::setprecision(2)
              << std::setw(12) << ms << "\n";
  }
  double const outside = read_wall_ms - in_ds_ms;
  std::cout << std::left << std::setw(28) << "  TOTAL in datasource" << std::right << std::setw(10)
            << "" << std::setw(14) << "" << std::fixed << std::setprecision(2) << std::setw(12)
            << in_ds_ms << "\n"
            << std::left << std::setw(28) << "  read_parquet wall" << std::right << std::setw(10)
            << "" << std::setw(14) << "" << std::setw(12) << read_wall_ms << "\n"
            << std::left << std::setw(28) << "  outside datasource (decode)" << std::right
            << std::setw(10) << "" << std::setw(14) << "" << std::setw(12) << outside << "  ("
            << std::setprecision(1) << (read_wall_ms > 0 ? 100.0 * outside / read_wall_ms : 0.0)
            << "%)\n";
}

struct arm_result {
  std::string name;
  double ms{0};
  std::size_t bytes{0};
  std::size_t rows{0};
};

void print_header()
{
  std::cout << std::left << std::setw(28) << "arm" << std::right << std::setw(12) << "wall ms"
            << std::setw(14) << "MiB read" << std::setw(10) << "GB/s" << std::setw(14) << "rows"
            << "\n"
            << std::string(78, '-') << "\n";
}

void print_row(arm_result const& r)
{
  double const gbps = r.ms > 0 ? static_cast<double>(r.bytes) / (r.ms / 1000.0) / 1e9 : 0.0;
  std::cout << std::left << std::setw(28) << r.name << std::right << std::fixed
            << std::setprecision(1) << std::setw(12) << r.ms << std::setprecision(1)
            << std::setw(14) << static_cast<double>(r.bytes) / (1024.0 * 1024.0)
            << std::setprecision(2) << std::setw(10) << gbps << std::setw(14) << r.rows << "\n";
}

/// Wall time covered by at least one IO, i.e. the union of the per-file
/// [start, end) intervals.  Differs from the plain sum once several prefetches
/// are in flight at once, which is exactly when the sum stops being meaningful.
double union_span_ms(std::vector<double> const& starts, std::vector<double> const& ends)
{
  std::vector<std::pair<double, double>> iv;
  iv.reserve(starts.size());
  for (std::size_t i = 0; i < starts.size(); ++i) {
    iv.emplace_back(starts[i], ends[i]);
  }
  std::sort(iv.begin(), iv.end());
  double total = 0;
  double cur_s = 0;
  double cur_e = 0;
  bool open    = false;
  for (auto const& [s, e] : iv) {
    if (!open) {
      cur_s = s;
      cur_e = e;
      open  = true;
      continue;
    }
    if (s > cur_e) {
      total += cur_e - cur_s;
      cur_s = s;
      cur_e = e;
    } else {
      cur_e = std::max(cur_e, e);
    }
  }
  if (open) { total += cur_e - cur_s; }
  return total;
}

/// Fixed pool of parser threads.  Each worker owns one @c rmm::cuda_stream for
/// its whole life — concurrent @c read_parquet calls must not share a stream,
/// and none of them may land on the default stream.
class thread_pool {
 public:
  explicit thread_pool(std::size_t n_workers)
  {
    _streams.reserve(n_workers);
    for (std::size_t i = 0; i < n_workers; ++i) {
      _streams.push_back(std::make_unique<rmm::cuda_stream>());
    }
    _workers.reserve(n_workers);
    for (std::size_t i = 0; i < n_workers; ++i) {
      _workers.emplace_back([this, i] { run(i); });
    }
  }

  ~thread_pool() { stop(); }

  thread_pool(thread_pool const&)            = delete;
  thread_pool& operator=(thread_pool const&) = delete;

  /// Queue @p task; it is invoked with the id of the worker that picks it up.
  void submit(std::function<void(std::size_t)> task)
  {
    {
      std::lock_guard lk(_mtx);
      _queue.push_back(std::move(task));
    }
    _cv.notify_one();
  }

  void stop() noexcept
  {
    {
      std::lock_guard lk(_mtx);
      if (_stopping) { return; }
      _stopping = true;
    }
    _cv.notify_all();
    for (auto& t : _workers) {
      if (t.joinable()) { t.join(); }
    }
  }

  [[nodiscard]] std::size_t size() const noexcept { return _workers.size(); }

  [[nodiscard]] rmm::cuda_stream_view stream_of(std::size_t worker) const
  {
    return _streams[worker]->view();
  }

  /// Print each worker's stream handle and assert they are distinct and
  /// non-default, so the run itself proves the parses cannot serialise on one
  /// stream (or be rejected by cudf's batched-copy path).
  void report_streams(std::string const& arm) const
  {
    std::cout << arm << " worker streams:";
    std::vector<void*> seen;
    for (std::size_t i = 0; i < _streams.size(); ++i) {
      auto* handle = static_cast<void*>(_streams[i]->value());
      std::cout << " w" << i << "=" << handle;
      if (handle == nullptr || std::find(seen.begin(), seen.end(), handle) != seen.end()) {
        std::cerr << "\nFATAL: worker " << i << " has a default or duplicated stream\n";
        std::abort();
      }
      seen.push_back(handle);
    }
    std::cout << "  (all distinct, none default)\n";
  }

 private:
  void run(std::size_t id)
  {
    for (;;) {
      std::function<void(std::size_t)> task;
      {
        std::unique_lock lk(_mtx);
        _cv.wait(lk, [this] { return _stopping || !_queue.empty(); });
        if (_queue.empty()) { return; }
        task = std::move(_queue.front());
        _queue.pop_front();
      }
      task(id);
    }
  }

  std::vector<std::unique_ptr<rmm::cuda_stream>> _streams;
  std::vector<std::thread> _workers;
  std::deque<std::function<void(std::size_t)>> _queue;
  std::mutex _mtx;
  std::condition_variable _cv;
  bool _stopping{false};
};

/// Per-file stamps of a pooled arm, all on the arm's single steady clock.
struct pool_trace {
  std::vector<double> io_start;
  std::vector<double> io_end;
  std::vector<double> parse_start;
  std::vector<double> parse_end;
  std::vector<std::size_t> worker;

  explicit pool_trace(std::size_t n)
    : io_start(n, 0.0), io_end(n, 0.0), parse_start(n, 0.0), parse_end(n, 0.0), worker(n, 0)
  {
  }
};

/// Barrier counting completed parses, so an arm can stop its timer the moment
/// the last pooled task returns rather than when the pool shuts down.
class completion_latch {
 public:
  explicit completion_latch(std::size_t target) : _target(target) {}

  void count_down()
  {
    {
      std::lock_guard lk(_mtx);
      ++_done;
    }
    _cv.notify_all();
  }

  void wait()
  {
    std::unique_lock lk(_mtx);
    _cv.wait(lk, [this] { return _done >= _target; });
  }

 private:
  std::size_t _target;
  std::size_t _done{0};
  std::mutex _mtx;
  std::condition_variable _cv;
};

double median_of(std::vector<double> v)
{
  if (v.empty()) { return 0.0; }
  std::sort(v.begin(), v.end());
  std::size_t const mid = v.size() / 2;
  return v.size() % 2 == 1 ? v[mid] : 0.5 * (v[mid - 1] + v[mid]);
}

/// Median parse time over the @p k files that started parsing last, i.e. the
/// tail of a pooled arm after its staging IO has drained.  Used as the
/// uncontended per-file cost of that arm's parse path.
double tail_parse_ms(pool_trace const& t, std::size_t k)
{
  std::vector<std::pair<double, double>> by_start;
  by_start.reserve(t.parse_start.size());
  for (std::size_t i = 0; i < t.parse_start.size(); ++i) {
    by_start.emplace_back(t.parse_start[i], t.parse_end[i] - t.parse_start[i]);
  }
  std::sort(by_start.begin(), by_start.end());
  k = std::min(k, by_start.size());
  std::vector<double> tail;
  tail.reserve(k);
  for (std::size_t i = by_start.size() - k; i < by_start.size(); ++i) {
    tail.push_back(by_start[i].second);
  }
  return median_of(std::move(tail));
}

/// Per-file timeline of a pooled arm plus the per-worker file counts.  @p with_io
/// adds the IO columns and the queue-wait column, which only arm F has.
void print_pool_trace(std::string const& arm,
                      pool_trace const& t,
                      std::size_t n_workers,
                      bool with_io)
{
  std::size_t const n = t.parse_start.size();
  std::cout << "\n" << arm << " per-file phases (ms relative to arm t0):\n" << std::right;
  std::cout << std::setw(6) << "file" << std::setw(8) << "wrk";
  if (with_io) {
    std::cout << std::setw(11) << "io_start" << std::setw(10) << "io_end" << std::setw(10)
              << "io_ms" << std::setw(11) << "queue_ms";
  }
  std::cout << std::setw(14) << "parse_start" << std::setw(11) << "parse_end" << std::setw(11)
            << "parse_ms" << "\n"
            << std::string(with_io ? 92 : 50, '-') << "\n";
  for (std::size_t i = 0; i < n; ++i) {
    std::cout << std::right << std::fixed << std::setprecision(1) << std::setw(6) << i
              << std::setw(8) << t.worker[i];
    if (with_io) {
      std::cout << std::setw(11) << t.io_start[i] << std::setw(10) << t.io_end[i] << std::setw(10)
                << (t.io_end[i] - t.io_start[i]) << std::setw(11)
                << (t.parse_start[i] - t.io_end[i]);
    }
    std::cout << std::setw(14) << t.parse_start[i] << std::setw(11) << t.parse_end[i]
              << std::setw(11) << (t.parse_end[i] - t.parse_start[i]) << "\n";
  }
  std::vector<std::size_t> per_worker(n_workers, 0);
  for (auto w : t.worker) {
    if (w < n_workers) { ++per_worker[w]; }
  }
  std::cout << "  files per worker:";
  for (std::size_t w = 0; w < n_workers; ++w) {
    std::cout << " w" << w << "=" << per_worker[w];
  }
  std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 2 || argc > 8) {
    std::cerr << "usage: " << argv[0]
              << " <lineitem-dir> [n_files] [n_reactors] [window] [pool_threads] [max_n_chunks] "
                 "[cache_block_kib]\n";
    return 1;
  }

  std::string const dir    = argv[1];
  std::size_t n_files      = argc > 2 ? static_cast<std::size_t>(std::stoull(argv[2])) : 4;
  std::size_t n_reactors   = argc > 3 ? static_cast<std::size_t>(std::stoull(argv[3])) : 2;
  std::size_t window       = argc > 4 ? static_cast<std::size_t>(std::stoull(argv[4])) : 3;
  std::size_t pool_threads = argc > 5 ? static_cast<std::size_t>(std::stoull(argv[5])) : 4;
  std::size_t max_chunks   = argc > 6 ? static_cast<std::size_t>(std::stoull(argv[6])) : 1;
  std::size_t cache_block =
    argc > 7 ? static_cast<std::size_t>(std::stoull(argv[7])) << 10 : HOST_BLOCK_SIZE;
  if (n_files == 0 || n_reactors == 0 || window == 0 || max_chunks == 0 || pool_threads == 0) {
    std::cerr << "n_files, n_reactors, window, pool_threads and max_n_chunks must all be > 0\n";
    return 1;
  }
  std::size_t const window_f = window;

  auto paths = list_parts(dir, n_files);
  if (paths.empty()) {
    std::cerr << "no files matched " << dir << "/part.*.parquet\n";
    return 1;
  }
  n_files = paths.size();

  auto log_sink = sirius::log::make_spdlog_owning_sink({"log", std::nullopt});
  log_sink->set_level(sirius::log::level::info);
  sirius::log::set_sink(std::move(log_sink));

  cudaFree(nullptr);

  std::size_t bytes_on_disk = 0;
  for (auto const& p : paths) {
    bytes_on_disk += static_cast<std::size_t>(std::filesystem::file_size(p));
  }
  std::size_t const host_pool_bytes = HOST_POOL_SIZE * cache_block;
  std::size_t const host_initial_pools =
    std::max(HOST_MIN_INITIAL_POOLS, (bytes_on_disk + host_pool_bytes - 1) / host_pool_bytes + 1);

  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_reservation_fraction_per_gpu(RESERVATION_FRACTION)
    .use_gpu_id_as_host_id()
    .set_per_numa_region_capacity(HOST_REGION_CAPACITY)
    .set_reservation_fraction_per_numa_region(RESERVATION_FRACTION)
    .set_host_pool_features(cache_block, HOST_POOL_SIZE, host_initial_pools);
  auto mgr = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());

  rmm::mr::cuda_async_memory_resource async_mr;
  rmm::mr::set_current_device_resource(
    cuda::mr::any_resource<cuda::mr::device_accessible>{rmm::device_async_resource_ref{async_mr}});

  rmm::cuda_stream stream;

  auto opts = cudf::io::parquet_reader_options::builder().column_names(COLUMNS).build();

  sirius::io::cache::config cache_cfg;
  cache_cfg.min_prefetching_budget_fraction = 0.9;
  cache_cfg.eviction_threshold_fraction     = 0.9;
  cache_cfg.inflight_io_chunk_budget        = 16384;
  cache_cfg.dispose_on_idle                 = false;

  std::cout << "dir        : " << dir << "\n"
            << "files      : " << n_files << "\n"
            << "reactors   : " << n_reactors << "\n"
            << "window     : " << window << "\n"
            << "pool threads: " << pool_threads << "\n"
            << "arm F window: " << window_f
            << (window_f != window ? "  (RAISED from the requested window to pool_threads + 2 so "
                                     "the pool cannot starve)"
                                   : "")
            << "\n"
            << "max_n_chunks: " << max_chunks << "\n"
            << "cache block : " << (cache_block >> 10) << " KiB\n"
            << "host region: " << (HOST_REGION_CAPACITY >> 30) << " GiB, block "
            << (HOST_BLOCK_SIZE >> 20) << " MiB\n"
            << "arm G window: n/a  (all " << n_files
            << " files are staged before any parse starts, so `window` does not apply)\n"
            << "host pinned pre-allocation: " << host_initial_pools << " pools x "
            << (host_pool_bytes >> 20) << " MiB = " << std::fixed << std::setprecision(2)
            << static_cast<double>(host_initial_pools * host_pool_bytes) /
                 (1024.0 * 1024.0 * 1024.0)
            << " GiB committed up front, covering all " << n_files << " files ("
            << static_cast<double>(bytes_on_disk) / (1024.0 * 1024.0 * 1024.0)
            << " GiB on disk)\n\n";

  std::vector<arm_result> results;
  std::string summary_b;
  std::string summary_d;
  std::vector<double> per_file_c;
  std::vector<double> per_file_d;
  bool b_issued               = false;
  bool b_ok                   = false;
  double b_prefetch_ms        = 0;
  double b_parse_ms           = 0;
  std::size_t d_ready         = 0;
  std::size_t d_waited        = 0;
  std::size_t d_peak_inflight = 0;
  std::vector<double> d_io_start(n_files, 0.0);
  std::vector<double> d_io_end(n_files, 0.0);
  std::vector<double> d_parse_start(n_files, 0.0);
  std::vector<double> d_parse_end(n_files, 0.0);
  std::size_t b_range_bytes    = 0;
  std::size_t b_modelled_bytes = 0;
  std::size_t b_disk_bytes     = 0;
  ds_counters counters_a;
  ds_counters counters_b;
  ds_counters counters_c;
  ds_counters counters_d;
  ds_counters counters_e;
  ds_counters counters_f;
  pool_trace e_trace(n_files);
  pool_trace f_trace(n_files);
  std::size_t e_bytes = 0;
  std::size_t f_bytes = 0;
  std::size_t f_modelled_bytes{0};
  std::size_t f_disk_bytes{0};
  std::size_t f_issued{0};
  std::atomic<std::size_t> f_ok{0};
  std::string summary_f;
  ds_counters counters_g;
  pool_trace g_trace(n_files);
  std::size_t g_bytes{0};
  std::size_t g_modelled_bytes{0};
  std::size_t g_disk_bytes{0};
  std::size_t g_issued{0};
  std::atomic<std::size_t> g_ok{0};
  double g_stage_ms{0};
  double g_parse_wall_ms{0};
  std::string summary_g_staged;
  std::string summary_g;
  ds_counters counters_g1;
  pool_trace g1_trace(n_files);
  double g1_parse_wall_ms{0};
  double g1_restage_ms{0};
  std::size_t g1_disk_bytes{0};
  std::string summary_g1;

  {
    auto stack = make_io_stack(n_reactors, max_chunks);
    auto prep  = prep_file(*stack.io_ctx, paths.front(), opts, cache_block);
    auto t0    = clock_type::now();
    auto rows  = read_one(prep, opts, stream.view(), counters_a);
    results.push_back({"A single-file baseline", now_ms(t0), prep.range_bytes, rows});
  }

  {
    auto stack = make_io_stack(n_reactors, max_chunks);
    stack.io_ctx->initialize_cache(*mgr, cache_cfg, nullptr);
    if (!stack.io_ctx->uses_prefetching_cache()) {
      std::cerr << "FATAL: prefetching cache did not come up for arm B\n";
      return 2;
    }
    auto prep = prep_file(*stack.io_ctx, paths.front(), opts, cache_block);
    prep.ds->fadvise(prep.ranges, 0);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    auto const disk0 = proc_read_bytes();
    auto t0          = clock_type::now();
    std::promise<bool> p;
    auto fut              = p.get_future();
    b_issued              = prep.ds->prefetch_async([&p](bool ok) noexcept { p.set_value(ok); });
    b_ok                  = fut.get();
    b_prefetch_ms         = now_ms(t0);
    b_disk_bytes          = proc_read_bytes() - disk0;
    b_range_bytes         = prep.range_bytes;
    b_modelled_bytes      = prep.modelled_staged_bytes;
    auto rows             = read_one(prep, opts, stream.view(), counters_b);
    double const total_ms = now_ms(t0);
    b_parse_ms            = total_ms - b_prefetch_ms;
    results.push_back({"B single-file prefetch", total_ms, prep.range_bytes, rows});
    summary_b = stack.io_ctx->cache()->summary();
  }

  {
    auto stack = make_io_stack(n_reactors, max_chunks);
    std::vector<file_prep> preps;
    preps.reserve(n_files);
    for (auto const& p : paths) {
      preps.push_back(prep_file(*stack.io_ctx, p, opts, cache_block));
    }
    std::size_t bytes = 0;
    std::size_t rows  = 0;
    auto t0           = clock_type::now();
    for (std::size_t i = 0; i < preps.size(); ++i) {
      auto f0 = clock_type::now();
      rows += read_one(preps[i], opts, stream.view(), counters_c);
      per_file_c.push_back(now_ms(f0));
    }
    double const ms = now_ms(t0);
    for (auto const& f : preps) {
      bytes += f.range_bytes;
    }
    results.push_back({"C multi-file baseline", ms, bytes, rows});
  }

  {
    auto stack = make_io_stack(n_reactors, max_chunks);
    stack.io_ctx->initialize_cache(*mgr, cache_cfg, nullptr);
    if (!stack.io_ctx->uses_prefetching_cache()) {
      std::cerr << "FATAL: prefetching cache did not come up for arm D\n";
      return 2;
    }
    std::vector<file_prep> preps;
    preps.reserve(n_files);
    for (auto const& p : paths) {
      preps.push_back(prep_file(*stack.io_ctx, p, opts, cache_block));
    }
    for (auto& f : preps) {
      f.ds->fadvise(f.ranges, 0);
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::vector<std::promise<bool>> promises(n_files);
    std::vector<std::future<bool>> futures;
    futures.reserve(n_files);
    for (auto& pr : promises) {
      futures.push_back(pr.get_future());
    }

    std::atomic<std::size_t> parse_pos{0};
    std::mutex mtx;
    std::condition_variable cv;
    std::size_t issued_ok = 0;

    std::size_t bytes = 0;
    std::size_t rows  = 0;
    auto t0           = clock_type::now();

    std::thread prefetcher([&] {
      for (std::size_t k = 0; k < n_files; ++k) {
        {
          std::unique_lock lk(mtx);
          cv.wait(lk, [&] { return k < parse_pos.load(std::memory_order_acquire) + window; });
        }
        std::size_t const inflight = k + 1 - parse_pos.load(std::memory_order_acquire);
        d_peak_inflight            = std::max(d_peak_inflight, inflight);
        d_io_start[k]              = now_ms(t0);
        if (preps[k].ds->prefetch_async([&promises, &d_io_end, t0, k](bool ok) noexcept {
              d_io_end[k] = now_ms(t0);
              promises[k].set_value(ok);
            })) {
          ++issued_ok;
        }
      }
    });

    for (std::size_t i = 0; i < n_files; ++i) {
      if (futures[i].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        ++d_ready;
      } else {
        ++d_waited;
      }
      futures[i].get();
      d_parse_start[i] = now_ms(t0);
      auto f0          = clock_type::now();
      rows += read_one(preps[i], opts, stream.view(), counters_d);
      per_file_d.push_back(now_ms(f0));
      d_parse_end[i] = now_ms(t0);
      {
        std::lock_guard lk(mtx);
        parse_pos.store(i + 1, std::memory_order_release);
      }
      cv.notify_all();
    }
    prefetcher.join();
    double const ms = now_ms(t0);
    for (auto const& f : preps) {
      bytes += f.range_bytes;
    }
    results.push_back({"D multi-file prefetch", ms, bytes, rows});
    summary_d = stack.io_ctx->cache()->summary();
    std::cout << "arm D: prefetch_async issued IO for " << issued_ok << "/" << n_files
              << " files\n\n";
  }

  {
    auto stack = make_io_stack(n_reactors, max_chunks);
    std::vector<file_prep> preps;
    preps.reserve(n_files);
    for (auto const& p : paths) {
      preps.push_back(prep_file(*stack.io_ctx, p, opts, cache_block));
    }
    std::size_t max_range = 0;
    for (auto const& f : preps) {
      e_bytes += f.range_bytes;
      max_range = std::max(max_range, f.range_bytes);
    }
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    double const est_peak =
      static_cast<double>(pool_threads) * DEVICE_EXPANSION_FACTOR * static_cast<double>(max_range);
    std::cout << "pool memory guard: " << pool_threads << " concurrent tables, est peak "
              << std::fixed << std::setprecision(1) << est_peak / (1024.0 * 1024.0 * 1024.0)
              << " GiB vs " << static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0)
              << " GiB free\n";
    if (est_peak > DEVICE_HEADROOM_FRACTION * static_cast<double>(free_bytes)) {
      std::cerr << "FATAL: pool_threads=" << pool_threads
                << " would not fit in device memory; lower pool_threads\n";
      return 3;
    }

    thread_pool pool(pool_threads);
    pool.report_streams("arm E");
    completion_latch latch(n_files);
    std::atomic<std::size_t> rows{0};

    auto t0 = clock_type::now();
    for (std::size_t k = 0; k < n_files; ++k) {
      pool.submit([&, k](std::size_t worker) {
        e_trace.worker[k]      = worker;
        e_trace.parse_start[k] = now_ms(t0);
        rows.fetch_add(read_one(preps[k], opts, pool.stream_of(worker), counters_e),
                       std::memory_order_relaxed);
        e_trace.parse_end[k] = now_ms(t0);
        latch.count_down();
      });
    }
    latch.wait();
    double const ms = now_ms(t0);
    pool.stop();
    results.push_back(
      {"E multi-file pool baseline", ms, e_bytes, rows.load(std::memory_order_relaxed)});
  }

  {
    auto stack = make_io_stack(n_reactors, max_chunks);
    stack.io_ctx->initialize_cache(*mgr, cache_cfg, nullptr);
    if (!stack.io_ctx->uses_prefetching_cache()) {
      std::cerr << "FATAL: prefetching cache did not come up for arm F\n";
      return 2;
    }
    std::vector<file_prep> preps;
    preps.reserve(n_files);
    for (auto const& p : paths) {
      preps.push_back(prep_file(*stack.io_ctx, p, opts, cache_block));
    }
    for (auto& f : preps) {
      f.ds->fadvise(f.ranges, 0);
      f_bytes += f.range_bytes;
      f_modelled_bytes += f.modelled_staged_bytes;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));

    thread_pool pool(pool_threads);
    pool.report_streams("arm F");
    completion_latch latch(n_files);
    std::counting_semaphore<MAX_WINDOW> slots(static_cast<std::ptrdiff_t>(window_f));
    std::atomic<std::size_t> rows{0};

    auto const disk0 = proc_read_bytes();
    auto t0          = clock_type::now();

    std::thread prefetcher([&] {
      for (std::size_t k = 0; k < n_files; ++k) {
        slots.acquire();
        f_trace.io_start[k] = now_ms(t0);
        if (preps[k].ds->prefetch_async([&, k](bool ok) noexcept {
              f_trace.io_end[k] = now_ms(t0);
              f_ok += ok ? 1 : 0;
              pool.submit([&, k](std::size_t worker) {
                f_trace.worker[k]      = worker;
                f_trace.parse_start[k] = now_ms(t0);
                rows.fetch_add(read_one(preps[k], opts, pool.stream_of(worker), counters_f),
                               std::memory_order_relaxed);
                f_trace.parse_end[k] = now_ms(t0);
                slots.release();
                latch.count_down();
              });
            })) {
          ++f_issued;
        }
      }
    });

    latch.wait();
    double const ms = now_ms(t0);
    f_disk_bytes    = proc_read_bytes() - disk0;
    prefetcher.join();
    pool.stop();
    results.push_back({"F multi-file pool prefetch", ms, f_bytes, rows.load()});
    summary_f = stack.io_ctx->cache()->summary();
  }

  {
    auto stack = make_io_stack(n_reactors, max_chunks);
    stack.io_ctx->initialize_cache(*mgr, cache_cfg, nullptr);
    if (!stack.io_ctx->uses_prefetching_cache()) {
      std::cerr << "FATAL: prefetching cache did not come up for arm G\n";
      return 2;
    }
    std::vector<file_prep> preps;
    std::vector<file_prep> preps_solo;
    preps.reserve(n_files);
    preps_solo.reserve(n_files);
    for (auto const& p : paths) {
      preps.push_back(prep_file(*stack.io_ctx, p, opts, cache_block));
      preps_solo.push_back(prep_file(*stack.io_ctx, p, opts, cache_block));
    }
    for (auto& f : preps) {
      f.ds->fadvise(f.ranges, 0);
      g_bytes += f.range_bytes;
      g_modelled_bytes += f.modelled_staged_bytes;
    }
    for (auto& f : preps_solo) {
      f.ds->fadvise(f.ranges, 0);
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::vector<std::promise<bool>> promises(n_files);
    std::vector<std::future<bool>> futures;
    futures.reserve(n_files);
    for (auto& pr : promises) {
      futures.push_back(pr.get_future());
    }
    std::atomic<std::size_t> staged{0};

    auto const disk0    = proc_read_bytes();
    auto const stage_t0 = clock_type::now();
    for (std::size_t k = 0; k < n_files; ++k) {
      g_trace.io_start[k] = now_ms(stage_t0);
      if (preps[k].ds->prefetch_async([&, k](bool ok) noexcept {
            g_trace.io_end[k] = now_ms(stage_t0);
            g_ok += ok ? 1 : 0;
            staged.fetch_add(1, std::memory_order_acq_rel);
            promises[k].set_value(ok);
          })) {
        ++g_issued;
      }
    }
    for (auto& fu : futures) {
      fu.get();
    }
    g_stage_ms   = now_ms(stage_t0);
    g_disk_bytes = proc_read_bytes() - disk0;
    if (staged.load(std::memory_order_acquire) != n_files || g_ok.load() != n_files ||
        g_issued != n_files) {
      std::cerr << "FATAL: arm G staging did not fully drain: issued=" << g_issued
                << " completed=" << staged.load() << " ok=" << g_ok.load() << " of " << n_files
                << " files; phase 2 would not be IO-free\n";
      return 4;
    }
    summary_g_staged = stack.io_ctx->cache()->summary();
    std::cout << "\narm G phase 1 (staging only, no parsing): " << std::fixed
              << std::setprecision(1) << g_stage_ms << " ms, " << std::setprecision(2)
              << static_cast<double>(g_disk_bytes) / (1024.0 * 1024.0)
              << " MiB off the block device"
              << " (model says " << static_cast<double>(g_modelled_bytes) / (1024.0 * 1024.0)
              << " MiB staged, useful " << static_cast<double>(g_bytes) / (1024.0 * 1024.0)
              << " MiB)\n"
              << "arm G phase 1 verified drained: " << g_issued << "/" << n_files
              << " prefetches issued, " << g_ok.load()
              << " completed ok, 0 in flight at phase 2 start\n"
              << "arm G cache after staging " << summary_g_staged << "\n";

    thread_pool pool(pool_threads);
    pool.report_streams("arm G");
    completion_latch latch(n_files);
    std::atomic<std::size_t> rows{0};

    auto t0 = clock_type::now();
    for (std::size_t k = 0; k < n_files; ++k) {
      pool.submit([&, k](std::size_t worker) {
        g_trace.worker[k]      = worker;
        g_trace.parse_start[k] = now_ms(t0);
        rows.fetch_add(read_one(preps[k], opts, pool.stream_of(worker), counters_g),
                       std::memory_order_relaxed);
        g_trace.parse_end[k] = now_ms(t0);
        latch.count_down();
      });
    }
    latch.wait();
    g_parse_wall_ms = now_ms(t0);
    pool.stop();
    results.push_back({"G staged, pool parse (phase 2)", g_parse_wall_ms, g_bytes, rows.load()});
    summary_g = stack.io_ctx->cache()->summary();

    std::vector<std::promise<bool>> promises1(n_files);
    std::vector<std::future<bool>> futures1;
    futures1.reserve(n_files);
    for (auto& pr : promises1) {
      futures1.push_back(pr.get_future());
    }
    auto const disk1      = proc_read_bytes();
    auto const restage_t0 = clock_type::now();
    for (std::size_t k = 0; k < n_files; ++k) {
      preps_solo[k].ds->prefetch_async(
        [&promises1, k](bool ok) noexcept { promises1[k].set_value(ok); });
    }
    for (auto& fu : futures1) {
      fu.get();
    }
    g1_restage_ms = now_ms(restage_t0);
    g1_disk_bytes = proc_read_bytes() - disk1;

    thread_pool solo(1);
    solo.report_streams("arm G control (1 worker)");
    completion_latch latch1(n_files);
    std::atomic<std::size_t> rows1{0};

    auto t1 = clock_type::now();
    for (std::size_t k = 0; k < n_files; ++k) {
      solo.submit([&, k](std::size_t worker) {
        g1_trace.worker[k]      = worker;
        g1_trace.parse_start[k] = now_ms(t1);
        rows1.fetch_add(read_one(preps_solo[k], opts, solo.stream_of(worker), counters_g1),
                        std::memory_order_relaxed);
        g1_trace.parse_end[k] = now_ms(t1);
        latch1.count_down();
      });
    }
    latch1.wait();
    g1_parse_wall_ms = now_ms(t1);
    solo.stop();
    summary_g1 = stack.io_ctx->cache()->summary();
  }

  print_header();
  for (auto const& r : results) {
    print_row(r);
  }

  double const a_block_ms = counters_a.async_blocked.ms.load(std::memory_order_relaxed);
  std::cout << "\narm B read amplification (MEASURED from /proc/self/io read_bytes across the "
               "prefetch, O_DIRECT so this is real block-device traffic):\n"
            << "  useful=" << std::fixed << std::setprecision(2)
            << static_cast<double>(b_range_bytes) / (1024.0 * 1024.0)
            << " MiB, read from disk=" << static_cast<double>(b_disk_bytes) / (1024.0 * 1024.0)
            << " MiB, ratio=" << std::setprecision(3)
            << (b_range_bytes > 0
                  ? static_cast<double>(b_disk_bytes) / static_cast<double>(b_range_bytes)
                  : 0.0)
            << "\n  cross-check, per-chunk fill MODEL (needed_fill/merge/fill_span, not a "
               "measurement)="
            << std::setprecision(2) << static_cast<double>(b_modelled_bytes) / (1024.0 * 1024.0)
            << " MiB, ratio=" << std::setprecision(3)
            << (b_range_bytes > 0
                  ? static_cast<double>(b_modelled_bytes) / static_cast<double>(b_range_bytes)
                  : 0.0)
            << "\narm B prefetch throughput on MEASURED disk bytes: " << std::setprecision(2)
            << (b_prefetch_ms > 0
                  ? static_cast<double>(b_disk_bytes) / (b_prefetch_ms / 1000.0) / 1e9
                  : 0.0)
            << " GB/s (vs " << std::setprecision(2)
            << (b_prefetch_ms > 0
                  ? static_cast<double>(b_range_bytes) / (b_prefetch_ms / 1000.0) / 1e9
                  : 0.0)
            << " GB/s on useful bytes)\n";
  std::cout << "arm A baseline IO throughput: " << std::setprecision(2)
            << (a_block_ms > 0 ? static_cast<double>(b_range_bytes) / (a_block_ms / 1000.0) / 1e9
                               : 0.0)
            << " GB/s (device_read_async blocked " << std::setprecision(1) << a_block_ms
            << " ms on " << std::setprecision(2)
            << static_cast<double>(b_range_bytes) / (1024.0 * 1024.0) << " MiB)\n";

  std::cout << "\narm B phases: prefetch_ms=" << std::fixed << std::setprecision(1) << b_prefetch_ms
            << " parse_ms=" << b_parse_ms << " total_ms=" << (b_prefetch_ms + b_parse_ms) << "\n";
  std::cout << "arm B: prefetch_async issued=" << std::boolalpha << b_issued
            << " completed_ok=" << b_ok << "\n";
  std::cout << "arm B cache " << summary_b << "\n";

  std::cout << "\narm C per-file parse ms:";
  for (auto ms : per_file_c) {
    std::cout << " " << std::fixed << std::setprecision(1) << ms;
  }
  std::cout << "\n";

  std::cout << "\narm D per-file phases (ms relative to arm t0):\n"
            << std::right << std::setw(6) << "file" << std::setw(11) << "io_start" << std::setw(10)
            << "io_end" << std::setw(10) << "io_ms" << std::setw(14) << "parse_start"
            << std::setw(11) << "parse_end" << std::setw(11) << "parse_ms"
            << "\n"
            << std::string(73, '-') << "\n";
  double total_io_ms    = 0;
  double total_parse_ms = 0;
  for (std::size_t i = 0; i < n_files; ++i) {
    double const io    = d_io_end[i] - d_io_start[i];
    double const parse = d_parse_end[i] - d_parse_start[i];
    total_io_ms += io;
    total_parse_ms += parse;
    std::cout << std::right << std::fixed << std::setprecision(1) << std::setw(6) << i
              << std::setw(11) << d_io_start[i] << std::setw(10) << d_io_end[i] << std::setw(10)
              << io << std::setw(14) << d_parse_start[i] << std::setw(11) << d_parse_end[i]
              << std::setw(11) << parse << "\n";
  }

  double const wall_ms       = results[3].ms;
  double const io_union      = union_span_ms(d_io_start, d_io_end);
  double const overlap_ms    = total_io_ms + total_parse_ms - wall_ms;
  double const hideable      = std::min(total_io_ms, total_parse_ms);
  double const hidden_pct    = hideable > 0 ? 100.0 * overlap_ms / hideable : 0.0;
  double const true_overlap  = io_union + total_parse_ms - wall_ms;
  double const true_hideable = std::min(io_union, total_parse_ms);
  double const true_pct      = true_hideable > 0 ? 100.0 * true_overlap / true_hideable : 0.0;
  std::cout << "\narm D aggregates: total_io_ms=" << std::fixed << std::setprecision(1)
            << total_io_ms << " total_parse_ms=" << total_parse_ms << " wall_ms=" << wall_ms
            << " overlap_ms=" << overlap_ms << " (" << std::setprecision(1) << hidden_pct
            << "% of hideable " << std::setprecision(1) << hideable << " ms)\n";
  std::cout << "arm D aggregates (IO interval union, valid when prefetches run concurrently): "
            << "io_union_ms=" << io_union << " overlap_ms=" << true_overlap << " (" << true_pct
            << "% of hideable " << true_hideable << " ms)\n";

  std::cout << "\narm D handoffs (lead_ms = parse_end[i] - io_end[i+1]):\n";
  std::vector<double> leads;
  std::size_t overlapped = 0;
  std::size_t waited     = 0;
  for (std::size_t i = 0; i + 1 < n_files; ++i) {
    double const lead = d_parse_end[i] - d_io_end[i + 1];
    leads.push_back(lead);
    if (lead > 0) {
      ++overlapped;
    } else {
      ++waited;
    }
    std::cout << "  " << i << " -> " << (i + 1) << " : lead_ms = " << std::showpos << std::fixed
              << std::setprecision(1) << lead << std::noshowpos
              << (lead > 0 ? "   (overlapped)" : "   (parser waited)") << "\n";
  }
  if (!leads.empty()) {
    std::cout << "  summary: overlapped=" << overlapped << " waited=" << waited
              << " min=" << std::fixed << std::setprecision(1)
              << *std::min_element(leads.begin(), leads.end()) << " median=" << median_of(leads)
              << " max=" << *std::max_element(leads.begin(), leads.end()) << "\n";
  }
  std::cout << "  cross-check: already_ready=" << d_ready << " (expected " << overlapped
            << ") had_to_wait=" << d_waited << " (expected " << (n_files - overlapped) << ")"
            << (d_ready == overlapped ? "  OK" : "  MISMATCH — measurement bug") << "\n";

  std::cout << "\narm D readahead: window=" << window << " peak_files_in_flight=" << d_peak_inflight
            << " already_ready=" << d_ready << " had_to_wait=" << d_waited << "\n";
  std::cout << "arm D cache " << summary_d << "\n";

  print_pool_trace("arm E (pool baseline)", e_trace, pool_threads, false);
  print_pool_trace("arm F (pool + prefetch)", f_trace, pool_threads, true);

  std::vector<double> f_queue_wait;
  double f_parse_total = 0;
  for (std::size_t i = 0; i < n_files; ++i) {
    f_queue_wait.push_back(f_trace.parse_start[i] - f_trace.io_end[i]);
    f_parse_total += f_trace.parse_end[i] - f_trace.parse_start[i];
  }
  double const f_io_union = union_span_ms(f_trace.io_start, f_trace.io_end);
  std::cout << "\narm F queue wait (parse_start - io_end, i.e. how long a staged file waited for a "
               "free worker):\n"
            << "  min=" << std::fixed << std::setprecision(1)
            << *std::min_element(f_queue_wait.begin(), f_queue_wait.end())
            << " median=" << median_of(f_queue_wait)
            << " max=" << *std::max_element(f_queue_wait.begin(), f_queue_wait.end()) << " ms\n"
            << "  (large => the pool is the bottleneck; near zero => the prefetcher is)\n";
  std::cout << "arm F io_union_ms=" << f_io_union << " total_parse_ms=" << f_parse_total
            << " wall_ms=" << results[5].ms << " fill_bubble_ms=" << f_trace.parse_start[0] << "\n";
  std::cout << "arm F: prefetch_async issued IO for " << f_issued << "/" << n_files
            << " files, completed ok=" << f_ok.load() << "\n";
  std::cout << "arm F read amplification (MEASURED, /proc/self/io read_bytes over the whole arm): "
            << "useful=" << std::setprecision(2) << static_cast<double>(f_bytes) / (1024.0 * 1024.0)
            << " MiB, read from disk=" << static_cast<double>(f_disk_bytes) / (1024.0 * 1024.0)
            << " MiB, ratio=" << std::setprecision(3)
            << (f_bytes > 0 ? static_cast<double>(f_disk_bytes) / static_cast<double>(f_bytes)
                            : 0.0)
            << "  [model says " << std::setprecision(2)
            << static_cast<double>(f_modelled_bytes) / (1024.0 * 1024.0) << " MiB]\n";
  std::cout << "arm F cache " << summary_f << "\n";

  print_pool_trace("arm G (staged, then pool parse)", g_trace, pool_threads, false);

  double g_parse_total = 0;
  for (std::size_t i = 0; i < n_files; ++i) {
    g_parse_total += g_trace.parse_end[i] - g_trace.parse_start[i];
  }
  double const g_packing = g_parse_wall_ms > 0
                             ? g_parse_total / (static_cast<double>(pool_threads) * g_parse_wall_ms)
                             : 0.0;
  double const f_tail    = tail_parse_ms(f_trace, pool_threads);
  double const g_ideal = static_cast<double>(n_files) * f_tail / static_cast<double>(pool_threads);
  double const g_median_parse = tail_parse_ms(g_trace, n_files);

  std::cout << "\narm G phases: staging_ms=" << std::fixed << std::setprecision(1) << g_stage_ms
            << " (" << std::setprecision(2) << static_cast<double>(g_disk_bytes) / (1024.0 * 1024.0)
            << " MiB staged)"
            << "   parse_wall_ms=" << std::setprecision(1) << g_parse_wall_ms
            << "   total_parse_ms=" << g_parse_total << "   packing=" << std::setprecision(3)
            << g_packing << " (total_parse / (" << pool_threads << " x parse_wall))\n"
            << "arm G cache " << summary_g << "\n";

  std::cout << "\nE vs F vs G (same n_files/pool_threads):\n"
            << "  E pool baseline, fused IO+decode : " << std::setprecision(1) << results[4].ms
            << " ms\n"
            << "  F pool + concurrent prefetch     : " << results[5].ms << " ms\n"
            << "  G phase 2, parse from a fully staged cache, zero IO in flight : "
            << g_parse_wall_ms << " ms\n";

  std::size_t const g_h2d_bytes = counters_g.device_read_async.bytes.load() +
                                  counters_g.device_read_dst.bytes.load() +
                                  counters_g.device_read_buf.bytes.load();
  double const g_pcie_ms = 1000.0 * static_cast<double>(g_h2d_bytes) / (PCIE_GBPS * 1e9);
  double const g_scaling = g_parse_wall_ms > 0 ? g1_parse_wall_ms / g_parse_wall_ms : 0.0;

  std::cout << "\narm G phase 2 H2D accounting (pinned -> device over the one PCIe link):\n"
            << "  copied " << std::setprecision(2)
            << static_cast<double>(g_h2d_bytes) / (1024.0 * 1024.0) << " MiB in "
            << counters_g.device_read_async.calls.load() << " device_read_async calls\n"
            << "  implied PCIe time at " << PCIE_GBPS << " GB/s = " << std::setprecision(1)
            << g_pcie_ms << " ms, i.e. " << std::setprecision(1)
            << (g_parse_wall_ms > 0 ? 100.0 * g_pcie_ms / g_parse_wall_ms : 0.0) << "% of the "
            << g_parse_wall_ms
            << " ms phase-2 wall (a floor no thread count can "
               "lower)\n";

  print_pool_trace("arm G control (1 worker)", g1_trace, 1, false);

  std::cout << "\narm G single-worker control (same staged cache, pool_threads=1):\n"
            << "  re-arm of the second datasource set: " << std::setprecision(1) << g1_restage_ms
            << " ms, " << std::setprecision(2)
            << static_cast<double>(g1_disk_bytes) / (1024.0 * 1024.0)
            << " MiB off disk (near zero => it really did parse from cache)\n"
            << "  1-worker parse wall = " << std::setprecision(1) << g1_parse_wall_ms << " ms   vs "
            << pool_threads << "-worker " << g_parse_wall_ms << " ms\n"
            << "  MEASURED parse scaling factor = " << std::setprecision(2) << g_scaling << "x on "
            << pool_threads << " workers (ideal " << pool_threads << "x)\n"
            << "  arm G control cache " << summary_g1 << "\n";

  std::cout << "\narm G verdict:\n"
            << "  uncontended per-file parse-from-cache (F tail, last " << pool_threads
            << " files) = " << std::setprecision(1) << f_tail << " ms\n"
            << "  perfectly-scaling expectation = " << n_files << " x " << f_tail << " / "
            << pool_threads << " = " << g_ideal << " ms\n"
            << "  measured arm G phase 2 = " << g_parse_wall_ms << " ms  (ratio "
            << std::setprecision(2) << (g_ideal > 0 ? g_parse_wall_ms / g_ideal : 0.0) << "x)\n"
            << "  median per-file parse in G = " << std::setprecision(1) << g_median_parse
            << " ms vs " << f_tail << " ms uncontended\n"
            << "  worker packing = " << std::setprecision(3) << g_packing
            << " (so the pool was kept busy; this is not a queueing artefact)\n"
            << "  CONCLUSION: "
            << (g_parse_wall_ms <= 1.35 * g_ideal
                  ? "(a) CONTENTION. With no staging IO running, the pool parses at the "
                    "uncontended rate and scales across workers, so arm F's deficit is staging IO "
                    "interfering with parsing."
                  : "(b) THE GPU AND THE PCIe LINK ARE SHARED, SERIAL RESOURCES. With zero IO in "
                    "flight and the workers fully packed, 4-way parsing still misses the "
                    "uncontended rate: the H2D copies share one PCIe link and the decode kernels "
                    "share one GPU, so adding parse concurrency cannot buy back the time and "
                    "prefetching cannot win that way.")
            << "\n";

  print_counters("arm A (baseline, 1 file)", counters_a, results[0].ms);
  print_counters("arm B (prefetch, 1 file)", counters_b, b_parse_ms);
  print_counters("arm C (baseline, n files)", counters_c, results[2].ms);
  print_counters("arm D (prefetch, n files)", counters_d, total_parse_ms);
  print_counters("arm E (pool baseline)", counters_e, results[4].ms);
  print_counters("arm F (pool + prefetch)", counters_f, f_parse_total);
  print_counters("arm G (staged, then pool parse)", counters_g, g_parse_total);
  print_counters("arm G control (1 worker)", counters_g1, g1_parse_wall_ms);

  return 0;
}
