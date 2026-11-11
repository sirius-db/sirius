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

#include "io/gds/gds_ioctx.hpp"

#include "log/logging.hpp"

#include <kvikio/cufile/driver.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/shim/cufile.hpp>
#include <kvikio/utils.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda_runtime.h>

#include <cucascade/io/io_utils.hpp>
#include <cucascade/io/kvikio/kvikio_context.hpp>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>

namespace sirius::io::gds {

namespace {

using cucascade::io::io_object;
using cucascade::io::kvikio_io_object;
using cucascade::io::prepared_io_slice;

/// How long reads must stay slow before the context gives up on the path.
constexpr std::int64_t k_slow_read_hold_ns = 2'000'000'000;

/// Device bytes read before the share of POSIX reads is judged.
constexpr std::uint64_t k_fallback_min_bytes = 1ULL << 30;

[[nodiscard]] std::int64_t now_ns() noexcept
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

kvikio_io_object const& as_local(io_object const& obj)
{
  auto const* typed = dynamic_cast<kvikio_io_object const*>(&obj);
  if (typed == nullptr) {
    throw std::invalid_argument("gds_ioctx: io_object '" + obj.object_path() +
                                "' was not created by this backend");
  }
  return *typed;
}

[[nodiscard]] std::size_t clamp_to_object(io_object const& obj,
                                          std::size_t offset,
                                          std::size_t size) noexcept
{
  return obj.size() > offset ? std::min(size, obj.size() - offset) : 0;
}

std::size_t posix_read(int fd, std::uint8_t* dst, std::size_t size, std::size_t offset)
{
  std::size_t done = 0;
  while (done < size) {
    auto const n = ::pread(fd, dst + done, size - done, static_cast<off_t>(offset + done));
    if (n < 0) {
      if (errno == EINTR) { continue; }
      throw std::system_error(errno, std::generic_category(), "gds_ioctx: pread");
    }
    if (n == 0) { break; }
    done += static_cast<std::size_t>(n);
  }
  return done;
}

/// CPUs the calling thread may run on; the read threads inherit them.
[[nodiscard]] int allowed_cpus() noexcept
{
  cpu_set_t set;
  return sched_getaffinity(0, sizeof(set), &set) == 0 ? CPU_COUNT(&set) : 0;
}

void notify(prepared_io_slice& slice, bool success) noexcept
{
  if (slice.on_complete != nullptr) { (*slice.on_complete)(slice.h_buffer.fragments(), success); }
}

}  // namespace

/// Where a device destination sits in its CUDA allocation.
struct device_target {
  void* base{nullptr};
  std::size_t offset{0};
  CUcontext context{};
};

// kvikio::FileHandle::pread hands cuFile the allocation's base and an offset.
// The lookup needs a current CUDA context, which the submitting thread has and
// the pool threads do not. CUDA cannot name the base of memory from a
// stream-ordered pool (cuda_async_memory_resource); cuFile then gets the
// pointer itself, which is all an unregistered buffer needs.
static device_target locate(void* dst)
{
  device_target target;
  try {
    // get_alloc_info reads the context it is given; it does not fill it in.
    target.context   = kvikio::get_context_from_pointer(dst);
    auto const alloc = kvikio::get_alloc_info(dst, &target.context);
    target.base      = std::get<0>(alloc);
    target.offset    = std::get<2>(alloc);
  } catch (...) {
    target.base   = dst;
    target.offset = 0;
    if (kvikio::cudaAPI::instance().CtxGetCurrent(&target.context) != CUDA_SUCCESS) {
      throw std::runtime_error("gds_ioctx: no current CUDA context for a device read");
    }
  }
  return target;
}

struct gds_ioctx::request {
  std::shared_ptr<io_object const> object;
  std::vector<prepared_io_slice> slices;
  std::vector<device_target> targets;
  cucascade::exec::promise<std::size_t> promise;
  std::atomic<std::size_t> pending{0};
  std::atomic<std::size_t> bytes{0};
  std::atomic<bool> settled{false};
  std::mutex error_mutex;
  std::exception_ptr error;
};

gds_ioctx::gds_ioctx(gds_config config, options opts) : _config(config), _options(std::move(opts))
{
  if (_config.num_threads == 0) { throw std::invalid_argument("gds: num_threads must be > 0"); }
  if (_config.task_size == 0) { throw std::invalid_argument("gds: task_size must be > 0"); }
  if (!_options.on_fatal_stall) {
    _options.on_fatal_stall = [] {
      SIRIUS_LOG_ERROR("[gds] a read is stuck; exiting with status 70");
      // The log may sit on the filesystem that stopped answering: the flush gets
      // two seconds, and the exit does not wait for it.
      std::thread([] { sirius::log::get_sink()->flush(); }).detach();
      std::this_thread::sleep_for(std::chrono::seconds(2));
      std::_Exit(70);
    };
  }

  if (_options.compat_mode == kvikio::CompatMode::OFF) {
    // kvikIO reports GDS as unavailable until the cuFile driver is open.
    kvikio::cuFileAPI::instance().driver_open();
    kvikio::DriverProperties props;
    if (!props.is_gds_available()) {
      throw std::runtime_error(
        "[gds] cuFile does not report GPUDirect Storage as available on this machine");
    }
    bool const may_fall_back = props.get_nvfs_allow_compat_mode();
    if (may_fall_back && _config.require) {
      throw std::runtime_error(
        "[gds] cuFile may fall back to POSIX reads without an error (allow_compat_mode is "
        "true). Point CUFILE_ENV_PATH_JSON at a cufile.json with \"allow_compat_mode\": false, "
        "or set scan_manager.gds.require: false");
    }
    _cufile_counters = enable_cufile_counters();
    if (auto counters = read_cufile_counters()) {
      _cufile_baseline = *counters;
    } else {
      _cufile_counters = false;
    }
    SIRIUS_LOG_INFO(
      "[gds] cuFile ready: nvidia-fs {}.{}, posix_fallback_allowed={}, cufile_counters={}, "
      "num_threads={}, task_size={}, read_timeout_s={}, slow_read_ms_per_mib={}, require={}, "
      "cpus_allowed={}",
      props.get_nvfs_major_version(),
      props.get_nvfs_minor_version(),
      may_fall_back,
      _cufile_counters,
      _config.num_threads,
      _config.task_size,
      _config.read_timeout.count(),
      _config.slow_read_ms_per_mib,
      _config.require,
      allowed_cpus());
  }

  _read_started = std::make_unique<std::atomic<std::int64_t>[]>(_config.num_threads);
  _reading      = std::make_unique<reading[]>(_config.num_threads);
  _workers.reserve(_config.num_threads);
  try {
    for (std::size_t i = 0; i < _config.num_threads; ++i) {
      _workers.emplace_back([this, i] { worker_loop(i); });
    }
    _watchdog = std::thread([this] { watchdog_loop(); });
  } catch (...) {
    // A joinable thread left behind would terminate the process on unwinding.
    shutdown();
    throw;
  }
}

gds_ioctx::~gds_ioctx()
{
  this->pre_destroy();
  shutdown();
}

void gds_ioctx::shutdown() noexcept
{
  {
    std::lock_guard lock{_queue_mutex};
    if (_stop) { return; }
    _stop = true;
  }
  _queue_cv.notify_all();
  // The watchdog keeps running while the workers are joined: a worker stuck in
  // a read would otherwise block this join for good, with nothing left to end
  // the process.
  for (auto& worker : _workers) {
    if (worker.joinable()) { worker.join(); }
  }
  {
    std::lock_guard lock{_queue_mutex};
    _watchdog_stop = true;
  }
  _watchdog_cv.notify_all();
  if (_watchdog.joinable()) { _watchdog.join(); }
}

bool gds_ioctx::supports(std::string_view path) const noexcept
{
  return path.find("://") == std::string_view::npos;
}

std::vector<cucascade::io::byte_range> gds_ioctx::align_and_coalesce(
  std::span<const cucascade::io::byte_range> ranges,
  std::optional<std::size_t> /*alignment*/) const noexcept
{
  try {
    return ::io_utils::align_ranges(ranges, 0);
  } catch (...) {
    return {};
  }
}

std::shared_ptr<io_object> gds_ioctx::create_io_object(std::string path)
{
  if (!supports(path)) {
    throw std::runtime_error("gds_ioctx: '" + path + "' is not a local file");
  }
  kvikio::FileHandle handle{path, "r", kvikio::FileHandle::m644, _options.compat_mode};
  auto const size = handle.nbytes();
  return std::make_shared<kvikio_io_object>(std::move(path), std::move(handle), size);
}

std::size_t gds_ioctx::host_read_io(const io_object& obj,
                                    std::size_t offset,
                                    std::size_t size,
                                    std::uint8_t* dst)
{
  size         = clamp_to_object(obj, offset, size);
  auto const n = posix_read(as_local(obj).handle().fd(), dst, size, offset);
  _host_bytes.fetch_add(n, std::memory_order_relaxed);
  return n;
}

std::exception_ptr gds_ioctx::failure() const
{
  switch (_state.load()) {
    case state::ok: return nullptr;
    case state::stalled:
      return std::make_exception_ptr(
        read_stalled_error("GDS read stalled: a cuFile read did not return within " +
                           std::to_string(_config.read_timeout.count()) + " s"));
    case state::slow:
      return std::make_exception_ptr(read_stalled_error(
        "GDS reads slowed down: they stayed above " + std::to_string(_config.slow_read_ms_per_mib) +
        " ms per MiB; no further GDS reads are issued"));
    case state::posix_fallback:
      return std::make_exception_ptr(
        posix_fallback_error("GDS required, but cuFile served a read through its POSIX fallback"));
  }
  return nullptr;
}

void gds_ioctx::settle(std::shared_ptr<request> const& req, std::exception_ptr error) noexcept
{
  if (req->settled.exchange(true)) { return; }
  {
    std::lock_guard lock{_live_mutex};
    _live.erase(req);
  }
  for (auto& slice : req->slices) {
    notify(slice, error == nullptr);
  }
  if (error) {
    req->promise.set_exception(std::move(error));
  } else {
    req->promise.set_value(req->bytes.load());
  }
}

void gds_ioctx::run_piece(std::shared_ptr<request> const& req,
                          std::size_t slice_index,
                          std::size_t piece_offset,
                          std::size_t piece_size,
                          std::int64_t queued_at,
                          std::size_t worker)
{
  std::exception_ptr error = failure();
  if (!error && !req->settled.load()) {
    try {
      auto& slice        = req->slices[slice_index];
      auto const& object = as_local(*req->object);
      auto const offset  = slice.offset() + piece_offset;
      std::size_t done   = 0;
      _reading[worker].object.store(req->object.get());
      _reading[worker].offset.store(offset);
      _reading[worker].size.store(piece_size);
      _reading[worker].device.store(slice.has_device_request());
      if (slice.has_device_request()) {
        auto const started = now_ns();
        _read_started[worker].store(started);
        if (_options.reader) {
          done = _options.reader(
            object.handle(), slice.d_buffer.data + piece_offset, piece_size, offset);
        } else {
          // What kvikio::FileHandle::pread does for each of its tasks.
          auto const& target = req->targets[slice_index];
          kvikio::PushAndPopContext const guard{target.context};
          done = object.handle().read(
            target.base, piece_size, offset, target.offset + piece_offset, false);
        }
        _read_started[worker].store(0);
        _device_read_ns.fetch_add(now_ns() - started, std::memory_order_relaxed);
        _device_wait_ns.fetch_add(started - queued_at, std::memory_order_relaxed);
        _device_bytes.fetch_add(done, std::memory_order_relaxed);
        // A read has a fixed cost: small ones are charged as one MiB.
        _device_charged_bytes.fetch_add(std::max<std::size_t>(done, 1 << 20),
                                        std::memory_order_relaxed);
        _device_reads.fetch_add(1, std::memory_order_relaxed);
      } else {
        auto* host = std::get<std::uint8_t*>(slice.h_buffer.buffer);
        _read_started[worker].store(now_ns());
        done = posix_read(object.handle().fd(), host + piece_offset, piece_size, offset);
        _read_started[worker].store(0);
        _host_bytes.fetch_add(done, std::memory_order_relaxed);
      }
      if (done != piece_size) { throw std::runtime_error("gds_ioctx: short read"); }
      req->bytes.fetch_add(done);
    } catch (...) {
      _read_started[worker].store(0);
      error = std::current_exception();
    }
  }
  if (error) {
    std::lock_guard lock{req->error_mutex};
    if (!req->error) { req->error = error; }
  }
  if (req->pending.fetch_sub(1) == 1) {
    std::exception_ptr first;
    {
      std::lock_guard lock{req->error_mutex};
      first = req->error;
    }
    settle(req, first);
  }
}

cucascade::exec::semi_future<std::size_t> gds_ioctx::mixed_readv_async_io(
  const io_object& obj, std::vector<prepared_io_slice>&& slices) noexcept
{
  auto req              = std::make_shared<request>();
  req->slices           = std::move(slices);
  auto future           = req->promise.get_semi_future();
  std::size_t submitted = 0;
  try {
    if (auto error = failure()) { std::rethrow_exception(error); }
    req->object = obj.shared_from_this();

    // cuFileRead is not ordered on a CUDA stream, and a destination from a
    // stream-ordered pool may still be in use by work queued on its stream.
    cudaStream_t drained = nullptr;
    bool any_drained     = false;
    std::size_t pieces   = 0;
    req->targets.resize(req->slices.size());
    for (std::size_t index = 0; index < req->slices.size(); ++index) {
      auto& slice    = req->slices[index];
      slice.rng.size = clamp_to_object(obj, slice.offset(), slice.size());
      if (slice.size() == 0) { continue; }
      if (slice.is_fragmented() || (slice.has_device_request() && slice.is_contiguous())) {
        throw std::invalid_argument("gds_ioctx: cache-backed slices are not supported");
      }
      if (!slice.has_device_request() && !slice.is_contiguous()) {
        throw std::invalid_argument("gds_ioctx: prepared slice has no destination");
      }
      if (slice.has_device_request()) {
        std::optional<rmm::cuda_set_device_raii> device;
        if (slice.d_buffer.device_id >= 0) {
          device.emplace(rmm::cuda_device_id{slice.d_buffer.device_id});
        }
        if (!_options.reader) { req->targets[index] = locate(slice.d_buffer.data); }
        auto const stream = slice.d_buffer.stream.get();
        if (!any_drained || stream != drained) {
          if (auto const status = cudaStreamSynchronize(stream); status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
          }
          drained     = stream;
          any_drained = true;
        }
      }
      pieces += (slice.size() + _config.task_size - 1) / _config.task_size;
    }

    if (pieces == 0) {
      settle(req, nullptr);
      return future;
    }
    req->pending.store(pieces);
    {
      std::lock_guard lock{_live_mutex};
      _live.insert(req);
    }
    for (std::size_t index = 0; index < req->slices.size(); ++index) {
      auto const size = req->slices[index].size();
      for (std::size_t offset = 0; offset < size; offset += _config.task_size) {
        auto const piece = std::min(_config.task_size, size - offset);
        submit([this, req, index, offset, piece, queued_at = now_ns()](std::size_t worker) {
          run_piece(req, index, offset, piece, queued_at, worker);
        });
        ++submitted;
      }
    }
  } catch (...) {
    auto const error   = std::current_exception();
    auto const pending = req->pending.load();
    if (submitted == 0) {
      settle(req, error);
    } else {
      // Pieces already queued still write into the caller's buffers: the request
      // settles when the last of them returns, not now.
      {
        std::lock_guard lock{req->error_mutex};
        if (!req->error) { req->error = error; }
      }
      auto const missing = pending - submitted;
      if (req->pending.fetch_sub(missing) == missing) { settle(req, error); }
    }
  }
  return future;
}

void gds_ioctx::submit(std::function<void(std::size_t)> task)
{
  {
    std::lock_guard lock{_queue_mutex};
    _queue.push_back(std::move(task));
  }
  _queue_cv.notify_one();
}

void gds_ioctx::worker_loop(std::size_t worker)
{
  for (;;) {
    std::function<void(std::size_t)> task;
    {
      std::unique_lock lock{_queue_mutex};
      _queue_cv.wait(lock, [this] { return _stop || !_queue.empty(); });
      if (_queue.empty()) { return; }
      task = std::move(_queue.front());
      _queue.pop_front();
    }
    task(worker);
  }
}

void gds_ioctx::watchdog_loop()
{
  auto const timeout_ns =
    std::chrono::duration_cast<std::chrono::nanoseconds>(_config.read_timeout).count();
  bool fatal_reported        = false;
  std::uint64_t seen_reads   = 0;
  std::uint64_t seen_read_ns = 0;
  std::uint64_t seen_charged = 0;
  std::int64_t slow_since    = 0;

  std::unique_lock lock{_queue_mutex};
  while (
    !_watchdog_cv.wait_for(lock, _options.watchdog_period, [this] { return _watchdog_stop; })) {
    lock.unlock();

    auto const now      = now_ns();
    std::int64_t oldest = 0;
    for (std::size_t i = 0; i < _config.num_threads; ++i) {
      auto const started = _read_started[i].load();
      if (started != 0 && (oldest == 0 || started < oldest)) { oldest = started; }
    }

    auto current = _state.load();
    if (current == state::ok && _config.require && _cufile_counters) {
      // cuFile serves reads of a few KiB (a tiny file, a file's tail) through POSIX
      // even under GDS, so a POSIX read alone proves nothing. A real fallback sends
      // every read that way.
      if (auto counters = read_cufile_counters()) {
        auto const delta = *counters - _cufile_baseline;
        if (_device_bytes.load() >= k_fallback_min_bytes &&
            delta.posix_reads * 2 > delta.total_reads) {
          _state.store(current = state::posix_fallback);
          SIRIUS_LOG_ERROR(
            "[gds] cuFile served {} of {} reads through POSIX: it is not reading through GDS. "
            "Failing reads.",
            delta.posix_reads,
            delta.total_reads);
        }
      }
    }
    // Time per MiB of the reads that finished since the last pass. Read time
    // grows with read size, so a limit per read would be wrong for one of small
    // or large reads.
    auto const reads   = _device_reads.load();
    auto const read_ns = _device_read_ns.load();
    auto const charged = _device_charged_bytes.load();
    if (_config.slow_read_ms_per_mib > 0 && reads - seen_reads >= _options.slow_read_min_reads) {
      auto const ms_per_mib = static_cast<double>(read_ns - seen_read_ns) / 1e6 /
                              (static_cast<double>(charged - seen_charged) / (1 << 20));
      if (ms_per_mib <= _config.slow_read_ms_per_mib) {
        slow_since = 0;
      } else if (slow_since == 0) {
        slow_since = now;
      } else if (current == state::ok && now - slow_since >= k_slow_read_hold_ns) {
        _state.store(current = state::slow);
        SIRIUS_LOG_ERROR(
          "[gds] cuFile reads take {:.1f} ms per MiB (limit {:.1f}): the storage path is "
          "degrading. Failing further GDS reads.",
          ms_per_mib,
          _config.slow_read_ms_per_mib);
      }
      seen_reads   = reads;
      seen_read_ns = read_ns;
      seen_charged = charged;
    }
    if (current != state::stalled && oldest != 0 && now - oldest > timeout_ns) {
      _state.store(current = state::stalled);
      SIRIUS_LOG_ERROR(
        "[gds] a read has been in flight for {} s (limit {} s); ending the process.",
        (now - oldest) / 1'000'000'000,
        _config.read_timeout.count());
      // A request cannot be settled, and so cannot release its object, without
      // this mutex: the objects named below stay alive while it is held.
      std::lock_guard live_lock{_live_mutex};
      for (std::size_t i = 0; i < _config.num_threads; ++i) {
        auto const started = _read_started[i].load();
        auto const* object = _reading[i].object.load();
        // Reads younger than a second are healthy ones caught in passing.
        if (started == 0 || object == nullptr || now - started < 1'000'000'000) { continue; }
        SIRIUS_LOG_ERROR("[gds] in flight: worker {} {} read for {:.1f} s, {} bytes at {} of {}",
                         i,
                         _reading[i].device.load() ? "cuFile" : "host",
                         static_cast<double>(now - started) / 1e9,
                         _reading[i].size.load(),
                         _reading[i].offset.load(),
                         object->object_path());
      }
    }
    if (current == state::stalled && !fatal_reported) {
      // A stuck cuFile read cannot be cancelled and still owns its destination.
      // Failing its request would let the caller free that buffer under the
      // read: on GPFS this crashed the rank inside cuFile, and the launcher then
      // restarted it onto the same path. So the process ends here, before any
      // request is settled; the default handler does not return.
      fatal_reported = true;
      _options.on_fatal_stall();
      // Only a test's handler returns. Its reader holds no real buffer.
      std::vector<std::shared_ptr<request>> live;
      {
        std::lock_guard live_lock{_live_mutex};
        live.assign(_live.begin(), _live.end());
      }
      for (auto const& req : live) {
        settle(req, failure());
      }
    }

    lock.lock();
  }
}

gds_ioctx::io_counters gds_ioctx::counters() const noexcept
{
  return {_device_bytes.load(),
          _device_reads.load(),
          _host_bytes.load(),
          _device_read_ns.load(),
          _device_wait_ns.load()};
}

std::optional<cufile_counters> gds_ioctx::cufile() const noexcept
{
  if (!_cufile_counters) { return std::nullopt; }
  auto counters = read_cufile_counters();
  if (!counters) { return std::nullopt; }
  return *counters - _cufile_baseline;
}

}  // namespace sirius::io::gds
