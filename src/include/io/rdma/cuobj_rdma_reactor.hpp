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

#pragma once

#include "io/cache/types.hpp"
#include "io/details/slot_pool.hpp"
#include "io/io_request.hpp"
#include "io/rdma/rdma_client.hpp"
#include "io/types.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace sirius::io::rdma {

/// GPUDirect RDMA write-ordering: a flush before the consuming copy is needed
/// unless the platform orders NIC writes for all devices.  @p writes_ordering
/// is the CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING value
/// (NONE = 0, OWNER = 100, ALL_DEVICES = 200); whether OWNER truly needs the
/// flush for same-device consumers is validated on the rig — until then this
/// stays conservative.
[[nodiscard]] constexpr bool flush_required(int writes_ordering) noexcept
{
  constexpr int k_all_devices_ordered = 200;  // CU_..._ORDERING_ALL_DEVICES
  return writes_ordering < k_all_devices_ordered;
}

/// Pull-based transfer counters, snapshotted lock-free from the reactor's
/// atomics (the REST backend's perf-snapshot pattern).  Values accumulate for
/// the reactor's lifetime and are never reset.
struct rdma_perf_snapshot {
  uint64_t bytes_total{0};       ///< bytes successfully delivered (device + host paths)
  uint64_t requests_total{0};    ///< chunks / sync host reads processed (a retried one counts once)
  uint64_t retries_total{0};     ///< retry attempts performed (attempts beyond the first)
  uint64_t short_read_total{0};  ///< short reads observed (every occurrence, retried or terminal)
  uint64_t error_total{0};       ///< chunks / sync host reads that failed terminally
  uint64_t slot_wait_total{0};   ///< arena-slot acquisitions that had to block (0 by construction
                                 ///< while slots == workers; kept for staged-completion shapes)
  uint64_t flush_total{0};       ///< GPUDirect write flushes performed (0 while flushing is off)
  uint64_t inflight_peak{0};     ///< max chunks concurrently being processed by the worker pool
};

/// s3://bucket/key object handle resolved by @c s3_rdma_ioctx::create_io_object
/// (HEAD via the client).
class cuobj_rdma_io_object : public sirius_io_object {
 public:
  cuobj_rdma_io_object(std::string path, std::string bucket, std::string key, size_t size)
    : _path(std::move(path)), _bucket(std::move(bucket)), _key(std::move(key)), _file_size(size)
  {
  }

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] size_t size() const noexcept override { return _file_size; }

  [[nodiscard]] const std::string& bucket() const noexcept { return _bucket; }
  [[nodiscard]] const std::string& key() const noexcept { return _key; }

 private:
  std::string _path;
  std::string _bucket;
  std::string _key;
  size_t _file_size;
};

/// One transfer chunk: a slot-sized (or smaller) contiguous file range bound
/// to its final destination.  Device chunks stage through a landing-arena slot
/// before a device-to-device copy on @c stream; host chunks deliver directly.
struct cuobj_chunked_rx_request {
  std::string bucket;
  std::string key;
  size_t offset{0};
  size_t size{0};
  uint8_t* dst{nullptr};
  bool is_device{false};
  rmm::cuda_stream_view stream{};
  int device_id{-1};
  std::shared_ptr<request_manager> manager;
};

/**
 * @brief S3-over-RDMA reactor: a pool of @c max_inflight blocking workers over
 *        an @c rdma_client, delivering device reads through a per-device
 *        landing arena (cudaMalloc, slot-pooled) + a D2D copy on the request
 *        stream.
 *
 * Concurrency model: the client's @c get blocks, so the worker count IS the
 * in-flight ceiling; slot exhaustion is natural backpressure.  A chunk
 * terminates exactly once (chunk_complete / report_error) on a worker thread —
 * the read's future resolves when its last chunk releases the shared
 * @c request_manager.  With no client configured every path fails loudly with
 * a "not implemented" error (the transport-selection contract).  @c shutdown
 * drains queued work, then joins the workers; it is idempotent, and work
 * enqueued afterwards fails cleanly.
 */
class cuobj_rdma_reactor {
 public:
  struct config {
    size_t max_inflight{8};
    size_t arena_slot_size{4UL << 20};
    /// Total transfer attempts per chunk / sync host read (the first + retries).
    /// Only client transport failures and short reads retry — never CUDA work.
    size_t max_get_attempts{3};
    std::chrono::milliseconds retry_backoff_base{5};
    std::chrono::milliseconds retry_jitter{5};
  };

  /// Shared per-pool state: the effective config + the transfer client (may be
  /// null until the real data path is configured).
  class reactor_context {
   public:
    reactor_context(config cfg, std::shared_ptr<rdma_client> client)
      : _config(cfg), _client(std::move(client))
    {
    }

    [[nodiscard]] const config& cfg() const noexcept { return _config; }
    [[nodiscard]] const std::shared_ptr<rdma_client>& client() const noexcept { return _client; }

    /// Flush GPUDirect writes before the consuming D2D copy.  Off by default;
    /// set at init from the device's writes-ordering attribute
    /// (see @c flush_required) once the real data path is enabled.
    [[nodiscard]] bool flush_before_copy() const noexcept { return _flush_before_copy; }
    void set_flush_before_copy(bool value) noexcept { _flush_before_copy = value; }

   private:
    config _config;
    std::shared_ptr<rdma_client> _client;
    bool _flush_before_copy{false};
  };

  using io_object_type       = cuobj_rdma_io_object;
  using request_type         = rx_request_t<cuobj_chunked_rx_request>;
  using request_type_ptr     = std::unique_ptr<request_type>;
  using reactor_config_type  = config;
  using reactor_context_type = reactor_context;

  explicit cuobj_rdma_reactor(std::shared_ptr<reactor_context> ctx);
  ~cuobj_rdma_reactor();

  cuobj_rdma_reactor(const cuobj_rdma_reactor&)            = delete;
  cuobj_rdma_reactor& operator=(const cuobj_rdma_reactor&) = delete;

  [[nodiscard]] const config& get_config() const noexcept { return _config; }

  /// Static request builders (the templated_ioctx dispatch contract).  They
  /// assume @p file was created by this backend's create_io_object; the ioctx
  /// fail-fast overrides guarantee no request is built without a client.
  static request_type_ptr prep_host_rx_request(const config& cfg,
                                               const io_object_type& file,
                                               const io_object_segment& segment);

  static request_type_ptr prep_device_rx_request(const config& cfg,
                                                 const io_object_type& file,
                                                 uint8_t* dst,
                                                 size_t offset,
                                                 size_t size,
                                                 rmm::cuda_stream_view stream,
                                                 int device_id);

  void enqueue(request_type_ptr req);

  size_t host_read(const io_object_type& file, size_t offset, size_t size, uint8_t* dst);

  void start();
  void shutdown();
  void interrupt();

  [[nodiscard]] rdma_perf_snapshot perf_snapshot() const noexcept;

  /// Concept stub: real object creation needs the client (HEAD) and lives in
  /// @c s3_rdma_ioctx::create_io_object.  Always throws.
  static std::unique_ptr<io_object_type> create_io_object(std::string path);
  static bool supports(std::string_view path);
  static cache::prefetching_stage preferred_prefetching_stage() noexcept
  {
    return cache::prefetching_stage::none;
  }

 private:
  struct arena {
    uint8_t* base{nullptr};
    std::unique_ptr<slot_pool> pool;
    std::shared_ptr<rdma_client> registrar;  // deregisters base on teardown
    ~arena();
  };

  void worker_loop();
  void process_chunk(cuobj_chunked_rx_request& chunk);
  arena& arena_for_device(int device_id);
  [[nodiscard]] bool stopping();
  /// Bounded-retry transfer into @p dst: up to max_get_attempts client gets
  /// (+ the short-read check), backoff between attempts, abort when stopping.
  /// Counts retries/short reads; throws the last error on exhaustion.
  size_t get_with_retry(
    std::string_view bucket, std::string_view key, size_t offset, size_t size, void* dst);

  std::shared_ptr<reactor_context> _ctx;
  config _config;

  std::mutex _mtx;
  std::condition_variable _cv;
  std::condition_variable _drained_cv;
  std::deque<std::unique_ptr<cuobj_chunked_rx_request>> _queue;
  size_t _active{0};
  bool _started{false};
  bool _stopping{false};
  bool _joined{false};
  std::vector<std::thread> _workers;

  std::mutex _arena_mtx;
  std::map<int, std::unique_ptr<arena>> _arenas;

  std::atomic<uint64_t> _bytes_total{0};
  std::atomic<uint64_t> _requests_total{0};
  std::atomic<uint64_t> _retries_total{0};
  std::atomic<uint64_t> _short_read_total{0};
  std::atomic<uint64_t> _error_total{0};
  std::atomic<uint64_t> _slot_wait_total{0};
  std::atomic<uint64_t> _flush_total{0};
  std::atomic<uint64_t> _inflight_peak{0};
};

}  // namespace sirius::io::rdma
