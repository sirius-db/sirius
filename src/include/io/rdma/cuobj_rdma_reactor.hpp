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
#include "io/rdma/rdma_admission_gate.hpp"
#include "io/rdma/rdma_client.hpp"
#include "io/types.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace sirius::io::rdma {

/// GPUDirect RDMA write-ordering: a flush before the consuming copy is needed
/// unless the platform orders NIC writes for all devices.  @p writes_ordering
/// is the CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING value
/// (NONE = 0, OWNER = 100, ALL_DEVICES = 200).  Whether OWNER suffices for
/// same-device consumers has not been validated on hardware, so OWNER is
/// treated as requiring the flush.
[[nodiscard]] constexpr bool flush_required(int writes_ordering) noexcept
{
  constexpr int k_all_devices_ordered = 200;  // CU_..._ORDERING_ALL_DEVICES
  return writes_ordering < k_all_devices_ordered;
}

/// Default @c cuda_delivery_ops::fatal_hook: raw stderr writes, then return
/// (the wrapper terminates).
void default_fatal_hook(const char* what, cudaError_t rc) noexcept;

/// Sticky/context-fatal CUDA codes: the single source of truth for
/// classifying CUDA errors before the delivery boundary (safety contract:
/// experimental/s3-rdma-transport-design.md, Section 3).  `is_context_fatal`
/// scans this array and the parameterized death tests iterate it, so dropping
/// a code fails a test.  A code belongs here iff NVIDIA documentation demands
/// process terminate/relaunch, declares the context/device unusable, or
/// requires a GPU reset / node reboot.
inline constexpr auto k_sticky_context_fatal_codes = std::to_array<cudaError_t>({
  cudaErrorECCUncorrectable,
  cudaErrorNvlinkUncorrectable,
#if CUDART_VERSION >= 12080
  cudaErrorContained,
#endif
  cudaErrorIllegalAddress,
  cudaErrorLaunchTimeout,
  cudaErrorAssert,
  cudaErrorHardwareStackError,
  cudaErrorIllegalInstruction,
  cudaErrorMisalignedAddress,
  cudaErrorInvalidAddressSpace,
  cudaErrorInvalidPc,
  cudaErrorLaunchFailure,
#if CUDART_VERSION >= 12080
  cudaErrorTensorMemoryLeak,
#endif
  cudaErrorMpsClientTerminated,
  cudaErrorExternalDevice,
});

/// CUDA delivery seam: every CUDA call on the device-chunk delivery path goes
/// through these, so tests can inject failures without faking CUDA side
/// effects.  Defaults are the real CUDA runtime entry points; the indirection
/// costs one std::function call per op against a 100+ µs blocking GET.
/// Injected at construction only (via the @c s3_rdma_ioctx constructor);
/// there is no runtime setter, so workers never race a swap.
struct cuda_delivery_ops {
  // Lambdas rather than &function: several runtime entry points carry C++
  // overload sets in cuda_runtime.h, so a bare address is ambiguous.
  std::function<cudaError_t(cudaEvent_t*, unsigned int)> event_create =
    [](cudaEvent_t* event, unsigned int flags) { return cudaEventCreateWithFlags(event, flags); };
  std::function<cudaError_t(cudaEvent_t, cudaStream_t)> event_record =
    [](cudaEvent_t event, cudaStream_t stream) { return cudaEventRecord(event, stream); };
  std::function<cudaError_t(cudaEvent_t)> event_synchronize = [](cudaEvent_t event) {
    return cudaEventSynchronize(event);
  };
  std::function<cudaError_t(cudaEvent_t)> event_destroy = [](cudaEvent_t event) {
    return cudaEventDestroy(event);
  };
  std::function<cudaError_t(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t)>
    memcpy_async =
      [](void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
        return cudaMemcpyAsync(dst, src, count, kind, stream);
      };
  /// GPUDirect write-visibility flush (pre-boundary leg; one call per exact
  /// completion when the platform's ordering attribute requires it).
  std::function<cudaError_t()> flush = [] {
    return cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                              cudaFlushGPUDirectRDMAWritesToOwner);
  };
  /// Stream-capture probe (pre-boundary leg; runs BEFORE the RDMA GET so a
  /// doomed request makes no remote side effect).
  std::function<cudaError_t(cudaStream_t, cudaStreamCaptureStatus*)> stream_capture_query =
    [](cudaStream_t stream, cudaStreamCaptureStatus* status) {
      return cudaStreamIsCapturing(stream, status);
    };
  /// Delivery-fatal diagnostic sink.  Reached ONLY through @c invoke_fatal;
  /// the process dies whether this returns or throws.  The default performs
  /// raw writes only — no allocation, no formatting, no throwing.
  std::function<void(const char*, cudaError_t)> fatal_hook = default_fatal_hook;
};

/// The only entry to the fatal hook: noexcept and non-returning.  It invokes
/// the hook, swallows anything the hook throws, and calls std::terminate().
/// No stack unwinding, no slot release, and no future resolution can follow.
[[noreturn]] void invoke_fatal(const cuda_delivery_ops& ops,
                               const char* what,
                               cudaError_t rc) noexcept;

/// Throws std::invalid_argument when any member is null — every op must stay
/// callable (the defaults above; partial injections keep the rest real).
void validate(const cuda_delivery_ops& ops);

/// Pull-based transfer counters, snapshotted lock-free from the reactor's
/// atomics (the REST backend's perf-snapshot pattern).  Values accumulate for
/// the reactor's lifetime and are never reset.
struct rdma_perf_snapshot {
  uint64_t bytes_total{0};     ///< bytes successfully delivered (device + host paths)
  uint64_t requests_total{0};  ///< logical read requests admitted (one per envelope, not per chunk)
  uint64_t retries_total{0};   ///< always 0: a transport failure fail-stops instead of retrying
  uint64_t short_read_total{0};  ///< short reads observed (each one is terminal)
  uint64_t error_total{0};       ///< chunks / sync host reads that failed terminally
  uint64_t slot_wait_total{0};   ///< arena-slot acquisitions that had to block (0 by construction
                                 ///< while slots == workers; kept for staged-completion shapes)
  uint64_t flush_total{0};       ///< GPUDirect write flushes performed (0 while flushing is off)
  uint64_t inflight_peak{0};     ///< max chunks concurrently being processed by the worker pool
  uint64_t envelope_wait_total{0};     ///< submits that blocked at the envelope-queue cap
  uint64_t envelope_wait_ns_total{0};  ///< total nanoseconds submitters spent blocked at the cap
  uint64_t envelope_depth_peak{0};     ///< max queued envelopes (logical requests)
  uint64_t slots_in_use_peak{0};       ///< max landing-arena slots concurrently held
  uint64_t fail_stop_total{0};         ///< fail-stop transitions (exactly-once per reactor)
  uint64_t arena_leak_total{0};        ///< arenas made non-freeable by a fail-stop (leaked at
                                       ///< teardown: no deregister, no free)
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

/**
 * @brief S3-over-RDMA reactor: a pool of @c max_inflight blocking workers over
 *        an @c rdma_client, delivering device reads through a per-device
 *        landing arena (cudaMalloc, slot-pooled) + a D2D copy on the request
 *        stream.
 *
 * Concurrency model: intake goes through the context's @c admission_gate —
 * one envelope per logical read request, bounded by @c queue_cap; workers
 * claim chunks lazily from the front envelope.  The client's @c get blocks,
 * so the worker count IS the in-flight ceiling; slot exhaustion is natural
 * backpressure.  A chunk terminates exactly once (chunk_complete /
 * report_error) on a worker thread — the read's future resolves when its last
 * chunk releases the shared @c request_manager.  A transport failure (thrown
 * GET or short read) is one-shot: it fail-stops the gate, marks every landing
 * arena non-freeable, and error-completes queued work; there is no retry and
 * no fallback.  With no client configured every path fails loudly with a
 * "not implemented" error.  @c shutdown closes admission, error-completes
 * unissued work, waits for issued work to publish, then joins the workers;
 * concurrent callers block until the elected joiner finishes.
 */
class cuobj_rdma_reactor {
 public:
  struct config {
    size_t max_inflight{8};
    size_t arena_slot_size{4UL << 20};
    /// Envelope-queue bound (logical requests).  Unset derives
    /// 4 x max_inflight after sanitizing; an explicit zero is rejected.
    std::optional<size_t> queue_cap{};
  };

  /// Shared per-pool state: the sanitized config + the transfer client (may be
  /// null until the real data path is configured) + the CUDA delivery seam +
  /// the admission gate (all immutable bindings after construction).  Config
  /// errors surface here, at construction.
  class reactor_context {
   public:
    reactor_context(config cfg,
                    std::shared_ptr<rdma_client> client,
                    cuda_delivery_ops delivery = {});

    [[nodiscard]] const config& cfg() const noexcept { return _config; }
    [[nodiscard]] const std::shared_ptr<rdma_client>& client() const noexcept { return _client; }
    [[nodiscard]] const cuda_delivery_ops& delivery_ops() const noexcept { return _delivery; }
    [[nodiscard]] admission_gate& gate() noexcept { return _gate; }
    [[nodiscard]] const admission_gate& gate() const noexcept { return _gate; }

    /// Flush GPUDirect writes before the consuming D2D copy.  Off by default;
    /// set at init from the device's writes-ordering attribute
    /// (see @c flush_required) once the real data path is enabled.
    [[nodiscard]] bool flush_before_copy() const noexcept { return _flush_before_copy; }
    void set_flush_before_copy(bool value) noexcept { _flush_before_copy = value; }

   private:
    config _config;
    std::shared_ptr<rdma_client> _client;
    cuda_delivery_ops _delivery;
    admission_gate _gate;
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
    bool leaked{false};                      // non-freeable flag, set for every arena by the
                                             // fail-stop marker: the destructor then skips
                                             // deregister + cudaFree, because touching
                                             // registered memory under an un-quiesced NIC is
                                             // a use-after-free.
    ~arena();
  };

  void worker_loop();
  /// One chunk end to end: arena/slot resolution, the pre-GET checks, the
  /// one-shot client GET under the get permit, then the CUDA delivery
  /// boundary.  A transport failure fail-stops the gate; the failing chunk is
  /// error-reported before its guard or permit releases.
  void process_claimed(admission_gate::claimed_chunk claimed_arg);
  /// Single client GET + exact-completion check.  Sets @p transport_failure
  /// before rethrowing so the caller can tell an RDMA failure (fail-stop)
  /// from a CUDA-side one (error-complete only).
  size_t one_shot_get(const cuobj_chunked_rx_request& chunk, void* dst, bool& transport_failure);
  arena& arena_for_device(int device_id);
  /// The gate's whole-arena marker: flips every arena non-freeable and counts
  /// each, called under the held gate mutex (lock order gate -> arena).
  static void mark_arenas_non_freeable(void* opaque) noexcept;

  std::shared_ptr<reactor_context> _ctx;
  config _config;

  std::mutex _lifecycle_mtx;
  std::condition_variable _joined_cv;
  bool _started{false};
  bool _closing{false};
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
  std::atomic<uint64_t> _inflight{0};
  std::atomic<uint64_t> _inflight_peak{0};
  std::atomic<uint64_t> _slots_in_use{0};
  std::atomic<uint64_t> _slots_in_use_peak{0};
  std::atomic<uint64_t> _arena_leak_total{0};
};

/// Sanitized copy of @p cfg: zero worker count and slot size fall back to
/// their minimum viable values, and the envelope cap resolves to an explicit
/// positive value — 4 x max_inflight when unset (overflow-checked), rejected
/// when explicitly zero.
[[nodiscard]] cuobj_rdma_reactor::config sanitized(cuobj_rdma_reactor::config cfg);

}  // namespace sirius::io::rdma
