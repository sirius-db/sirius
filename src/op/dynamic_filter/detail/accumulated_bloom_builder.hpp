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

#include "log/logging.hpp"
#include "op/dynamic_filter/complete_build_inventory.hpp"
#include "op/dynamic_filter/dynamic_filter_replica_space.hpp"

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <cucascade/error.hpp>

#include <concepts>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op {
class sirius_dynamic_bloom_filter;
}  // namespace sirius::op

namespace sirius::op::detail {

/**
 * @brief A CUDA runtime failure on the accumulation path, carrying its error code.
 */
class accumulation_cuda_error : public cucascade::cuda_error {
 public:
  /**
   * @brief Describes a failed CUDA call.
   *
   * @param code The error the call returned
   * @param kernel_launch Whether the call launched a kernel
   * @param message The failing call and the error text
   */
  accumulation_cuda_error(cudaError_t code, bool kernel_launch, std::string const& message)
    : cucascade::cuda_error{message}, _code{code}, _kernel_launch{kernel_launch}
  {
  }

  /**
   * @brief The CUDA error code.
   */
  [[nodiscard]] cudaError_t code() const noexcept { return _code; }

  /**
   * @brief True for the kernel-launch codes that the mandatory path retries
   * (`cudaErrorLaunchOutOfResources`, `cudaErrorInvalidValue`).
   */
  [[nodiscard]] bool transient_launch_failure() const noexcept
  {
    return _kernel_launch &&
           (_code == cudaErrorLaunchOutOfResources || _code == cudaErrorInvalidValue);
  }

 private:
  cudaError_t _code;
  bool _kernel_launch;
};

/**
 * @brief Thrown by `join_on_failure` when a host join failed and a listed stream still reports
 * unfinished work.
 *
 * Storage that such work may touch must be leaked rather than released. The original exception is
 * nested.
 *
 * This is defence in depth. The synchronization failures seen in practice are sticky context
 * errors, after which a stream reports that error instead of `cudaErrorNotReady`, so no test
 * reaches this path. CUDA does not promise that a failed synchronization leaves a stream idle,
 * however, and releasing storage that queued work may still write would silently corrupt whatever
 * reuses it, so the builder leaks instead and the session counts the leak.
 */
class unjoined_gpu_work : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/**
 * @brief Runs @p body; if it throws, host-synchronizes every stream in @p streams before the
 * exception continues.
 *
 * A failed synchronization is logged at ERROR and its error is cleared. If any listed stream then
 * still reports `cudaErrorNotReady`, the exception is replaced by `unjoined_gpu_work` with the
 * original nested; otherwise the original is rethrown. On success nothing waits.
 *
 * @tparam Body A callable taking no arguments
 * @param streams Streams that may run work @p body enqueued
 * @param body The enqueue sequence
 * @return Whatever @p body returns
 */
template <std::invocable Body>
decltype(auto) join_on_failure(std::span<::cuda::stream_ref const> streams, Body&& body)
{
  try {
    return std::invoke(std::forward<Body>(body));
  } catch (...) {
    bool unjoined = false;
    for (auto const stream : streams) {
      auto const synced = cudaStreamSynchronize(stream.get());
      if (synced == cudaSuccess) { continue; }
      (void)cudaGetLastError();
      unjoined = unjoined || cudaStreamQuery(stream.get()) == cudaErrorNotReady;
      (void)cudaGetLastError();
      try {
        SIRIUS_LOG_ERROR("[join_on_failure] host join after a failure failed: {}",
                         cudaGetErrorString(synced));
      } catch (...) {  // The failure being unwound is reported by the caller.
      }
    }
    if (unjoined) {
      std::throw_with_nested(
        unjoined_gpu_work{"[join_on_failure] GPU work may still run after a failed host join"});
    }
    throw;
  }
}

/**
 * @brief Private per-GPU Bloom partials of one accumulation; never visible to consumers.
 *
 * `dynamic_filter_publication_session` creates one builder when accumulation begins. Each replica
 * GPU gets a partial: one zeroed array per key and an exclusive, non-blocking stream that orders
 * every later use and every free of that GPU's arrays. Contributions insert keys on the
 * contributing task's stream and fold their completion into the partial's stream; `finish` reduces
 * the partials and publishes the union. Every member returns or throws only after the GPU work it
 * enqueued is complete or folded into a partial's stream, so storage is always released
 * stream-ordered after its last use.
 */
class accumulated_bloom_builder final {
 public:
  /**
   * @brief One key column: its ordinal in the build table and its storage type.
   */
  struct key {
    cudf::size_type ordinal;
    cudf::data_type type;
  };

  /**
   * @brief Whether keys of type @p type can be accumulated: INT32 and INT64 only.
   *
   * Narrower than `sirius_dynamic_bloom_filter::supports`: every other key type publishes only from
   * a whole build.
   */
  [[nodiscard]] static constexpr bool supports(cudf::data_type type) noexcept
  {
    return type.id() == cudf::type_id::INT32 || type.id() == cudf::type_id::INT64;
  }

  /**
   * @brief Allocates zeroed arrays on every GPU of @p targets without waiting on the host.
   *
   * Each GPU's arrays are allocated under their own lease, attached to a new exclusive stream of
   * that GPU and sized to the exact allocation charge. The calling thread must hold no allocation
   * tracker for those GPUs; a tracked caller is refused like missing capacity.
   *
   * @param keys The active keys, in publication order
   * @param geometry The shared array geometry
   * @param targets The replica GPUs; each distinct GPU receives one partial
   * @return The builder, or nullopt iff a lease was refused, after releasing everything built so
   * far
   */
  [[nodiscard]] static std::optional<accumulated_bloom_builder> try_create(
    std::vector<key> keys,
    accumulated_bloom_geometry geometry,
    std::span<dynamic_filter_replica_space const> targets);

  ~accumulated_bloom_builder();
  accumulated_bloom_builder(accumulated_bloom_builder const&)            = delete;
  accumulated_bloom_builder& operator=(accumulated_bloom_builder const&) = delete;
  accumulated_bloom_builder(accumulated_bloom_builder&&) noexcept;
  accumulated_bloom_builder& operator=(accumulated_bloom_builder&&) noexcept;

  /**
   * @brief Whether @p device holds a partial.
   */
  [[nodiscard]] bool has_partial(rmm::cuda_device_id device) const noexcept;

  /**
   * @brief Inserts @p input's key columns into @p device's partial, without a host wait or a device
   * allocation.
   *
   * Thread-safe. @p stream first waits for the partial's zero fill; the partial's stream then waits
   * for the inserts, so every later use or free of the arrays follows them.
   *
   * @pre The caller retires @p stream before releasing @p input
   * @throw std::invalid_argument if a key column is missing or has another type, before anything is
   * enqueued
   * @throw std::logic_error if @p device has no partial or @p stream belongs to another GPU, before
   * anything is enqueued
   * @throw accumulation_cuda_error if a CUDA call fails; work already enqueued is joined first
   * @param input A build batch in the certified schema
   * @param device The GPU that holds @p input
   * @param stream The contributing task's stream
   */
  void enqueue_add(cudf::table_view const& input,
                   rmm::cuda_device_id device,
                   ::cuda::stream_ref stream);

  /**
   * @brief Reduces every contributed partial into @p root's, overwrites every other partial with
   * the union, and waits on the host once.
   *
   * Moves the arrays into the returned filters: each filter holds one replica per partial GPU. The
   * chunk pipeline overlaps peer copies into a scratch buffer on @p root (on @p stream), the OR of
   * each chunk (on the root partial's stream), and the peer copies of the union back to every other
   * partial (on their streams).
   *
   * @pre Every `enqueue_add` returned; the caller is the only user of the builder; @p stream
   * belongs to @p root; @p stream is not tracked by @p root's allocator
   * @throw std::logic_error if @p root has no partial or @p stream is tracked, before anything is
   * enqueued
   * @throw accumulation_cuda_error if a CUDA call fails; work already enqueued is joined first
   * @param root The GPU that reduces; the calling thread's current device
   * @param stream An exclusive stream of @p root
   * @return One filter per key in key order, or nullopt iff the scratch lease was refused (nothing
   * was enqueued)
   */
  [[nodiscard]] std::optional<std::vector<std::shared_ptr<sirius_dynamic_bloom_filter>>> finish(
    rmm::cuda_device_id root, ::cuda::stream_ref stream);

  /**
   * @brief True iff releasing this builder's storage now credits no allocation tracker.
   *
   * cuCascade credits a free to the calling thread's (or stream's) tracker for the freeing
   * allocator. A builder must therefore not be destroyed while the thread tracks any partial's
   * allocator. Never throws; an unexpected failure reports false.
   */
  [[nodiscard]] bool releasable_here() const noexcept;

  /**
   * @brief The state of one published replica, copied for verification.
   */
  struct replica_contents {
    std::vector<std::byte> bytes;  ///< The replica's filter array
    bool stream_idle;              ///< Whether the replica's stream had no pending work at the call
  };

  /**
   * @brief Copies @p filter's accumulated replica on @p device to the host.
   *
   * Queries the replica's stream before enqueueing the copy on it, then waits for the copy on the
   * host. Meant for tests and diagnostics; publication never calls it.
   *
   * @pre @p filter was published: no publishing job still owns it
   * @param filter An accumulated filter returned by `finish`
   * @param device The GPU whose replica is copied
   * @return The replica's contents, or nullopt if @p filter has no accumulated replica on @p device
   */
  [[nodiscard]] static std::optional<replica_contents> inspect_replica(
    sirius_dynamic_bloom_filter const& filter, rmm::cuda_device_id device);

 private:
  struct impl;
  explicit accumulated_bloom_builder(std::unique_ptr<impl> state) noexcept;

  /**
   * @brief Moves the partial arrays into one filter per key, in key order, with one replica per
   * partial.
   *
   * @param root_device The GPU that reduced the partials, recorded as each filter's source device
   * @return One filter per key
   */
  [[nodiscard]] std::vector<std::shared_ptr<sirius_dynamic_bloom_filter>> make_filters(
    int root_device);

  std::unique_ptr<impl> _impl;
};

}  // namespace sirius::op::detail
