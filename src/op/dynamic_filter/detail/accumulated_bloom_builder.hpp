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

#include "helper/cuda_launch_error.hpp"
#include "log/logging.hpp"
#include "op/dynamic_filter/complete_build_inventory.hpp"
#include "op/dynamic_filter/dynamic_filter_replica_space.hpp"
#include "sirius/exception.hpp"

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace sirius::op {
class sirius_dynamic_bloom_filter;
}  // namespace sirius::op

namespace sirius::op::detail {

/**
 * @brief A CUDA runtime failure whose construction cannot turn a fatal error into recoverable host
 * OOM.
 */
class accumulation_cuda_error : public std::exception {
 public:
  /**
   * @brief Describes a failed CUDA call.
   *
   * @param code The error the call returned
   * @param kernel_launch Whether the call launched a kernel
   * @param message Static-lifetime context for the failed call
   */
  accumulation_cuda_error(cudaError_t code, bool kernel_launch, char const* message) noexcept
    : _code{code}, _kernel_launch{kernel_launch}, _message{message}
  {
  }

  [[nodiscard]] char const* what() const noexcept override { return _message; }

  [[nodiscard]] cudaError_t code() const noexcept { return _code; }

  /**
   * @brief True for a kernel launch that failed with a code the mandatory path retries
   * (`is_retryable_launch_error`).
   */
  [[nodiscard]] bool transient_launch_failure() const noexcept
  {
    return _kernel_launch && is_retryable_launch_error(_code);
  }

 private:
  cudaError_t _code;
  bool _kernel_launch;
  char const* _message;
};

/**
 * @brief An internal accumulation contract failure that remains fatal under host allocation
 * refusal.
 */
class accumulation_invariant_error : public std::exception {
 public:
  explicit accumulation_invariant_error(char const* static_context) noexcept
    : _context{static_context}
  {
  }
  [[nodiscard]] char const* what() const noexcept override { return _context; }

 private:
  char const* _context;
};

/**
 * @brief Runs @p body; if it throws, host-synchronizes every stream in @p streams before the
 * exception continues.
 *
 * A failed synchronization replaces the original exception with `accumulation_cuda_error`. If a
 * stream still reports unfinished work, `unjoined_gpu_work` takes precedence so its storage remains
 * protected. On success nothing waits.
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
    bool unjoined           = false;
    cudaError_t failed_join = cudaSuccess;
    for (auto const stream : streams) {
      auto const synced = cudaStreamSynchronize(stream.get());
      if (synced == cudaSuccess) { continue; }
      failed_join = synced;
      (void)cudaGetLastError();
      unjoined = unjoined || cudaStreamQuery(stream.get()) == cudaErrorNotReady;
      (void)cudaGetLastError();
      // The failure being unwound is reported by the caller.
      sirius::log::log_noexcept([&] {
        SIRIUS_LOG_ERROR("[join_on_failure] host join after a failure failed: {}",
                         cudaGetErrorString(synced));
      });
    }
    if (unjoined) { std::throw_with_nested(unjoined_gpu_work{}); }
    if (failed_join != cudaSuccess) {
      std::throw_with_nested(accumulation_cuda_error{
        failed_join, false, "cudaStreamSynchronize during accumulation failure cleanup"});
    }
    throw;
  }
}

/**
 * @brief Owns per-GPU Bloom partials and the complete filters produced from them.
 *
 * `dynamic_filter_publication_session` creates one builder when accumulation begins. Each replica
 * GPU gets a partial: one zeroed array per key and an exclusive, non-blocking stream that orders
 * every later use and every free of that GPU's filter arrays. Contributions insert keys on the
 * contributing task's stream and fold their completion into the partial's stream; `finish` reduces
 * the partials and publishes the union. Every member returns or throws only after the GPU work it
 * enqueued is complete or folded into a partial's stream. An unjoined fatal failure retains storage
 * instead of freeing memory still reachable by GPU work.
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
   *
   * @param type The storage type of a build key column
   * @return True if @p type is INT32 or INT64
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
   * @brief Inserts @p input's key columns into @p device's partial, without a host wait or a device
   * allocation.
   *
   * Thread-safe. @p stream first waits for the partial's zero fill; the partial's stream then waits
   * for the inserts, so every later use or free of the arrays follows them.
   *
   * @pre @p input holds every key column at its ordinal with its type, and the caller retires the
   * stream before releasing @p input
   * @throw accumulation_invariant_error if @p stream belongs to another GPU, before anything is
   * enqueued
   * @throw accumulation_cuda_error if a CUDA call fails, or `cucascade::cuda_error` if creating its
   * completion event fails; work already enqueued is joined first
   * @param input A build batch with the admitted key types
   * @param device The GPU that holds @p input
   * @param stream The contributing task's stream
   * @return false, before anything is enqueued, iff @p device holds no partial
   */
  [[nodiscard]] bool enqueue_add(cudf::table_view const& input,
                                 rmm::cuda_device_id device,
                                 ::cuda::stream_ref stream);

  /**
   * @brief How `finish` moved partials and the union between GPUs.
   */
  enum class publication_transport : std::uint8_t {
    PEER_COPY,   ///< `cudaMemcpyPeerAsync`; every replica pool grants every other replica GPU
    HOST_STAGED  ///< Bounded pinned HOST staging, when a grant is unavailable or direct scratch
                 ///< is refused
  };

  /**
   * @brief A completed publication.
   */
  struct publication {
    /// One ready filter per key; borrows owners the builder retains until its destruction
    std::span<std::shared_ptr<sirius_dynamic_bloom_filter> const> filters;
    publication_transport transport;  ///< The transport that produced every replica
  };

  /**
   * @brief Reduces every contributed partial into @p root's, overwrites every other partial with
   * the union, and waits for every replica on the host.
   *
   * Uses peer copies when every replica pool grants every other replica GPU and direct scratch is
   * admitted; otherwise stages through the root's HOST space. The transport is chosen before any
   * transfer is submitted and never changes afterwards. The builder retains every completed filter
   * owner; channels may share them. Scratch uses @p scratch_mr only during this call and is freed
   * after every submitted transfer has retired.
   *
   * @pre Every `enqueue_add` returned; this is the only `finish` call; the caller is the only user
   *      of the builder
   * @throw accumulation_invariant_error if @p root is not the current device or @p stream belongs
   * to another GPU, before anything is enqueued
   * @param root The reducing GPU, current on the calling thread
   * @param stream The publication stream of @p root
   * @param scratch_mr Resource borrowing the task's reservation
   * @return The published filters, or no value when @p root holds no partial or publication
   *         workspace allocation is refused
   */
  [[nodiscard]] std::optional<publication> finish(rmm::cuda_device_id root,
                                                  ::cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref scratch_mr);

  /**
   * @brief True iff releasing this builder's storage now credits no allocation tracker.
   *
   * cuCascade credits a free to the calling thread's (or stream's) tracker for the freeing
   * allocator. A builder must therefore not be destroyed while the thread tracks any partial's
   * allocator. Never throws; an unexpected failure reports false.
   */
  [[nodiscard]] bool releasable_here() const noexcept;

  /**
   * @brief One GPU's Bloom bit array for one key, copied to the host by `inspect_partial` or
   *        `inspect_replica`.
   *
   * Both inspection functions below query the array's stream, enqueue a device-to-host copy on that
   * same stream so the copy runs after the work already queued there, and then wait on the host for
   * the copy. Both are meant for tests and diagnostics; publication never calls them. Tests compare
   * every replica of a key with the bitwise OR of that key's partials.
   */
  struct replica_contents {
    std::vector<std::byte> bytes;  ///< The copied bit array
    bool stream_idle;  ///< Whether the array's stream had no pending work before the copy
  };

  /**
   * @brief Copies the private partial (before publication) of @p device for key @p key_index to the
   *        host.
   *
   * A partial holds only the keys that `enqueue_add` inserted from the batches @p device
   * contributed. `finish` ORs every GPU's partial into the root and copies that union into each
   * replica, after which the partials no longer exist. The copy runs after the pending insertions
   * on the partial's stream.
   *
   * @pre All `enqueue_add` calls returned and publication has not begun; the caller is the
   *      builder's only user
   * @param key_index The key whose partial is copied
   * @param device The GPU whose partial is copied
   * @return The partial contents, or no value for an absent key or GPU
   */
  [[nodiscard]] std::optional<replica_contents> inspect_partial(std::size_t key_index,
                                                                rmm::cuda_device_id device) const;

  /**
   * @brief Copies the replica (after publication) of @p filter on @p device to the host.
   *
   * A replica is one GPU's copy of the union of all partials for the key of @p filter, read through
   * the `sirius_dynamic_bloom_filter` that `finish` returned. The function is static because it
   * reads only @p filter, which may outlive the builder. The copy runs after the publication copies
   * on the replica's stream.
   *
   * @pre @p filter was published: the `finish` call that returned it has returned
   * @param filter An accumulated filter returned by `finish`
   * @param device The GPU whose replica is copied
   * @return The replica's contents, or nullopt if @p filter has no accumulated replica on @p device
   */
  [[nodiscard]] static std::optional<replica_contents> inspect_replica(
    sirius_dynamic_bloom_filter const& filter, rmm::cuda_device_id device);

 private:
  /**
   * @brief Successfully acquired workspace charges for one finish attempt, retained after cleanup.
   */
  struct workspace_observation {
    std::size_t device_scratch_bytes{};  ///< Aligned device allocation charge
    std::size_t host_staging_bytes{};    ///< Physical fixed-block HOST allocation charge
  };

  friend struct accumulated_bloom_test_access;

  /**
   * @brief `finish`, optionally forcing HOST staging and reporting the workspace it acquired.
   *
   * Tests reach it through `accumulated_bloom_test_access`; production calls `finish`.
   */
  [[nodiscard]] std::optional<publication> finish_impl(rmm::cuda_device_id root,
                                                       ::cuda::stream_ref stream,
                                                       rmm::device_async_resource_ref scratch_mr,
                                                       bool force_host,
                                                       workspace_observation* observation);

  struct impl;
  explicit accumulated_bloom_builder(std::unique_ptr<impl> state) noexcept;

  void make_filters(int root_device);

  std::unique_ptr<impl> _impl;
};

}  // namespace sirius::op::detail
