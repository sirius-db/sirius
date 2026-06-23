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

#include "exec/semi_future.hpp"
#include "io/cache/prefetching_cache.hpp"
#include "io/cache/types.hpp"
#include "io/io_context.hpp"
#include "io/types.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <exception>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius::io {

namespace detail {

/// Wrap cudaGetDevice with an exception on failure.  Without the check a
/// failure leaves the out-param at -1, the reactor's `if (device_id >= 0)
/// cudaSetDevice` skips, and on multi-GPU the H2D copy lands on whichever
/// device the reactor thread happens to have current — wrong-device data
/// corruption or a downstream cudaMemcpyAsync error on a different code
/// path.  Throwing here surfaces the original error at the dispatch site.
[[nodiscard]] inline int current_cuda_device()
{
  int id          = -1;
  cudaError_t err = cudaGetDevice(&id);
  if (err != cudaSuccess) {
    cudaGetLastError();
    throw std::runtime_error(std::string("templated_ioctx: cudaGetDevice failed: ") +
                             cudaGetErrorString(err));
  }
  return id;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Concept: io_reactor_c
// ---------------------------------------------------------------------------

// Baseline reactor contract.  Every reactor must provide buffered host reads
// (the single-range prep_host_rx_request + the synchronous host_read),
// lifecycle (shutdown / interrupt), dispatch (enqueue), io_object creation,
// and the static capability queries.  The optional dispatch paths (device
// reads, bounce-staged device reads, vectored host reads) are detected
// separately via reactor_traits below, so a reactor only defines the prep_*
// overloads for the paths it actually supports.
template <class R>
concept io_reactor_c = requires(R r,
                                typename R::io_object_type file,
                                typename R::request_type_ptr rx_req,
                                typename R::reactor_config_type cfg,
                                size_t offset,
                                size_t size,
                                uint8_t* dst,
                                std::string_view path,
                                std::string path_str) {
  typename R::io_object_type;
  typename R::request_type;
  typename R::request_type_ptr;
  typename R::reactor_config_type;

  {
    r.prep_host_rx_request(cfg, file, io_object_segment{offset, size, dst})
  } -> std::same_as<typename R::request_type_ptr>;

  { r.enqueue(std::move(rx_req)) };
  { r.host_read(file, offset, size, dst) } -> std::same_as<size_t>;

  { r.shutdown() };
  { r.interrupt() };

  {
    R::create_io_object(std::move(path_str))
  } -> std::same_as<std::unique_ptr<typename R::io_object_type>>;

  // -- capabilities --------------------------------------------------------
  { R::supports(path) } -> std::same_as<bool>;
  { R::preferred_prefetching_stage() } -> std::same_as<cache::prefetching_stage>;
};

// ---------------------------------------------------------------------------
// Reactor capability detection
// ---------------------------------------------------------------------------
//
// Optional dispatch paths are advertised structurally: a reactor supports a
// path iff it provides the matching create_* overload.  reactor_traits<R>
// probes for each one so a reactor no longer hand-maintains
// supports_device_read / supports_vector_host_read booleans that must be kept
// in sync with the methods it actually defines.

template <class R>
concept reactor_has_device_rx = requires(R r,
                                         typename R::io_object_type file,
                                         typename R::reactor_config_type cfg,
                                         uint8_t* dst,
                                         size_t offset,
                                         size_t size,
                                         rmm::cuda_stream_view stream) {
  {
    r.prep_device_rx_request(cfg, file, dst, offset, size, stream, 1)
  } -> std::same_as<typename R::request_type_ptr>;
};

template <class R>
concept reactor_has_host_to_device_rx = requires(R r,
                                                 typename R::io_object_type file,
                                                 typename R::reactor_config_type cfg,
                                                 uint8_t* dst,
                                                 size_t offset,
                                                 size_t size,
                                                 rmm::cuda_stream_view stream,
                                                 std::span<io_object_segment> bounce) {
  {
    r.prep_host_to_device_rx_request(cfg, file, bounce, dst, offset, size, stream, 1)
  } -> std::same_as<typename R::request_type_ptr>;
};

template <class R>
concept reactor_has_vector_host_rx =
  requires(R r,
           typename R::io_object_type file,
           typename R::reactor_config_type cfg,
           std::span<io_object_segment> dsts,
           std::span<const cudf::io::text::byte_range_info> ranges,
           std::optional<size_t> alignment) {
    { r.prep_host_rxv_request(cfg, file, dsts) } -> std::same_as<typename R::request_type_ptr>;
    {
      r.align_and_coalesce(ranges, alignment)
    } -> std::same_as<std::vector<cudf::io::text::byte_range_info>>;
  };

template <class R>
struct reactor_traits {
  /// BYO-device-buffer reads: caller supplies the device destination and the
  /// reactor streams file → device (directly or via its own bounce slots).
  static constexpr bool supports_device_read = reactor_has_device_rx<R>;

  /// Device reads staged through a caller-supplied pinned host bounce buffer
  static constexpr bool supports_host_to_device_read = reactor_has_host_to_device_rx<R>;

  /// Batched (vectored) host reads dispatched in a single call.
  static constexpr bool supports_vector_host_read = reactor_has_vector_host_rx<R>;
};

// ---------------------------------------------------------------------------
// templated_ioctx
// ---------------------------------------------------------------------------
//
// Generic ioctx implementation parameterised on a backend Reactor.  Owns a
// pool of reactors, the cache (via sirius_ioctx base), and implements all of
// the sirius_ioctx read API using only the operations named by io_reactor_c
// and io_object_c.  Backends plug in by providing a Reactor + io_object type
// pair that satisfies those concepts.

template <io_reactor_c Reactor>
class templated_ioctx : public sirius_ioctx {
 public:
  using reactor_type        = Reactor;
  using io_object_type      = typename Reactor::io_object_type;
  using request_type        = typename Reactor::request_type;
  using request_type_ptr    = typename Reactor::request_type_ptr;
  using reactor_traits_t    = reactor_traits<Reactor>;
  using reactor_config_type = typename Reactor::reactor_config_type;

  enum class io_op_type { host, host_async, device_async, host_vector_async };

  /// Construct with a pre-built vector of reactors (most flexible).
  explicit templated_ioctx(const reactor_config_type& cfg,
                           std::vector<std::unique_ptr<Reactor>> reactors)
    : _config(cfg), _reactors(std::move(reactors))
  {
  }

  /// Construct by invoking @p factory @p n_reactors times to build the pool.
  template <class Factory>
    requires std::invocable<Factory&, const reactor_config_type&> &&
             std::convertible_to<std::invoke_result_t<Factory&, const reactor_config_type&>,
                                 std::unique_ptr<Reactor>>
  templated_ioctx(size_t n_reactors, const reactor_config_type& cfg, Factory factory) : _config(cfg)
  {
    _reactors.reserve(n_reactors);
    for (size_t i = 0; i < n_reactors; ++i)
      _reactors.emplace_back(factory(_config));
  }

  ~templated_ioctx() override
  {
    // pre_destroy() drains the cache (if any) so its workers stop
    // issuing IO BEFORE we tear down the reactors below.  Must be the
    // first statement in every derived dtor — see sirius_ioctx for
    // the full contract.
    this->pre_destroy();
    shutdown();
  }

  /// Start every reactor in the pool (launches their worker threads and
  /// allocates their per-reactor staging).  The pool itself is built cheaply at
  /// construction; the worker threads and pinned bounce buffers are not spun up
  /// until here.  Idempotent — a second call is a no-op.
  void start() override
  {
    if (_started) { return; }
    for (auto& r : _reactors) {
      r->start();
    }
    _started = true;
  }

  void shutdown() noexcept override
  {
    for (auto& r : _reactors) {
      try {
        r->shutdown();
      } catch (const std::exception& e) {
        SIRIUS_LOG_ERROR("templated_ioctx: reactor shutdown failed: {}", e.what());
      } catch (...) {
        SIRIUS_LOG_ERROR("templated_ioctx: reactor shutdown failed: unknown error");
      }
    }
  }

  [[nodiscard]] bool supports(std::string_view path) const noexcept final
  {
    return Reactor::supports(path);
  }

  // -- Capabilities (delegate to the reactor) -------------------------------

  [[nodiscard]] bool supports_device_read() const noexcept final
  {
    return reactor_traits_t::supports_device_read;
  }

  [[nodiscard]] bool supports_host_to_device_read() const noexcept final
  {
    return reactor_traits_t::supports_host_to_device_read;
  }

  [[nodiscard]] bool supports_vector_host_read() const noexcept final
  {
    return reactor_traits_t::supports_vector_host_read;
  }

  [[nodiscard]] cache::prefetching_stage preferred_prefetching_stage() const noexcept override
  {
    // Vector host read is the cheap dispatch path the prefetcher relies on.
    // Without it, force prefetching off regardless of the reactor's preference.
    if (!reactor_traits_t::supports_vector_host_read) { return cache::prefetching_stage::none; }
    return Reactor::preferred_prefetching_stage();
  }

  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> align_and_coalesce(
    std::span<const cudf::io::text::byte_range_info> ranges,
    std::optional<size_t> alignment = std::nullopt) const noexcept override
  {
    if constexpr (reactor_traits_t::supports_vector_host_read) {
      return Reactor::align_and_coalesce(ranges, alignment);
    } else {
      // Backends without vectored host reads impose no alignment of their own;
      // pass the ranges through unchanged.
      return {ranges.begin(), ranges.end()};
    }
  }

  virtual std::vector<Reactor*> next_reactor([[maybe_unused]] const io_object_type& obj,
                                             [[maybe_unused]] size_t n_chunks,
                                             [[maybe_unused]] io_op_type type,
                                             [[maybe_unused]] int device_id = -1) noexcept
  {
    size_t idx = _next.fetch_add(1, std::memory_order_relaxed) % _reactors.size();
    return {_reactors.at(idx).get()};
  }

  // -- Host reads (generic: delegate to io_object_type + std::async) --------

  size_t host_read_io(const sirius_io_object& obj,
                      size_t offset,
                      size_t size,
                      uint8_t* dst) override
  {
    auto& tobj = as_typed(obj);
    size       = std::min(size, tobj.size() > offset ? tobj.size() - offset : size_t{0});
    if (size == 0) return 0;
    return next_reactor(tobj, 1, io_op_type::host).at(0)->host_read(tobj, offset, size, dst);
  }

  exec::semi_future<size_t> host_read_async_io(const sirius_io_object& obj,
                                               size_t offset,
                                               size_t size,
                                               uint8_t* dst) noexcept override
  {
    try {
      auto& tobj = as_typed(obj);
      size       = std::min(size, tobj.size() > offset ? tobj.size() - offset : size_t{0});

      auto reactors = next_reactor(tobj, 1, io_op_type::host_async);
      if (reactors.empty()) {
        return exec::make_semi_future<size_t>(
          std::make_exception_ptr(std::runtime_error("host_read_async_io: no available reactors")));
      }

      auto req = Reactor::prep_host_rx_request(_config, tobj, io_object_segment{offset, size, dst});
      auto semi = req->get_future();
      auto reqs = request_type::splits(std::move(req), reactors.size());
      assert(reqs.size() <= reactors.size());
      std::for_each(std::make_move_iterator(reqs.begin()),
                    std::make_move_iterator(reqs.end()),
                    [&reactors, i = 0](auto&& r) mutable { reactors[i++]->enqueue(std::move(r)); });
      return semi;
    } catch (...) {
      return exec::make_semi_future<size_t>(std::current_exception());
    }
  }

  // -- Device reads (generic chunking; reactor-backed) ----------------------

  exec::semi_future<size_t> device_read_async_io(const sirius_io_object& obj,
                                                 size_t offset,
                                                 size_t size,
                                                 uint8_t* dst,
                                                 rmm::cuda_stream_view stream) noexcept override
  {
    if constexpr (reactor_traits_t::supports_device_read) {
      try {
        auto& tobj    = as_typed(obj);
        int device_id = rmm::get_current_cuda_device().value();
        auto reactors = next_reactor(tobj, 0, io_op_type::device_async, device_id);
        if (reactors.empty()) {
          return exec::make_semi_future<size_t>(std::make_exception_ptr(
            std::runtime_error("device_read_async_io: no available reactors")));
        }
        request_type_ptr req =
          Reactor::prep_device_rx_request(_config, tobj, dst, offset, size, stream, device_id);
        auto semi = req->get_future();
        auto reqs = request_type::splits(std::move(req), reactors.size());
        assert(reqs.size() <= reactors.size());
        std::for_each(
          std::make_move_iterator(reqs.begin()),
          std::make_move_iterator(reqs.end()),
          [&reactors, i = 0](auto&& r) mutable { reactors[i++]->enqueue(std::move(r)); });
        return semi;
      } catch (...) {
        return exec::make_semi_future<size_t>(std::current_exception());
      }
    } else {
      return exec::make_semi_future<size_t>(
        std::make_exception_ptr(std::runtime_error("device_read_async_io: unsupported operation")));
    }
  }

  exec::semi_future<size_t> host_to_device_read_async_io(
    const sirius_io_object& obj,
    std::span<io_object_segment> slices,
    size_t offset,
    size_t size,
    uint8_t* dst,
    rmm::cuda_stream_view stream) noexcept override
  {
    if constexpr (reactor_traits_t::supports_host_to_device_read) {
      try {
        auto& tobj    = as_typed(obj);
        int device_id = rmm::get_current_cuda_device().value();
        auto reactors = next_reactor(tobj, 0, io_op_type::device_async, device_id);
        if (reactors.empty()) {
          return exec::make_semi_future<size_t>(std::make_exception_ptr(
            std::runtime_error("host_to_device_read_async_io: no available reactors")));
        }
        request_type_ptr req = Reactor::prep_host_to_device_rx_request(
          _config, tobj, slices, dst, offset, size, stream, device_id);
        auto semi = req->get_future();
        auto reqs = request_type::splits(std::move(req), reactors.size());
        assert(reqs.size() <= reactors.size());
        std::for_each(
          std::make_move_iterator(reqs.begin()),
          std::make_move_iterator(reqs.end()),
          [&reactors, i = 0](auto&& r) mutable { reactors[i++]->enqueue(std::move(r)); });
        return semi;
      } catch (...) {
        return exec::make_semi_future<size_t>(std::current_exception());
      }
    } else {
      return exec::make_semi_future<size_t>(std::make_exception_ptr(
        std::runtime_error("host_to_device_read_async_io: unsupported operation")));
    }
  }
  // -- Batch host reads (generic: dispatch to reactor host_read_async) ------

  exec::semi_future<size_t> host_read_ranges_async_io(
    const sirius_io_object& obj, std::span<io_object_segment> segments) noexcept override
  {
    if constexpr (reactor_traits_t::supports_vector_host_read) {
      try {
        auto& tobj = as_typed(obj);

        auto reactors = next_reactor(tobj, segments.size(), io_op_type::host_vector_async);
        if (reactors.empty()) {
          return exec::make_semi_future<size_t>(std::make_exception_ptr(
            std::runtime_error("host_read_ranges_async_io: no available reactors")));
        }

        auto req  = Reactor::prep_host_rxv_request(_config, tobj, segments);
        auto semi = req->get_future();
        auto reqs = request_type::splits(std::move(req), reactors.size());
        assert(reqs.size() <= reactors.size());
        std::for_each(
          std::make_move_iterator(reqs.begin()),
          std::make_move_iterator(reqs.end()),
          [&reactors, i = 0](auto&& r) mutable { reactors[i++]->enqueue(std::move(r)); });
        return semi;
      } catch (...) {
        return exec::make_semi_future<size_t>(std::current_exception());
      }
    } else {
      return exec::make_semi_future<size_t>(std::make_exception_ptr(
        std::runtime_error("host_read_ranges_async_io: unsupported operation")));
    }
  }

 protected:
  std::shared_ptr<sirius_io_object> create_io_object(std::string path) override
  {
    return std::shared_ptr<sirius_io_object>(Reactor::create_io_object(std::move(path)));
  }

  reactor_config_type _config;
  std::vector<std::unique_ptr<Reactor>> _reactors;
  std::atomic<size_t> _next{0};
  bool _started{false};

 private:
  static const io_object_type& as_typed(const sirius_io_object& obj) noexcept
  {
    return static_cast<const io_object_type&>(obj);
  }
};

}  // namespace sirius::io
