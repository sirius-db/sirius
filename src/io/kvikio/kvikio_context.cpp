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

#include "io/kvikio/kvikio_context.hpp"

#include "io/cache/prefetching_cache.hpp"
#include "io/cache/types.hpp"
#include "io/io_errors.hpp"
#include "io/rest/rest_reactor.hpp"

#include <kvikio/defaults.hpp>
#include <kvikio/remote_handle.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

namespace sirius::io {

namespace {

constexpr std::string_view k_s3_scheme = "s3://";

[[nodiscard]] bool is_s3_uri(std::string_view path) noexcept
{
  return path.size() > k_s3_scheme.size() && path.compare(0, k_s3_scheme.size(), k_s3_scheme) == 0;
}

const kvikio_object& as_kvikio(const io_object& obj)
{
  // A mismatch is a programmer error (mixing io_objects across backends), but
  // the cast is checked so it surfaces as a clear error instead of UB.
  const auto* typed = dynamic_cast<const kvikio_object*>(&obj);
  if (typed == nullptr) {
    throw std::invalid_argument("kvikio_context: io_object '" + obj.object_path() +
                                "' was not created by this backend");
  }
  return *typed;
}

/// Bytes actually available at @p offset.  kvikIO reads are unclamped, unlike
/// the @c cudf::io::datasource this backend used to wrap, so callers past EOF
/// would otherwise depend on kvikIO's short-read behaviour.
[[nodiscard]] size_t clamp_to_object(const io_object& obj, size_t offset, size_t size) noexcept
{
  return obj.size() > offset ? std::min(size, obj.size() - offset) : 0;
}

/// kvikIO falls back to the AWS_* environment when an argument is nullopt, so
/// every credential is passed engaged — an unconfigured store is rejected in
/// create_io_object before we get here.
[[nodiscard]] std::optional<std::string> engaged(std::string const& value)
{
  return std::optional<std::string>{value};
}

}  // namespace

std::size_t kvikio_io_object::read_at(void* dst, std::size_t size, std::size_t offset) const
{
  size = clamp_to_object(*this, offset, size);
  if (size == 0) { return 0; }
  return handle().pread(dst, size, offset).get();
}

std::size_t kvikio_remote_io_object::read_at(void* dst, std::size_t size, std::size_t offset) const
{
  size = clamp_to_object(*this, offset, size);
  if (size == 0) { return 0; }
  return read_async(dst, size, offset).get();
}

std::future<std::size_t> kvikio_io_object::read_async(void* dst,
                                                      std::size_t size,
                                                      std::size_t offset) const
{
  return handle().pread(dst, size, offset);
}

std::future<std::size_t> kvikio_remote_io_object::read_async(void* dst,
                                                             std::size_t size,
                                                             std::size_t offset) const
{
  if (read_from_stash(dst, size, offset)) {
    std::promise<std::size_t> done;
    done.set_value(size);
    return done.get_future();
  }
  return handle().pread(dst, size, offset);
}

bool kvikio_remote_io_object::read_from_stash(void* dst, std::size_t size, std::size_t offset) const
{
  if (_stash == nullptr || offset < _window_lo || offset - _window_lo > _stash->size() ||
      size > _stash->size() - (offset - _window_lo)) {
    return false;
  }
  auto const* src = _stash->data() + (offset - _window_lo);
  cudaPointerAttributes attrs{};
  auto const attr_status = cudaPointerGetAttributes(&attrs, dst);
  if (attr_status != cudaSuccess) { std::ignore = cudaGetLastError(); }  // Not CUDA memory.
  if (attr_status == cudaSuccess && attrs.type == cudaMemoryTypeDevice) {
    auto const status = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
  } else {
    std::memcpy(dst, src, size);
  }
  return true;
}

object_identity kvikio_remote_io_object::identity() const
{
  auto result = io_object::identity();
  if (!rest::rest_io_object::is_strong_tag(_etag)) {
    result.kind    = identity_kind::opened_handle;
    result.version = std::to_string(open_generation());
  }
  return result;
}

void apply_kvikio_defaults(kvikio_config const& cfg)
{
  // Validate before touching anything so a bad config leaves kvikIO's globals
  // untouched rather than half-applied.
  if (cfg.nthreads && *cfg.nthreads == 0) {
    throw std::invalid_argument("kvikio_config: nthreads must be non-zero");
  }
  if (cfg.task_size && *cfg.task_size == 0) {
    throw std::invalid_argument("kvikio_config: task_size must be non-zero");
  }
  if (cfg.bounce_buffer_size && *cfg.bounce_buffer_size == 0) {
    throw std::invalid_argument("kvikio_config: bounce_buffer_size must be non-zero");
  }

  // Only engaged fields are pushed, so an unset field keeps whatever kvikIO
  // seeded from its environment variable.
  //
  // Order matters for the two thread-pool knobs: set the per-block-device flag
  // first so that pools created afterwards are sized by the nthreads below,
  // rather than rebuilding a global pool we are about to replace anyway.
  if (cfg.thread_pool_per_block_device) {
    kvikio::defaults::set_thread_pool_per_block_device(*cfg.thread_pool_per_block_device);
  }
  if (cfg.nthreads) { kvikio::defaults::set_thread_pool_nthreads(*cfg.nthreads); }
  if (cfg.task_size) { kvikio::defaults::set_task_size(*cfg.task_size); }
  if (cfg.gds_threshold) { kvikio::defaults::set_gds_threshold(*cfg.gds_threshold); }
  if (cfg.bounce_buffer_size) { kvikio::defaults::set_bounce_buffer_size(*cfg.bounce_buffer_size); }
  if (cfg.auto_direct_io_read) {
    kvikio::defaults::set_auto_direct_io_read(*cfg.auto_direct_io_read);
  }
  if (cfg.auto_direct_io_read_overread) {
    kvikio::defaults::set_auto_direct_io_read_overread(*cfg.auto_direct_io_read_overread);
  }
  // compat_mode is deliberately NOT set globally — it rides the FileHandle
  // constructor in create_io_object so it scopes to this ioctx's files.
}

kvikio_context::kvikio_context(kvikio_config cfg,
                               object_store_config os,
                               std::shared_ptr<rest::rest_reactor> control_plane)
  : _config(std::move(cfg)), _object_store(std::move(os)), _control_plane(std::move(control_plane))
{
  apply_kvikio_defaults(_config);
}

std::shared_ptr<io_object> kvikio_context::create_remote_io_object(
  std::string path, std::optional<std::uint64_t> known_size, bool footer_probe)
{
  if (_object_store.endpoint.empty() || _object_store.region.empty() ||
      _object_store.access_key.empty() || _object_store.secret_key.empty()) {
    throw std::runtime_error(
      "kvikio_context: cannot open '" + path +
      "': object store not configured (endpoint / region / credentials missing)");
  }
  auto bucket_and_object = kvikio::S3Endpoint::parse_s3_url(path);

  // Resolve the size (and, through the control plane, the ETag and possibly
  // the footer) before building the handle, which then skips its own HEAD.
  std::string etag;
  std::size_t window_lo = 0;
  kvikio_remote_io_object::stash_type stash;
  if (!known_size && _control_plane != nullptr) {
    auto const& [bucket, key] = bucket_and_object;
    rest::footer_probe probe;
    if (footer_probe) {
      probe =
        _control_plane->fetch_footer_suffix(bucket, key, _control_plane->get_config().footer_probe_bytes);
    }
    if (probe.bytes) {
      known_size = probe.object_size;
      etag       = std::move(probe.etag);
      window_lo  = probe.window_lo;
      stash      = std::move(probe.bytes);
    } else {
      // No probe asked for, or an unusable suffix response: a plain HEAD.
      auto head  = _control_plane->head_object(bucket, key);
      known_size = head.object_size;
      etag       = std::move(head.etag);
    }
  }

  auto endpoint = std::make_unique<kvikio::S3Endpoint>(std::move(bucket_and_object),
                                                       engaged(_object_store.region),
                                                       engaged(_object_store.access_key),
                                                       engaged(_object_store.secret_key),
                                                       engaged(_object_store.endpoint),
                                                       engaged(_object_store.session_token));
  // With the size known, the handle skips the HEAD it would otherwise send.
  auto handle = known_size ? kvikio::RemoteHandle{std::move(endpoint), *known_size}
                           : kvikio::RemoteHandle{std::move(endpoint)};
  auto const object_size = handle.nbytes();
  return std::make_shared<kvikio_remote_io_object>(
    std::move(path), std::move(handle), object_size, std::move(etag), window_lo, std::move(stash));
}

std::shared_ptr<io_object> kvikio_context::create_io_object(std::string path,
                                                           std::uint64_t known_size)
{
  if (is_s3_uri(path)) { return create_remote_io_object(std::move(path), known_size, false); }
  return create_io_object(std::move(path));
}

std::shared_ptr<io_object> kvikio_context::create_io_object(std::string path, open_hint hint)
{
  if (is_s3_uri(path)) {
    return create_remote_io_object(
      std::move(path), std::nullopt, hint == open_hint::parquet_footer_probe);
  }
  return create_io_object(std::move(path));
}

std::shared_ptr<io_object> kvikio_context::create_io_object(std::string path)
{
  if (is_s3_uri(path)) { return create_remote_io_object(std::move(path), std::nullopt, false); }
  // Read-only: this ioctx serves the scan path only.  The handle owns the fd
  // (and any cuFile registration) for the io_object's lifetime, and the
  // io_object outlives any single datasource wrapping it.
  //
  // compat_mode is passed per handle (rather than through kvikio::defaults) so
  // it applies only to files this ioctx opens; unset falls back to kvikIO's own
  // default, which honours KVIKIO_COMPAT_MODE.
  for (int attempt = 0; attempt < 2; ++attempt) {
    kvikio::FileHandle handle =
      _config.compat_mode
        ? kvikio::FileHandle{path, "r", kvikio::FileHandle::m644, *_config.compat_mode}
        : kvikio::FileHandle{path, "r"};
    if (handle.fd(true) >= 0) {
      object_identity buffered_identity, direct_identity;
      if (!capture_local_identity(handle.fd(false), buffered_identity) ||
          !capture_local_identity(handle.fd(true), direct_identity)) {
        throw descriptor_mismatch_error(
          "kvikio_context::create_io_object: identity_descriptor_mismatch");
      }
      if (buffered_identity.device != direct_identity.device ||
          buffered_identity.inode != direct_identity.inode) {
        // FileHandle owns both descriptors; discard both before retrying.
        continue;
      }
    }
    auto const file_size = handle.nbytes();
    return std::make_shared<kvikio_io_object>(std::move(path), std::move(handle), file_size);
  }
  throw descriptor_mismatch_error("kvikio_context::create_io_object: identity_descriptor_mismatch");
}

bool kvikio_context::prefers_bulk_io(io_object const& obj) const noexcept
{
  return prefers_bulk_io() && dynamic_cast<kvikio_remote_io_object const*>(&obj) != nullptr;
}

bool kvikio_context::supports(std::string_view /*path*/) const noexcept
{
  // Universal fallback: kvikIO handles local paths, and the actual feasibility
  // check happens at create_io_object time, where opening the file may throw.
  // The registry consults this last, so an explicit backend always wins.
  return true;
}

std::vector<cudf::io::text::byte_range_info> kvikio_context::align_and_coalesce(
  std::span<const cudf::io::text::byte_range_info> ranges,
  std::optional<size_t> /*alignment*/) const noexcept
{
  return {ranges.begin(), ranges.end()};
}

size_t kvikio_context::host_read_io(const io_object& obj, size_t offset, size_t size, uint8_t* dst)
{
  // read_at dispatches on the destination pointer type, so the same call serves
  // host and device buffers; here it is always host memory.
  return as_kvikio(obj).read_at(dst, size, offset);
}

namespace {

/// @p device_id, or the calling thread's current device when it is negative.
[[nodiscard]] rmm::cuda_device_id resolve_device(int device_id)
{
  if (device_id < 0) {
    auto const status = cudaGetDevice(&device_id);
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
  }
  return rmm::cuda_device_id{device_id};
}

void check_cuda(cudaError_t status)
{
  if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}

/// The device that owns @p ptr, or the calling thread's current device when
/// the pointer is not device memory.
[[nodiscard]] int device_of(void const* ptr)
{
  cudaPointerAttributes attrs{};
  auto const status = cudaPointerGetAttributes(&attrs, ptr);
  if (status == cudaSuccess && attrs.type == cudaMemoryTypeDevice) { return attrs.device; }
  if (status != cudaSuccess) { std::ignore = cudaGetLastError(); }  // Not CUDA memory.
  int device = 0;
  check_cuda(cudaGetDevice(&device));
  return device;
}

/// One batch handed to @c mixed_readv_async_io, from issue to completion.
struct readv_batch {
  std::vector<prepared_io_slice> slices;
  /// The reads started for each slice: one for a contiguous or device
  /// destination, one per cache chunk for a fragmented one.
  std::vector<std::vector<std::future<std::size_t>>> reads;
  /// Bytes each slice's reads must return in total.
  std::vector<std::size_t> expected;
  exec::promise<size_t> promise;

  /// Wait for every started read: a destination stays in use until its read finishes.
  void wait_all() noexcept
  {
    for (auto& slice_reads : reads) {
      for (auto& read : slice_reads) {
        if (read.valid()) { read.wait(); }
      }
    }
  }

  /// Fail every slice, then the batch.
  void fail(std::exception_ptr error) noexcept
  {
    for (auto& slice : slices) {
      if (slice.on_complete != nullptr) { (*slice.on_complete)(slice.h_buffer.fragments(), false); }
    }
    promise.set_exception(std::move(error));
  }
};

/// Record @p device_id / @p stream in @p streams unless already there.
void add_stream(std::vector<std::pair<int, cudaStream_t>>& streams, device_buffer const& d)
{
  auto const key = std::pair{d.device_id, d.stream.get()};
  if (std::find(streams.begin(), streams.end(), key) == streams.end()) { streams.push_back(key); }
}

}  // namespace

exec::semi_future<size_t> kvikio_context::mixed_readv_async_io(
  const io_object& obj, std::vector<prepared_io_slice>&& slices) noexcept
{
  auto batch    = std::make_shared<readv_batch>();
  batch->slices = std::move(slices);
  auto result   = batch->promise.get_semi_future();
  // Fragmented slices fill prefetching-cache chunks, which exist only with a cache.
  auto const chunk_bytes = cache() != nullptr ? cache()->chunk_size() : std::size_t{0};

  kvikio_object const* object = nullptr;
  std::exception_ptr issue_error;
  try {
    object = &as_kvikio(obj);
    batch->reads.resize(batch->slices.size());
    batch->expected.resize(batch->slices.size(), 0);

    // The settle task runs on a pool thread whose current device is not the
    // caller's, so pin every device destination to its own device now.
    for (auto& slice : batch->slices) {
      if (slice.has_device_request() && slice.d_buffer.device_id < 0) {
        slice.d_buffer.device_id = device_of(slice.d_buffer.data);
      }
    }

    // Start every host and cache read now, before waiting on anything, so
    // kvikIO serves them concurrently.  Reads straight into the device start
    // in the settle task below: they first wait for their destination streams.
    for (std::size_t i = 0; i < batch->slices.size(); ++i) {
      auto& slice      = batch->slices[i];
      auto& reads      = batch->reads[i];
      auto const bytes = clamp_to_object(obj, slice.offset(), slice.size());
      slice.rng.size   = bytes;
      if (bytes == 0) { continue; }
      if (slice.is_fragmented()) {
        // Fill each chunk to the extent the cache advertises for it, which
        // covers this slice's part of the chunk.
        if (chunk_bytes == 0) {
          throw std::runtime_error("kvikio_context: cache buffers without a prefetching cache");
        }
        for (auto* chunk : slice.h_buffer.fragments()) {
          auto const [fill_lo, fill_hi] =
            cache::fill_span(chunk->state.get_fill(), chunk->offset, chunk_bytes);
          auto const hi = std::min<std::size_t>(fill_hi, obj.size());
          if (hi <= fill_lo) { continue; }
          if (chunk->data == nullptr) {
            throw std::runtime_error("kvikio_context: cache chunk has no buffer");
          }
          reads.push_back(
            object->read_async(chunk->data + (fill_lo - chunk->offset), hi - fill_lo, fill_lo));
          batch->expected[i] += hi - fill_lo;
        }
      } else if (slice.is_contiguous()) {
        // Host destination, copied on to the device afterwards when requested.
        reads.push_back(object->read_async(
          std::get<std::uint8_t*>(slice.h_buffer.buffer), bytes, slice.offset()));
        batch->expected[i] = bytes;
      } else if (slice.has_device_request()) {
        batch->expected[i] = bytes;  // Started in the settle task.
      } else {
        throw std::invalid_argument("prepared slice has no destination");
      }
    }
  } catch (...) {
    issue_error = std::current_exception();
  }

  // Settle the batch off the caller's thread: start the device reads, wait for
  // all reads, copy host-staged slices on to the device, then complete the
  // slices and the batch.
  auto settle = [batch, object, issue_error, chunk_bytes]() noexcept {
    if (issue_error) {
      batch->wait_all();
      batch->fail(issue_error);
      return;
    }
    try {
      // kvikIO writes device destinations from its own stream, not the
      // slice's, so work already queued on a destination stream (e.g. the
      // stream-ordered allocation of the buffer) has to finish first.  Drain
      // each distinct stream once for the whole batch.  Slices staged through
      // host memory are copied on their own stream and need no drain.
      std::vector<std::pair<int, cudaStream_t>> drained;
      for (auto const& slice : batch->slices) {
        if (slice.size() != 0 && slice.has_device_request() && slice.needs_staging()) {
          add_stream(drained, slice.d_buffer);
        }
      }
      for (auto const& [device_id, stream] : drained) {
        rmm::cuda_set_device_raii const guard{resolve_device(device_id)};
        check_cuda(cudaStreamSynchronize(stream));
      }
      for (std::size_t i = 0; i < batch->slices.size(); ++i) {
        auto& slice = batch->slices[i];
        if (slice.size() == 0 || !slice.has_device_request() || !slice.needs_staging()) {
          continue;
        }
        rmm::cuda_set_device_raii const guard{resolve_device(slice.d_buffer.device_id)};
        batch->reads[i].push_back(
          object->read_async(slice.d_buffer.data, slice.size(), slice.offset()));
      }
    } catch (...) {
      batch->wait_all();
      batch->fail(std::current_exception());
      return;
    }
    batch->wait_all();
    std::size_t total = 0;
    try {
      std::vector<std::pair<int, cudaStream_t>> copied;
      for (std::size_t i = 0; i < batch->slices.size(); ++i) {
        auto& slice      = batch->slices[i];
        std::size_t read = 0;
        for (auto& future : batch->reads[i]) {
          read += future.get();
        }
        if (read != batch->expected[i]) { throw std::runtime_error("kvikio_context: short read"); }
        if (slice.size() == 0 || !slice.has_device_request() || slice.needs_staging()) {
          total += slice.size();
          continue;
        }
        // Copy the slice's logical range from host staging to the device.
        rmm::cuda_set_device_raii const guard{resolve_device(slice.d_buffer.device_id)};
        auto const stream = slice.d_buffer.stream.get();
        if (slice.is_contiguous()) {
          check_cuda(cudaMemcpyAsync(slice.d_buffer.data,
                                     std::get<std::uint8_t*>(slice.h_buffer.buffer),
                                     slice.size(),
                                     cudaMemcpyDefault,
                                     stream));
        } else {
          for (auto* chunk : slice.h_buffer.fragments()) {
            auto const part = intersect(slice.rng, range{chunk->offset, chunk_bytes});
            if (part.empty()) { continue; }
            check_cuda(cudaMemcpyAsync(slice.d_buffer.data + (part.offset - slice.offset()),
                                       chunk->data + (part.offset - chunk->offset),
                                       part.size,
                                       cudaMemcpyDefault,
                                       stream));
          }
        }
        add_stream(copied, slice.d_buffer);
        total += slice.size();
      }
      for (auto const& [device_id, stream] : copied) {
        rmm::cuda_set_device_raii const guard{resolve_device(device_id)};
        check_cuda(cudaStreamSynchronize(stream));
      }
    } catch (...) {
      batch->fail(std::current_exception());
      return;
    }
    for (auto& slice : batch->slices) {
      if (slice.on_complete != nullptr) { (*slice.on_complete)(slice.h_buffer.fragments(), true); }
    }
    batch->promise.set_value(total);
  };

  try {
    _completion_pool.detach_task(settle);
  } catch (...) {
    // The pool could not take the task: settle the batch here.
    settle();
  }
  return result;
}

}  // namespace sirius::io
