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

#include <kvikio/defaults.hpp>
#include <kvikio/remote_handle.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
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
  return handle().pread(dst, size, offset).get();
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

kvikio_context::kvikio_context(kvikio_config cfg, object_store_config os)
  : _config(std::move(cfg)), _object_store(std::move(os))
{
  apply_kvikio_defaults(_config);
}

std::shared_ptr<io_object> kvikio_context::create_io_object(std::string path)
{
  if (is_s3_uri(path)) {
    if (_object_store.endpoint.empty() || _object_store.region.empty() ||
        _object_store.access_key.empty() || _object_store.secret_key.empty()) {
      throw std::runtime_error(
        "kvikio_context: cannot open '" + path +
        "': object store not configured (endpoint / region / credentials missing)");
    }
    auto bucket_and_object = kvikio::S3Endpoint::parse_s3_url(path);
    auto endpoint          = std::make_unique<kvikio::S3Endpoint>(std::move(bucket_and_object),
                                                         engaged(_object_store.region),
                                                         engaged(_object_store.access_key),
                                                         engaged(_object_store.secret_key),
                                                         engaged(_object_store.endpoint),
                                                         engaged(_object_store.session_token));
    kvikio::RemoteHandle handle{std::move(endpoint)};
    auto const object_size = handle.nbytes();
    return std::make_shared<kvikio_remote_io_object>(
      std::move(path), std::move(handle), object_size);
  }
  // Read-only: this ioctx serves the scan path only.  The handle owns the fd
  // (and any cuFile registration) for the io_object's lifetime, and the
  // io_object outlives any single datasource wrapping it.
  //
  // compat_mode is passed per handle (rather than through kvikio::defaults) so
  // it applies only to files this ioctx opens; unset falls back to kvikIO's own
  // default, which honours KVIKIO_COMPAT_MODE.
  kvikio::FileHandle handle =
    _config.compat_mode
      ? kvikio::FileHandle{path, "r", kvikio::FileHandle::m644, *_config.compat_mode}
      : kvikio::FileHandle{path, "r"};
  auto const file_size = handle.nbytes();
  return std::make_shared<kvikio_io_object>(std::move(path), std::move(handle), file_size);
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

exec::semi_future<size_t> kvikio_context::host_read_async_io(const io_object& obj,
                                                             size_t offset,
                                                             size_t size,
                                                             uint8_t* dst) noexcept
{
  // make_semi_future_with invokes eagerly, so the kvikIO future is consumed
  // here and the returned semi_future is already satisfied.  Callers that need
  // true overlap use the uring backend.
  return exec::make_semi_future_with(
    [&obj, offset, size, dst]() -> size_t { return as_kvikio(obj).read_at(dst, size, offset); });
}

exec::semi_future<size_t> kvikio_context::device_read_async_io(
  const io_object& obj,
  size_t offset,
  size_t size,
  uint8_t* dst,
  rmm::cuda_stream_view stream) noexcept
{
  size = clamp_to_object(obj, offset, size);
  return exec::make_semi_future_with([&obj, offset, size, dst, stream]() -> size_t {
    if (size == 0) { return 0; }
    const auto* local = dynamic_cast<const kvikio_io_object*>(&obj);
    if (local == nullptr) { return as_kvikio(obj).read_at(dst, size, offset); }
    auto fut = local->handle().read_async(dst,
                                          size,
                                          static_cast<off_t>(offset),
                                          /*devPtr_offset=*/0,
                                          stream.value());
    return fut.check_bytes_done();
  });
}

exec::semi_future<size_t> kvikio_context::host_to_device_read_async_io(
  const io_object& /*obj*/,
  std::span<io_object_segment> /*slices*/,
  size_t /*offset*/,
  size_t /*size*/,
  uint8_t* /*device_dst*/,
  rmm::cuda_stream_view /*stream*/) noexcept
{
  return exec::make_semi_future<size_t>(std::make_exception_ptr(
    std::runtime_error("kvikio_context does not support host_to_device_read_async_io; use "
                       "device_read_async instead")));
}

exec::semi_future<size_t> kvikio_context::host_read_ranges_async_io(
  const io_object& /*obj*/, std::span<io_object_segment> /*segments*/) noexcept
{
  return exec::make_semi_future<size_t>(std::make_exception_ptr(
    std::runtime_error("kvikio_context does not support host_read_ranges_async_io")));
}

}  // namespace sirius::io
