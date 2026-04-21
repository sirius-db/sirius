/*
 * Copyright 2025, Sirius Contributors.
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

#include "io/cucascade_datasource.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace sirius::io {

namespace {

/**
 * @brief RAII wrapper for a pinned (page-locked) host allocation.
 *
 * Uses cudaMallocHost directly rather than cucascade::memory::fixed_size_host_memory_resource
 * because that resource is per-memory-space and owned by SiriusContext; the datasource
 * stays decoupled from SiriusContext by allocating pinned memory via the CUDA runtime.
 *
 * See 05-RESEARCH.md Pitfall 2: pinned memory is required so cuDF's
 * cuda_memcpy_async on the returned buffer stays truly asynchronous (IO-03).
 *
 * Exposes .data() and .size() so it can be stored in a cudf::io::datasource::owning_buffer
 * via cudf::io::datasource::buffer::create(...).
 */
struct pinned_host_buffer {
  uint8_t* _ptr{nullptr};
  std::size_t _bytes{0};

  pinned_host_buffer() = default;

  explicit pinned_host_buffer(std::size_t n) : _bytes(n)
  {
    if (n == 0) { return; }
    void* p         = nullptr;
    auto const err  = cudaMallocHost(&p, n);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cucascade_datasource: cudaMallocHost failed: ") +
                               cudaGetErrorString(err));
    }
    _ptr = static_cast<uint8_t*>(p);
  }

  ~pinned_host_buffer()
  {
    if (_ptr != nullptr) { cudaFreeHost(_ptr); }
  }

  pinned_host_buffer(pinned_host_buffer const&)            = delete;
  pinned_host_buffer& operator=(pinned_host_buffer const&) = delete;

  pinned_host_buffer(pinned_host_buffer&& other) noexcept
    : _ptr(other._ptr), _bytes(other._bytes)
  {
    other._ptr   = nullptr;
    other._bytes = 0;
  }

  pinned_host_buffer& operator=(pinned_host_buffer&& other) noexcept
  {
    if (this != &other) {
      if (_ptr != nullptr) { cudaFreeHost(_ptr); }
      _ptr         = other._ptr;
      _bytes       = other._bytes;
      other._ptr   = nullptr;
      other._bytes = 0;
    }
    return *this;
  }

  // Interface expected by cudf::io::datasource::owning_buffer<T>.
  [[nodiscard]] uint8_t const* data() const { return _ptr; }
  [[nodiscard]] std::size_t size() const { return _bytes; }
};

/**
 * @brief Remote URI schemes rejected at construction — out of scope per PROJECT.md.
 *
 * The adapter is local-filesystem-only; remote schemes must fail fast rather than
 * silently proceeding to a read that will error inside the backend. CONTEXT.md
 * locks this behavior.
 */
constexpr std::string_view kRemotePrefixes[] = {
  "s3://", "http://", "https://", "hdfs://", "gs://", "azure://"};

[[nodiscard]] bool has_remote_scheme(std::string const& path_str)
{
  for (auto const& prefix : kRemotePrefixes) {
    if (path_str.size() >= prefix.size() &&
        std::string_view(path_str.data(), prefix.size()) == prefix) {
      return true;
    }
  }
  return false;
}

}  // namespace

cucascade_datasource::cucascade_datasource(std::shared_ptr<cucascade::idisk_io_backend> backend,
                                           std::filesystem::path path,
                                           std::size_t file_size)
  : _backend(std::move(backend)), _path(std::move(path)), _file_size(file_size)
{
  if (_backend == nullptr) {
    throw std::invalid_argument("cucascade_datasource: backend must not be null");
  }

  auto const path_str = _path.string();
  if (has_remote_scheme(path_str)) {
    throw std::invalid_argument(
      "cucascade_datasource: remote URI scheme not supported — local filesystem only. Path: " +
      path_str);
  }
}

cucascade_datasource::~cucascade_datasource() = default;

//===----------------------------------------------------------------------===//
// Host reads (sync)
//===----------------------------------------------------------------------===//

std::size_t cucascade_datasource::host_read(std::size_t offset, std::size_t size, uint8_t* dst)
{
  // Clip request to the known file size. Mirrors kvikio_source::clamped_read_to_vector
  // behavior so callers can request a range past EOF without error (cuDF's parquet
  // footer planning occasionally does this).
  if (offset >= _file_size) { return 0; }
  auto const read_size = std::min(size, _file_size - offset);
  if (read_size == 0) { return 0; }

  _backend->read(_path, dst, read_size, offset);
  return read_size;
}

std::unique_ptr<cudf::io::datasource::buffer> cucascade_datasource::host_read(std::size_t offset,
                                                                              std::size_t size)
{
  // Clip request to file size (same contract as the dst overload).
  std::size_t read_size = 0;
  if (offset < _file_size) { read_size = std::min(size, _file_size - offset); }

  // Allocate a pinned host buffer for the (possibly clipped) read. Pinned memory is
  // load-bearing for IO-03: cuDF downstream issues cuda_memcpy_async on this buffer
  // and that call silently serializes on pageable memory.
  // Using cudaMallocHost directly; see 05-RESEARCH.md Pitfall 2.
  pinned_host_buffer buf{read_size};

  if (read_size > 0) { _backend->read(_path, buf._ptr, read_size, offset); }

  return cudf::io::datasource::buffer::create(std::move(buf));
}

//===----------------------------------------------------------------------===//
// Host reads (async)
//
// std::launch::async is locked by CONTEXT.md + 05-RESEARCH.md Pitfall 3.
// Intentionally differs from prefetched_data_source which uses std::launch::deferred —
// that class wraps an already-issued CUDA event, so deferred is correct there. Our
// backend call is a blocking host read, so deferred would collapse concurrency.
//===----------------------------------------------------------------------===//

std::future<std::size_t> cucascade_datasource::host_read_async(std::size_t offset,
                                                               std::size_t size,
                                                               uint8_t* dst)
{
  return std::async(std::launch::async, [this, offset, size, dst]() -> std::size_t {
    return this->host_read(offset, size, dst);
  });
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>>
cucascade_datasource::host_read_async(std::size_t offset, std::size_t size)
{
  return std::async(std::launch::async,
                    [this, offset, size]() -> std::unique_ptr<cudf::io::datasource::buffer> {
                      return this->host_read(offset, size);
                    });
}

}  // namespace sirius::io
