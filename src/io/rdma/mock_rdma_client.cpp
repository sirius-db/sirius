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

#include "io/rdma/mock_rdma_client.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace sirius::io::rdma {

namespace {

/// Copy that works for host and device destinations, and on hosts with no
/// usable CUDA device at all (host-only tests): a plain memcpy unless the
/// destination is identifiably device memory.
void copy_to_destination(void* dst, const std::uint8_t* src, size_t n)
{
  if (n == 0) { return; }
  cudaPointerAttributes attr{};
  auto err        = cudaPointerGetAttributes(&attr, dst);
  bool device_dst = err == cudaSuccess && attr.type == cudaMemoryTypeDevice;
  if (err != cudaSuccess) { (void)cudaGetLastError(); }
  if (device_dst) {
    auto copy_err = cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice);
    if (copy_err != cudaSuccess) {
      throw std::runtime_error(std::string("mock_rdma_client: H2D copy failed: ") +
                               cudaGetErrorString(copy_err));
    }
    return;
  }
  std::memcpy(dst, src, n);
}

}  // namespace

void mock_rdma_client::put_object(std::string bucket,
                                  std::string key,
                                  std::vector<std::uint8_t> bytes)
{
  std::lock_guard lk{_mtx};
  _objects[{std::move(bucket), std::move(key)}] = std::move(bytes);
}

void mock_rdma_client::fail_gets(std::string message)
{
  std::lock_guard lk{_mtx};
  _fail_message = std::move(message);
}

void mock_rdma_client::fail_next_gets(size_t count, std::string message)
{
  std::lock_guard lk{_mtx};
  _fail_next_count   = count;
  _fail_next_message = std::move(message);
}

void mock_rdma_client::short_read(size_t bytes)
{
  std::lock_guard lk{_mtx};
  _short_read = bytes;
}

void mock_rdma_client::clear_fault()
{
  std::lock_guard lk{_mtx};
  _fail_message.reset();
  _fail_next_count = 0;
  _short_read.reset();
}

void mock_rdma_client::close_gate()
{
  std::lock_guard lk{_mtx};
  _gate_closed = true;
}

void mock_rdma_client::open_gate()
{
  {
    std::lock_guard lk{_mtx};
    _gate_closed = false;
  }
  _gate_cv.notify_all();
}

size_t mock_rdma_client::gets_issued() const noexcept
{
  std::lock_guard lk{_mtx};
  return _gets_issued;
}

size_t mock_rdma_client::peak_concurrent_gets() const noexcept
{
  std::lock_guard lk{_mtx};
  return _peak_concurrent;
}

size_t mock_rdma_client::head(std::string_view bucket, std::string_view key)
{
  std::lock_guard lk{_mtx};
  auto it = _objects.find({std::string(bucket), std::string(key)});
  if (it == _objects.end()) {
    throw std::runtime_error("mock_rdma_client: no such object s3://" + std::string(bucket) + "/" +
                             std::string(key));
  }
  return it->second.size();
}

size_t mock_rdma_client::get(
  std::string_view bucket, std::string_view key, size_t offset, size_t size, void* dst)
{
  std::unique_lock lk{_mtx};
  ++_gets_issued;
  ++_concurrent;
  _peak_concurrent = std::max(_peak_concurrent, _concurrent);
  struct concurrent_guard {
    size_t& count;
    ~concurrent_guard() { --count; }
  } guard{_concurrent};

  _gate_cv.wait(lk, [&] { return !_gate_closed; });

  if (_fail_message) { throw std::runtime_error(*_fail_message); }
  if (_fail_next_count > 0) {
    --_fail_next_count;
    throw std::runtime_error(_fail_next_message);
  }
  auto it = _objects.find({std::string(bucket), std::string(key)});
  if (it == _objects.end()) {
    throw std::runtime_error("mock_rdma_client: no such object s3://" + std::string(bucket) + "/" +
                             std::string(key));
  }
  const auto& bytes = it->second;
  if (offset >= bytes.size()) { return 0; }
  size_t n = std::min(size, bytes.size() - offset);
  if (_short_read) { n = std::min(n, *_short_read); }
  copy_to_destination(dst, bytes.data() + offset, n);
  return n;
}

}  // namespace sirius::io::rdma
