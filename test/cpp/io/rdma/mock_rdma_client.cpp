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
#include <utility>

namespace sirius::io::rdma {

namespace {

constexpr const char* k_mock_reply_tag = "mock-reply-tag";

/// Deliver into host OR device memory: tests run with and without a usable
/// CUDA device (host-only tests): a plain memcpy unless the destination is
/// identifiably device memory.
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
      throw std::runtime_error(std::string("mock transport: H2D copy failed: ") +
                               cudaGetErrorString(copy_err));
    }
    return;
  }
  std::memcpy(dst, src, n);
}

}  // namespace

// ---- mock_s3_control_client ------------------------------------------------

void mock_s3_control_client::put_object(std::string bucket,
                                        std::string key,
                                        std::vector<std::uint8_t> bytes)
{
  std::lock_guard lk{_mtx};
  _objects[{std::move(bucket), std::move(key)}] = std::move(bytes);
}

void mock_s3_control_client::fail_transport(std::string curl_error)
{
  std::lock_guard lk{_mtx};
  _next_transport_error = std::move(curl_error);
}

void mock_s3_control_client::respond_status(long http_status)
{
  std::lock_guard lk{_mtx};
  _next_status = http_status;
}

void mock_s3_control_client::fail_next_n_range_gets(std::size_t count, long http_status)
{
  std::lock_guard lk{_mtx};
  _fail_next_range_gets = count;
  _fail_next_status     = http_status;
}

void mock_s3_control_client::override_content_range(std::string value)
{
  std::lock_guard lk{_mtx};
  _content_range_override = std::move(value);
}

void mock_s3_control_client::close_gate()
{
  std::lock_guard lk{_mtx};
  _gate_closed = true;
}

void mock_s3_control_client::open_gate()
{
  {
    std::lock_guard lk{_mtx};
    _gate_closed = false;
  }
  _gate_cv.notify_all();
}

size_t mock_s3_control_client::heads_issued() const
{
  std::lock_guard lk{_mtx};
  return _heads_issued;
}

size_t mock_s3_control_client::range_gets_issued() const
{
  std::lock_guard lk{_mtx};
  return _range_gets_issued;
}

uint64_t mock_s3_control_client::attempts_total() const noexcept
{
  std::lock_guard lk{_mtx};
  return static_cast<uint64_t>(_heads_issued + _range_gets_issued);
}

uint64_t mock_s3_control_client::connections_total() const noexcept
{
  std::lock_guard lk{_mtx};
  return _connections_total;
}

head_result mock_s3_control_client::head(const rx_route& route)
{
  std::lock_guard lk{_mtx};
  ++_heads_issued;
  if (!_connected) {
    _connected = true;
    ++_connections_total;
  }
  head_result result;
  if (_next_transport_error) {
    result.outcome.transport_error = std::move(*_next_transport_error);
    _next_transport_error.reset();
    return result;
  }
  if (_next_status) {
    result.outcome.http_status = *_next_status;
    _next_status.reset();
    return result;
  }
  auto it = _objects.find({route.bucket, route.key});
  if (it == _objects.end()) {
    result.outcome.http_status = 404;
    return result;
  }
  result.outcome.http_status = 200;
  result.object_size         = it->second.size();
  return result;
}

range_get_result mock_s3_control_client::range_get(const rx_route& route,
                                                   size_t offset,
                                                   size_t size,
                                                   uint8_t* dst)
{
  std::unique_lock lk{_mtx};
  ++_range_gets_issued;
  if (!_connected) {
    _connected = true;
    ++_connections_total;
  }
  _gate_cv.wait(lk, [&] { return !_gate_closed; });

  range_get_result result;
  if (_fail_next_range_gets > 0) {
    --_fail_next_range_gets;
    result.outcome.http_status = _fail_next_status;  // exercises the host-plane retry
    return result;
  }
  if (_next_transport_error) {
    result.outcome.transport_error = std::move(*_next_transport_error);
    _next_transport_error.reset();
    return result;
  }
  if (_next_status) {
    result.outcome.http_status = *_next_status;
    _next_status.reset();
    return result;
  }
  auto it = _objects.find({route.bucket, route.key});
  if (it == _objects.end()) {
    result.outcome.http_status = 404;
    return result;
  }
  const auto& bytes = it->second;
  if (offset >= bytes.size()) {
    result.outcome.http_status = 416;
    return result;
  }
  const size_t n = std::min(size, bytes.size() - offset);
  std::memcpy(dst, bytes.data() + offset, n);
  result.outcome.http_status = 206;
  result.delivered_bytes     = n;
  if (_content_range_override) {
    result.content_range = std::move(*_content_range_override);
    _content_range_override.reset();
  } else {
    result.content_range = "bytes " + std::to_string(offset) + "-" +
                           std::to_string(offset + n - 1) + "/" + std::to_string(bytes.size());
  }
  return result;
}

// ---- mock_rdma_data_session_factory ----------------------------------------

/// Thin per-worker view over the factory's shared state (the friend named in
/// the header).  Sessions carry no state of their own.
class mock_rdma_data_session final : public rdma_data_session {
 public:
  explicit mock_rdma_data_session(std::shared_ptr<mock_rdma_data_session_factory> state)
    : _state(std::move(state))
  {
    _state->on_session_created();
  }
  ~mock_rdma_data_session() override { _state->on_session_destroyed(); }
  void register_memory(void* /*base*/, size_t /*bytes*/) override { _state->count_register(); }
  void deregister_memory(void* /*base*/) noexcept override { _state->count_deregister(); }
  data_get_result get(const rx_route& route, size_t offset, size_t size, void* dst) override
  {
    return _state->serve_get(route, offset, size, dst);
  }

 private:
  std::shared_ptr<mock_rdma_data_session_factory> _state;
};

void mock_rdma_data_session_factory::put_object(std::string bucket,
                                                std::string key,
                                                std::vector<std::uint8_t> bytes)
{
  std::lock_guard lk{_mtx};
  _objects[{std::move(bucket), std::move(key)}] = std::move(bytes);
}

void mock_rdma_data_session_factory::script_result(data_get_result result)
{
  std::lock_guard lk{_mtx};
  _scripted.push_back(std::move(result));
}

void mock_rdma_data_session_factory::close_get_gate()
{
  std::lock_guard lk{_mtx};
  _gate_closed = true;
}

void mock_rdma_data_session_factory::open_get_gate()
{
  {
    std::lock_guard lk{_mtx};
    _gate_closed = false;
  }
  _gate_cv.notify_all();
}

void mock_rdma_data_session_factory::fail_gets(std::string transport_error)
{
  std::lock_guard lk{_mtx};
  _fail_message = std::move(transport_error);
}

void mock_rdma_data_session_factory::throw_gets(std::string what)
{
  std::lock_guard lk{_mtx};
  _throw_message = std::move(what);
}

void mock_rdma_data_session_factory::short_write(size_t bytes)
{
  std::lock_guard lk{_mtx};
  _short_write = bytes;
}

void mock_rdma_data_session_factory::fail_acquire(std::string what)
{
  std::lock_guard lk{_mtx};
  _fail_acquire = std::move(what);
}

void mock_rdma_data_session_factory::null_acquire_after(size_t count)
{
  std::lock_guard lk{_mtx};
  _null_acquire_after = count;
}

void mock_rdma_data_session_factory::fail_register(size_t count, std::string message)
{
  std::lock_guard lk{_mtx};
  _fail_register_count   = count;
  _fail_register_message = std::move(message);
}

size_t mock_rdma_data_session_factory::live_sessions() const
{
  return _live_sessions.load(std::memory_order_relaxed);
}

size_t mock_rdma_data_session_factory::gets_issued() const
{
  std::lock_guard lk{_mtx};
  return _gets_issued;
}

size_t mock_rdma_data_session_factory::peak_concurrent_gets() const
{
  return _peak_concurrent.load(std::memory_order_relaxed);
}

size_t mock_rdma_data_session_factory::register_count() const
{
  std::lock_guard lk{_mtx};
  return _register_count;
}

size_t mock_rdma_data_session_factory::deregister_count() const
{
  std::lock_guard lk{_mtx};
  return _deregister_count;
}

size_t mock_rdma_data_session_factory::acquired_total() const
{
  std::lock_guard lk{_mtx};
  return _acquire_threads.size();
}

std::vector<std::thread::id> mock_rdma_data_session_factory::acquisition_thread_ids() const
{
  std::lock_guard lk{_mtx};
  return _acquire_threads;
}

std::unique_ptr<rdma_data_session> mock_rdma_data_session_factory::acquire()
{
  {
    std::lock_guard lk{_mtx};
    if (_fail_acquire) { throw std::runtime_error(*_fail_acquire); }
    const size_t call = _acquire_calls++;
    if (_null_acquire_after && call >= *_null_acquire_after) { return nullptr; }
    _acquire_threads.push_back(std::this_thread::get_id());
  }
  return std::make_unique<mock_rdma_data_session>(shared_from_this());
}

void mock_rdma_data_session_factory::count_register()
{
  std::lock_guard lk{_mtx};
  ++_register_count;
  if (_fail_register_count > 0) {
    --_fail_register_count;
    throw std::runtime_error(_fail_register_message);
  }
}

void mock_rdma_data_session_factory::count_deregister()
{
  std::lock_guard lk{_mtx};
  ++_deregister_count;
}

data_get_result mock_rdma_data_session_factory::serve_get(const rx_route& route,
                                                          size_t offset,
                                                          size_t size,
                                                          void* dst)
{
  // Atomic in-flight accounting: the guard's decrement must be safe after the
  // lock is released for the delivery copy (and even if that copy throws).
  const size_t now = _concurrent.fetch_add(1, std::memory_order_relaxed) + 1;
  for (size_t peak = _peak_concurrent.load(std::memory_order_relaxed);
       now > peak &&
       !_peak_concurrent.compare_exchange_weak(peak, now, std::memory_order_relaxed);) {}
  struct concurrent_release {
    std::atomic<size_t>* counter;
    ~concurrent_release() { counter->fetch_sub(1, std::memory_order_relaxed); }
  } release{&_concurrent};

  // The delivery copy happens after the lock is released, reading a PRIVATE
  // snapshot rather than a reference into _objects, so a concurrent
  // put_object of the same key cannot free the buffer under it.
  std::vector<std::uint8_t> payload;
  data_get_result result;
  {
    std::unique_lock lk{_mtx};
    ++_gets_issued;
    _gate_cv.wait(lk, [&] { return !_gate_closed; });

    if (_throw_message) { throw std::runtime_error(*_throw_message); }
    if (!_scripted.empty()) {
      result = std::move(_scripted.front());
      _scripted.pop_front();
      // A scripted successful completion still delivers the seeded bytes so
      // byte-exact assertions can ride on it.
      if (result.commit == data_commit_state::completed && result.delivered_bytes > 0) {
        if (auto it = _objects.find({route.bucket, route.key});
            it != _objects.end() && offset < it->second.size()) {
          const size_t n = std::min(size, it->second.size() - offset);
          payload.assign(it->second.begin() + static_cast<std::ptrdiff_t>(offset),
                         it->second.begin() + static_cast<std::ptrdiff_t>(offset + n));
        }
      }
    } else if (_fail_message) {
      result.commit          = data_commit_state::sent_unknown;
      result.transport_error = *_fail_message;
    } else if (auto it = _objects.find({route.bucket, route.key}); it == _objects.end()) {
      result.commit      = data_commit_state::completed;
      result.http_status = 404;
      result.reply_tag   = k_mock_reply_tag;
    } else {
      const auto& bytes = it->second;
      const size_t full = offset < bytes.size() ? std::min(size, bytes.size() - offset) : 0;
      const size_t n    = _short_write ? std::min(*_short_write, full) : full;
      payload.assign(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                     bytes.begin() + static_cast<std::ptrdiff_t>(offset + n));
      result.commit          = data_commit_state::completed;
      result.delivered_bytes = n;
      result.http_status     = 200;
      result.reply_tag       = k_mock_reply_tag;
    }
  }
  if (!payload.empty()) { copy_to_destination(dst, payload.data(), payload.size()); }
  return result;
}

}  // namespace sirius::io::rdma
