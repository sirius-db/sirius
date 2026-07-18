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

#include "catch.hpp"
#include "exec/semi_future.hpp"
#include "io/io_request.hpp"
#include "io/object_store_config.hpp"
#include "io/rdma/cuobj_rdma_reactor.hpp"
#include "io/rdma/mock_rdma_client.hpp"
#include "io/rdma/rdma_admission_gate.hpp"
#include "io/rdma/rdma_client.hpp"
#include "io/s3/s3_rdma_ioctx.hpp"
#include "io/sirius_datasource.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace s3_rdma_admission_tests {

using sirius::exec::semi_future;
using sirius::io::object_store_config;
using sirius::io::request_manager;
using sirius::io::rdma::admission_gate;
using sirius::io::rdma::cuda_delivery_ops;
using sirius::io::rdma::cuobj_rdma_reactor;
using sirius::io::rdma::mock_rdma_client;
using sirius::io::rdma::rdma_client;
using sirius::io::s3::s3_rdma_ioctx;
using namespace std::chrono_literals;

constexpr std::size_t k_slot_size       = 64UL << 10;
constexpr std::size_t k_default_workers = 1;
constexpr std::string_view k_bucket     = "bucket";
constexpr auto k_ready_timeout          = 5s;
constexpr auto k_block_probe            = 50ms;

static_assert(std::is_nothrow_move_constructible_v<admission_gate::envelope>);
static_assert(std::is_nothrow_destructible_v<admission_gate::envelope>);

std::size_t ceil_div(std::size_t value, std::size_t divisor)
{
  return (value + divisor - 1) / divisor;
}

std::exception_ptr test_error(std::string message)
{
  return std::make_exception_ptr(std::runtime_error(std::move(message)));
}

std::string exception_message(std::exception_ptr error)
{
  if (!error) { return {}; }
  try {
    std::rethrow_exception(error);
  } catch (std::exception const& e) {
    return e.what();
  } catch (...) {
    return "non-standard exception";
  }
}

template <typename Fn>
std::string captured_exception(Fn&& fn)
{
  try {
    std::forward<Fn>(fn)();
  } catch (std::exception const& e) {
    return e.what();
  } catch (...) {
    return "non-standard exception";
  }
  return {};
}

std::string consume_error(semi_future<std::size_t>& future)
{
  try {
    (void)std::move(future).get(k_ready_timeout);
  } catch (std::exception const& e) {
    return e.what();
  } catch (...) {
    return "non-standard exception";
  }
  return {};
}

std::string consume_error(std::future<std::size_t>& future)
{
  if (future.wait_for(k_ready_timeout) != std::future_status::ready) { return "future timed out"; }
  try {
    (void)future.get();
  } catch (std::exception const& e) {
    return e.what();
  } catch (...) {
    return "non-standard exception";
  }
  return {};
}

struct gate_request {
  std::unique_ptr<std::uint8_t[]> destination;
  admission_gate::envelope envelope;
  semi_future<std::size_t> future;
};

gate_request make_gate_request(std::string key,
                               std::size_t size       = k_slot_size,
                               std::size_t slot_bytes = k_slot_size)
{
  auto destination = std::make_unique<std::uint8_t[]>(std::max<std::size_t>(size, 1));
  auto manager     = std::make_shared<request_manager>(size, ceil_div(size, slot_bytes));
  auto future      = manager->get_future();
  admission_gate::envelope envelope{std::string{k_bucket},
                                    std::move(key),
                                    0,
                                    size,
                                    destination.get(),
                                    false,
                                    rmm::cuda_stream_view{},
                                    -1,
                                    std::move(manager),
                                    slot_bytes};
  return gate_request{std::move(destination), std::move(envelope), std::move(future)};
}

void complete_batch(admission_gate& gate,
                    admission_gate::drain_batch& batch,
                    std::exception_ptr error)
{
  if (!batch.has_token()) {
    CHECK(batch.empty());
    return;
  }
  batch.error_complete_all(std::move(error));
  auto token = std::move(batch).take_token();
  gate.complete_drain(std::move(token));
}

struct marker_state {
  std::atomic<std::size_t> calls{0};
};

void record_marker(void* opaque) noexcept
{
  static_cast<marker_state*>(opaque)->calls.fetch_add(1, std::memory_order_relaxed);
}

class observing_rdma_client final : public rdma_client {
 public:
  observing_rdma_client() : _inner(std::make_shared<mock_rdma_client>()) {}

  void put_object(std::string key, std::vector<std::uint8_t> bytes)
  {
    _inner->put_object(std::string{k_bucket}, std::move(key), std::move(bytes));
  }

  void close_get_gate() { _inner->close_gate(); }
  void open_get_gate() { _inner->open_gate(); }
  void fail_gets(std::string message) { _inner->fail_gets(std::move(message)); }
  void short_read(std::size_t bytes) { _inner->short_read(bytes); }

  [[nodiscard]] std::size_t get_count() const
  {
    std::lock_guard lock{_mutex};
    return _get_count;
  }

  [[nodiscard]] std::size_t register_count() const
  {
    std::lock_guard lock{_mutex};
    return _register_count;
  }

  [[nodiscard]] std::size_t deregister_count() const
  {
    std::lock_guard lock{_mutex};
    return _deregister_count;
  }

  bool wait_for_get_count(std::size_t expected, std::chrono::milliseconds timeout = k_ready_timeout)
  {
    std::unique_lock lock{_mutex};
    return _get_cv.wait_for(lock, timeout, [&] { return _get_count >= expected; });
  }

  std::size_t head(std::string_view bucket, std::string_view key) override
  {
    return _inner->head(bucket, key);
  }

  std::size_t get(std::string_view bucket,
                  std::string_view key,
                  std::size_t offset,
                  std::size_t size,
                  void* dst) override
  {
    {
      std::lock_guard lock{_mutex};
      ++_get_count;
    }
    _get_cv.notify_all();
    return _inner->get(bucket, key, offset, size, dst);
  }

  void register_memory(void* base, std::size_t bytes) override
  {
    {
      std::lock_guard lock{_mutex};
      ++_register_count;
    }
    _inner->register_memory(base, bytes);
  }

  void deregister_memory(void* base) noexcept override
  {
    {
      std::lock_guard lock{_mutex};
      ++_deregister_count;
    }
    _inner->deregister_memory(base);
  }

 private:
  std::shared_ptr<mock_rdma_client> _inner;
  mutable std::mutex _mutex;
  std::condition_variable _get_cv;
  std::size_t _get_count{0};
  std::size_t _register_count{0};
  std::size_t _deregister_count{0};
};

class blocking_event_wait {
 public:
  cudaError_t wait(cudaEvent_t event)
  {
    {
      std::unique_lock lock{_mutex};
      _entered = true;
      _cv.notify_all();
      _cv.wait(lock, [&] { return _released; });
    }
    return cudaEventSynchronize(event);
  }

  bool wait_until_entered(std::chrono::milliseconds timeout = k_ready_timeout)
  {
    std::unique_lock lock{_mutex};
    return _cv.wait_for(lock, timeout, [&] { return _entered; });
  }

  void release()
  {
    {
      std::lock_guard lock{_mutex};
      _released = true;
    }
    _cv.notify_all();
  }

 private:
  std::mutex _mutex;
  std::condition_variable _cv;
  bool _entered{false};
  bool _released{false};
};

object_store_config reactor_object_store_config(std::size_t max_inflight = k_default_workers)
{
  object_store_config cfg;
  cfg.s3_transport            = object_store_config::transport::RDMA;
  cfg.s3_rdma_max_inflight    = max_inflight;
  cfg.s3_rdma_arena_slot_size = k_slot_size;
  cfg.endpoint                = "mock-rdma-endpoint";
  cfg.region                  = "us-east-1";
  cfg.access_key              = "mock-access-key";
  cfg.secret_key              = "mock-secret-key";
  return cfg;
}

std::vector<std::uint8_t> payload_bytes(std::uint8_t salt = 23)
{
  std::vector<std::uint8_t> payload(k_slot_size);
  for (std::size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<std::uint8_t>((i * 131U + salt) & 0xffU);
  }
  return payload;
}

bool cuda_device_available()
{
  int count       = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count == 0) {
    WARN("Skipping S3 RDMA admission-gate reactor test: no CUDA device is available");
    return false;
  }
  REQUIRE(cudaSetDevice(0) == cudaSuccess);
  return true;
}

std::shared_ptr<s3_rdma_ioctx> make_started_ioctx(std::shared_ptr<observing_rdma_client> client,
                                                  cuda_delivery_ops delivery = {},
                                                  std::size_t max_inflight   = k_default_workers)
{
  auto ctx = std::make_shared<s3_rdma_ioctx>(
    reactor_object_store_config(max_inflight), std::move(client), std::move(delivery));
  ctx->start();
  return ctx;
}

std::unique_ptr<sirius::io::sirius_datasource> open_datasource(
  std::shared_ptr<s3_rdma_ioctx> const& ctx, std::string const& key)
{
  return ctx->open_datasource("s3://" + std::string{k_bucket} + "/" + key);
}

std::future<std::size_t> issue_device_read(sirius::io::sirius_datasource& datasource,
                                           rmm::device_buffer& destination,
                                           rmm::cuda_stream_view stream)
{
  return datasource.device_read_async(
    0, destination.size(), static_cast<std::uint8_t*>(destination.data()), stream);
}

}  // namespace s3_rdma_admission_tests

TEST_CASE("s3_rdma AC1 incomplete request manager resolves an internal error",
          "[s3][rdma][admission]")
{
  using namespace s3_rdma_admission_tests;

  auto manager = std::make_shared<request_manager>(k_slot_size, 2);
  auto future  = manager->get_future();
  manager->chunk_complete(k_slot_size / 2);
  manager.reset();

  auto const message = consume_error(future);
  CHECK_FALSE(message.empty());
  CHECK((message.find("internal") != std::string::npos ||
         message.find("incomplete") != std::string::npos ||
         message.find("chunk") != std::string::npos));
}

TEST_CASE("s3_rdma AC2 envelope admission blocks only at the configured cap",
          "[s3][rdma][admission]")
{
  using namespace s3_rdma_admission_tests;

  admission_gate gate{1};
  auto first  = make_gate_request("cap-first");
  auto second = make_gate_request("cap-second");
  gate.submit(std::move(first.envelope));

  std::promise<void> submit_started;
  auto started = submit_started.get_future();
  auto submitter =
    std::async(std::launch::async, [&, envelope = std::move(second.envelope)]() mutable {
      submit_started.set_value();
      return captured_exception([&] { gate.submit(std::move(envelope)); });
    });

  auto const submit_started_ready = started.wait_for(k_ready_timeout) == std::future_status::ready;
  auto const blocked_at_cap =
    submit_started_ready && submitter.wait_for(k_block_probe) != std::future_status::ready;

  auto claimed = gate.claim();
  if (!claimed) {
    auto batch = gate.fail_stop(test_error("AC2 cleanup"));
    complete_batch(gate, batch, test_error("AC2 cleanup"));
    (void)submitter.wait_for(k_ready_timeout);
    FAIL("the first envelope was not claimable");
  }
  claimed->report_error(test_error("AC2 first complete"));
  claimed.reset();

  auto const woke_after_claim = submitter.wait_for(k_ready_timeout) == std::future_status::ready;
  if (!woke_after_claim) {
    auto batch = gate.fail_stop(test_error("AC2 blocked submitter cleanup"));
    complete_batch(gate, batch, test_error("AC2 blocked submitter cleanup"));
  }
  REQUIRE(submitter.wait_for(k_ready_timeout) == std::future_status::ready);
  CHECK(submitter.get().empty());
  CHECK(submit_started_ready);
  CHECK(blocked_at_cap);
  CHECK(woke_after_claim);

  auto batch = gate.begin_close();
  REQUIRE(batch.size() == 1);
  complete_batch(gate, batch, test_error("AC2 close"));
  gate.await_closed();

  CHECK_FALSE(consume_error(first.future).empty());
  CHECK_FALSE(consume_error(second.future).empty());
}

TEST_CASE("s3_rdma AC3 submit commit is no-throw and terminal wake consumes no slot",
          "[s3][rdma][admission]")
{
  using namespace s3_rdma_admission_tests;

  SECTION("terminal submit never enters the queue")
  {
    admission_gate gate{1};
    auto batch = gate.begin_close();
    CHECK(batch.empty());
    CHECK_FALSE(batch.has_token());

    auto rejected      = make_gate_request("already-terminal");
    auto const message = captured_exception([&] { gate.submit(std::move(rejected.envelope)); });
    CHECK_FALSE(message.empty());
    CHECK_FALSE(gate.claim().has_value());
    gate.await_closed();
    CHECK_FALSE(consume_error(rejected.future).empty());
  }

  SECTION("fail-stop wakes a cap waiter without admitting its envelope")
  {
    admission_gate gate{1};
    auto admitted = make_gate_request("admitted");
    auto rejected = make_gate_request("blocked-then-failed");
    gate.submit(std::move(admitted.envelope));

    std::promise<void> submit_started;
    auto started = submit_started.get_future();
    auto submitter =
      std::async(std::launch::async, [&, envelope = std::move(rejected.envelope)]() mutable {
        submit_started.set_value();
        return captured_exception([&] { gate.submit(std::move(envelope)); });
      });
    auto const submit_started_ready =
      started.wait_for(k_ready_timeout) == std::future_status::ready;
    auto const blocked_at_cap =
      submit_started_ready && submitter.wait_for(k_block_probe) != std::future_status::ready;

    auto const fatal = test_error("AC3 terminal wake");
    auto batch       = gate.fail_stop(fatal);
    REQUIRE(batch.size() == 1);
    CHECK(batch.has_token());
    REQUIRE(submitter.wait_for(k_ready_timeout) == std::future_status::ready);
    CHECK(submitter.get().find("AC3 terminal wake") != std::string::npos);
    CHECK(submit_started_ready);
    CHECK(blocked_at_cap);

    complete_batch(gate, batch, fatal);
    gate.await_closed();
    CHECK_FALSE(consume_error(admitted.future).empty());
    CHECK_FALSE(consume_error(rejected.future).empty());
  }
}

TEST_CASE("s3_rdma AC4 claim guard keeps close pending until error publication",
          "[s3][rdma][admission]")
{
  using namespace s3_rdma_admission_tests;

  admission_gate gate{1};
  auto request = make_gate_request("claimed-not-issued");
  gate.submit(std::move(request.envelope));
  auto claimed = gate.claim();
  REQUIRE(claimed.has_value());

  auto batch = gate.begin_close();
  CHECK(batch.empty());
  CHECK_FALSE(batch.has_token());

  std::promise<void> waiter_started;
  auto started                    = waiter_started.get_future();
  auto waiter                     = std::async(std::launch::async, [&] {
    waiter_started.set_value();
    gate.await_closed();
    return request.future.is_ready();
  });
  auto const waiter_started_ready = started.wait_for(k_ready_timeout) == std::future_status::ready;
  auto const blocked_on_claim =
    waiter_started_ready && waiter.wait_for(k_block_probe) != std::future_status::ready;

  claimed->report_error(test_error("AC4 claimed abort"));
  claimed.reset();

  REQUIRE(waiter.wait_for(k_ready_timeout) == std::future_status::ready);
  CHECK(waiter.get());
  CHECK(waiter_started_ready);
  CHECK(blocked_on_claim);
  CHECK(consume_error(request.future).find("AC4 claimed abort") != std::string::npos);
}

TEST_CASE("s3_rdma AC5 every pre-GET terminal path publishes error before close",
          "[s3][rdma][admission]")
{
  using namespace s3_rdma_admission_tests;

  auto exercise_abort = [](bool at_get_acquire) {
    admission_gate gate{1};
    auto request = make_gate_request(at_get_acquire ? "acquire-get-abort" : "creation-abort");
    gate.submit(std::move(request.envelope));
    auto claimed = gate.claim();
    REQUIRE(claimed.has_value());

    auto batch = gate.begin_close();
    CHECK(batch.empty());
    CHECK_FALSE(batch.has_token());

    std::promise<void> waiter_started;
    auto started = waiter_started.get_future();
    auto waiter  = std::async(std::launch::async, [&] {
      waiter_started.set_value();
      gate.await_closed();
      return request.future.is_ready();
    });
    auto const waiter_started_ready =
      started.wait_for(k_ready_timeout) == std::future_status::ready;
    auto const blocked =
      waiter_started_ready && waiter.wait_for(k_block_probe) != std::future_status::ready;

    std::string terminal_message;
    if (at_get_acquire) {
      terminal_message = captured_exception([&] { (void)gate.acquire_get(std::move(*claimed)); });
    } else {
      terminal_message = captured_exception([&] { (void)gate.enter_creation(); });
    }
    CHECK_FALSE(terminal_message.empty());

    claimed->report_error(
      test_error(at_get_acquire ? "AC5 acquire_get abort" : "AC5 enter_creation abort"));
    claimed.reset();

    REQUIRE(waiter.wait_for(k_ready_timeout) == std::future_status::ready);
    CHECK(waiter.get());
    CHECK(waiter_started_ready);
    CHECK(blocked);
    CHECK_FALSE(consume_error(request.future).empty());
  };

  SECTION("terminal at acquire_get") { exercise_abort(true); }
  SECTION("terminal before the first lazy arena creation") { exercise_abort(false); }
}

TEST_CASE("s3_rdma AC6 two-phase close never waits on the caller permit", "[s3][rdma][admission]")
{
  using namespace s3_rdma_admission_tests;

  admission_gate gate{1};
  std::optional<admission_gate::admission_permit> permit;
  permit.emplace(gate.acquire_control());

  auto batch = gate.begin_close();
  CHECK(batch.empty());
  CHECK_FALSE(batch.has_token());

  std::promise<void> first_started;
  std::promise<void> second_started;
  auto first_ready   = first_started.get_future();
  auto second_ready  = second_started.get_future();
  auto first_waiter  = std::async(std::launch::async, [&] {
    first_started.set_value();
    gate.await_closed();
  });
  auto second_waiter = std::async(std::launch::async, [&] {
    second_started.set_value();
    gate.await_closed();
  });
  auto const first_started_ready =
    first_ready.wait_for(k_ready_timeout) == std::future_status::ready;
  auto const second_started_ready =
    second_ready.wait_for(k_ready_timeout) == std::future_status::ready;
  auto const first_blocked =
    first_started_ready && first_waiter.wait_for(k_block_probe) != std::future_status::ready;
  auto const second_blocked =
    second_started_ready && second_waiter.wait_for(k_block_probe) != std::future_status::ready;

  auto repeated = gate.begin_close();
  CHECK(repeated.empty());
  CHECK_FALSE(repeated.has_token());
  permit.reset();

  REQUIRE(first_waiter.wait_for(k_ready_timeout) == std::future_status::ready);
  REQUIRE(second_waiter.wait_for(k_ready_timeout) == std::future_status::ready);
  first_waiter.get();
  second_waiter.get();
  CHECK(first_started_ready);
  CHECK(second_started_ready);
  CHECK(first_blocked);
  CHECK(second_blocked);
}

TEST_CASE("s3_rdma AC7 fatal after close remains the permanent terminal error",
          "[s3][rdma][admission]")
{
  using namespace s3_rdma_admission_tests;

  admission_gate gate{1};
  marker_state marker;
  gate.bind_arena_marker(record_marker, &marker);

  auto request = make_gate_request("fatal-after-close", 2 * k_slot_size);
  gate.submit(std::move(request.envelope));
  auto first  = gate.claim();
  auto second = gate.claim();
  REQUIRE(first.has_value());
  REQUIRE(second.has_value());

  std::optional<admission_gate::admission_permit> get_permit;
  get_permit.emplace(gate.acquire_get(std::move(*first)));
  auto close_batch = gate.begin_close();
  CHECK(close_batch.empty());

  auto const fatal = test_error("AC7 issued GET failed");
  auto fatal_batch = gate.fail_stop(fatal);
  CHECK(fatal_batch.empty());
  CHECK(gate.first_fatal() == fatal);
  CHECK(marker.calls.load(std::memory_order_relaxed) == 1);

  auto const control_error  = captured_exception([&] { (void)gate.acquire_control(); });
  auto const creation_error = captured_exception([&] { (void)gate.enter_creation(); });
  auto const get_error = captured_exception([&] { (void)gate.acquire_get(std::move(*second)); });
  CHECK(control_error.find("AC7 issued GET failed") != std::string::npos);
  CHECK(creation_error.find("AC7 issued GET failed") != std::string::npos);
  CHECK(get_error.find("AC7 issued GET failed") != std::string::npos);

  first->report_error(fatal);
  second->report_error(fatal);
  first.reset();
  second.reset();
  get_permit.reset();
  gate.await_closed();

  CHECK(consume_error(request.future).find("AC7 issued GET failed") != std::string::npos);
  CHECK(exception_message(gate.first_fatal()).find("AC7 issued GET failed") != std::string::npos);
}

TEST_CASE("s3_rdma AC8 exactly one concurrent transition owns the drain token",
          "[s3][rdma][admission]")
{
  using namespace s3_rdma_admission_tests;

  admission_gate gate{2};
  auto first  = make_gate_request("drain-first", 3 * k_slot_size);
  auto second = make_gate_request("drain-second", 2 * k_slot_size);
  gate.submit(std::move(first.envelope));
  gate.submit(std::move(second.envelope));

  auto const fatal = test_error("AC8 concurrent drain");
  std::barrier transition_start{3};
  auto closer  = std::async(std::launch::async, [&] {
    transition_start.arrive_and_wait();
    return gate.begin_close();
  });
  auto failure = std::async(std::launch::async, [&] {
    transition_start.arrive_and_wait();
    return gate.fail_stop(fatal);
  });
  transition_start.arrive_and_wait();

  auto close_batch = closer.get();
  auto fail_batch  = failure.get();
  CHECK(close_batch.has_token() != fail_batch.has_token());
  CHECK((close_batch.size() == 2 || fail_batch.size() == 2));
  CHECK((close_batch.empty() || fail_batch.empty()));

  std::promise<void> waiter_started;
  auto started                    = waiter_started.get_future();
  auto waiter                     = std::async(std::launch::async, [&] {
    waiter_started.set_value();
    gate.await_closed();
  });
  auto const waiter_started_ready = started.wait_for(k_ready_timeout) == std::future_status::ready;
  auto const blocked_on_drain =
    waiter_started_ready && waiter.wait_for(k_block_probe) != std::future_status::ready;

  auto finish_winner = [&](admission_gate::drain_batch& winner) {
    REQUIRE(winner.has_token());
    REQUIRE(winner.size() == 2);
    winner.error_complete_all(fatal);
    CHECK(first.future.is_ready());
    CHECK(second.future.is_ready());
    auto token = std::move(winner).take_token();
    gate.complete_drain(std::move(token));
  };
  if (close_batch.has_token()) {
    CHECK(fail_batch.empty());
    CHECK_FALSE(fail_batch.has_token());
    finish_winner(close_batch);
  } else {
    CHECK(close_batch.empty());
    CHECK_FALSE(close_batch.has_token());
    finish_winner(fail_batch);
  }

  REQUIRE(waiter.wait_for(k_ready_timeout) == std::future_status::ready);
  waiter.get();
  CHECK(waiter_started_ready);
  CHECK(blocked_on_drain);
  CHECK(consume_error(first.future).find("AC8 concurrent drain") != std::string::npos);
  CHECK(consume_error(second.future).find("AC8 concurrent drain") != std::string::npos);
}

TEST_CASE("s3_rdma AC9 poisoned gate fails both planes with one stable error",
          "[s3][rdma][admission]")
{
  using namespace s3_rdma_admission_tests;

  admission_gate gate{1};
  auto claimed_request = make_gate_request("poisoned-claimed");
  gate.submit(std::move(claimed_request.envelope));
  auto claimed = gate.claim();
  REQUIRE(claimed.has_value());

  auto const fatal = test_error("AC9 stable terminal");
  auto batch       = gate.fail_stop(fatal);
  CHECK(batch.empty());
  CHECK_FALSE(batch.has_token());

  auto rejected           = make_gate_request("poisoned-submit");
  auto const host_error   = captured_exception([&] { (void)gate.acquire_control(); });
  auto const submit_error = captured_exception([&] { gate.submit(std::move(rejected.envelope)); });
  auto const get_error = captured_exception([&] { (void)gate.acquire_get(std::move(*claimed)); });
  CHECK(host_error.find("AC9 stable terminal") != std::string::npos);
  CHECK(submit_error == host_error);
  CHECK(get_error == host_error);

  claimed->report_error(fatal);
  claimed.reset();
  gate.await_closed();
  CHECK(consume_error(claimed_request.future).find("AC9 stable terminal") != std::string::npos);
  CHECK_FALSE(consume_error(rejected.future).empty());
}

TEST_CASE("s3_rdma AC10 one RDMA failure fail-stops without retry or arena release",
          "[s3][rdma][admission][gpu]")
{
  using namespace s3_rdma_admission_tests;
  if (!cuda_device_available()) { return; }

  auto exercise_failure = [](bool short_read) {
    auto payload = payload_bytes(short_read ? 31 : 29);
    auto client  = std::make_shared<observing_rdma_client>();
    client->put_object(short_read ? "one-shot-short" : "one-shot-throw", payload);
    if (short_read) {
      client->short_read(payload.size() / 2);
    } else {
      client->fail_gets("AC10 transport failure");
    }

    auto ctx        = make_started_ioctx(client);
    auto datasource = open_datasource(ctx, short_read ? "one-shot-short" : "one-shot-throw");
    rmm::cuda_stream stream;
    rmm::device_buffer device(payload.size(), stream);
    auto future        = issue_device_read(*datasource, device, stream);
    auto const message = consume_error(future);
    CHECK_FALSE(message.empty());
    CHECK(client->get_count() == 1);
    CHECK(client->register_count() == 1);

    auto const snapshot = ctx->perf_snapshot();
    CHECK(snapshot.requests_total == 1);
    CHECK(snapshot.retries_total == 0);
    CHECK(snapshot.short_read_total == (short_read ? 1 : 0));
    CHECK(snapshot.error_total == 1);
    CHECK(snapshot.fail_stop_total == 1);
    CHECK(snapshot.arena_leak_total == 1);

    datasource.reset();
    ctx.reset();
    CHECK(client->deregister_count() == 0);
  };

  SECTION("transport exception") { exercise_failure(false); }
  SECTION("short completion") { exercise_failure(true); }
}

TEST_CASE("s3_rdma AC11 teardown waits for GET and D2D and never frees a failed arena",
          "[s3][rdma][admission][gpu]")
{
  using namespace s3_rdma_admission_tests;
  if (!cuda_device_available()) { return; }

  SECTION("normal close waits for an issued GET")
  {
    auto payload = payload_bytes(37);
    auto client  = std::make_shared<observing_rdma_client>();
    client->put_object("teardown-get", payload);
    client->close_get_gate();
    auto ctx        = make_started_ioctx(client);
    auto datasource = open_datasource(ctx, "teardown-get");
    rmm::cuda_stream stream;
    rmm::device_buffer device(payload.size(), stream);
    auto future            = issue_device_read(*datasource, device, stream);
    auto const get_entered = client->wait_for_get_count(1);
    if (!get_entered) { client->open_get_gate(); }
    REQUIRE(get_entered);

    std::promise<void> shutdown_started;
    auto started  = shutdown_started.get_future();
    auto shutdown = std::async(std::launch::async, [&] {
      shutdown_started.set_value();
      ctx->shutdown();
    });
    auto const shutdown_started_ready =
      started.wait_for(k_ready_timeout) == std::future_status::ready;
    auto const blocked =
      shutdown_started_ready && shutdown.wait_for(k_block_probe) != std::future_status::ready;
    CHECK(client->deregister_count() == 0);

    client->open_get_gate();
    REQUIRE(future.wait_for(k_ready_timeout) == std::future_status::ready);
    CHECK(future.get() == payload.size());
    REQUIRE(shutdown.wait_for(k_ready_timeout) == std::future_status::ready);
    shutdown.get();
    CHECK(shutdown_started_ready);
    CHECK(blocked);
    CHECK(client->deregister_count() == 0);

    datasource.reset();
    ctx.reset();
    CHECK(client->deregister_count() == 1);
  }

  SECTION("normal close waits for D2D confirmation")
  {
    auto payload = payload_bytes(41);
    auto client  = std::make_shared<observing_rdma_client>();
    client->put_object("teardown-d2d", payload);
    auto event_wait = std::make_shared<blocking_event_wait>();
    cuda_delivery_ops ops;
    ops.event_synchronize = [event_wait](cudaEvent_t event) { return event_wait->wait(event); };

    auto ctx        = make_started_ioctx(client, std::move(ops));
    auto datasource = open_datasource(ctx, "teardown-d2d");
    rmm::cuda_stream stream;
    rmm::device_buffer device(payload.size(), stream);
    auto future                   = issue_device_read(*datasource, device, stream);
    auto const event_wait_entered = event_wait->wait_until_entered();
    if (!event_wait_entered) { event_wait->release(); }
    REQUIRE(event_wait_entered);

    std::promise<void> shutdown_started;
    auto started  = shutdown_started.get_future();
    auto shutdown = std::async(std::launch::async, [&] {
      shutdown_started.set_value();
      ctx->shutdown();
    });
    auto const shutdown_started_ready =
      started.wait_for(k_ready_timeout) == std::future_status::ready;
    auto const blocked =
      shutdown_started_ready && shutdown.wait_for(k_block_probe) != std::future_status::ready;
    CHECK(client->deregister_count() == 0);

    event_wait->release();
    REQUIRE(future.wait_for(k_ready_timeout) == std::future_status::ready);
    CHECK(future.get() == payload.size());
    REQUIRE(shutdown.wait_for(k_ready_timeout) == std::future_status::ready);
    shutdown.get();
    CHECK(shutdown_started_ready);
    CHECK(blocked);

    datasource.reset();
    ctx.reset();
    CHECK(client->deregister_count() == 1);
  }

  SECTION("fail-stop leaves the registered arena non-freeable")
  {
    auto payload = payload_bytes(43);
    auto client  = std::make_shared<observing_rdma_client>();
    client->put_object("teardown-failed", payload);
    client->fail_gets("AC11 fail-stop");
    auto ctx        = make_started_ioctx(client);
    auto datasource = open_datasource(ctx, "teardown-failed");
    rmm::cuda_stream stream;
    rmm::device_buffer device(payload.size(), stream);
    auto future = issue_device_read(*datasource, device, stream);
    CHECK(consume_error(future).find("AC11 fail-stop") != std::string::npos);

    datasource.reset();
    ctx.reset();
    CHECK(client->register_count() == 1);
    CHECK(client->deregister_count() == 0);
  }
}

TEST_CASE("s3_rdma AC12 admission and slot metrics count logical requests",
          "[s3][rdma][admission][metrics][gpu]")
{
  using namespace s3_rdma_admission_tests;
  if (!cuda_device_available()) { return; }

  constexpr std::size_t request_count = 6;
  auto payload                        = payload_bytes(47);
  auto client                         = std::make_shared<observing_rdma_client>();
  client->put_object("admission-metrics", payload);
  client->close_get_gate();
  auto ctx        = make_started_ioctx(client, cuda_delivery_ops{}, 1);
  auto datasource = open_datasource(ctx, "admission-metrics");
  rmm::cuda_stream stream;

  std::vector<std::unique_ptr<rmm::device_buffer>> buffers;
  buffers.reserve(request_count);
  for (std::size_t i = 0; i < request_count; ++i) {
    buffers.push_back(std::make_unique<rmm::device_buffer>(payload.size(), stream));
  }

  std::vector<std::future<std::size_t>> futures;
  futures.reserve(request_count);
  futures.push_back(issue_device_read(*datasource, *buffers[0], stream));
  auto const get_entered = client->wait_for_get_count(1);
  if (!get_entered) { client->open_get_gate(); }
  REQUIRE(get_entered);
  for (std::size_t i = 1; i < request_count - 1; ++i) {
    futures.push_back(issue_device_read(*datasource, *buffers[i], stream));
  }

  std::promise<void> submit_started;
  auto started                    = submit_started.get_future();
  auto last_submit                = std::async(std::launch::async, [&] {
    submit_started.set_value();
    return issue_device_read(*datasource, *buffers.back(), stream);
  });
  auto const submit_started_ready = started.wait_for(k_ready_timeout) == std::future_status::ready;
  auto const blocked_at_cap =
    submit_started_ready && last_submit.wait_for(k_block_probe) != std::future_status::ready;

  client->open_get_gate();
  REQUIRE(last_submit.wait_for(k_ready_timeout) == std::future_status::ready);
  futures.push_back(last_submit.get());
  REQUIRE(futures.size() == request_count);
  for (auto& future : futures) {
    REQUIRE(future.wait_for(k_ready_timeout) == std::future_status::ready);
    CHECK(future.get() == payload.size());
  }

  auto const snapshot = ctx->perf_snapshot();
  CHECK(submit_started_ready);
  CHECK(blocked_at_cap);
  CHECK(snapshot.requests_total == request_count);
  CHECK(snapshot.bytes_total == request_count * payload.size());
  CHECK(snapshot.retries_total == 0);
  CHECK(snapshot.error_total == 0);
  CHECK(snapshot.envelope_wait_total == 1);
  CHECK(snapshot.envelope_wait_ns_total > 0);
  CHECK(snapshot.envelope_depth_peak == 4);
  CHECK(snapshot.slots_in_use_peak == 1);
  CHECK(snapshot.fail_stop_total == 0);
  CHECK(client->get_count() == request_count);
}

TEST_CASE("s3_rdma AC13 queue cap is sanitized after max inflight", "[s3][rdma][admission][config]")
{
  using namespace s3_rdma_admission_tests;

  SECTION("default follows the sanitized worker count")
  {
    cuobj_rdma_reactor::config cfg;
    cfg.max_inflight     = 0;
    auto const sanitized = sirius::io::rdma::sanitized(cfg);
    CHECK(sanitized.max_inflight == 1);
    CHECK(sanitized.queue_cap == 4);
  }

  SECTION("an explicit nonzero cap is preserved")
  {
    cuobj_rdma_reactor::config cfg;
    cfg.max_inflight = 2;
    cfg.queue_cap    = 3;
    CHECK(sirius::io::rdma::sanitized(cfg).queue_cap == 3);
  }

  SECTION("an explicit zero cap is rejected")
  {
    cuobj_rdma_reactor::config cfg;
    cfg.queue_cap = 0;
    CHECK_THROWS_AS(sirius::io::rdma::sanitized(cfg), std::invalid_argument);
  }

  SECTION("the derived default is overflow checked")
  {
    cuobj_rdma_reactor::config cfg;
    cfg.max_inflight = std::numeric_limits<std::size_t>::max() / 4 + 1;
    CHECK_THROWS_AS(sirius::io::rdma::sanitized(cfg), std::overflow_error);
  }
}
