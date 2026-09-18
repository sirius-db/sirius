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
#include "io/object_store_config.hpp"
#include "io/rdma/mock_rdma_client.hpp"
#include "io/rest/rest_ioctx.hpp"
#include "io/s3/s3_rdma_ioctx.hpp"
#include "io/sirius_datasource.hpp"
#include "rdma_test_transport.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace s3_rdma_listing_tests {

using sirius::io::object_store_config;
using sirius::io::rdma::head_result;
using sirius::io::rdma::list_page_result;
using sirius::io::rdma::mock_rdma_data_session_factory;
using sirius::io::rdma::range_get_result;
using sirius::io::rdma::rdma_transport_clients;
using sirius::io::rdma::rx_route;
using sirius::io::rdma::s3_control_client;
using sirius::io::s3::list_entry;
using sirius::io::s3::list_objects_v2_page;
using sirius::io::s3::s3_rdma_ioctx;
using sirius::test::rdma::mock_transport_fixture;
using namespace std::chrono_literals;

constexpr std::string_view k_bucket = "list-bucket";
constexpr std::string_view k_prefix = "prefix/";
constexpr std::size_t k_slot_size   = 64UL << 10;

object_store_config list_config()
{
  object_store_config cfg;
  cfg.endpoint                     = "http://control.example.invalid";
  cfg.region                       = "us-east-1";
  cfg.access_key                   = "list-access-key";
  cfg.secret_key                   = "list-secret-key";
  cfg.s3_signing_mode              = object_store_config::signing_mode::header;
  cfg.s3_transport                 = object_store_config::transport::RDMA;
  cfg.s3_rdma_max_inflight         = 1;
  cfg.s3_rdma_arena_slot_size      = k_slot_size;
  cfg.s3_rdma_data.endpoint        = "http://data.example.invalid";
  cfg.s3_rdma_data.region          = cfg.region;
  cfg.s3_rdma_data.access_key      = cfg.access_key;
  cfg.s3_rdma_data.secret_key      = cfg.secret_key;
  cfg.s3_rdma_data.s3_signing_mode = object_store_config::signing_mode::header;
  return cfg;
}

std::shared_ptr<mock_transport_fixture> seeded_listing_transport()
{
  auto transport = std::make_shared<mock_transport_fixture>();
  transport->put_object(std::string{k_bucket}, "prefix/a.bin", std::vector<std::uint8_t>(11, 1));
  transport->put_object(std::string{k_bucket}, "prefix/b.bin", std::vector<std::uint8_t>(22, 2));
  transport->put_object(std::string{k_bucket}, "prefix/c.bin", std::vector<std::uint8_t>(33, 3));
  transport->put_object(std::string{k_bucket}, "other.bin", std::vector<std::uint8_t>(44, 4));
  transport->put_object("other-bucket", "prefix/d.bin", std::vector<std::uint8_t>(55, 5));
  return transport;
}

std::shared_ptr<s3_rdma_ioctx> make_started_ioctx(
  std::shared_ptr<mock_transport_fixture> const& transport)
{
  auto ctx = std::make_shared<s3_rdma_ioctx>(list_config(), transport->clients());
  ctx->start();
  return ctx;
}

template <typename Fn>
std::string exception_message(Fn&& fn)
{
  try {
    std::forward<Fn>(fn)();
  } catch (std::exception const& error) {
    return error.what();
  }
  return {};
}

bool cuda_device_available()
{
  int count       = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count == 0) {
    WARN("Skipping S3 RDMA LIST fail-stop control: no CUDA device is available");
    return false;
  }
  REQUIRE(cudaSetDevice(0) == cudaSuccess);
  return true;
}

std::string ready_error(std::future<std::size_t>& future)
{
  REQUIRE(future.wait_for(5s) == std::future_status::ready);
  try {
    (void)future.get();
    FAIL("expected device read failure");
  } catch (std::exception const& error) {
    return error.what();
  }
  return {};
}

class blocking_list_control_client final : public s3_control_client {
 public:
  head_result head(rx_route const&) override { return {}; }

  range_get_result range_get(rx_route const&, std::size_t, std::size_t, std::uint8_t*) override
  {
    return {};
  }

  list_page_result list_page(std::string_view,
                             std::string_view,
                             std::size_t,
                             std::string_view) override
  {
    std::unique_lock lock{_mutex};
    ++_attempts;
    _entered = true;
    _cv.notify_all();
    _cv.wait(lock, [&] { return _released; });

    list_page_result result;
    result.outcome.http_status = 200;
    result.page.entries.push_back({"prefix/blocked.bin", 7});
    return result;
  }

  uint64_t attempts_total() const noexcept override
  {
    std::lock_guard lock{_mutex};
    return _attempts;
  }

  uint64_t connections_total() const noexcept override { return 1; }

  bool wait_until_entered(std::chrono::milliseconds timeout)
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
  mutable std::mutex _mutex;
  std::condition_variable _cv;
  bool _entered{false};
  bool _released{false};
  uint64_t _attempts{0};
};

void require_listing_succeeds(std::shared_ptr<s3_rdma_ioctx> const& ctx)
{
  std::vector<list_entry> entries;
  ctx->list_objects_paged(k_bucket, k_prefix, 1000, [&](list_objects_v2_page const& page) {
    entries.insert(entries.end(), page.entries.begin(), page.entries.end());
    return true;
  });
  REQUIRE(entries.size() == 3);
}

}  // namespace s3_rdma_listing_tests

TEST_CASE("s3_rdma LIST paginates through the mock control plane", "[s3][rdma][list]")
{
  using namespace s3_rdma_listing_tests;

  SECTION("continuation preserves ordered keys and sizes")
  {
    auto transport = seeded_listing_transport();
    auto ctx       = make_started_ioctx(transport);
    std::vector<list_objects_v2_page> pages;

    ctx->list_objects_paged(k_bucket, k_prefix, 2, [&](list_objects_v2_page const& page) {
      pages.push_back(page);
      return true;
    });

    REQUIRE(pages.size() == 2);
    REQUIRE(pages[0].entries.size() == 2);
    CHECK(pages[0].entries[0].key == "prefix/a.bin");
    CHECK(pages[0].entries[0].size == 11);
    CHECK(pages[0].entries[1].key == "prefix/b.bin");
    CHECK(pages[0].entries[1].size == 22);
    CHECK(pages[0].is_truncated);
    CHECK(pages[0].next_continuation_token == "prefix/b.bin");
    REQUIRE(pages[1].entries.size() == 1);
    CHECK(pages[1].entries[0].key == "prefix/c.bin");
    CHECK(pages[1].entries[0].size == 33);
    CHECK_FALSE(pages[1].is_truncated);
    CHECK(pages[1].next_continuation_token.empty());
    CHECK(transport->control->list_pages_issued() == 2);
    ctx->shutdown();
  }

  SECTION("a false sink result stops after the first page")
  {
    auto transport         = seeded_listing_transport();
    auto ctx               = make_started_ioctx(transport);
    std::size_t pages_seen = 0;

    ctx->list_objects_paged(k_bucket, k_prefix, 2, [&](list_objects_v2_page const&) {
      ++pages_seen;
      return false;
    });

    CHECK(pages_seen == 1);
    CHECK(transport->control->list_pages_issued() == 1);
    ctx->shutdown();
  }

  SECTION("the scanned-object cap throws instead of truncating")
  {
    auto transport = seeded_listing_transport();
    auto ctx       = make_started_ioctx(transport);

    auto const message = exception_message([&] {
      ctx->list_objects_paged(
        k_bucket, k_prefix, 2, [](list_objects_v2_page const&) { return true; }, 2);
    });

    CHECK(message.find("scanned") != std::string::npos);
    CHECK(message.find("2") != std::string::npos);
    CHECK(transport->control->list_pages_issued() == 2);
    CHECK(ctx->perf_snapshot().fail_stop_total == 0);
    ctx->shutdown();
  }
}

TEST_CASE("s3_rdma LIST reports the admission gate terminal error", "[s3][rdma][list][gpu]")
{
  using namespace s3_rdma_listing_tests;

  SECTION("first fatal is preserved")
  {
    if (!cuda_device_available()) { return; }

    auto transport = seeded_listing_transport();
    transport->data->throw_gets("LIST first-fatal sentinel");
    auto ctx        = make_started_ioctx(transport);
    auto datasource = ctx->open_datasource("s3://list-bucket/prefix/a.bin");
    rmm::cuda_stream stream;
    rmm::device_buffer destination(11, stream);
    auto read = datasource->device_read_async(
      0, destination.size(), static_cast<std::uint8_t*>(destination.data()), stream);
    auto const terminal_error = ready_error(read);

    auto const list_error = exception_message([&] {
      ctx->list_objects_paged(
        k_bucket, k_prefix, 2, [](list_objects_v2_page const&) { return true; });
    });

    CHECK(list_error == terminal_error);
    CHECK(transport->control->list_pages_issued() == 0);
    CHECK(ctx->perf_snapshot().fail_stop_total == 1);
    ctx->shutdown();
  }

  SECTION("plain shutdown preserves the stable closed error")
  {
    auto transport = seeded_listing_transport();
    auto ctx       = make_started_ioctx(transport);
    ctx->shutdown();

    auto const list_error = exception_message([&] {
      ctx->list_objects_paged(
        k_bucket, k_prefix, 2, [](list_objects_v2_page const&) { return true; });
    });

    CHECK(list_error == "s3_rdma admission_gate: transport closed");
    CHECK(transport->control->list_pages_issued() == 0);
  }
}

TEST_CASE("s3_rdma shutdown waits for an issued LIST page", "[s3][rdma][list]")
{
  using namespace s3_rdma_listing_tests;

  auto control = std::make_shared<blocking_list_control_client>();
  auto data    = std::make_shared<mock_rdma_data_session_factory>();
  auto ctx     = std::make_shared<s3_rdma_ioctx>(list_config(),
                                             rdma_transport_clients{control, std::move(data)});
  ctx->start();
  std::atomic<std::size_t> pages_seen{0};

  auto listing = std::async(std::launch::async, [&] {
    ctx->list_objects_paged(k_bucket, k_prefix, 1, [&](list_objects_v2_page const&) {
      pages_seen.fetch_add(1, std::memory_order_relaxed);
      return true;
    });
  });

  if (!control->wait_until_entered(2s)) {
    control->release();
    ctx->shutdown();
    REQUIRE(listing.wait_for(2s) == std::future_status::ready);
    (void)exception_message([&] { listing.get(); });
    FAIL("LIST never reached the control client");
  }

  auto closing = std::async(std::launch::async, [&] { ctx->shutdown(); });
  CHECK(closing.wait_for(100ms) == std::future_status::timeout);

  control->release();
  REQUIRE(listing.wait_for(2s) == std::future_status::ready);
  CHECK_NOTHROW(listing.get());
  REQUIRE(closing.wait_for(2s) == std::future_status::ready);
  CHECK_NOTHROW(closing.get());
  CHECK(pages_seen.load(std::memory_order_relaxed) == 1);
  CHECK(control->attempts_total() == 1);
}

TEST_CASE("s3_rdma LIST sink can close its ioctx without self-deadlock", "[s3][rdma][list]")
{
  using namespace s3_rdma_listing_tests;

  SECTION("continuing after shutdown reports the stable closed error")
  {
    auto transport         = seeded_listing_transport();
    auto ctx               = make_started_ioctx(transport);
    std::size_t pages_seen = 0;

    auto const message = exception_message([&] {
      ctx->list_objects_paged(k_bucket, k_prefix, 1, [&](list_objects_v2_page const&) {
        ++pages_seen;
        ctx->shutdown();
        return true;
      });
    });

    CHECK(message == "s3_rdma admission_gate: transport closed");
    CHECK(pages_seen == 1);
    CHECK(transport->control->list_pages_issued() == 1);
  }

  SECTION("stopping after shutdown returns normally")
  {
    auto transport         = seeded_listing_transport();
    auto ctx               = make_started_ioctx(transport);
    std::size_t pages_seen = 0;

    CHECK_NOTHROW(ctx->list_objects_paged(k_bucket, k_prefix, 1, [&](list_objects_v2_page const&) {
      ++pages_seen;
      ctx->shutdown();
      return false;
    }));

    CHECK(pages_seen == 1);
    CHECK(transport->control->list_pages_issued() == 1);
  }
}

TEST_CASE("s3_rdma LIST failures stay on the benign host plane", "[s3][rdma][list]")
{
  using namespace s3_rdma_listing_tests;

  SECTION("transport failure")
  {
    auto transport = seeded_listing_transport();
    auto ctx       = make_started_ioctx(transport);
    transport->control->fail_transport("injected LIST transport failure");

    auto const message = exception_message([&] {
      ctx->list_objects_paged(
        k_bucket, k_prefix, 1000, [](list_objects_v2_page const&) { return true; });
    });
    CHECK(message.find("injected LIST transport failure") != std::string::npos);
    CHECK(transport->control->list_pages_issued() == 1);
    CHECK(ctx->perf_snapshot().fail_stop_total == 0);

    require_listing_succeeds(ctx);
    CHECK(transport->control->list_pages_issued() == 2);
    CHECK(ctx->perf_snapshot().fail_stop_total == 0);
    ctx->shutdown();
  }

  SECTION("HTTP 403")
  {
    auto transport = seeded_listing_transport();
    auto ctx       = make_started_ioctx(transport);
    transport->control->respond_status(403);

    auto const message = exception_message([&] {
      ctx->list_objects_paged(
        k_bucket, k_prefix, 1000, [](list_objects_v2_page const&) { return true; });
    });
    CHECK(message.find("403") != std::string::npos);
    CHECK(transport->control->list_pages_issued() == 1);
    CHECK(ctx->perf_snapshot().fail_stop_total == 0);

    require_listing_succeeds(ctx);
    CHECK(transport->control->list_pages_issued() == 2);
    CHECK(ctx->perf_snapshot().fail_stop_total == 0);
    ctx->shutdown();
  }
}

TEST_CASE("s3_rdma LIST match cap equals the REST default", "[s3][rdma][list]")
{
  using namespace s3_rdma_listing_tests;

  auto transport = seeded_listing_transport();
  auto rdma      = std::make_shared<s3_rdma_ioctx>(list_config(), transport->clients());
  sirius::io::rest::rest_ioctx rest_default{0, nullptr};

  CHECK(rdma->list_max_matches() == rest_default.list_max_matches());
}
