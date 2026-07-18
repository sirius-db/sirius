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
#include "io/rdma/cuobj_rdma_client.hpp"
#include "io/rdma/cuobj_rdma_reactor.hpp"
#include "io/rdma/mock_rdma_client.hpp"
#include "io/rdma/rdma_client.hpp"
#include "io/s3/s3_rdma_ioctx.hpp"
#include "io/s3/sirius_sigv4_authorizer.hpp"
#include "io/s3/static_credentials.hpp"
#include "io/sirius_datasource.hpp"
#include "rdma_test_transport.hpp"
#include "utils/log_test_utils.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#ifdef SIRIUS_HAVE_TESTCONTAINERS
#include "utils/s3_container.hpp"

#include <curl/curl.h>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace s3_rdma_client_seam_tests {

using sirius::io::object_store_config;
using sirius::io::rdma::curl_s3_control_client;
using sirius::io::rdma::data_commit_state;
using sirius::io::rdma::data_get_result;
using sirius::io::rdma::rx_route;
using sirius::io::s3::s3_rdma_ioctx;
using sirius::test::rdma::mock_transport_fixture;
using sirius::test::rdma::seeded_mock_transport;
using namespace std::chrono_literals;

constexpr std::size_t k_slot_size   = 64UL << 10;
constexpr std::string_view k_bucket = "bucket";

object_store_config mock_config(std::size_t max_inflight = 1)
{
  object_store_config cfg;
  cfg.endpoint                     = "http://control.example.invalid";
  cfg.region                       = "us-east-1";
  cfg.access_key                   = "mock-access-key";
  cfg.secret_key                   = "mock-secret-key";
  cfg.s3_signing_mode              = object_store_config::signing_mode::header;
  cfg.s3_transport                 = object_store_config::transport::RDMA;
  cfg.s3_rdma_max_inflight         = max_inflight;
  cfg.s3_rdma_arena_slot_size      = k_slot_size;
  cfg.s3_rdma_data.endpoint        = "http://data.example.invalid";
  cfg.s3_rdma_data.region          = cfg.region;
  cfg.s3_rdma_data.access_key      = cfg.access_key;
  cfg.s3_rdma_data.secret_key      = cfg.secret_key;
  cfg.s3_rdma_data.s3_signing_mode = object_store_config::signing_mode::header;
  cfg.s3_rdma_data.tls_verify      = false;
  return cfg;
}

std::vector<std::uint8_t> pattern_bytes(std::size_t size, std::uint8_t salt = 61)
{
  std::vector<std::uint8_t> bytes(size);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>((i * 131U + salt) & 0xffU);
  }
  return bytes;
}

bool cuda_device_available()
{
  int count       = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count == 0) {
    WARN("Skipping S3 RDMA client-seam device test: no CUDA device is available");
    return false;
  }
  REQUIRE(cudaSetDevice(0) == cudaSuccess);
  return true;
}

std::shared_ptr<s3_rdma_ioctx> make_started_ioctx(
  std::shared_ptr<mock_transport_fixture> const& transport,
  object_store_config cfg                         = mock_config(),
  sirius::io::rdma::reply_tag_predicate predicate = &sirius::io::rdma::non_empty_reply_tag)
{
  auto ctx = std::make_shared<s3_rdma_ioctx>(
    std::move(cfg), transport->clients(predicate), sirius::io::rdma::cuda_delivery_ops{});
  ctx->start();
  return ctx;
}

std::unique_ptr<sirius::io::sirius_datasource> open_ds(std::shared_ptr<s3_rdma_ioctx> const& ctx,
                                                       std::string_view key)
{
  return ctx->open_datasource("s3://" + std::string{k_bucket} + "/" + std::string{key});
}

std::string ready_error(std::future<std::size_t>& future, std::chrono::milliseconds timeout = 5s)
{
  REQUIRE(future.wait_for(timeout) == std::future_status::ready);
  try {
    (void)future.get();
    FAIL("expected S3 RDMA read to fail");
  } catch (std::exception const& error) {
    return error.what();
  }
  return {};
}

std::future<std::size_t> issue_device_read(sirius::io::sirius_datasource& datasource,
                                           rmm::device_buffer& destination,
                                           rmm::cuda_stream_view stream)
{
  return datasource.device_read_async(
    0, destination.size(), static_cast<std::uint8_t*>(destination.data()), stream);
}

bool wait_until(auto&& predicate, std::chrono::milliseconds timeout = 5s)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) { return true; }
    std::this_thread::sleep_for(2ms);
  }
  return predicate();
}

bool accepts_test_tag(std::string_view tag) noexcept { return tag == "accepted-test-tag"; }

void require_no_long_fragment(std::string_view text, std::string_view secret)
{
  REQUIRE(secret.size() > 8);
  for (std::size_t begin = 0; begin + 9 <= secret.size(); ++begin) {
    CHECK(text.find(secret.substr(begin, 9)) == std::string_view::npos);
  }
}

#ifdef SIRIUS_HAVE_TESTCONTAINERS

constexpr std::string_view k_fixture_key = "small.bin";

std::string env_or(std::string_view name, std::string fallback = {})
{
  if (auto* value = std::getenv(std::string{name}.c_str()); value != nullptr) { return value; }
  return fallback;
}

std::string require_env(std::string_view name)
{
  auto value = env_or(name);
  REQUIRE_FALSE(value.empty());
  return value;
}

bool ensure_minio_env()
{
  if (sirius::test::ensure_s3_container_env()) { return true; }
  SUCCEED("SIRIUS_TEST_S3_* not set; skipping S3 RDMA client-seam MinIO test");
  return false;
}

struct minio_env {
  std::string endpoint;
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string bucket;
  std::filesystem::path local_dir;
};

minio_env read_minio_env()
{
  return minio_env{require_env("SIRIUS_TEST_S3_ENDPOINT"),
                   env_or("SIRIUS_TEST_S3_REGION", "us-east-1"),
                   require_env("SIRIUS_TEST_S3_ACCESS_KEY"),
                   require_env("SIRIUS_TEST_S3_SECRET_KEY"),
                   require_env("SIRIUS_TEST_S3_BUCKET"),
                   std::filesystem::path{require_env("SIRIUS_TEST_S3_LOCAL_DIR")}};
}

std::vector<std::uint8_t> read_binary_file(std::filesystem::path const& path)
{
  std::ifstream input(path, std::ios::binary);
  REQUIRE(input.good());
  std::vector<char> chars((std::istreambuf_iterator<char>(input)),
                          std::istreambuf_iterator<char>());
  return std::vector<std::uint8_t>(chars.begin(), chars.end());
}

std::shared_ptr<sirius::io::s3::sirius_sigv4_header_authorizer> make_header_authorizer(
  minio_env const& env, std::string endpoint = {})
{
  sirius::io::s3::static_credentials credentials;
  credentials.access_key_id     = env.access_key;
  credentials.secret_access_key = env.secret_key;
  return std::make_shared<sirius::io::s3::sirius_sigv4_header_authorizer>(
    std::move(credentials), env.region, endpoint.empty() ? env.endpoint : std::move(endpoint));
}

std::unique_ptr<curl_s3_control_client> make_control_client(minio_env const& env,
                                                            std::string endpoint = {})
{
  return std::make_unique<curl_s3_control_client>(
    make_header_authorizer(env, std::move(endpoint)), std::string{}, false);
}

std::string header_value(std::vector<std::pair<std::string, std::string>> const& headers,
                         std::string_view name)
{
  for (auto const& [key, value] : headers) {
    if (key.size() == name.size() &&
        std::equal(key.begin(), key.end(), name.begin(), [](unsigned char lhs, unsigned char rhs) {
          return std::tolower(lhs) == std::tolower(rhs);
        })) {
      return value;
    }
  }
  return {};
}

std::size_t curl_write(char* data, std::size_t size, std::size_t count, void* opaque)
{
  auto* bytes  = static_cast<std::vector<std::uint8_t>*>(opaque);
  auto const n = size * count;
  bytes->insert(
    bytes->end(), reinterpret_cast<std::uint8_t*>(data), reinterpret_cast<std::uint8_t*>(data) + n);
  return n;
}

struct wire_response {
  long status{0};
  std::vector<std::uint8_t> body;
};

wire_response perform_get(sirius::io::s3::s3_authorized_request const& request)
{
  CURL* curl = curl_easy_init();
  REQUIRE(curl != nullptr);
  curl_slist* headers = nullptr;
  for (auto const& [name, value] : request.headers) {
    auto line = name + ": " + value;
    headers   = curl_slist_append(headers, line.c_str());
    REQUIRE(headers != nullptr);
  }

  wire_response response;
  curl_easy_setopt(curl, CURLOPT_URL, request.url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &curl_write);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response.body);
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
  auto const rc = curl_easy_perform(curl);
  if (rc == CURLE_OK) { (void)curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response.status); }
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  REQUIRE(rc == CURLE_OK);
  return response;
}

#endif

}  // namespace s3_rdma_client_seam_tests

#ifdef SIRIUS_HAVE_TESTCONTAINERS

TEST_CASE("s3_rdma AC1 control HEAD reports success and missing keys without throwing",
          "[s3][rdma][client-seam][integration]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!ensure_minio_env()) { return; }

  auto const env     = read_minio_env();
  auto const payload = read_binary_file(env.local_dir / std::string{k_fixture_key});
  auto client        = make_control_client(env);

  auto const found = client->head(rx_route{env.bucket, std::string{k_fixture_key}});
  CHECK(found.outcome.http_status == 200);
  CHECK(found.outcome.transport_error.empty());
  CHECK(found.object_size == payload.size());

  auto const missing = client->head(rx_route{env.bucket, "step5-ac1-missing.bin"});
  CHECK(missing.outcome.http_status == 404);
  CHECK(missing.outcome.transport_error.empty());
}

TEST_CASE("s3_rdma AC1 control range GET reports partial and past-EOF results",
          "[s3][rdma][client-seam][integration]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!ensure_minio_env()) { return; }

  auto const env     = read_minio_env();
  auto const payload = read_binary_file(env.local_dir / std::string{k_fixture_key});
  REQUIRE(payload.size() > 64);
  auto client = make_control_client(env);
  rx_route route{env.bucket, std::string{k_fixture_key}};

  std::array<std::uint8_t, 32> bytes{};
  auto const partial = client->range_get(route, 17, bytes.size(), bytes.data());
  CHECK(partial.outcome.http_status == 206);
  CHECK(partial.outcome.transport_error.empty());
  CHECK(partial.delivered_bytes == bytes.size());
  CHECK(partial.content_range == "bytes 17-48/" + std::to_string(payload.size()));
  CHECK(std::equal(bytes.begin(), bytes.end(), payload.begin() + 17));

  auto const past_eof = client->range_get(route, payload.size(), bytes.size(), bytes.data());
  CHECK(past_eof.outcome.http_status == 416);
  CHECK(past_eof.outcome.transport_error.empty());
  CHECK(past_eof.delivered_bytes == 0);
}

TEST_CASE("s3_rdma AC1 control transport failure is a result rather than an exception",
          "[s3][rdma][client-seam][integration]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!ensure_minio_env()) { return; }

  auto client       = make_control_client(read_minio_env(), "http://127.0.0.1:1");
  auto const result = client->head(rx_route{"bucket", "unreachable"});
  CHECK(result.outcome.http_status == 0);
  CHECK_FALSE(result.outcome.transport_error.empty());
}

TEST_CASE("s3_rdma AC2 control calls make exactly one HTTP attempt",
          "[s3][rdma][client-seam][integration]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!ensure_minio_env()) { return; }
  auto const env = read_minio_env();

  SECTION("successful HEAD")
  {
    auto client = make_control_client(env);
    (void)client->head(rx_route{env.bucket, std::string{k_fixture_key}});
    CHECK(client->attempts_total() == 1);
  }
  SECTION("successful range GET")
  {
    auto client = make_control_client(env);
    std::array<std::uint8_t, 16> bytes{};
    (void)client->range_get(
      rx_route{env.bucket, std::string{k_fixture_key}}, 0, bytes.size(), bytes.data());
    CHECK(client->attempts_total() == 1);
  }
  SECTION("HTTP failure")
  {
    auto client = make_control_client(env);
    (void)client->head(rx_route{env.bucket, "step5-ac2-missing.bin"});
    CHECK(client->attempts_total() == 1);
  }
  SECTION("transport failure")
  {
    auto client = make_control_client(env, "http://127.0.0.1:1");
    std::array<std::uint8_t, 16> bytes{};
    (void)client->range_get(rx_route{"bucket", "unreachable"}, 0, bytes.size(), bytes.data());
    CHECK(client->attempts_total() == 1);
  }
}

TEST_CASE("s3_rdma AC3 one control client reuses its persistent connection",
          "[s3][rdma][client-seam][integration]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!ensure_minio_env()) { return; }

  auto const env = read_minio_env();
  auto client    = make_control_client(env);
  rx_route route{env.bucket, std::string{k_fixture_key}};
  std::array<std::uint8_t, 16> first{};
  std::array<std::uint8_t, 16> second{};
  auto const attempts_before    = client->attempts_total();
  auto const connections_before = client->connections_total();

  CHECK(client->range_get(route, 0, first.size(), first.data()).outcome.http_status == 206);
  CHECK(client->range_get(route, first.size(), second.size(), second.data()).outcome.http_status ==
        206);
  CHECK(client->attempts_total() - attempts_before == 2);
  CHECK(client->connections_total() - connections_before <= 1);
}

TEST_CASE("s3_rdma AC7 data headers are signed and accepted on the wire",
          "[s3][rdma][client-seam][integration]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!ensure_minio_env()) { return; }

  auto const env  = read_minio_env();
  auto authorizer = make_header_authorizer(env);
  std::vector<std::pair<std::string, std::string>> extra_headers{
    {"x-amz-rdma-token", "step5-wire-token"}, {"Range", "bytes=0-15"}};
  auto request = authorizer->authorize_with_headers({env.bucket, std::string{k_fixture_key}},
                                                    sirius::io::s3::s3_request_method::GET,
                                                    30s,
                                                    extra_headers);

  auto const authorization = header_value(request.headers, "Authorization");
  CHECK(authorization.find("range") != std::string::npos);
  CHECK(authorization.find("x-amz-rdma-token") != std::string::npos);
  CHECK(header_value(request.headers, "Range") == "bytes=0-15");
  CHECK(header_value(request.headers, "x-amz-rdma-token") == "step5-wire-token");

  auto const response = perform_get(request);
  CHECK(response.status == 206);
  CHECK(response.body.size() == 16);
}

#endif

TEST_CASE("s3_rdma AC4 not-sent data errors do not poison the transport",
          "[s3][rdma][client-seam][gpu]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(k_slot_size);
  auto transport = seeded_mock_transport(std::string{k_bucket}, "not-sent", payload);
  transport->data->script_result(
    data_get_result{data_commit_state::not_sent, 0, 0, {}, "request not sent"});
  auto ctx = make_started_ioctx(transport);
  auto ds  = open_ds(ctx, "not-sent");
  rmm::cuda_stream stream;
  rmm::device_buffer first(payload.size(), stream);

  auto failed = issue_device_read(*ds, first, stream);
  CHECK(ready_error(failed).find("not sent") != std::string::npos);
  CHECK(ctx->perf_snapshot().fail_stop_total == 0);
  CHECK(ctx->perf_snapshot().arena_leak_total == 0);

  rmm::device_buffer follow_up(payload.size(), stream);
  auto succeeded = issue_device_read(*ds, follow_up, stream);
  REQUIRE(succeeded.wait_for(5s) == std::future_status::ready);
  CHECK(succeeded.get() == payload.size());
}

TEST_CASE("s3_rdma AC4 sent-unknown data errors fail-stop", "[s3][rdma][client-seam][gpu]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(k_slot_size);
  auto transport = seeded_mock_transport(std::string{k_bucket}, "sent-unknown", payload);
  transport->data->script_result(
    data_get_result{data_commit_state::sent_unknown, 0, 0, {}, "completion unknown"});
  auto ctx = make_started_ioctx(transport);
  auto ds  = open_ds(ctx, "sent-unknown");
  rmm::cuda_stream stream;
  rmm::device_buffer destination(payload.size(), stream);

  auto failed = issue_device_read(*ds, destination, stream);
  CHECK(ready_error(failed).find("completion unknown") != std::string::npos);
  CHECK(ctx->perf_snapshot().fail_stop_total == 1);
  CHECK(ctx->perf_snapshot().arena_leak_total == 1);
}

TEST_CASE("s3_rdma AC4 completed data succeeds only with all authority legs",
          "[s3][rdma][client-seam][gpu]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!cuda_device_available()) { return; }

  auto payload   = pattern_bytes(k_slot_size);
  auto transport = seeded_mock_transport(std::string{k_bucket}, "completed", payload);
  transport->data->script_result(
    data_get_result{data_commit_state::completed, payload.size(), 200, "accepted-test-tag", {}});
  auto ctx = make_started_ioctx(transport, mock_config(), &accepts_test_tag);
  auto ds  = open_ds(ctx, "completed");
  rmm::cuda_stream stream;
  rmm::device_buffer destination(payload.size(), stream);

  auto succeeded = issue_device_read(*ds, destination, stream);
  REQUIRE(succeeded.wait_for(5s) == std::future_status::ready);
  CHECK(succeeded.get() == payload.size());
  CHECK(ctx->perf_snapshot().fail_stop_total == 0);
}

TEST_CASE("s3_rdma AC5 completion authority validates tag bytes and status",
          "[s3][rdma][client-seam][gpu]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!cuda_device_available()) { return; }

  struct row {
    std::string_view name;
    data_get_result result;
    bool succeeds;
  };
  auto const expected = k_slot_size;
  std::vector<row> rows{
    {"missing tag", {data_commit_state::completed, expected, 200, {}, {}}, false},
    {"predicate rejected",
     {data_commit_state::completed, expected, 200, "rejected-test-tag", {}},
     false},
    {"short bytes",
     {data_commit_state::completed, expected - 1, 200, "accepted-test-tag", {}},
     false},
    {"over bytes",
     {data_commit_state::completed, expected + 1, 200, "accepted-test-tag", {}},
     false},
    {"negative status",
     {data_commit_state::completed, expected, 503, "accepted-test-tag", {}},
     false},
    {"all positive", {data_commit_state::completed, expected, 200, "accepted-test-tag", {}}, true},
  };

  for (auto const& test : rows) {
    DYNAMIC_SECTION(test.name)
    {
      auto payload   = pattern_bytes(expected);
      auto transport = seeded_mock_transport(std::string{k_bucket}, "authority", payload);
      transport->data->script_result(test.result);
      auto ctx = make_started_ioctx(transport, mock_config(), &accepts_test_tag);
      auto ds  = open_ds(ctx, "authority");
      rmm::cuda_stream stream;
      rmm::device_buffer destination(expected, stream);
      auto future = issue_device_read(*ds, destination, stream);

      if (test.succeeds) {
        REQUIRE(future.wait_for(5s) == std::future_status::ready);
        CHECK(future.get() == expected);
        CHECK(ctx->perf_snapshot().fail_stop_total == 0);
      } else {
        CHECK_FALSE(ready_error(future).empty());
        auto const gets_before_follow_up = transport->data->gets_issued();
        auto follow_up                   = issue_device_read(*ds, destination, stream);
        CHECK_FALSE(ready_error(follow_up).empty());
        auto const snapshot = ctx->perf_snapshot();
        CHECK(snapshot.fail_stop_total == 1);
        CHECK(snapshot.retries_total == 0);
        CHECK(snapshot.arena_leak_total == 1);
        CHECK(transport->data->gets_issued() == gets_before_follow_up);
      }
    }
  }
}

TEST_CASE("s3_rdma AC6 host reads use only the control plane", "[s3][rdma][client-seam]")
{
  using namespace s3_rdma_client_seam_tests;

  auto payload   = pattern_bytes(256);
  auto transport = seeded_mock_transport(std::string{k_bucket}, "host-plane", payload);
  auto ctx       = make_started_ioctx(transport);
  auto ds        = open_ds(ctx, "host-plane");
  std::array<std::uint8_t, 32> destination{};

  CHECK(ds->host_read(11, destination.size(), destination.data()) == destination.size());
  CHECK(transport->control->range_gets_issued() == 1);
  CHECK(transport->data->gets_issued() == 0);
  CHECK(std::equal(destination.begin(), destination.end(), payload.begin() + 11));
}

TEST_CASE("s3_rdma AC6 control failure is nonfatal and the next host read succeeds",
          "[s3][rdma][client-seam]")
{
  using namespace s3_rdma_client_seam_tests;

  auto payload   = pattern_bytes(256);
  auto transport = seeded_mock_transport(std::string{k_bucket}, "host-recovery", payload);
  auto ctx       = make_started_ioctx(transport);
  auto ds        = open_ds(ctx, "host-recovery");
  std::array<std::uint8_t, 32> destination{};
  transport->control->respond_status(503);

  CHECK_THROWS(ds->host_read(0, destination.size(), destination.data()));
  CHECK(ctx->perf_snapshot().fail_stop_total == 0);
  CHECK(ds->host_read(0, destination.size(), destination.data()) == destination.size());
  CHECK(ctx->perf_snapshot().fail_stop_total == 0);
  CHECK(transport->data->gets_issued() == 0);
}

TEST_CASE("s3_rdma AC10 token-bearing diagnostics are redacted at publication",
          "[s3][rdma][client-seam][gpu]")
{
  using namespace s3_rdma_client_seam_tests;
  if (!cuda_device_available()) { return; }

  constexpr std::string_view sentinel = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef";
  auto payload                        = pattern_bytes(k_slot_size);
  auto transport = seeded_mock_transport(std::string{k_bucket}, "redaction", payload);
  transport->data->script_result(data_get_result{
    data_commit_state::sent_unknown,
    0,
    0,
    {},
    "gateway error; x-amz-rdma-token: " + std::string{sentinel} + "; completion unknown"});
  sirius::test::scoped_recording_log_sink logs{"trace"};
  auto ctx = make_started_ioctx(transport);
  auto ds  = open_ds(ctx, "redaction");
  rmm::cuda_stream stream;
  rmm::device_buffer destination(payload.size(), stream);

  auto future      = issue_device_read(*ds, destination, stream);
  auto const error = ready_error(future);
  CHECK(error.find("x-amz-rdma-token") != std::string::npos);
  require_no_long_fragment(error, sentinel);

  auto const records = logs.records();
  REQUIRE(std::any_of(records.begin(), records.end(), [](auto const& record) {
    return record.message.find("x-amz-rdma-token") != std::string::npos;
  }));
  for (auto const& record : records) {
    require_no_long_fragment(record.message, sentinel);
  }
}

TEST_CASE("s3_rdma AC13 start acquires exactly one data session per worker",
          "[s3][rdma][client-seam]")
{
  using namespace s3_rdma_client_seam_tests;
  constexpr std::size_t workers = 4;
  auto transport                = std::make_shared<mock_transport_fixture>();
  auto ctx                      = make_started_ioctx(transport, mock_config(workers));

  REQUIRE(wait_until([&] { return transport->data->acquired_total() == workers; }));
  auto const threads = transport->data->acquisition_thread_ids();
  CHECK(transport->data->acquired_total() == workers);
  REQUIRE(threads.size() == workers);
  std::set<std::thread::id> unique_threads(threads.begin(), threads.end());
  CHECK(unique_threads.size() == workers);
  ctx->shutdown();
}
