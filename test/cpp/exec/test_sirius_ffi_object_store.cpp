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

// Only the public FFI surface is exercised. No query, S3 service or private engine header.
// Construction-only tests use the same small memory configuration as the existing FFI guards.

#include "sirius/ffi.hpp"
#include "utils/ffi_object_store_fixture.hpp"

#include <arpa/inet.h>
#include <catch.hpp>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <latch>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

using sirius::ffi::make_context_from_config;
using sirius::ffi::ObjectStoreConfig;
using sirius::test::object_store_inputs;
using sirius::test::object_store_sentinels;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

static_assert(std::is_aggregate_v<ObjectStoreConfig>);
static_assert(std::is_const_v<decltype(ObjectStoreConfig::endpoint)> &&
              std::is_const_v<decltype(ObjectStoreConfig::region)> &&
              std::is_const_v<decltype(ObjectStoreConfig::access_key)> &&
              std::is_const_v<decltype(ObjectStoreConfig::secret_key)> &&
              std::is_const_v<decltype(ObjectStoreConfig::session_token)> &&
              std::is_const_v<decltype(ObjectStoreConfig::transport)> &&
              std::is_const_v<decltype(ObjectStoreConfig::signing_mode)> &&
              std::is_const_v<decltype(ObjectStoreConfig::ca_bundle_path)>);

fs::path fixture_path(std::string const& name) { return fs::path(__FILE__).parent_path() / name; }

std::string memory_config()
{
  return (fs::path(__FILE__).parent_path().parent_path() / "scan" / "memory.yaml").string();
}

void check_no_sentinels(std::string const& message)
{
  for (auto sentinel : object_store_sentinels) {
    CHECK(message.find(sentinel) == std::string::npos);
  }
}

template <typename Operation>
void check_invalid(Operation&& operation, std::string const& field)
{
  bool invalid_argument = false;
  std::string message;
  try {
    operation();
  } catch (std::invalid_argument const& error) {
    invalid_argument = true;
    message          = error.what();
  } catch (std::exception const& error) {
    message = error.what();
  }
  INFO("exception text: " << message);
  CHECK(invalid_argument);
  if (invalid_argument) { CHECK(message.starts_with("ObjectStoreConfig." + field + ":")); }
  check_no_sentinels(message);
}

void check_fields(ObjectStoreConfig const& actual, object_store_inputs const& expected)
{
  CHECK(actual.endpoint == expected.endpoint);
  CHECK(actual.region == expected.region);
  CHECK(actual.access_key == expected.access_key);
  CHECK(actual.secret_key == expected.secret_key);
  CHECK(actual.session_token == expected.session_token);
  CHECK(actual.transport == expected.transport);
  CHECK(actual.signing_mode == expected.signing_mode);
  CHECK(actual.ca_bundle_path == expected.ca_bundle_path);
}

struct invalid_input {
  char const* name;
  std::string object_store_inputs::* member;
  std::string value;
  char const* field;
};

std::vector<invalid_input> invalid_inputs()
{
  auto const missing_ca =
    fixture_path("missing_" + std::string(object_store_sentinels[2]) + ".pem");
  REQUIRE_FALSE(fs::exists(missing_ca));
  return {
    {"empty endpoint", &object_store_inputs::endpoint, "", "endpoint"},
    {"empty region", &object_store_inputs::region, "", "region"},
    {"empty access key", &object_store_inputs::access_key, "", "access_key"},
    {"empty secret key", &object_store_inputs::secret_key, "", "secret_key"},
    {"missing scheme", &object_store_inputs::endpoint, "localhost:9000", "endpoint"},
    {"endpoint path",
     &object_store_inputs::endpoint,
     "https://localhost/" + std::string(object_store_sentinels[1]),
     "endpoint"},
    {"endpoint query",
     &object_store_inputs::endpoint,
     "https://localhost?credential=" + std::string(object_store_sentinels[0]),
     "endpoint"},
    {"endpoint fragment",
     &object_store_inputs::endpoint,
     "https://localhost#" + std::string(object_store_sentinels[2]),
     "endpoint"},
    {"unsupported scheme", &object_store_inputs::endpoint, "ftp://localhost", "endpoint"},
    {"missing host", &object_store_inputs::endpoint, "https://", "endpoint"},
    {"port without host", &object_store_inputs::endpoint, "https://:443", "endpoint"},
    {"empty transport", &object_store_inputs::transport, "", "transport"},
    {"internal transport alias", &object_store_inputs::transport, "https", "transport"},
    {"unknown transport",
     &object_store_inputs::transport,
     std::string(object_store_sentinels[0]),
     "transport"},
    {"empty signing mode", &object_store_inputs::signing_mode, "", "signing_mode"},
    {"unknown signing mode",
     &object_store_inputs::signing_mode,
     std::string(object_store_sentinels[1]),
     "signing_mode"},
    {"missing CA file",
     &object_store_inputs::ca_bundle_path,
     missing_ca.string(),
     "ca_bundle_path"},
  };
}

class socket_handle {
 public:
  explicit socket_handle(int fd) : fd_(fd)
  {
    if (fd_ < 0) { throw std::system_error(errno, std::generic_category(), "socket"); }
  }
  ~socket_handle() { ::close(fd_); }
  socket_handle(socket_handle const&)            = delete;
  socket_handle& operator=(socket_handle const&) = delete;
  [[nodiscard]] int get() const { return fd_; }

 private:
  int fd_;
};

// A live listener detects connections even when no HTTP request is completed. Its positive
// control proves observation works; final draining avoids mistaking an unaccepted backlog for
// "no network". Connections are closed immediately, so an accidental request fails promptly.
class connection_observer {
 public:
  connection_observer()
  {
    address_.sin_family      = AF_INET;
    address_.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::bind(listener_.get(), reinterpret_cast<sockaddr*>(&address_), sizeof(address_)) != 0 ||
        ::listen(listener_.get(), 16) != 0) {
      throw std::system_error(errno, std::generic_category(), "listen");
    }
    socklen_t length = sizeof(address_);
    if (::getsockname(listener_.get(), reinterpret_cast<sockaddr*>(&address_), &length) != 0) {
      throw std::system_error(errno, std::generic_category(), "getsockname");
    }
    worker_ = std::jthread([this](std::stop_token stop) {
      while (!stop.stop_requested()) {
        pollfd event{listener_.get(), POLLIN, 0};
        auto const ready = ::poll(&event, 1, 20);
        if (ready > 0) {
          drain();
        } else if (ready < 0 && errno != EINTR) {
          record_error(errno);
          break;
        }
      }
      drain();
    });
  }

  ~connection_observer() { finish(); }

  [[nodiscard]] std::string endpoint() const
  {
    return "http://127.0.0.1:" + std::to_string(ntohs(address_.sin_port));
  }

  void positive_control()
  {
    socket_handle probe(::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
    if (::connect(probe.get(), reinterpret_cast<sockaddr*>(&address_), sizeof(address_)) != 0) {
      throw std::system_error(errno, std::generic_category(), "positive control connect");
    }
    std::unique_lock lock(mutex_);
    if (!observed_.wait_for(lock, 2s, [this] { return accepted_ > 0 || error_ != 0; }) ||
        error_ != 0) {
      throw std::runtime_error("connection observer positive control failed");
    }
  }

  void finish()
  {
    if (worker_.joinable()) {
      worker_.request_stop();
      worker_.join();
    }
  }

  // Called after finish(), so the worker has published its final observations.
  [[nodiscard]] std::size_t accepted() const { return accepted_; }
  [[nodiscard]] int error() const { return error_; }

 private:
  void record_error(int error)
  {
    std::lock_guard lock(mutex_);
    error_ = error;
    observed_.notify_all();
  }

  void drain()
  {
    for (;;) {
      auto const accepted =
        ::accept4(listener_.get(), nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK);
      if (accepted < 0) {
        if (errno == EINTR) { continue; }
        if (errno != EAGAIN && errno != EWOULDBLOCK) { record_error(errno); }
        return;
      }
      ::close(accepted);
      std::lock_guard lock(mutex_);
      ++accepted_;
      observed_.notify_all();
    }
  }

  socket_handle listener_{::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0)};
  sockaddr_in address_{};
  std::mutex mutex_;
  std::condition_variable observed_;
  std::size_t accepted_{};
  int error_{};
  std::jthread worker_;
};

}  // namespace

TEST_CASE("FFI object store factory rejects invalid configuration without leaking credentials",
          "[isolated_context][ffi][object_store]")
{
  for (auto const& item : invalid_inputs()) {
    CAPTURE(item.name);
    object_store_inputs inputs;
    inputs.*(item.member) = item.value;
    // No other Context is alive, even if a broken validator unexpectedly constructs this one.
    check_invalid([&] { (void)make_context_from_config(memory_config(), inputs.snapshot()); },
                  item.field);
  }
}

TEST_CASE("FFI object store setter rejects invalid configuration and preserves its snapshot",
          "[isolated_context][ffi][object_store]")
{
  auto const original = object_store_inputs{}.snapshot();
  auto context        = make_context_from_config(memory_config(), original);
  for (auto const& item : invalid_inputs()) {
    CAPTURE(item.name);
    object_store_inputs inputs;
    inputs.*(item.member) = item.value;
    check_invalid([&] { context->set_object_store(inputs.snapshot()); }, item.field);
    CHECK(context->object_store_config().get() == original.get());
    // Keep later matrix entries independent after an unexpected successful publication.
    context->set_object_store(original);
  }
}

TEST_CASE("FFI object store backend policy rejects RDMA and kvikIO options",
          "[isolated_context][ffi][object_store]")
{
  auto const ca = fixture_path("ffi_object_store_test_ca.pem");
  REQUIRE(fs::is_regular_file(ca));
  for (auto const& item : std::vector<invalid_input>{
         {"RDMA", &object_store_inputs::transport, "rdma", "transport"},
         {"kvikIO header", &object_store_inputs::signing_mode, "header", "signing_mode"},
         {"kvikIO CA", &object_store_inputs::ca_bundle_path, ca.string(), "ca_bundle_path"}}) {
    CAPTURE(item.name);
    auto const config = item.member == &object_store_inputs::transport
                          ? memory_config()
                          : fixture_path("ffi_object_store_kvikio.yaml").string();
    object_store_inputs inputs;
    auto const original   = inputs.snapshot();
    inputs.*(item.member) = item.value;
    auto const rejected   = inputs.snapshot();
    check_invalid([&] { (void)make_context_from_config(config, rejected); }, item.field);
    auto context = make_context_from_config(config, original);
    check_invalid([&] { context->set_object_store(rejected); }, item.field);
    CHECK(context->object_store_config().get() == original.get());
  }
}

TEST_CASE("FFI object store factories preserve pointers without network access",
          "[isolated_context][ffi][object_store]")
{
  connection_observer observer;
  observer.positive_control();
  object_store_inputs inputs;
  inputs.endpoint = observer.endpoint();
  inputs.endpoint.replace(0, 4, "https");
  inputs.transport      = "http";
  inputs.signing_mode   = "header";
  inputs.ca_bundle_path = fixture_path("ffi_object_store_test_ca.pem").string();
  auto const config     = inputs.snapshot();
  {
    auto context = make_context_from_config("", nullptr);
    REQUIRE(context);
    CHECK(context->object_store_config() == nullptr);
    context->set_object_store(config);
    CHECK(context->object_store_config().get() == config.get());
  }
  {
    auto context = make_context_from_config("", config);
    REQUIRE(context);
    CHECK(context->object_store_config().get() == config.get());
    context->set_object_store(nullptr);
  }
  observer.finish();
  CHECK(observer.error() == 0);
  CHECK(observer.accepted() == 1);  // Only the positive control.
}

TEST_CASE("FFI object store factory refusal leaves the next context usable",
          "[isolated_context][ffi][object_store]")
{
  object_store_inputs inputs;
  inputs.region.clear();
  check_invalid([&] { (void)make_context_from_config(memory_config(), inputs.snapshot()); },
                "region");
  {
    auto const valid = object_store_inputs{}.snapshot();
    auto context     = make_context_from_config(memory_config(), valid);
    REQUIRE(context);
    CHECK(context->object_store_config().get() == valid.get());
  }

  auto const missing_config = fixture_path("ffi_object_store_missing_config.yaml");
  REQUIRE_FALSE(fs::exists(missing_config));
  std::string message;
  try {
    (void)make_context_from_config(missing_config.string(), inputs.snapshot());
  } catch (std::exception const& error) {
    message = error.what();
  }
  // The documented order parses the engine YAML before validating the object-store value.
  CHECK(message.find(missing_config.filename().string()) != std::string::npos);
  CHECK(message.find("ObjectStoreConfig.") == std::string::npos);
  check_no_sentinels(message);
  // This case observes early validation/recovery, not a post-initialization MR rollback.
}

TEST_CASE("FFI object store publishes complete snapshots atomically",
          "[isolated_context][ffi][object_store]")
{
  object_store_inputs inputs;
  auto const first = inputs.snapshot();
  inputs.region    = "replacement-region";
  inputs.endpoint  = "http://127.0.0.1:9";
  inputs.access_key += "_replacement";
  inputs.secret_key += "_replacement";
  inputs.session_token += "_replacement";
  auto const second = inputs.snapshot();
  auto context      = make_context_from_config(memory_config(), first);
  std::latch started{1};
  std::atomic<std::size_t> invalid{}, seen_first{}, seen_second{};
  std::jthread reader([&](std::stop_token stop) {
    auto observe = [&] {
      auto snapshot = context->object_store_config();
      if (snapshot.get() == first.get()) {
        ++seen_first;
      } else if (snapshot.get() == second.get()) {
        ++seen_second;
      } else {
        ++invalid;
      }
    };
    observe();
    started.count_down();
    while (!stop.stop_requested()) {
      observe();
      std::this_thread::yield();
    }
  });
  started.wait();
  for (std::size_t i = 0; i < 256; ++i) {
    context->set_object_store(i % 2 == 0 ? second : first);
    std::this_thread::yield();
  }
  context->set_object_store(second);
  auto const deadline = std::chrono::steady_clock::now() + 2s;
  while (seen_second.load() == 0 && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  reader.request_stop();
  reader.join();
  CHECK(seen_first.load() > 0);
  CHECK(seen_second.load() > 0);
  CHECK(invalid.load() == 0);
  CHECK(context->object_store_config().get() == second.get());
  check_fields(*first, object_store_inputs{});
  check_fields(*second, inputs);
}

TEST_CASE("FFI object store credentials derive an immutable snapshot",
          "[isolated_context][ffi][object_store]")
{
  ObjectStoreConfig defaults{};
  CHECK(defaults.transport == "auto");
  CHECK(defaults.signing_mode == "presigned");
  CHECK(defaults.session_token.empty());
  CHECK(defaults.ca_bundle_path.empty());

  connection_observer observer;
  observer.positive_control();
  object_store_inputs inputs;
  inputs.endpoint       = observer.endpoint();
  inputs.transport      = "http";
  inputs.signing_mode   = "header";
  inputs.ca_bundle_path = fixture_path("ffi_object_store_test_ca.pem").string();
  auto const original   = inputs.snapshot();
  auto const unchanged  = inputs;
  inputs.access_key += "_new";
  inputs.secret_key += "_new";
  inputs.session_token += "_new";
  auto const derived =
    original->with_credentials(inputs.access_key, inputs.secret_key, inputs.session_token);
  REQUIRE(derived);
  CHECK(derived.get() != original.get());
  check_fields(*derived, inputs);
  check_fields(*original, unchanged);

  // Derivation does not validate: even empty credentials produce a new immutable value.
  auto const empty = original->with_credentials("", "", "");
  REQUIRE(empty);
  auto expected_empty          = unchanged;
  expected_empty.access_key    = "";
  expected_empty.secret_key    = "";
  expected_empty.session_token = "";
  check_fields(*empty, expected_empty);
  observer.finish();
  CHECK(observer.error() == 0);
  CHECK(observer.accepted() == 1);
}

TEST_CASE("FFI object store YAML remains rejected", "[isolated_context][ffi][object_store]")
{
  std::string message;
  try {
    (void)make_context_from_config(fixture_path("ffi_object_store_yaml_rejected.yaml").string(),
                                   nullptr);
  } catch (std::exception const& error) {
    message = error.what();
  }
  CHECK(message.find("sirius.executor.scan_manager.object_store") != std::string::npos);
  CHECK(message.find("no longer YAML-loadable") != std::string::npos);
  check_no_sentinels(message);
}

TEST_CASE("FFI object store clears without invalidating retained snapshots",
          "[isolated_context][ffi][object_store]")
{
  object_store_inputs const inputs;
  auto const original = inputs.snapshot();
  std::shared_ptr<ObjectStoreConfig> retained;
  {
    auto context = make_context_from_config(memory_config(), original);
    retained     = context->object_store_config();

    auto const equal_values = inputs.snapshot();
    REQUIRE(equal_values.get() != original.get());
    context->set_object_store(equal_values);
    CHECK(context->object_store_config().get() == equal_values.get());
    context->set_object_store(nullptr);
    CHECK(context->object_store_config() == nullptr);
    REQUIRE(retained);
    check_fields(*retained, inputs);
    context->set_object_store(retained);
    CHECK(context->object_store_config().get() == retained.get());
  }
  CHECK(retained.get() == original.get());
  check_fields(*retained, inputs);
  auto const derived = retained->with_credentials(inputs.access_key, inputs.secret_key, "");
  REQUIRE(derived);
  CHECK(derived->session_token.empty());
  CHECK(retained->session_token == inputs.session_token);
}
