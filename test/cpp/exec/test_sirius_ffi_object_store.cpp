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

// Public snapshot validation and refusal of remote plans before binding. No S3 service is
// required; isolated contexts use the same small memory configuration as the existing FFI guards.

#include "sirius/ffi.hpp"
#include "sirius_ffi_testing.hpp"
#include "utils/ffi_object_store_fixture.hpp"

#include <arpa/inet.h>
#include <catch.hpp>
#include <duckdb/common/arrow/arrow.hpp>
#include <poll.h>
#include <substrait/plan.pb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <latch>
#include <limits>
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
              std::is_const_v<decltype(ObjectStoreConfig::ca_bundle_path)> &&
              std::is_const_v<decltype(ObjectStoreConfig::uri_prefix)>);

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
  CHECK(actual.uri_prefix == expected.uri_prefix);
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
    {"non-S3 prefix", &object_store_inputs::uri_prefix, "http://b/", "uri_prefix"},
    {"empty prefix bucket", &object_store_inputs::uri_prefix, "s3://", "uri_prefix"},
    {"missing prefix bucket", &object_store_inputs::uri_prefix, "s3:///p/", "uri_prefix"},
    {"bucket without separator", &object_store_inputs::uri_prefix, "s3://b", "uri_prefix"},
    {"prefix without separator", &object_store_inputs::uri_prefix, "s3://b/p", "uri_prefix"},
    {"prefix query", &object_store_inputs::uri_prefix, "s3://b/p/?x", "uri_prefix"},
    {"prefix fragment", &object_store_inputs::uri_prefix, "s3://b/p/#f", "uri_prefix"},
    {"prefix query with credential",
     &object_store_inputs::uri_prefix,
     "s3://b/p/?token=" + std::string(object_store_sentinels[2]),
     "uri_prefix"},
    {"prefix fragment with credential",
     &object_store_inputs::uri_prefix,
     "s3://b/p/#" + std::string(object_store_sentinels[1]),
     "uri_prefix"},
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

struct scan_item {
  std::string path;
  std::uint64_t start{};
  std::uint64_t length{};
  bool uri_path{};
};

// The same LocalFiles construction as the byte-range tests, with both supported path forms.
std::string read_plan(std::vector<scan_item> const& items)
{
  substrait::Plan plan;
  auto* root = plan.add_relations()->mutable_root();
  root->add_names("a");
  auto* files = root->mutable_input()->mutable_read()->mutable_local_files();
  for (auto const& item : items) {
    auto* file = files->add_items();
    if (item.uri_path) {
      file->set_uri_path(item.path);
    } else {
      file->set_uri_file(item.path);
    }
    file->mutable_parquet();
    file->set_start(item.start);
    file->set_length(item.length);
  }
  std::string bytes;
  REQUIRE(plan.SerializeToString(&bytes));
  return bytes;
}

enum class entry_point { execute, fragment_build };

void invoke_plan(sirius::ffi::Context& context, entry_point entry, std::string const& plan)
{
  if (entry == entry_point::fragment_build) {
    auto fragment = sirius::ffi::make_fragment(context);
    fragment->build(plan);
  } else {
    // Also release a stream if a broken guard unexpectedly lets execution succeed.
    struct output_stream {
      ArrowArrayStream stream{};
      ~output_stream()
      {
        if (stream.release) { stream.release(&stream); }
      }
    } output;
    context.execute_substrait(plan, reinterpret_cast<std::uintptr_t>(&output.stream));
  }
}

template <typename Operation>
std::string check_refusal(Operation&& operation, std::string const& prefix, std::string const& item)
{
  bool threw = false;
  std::string message;
  try {
    operation();
  } catch (std::exception const& error) {
    threw   = true;
    message = error.what();
  }
  INFO("exception text: " << message);
  CHECK(threw);
  CHECK(message.starts_with(prefix));
  CHECK(message.find(item) != std::string::npos);
  check_no_sentinels(message);
  CHECK(message.find('?') == std::string::npos);
  CHECK(message.find('#') == std::string::npos);
  CHECK(message.find("X-Amz-") == std::string::npos);
  return message;
}

std::string authentication_suffix()
{
  return "?X-Amz-Security-Token=" + std::string(object_store_sentinels[2]) +
         "&X-Amz-Signature=" + std::string(object_store_sentinels[1]) + "#" +
         std::string(object_store_sentinels[0]);
}

// Cleanup releases and joins the worker even if a main-thread assertion aborts the case.
struct capture_pause_guard {
  ~capture_pause_guard() { finish(); }

  void pause()
  {
    std::unique_lock lock(mutex);
    ++calls;
    hook_thread = std::this_thread::get_id();
    paused      = true;
    cv.notify_all();
    cv.wait(lock, [&] { return resumed; });
  }

  bool wait_until_paused()
  {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [&] { return paused; });
  }

  void finish()
  {
    {
      std::lock_guard lock(mutex);
      resumed = true;
    }
    cv.notify_all();
    if (worker.joinable()) { worker.join(); }
    sirius::ffi::testing::set_capture_pause_hook({});
  }

  std::mutex mutex;
  std::condition_variable cv;
  bool paused{};
  bool resumed{};
  std::size_t calls{};
  std::thread::id hook_thread;
  std::exception_ptr failure;
  std::thread worker;
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
    REQUIRE(context->object_store_config());
    check_fields(*context->object_store_config(), inputs);
  }
  {
    auto context = make_context_from_config("", config);
    REQUIRE(context);
    CHECK(context->object_store_config().get() == config.get());
    REQUIRE(context->object_store_config());
    check_fields(*context->object_store_config(), inputs);
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
  CHECK(defaults.uri_prefix.empty());

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

TEST_CASE("FFI object store accepts bucket and nested URI prefixes",
          "[isolated_context][ffi][object_store]")
{
  for (auto const* prefix : {"s3://b/", "s3://b/p/q/"}) {
    CAPTURE(prefix);
    object_store_inputs inputs;
    inputs.uri_prefix     = prefix;
    auto const configured = inputs.snapshot();
    auto context          = make_context_from_config(memory_config(), configured);
    CHECK(context->object_store_config().get() == configured.get());
    REQUIRE(context->object_store_config());
    check_fields(*context->object_store_config(), inputs);
    context->set_object_store(nullptr);
    context->set_object_store(configured);
    CHECK(context->object_store_config().get() == configured.get());
    REQUIRE(context->object_store_config());
    check_fields(*context->object_store_config(), inputs);
  }
}

TEST_CASE("FFI object store scopes remote plans before either entry point binds",
          "[isolated_context][ffi][object_store]")
{
  connection_observer observer;
  observer.positive_control();
  {
    object_store_inputs inputs;
    inputs.endpoint   = observer.endpoint();
    inputs.uri_prefix = "s3://bucket/allowed/";
    auto context      = make_context_from_config(memory_config(), inputs.snapshot());
    struct scope_case {
      std::string path;
      std::string display;
      bool in_scope;
    };
    std::vector<scope_case> const cases{
      {"s3://bucket/other/k.parquet", "s3://bucket/other/k.parquet", false},
      {"S3://bucket/allowed/k.parquet", "S3://bucket/allowed/k.parquet", true},
      {"s3://Bucket/allowed/k.parquet", "s3://Bucket/allowed/k.parquet", false},
      {"s3://bucket/allowed/../other/k.parquet", "s3://bucket/allowed/../other/k.parquet", false},
      {"s3://bucket/allowed//k.parquet", "s3://bucket/allowed//k.parquet", false},
      {"s3://bucket/allowed/%2e%2e/k.parquet", "s3://bucket/allowed/%2e%2e/k.parquet", false},
      {"s3://bucket/allowed/k.parquet", "s3://bucket/allowed/k.parquet", true},
      {"s3://u:part@" + std::string(object_store_sentinels[2]) + "@bucket/allowed/k@part.parquet",
       "s3://bucket/allowed/k@part.parquet",
       false},
    };
    for (auto const& item : cases) {
      for (auto const entry : {entry_point::execute, entry_point::fragment_build}) {
        CAPTURE(item.path, entry == entry_point::execute);
        auto const plan = read_plan({{item.path + authentication_suffix()}});
        check_refusal([&] { invoke_plan(*context, entry, plan); },
                      item.in_scope ? "object_store_route_unavailable:" : "object_store_scope:",
                      item.display);
      }
    }
  }
  observer.finish();
  CHECK(observer.error() == 0);
  CHECK(observer.accepted() == 1);  // Only the positive control.
}

TEST_CASE("FFI object store refuses remote plans before range validation without network access",
          "[isolated_context][ffi][object_store]")
{
  connection_observer observer;
  observer.positive_control();
  {
    object_store_inputs inputs;
    inputs.endpoint = observer.endpoint();
    inputs.uri_prefix.clear();
    auto const configured = inputs.snapshot();
    auto context          = make_context_from_config(memory_config(), nullptr);
    for (bool const has_snapshot : {false, true}) {
      context->set_object_store(has_snapshot ? configured : nullptr);
      std::string const token =
        has_snapshot ? "object_store_route_unavailable:" : "object_store_unconfigured:";
      for (auto const entry : {entry_point::execute, entry_point::fragment_build}) {
        for (bool const uri_path : {false, true}) {
          for (auto const& path :
               std::vector<std::string>{"s3://bucket/k@part.parquet",
                                        "s3://u:part@" + std::string(object_store_sentinels[2]) +
                                          "@bucket/k@part.parquet"}) {
            for (auto const& item :
                 std::vector<scan_item>{{path + authentication_suffix(), 0, 0, uri_path},
                                        {path + authentication_suffix(), 1, 1, uri_path},
                                        {path + authentication_suffix(),
                                         std::numeric_limits<std::uint64_t>::max(),
                                         1,
                                         uri_path}}) {
              CAPTURE(has_snapshot,
                      entry == entry_point::execute,
                      uri_path,
                      item.start,
                      item.length,
                      path);
              auto const plan    = read_plan({item});
              auto const message = check_refusal(
                [&] { invoke_plan(*context, entry, plan); }, token, "s3://bucket/k@part.parquet");
              CHECK(message.find("byte-range splits are only supported") == std::string::npos);
              CHECK(message.find("overflows a 64-bit offset") == std::string::npos);
              CHECK(message.find("path type other than uri_file") == std::string::npos);
            }
          }
        }
        // The invalid local read precedes the remote item in the plan. Removing the remote
        // item must restore the original local validation error without trying to open a file.
        std::vector<scan_item> local{{"/unopened/object_store.parquet"},
                                     {"/unopened/object_store.parquet", 1, 1}};
        auto mixed = local;
        mixed.push_back({"s3://bucket/k.parquet" + authentication_suffix()});
        auto const mixed_plan = read_plan(mixed);
        check_refusal(
          [&] { invoke_plan(*context, entry, mixed_plan); }, token, "s3://bucket/k.parquet");
        auto const local_plan = read_plan(local);
        auto const message    = check_refusal([&] { invoke_plan(*context, entry, local_plan); },
                                           "a read names",
                                           "/unopened/object_store.parquet");
        CHECK(message.find("both whole and by byte range") != std::string::npos);
      }
    }
  }
  observer.finish();
  CHECK(observer.error() == 0);
  CHECK(observer.accepted() == 1);  // Only the positive control.
}

TEST_CASE("FFI object store failed fragment builds recapture on retry",
          "[isolated_context][ffi][object_store]")
{
  connection_observer observer;
  observer.positive_control();
  {
    object_store_inputs inputs;
    inputs.endpoint = observer.endpoint();
    inputs.uri_prefix.clear();
    auto context    = make_context_from_config(memory_config(), nullptr);
    auto fragment   = sirius::ffi::make_fragment(*context);
    auto const plan = read_plan({{"s3://bucket/k.parquet" + authentication_suffix()}});
    auto build      = [&] { fragment->build(plan); };
    check_refusal(build, "object_store_unconfigured:", "s3://bucket/k.parquet");
    context->set_object_store(inputs.snapshot());
    check_refusal(build, "object_store_route_unavailable:", "s3://bucket/k.parquet");
    context->set_object_store(nullptr);
    check_refusal(build, "object_store_unconfigured:", "s3://bucket/k.parquet");
  }
  observer.finish();
  CHECK(observer.error() == 0);
  CHECK(observer.accepted() == 1);  // Only the positive control.
}

TEST_CASE("FFI object store preserves captured snapshots across publication changes",
          "[isolated_context][ffi][object_store]")
{
  connection_observer observer_a;
  connection_observer observer_b;
  observer_a.positive_control();
  observer_b.positive_control();
  REQUIRE(observer_a.endpoint() != observer_b.endpoint());
  {
    object_store_inputs inputs_a;
    inputs_a.endpoint = observer_a.endpoint();
    inputs_a.uri_prefix.clear();
    auto inputs_b         = inputs_a;
    inputs_b.endpoint     = observer_b.endpoint();
    inputs_b.uri_prefix   = "s3://other-bucket/";
    auto const snapshot_b = inputs_b.snapshot();
    auto const plan       = read_plan({{"s3://bucket/k.parquet" + authentication_suffix()}});

    for (auto const entry : {entry_point::execute, entry_point::fragment_build}) {
      for (bool const replace_with_b : {false, true}) {
        CAPTURE(entry == entry_point::execute, replace_with_b);
        auto context    = make_context_from_config(memory_config(), inputs_a.snapshot());
        auto const next = replace_with_b ? snapshot_b : nullptr;
        capture_pause_guard pause;
        sirius::ffi::testing::set_capture_pause_hook([&] { pause.pause(); });
        pause.worker         = std::thread([&] {
          try {
            invoke_plan(*context, entry, plan);
          } catch (...) {
            pause.failure = std::current_exception();
          }
        });
        auto const worker_id = pause.worker.get_id();
        REQUIRE(pause.wait_until_paused());
        context->set_object_store(next);
        CHECK(context->object_store_config().get() == next.get());

        pause.finish();
        CHECK(pause.calls == 1);
        CHECK(pause.hook_thread == worker_id);
        check_refusal(
          [&] {
            if (pause.failure) { std::rethrow_exception(pause.failure); }
          },
          "object_store_route_unavailable:",
          "s3://bucket/k.parquet");
        check_refusal([&] { invoke_plan(*context, entry, plan); },
                      replace_with_b ? "object_store_scope:" : "object_store_unconfigured:",
                      "s3://bucket/k.parquet");
        CHECK(pause.calls == 1);
      }
    }
  }
  observer_a.finish();
  observer_b.finish();
  CHECK(observer_a.error() == 0);
  CHECK(observer_b.error() == 0);
  CHECK(observer_a.accepted() == 1);
  CHECK(observer_b.accepted() == 1);
}
