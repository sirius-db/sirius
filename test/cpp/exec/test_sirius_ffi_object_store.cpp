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
#include "utils/parquet_fixture_utils.hpp"

#include <arpa/inet.h>
#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/common/arrow/arrow.hpp>
#include <poll.h>
#include <substrait/plan.pb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <barrier>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <latch>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

using sirius::ffi::make_context_from_config;
using sirius::ffi::ObjectStoreConfig;
using sirius::ffi::SigningMode;
using sirius::ffi::Transport;
using sirius::test::object_store_inputs;
using sirius::test::object_store_sentinels;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

constexpr char kFixtureUri[] = "s3://bucket/prefix/object";

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
  CHECK(message.find("X-Amz-Signature") == std::string::npos);
}

template <typename Operation>
std::string check_invalid(Operation&& operation,
                          std::string const& field,
                          char const* reason = nullptr)
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
  if (reason) { CHECK(message.find(reason) != std::string::npos); }
  check_no_sentinels(message);
  return message;
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
  char const* reason = nullptr;
};

std::vector<invalid_input> invalid_inputs()
{
  auto const missing_ca =
    fixture_path("missing_" + std::string(object_store_sentinels[2]) + ".pem");
  REQUIRE_FALSE(fs::exists(missing_ca));
  auto const ca_directory = fixture_path("");
  REQUIRE(fs::is_directory(ca_directory));
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
    {"zero port", &object_store_inputs::endpoint, "https://h:0", "endpoint"},
    {"out-of-range port", &object_store_inputs::endpoint, "https://h:65536", "endpoint"},
    {"nondecimal port", &object_store_inputs::endpoint, "https://h:12a", "endpoint"},
    {"empty port", &object_store_inputs::endpoint, "https://h:", "endpoint"},
    {"endpoint control byte", &object_store_inputs::endpoint, "https://h\x01", "endpoint"},
    {"endpoint DEL byte", &object_store_inputs::endpoint, "https://h\x7f", "endpoint"},
    {"endpoint space", &object_store_inputs::endpoint, "https://h ost", "endpoint"},
    {"unterminated IPv6 literal", &object_store_inputs::endpoint, "https://[::1", "endpoint"},
    {"missing CA file",
     &object_store_inputs::ca_bundle_path,
     missing_ca.string(),
     "ca_bundle_path"},
    {"CA directory",
     &object_store_inputs::ca_bundle_path,
     ca_directory.string(),
     "ca_bundle_path",
     "is not a regular file"},
    {"non-S3 prefix", &object_store_inputs::uri_prefix, "http://b/", "uri_prefix"},
    {"empty prefix bucket", &object_store_inputs::uri_prefix, "s3://", "uri_prefix"},
    {"missing prefix bucket", &object_store_inputs::uri_prefix, "s3:///p/", "uri_prefix"},
    {"bucket without separator", &object_store_inputs::uri_prefix, "s3://b", "uri_prefix"},
    {"prefix without separator", &object_store_inputs::uri_prefix, "s3://b/p", "uri_prefix"},
    {"empty prefix segment", &object_store_inputs::uri_prefix, "s3://b/a//", "uri_prefix"},
    {"parent prefix segment", &object_store_inputs::uri_prefix, "s3://b/../", "uri_prefix"},
    {"dot prefix segment", &object_store_inputs::uri_prefix, "s3://b/./", "uri_prefix"},
    {"percent prefix segment", &object_store_inputs::uri_prefix, "s3://b/%41/", "uri_prefix"},
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
         "&X-Amz-Signature=" + std::string(object_store_sentinels[4]) + "#" +
         std::string(object_store_sentinels[0]);
}

std::string iceberg_plan(std::string const& metadata, bool nested = false)
{
  substrait::Plan plan;
  auto* root = plan.add_relations()->mutable_root();
  root->add_names("a");
  auto* input = root->mutable_input();
  if (nested) { input = input->mutable_project()->mutable_input(); }
  input->mutable_read()->mutable_iceberg_table()->mutable_direct()->set_metadata_uri(metadata);
  std::string bytes;
  REQUIRE(plan.SerializeToString(&bytes));
  return bytes;
}

template <typename Operation>
std::string exception_text(Operation&& operation)
{
  std::string message;
  bool threw = false;
  try {
    operation();
  } catch (std::exception const& error) {
    threw   = true;
    message = error.what();
  }
  REQUIRE(threw);
  return message;
}

// A deadlocked deleter would also block thread/fixture destruction. End the test process
// with a distinct exit code rather than leaving a stuck GPU process after a failed guard.
struct operation_deadline {
  explicit operation_deadline(char const* operation)
    : worker([this, operation] {
        std::unique_lock lock(mutex);
        if (!cv.wait_for(lock, 60s, [&] { return done; })) {
          std::fprintf(stderr, "object-store test timeout: %s\n", operation);
          std::fflush(stderr);
          std::_Exit(124);
        }
      })
  {
  }
  ~operation_deadline()
  {
    {
      std::lock_guard lock(mutex);
      done = true;
    }
    cv.notify_all();
    worker.join();
  }
  std::mutex mutex;
  std::condition_variable cv;
  bool done{};
  std::thread worker;
};

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

struct queued_binding_guard {
  ~queued_binding_guard() { finish(); }
  void capture()
  {
    std::unique_lock lock(mutex);
    if (++captures == 1) {
      first_captured = true;
      cv.notify_all();
      cv.wait(lock, [&] { return release; });
    }
  }
  void entry()
  {
    // Replacing this hook from its own invocation must not need a held hook mutex.
    sirius::ffi::testing::set_transaction_entry_hook({});
    std::lock_guard lock(mutex);
    second_entered = true;
    cv.notify_all();
    // Return immediately: the second call queues on the connection held by the first call.
  }
  bool wait(bool const& condition)
  {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [&] { return condition; });
  }
  void finish()
  {
    {
      std::lock_guard lock(mutex);
      release = true;
    }
    cv.notify_all();
    if (first.joinable()) { first.join(); }
    if (second.joinable()) { second.join(); }
    sirius::ffi::testing::set_transaction_entry_hook({});
    sirius::ffi::testing::set_capture_pause_hook({});
  }
  std::mutex mutex;
  std::condition_variable cv;
  bool first_captured{}, second_entered{}, release{};
  unsigned captures{};
  std::exception_ptr first_error, second_error;
  std::thread first, second;
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
                  item.field,
                  item.reason);
  }
}

TEST_CASE("FFI object store publication rejects invalid configuration and preserves its snapshot",
          "[isolated_context][ffi][object_store]")
{
  auto const original = object_store_inputs{}.snapshot();
  auto context        = make_context_from_config(memory_config(), original);
  for (auto const& item : invalid_inputs()) {
    CAPTURE(item.name);
    object_store_inputs inputs;
    inputs.*(item.member) = item.value;
    check_invalid([&] { context->put_object_store(inputs.snapshot()); }, item.field, item.reason);
    CHECK(context->object_store_config_for(kFixtureUri).get() == original.get());
    // Keep later matrix entries independent after an unexpected successful publication.
    context->put_object_store(original);
  }
}

TEST_CASE("FFI object store rejects undeclared enum values before publication",
          "[isolated_context][ffi][object_store]")
{
  for (bool invalid_transport : {true, false}) {
    auto const* field = invalid_transport ? "transport" : "signing_mode";
    CAPTURE(field);
    object_store_inputs inputs;
    if (invalid_transport) {
      inputs.transport = static_cast<Transport>(255);
    } else {
      inputs.signing_mode = static_cast<SigningMode>(255);
    }
    auto const rejected = inputs.snapshot();
    // A broken factory validator must not bring up a second live engine.
    auto const factory_error =
      check_invalid([&] { (void)make_context_from_config(memory_config(), rejected); }, field);
    CHECK(factory_error.find("255") == std::string::npos);
    CHECK(factory_error.find("0xff") == std::string::npos);

    auto const original = object_store_inputs{}.snapshot();
    auto context        = make_context_from_config(memory_config(), original);
    object_store_inputs fallback_inputs;
    fallback_inputs.uri_prefix.clear();
    auto const fallback = fallback_inputs.snapshot();
    context->put_object_store(fallback);
    auto const publication_error =
      check_invalid([&] { context->put_object_store(rejected); }, field);
    CHECK(publication_error.find("255") == std::string::npos);
    CHECK(publication_error.find("0xff") == std::string::npos);
    CHECK(context->object_store_config_for(kFixtureUri).get() == original.get());
    CHECK(context->object_store_config_for("s3://other/object").get() == fallback.get());
  }
}

TEST_CASE("FFI object store accepts IPv6 literals and decimal endpoint ports",
          "[isolated_context][ffi][object_store]")
{
  auto context = make_context_from_config(memory_config());
  for (auto const* endpoint : {"https://h:9000",
                               "https://[::1]:9000",
                               "https://[::1]",
                               "https://h:0080",
                               "https://h:1",
                               "https://h:65535"}) {
    CAPTURE(endpoint);
    object_store_inputs inputs;
    inputs.endpoint = endpoint;
    auto snapshot   = inputs.snapshot();
    REQUIRE_NOTHROW(context->put_object_store(snapshot));
    CHECK(context->object_store_config_for(kFixtureUri).get() == snapshot.get());
  }
}

TEST_CASE("FFI object store backend policy rejects RDMA and kvikIO options",
          "[isolated_context][ffi][object_store]")
{
  auto const ca = fixture_path("ffi_object_store_test_ca.pem");
  REQUIRE(fs::is_regular_file(ca));
  for (auto const field : {"transport", "signing_mode", "ca_bundle_path"}) {
    CAPTURE(field);
    auto const config = std::string_view(field) == "transport"
                          ? memory_config()
                          : fixture_path("ffi_object_store_kvikio.yaml").string();
    object_store_inputs inputs;
    auto const original = inputs.snapshot();
    if (std::string_view(field) == "transport") {
      inputs.transport = Transport::kRdma;
    } else if (std::string_view(field) == "signing_mode") {
      inputs.signing_mode = SigningMode::kHeader;
    } else {
      inputs.ca_bundle_path = ca.string();
    }
    auto const rejected = inputs.snapshot();
    check_invalid([&] { (void)make_context_from_config(config, rejected); }, field);
    auto context = make_context_from_config(config, original);
    check_invalid([&] { context->put_object_store(rejected); }, field);
    CHECK(context->object_store_config_for(kFixtureUri).get() == original.get());
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
  inputs.transport      = Transport::kHttp;
  inputs.signing_mode   = SigningMode::kHeader;
  inputs.ca_bundle_path = fixture_path("ffi_object_store_test_ca.pem").string();
  auto const config     = inputs.snapshot();
  {
    auto context = make_context_from_config("", nullptr);
    REQUIRE(context);
    CHECK(context->object_store_config_for(kFixtureUri) == nullptr);
    context->put_object_store(config);
    CHECK(context->object_store_config_for(kFixtureUri).get() == config.get());
    REQUIRE(context->object_store_config_for(kFixtureUri));
    check_fields(*context->object_store_config_for(kFixtureUri), inputs);
  }
  {
    auto context = make_context_from_config("", config);
    REQUIRE(context);
    CHECK(context->object_store_config_for(kFixtureUri).get() == config.get());
    REQUIRE(context->object_store_config_for(kFixtureUri));
    check_fields(*context->object_store_config_for(kFixtureUri), inputs);
    context->drop_object_store(config->uri_prefix);
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
    CHECK(context->object_store_config_for(kFixtureUri).get() == valid.get());
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
      auto snapshot = context->object_store_config_for(kFixtureUri);
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
    context->put_object_store(i % 2 == 0 ? second : first);
    std::this_thread::yield();
  }
  context->put_object_store(second);
  auto const deadline = std::chrono::steady_clock::now() + 2s;
  while (seen_second.load() == 0 && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  reader.request_stop();
  reader.join();
  CHECK(seen_first.load() > 0);
  CHECK(seen_second.load() > 0);
  CHECK(invalid.load() == 0);
  CHECK(context->object_store_config_for(kFixtureUri).get() == second.get());
  check_fields(*first, object_store_inputs{});
  check_fields(*second, inputs);
}

TEST_CASE("FFI object store credentials derive an immutable snapshot",
          "[isolated_context][ffi][object_store]")
{
  ObjectStoreConfig defaults{};
  CHECK(defaults.transport == Transport::kAuto);
  CHECK(defaults.signing_mode == SigningMode::kPresigned);
  CHECK(defaults.session_token.empty());
  CHECK(defaults.ca_bundle_path.empty());
  CHECK(defaults.uri_prefix.empty());

  connection_observer observer;
  observer.positive_control();
  object_store_inputs inputs;
  inputs.endpoint       = observer.endpoint();
  inputs.transport      = Transport::kHttp;
  inputs.signing_mode   = SigningMode::kHeader;
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
    retained     = context->object_store_config_for(kFixtureUri);

    auto const equal_values = inputs.snapshot();
    REQUIRE(equal_values.get() != original.get());
    context->put_object_store(equal_values);
    CHECK(context->object_store_config_for(kFixtureUri).get() == equal_values.get());
    context->drop_object_store(inputs.uri_prefix);
    CHECK(context->object_store_config_for(kFixtureUri) == nullptr);
    REQUIRE(retained);
    check_fields(*retained, inputs);
    context->put_object_store(retained);
    CHECK(context->object_store_config_for(kFixtureUri).get() == retained.get());
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
  for (auto const* prefix : {"s3://b/", "s3://b/p/q/", "s3://b/a/b/"}) {
    CAPTURE(prefix);
    object_store_inputs inputs;
    inputs.uri_prefix     = prefix;
    auto const configured = inputs.snapshot();
    auto context          = make_context_from_config(memory_config(), configured);
    auto const uri        = inputs.uri_prefix + "object";
    CHECK(context->object_store_config_for(uri).get() == configured.get());
    REQUIRE(context->object_store_config_for(uri));
    check_fields(*context->object_store_config_for(uri), inputs);
    context->drop_object_store(prefix);
    context->put_object_store(configured);
    CHECK(context->object_store_config_for(uri).get() == configured.get());
    REQUIRE(context->object_store_config_for(uri));
    check_fields(*context->object_store_config_for(uri), inputs);
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
       "s3://part.parquet",  // Conservative last-@ redaction.
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

TEST_CASE("FFI object store scopes bare empty and trailing object keys",
          "[isolated_context][ffi][object_store]")
{
  connection_observer observer;
  observer.positive_control();
  {
    object_store_inputs inputs;
    inputs.endpoint = observer.endpoint();
    auto context    = make_context_from_config(memory_config(), nullptr);
    struct scope_case {
      std::string prefix;
      std::string path;
      bool in_scope;
    };
    std::vector<scope_case> const cases{
      {"s3://bucket/", "s3://bucket/", false},
      {"s3://bucket/", "s3://bucket/allowed/k.parquet", true},
      {"s3://bucket/allowed/", "s3://bucket/allowed/", false},
      {"s3://bucket/allowed/", "s3://bucket/allowed/k.parquet", true},
      {"", "s3://bucket/", false},
      {"", "s3://bucket/allowed/", false},
    };
    for (auto const& item : cases) {
      context->drop_object_store(inputs.uri_prefix);
      inputs.uri_prefix = item.prefix;
      context->put_object_store(inputs.snapshot());
      for (auto const entry : {entry_point::execute, entry_point::fragment_build}) {
        CAPTURE(item.prefix, item.path, entry == entry_point::execute);
        auto const plan = read_plan({{item.path}});
        check_refusal([&] { invoke_plan(*context, entry, plan); },
                      item.in_scope ? "object_store_route_unavailable:" : "object_store_scope:",
                      item.path);
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
      if (has_snapshot) {
        context->put_object_store(configured);
      } else {
        context->drop_object_store("");
      }
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
              auto const plan = read_plan({item});
              // Last-@ display contract.
              auto const message = check_refusal(
                [&] { invoke_plan(*context, entry, plan); }, token, "s3://part.parquet");
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
    context->put_object_store(inputs.snapshot());
    check_refusal(build, "object_store_route_unavailable:", "s3://bucket/k.parquet");
    context->drop_object_store("");
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
    inputs_b.uri_prefix.clear();
    auto const replacement_b = inputs_b.snapshot();
    auto const plan          = read_plan({{"s3://bucket/k.parquet" + authentication_suffix()}});

    for (auto const entry : {entry_point::execute, entry_point::fragment_build}) {
      for (int action = 0; action != 4; ++action) {
        CAPTURE(entry == entry_point::execute, action);
        auto context = make_context_from_config(memory_config(), inputs_a.snapshot());
        std::weak_ptr<ObjectStoreConfig> captured =
          context->object_store_config_for("s3://bucket/k.parquet");
        auto const next = action == 1 ? snapshot_b : action == 2 ? replacement_b : nullptr;
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
        if (action == 1) {
          // The execution already holds its capture; finish both scoped edits before resuming it.
          context->drop_object_store("");
          context->put_object_store(next);
        } else if (action == 2) {
          context->put_object_store(next);
        } else {
          context->drop_object_store("");
        }
        auto const* observed_uri =
          action == 1 ? "s3://other-bucket/object" : "s3://bucket/k.parquet";
        CHECK(context->object_store_config_for(observed_uri).get() == next.get());
        CHECK_FALSE(captured.expired());

        pause.finish();
        CHECK(captured.expired());
        CHECK(pause.calls == 1);
        CHECK(pause.hook_thread == worker_id);
        check_refusal(
          [&] {
            if (pause.failure) { std::rethrow_exception(pause.failure); }
          },
          "object_store_route_unavailable:",
          "s3://bucket/k.parquet");
        check_refusal([&] { invoke_plan(*context, entry, plan); },
                      action == 1   ? "object_store_scope:"
                      : action == 2 ? "object_store_route_unavailable:"
                                    : "object_store_unconfigured:",
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

TEST_CASE("FFI object store redacts credential-bearing URI forms at both entries",
          "[isolated_context][ffi][object_store]")
{
  connection_observer observer;
  observer.positive_control();
  object_store_inputs inputs;
  inputs.endpoint   = observer.endpoint();
  inputs.secret_key = object_store_sentinels[3];
  auto const secret = inputs.secret_key;
  struct row {
    std::string uri;
    std::string display;
    bool iceberg{};
    char const* default_token = "object_store_route_unavailable:";
  };
  // Explicit protocol examples, not an oracle computed by the production formatter.
  std::vector<row> const cases{
    {"s3://u:" + secret + "/with/slash@bucket/k", "s3://bucket/k"},
    {"s3://u:" + secret + "?x@bucket/k", "<redacted-uri>"},
    {"s3://u:" + secret + "#x@bucket/k", "<redacted-uri>"},
    {"//u:" + secret + "@host/k.metadata.json", "<redacted-uri>", true},
    {"//u:" + secret + "@host/k?next=https://x@y", "<redacted-uri>", true},
    {"s3://bucket/k@part.parquet", "s3://part.parquet"},
    {"s3://bucket/k", "s3://bucket/k"},
    {"s3://bucket/k%40part", "s3://bucket/k%40part", false, "object_store_scope:"},
  };
  auto context = make_context_from_config(memory_config());
  for (int state = 0; state != 3; ++state) {
    context->drop_object_store("s3://bucket/allowed/");
    context->drop_object_store("");
    inputs.uri_prefix = state == 1 ? "s3://bucket/allowed/" : "";
    if (state != 0) { context->put_object_store(inputs.snapshot()); }
    for (auto const& item : cases) {
      std::string const token = state == 0   ? "object_store_unconfigured:"
                                : state == 1 ? "object_store_scope:"
                                             : item.default_token;
      auto const plan         = item.iceberg ? iceberg_plan(item.uri) : read_plan({{item.uri}});
      for (auto entry : {entry_point::execute, entry_point::fragment_build}) {
        CAPTURE(state, item.uri, entry == entry_point::execute);
        check_refusal([&] { invoke_plan(*context, entry, plan); }, token, item.display);
      }
    }
  }
  observer.finish();
  CHECK(observer.error() == 0);
  CHECK(observer.accepted() == 1);
}

TEST_CASE("FFI object store local range errors redact user-info before binding",
          "[isolated_context][ffi][object_store]")
{
  auto context = make_context_from_config(memory_config());
  auto const uri =
    "file://u:" + std::string(object_store_sentinels[3]) + "@localhost/tmp/k.parquet";
  auto const plan = read_plan({{uri, std::numeric_limits<std::uint64_t>::max(), 1}});
  for (auto entry : {entry_point::execute, entry_point::fragment_build}) {
    auto const message = exception_text([&] { invoke_plan(*context, entry, plan); });
    INFO(message);
    CHECK(message.find("overflow") != std::string::npos);
    CHECK(message.find("file://localhost/tmp/k.parquet") != std::string::npos);
    CHECK(message.find("object_store_") == std::string::npos);
    check_no_sentinels(message);
  }
}

TEST_CASE("FFI object store inventories direct and nested Iceberg metadata before binding",
          "[isolated_context][ffi][object_store]")
{
  connection_observer observer;
  observer.positive_control();
  object_store_inputs inputs;
  inputs.endpoint = observer.endpoint();
  auto context    = make_context_from_config(memory_config());
  for (int state = 0; state != 3; ++state) {
    context->drop_object_store("s3://bucket/allowed/");
    context->drop_object_store("");
    inputs.uri_prefix = state == 1 ? "s3://bucket/allowed/" : "";
    if (state != 0) { context->put_object_store(inputs.snapshot()); }
    std::string const token = state == 0   ? "object_store_unconfigured:"
                              : state == 1 ? "object_store_scope:"
                                           : "object_store_route_unavailable:";
    for (auto const& uri : std::vector<std::string>{"s3://other-bucket/t/metadata/v1.json",
                                                    "/tmp/m.json",
                                                    "file:///tmp/m.json",
                                                    observer.endpoint() + "/m.json"}) {
      for (bool nested : {false, true}) {
        auto const plan = iceberg_plan(uri, nested);
        for (auto entry : {entry_point::execute, entry_point::fragment_build}) {
          CAPTURE(state, uri, nested, entry == entry_point::execute);
          check_refusal([&] { invoke_plan(*context, entry, plan); }, token, uri);
        }
      }
    }
    for (auto entry : {entry_point::execute, entry_point::fragment_build}) {
      auto const message = exception_text([&] { invoke_plan(*context, entry, iceberg_plan("")); });
      INFO(message);
      CHECK(message.find("object_store_") == std::string::npos);
      CHECK(message.find("Metadata file missing") != std::string::npos);
      check_no_sentinels(message);
    }
  }
  observer.finish();
  CHECK(observer.error() == 0);
  CHECK(observer.accepted() == 1);
}

TEST_CASE("FFI object store scoped sets preserve longest matches and supplied pointers",
          "[isolated_context][ffi][object_store]")
{
  auto context = make_context_from_config(memory_config());
  object_store_inputs inputs;
  inputs.uri_prefix = "s3://b/x/";
  auto a            = inputs.snapshot();
  inputs.uri_prefix = "s3://b/x/y/";
  auto b            = inputs.snapshot();
  inputs.uri_prefix.clear();
  auto d = inputs.snapshot();
  context->put_object_store(a);
  CHECK(context->object_store_config_for("s3://b/x/object").get() == a.get());
  CHECK(context->object_store_config_for("s3://b/x/f").get() == a.get());
  CHECK(context->object_store_config_for("s3://other/f") == nullptr);
  context->put_object_store(b);
  CHECK(context->object_store_config_for("s3://unclaimed/object") == nullptr);
  context->put_object_store(d);
  CHECK(context->object_store_config_for("s3://unclaimed/object").get() == d.get());
  CHECK(context->object_store_config_for("s3://b/x/f").get() == a.get());
  CHECK(context->object_store_config_for("s3://b/x/y/f").get() == b.get());
  CHECK(context->object_store_config_for("S3://b/x/y/f").get() == b.get());
  CHECK(context->object_store_config_for("s3://B/x/y/f").get() == d.get());
  CHECK(context->object_store_config_for("s3://other/f").get() == d.get());
  inputs.uri_prefix = "s3://b/x/y/";
  auto replacement  = inputs.snapshot();
  context->put_object_store(replacement);
  CHECK(context->object_store_config_for("s3://b/x/y/f").get() == replacement.get());
  CHECK(context->object_store_config_for("s3://b/x/f").get() == a.get());
  REQUIRE_THROWS_AS(context->put_object_store(nullptr), std::invalid_argument);
  CHECK(context->object_store_config_for("s3://b/x/y/f").get() == replacement.get());
  context->drop_object_store(b->uri_prefix);
  CHECK(context->object_store_config_for("s3://b/x/y/f").get() == a.get());
  context->drop_object_store("");
  CHECK(context->object_store_config_for("s3://other/f") == nullptr);
  context->drop_object_store(a->uri_prefix);
  context->put_object_store(b);
  CHECK(context->object_store_config_for("s3://b/x/y/object").get() == b.get());
  CHECK(context->object_store_config_for("s3://b/x/f") == nullptr);
  context->drop_object_store(b->uri_prefix);
  CHECK(context->object_store_config_for("s3://unclaimed/object") == nullptr);
  CHECK(context->object_store_config_for("s3://b/x/y/f") == nullptr);
}

TEST_CASE("FFI object store applies the raw-key gate before selecting a scope",
          "[isolated_context][ffi][object_store]")
{
  enum class resolution { none, scoped, fallback, refused };
  struct scope_state {
    char const* name;
    bool has_prefix;
    bool has_default;
  };
  std::array<scope_state, 4> const states{{
    {"empty", false, false},
    {"default only", false, true},
    {"prefix only", true, false},
    {"prefix and default", true, true},
  }};
  struct scope_case {
    char const* uri;
    std::array<resolution, 4> expected;
  };
  std::vector<scope_case> const cases{
    {"s3://b/allowed/x",
     {resolution::none, resolution::fallback, resolution::scoped, resolution::scoped}},
    {"s3://b/other/x",
     {resolution::none, resolution::fallback, resolution::none, resolution::fallback}},
    {"s3://b", {resolution::none, resolution::fallback, resolution::none, resolution::fallback}},
    {"S3://b/allowed/x",
     {resolution::none, resolution::fallback, resolution::scoped, resolution::scoped}},
    {"s3://b/other/../allowed/x",
     {resolution::none, resolution::refused, resolution::refused, resolution::refused}},
    {"s3://b/./x",
     {resolution::none, resolution::refused, resolution::refused, resolution::refused}},
    {"s3://b//x",
     {resolution::none, resolution::refused, resolution::refused, resolution::refused}},
    {"s3://b/%61llowed/x",
     {resolution::none, resolution::refused, resolution::refused, resolution::refused}},
    {"s3://b/allowed/report%202026.parquet",
     {resolution::none, resolution::refused, resolution::refused, resolution::refused}},
    {"s3://b/allowed/x/",
     {resolution::none, resolution::refused, resolution::refused, resolution::refused}},
    {"s3://b/", {resolution::none, resolution::refused, resolution::refused, resolution::refused}},
  };

  connection_observer observer;
  observer.positive_control();
  {
    auto context = make_context_from_config(memory_config());
    object_store_inputs inputs;
    inputs.endpoint   = observer.endpoint();
    inputs.uri_prefix = "s3://b/allowed/";
    auto scoped       = inputs.snapshot();
    inputs.uri_prefix.clear();
    inputs.access_key += "_default";
    auto fallback = inputs.snapshot();

    for (std::size_t index = 0; index < states.size(); ++index) {
      auto const& state = states[index];
      context->drop_object_store(scoped->uri_prefix);
      context->drop_object_store("");
      if (state.has_prefix) { context->put_object_store(scoped); }
      if (state.has_default) { context->put_object_store(fallback); }
      for (auto const& item : cases) {
        CAPTURE(state.name, item.uri);
        auto const expected = item.expected[index];
        if (expected == resolution::refused) {
          check_refusal([&] { (void)context->object_store_config_for(item.uri); },
                        "object_store_scope:",
                        item.uri);
        } else {
          ObjectStoreConfig const* snapshot = nullptr;
          if (expected == resolution::scoped) { snapshot = scoped.get(); }
          if (expected == resolution::fallback) { snapshot = fallback.get(); }
          CHECK(context->object_store_config_for(item.uri).get() == snapshot);
        }

        auto const empty = !state.has_prefix && !state.has_default;
        auto const token = empty ? "object_store_unconfigured:"
                           : expected == resolution::none || expected == resolution::refused
                             ? "object_store_scope:"
                             : "object_store_route_unavailable:";
        auto const plan  = read_plan({{item.uri}});
        for (auto const entry : {entry_point::execute, entry_point::fragment_build}) {
          CAPTURE(entry == entry_point::execute);
          auto const message =
            check_refusal([&] { invoke_plan(*context, entry, plan); }, token, item.uri);
          if (empty) {
            CHECK(message.find("or through an Iceberg metadata path") != std::string::npos);
          }
        }
      }
    }
  }
  observer.finish();
  CHECK(observer.error() == 0);
  CHECK(observer.accepted() == 1);  // Only the positive control.
}

TEST_CASE("FFI object store getters and execution refuse invalid selected keys without fallback",
          "[isolated_context][ffi][object_store]")
{
  auto context = make_context_from_config(memory_config());
  object_store_inputs inputs;
  inputs.uri_prefix.clear();
  auto fallback = inputs.snapshot();
  for (auto const* prefix : {"s3://bucket/", "s3://bucket/allowed/"}) {
    inputs.uri_prefix = prefix;
    auto scoped       = inputs.snapshot();
    for (bool with_default : {false, true}) {
      context->drop_object_store("s3://bucket/");
      context->drop_object_store("s3://bucket/allowed/");
      context->drop_object_store("");
      context->put_object_store(scoped);
      if (with_default) { context->put_object_store(fallback); }
      for (auto const* suffix : {"", ".", "..", "./f", "../f", "a//f", "/f", "a%2fb", "f/"}) {
        std::string const uri = std::string(prefix) + suffix;
        CAPTURE(uri, with_default);
        check_refusal(
          [&] { (void)context->object_store_config_for(uri); }, "object_store_scope:", uri);
        auto const plan = read_plan({{uri}});
        for (auto entry : {entry_point::execute, entry_point::fragment_build}) {
          check_refusal([&] { invoke_plan(*context, entry, plan); }, "object_store_scope:", uri);
        }
      }
      CHECK(context->object_store_config_for(std::string(prefix) + "good").get() == scoped.get());
      CHECK(context->object_store_config_for("s3://elsewhere/good").get() ==
            (with_default ? fallback.get() : nullptr));
    }
  }
}

TEST_CASE("FFI object store drop validates keys and leaves absent keys unchanged",
          "[isolated_context][ffi][object_store]")
{
  object_store_inputs inputs;
  inputs.uri_prefix = "s3://bucket/allowed/";
  auto original     = inputs.snapshot();
  auto context      = make_context_from_config(memory_config(), original);
  for (auto const* key : {"s3://bucket/allowed",
                          "S3://bucket/allowed/",
                          "http://b/p/",
                          "s3://b/p/?x",
                          "s3://b/a//",
                          "s3://b/../",
                          "s3://b/%41/",
                          "s3://b/./"}) {
    CAPTURE(key);
    check_invalid([&] { context->drop_object_store(key); }, "uri_prefix");
    CHECK(context->object_store_config_for("s3://bucket/allowed/object").get() == original.get());
    CHECK(context->object_store_config_for("s3://bucket/allowed/f").get() == original.get());
  }
  operation_deadline deadline("absent drop while a capture is retained");
  capture_pause_guard pause;
  auto const plan = read_plan({{"s3://bucket/allowed/f"}});
  sirius::ffi::testing::set_capture_pause_hook([&] { pause.pause(); });
  pause.worker = std::thread([&] {
    try {
      invoke_plan(*context, entry_point::execute, plan);
    } catch (...) {
      pause.failure = std::current_exception();
    }
  });
  REQUIRE(pause.wait_until_paused());
  // An absent key leaves membership, config pointers and retained ownership unchanged.
  auto const owners = original.use_count();
  CHECK_NOTHROW(context->drop_object_store("s3://bucket/missing/"));
  CHECK_NOTHROW(context->drop_object_store(""));
  CHECK(original.use_count() == owners);
  CHECK(context->object_store_config_for("s3://bucket/allowed/object").get() == original.get());
  CHECK(context->object_store_config_for("s3://bucket/allowed/f").get() == original.get());
  pause.finish();
  check_refusal(
    [&] {
      if (pause.failure) { std::rethrow_exception(pause.failure); }
    },
    "object_store_route_unavailable:",
    "s3://bucket/allowed/f");
  check_refusal([&] { invoke_plan(*context, entry_point::execute, plan); },
                "object_store_route_unavailable:",
                "s3://bucket/allowed/f");
}

TEST_CASE("FFI object store concurrent writers have serializable membership and pointer identity",
          "[isolated_context][ffi][object_store]")
{
  auto context = make_context_from_config(memory_config());
  object_store_inputs inputs;
  inputs.uri_prefix = "s3://b/a/";
  auto a            = inputs.snapshot();
  inputs.uri_prefix = "s3://b/b/";
  auto b            = inputs.snapshot();
  inputs.uri_prefix = "s3://b/c/";
  auto c            = inputs.snapshot();
  operation_deadline deadline("concurrent scope-set writers");
  // Each pair has a coordinated start. This does not claim an internal writer-pause seam.
  for (int round = 0; round != 16; ++round) {
    for (int pair = 0; pair != 4; ++pair) {
      CAPTURE(round, pair);
      context->drop_object_store(a->uri_prefix);
      context->drop_object_store(b->uri_prefix);
      context->drop_object_store(c->uri_prefix);
      context->put_object_store(a);
      std::barrier start(3);
      std::exception_ptr first_error, second_error;
      std::jthread first([&] {
        start.arrive_and_wait();
        try {
          if (pair == 3) {
            // Race a same-key drop against put, keeping unrelated A published.
            context->drop_object_store(b->uri_prefix);
          } else {
            context->put_object_store(b);
          }
        } catch (...) {
          first_error = std::current_exception();
        }
      });
      std::jthread second([&] {
        start.arrive_and_wait();
        try {
          if (pair == 0) {
            context->put_object_store(c);
          } else if (pair == 1) {
            context->drop_object_store(a->uri_prefix);
          } else if (pair == 2) {
            context->drop_object_store("s3://b/missing/");
          } else {
            context->put_object_store(b);
          }
        } catch (...) {
          second_error = std::current_exception();
        }
      });
      start.arrive_and_wait();
      first.join();
      second.join();
      REQUIRE_FALSE(first_error);
      REQUIRE_FALSE(second_error);
      CHECK(context->object_store_config_for("s3://b/a/f").get() ==
            (pair == 1 ? nullptr : a.get()));
      auto const observed_b = context->object_store_config_for("s3://b/b/f");
      if (pair == 3) {
        CHECK((observed_b == nullptr || observed_b.get() == b.get()));
      } else {
        CHECK(observed_b.get() == b.get());
      }
      CHECK(context->object_store_config_for("s3://b/c/f").get() ==
            (pair == 0 ? c.get() : nullptr));
    }
  }
}

TEST_CASE("FFI object store retired owners can reenter publication methods",
          "[isolated_context][ffi][object_store]")
{
  auto context = std::shared_ptr<sirius::ffi::Context>(make_context_from_config(memory_config()));
  std::weak_ptr<sirius::ffi::Context> weak_context = context;
  object_store_inputs inputs;
  inputs.uri_prefix = "s3://b/new/";
  auto replacement  = inputs.snapshot();
  inputs.uri_prefix = "s3://b/extra/";
  auto extra        = inputs.snapshot();
  operation_deadline deadline("retired configuration reentry");
  for (int operation = 0; operation != 4; ++operation) {
    struct result {
      std::atomic<bool> deleted{}, saw_new{}, failed{};
    };
    auto state        = std::make_shared<result>();
    inputs.uri_prefix = replacement->uri_prefix;
    auto old          = std::shared_ptr<ObjectStoreConfig>(
      new ObjectStoreConfig(*inputs.snapshot()),
      [weak_context, state, replacement, extra, operation](ObjectStoreConfig* raw) {
        std::unique_ptr<ObjectStoreConfig> reclaim(raw);
        if (auto owner = weak_context.lock()) {
          try {
            bool saw_new =
              owner->object_store_config_for("s3://b/new/object").get() == replacement.get();
            if (operation == 0) { owner->put_object_store(extra); }
            if (operation == 1) { owner->drop_object_store(replacement->uri_prefix); }
            if (operation == 2) {
              owner->drop_object_store(replacement->uri_prefix);
              owner->put_object_store(extra);
            }
            if (operation == 3) {
              saw_new = saw_new &&
                        owner->object_store_config_for("s3://b/new/f").get() == replacement.get();
            }
            state->saw_new = saw_new;
          } catch (...) {
            state->failed = true;
          }
          state->deleted = true;
        }
      });
    context->put_object_store(old);
    old.reset();
    context->put_object_store(replacement);
    CAPTURE(operation);
    CHECK(state->deleted.load());
    CHECK(state->saw_new.load());
    CHECK_FALSE(state->failed.load());
    CHECK(context->object_store_config_for("s3://b/new/f").get() ==
          ((operation == 0 || operation == 3) ? replacement.get() : nullptr));
    CHECK(context->object_store_config_for("s3://b/extra/f").get() ==
          ((operation == 0 || operation == 2) ? extra.get() : nullptr));
    context->drop_object_store(replacement->uri_prefix);
    context->drop_object_store(extra->uri_prefix);
  }
}

TEST_CASE("FFI object store checks the full inventory before reporting an unavailable route",
          "[isolated_context][ffi][object_store]")
{
  object_store_inputs inputs;
  inputs.uri_prefix = "s3://bucket/allowed/";
  auto context      = make_context_from_config(memory_config(), inputs.snapshot());
  for (auto const* bad : {"s3://elsewhere/f", "s3://bucket/allowed/../outside"}) {
    auto const plan = read_plan({{"s3://bucket/allowed/good"}, {bad, 1, 1}});
    for (auto entry : {entry_point::execute, entry_point::fragment_build}) {
      check_refusal([&] { invoke_plan(*context, entry, plan); }, "object_store_scope:", bad);
    }
  }
}

TEST_CASE("FFI object store queued binding captures after transaction admission",
          "[isolated_context][ffi][object_store]")
{
  object_store_inputs inputs;
  inputs.uri_prefix      = "s3://b/a/";
  auto a                 = inputs.snapshot();
  inputs.uri_prefix      = "s3://b/b/";
  auto b                 = inputs.snapshot();
  auto const first_plan  = read_plan({{"s3://b/a/f"}});
  auto const second_plan = read_plan({{"s3://b/b/f"}});
  for (auto entry : {entry_point::execute, entry_point::fragment_build}) {
    auto context = make_context_from_config(memory_config(), a);
    operation_deadline deadline("queued transaction admission");
    queued_binding_guard queue;
    sirius::ffi::testing::set_capture_pause_hook([&] { queue.capture(); });
    queue.first = std::thread([&] {
      try {
        invoke_plan(*context, entry_point::execute, first_plan);
      } catch (...) {
        queue.first_error = std::current_exception();
      }
    });
    REQUIRE(queue.wait(queue.first_captured));
    sirius::ffi::testing::set_transaction_entry_hook([&] { queue.entry(); });
    queue.second = std::thread([&] {
      try {
        invoke_plan(*context, entry, second_plan);
      } catch (...) {
        queue.second_error = std::current_exception();
      }
    });
    REQUIRE(queue.wait(queue.second_entered));
    // The first query is paused with A captured, and the second is still awaiting admission.
    context->drop_object_store(a->uri_prefix);
    context->put_object_store(b);
    CHECK(context->object_store_config_for("s3://b/b/f").get() == b.get());
    queue.finish();
    CHECK(queue.captures == 2);
    check_refusal(
      [&] {
        if (queue.first_error) { std::rethrow_exception(queue.first_error); }
      },
      "object_store_route_unavailable:",
      "s3://b/a/f");
    // A pre-transaction capture would still contain A and produce object_store_scope here.
    check_refusal(
      [&] {
        if (queue.second_error) { std::rethrow_exception(queue.second_error); }
      },
      "object_store_route_unavailable:",
      "s3://b/b/f");
  }
}

TEST_CASE("FFI object store failed execution attempts release captures before retry",
          "[isolated_context][ffi][object_store]")
{
  for (auto entry : {entry_point::execute, entry_point::fragment_build}) {
    CAPTURE(entry == entry_point::execute);
    bool deleted = false, reentered = false;
    auto context = std::shared_ptr<sirius::ffi::Context>(make_context_from_config(memory_config()));
    std::weak_ptr<sirius::ffi::Context> weak_context = context;
    object_store_inputs inputs;
    inputs.uri_prefix = "s3://bucket/allowed/";
    auto a            = std::shared_ptr<ObjectStoreConfig>(
      new ObjectStoreConfig(*inputs.snapshot()), [&, weak_context](ObjectStoreConfig* raw) {
        std::unique_ptr<ObjectStoreConfig> reclaim(raw);
        if (auto owner = weak_context.lock()) {
          ArrowArrayStream output{};
          try {
            owner->execute_substrait("not a substrait plan",
                                     reinterpret_cast<std::uintptr_t>(&output));
          } catch (std::exception const&) {
            reentered = true;
          }
          if (output.release) { output.release(&output); }
          deleted = true;
        }
      });
    context->put_object_store(a);
    std::weak_ptr<ObjectStoreConfig> weak_a = a;
    a.reset();
    operation_deadline deadline("failed execution capture cleanup");
    auto fragment     = sirius::ffi::make_fragment(*context);
    auto const remote = read_plan({{"s3://bucket/allowed/f"}});
    auto invoke       = [&](std::string const& plan) {
      if (entry == entry_point::fragment_build) {
        fragment->build(plan);
      } else {
        invoke_plan(*context, entry, plan);
      }
    };
    capture_pause_guard pause;
    sirius::ffi::testing::set_capture_pause_hook([&] { pause.pause(); });
    pause.worker = std::thread([&] {
      try {
        invoke(remote);
      } catch (...) {
        pause.failure = std::current_exception();
      }
    });
    REQUIRE(pause.wait_until_paused());
    context->drop_object_store(inputs.uri_prefix);
    CHECK_FALSE(weak_a.expired());
    pause.finish();
    check_refusal(
      [&] {
        if (pause.failure) { std::rethrow_exception(pause.failure); }
      },
      "object_store_route_unavailable:",
      "s3://bucket/allowed/f");
    CHECK(deleted);
    CHECK(reentered);
    CHECK(weak_a.expired());
    check_refusal([&] { invoke(remote); }, "object_store_unconfigured:", "s3://bucket/allowed/f");
    CHECK_THROWS(invoke("not a substrait plan"));
  }
}

TEST_CASE("FFI object store successful local fragments retain captures through run",
          "[isolated_context][ffi][object_store]")
{
  sirius::test::scratch_dir files("ffi_object_store_local");
  {
    sirius::test::scoped_sirius_disable disable;
    duckdb::DuckDB db(nullptr);
    duckdb::Connection connection(db);
    auto copied = connection.Query("COPY (SELECT 42::BIGINT AS a) TO " +
                                   files.file_literal("one.parquet") + " (FORMAT PARQUET)");
    REQUIRE(copied);
    REQUIRE_FALSE(copied->HasError());
  }
  auto a                                    = object_store_inputs{}.snapshot();
  std::weak_ptr<ObjectStoreConfig> retained = a;
  auto context  = make_context_from_config(memory_config(), std::move(a));
  auto fragment = sirius::ffi::make_fragment(*context);
  fragment->build(read_plan({{files.file("one.parquet")}}));
  context->drop_object_store("s3://bucket/prefix/");
  CHECK_FALSE(retained.expired());
  fragment->run();
  CHECK_FALSE(retained.expired());
  {
    struct result {
      ArrowArrayStream stream{};
      ArrowArray batch{};
      ~result()
      {
        if (batch.release) { batch.release(&batch); }
        if (stream.release) { stream.release(&stream); }
      }
    } output;
    fragment->result_to_arrow(reinterpret_cast<std::uintptr_t>(&output.stream));
    REQUIRE(output.stream.get_next);
    REQUIRE(output.stream.get_next(&output.stream, &output.batch) == 0);
    REQUIRE(output.batch.release);
    REQUIRE(output.batch.n_children == 1);
    auto const* column = output.batch.children[0];
    REQUIRE(column->length == 1);
    REQUIRE(column->buffers[1]);
    CHECK(static_cast<std::int64_t const*>(column->buffers[1])[column->offset] == 42);
    output.batch.release(&output.batch);
    output.batch = {};
    REQUIRE(output.stream.get_next(&output.stream, &output.batch) == 0);
    CHECK(output.batch.release == nullptr);
  }
  fragment.reset();
  CHECK(retained.expired());
}
