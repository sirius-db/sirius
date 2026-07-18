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
#include "io/rdma/cuobj_rdma_reactor.hpp"
#include "io/rdma/mock_rdma_client.hpp"
#include "log/sink.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

extern char** environ;

namespace s3_rdma_f01b_tests {

using sirius::exec::semi_future;
using sirius::io::rdma::cuda_delivery_ops;
using sirius::io::rdma::cuobj_rdma_io_object;
using sirius::io::rdma::cuobj_rdma_reactor;
using sirius::io::rdma::data_get_result;
using sirius::io::rdma::mock_rdma_data_session_factory;
using sirius::io::rdma::rdma_data_session;
using sirius::io::rdma::rdma_data_session_factory;
using sirius::io::rdma::rx_route;
using namespace std::chrono_literals;

constexpr std::size_t k_slot_size               = 64UL << 10;
constexpr std::string_view k_bucket             = "bucket";
constexpr std::string_view k_key                = "fatal-boundary";
constexpr std::string_view k_child_env          = "SIRIUS_RDMA_DEATH_CHILD";
constexpr std::string_view k_sticky_env         = "SIRIUS_RDMA_STICKY_CODE";
constexpr std::string_view k_hook_mark          = "FATAL_HOOK_CALLED\n";
constexpr std::string_view k_term_mark          = "TERMINATE_CALLED\n";
constexpr std::string_view k_future_mark        = "FUTURE_RESOLVED\n";
constexpr std::string_view k_unwind_mark        = "EVENT_DESTROYED_DURING_FATAL\n";
constexpr std::string_view k_teardown_mark      = "NORMAL_TEARDOWN\n";
constexpr std::string_view k_arena_mark         = "ARENA_DEREGISTERED\n";
constexpr std::string_view k_full_pipe_mark     = "FULL_STDERR_PIPE_READY\n";
constexpr std::string_view k_fatal_trigger_mark = "DEFAULT_FATAL_TRIGGER_READY\n";

inline constexpr auto k_expected_sticky_codes = std::to_array<cudaError_t>({
  cudaErrorECCUncorrectable,
  cudaErrorNvlinkUncorrectable,
#if CUDART_VERSION >= 12080
  cudaErrorContained,
#endif
  cudaErrorIllegalAddress,
  cudaErrorLaunchTimeout,
  cudaErrorAssert,
  cudaErrorHardwareStackError,
  cudaErrorIllegalInstruction,
  cudaErrorMisalignedAddress,
  cudaErrorInvalidAddressSpace,
  cudaErrorInvalidPc,
  cudaErrorLaunchFailure,
#if CUDART_VERSION >= 12080
  cudaErrorTensorMemoryLeak,
#endif
  cudaErrorMpsClientTerminated,
  cudaErrorExternalDevice,
});

void emit_marker_to(int fd, std::string_view marker) noexcept
{
  auto* data        = marker.data();
  std::size_t bytes = marker.size();
  while (bytes != 0) {
    auto const written = ::write(fd, data, bytes);
    if (written > 0) {
      data += written;
      bytes -= static_cast<std::size_t>(written);
    } else if (written < 0 && errno == EINTR) {
      continue;
    } else {
      return;
    }
  }
}

void emit_marker(std::string_view marker) noexcept { emit_marker_to(STDERR_FILENO, marker); }

int install_full_stderr_pipe()
{
  int pipe_fds[2];
  if (::pipe(pipe_fds) != 0) {
    throw std::system_error(errno, std::generic_category(), "full-stderr pipe");
  }

  auto fail = [&](char const* what) -> void {
    auto const error = errno;
    (void)::close(pipe_fds[0]);
    (void)::close(pipe_fds[1]);
    throw std::system_error(error, std::generic_category(), what);
  };

  int const flags = ::fcntl(pipe_fds[1], F_GETFL);
  if (flags == -1 || ::fcntl(pipe_fds[1], F_SETFL, flags | O_NONBLOCK) == -1) {
    fail("full-stderr fcntl non-blocking");
  }

  std::array<char, 4096> fill{};
  for (;;) {
    auto const written = ::write(pipe_fds[1], fill.data(), fill.size());
    if (written > 0) { continue; }
    if (written < 0 && errno == EINTR) { continue; }
    if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) { break; }
    fail("full-stderr fill");
  }
  if (::fcntl(pipe_fds[1], F_SETFL, flags) == -1) { fail("full-stderr fcntl restore"); }

  emit_marker(k_full_pipe_mark);
  if (::dup2(pipe_fds[1], STDERR_FILENO) == -1) { fail("full-stderr dup2"); }
  (void)::close(pipe_fds[1]);
  return pipe_fds[0];  // kept open, but deliberately never read, until process exit
}

class throwing_log_sink final : public sirius::log::sink {
 public:
  void set_level(sirius::log::level) override {}
  [[nodiscard]] bool should_log(sirius::log::level) const override { return true; }

  void log(sirius::log::level, std::source_location const&, std::string_view) override
  {
    _calls.fetch_add(1, std::memory_order_relaxed);
    throw std::runtime_error("injected log sink failure");
  }

  bool flush() override { return true; }
  [[nodiscard]] int calls() const noexcept { return _calls.load(std::memory_order_relaxed); }

 private:
  std::atomic<int> _calls{0};
};

class scoped_log_sink {
 public:
  explicit scoped_log_sink(std::shared_ptr<sirius::log::sink> replacement)
    : _previous(sirius::log::get_sink())
  {
    sirius::log::set_sink(std::move(replacement));
  }

  ~scoped_log_sink() { sirius::log::set_sink(std::move(_previous)); }

  scoped_log_sink(scoped_log_sink const&)            = delete;
  scoped_log_sink& operator=(scoped_log_sink const&) = delete;

 private:
  std::shared_ptr<sirius::log::sink> _previous;
};

bool cuda_device_available()
{
  int count       = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count == 0) {
    WARN("Skipping S3 RDMA fatal-boundary test: no CUDA device is available");
    return false;
  }
  REQUIRE(cudaSetDevice(0) == cudaSuccess);
  return true;
}

struct recording_session_state {
  explicit recording_session_state(bool mark) : mark_deregister(mark) {}

  bool mark_deregister;
  mutable std::mutex mutex;
  std::vector<void*> destinations;
};

class recording_session final : public rdma_data_session {
 public:
  recording_session(std::unique_ptr<rdma_data_session> inner,
                    std::shared_ptr<recording_session_state> state)
    : _inner(std::move(inner)), _state(std::move(state))
  {
    if (!_inner) { throw std::invalid_argument("recording_session: null inner session"); }
  }

  void register_memory(void* base, std::size_t bytes) override
  {
    _inner->register_memory(base, bytes);
  }

  void deregister_memory(void* base) noexcept override
  {
    if (_state->mark_deregister) { emit_marker(k_arena_mark); }
    _inner->deregister_memory(base);
  }

  data_get_result get(const rx_route& route,
                      std::size_t offset,
                      std::size_t size,
                      void* dst) override
  {
    {
      std::lock_guard lock{_state->mutex};
      _state->destinations.push_back(dst);
    }
    return _inner->get(route, offset, size, dst);
  }

 private:
  std::unique_ptr<rdma_data_session> _inner;
  std::shared_ptr<recording_session_state> _state;
};

class recording_session_factory final : public rdma_data_session_factory {
 public:
  explicit recording_session_factory(bool mark_deregister = false)
    : _inner(std::make_shared<mock_rdma_data_session_factory>()),
      _state(std::make_shared<recording_session_state>(mark_deregister))
  {
  }

  void put_object(std::vector<std::uint8_t> bytes)
  {
    _inner->put_object(std::string{k_bucket}, std::string{k_key}, std::move(bytes));
  }

  [[nodiscard]] std::size_t get_count() const
  {
    std::lock_guard lock{_state->mutex};
    return _state->destinations.size();
  }

  [[nodiscard]] void* get_destination(std::size_t index) const
  {
    std::lock_guard lock{_state->mutex};
    return _state->destinations.at(index);
  }

  std::unique_ptr<rdma_data_session> acquire() override
  {
    return std::make_unique<recording_session>(_inner->acquire(), _state);
  }

 private:
  std::shared_ptr<mock_rdma_data_session_factory> _inner;
  std::shared_ptr<recording_session_state> _state;
};

cuobj_rdma_reactor::config reactor_config()
{
  cuobj_rdma_reactor::config cfg;
  cfg.max_inflight    = 1;
  cfg.arena_slot_size = k_slot_size;
  return cfg;
}

std::vector<std::uint8_t> payload_bytes(std::uint8_t salt = 53)
{
  std::vector<std::uint8_t> bytes(k_slot_size);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>((i * 131U + salt) & 0xffU);
  }
  return bytes;
}

class reactor_fixture {
 public:
  explicit reactor_fixture(cuda_delivery_ops ops,
                           bool flush_before_copy = false,
                           bool mark_deregister   = false)
    : client(std::make_shared<recording_session_factory>(mark_deregister)),
      control(std::make_shared<sirius::io::rdma::mock_s3_control_client>()),
      context(std::make_shared<cuobj_rdma_reactor::reactor_context>(
        reactor_config(),
        sirius::io::rdma::rdma_transport_clients{control, client},
        std::move(ops))),
      reactor(context),
      object("s3://bucket/fatal-boundary", std::string{k_bucket}, std::string{k_key}, k_slot_size)
  {
    client->put_object(payload_bytes());
    context->set_flush_before_copy(flush_before_copy);
    reactor.start();
  }

  ~reactor_fixture()
  {
    try {
      reactor.shutdown();
    } catch (...) {  // NOLINT(bugprone-empty-catch)
    }
  }

  reactor_fixture(reactor_fixture const&)            = delete;
  reactor_fixture& operator=(reactor_fixture const&) = delete;

  semi_future<std::size_t> issue(void* dst, rmm::cuda_stream_view stream)
  {
    auto request = cuobj_rdma_reactor::prep_device_rx_request(
      reactor_config(), object, static_cast<std::uint8_t*>(dst), 0, k_slot_size, stream, 0);
    auto future = request->get_future();
    reactor.enqueue(std::move(request));
    return future;
  }

  void shutdown() { reactor.shutdown(); }

  std::shared_ptr<recording_session_factory> client;

 private:
  std::shared_ptr<sirius::io::rdma::mock_s3_control_client> control;
  std::shared_ptr<cuobj_rdma_reactor::reactor_context> context;
  cuobj_rdma_reactor reactor;
  cuobj_rdma_io_object object;
};

std::string resolved_error(semi_future<std::size_t> future)
{
  try {
    (void)std::move(future).get(5s);
  } catch (std::exception const& error) {
    return error.what();
  } catch (...) {
    return "non-standard exception";
  }
  return {};
}

void require_success(semi_future<std::size_t> future)
{
  REQUIRE(std::move(future).get(5s) == k_slot_size);
}

std::vector<std::uint8_t> copy_device_to_host(void const* device, rmm::cuda_stream_view stream)
{
  std::vector<std::uint8_t> bytes(k_slot_size);
  REQUIRE(
    cudaMemcpyAsync(bytes.data(), device, bytes.size(), cudaMemcpyDeviceToHost, stream.value()) ==
    cudaSuccess);
  stream.synchronize();
  return bytes;
}

enum class death_scenario {
  memcpy_error,
  memcpy_throw,
  record_error,
  record_throw,
  wait_error,
  wait_throw,
  sticky_create,
  sticky_flush,
  sticky_capture,
  hook_return,
  hook_throw,
  full_stderr_default_hook
};

cuda_delivery_ops death_ops(death_scenario scenario,
                            std::optional<cudaError_t> sticky_code = std::nullopt)
{
  cuda_delivery_ops ops;
  ops.event_destroy = [](cudaEvent_t event) {
    emit_marker(k_unwind_mark);
    return cudaEventDestroy(event);
  };

  if (scenario == death_scenario::full_stderr_default_hook) {
    // Keep the production default hook: this scenario exercises its
    // non-blocking diagnostic path against a saturated stderr pipe.
  } else if (scenario == death_scenario::hook_return) {
    ops.fatal_hook = [](auto&&...) {
      emit_marker(k_hook_mark);
      std::this_thread::sleep_for(100ms);
    };
  } else if (scenario == death_scenario::hook_throw) {
    ops.fatal_hook = [](auto&&...) -> void {
      emit_marker(k_hook_mark);
      std::this_thread::sleep_for(100ms);
      throw 17;
    };
  } else {
    ops.fatal_hook = [](auto&&...) {
      emit_marker(k_hook_mark);
      std::this_thread::sleep_for(100ms);
      std::abort();
    };
  }

  switch (scenario) {
    case death_scenario::memcpy_error:
      ops.memcpy_async = [](auto&&...) { return cudaErrorInvalidValue; };
      break;
    case death_scenario::memcpy_throw:
      ops.memcpy_async = [](auto&&...) -> cudaError_t { throw 23; };
      break;
    case death_scenario::record_error:
    case death_scenario::hook_return:
    case death_scenario::hook_throw:
      ops.event_record = [](auto&&...) { return cudaErrorInvalidValue; };
      break;
    case death_scenario::record_throw:
      ops.event_record = [](auto&&...) -> cudaError_t { throw 29; };
      break;
    case death_scenario::wait_error:
      ops.event_synchronize = [](auto&&...) { return cudaErrorInvalidValue; };
      break;
    case death_scenario::wait_throw:
      ops.event_synchronize = [](auto&&...) -> cudaError_t { throw 31; };
      break;
    case death_scenario::sticky_create: {
      auto const code  = sticky_code.value_or(cudaErrorIllegalAddress);
      ops.event_create = [code](auto&&...) { return code; };
      break;
    }
    case death_scenario::sticky_flush: {
      auto const code = sticky_code.value_or(cudaErrorAssert);
      ops.flush       = [code](auto&&...) { return code; };
      break;
    }
    case death_scenario::sticky_capture: {
      auto const code          = sticky_code.value_or(cudaErrorAssert);
      ops.stream_capture_query = [code](auto&&...) { return code; };
      break;
    }
    case death_scenario::full_stderr_default_hook:
      ops.event_record = [](auto&&...) { return cudaErrorInvalidValue; };
      break;
  }
  return ops;
}

bool child_enabled(std::string_view child_name)
{
  auto const* selected = std::getenv(k_child_env.data());
  return selected != nullptr && std::string_view{selected} == child_name;
}

std::optional<cudaError_t> selected_sticky_code()
{
  auto const* selected = std::getenv(k_sticky_env.data());
  if (selected == nullptr) { return std::nullopt; }

  char* end      = nullptr;
  errno          = 0;
  long const raw = std::strtol(selected, &end, 10);
  if (errno != 0 || end == selected || *end != '\0') {
    throw std::runtime_error("invalid SIRIUS_RDMA_STICKY_CODE");
  }
  return static_cast<cudaError_t>(raw);
}

void run_death_child(death_scenario scenario, std::optional<cudaError_t> sticky_code = std::nullopt)
{
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0 || cudaSetDevice(0) != cudaSuccess) {
    emit_marker("NO_CUDA_DEVICE\n");
    return;
  }

  int stalled_reader = -1;
  if (scenario == death_scenario::full_stderr_default_hook) {
    stalled_reader = install_full_stderr_pipe();
    // Exit without diagnostics: abort/crash handlers may themselves write to
    // the deliberately full stderr pipe and obscure whether invoke_fatal was
    // reached.  A blocking default hook still hangs before this handler.
    std::set_terminate([] { std::_Exit(134); });
  } else {
    std::set_terminate([] {
      emit_marker(k_term_mark);
      std::abort();
    });
  }

  auto ops               = death_ops(scenario, sticky_code);
  bool const needs_flush = scenario == death_scenario::sticky_flush;
  reactor_fixture fixture(std::move(ops), needs_flush, true);
  rmm::cuda_stream stream;
  rmm::device_buffer device(k_slot_size, stream);
  if (scenario == death_scenario::full_stderr_default_hook) {
    emit_marker_to(STDOUT_FILENO, k_fatal_trigger_mark);
  }
  auto future = fixture.issue(device.data(), stream);

  try {
    (void)std::move(future).get(5s);
  } catch (...) {
  }
  emit_marker(k_future_mark);
  emit_marker(k_teardown_mark);
  fixture.shutdown();
  if (stalled_reader != -1) { (void)::close(stalled_reader); }
}

struct child_result {
  bool timed_out{false};
  bool exited{false};
  int exit_code{-1};
  int signal{-1};
  std::string output;
};

std::vector<char*> child_environment(std::string const& child_name,
                                     std::optional<cudaError_t> sticky_code,
                                     std::vector<std::string>& storage)
{
  auto const prefix        = std::string{k_child_env} + "=";
  auto const sticky_prefix = std::string{k_sticky_env} + "=";
  for (char** entry = environ; entry != nullptr && *entry != nullptr; ++entry) {
    std::string_view value{*entry};
    if (!value.starts_with(prefix) && !value.starts_with(sticky_prefix)) {
      storage.emplace_back(value);
    }
  }
  storage.push_back(prefix + child_name);
  if (sticky_code) {
    storage.push_back(sticky_prefix + std::to_string(static_cast<int>(*sticky_code)));
  }

  std::vector<char*> result;
  result.reserve(storage.size() + 1);
  for (auto& value : storage) {
    result.push_back(value.data());
  }
  result.push_back(nullptr);
  return result;
}

child_result spawn_child(std::string const& child_name,
                         std::optional<cudaError_t> sticky_code = std::nullopt)
{
  int pipe_fds[2];
  if (::pipe(pipe_fds) != 0) { throw std::system_error(errno, std::generic_category(), "pipe"); }

  posix_spawn_file_actions_t actions;
  int rc                         = posix_spawn_file_actions_init(&actions);
  bool const actions_initialized = rc == 0;
  if (rc == 0) { rc = posix_spawn_file_actions_adddup2(&actions, pipe_fds[1], STDOUT_FILENO); }
  if (rc == 0) { rc = posix_spawn_file_actions_adddup2(&actions, pipe_fds[1], STDERR_FILENO); }
  if (rc == 0) { rc = posix_spawn_file_actions_addclose(&actions, pipe_fds[0]); }
  if (rc == 0) { rc = posix_spawn_file_actions_addclose(&actions, pipe_fds[1]); }
  if (rc != 0) {
    if (actions_initialized) { (void)posix_spawn_file_actions_destroy(&actions); }
    (void)::close(pipe_fds[0]);
    (void)::close(pipe_fds[1]);
    throw std::system_error(rc, std::generic_category(), "posix_spawn file actions");
  }

  std::vector<std::string> environment_storage;
  auto environment = child_environment(child_name, sticky_code, environment_storage);
  std::string executable{"/proc/self/exe"};
  std::string argv_zero{"sirius_unittest"};
  std::vector<char*> argv{argv_zero.data(), const_cast<char*>(child_name.c_str()), nullptr};

  pid_t pid = -1;
  rc = ::posix_spawn(&pid, executable.c_str(), &actions, nullptr, argv.data(), environment.data());
  (void)posix_spawn_file_actions_destroy(&actions);
  (void)::close(pipe_fds[1]);
  if (rc != 0) {
    (void)::close(pipe_fds[0]);
    throw std::system_error(rc, std::generic_category(), "posix_spawn");
  }

  child_result result;
  int read_error = 0;
  std::thread reader([&] {
    char buffer[4096];
    for (;;) {
      auto const bytes = ::read(pipe_fds[0], buffer, sizeof(buffer));
      if (bytes > 0) {
        result.output.append(buffer, static_cast<std::size_t>(bytes));
      } else if (bytes == 0) {
        break;
      } else if (errno != EINTR) {
        read_error = errno;
        break;
      }
    }
    (void)::close(pipe_fds[0]);
  });

  int status          = 0;
  auto const deadline = std::chrono::steady_clock::now() + 15s;
  for (;;) {
    auto const waited = ::waitpid(pid, &status, WNOHANG);
    if (waited == pid) { break; }
    if (waited < 0) {
      auto const wait_error = errno;
      (void)::kill(pid, SIGKILL);
      (void)::waitpid(pid, &status, 0);
      reader.join();
      throw std::system_error(wait_error, std::generic_category(), "waitpid");
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      result.timed_out = true;
      (void)::kill(pid, SIGKILL);
      (void)::waitpid(pid, &status, 0);
      break;
    }
    std::this_thread::sleep_for(10ms);
  }
  reader.join();
  if (read_error != 0) {
    throw std::system_error(read_error, std::generic_category(), "death-child output read");
  }

  result.exited    = WIFEXITED(status);
  result.exit_code = result.exited ? WEXITSTATUS(status) : -1;
  result.signal    = WIFSIGNALED(status) ? WTERMSIG(status) : -1;
  return result;
}

void require_process_fatal(std::string const& child_name,
                           bool wrapper_must_terminate            = false,
                           std::optional<cudaError_t> sticky_code = std::nullopt)
{
  if (!cuda_device_available()) { return; }

  auto const result = spawn_child(child_name, sticky_code);
  INFO("death-child output:\n" << result.output);
  REQUIRE_FALSE(result.timed_out);
  CHECK(result.output.find(k_hook_mark) != std::string::npos);
  if (wrapper_must_terminate) { CHECK(result.output.find(k_term_mark) != std::string::npos); }
  CHECK(result.output.find(k_future_mark) == std::string::npos);
  CHECK(result.output.find(k_unwind_mark) == std::string::npos);
  CHECK(result.output.find(k_teardown_mark) == std::string::npos);
  CHECK(result.output.find(k_arena_mark) == std::string::npos);
  CHECK((result.signal == SIGABRT || (result.exited && result.exit_code != 0)));
}

void require_default_hook_exits_with_full_stderr(std::string const& child_name)
{
  if (!cuda_device_available()) { return; }

  auto const result = spawn_child(child_name);
  INFO("full-stderr death-child output:\n" << result.output);
  REQUIRE_FALSE(result.timed_out);
  CHECK(result.output.find(k_full_pipe_mark) != std::string::npos);
  CHECK(result.output.find(k_fatal_trigger_mark) != std::string::npos);
  CHECK((result.signal == SIGABRT || (result.exited && result.exit_code != 0)));
}

}  // namespace s3_rdma_f01b_tests

TEST_CASE("s3_rdma memcpy error return is process-fatal", "[s3][rdma][fatal]")
{
  s3_rdma_f01b_tests::require_process_fatal("s3_rdma death child memcpy error return");
}

TEST_CASE("s3_rdma death child memcpy error return", "[.][rdma-death-child]")
{
  using namespace s3_rdma_f01b_tests;
  if (child_enabled("s3_rdma death child memcpy error return")) {
    run_death_child(death_scenario::memcpy_error);
  }
}

TEST_CASE("s3_rdma memcpy throw is process-fatal", "[s3][rdma][fatal]")
{
  s3_rdma_f01b_tests::require_process_fatal("s3_rdma death child memcpy throw");
}

TEST_CASE("s3_rdma death child memcpy throw", "[.][rdma-death-child]")
{
  using namespace s3_rdma_f01b_tests;
  if (child_enabled("s3_rdma death child memcpy throw")) {
    run_death_child(death_scenario::memcpy_throw);
  }
}

TEST_CASE("s3_rdma event record error return is process-fatal", "[s3][rdma][fatal]")
{
  s3_rdma_f01b_tests::require_process_fatal("s3_rdma death child event record error return");
}

TEST_CASE("s3_rdma death child event record error return", "[.][rdma-death-child]")
{
  using namespace s3_rdma_f01b_tests;
  if (child_enabled("s3_rdma death child event record error return")) {
    run_death_child(death_scenario::record_error);
  }
}

TEST_CASE("s3_rdma event record throw is process-fatal", "[s3][rdma][fatal]")
{
  s3_rdma_f01b_tests::require_process_fatal("s3_rdma death child event record throw");
}

TEST_CASE("s3_rdma death child event record throw", "[.][rdma-death-child]")
{
  using namespace s3_rdma_f01b_tests;
  if (child_enabled("s3_rdma death child event record throw")) {
    run_death_child(death_scenario::record_throw);
  }
}

TEST_CASE("s3_rdma event wait error return is process-fatal", "[s3][rdma][fatal]")
{
  s3_rdma_f01b_tests::require_process_fatal("s3_rdma death child event wait error return");
}

TEST_CASE("s3_rdma death child event wait error return", "[.][rdma-death-child]")
{
  using namespace s3_rdma_f01b_tests;
  if (child_enabled("s3_rdma death child event wait error return")) {
    run_death_child(death_scenario::wait_error);
  }
}

TEST_CASE("s3_rdma event wait throw is process-fatal", "[s3][rdma][fatal]")
{
  s3_rdma_f01b_tests::require_process_fatal("s3_rdma death child event wait throw");
}

TEST_CASE("s3_rdma death child event wait throw", "[.][rdma-death-child]")
{
  using namespace s3_rdma_f01b_tests;
  if (child_enabled("s3_rdma death child event wait throw")) {
    run_death_child(death_scenario::wait_throw);
  }
}

TEST_CASE("s3_rdma non-sticky flush failure recovers and continues", "[s3][rdma][fatal]")
{
  using namespace s3_rdma_f01b_tests;
  if (!cuda_device_available()) { return; }

  auto calls = std::make_shared<std::atomic<int>>(0);
  cuda_delivery_ops ops;
  ops.flush = [calls](auto&&...) {
    return calls->fetch_add(1) == 0 ? cudaErrorInvalidValue : cudaSuccess;
  };
  reactor_fixture fixture(std::move(ops), true);
  rmm::cuda_stream stream;
  rmm::device_buffer first(k_slot_size, stream);
  rmm::device_buffer follow_up(k_slot_size, stream);

  auto const error = resolved_error(fixture.issue(first.data(), stream));
  REQUIRE_FALSE(error.empty());
  REQUIRE(fixture.client->get_count() == 1);
  auto* const released_slot = fixture.client->get_destination(0);

  require_success(fixture.issue(follow_up.data(), stream));
  REQUIRE(fixture.client->get_count() == 2);
  CHECK(fixture.client->get_destination(1) == released_slot);
}

TEST_CASE("s3_rdma event create throw recovers and continues", "[s3][rdma][fatal]")
{
  using namespace s3_rdma_f01b_tests;
  if (!cuda_device_available()) { return; }

  auto calls = std::make_shared<std::atomic<int>>(0);
  cuda_delivery_ops ops;
  ops.event_create = [calls](cudaEvent_t* event, unsigned int flags) {
    if (calls->fetch_add(1) == 0) { throw std::runtime_error("injected event-create failure"); }
    return cudaEventCreateWithFlags(event, flags);
  };
  reactor_fixture fixture(std::move(ops));
  rmm::cuda_stream stream;
  rmm::device_buffer first(k_slot_size, stream);
  rmm::device_buffer follow_up(k_slot_size, stream);

  auto const error = resolved_error(fixture.issue(first.data(), stream));
  REQUIRE_FALSE(error.empty());
  REQUIRE(fixture.client->get_count() == 1);
  auto* const released_slot = fixture.client->get_destination(0);

  require_success(fixture.issue(follow_up.data(), stream));
  REQUIRE(fixture.client->get_count() == 2);
  CHECK(fixture.client->get_destination(1) == released_slot);
}

TEST_CASE("s3_rdma captured-stream rejection recovers before the GET", "[s3][rdma][fatal]")
{
  using namespace s3_rdma_f01b_tests;
  if (!cuda_device_available()) { return; }

  auto calls = std::make_shared<std::atomic<int>>(0);
  cuda_delivery_ops ops;
  ops.stream_capture_query = [calls](cudaStream_t, cudaStreamCaptureStatus* status) {
    *status =
      calls->fetch_add(1) == 0 ? cudaStreamCaptureStatusActive : cudaStreamCaptureStatusNone;
    return cudaSuccess;
  };
  reactor_fixture fixture(std::move(ops));
  rmm::cuda_stream stream;
  rmm::device_buffer first(k_slot_size, stream);
  rmm::device_buffer follow_up(k_slot_size, stream);

  auto const error = resolved_error(fixture.issue(first.data(), stream));
  REQUIRE_FALSE(error.empty());
  CHECK(fixture.client->get_count() == 0);

  require_success(fixture.issue(follow_up.data(), stream));
  CHECK(fixture.client->get_count() == 1);
}

TEST_CASE("s3_rdma sticky event create code is process-fatal", "[s3][rdma][fatal]")
{
  s3_rdma_f01b_tests::require_process_fatal("s3_rdma death child sticky event create");
}

TEST_CASE("s3_rdma death child sticky event create", "[.][rdma-death-child]")
{
  using namespace s3_rdma_f01b_tests;
  if (child_enabled("s3_rdma death child sticky event create")) {
    run_death_child(death_scenario::sticky_create, selected_sticky_code());
  }
}

TEST_CASE("s3_rdma sticky flush code is process-fatal", "[s3][rdma][fatal]")
{
  s3_rdma_f01b_tests::require_process_fatal("s3_rdma death child sticky flush");
}

TEST_CASE("s3_rdma death child sticky flush", "[.][rdma-death-child]")
{
  using namespace s3_rdma_f01b_tests;
  if (child_enabled("s3_rdma death child sticky flush")) {
    run_death_child(death_scenario::sticky_flush, selected_sticky_code());
  }
}

TEST_CASE("s3_rdma sticky capture query code is process-fatal", "[s3][rdma][fatal]")
{
  s3_rdma_f01b_tests::require_process_fatal("s3_rdma death child sticky capture query");
}

TEST_CASE("s3_rdma death child sticky capture query", "[.][rdma-death-child]")
{
  using namespace s3_rdma_f01b_tests;
  if (child_enabled("s3_rdma death child sticky capture query")) {
    run_death_child(death_scenario::sticky_capture, selected_sticky_code());
  }
}

TEST_CASE("s3_rdma fatal hook returning terminates", "[s3][rdma][fatal]")
{
  s3_rdma_f01b_tests::require_process_fatal("s3_rdma death child fatal hook returns", true);
}

TEST_CASE("s3_rdma death child fatal hook returns", "[.][rdma-death-child]")
{
  using namespace s3_rdma_f01b_tests;
  if (child_enabled("s3_rdma death child fatal hook returns")) {
    run_death_child(death_scenario::hook_return);
  }
}

TEST_CASE("s3_rdma fatal hook throwing terminates", "[s3][rdma][fatal]")
{
  s3_rdma_f01b_tests::require_process_fatal("s3_rdma death child fatal hook throws", true);
}

TEST_CASE("s3_rdma death child fatal hook throws", "[.][rdma-death-child]")
{
  using namespace s3_rdma_f01b_tests;
  if (child_enabled("s3_rdma death child fatal hook throws")) {
    run_death_child(death_scenario::hook_throw);
  }
}

TEST_CASE("s3_rdma event destroy failure after successful wait is log-only", "[s3][rdma][fatal]")
{
  using namespace s3_rdma_f01b_tests;
  if (!cuda_device_available()) { return; }

  auto destroy_calls = std::make_shared<std::atomic<int>>(0);
  cuda_delivery_ops ops;
  ops.event_destroy = [destroy_calls](cudaEvent_t event) {
    destroy_calls->fetch_add(1);
    auto const result = cudaEventDestroy(event);
    return result == cudaSuccess ? cudaErrorUnknown : result;
  };
  reactor_fixture fixture(std::move(ops));
  rmm::cuda_stream stream;
  rmm::device_buffer device(k_slot_size, stream);

  require_success(fixture.issue(device.data(), stream));
  CHECK(destroy_calls->load() == 1);
  CHECK(copy_device_to_host(device.data(), stream) == payload_bytes());
}

TEST_CASE("s3_rdma event destroy failure remains successful when log sink throws",
          "[s3][rdma][fatal]")
{
  using namespace s3_rdma_f01b_tests;
  if (!cuda_device_available()) { return; }

  auto destroy_calls = std::make_shared<std::atomic<int>>(0);
  cuda_delivery_ops ops;
  ops.event_destroy = [destroy_calls](cudaEvent_t event) {
    destroy_calls->fetch_add(1);
    auto const result = cudaEventDestroy(event);
    return result == cudaSuccess ? cudaErrorUnknown : result;
  };
  reactor_fixture fixture(std::move(ops));
  rmm::cuda_stream stream;
  rmm::device_buffer device(k_slot_size, stream);
  auto throwing_sink = std::make_shared<throwing_log_sink>();

  {
    scoped_log_sink installed(throwing_sink);
    require_success(fixture.issue(device.data(), stream));
  }

  CHECK(destroy_calls->load() == 1);
  CHECK(throwing_sink->calls() == 1);
  CHECK(copy_device_to_host(device.data(), stream) == payload_bytes());
}

TEST_CASE("s3_rdma default fatal hook exits when stderr pipe is full", "[s3][rdma][fatal]")
{
  s3_rdma_f01b_tests::require_default_hook_exits_with_full_stderr(
    "s3_rdma death child default fatal hook with full stderr pipe");
}

TEST_CASE("s3_rdma death child default fatal hook with full stderr pipe", "[.][rdma-death-child]")
{
  using namespace s3_rdma_f01b_tests;
  if (child_enabled("s3_rdma death child default fatal hook with full stderr pipe")) {
    run_death_child(death_scenario::full_stderr_default_hook);
  }
}

TEST_CASE("s3_rdma every frozen sticky code is process-fatal", "[s3][rdma][fatal]")
{
  using namespace s3_rdma_f01b_tests;
  using sirius::io::rdma::k_sticky_context_fatal_codes;
  if (!cuda_device_available()) { return; }

  REQUIRE(k_sticky_context_fatal_codes.size() == k_expected_sticky_codes.size());
  REQUIRE(std::equal(k_sticky_context_fatal_codes.begin(),
                     k_sticky_context_fatal_codes.end(),
                     k_expected_sticky_codes.begin(),
                     k_expected_sticky_codes.end()));

  constexpr std::array<std::string_view, 3> children = {
    "s3_rdma death child sticky event create",
    "s3_rdma death child sticky flush",
    "s3_rdma death child sticky capture query",
  };
  for (std::size_t i = 0; i < k_sticky_context_fatal_codes.size(); ++i) {
    auto const code = k_sticky_context_fatal_codes[i];
    INFO("sticky code " << static_cast<int>(code) << " via leg " << (i % children.size()));
    require_process_fatal(std::string{children[i % children.size()]}, false, code);
  }
}
