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

#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace sirius::test::rest_redirect {

template <typename Config>
constexpr bool has_request_log = requires(Config& cfg) { cfg.request_log; };

template <typename Config>
void set_request_log(Config& cfg, bool enabled)
{
  if constexpr (has_request_log<Config>) { cfg.request_log = enabled; }
}

template <typename Config>
bool request_log_enabled(Config const& cfg)
{
  if constexpr (has_request_log<Config>) { return cfg.request_log; }
  return false;
}

template <typename Config>
constexpr bool has_retention = requires(Config& cfg) { cfg.cluster_redirect.retention; };

template <typename Config>
void set_retention(Config& cfg, bool expanded)
{
  if constexpr (has_retention<Config>) {
    using mode                     = std::remove_cvref_t<decltype(cfg.cluster_redirect.retention)>;
    cfg.cluster_redirect.retention = expanded ? mode::expanded : mode::legacy;
  }
}

template <typename Config>
constexpr bool has_route_reuse = requires(Config& cfg) { cfg.cluster_redirect.route_reuse; };

template <typename Config>
void set_route_reuse(Config& cfg, bool enabled)
{
  if constexpr (has_route_reuse<Config>) {
    using mode = std::remove_cvref_t<decltype(cfg.cluster_redirect.route_reuse)>;
    cfg.cluster_redirect.route_reuse = enabled ? mode::same_open : mode::off;
  }
}

template <typename Counters>
constexpr bool has_route_counters = requires(Counters const& counts) {
  counts.route_seeds;
  counts.route_hits;
  counts.route_refreshes;
};

struct route_counts {
  std::uint64_t seeds{0};
  std::uint64_t hits{0};
  std::uint64_t refreshes{0};
};

template <typename Reactor>
route_counts routes(Reactor const& reactor)
{
  auto const counts = reactor.redirects();
  if constexpr (has_route_counters<decltype(counts)>) {
    return {counts.route_seeds, counts.route_hits, counts.route_refreshes};
  }
  return {};
}

// Run the existing transport regressions in both instrumentation modes.
template <typename Config>
void request_log_from_test_env(Config& cfg)
{
  auto const* mode = std::getenv("SIRIUS_TEST_REST_REQUEST_LOG");
  if (mode != nullptr) { set_request_log(cfg, std::string_view{mode} == "1"); }
}

template <typename Config>
constexpr bool has_policy = requires(Config& cfg) {
  cfg.cluster_redirect.enabled;
  cfg.cluster_redirect.origin;
  cfg.cluster_redirect.allowed_targets;
};

// The baseline must exercise its real never-follow path, not a test implementation of delegation.
template <typename Config>
void set_policy(Config& cfg, bool enabled, std::string origin, std::vector<std::string> targets)
{
  if constexpr (has_policy<Config>) {
    cfg.cluster_redirect.enabled         = enabled;
    cfg.cluster_redirect.origin          = std::move(origin);
    cfg.cluster_redirect.allowed_targets = std::move(targets);
  }
}

template <typename Config>
void require_policy(Config const& cfg,
                    bool enabled,
                    std::string const& origin,
                    std::vector<std::string> const& targets)
{
  REQUIRE(has_policy<Config>);
  if constexpr (has_policy<Config>) {
    CHECK(cfg.cluster_redirect.enabled == enabled);
    CHECK(cfg.cluster_redirect.origin == origin);
    CHECK(cfg.cluster_redirect.allowed_targets == targets);
  }
}

template <typename Reactor>
void require_counters(Reactor const& reactor, std::uint64_t followed, std::uint64_t refused)
{
  constexpr bool available = requires { reactor.redirects(); };
  REQUIRE(available);
  if constexpr (available) {
    auto const counts = reactor.redirects();
    CHECK(counts.followed == followed);
    CHECK(counts.refused == refused);
  }
}

class loopback_relay {
 public:
  loopback_relay(std::filesystem::path const& script, std::string mode, std::uint16_t upstream)
    : _mode(std::move(mode))
  {
    auto pattern = (std::filesystem::temp_directory_path() / "sirius-rest-relay-XXXXXX").string();
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    if (::mkdtemp(name.data()) == nullptr) { throw std::runtime_error("relay mkdtemp failed"); }
    _directory               = name.data();
    auto const script_name   = script.string();
    auto const upstream_port = std::to_string(upstream);
    auto const directory     = _directory.string();
    auto const parent_pid    = ::getpid();
    try {
      _pid = ::fork();
      if (_pid < 0) { throw std::runtime_error("relay fork failed"); }
      if (_pid == 0) {
        // Exec before doing fixture or CUDA work in the child.
        if (::setpgid(0, 0) != 0 || ::prctl(PR_SET_PDEATHSIG, SIGTERM) != 0) { ::_exit(126); }
        if (::getppid() != parent_pid) { ::_exit(126); }
        ::execlp("python",
                 "python",
                 "-u",
                 script_name.c_str(),
                 _mode.c_str(),
                 upstream_port.c_str(),
                 directory.c_str(),
                 static_cast<char*>(nullptr));
        ::_exit(127);
      }
      (void)::setpgid(_pid, _pid);
      auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{15};
      while (std::chrono::steady_clock::now() < deadline) {
        std::ifstream ready(_directory / "ready");
        unsigned port = 0;
        if (ready >> port && port > 0 && port <= 65535) {
          _port = static_cast<std::uint16_t>(port);
          return;
        }
        int status = 0;
        if (::waitpid(_pid, &status, WNOHANG) == _pid) {
          _pid = -1;
          throw std::runtime_error("relay exited before readiness");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{20});
      }
      throw std::runtime_error("relay readiness timeout");
    } catch (...) {
      close();
      throw;
    }
  }

  ~loopback_relay() { close(); }
  loopback_relay(loopback_relay const&)            = delete;
  loopback_relay& operator=(loopback_relay const&) = delete;

  std::string endpoint() const
  {
    return (_mode == "tls" ? "https://127.0.0.1:" : "http://127.0.0.1:") + std::to_string(_port);
  }
  std::uint16_t port() const { return _port; }

  std::string events() const
  {
    std::ifstream input(_directory / "events");
    return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
  }

 private:
  void close() noexcept
  {
    if (_pid > 0) {
      (void)::kill(-_pid, SIGTERM);
      int status = 0;
      for (int attempt = 0; attempt < 100; ++attempt) {
        auto const waited = ::waitpid(_pid, &status, WNOHANG);
        if (waited == _pid || (waited < 0 && errno == ECHILD)) {
          _pid = -1;
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
      }
      if (_pid > 0) {
        (void)::kill(-_pid, SIGKILL);
        (void)::waitpid(_pid, &status, 0);
        _pid = -1;
      }
    }
    std::error_code error;
    std::filesystem::remove_all(_directory, error);
  }

  std::string _mode;
  std::filesystem::path _directory;
  pid_t _pid{-1};
  std::uint16_t _port{0};
};

}  // namespace sirius::test::rest_redirect
