/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */
#pragma once
#include <catch.hpp>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utils/child_process_environment.hpp>
#include <utils/sirius_test_env.hpp>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
namespace sirius::test {
using namespace std::chrono_literals;
struct child_result {
  bool timed_out = false;
  int exit_code  = -1;
  int signal     = -1;
  std::string output;
};

inline child_result run_test_child(std::string const& test_name, std::string const& variant = {})
{
  // The child owns its own runtime. Release the parent's idle GPU pools first.
  // There is no live query/fixture when this helper is entered.
  struct paused_environments {
    std::vector<shared_test_env*> active;
    paused_environments()
    {
      for (auto* env : {g_shared_env, g_integration_env, g_integration_env_2gpu}) {
        if (env && env->is_active()) {
          active.push_back(env);
          env->pause();
        }
      }
    }
    ~paused_environments()
    {
      for (auto* env : active) {
        env->resume();
      }
    }
  } paused;
  namespace fs = std::filesystem;
  static std::atomic<unsigned> next_id{0};
  auto const output_path =
    fs::temp_directory_path() / ("sirius_native_lease_child_" + std::to_string(::getpid()) + "_" +
                                 std::to_string(next_id.fetch_add(1)) + ".log");
  auto const fd = ::open(output_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
  REQUIRE(fd >= 0);

  std::vector<std::string> arguments{"sirius_unittest", test_name, "--reporter", "compact"};
  std::vector<char*> argv;
  for (auto& argument : arguments) {
    argv.push_back(argument.data());
  }
  argv.push_back(nullptr);
  sirius::test::child_process_environment environment(
    {{"SIRIUS_NATIVE_LEASE_UNAVAILABLE_VARIANT", variant},
     {"SIRIUS_NATIVE_LEASE_CHILD_CASE", test_name}});

  posix_spawn_file_actions_t actions;
  REQUIRE(::posix_spawn_file_actions_init(&actions) == 0);
  REQUIRE(::posix_spawn_file_actions_adddup2(&actions, fd, STDOUT_FILENO) == 0);
  REQUIRE(::posix_spawn_file_actions_adddup2(&actions, fd, STDERR_FILENO) == 0);
  REQUIRE(::posix_spawn_file_actions_addclose(&actions, fd) == 0);
  pid_t pid{};
  auto const spawn_result =
    ::posix_spawn(&pid, "/proc/self/exe", &actions, nullptr, argv.data(), environment.data());
  (void)::posix_spawn_file_actions_destroy(&actions);
  (void)::close(fd);
  REQUIRE(spawn_result == 0);

  int status      = 0;
  bool timed_out  = true;
  bool wait_error = false;
  auto const stop = std::chrono::steady_clock::now() + 90s;
  while (std::chrono::steady_clock::now() < stop) {
    auto const waited = ::waitpid(pid, &status, WNOHANG);
    if (waited == pid) {
      timed_out = false;
      break;
    }
    if (waited < 0 && errno != EINTR) {
      timed_out  = false;
      wait_error = true;
      break;
    }
    std::this_thread::sleep_for(50ms);
  }
  if (timed_out) {
    (void)::kill(pid, SIGKILL);
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
  }

  std::ifstream input(output_path);
  std::ostringstream output;
  output << input.rdbuf();
  std::error_code remove_error;
  fs::remove(output_path, remove_error);
  return {timed_out,
          !timed_out && !wait_error && WIFEXITED(status) ? WEXITSTATUS(status) : -1,
          !timed_out && !wait_error && WIFSIGNALED(status) ? WTERMSIG(status) : -1,
          output.str()};
}

// Concurrency regressions must fail within a bounded time even when std::future destruction joins
// a deadlocked worker. The parent kills/reaps the whole isolated child after the deadline.
inline bool run_isolated()
{
  auto const name   = Catch::getResultCapture().getCurrentTestName();
  auto const* child = std::getenv("SIRIUS_NATIVE_LEASE_CHILD_CASE");
  if (child && name == child) { return false; }
  auto result = run_test_child(name);
  INFO(result.output);
  REQUIRE_FALSE(result.timed_out);
  REQUIRE(result.signal == -1);
  REQUIRE(result.exit_code == 0);
  return true;
}

}  // namespace sirius::test
