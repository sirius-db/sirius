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

#include "log/logging.hpp"

#include <cucascade/exec/thread_util.hpp>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace sirius::exec {

namespace thread_util = cucascade::exec::thread_util;

/// Lazy startup uses the saved boundary, even from a caller pinned to one CPU.
/// The kernel may narrow it after a deployment cpuset change. Placement discovery
/// and affinity operations remain in cuCascade; this is the startup acceptance policy.
class scoped_startup_affinity {
 public:
  explicit scoped_startup_affinity(std::span<int const> cpus)
    : restore_(cpus.empty() ? std::vector<int>{} : read_affinity())
  {
    if (cpus.empty()) return;
    for (int cpu : cpus)
      if (cpu < 0 || cpu >= CPU_SETSIZE)
        throw std::invalid_argument("startup CPU id is outside CPU_SETSIZE");
    if (!thread_util::set_current_thread_affinity(cpus))
      throw std::runtime_error("startup CPU boundary has no kernel-granted subset");
    effective_ = read_affinity();
    for (int cpu : effective_)
      if (std::find(cpus.begin(), cpus.end(), cpu) == cpus.end())
        throw std::runtime_error("startup affinity escaped its saved boundary");
    if (std::any_of(cpus.begin(), cpus.end(), [&](int cpu) {
          return std::find(effective_.begin(), effective_.end(), cpu) == effective_.end();
        })) {
      log::log_noexcept([&] {
        std::string granted;
        for (int cpu : effective_) {
          if (!granted.empty()) granted += ',';
          granted += std::to_string(cpu);
        }
        SIRIUS_LOG_WARN("startup CPU boundary narrowed by kernel; granted CPUs {}", granted);
      });
    }
  }
  std::vector<int> const& effective_cpus() const noexcept { return effective_; }

 private:
  static std::vector<int> read_affinity()
  {
    auto cpus = thread_util::get_current_thread_affinity();
    if (cpus.empty()) throw std::runtime_error("cannot read nonempty startup CPU affinity");
    return cpus;
  }
  // Construct first: also restores the caller if applying or reporting the mask throws.
  thread_util::scoped_thread_affinity restore_;
  std::vector<int> effective_;
};

/// Run @p work to completion on a new thread whose affinity is @p cpus, then rethrow what it
/// threw.  Threads @p work starts inherit @p cpus; the caller's own mask is untouched.  Lets a
/// thread restricted to a subset of CPUs build something that must plan from a wider mask.
/// Accepts a nonempty kernel-granted subset of the saved mask; rejects an empty subset.
template <typename F>
void run_with_affinity(std::span<int const> cpus, F&& work)
{
  std::exception_ptr error;
  std::thread runner([&] {
    try {
      scoped_startup_affinity affinity(cpus);
      work();
    } catch (...) {
      error = std::current_exception();
    }
  });
  runner.join();
  if (error) { std::rethrow_exception(error); }
}

/// The CPUs the process may run on: the affinity of its main thread, which is what taskset,
/// numactl and a container's cpuset set. Empty when it cannot be read.
[[nodiscard]] inline std::vector<int> process_affinity() noexcept
{
  cpu_set_t mask{};
  if (::sched_getaffinity(::getpid(), sizeof(mask), &mask) != 0) { return {}; }
  std::vector<int> ids;
  try {
    for (int i = 0; i < CPU_SETSIZE; ++i) {
      if (CPU_ISSET(static_cast<std::size_t>(i), &mask)) { ids.push_back(i); }
    }
  } catch (...) {
    return {};
  }
  return ids;
}

/// Runs the calling thread with the process's affinity for the guard's lifetime, then restores
/// its own. An embedder may call into Sirius from a thread it pinned to one core (DuckDB pins
/// its workers on hosts with more than 64 cores); everything planned from, or started on, such
/// a thread would otherwise inherit that single core.
class scoped_process_affinity {
 public:
  scoped_process_affinity() : _saved(thread_util::get_current_thread_affinity())
  {
    auto const wide = process_affinity();
    if (_saved.empty() || wide.empty() || wide == _saved ||
        !thread_util::set_current_thread_affinity(wide)) {
      _saved.clear();
    }
  }
  ~scoped_process_affinity()
  {
    if (!_saved.empty()) { std::ignore = thread_util::set_current_thread_affinity(_saved); }
  }
  scoped_process_affinity(scoped_process_affinity const&)            = delete;
  scoped_process_affinity& operator=(scoped_process_affinity const&) = delete;

 private:
  std::vector<int> _saved;
};

}  // namespace sirius::exec
