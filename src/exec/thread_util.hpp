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

#include <cucascade/exec/thread_util.hpp>
#include <sched.h>
#include <unistd.h>

#include <exception>
#include <span>
#include <thread>
#include <tuple>
#include <vector>

namespace sirius::exec {

namespace thread_util = cucascade::exec::thread_util;

/// Run @p work to completion on a new thread whose affinity is @p cpus, then rethrow what it
/// threw.  Threads @p work starts inherit @p cpus; the caller's own mask is untouched.  Lets a
/// thread restricted to a subset of CPUs build something that must plan from a wider mask.
/// Should the mask not apply, @p work still runs, from the caller's mask.
template <typename F>
void run_with_affinity(std::span<int const> cpus, F&& work)
{
  std::exception_ptr error;
  std::thread runner([&] {
    std::ignore = thread_util::set_current_thread_affinity(cpus);
    try {
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
