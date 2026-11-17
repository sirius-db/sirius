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

#include <exception>
#include <span>
#include <thread>
#include <tuple>

namespace sirius::exec {

namespace thread_util = cucascade::exec::thread_util;

/// Run @p work to completion on a new thread whose affinity is @p cpus, then rethrow what it
/// threw.  Threads @p work starts inherit @p cpus; the caller's own mask is untouched.  Lets a
/// thread restricted to a subset of CPUs build something that must plan from a wider mask.
/// Throws before user work if the requested nonempty mask cannot be applied exactly.
template <typename F>
void run_with_affinity(std::span<int const> cpus, F&& work)
{
  std::exception_ptr error;
  std::thread runner([&] {
    try {
      thread_util::scoped_thread_affinity affinity(cpus);
      work();
    } catch (...) {
      error = std::current_exception();
    }
  });
  runner.join();
  if (error) { std::rethrow_exception(error); }
}

}  // namespace sirius::exec
