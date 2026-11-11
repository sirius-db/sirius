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

#include <sched.h>
#include <unistd.h>

namespace sirius::util {

/**
 * @brief For its lifetime, gives the calling thread the CPU set of the
 * process's main thread, then restores the thread's own.
 *
 * A new thread inherits its creator's CPU set. DuckDB pins each of its worker
 * threads to one CPU (64 threads or more, as many CPUs), and `LOAD sirius` can
 * run on such a worker; every thread Sirius then starts without an explicit
 * affinity would share that one CPU. The main thread is never pinned by
 * DuckDB, so its set is the one the process was started with (all CPUs, or
 * what `taskset` / the scheduler allowed).
 */
class scoped_process_affinity {
 public:
  scoped_process_affinity() noexcept
  {
    cpu_set_t process;
    if (sched_getaffinity(0, sizeof(_saved), &_saved) != 0 ||
        sched_getaffinity(getpid(), sizeof(process), &process) != 0 ||
        CPU_EQUAL(&_saved, &process)) {
      return;
    }
    _restore = sched_setaffinity(0, sizeof(process), &process) == 0;
  }
  ~scoped_process_affinity()
  {
    if (_restore) { sched_setaffinity(0, sizeof(_saved), &_saved); }
  }
  scoped_process_affinity(scoped_process_affinity const&)            = delete;
  scoped_process_affinity& operator=(scoped_process_affinity const&) = delete;

  /// Whether the calling thread's set was narrower and has been widened.
  [[nodiscard]] bool widened() const noexcept { return _restore; }

 private:
  cpu_set_t _saved;
  bool _restore{false};
};

}  // namespace sirius::util
