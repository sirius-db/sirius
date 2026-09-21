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

#include <pthread.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

/**
 * @file thread_util.hpp
 * @brief Wrappers for the OS-level knobs on a running thread: name, affinity,
 *        scheduling policy, and niceness.
 *
 * The point of a wrapper here is not to hide the syscall but to spare every
 * caller the same three lines --- truncate the name because pthread caps it at
 * 16 bytes, populate a @c cpu_set_t from a list of ids, hand-check the return
 * code --- and to give every subsystem the same story about what happens on
 * failure.  Every entry point is @c noexcept and returns @c false on failure:
 * these knobs are best-effort telemetry / tuning, never load-bearing, and a
 * subsystem that decides otherwise can still test the return.
 *
 * The pthread-handle-taking overloads are the primitive; the @c std::thread
 * and @c std::jthread overloads forward to them, and the "current thread"
 * overloads read the current handle from @c pthread_self.
 */

namespace sirius::exec::thread_util {

/// Linux caps thread names at 16 bytes including the terminator.  Callers get
/// truncation, not rejection: a name that is too long is silently kept by
/// @c pthread_setname_np only up to this length, so we do the truncation up
/// front and always hand the syscall something it will accept.
inline constexpr std::size_t max_thread_name_len = 15;

// ---------------------------------------------------------------------------
// thread name
// ---------------------------------------------------------------------------

/// Set the name of @p handle.  Names longer than @ref max_thread_name_len are
/// truncated.  Returns @c false when @c pthread_setname_np refuses the name
/// (only when it is @c nullptr in practice, since we truncate before calling).
[[nodiscard]] bool set_thread_name(pthread_t handle, std::string_view name) noexcept;

/// Set the name of @p t.  Undefined if @p t is not @c joinable().
[[nodiscard]] bool set_thread_name(std::thread& t, std::string_view name) noexcept;
[[nodiscard]] bool set_thread_name(std::jthread& t, std::string_view name) noexcept;

/// Set the name of the calling thread.
[[nodiscard]] bool set_current_thread_name(std::string_view name) noexcept;

/// Read the name of @p handle.  Returns an empty string on failure.
[[nodiscard]] std::string get_thread_name(pthread_t handle) noexcept;
[[nodiscard]] std::string get_current_thread_name() noexcept;

// ---------------------------------------------------------------------------
// CPU affinity
// ---------------------------------------------------------------------------

/// Pin @p handle to the CPUs named in @p cpu_ids.  An empty list is a no-op
/// (rather than "any CPU"); the caller who really wants "any" should compute
/// the CPU list themselves.  Ids are the same as @c sched_setaffinity's ---
/// zero-based logical CPU numbers.  Ids outside @c CPU_SETSIZE are silently
/// skipped.  Returns @c false on syscall failure.
[[nodiscard]] bool set_thread_affinity(pthread_t handle, std::span<int const> cpu_ids) noexcept;
[[nodiscard]] bool set_thread_affinity(std::thread& t, std::span<int const> cpu_ids) noexcept;
[[nodiscard]] bool set_thread_affinity(std::jthread& t, std::span<int const> cpu_ids) noexcept;
[[nodiscard]] bool set_current_thread_affinity(std::span<int const> cpu_ids) noexcept;

/// Read the affinity mask of @p handle as the list of allowed CPU ids.
/// Returns an empty vector on failure.
[[nodiscard]] std::vector<int> get_thread_affinity(pthread_t handle) noexcept;
[[nodiscard]] std::vector<int> get_current_thread_affinity() noexcept;

// ---------------------------------------------------------------------------
// scheduling policy and priority
// ---------------------------------------------------------------------------

/// The kernel scheduler classes we expose.  Named apart from the raw
/// @c SCHED_* macros so callers do not have to include @c <sched.h> to spell a
/// policy, and so the enumerators are namespaced.
enum class scheduling_policy : std::uint8_t {
  /// SCHED_OTHER -- the default time-sharing class.  Priority is always 0;
  /// per-thread weight is set through niceness instead.
  other,
  /// SCHED_BATCH -- like other, but the scheduler assumes the thread is
  /// CPU-bound and skips latency-focused wake-up boosting.
  batch,
  /// SCHED_IDLE -- the very-low-priority background class.
  idle,
  /// SCHED_FIFO -- real-time, non-preemptive within a priority level.
  /// Requires @c CAP_SYS_NICE.  Priority is 1..99.
  fifo,
  /// SCHED_RR -- real-time, round-robin within a priority level.
  /// Requires @c CAP_SYS_NICE.  Priority is 1..99.
  round_robin,
};

/// Set the scheduling policy and priority for @p handle.  For the
/// time-sharing policies (@c other / @c batch / @c idle) @p priority must be
/// @c 0; for the real-time policies it must be in @c [1, 99].  Returns
/// @c false on syscall failure --- most commonly @c EPERM when a real-time
/// policy is requested without @c CAP_SYS_NICE.
[[nodiscard]] bool set_thread_scheduling(pthread_t handle,
                                         scheduling_policy policy,
                                         int priority) noexcept;
[[nodiscard]] bool set_thread_scheduling(std::thread& t,
                                         scheduling_policy policy,
                                         int priority) noexcept;
[[nodiscard]] bool set_thread_scheduling(std::jthread& t,
                                         scheduling_policy policy,
                                         int priority) noexcept;
[[nodiscard]] bool set_current_thread_scheduling(scheduling_policy policy, int priority) noexcept;

/// One reading of a thread's current scheduling state.
struct scheduling_info {
  scheduling_policy policy{scheduling_policy::other};
  int priority{0};
};

/// Read the scheduling policy and priority of @p handle.  Returns
/// @c std::nullopt on syscall failure.
[[nodiscard]] std::optional<scheduling_info> get_thread_scheduling(pthread_t handle) noexcept;
[[nodiscard]] std::optional<scheduling_info> get_current_thread_scheduling() noexcept;

// ---------------------------------------------------------------------------
// niceness (SCHED_OTHER weight)
// ---------------------------------------------------------------------------

/// Set the calling thread's nice value in @c [-20, 19].  Lower is more
/// aggressive; raising it above 0 does not require privileges, lowering it
/// below 0 requires @c CAP_SYS_NICE.  Only meaningful under
/// @c scheduling_policy::other or @c batch.  Returns @c false on failure.
///
/// Only the current-thread overload is offered: @c setpriority() takes a
/// thread id (obtained from @c gettid()), and there is no portable way to
/// derive one from a @c pthread_t without asking the thread itself.
[[nodiscard]] bool set_current_thread_niceness(int nice_value) noexcept;

/// Read the calling thread's nice value.  Returns @c std::nullopt on failure.
[[nodiscard]] std::optional<int> get_current_thread_niceness() noexcept;

// ---------------------------------------------------------------------------
// identification
// ---------------------------------------------------------------------------

/// The kernel's TID for the calling thread (@c gettid).  Useful as a stable
/// key in logs and in per-thread telemetry, and as the "who" for the syscalls
/// that address a thread by TID rather than @c pthread_t.
[[nodiscard]] std::uint64_t current_thread_id() noexcept;

}  // namespace sirius::exec::thread_util
