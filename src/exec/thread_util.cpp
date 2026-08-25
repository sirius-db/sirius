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

#include "exec/thread_util.hpp"

#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>

namespace sirius::exec::thread_util {

namespace {

/// Translate one of our policy enumerators into the kernel @c SCHED_* macro.
[[nodiscard]] int to_native(scheduling_policy p) noexcept
{
  switch (p) {
    case scheduling_policy::other: return SCHED_OTHER;
    case scheduling_policy::batch: return SCHED_BATCH;
    case scheduling_policy::idle: return SCHED_IDLE;
    case scheduling_policy::fifo: return SCHED_FIFO;
    case scheduling_policy::round_robin: return SCHED_RR;
  }
  // Unreachable if the enum is exhaustive; keep SCHED_OTHER as the fallback
  // rather than passing an out-of-range value to the syscall.
  return SCHED_OTHER;
}

/// Inverse of @ref to_native.  Returns @c std::nullopt for a policy we do not
/// expose (e.g. @c SCHED_DEADLINE), so the caller sees "unknown" rather than
/// a misleading match.
[[nodiscard]] std::optional<scheduling_policy> from_native(int p) noexcept
{
  switch (p) {
    case SCHED_OTHER: return scheduling_policy::other;
    case SCHED_BATCH: return scheduling_policy::batch;
    case SCHED_IDLE: return scheduling_policy::idle;
    case SCHED_FIFO: return scheduling_policy::fifo;
    case SCHED_RR: return scheduling_policy::round_robin;
    default: return std::nullopt;
  }
}

/// Copy @p name into a NUL-terminated buffer of @c max_thread_name_len+1
/// bytes.  Truncation is silent: the kernel would refuse a name past 15 bytes,
/// so we always hand it something short enough.
[[nodiscard]] std::array<char, max_thread_name_len + 1> truncated_name(
  std::string_view name) noexcept
{
  std::array<char, max_thread_name_len + 1> buf{};
  auto const n = std::min(name.size(), max_thread_name_len);
  std::memcpy(buf.data(), name.data(), n);
  buf[n] = '\0';
  return buf;
}

}  // namespace

// ---------------------------------------------------------------------------
// thread name
// ---------------------------------------------------------------------------

bool set_thread_name(pthread_t handle, std::string_view name) noexcept
{
  auto const buf = truncated_name(name);
  return ::pthread_setname_np(handle, buf.data()) == 0;
}

bool set_thread_name(std::thread& t, std::string_view name) noexcept
{
  return set_thread_name(t.native_handle(), name);
}

bool set_thread_name(std::jthread& t, std::string_view name) noexcept
{
  return set_thread_name(t.native_handle(), name);
}

bool set_current_thread_name(std::string_view name) noexcept
{
  return set_thread_name(::pthread_self(), name);
}

std::string get_thread_name(pthread_t handle) noexcept
{
  // 16-byte buffer is what the kernel writes.  Any longer wastes the read;
  // any shorter and a name that hit the limit would be silently truncated.
  std::array<char, max_thread_name_len + 1> buf{};
  if (::pthread_getname_np(handle, buf.data(), buf.size()) != 0) { return {}; }
  return std::string{buf.data()};
}

std::string get_current_thread_name() noexcept { return get_thread_name(::pthread_self()); }

// ---------------------------------------------------------------------------
// affinity
// ---------------------------------------------------------------------------

namespace {

/// Populate @p out from @p cpu_ids, silently dropping any id past
/// @c CPU_SETSIZE.  Returns the count actually installed --- 0 means the
/// caller passed nothing usable and the syscall should not be attempted, or
/// it would pin the thread to an empty mask.
[[nodiscard]] std::size_t build_cpuset(std::span<int const> cpu_ids, cpu_set_t& out) noexcept
{
  CPU_ZERO(&out);
  std::size_t installed = 0;
  for (int id : cpu_ids) {
    if (id < 0 || id >= CPU_SETSIZE) { continue; }
    CPU_SET(id, &out);
    ++installed;
  }
  return installed;
}

}  // namespace

bool set_thread_affinity(pthread_t handle, std::span<int const> cpu_ids) noexcept
{
  if (cpu_ids.empty()) { return true; }
  cpu_set_t mask{};
  if (build_cpuset(cpu_ids, mask) == 0) { return false; }
  return ::pthread_setaffinity_np(handle, sizeof(mask), &mask) == 0;
}

bool set_thread_affinity(std::thread& t, std::span<int const> cpu_ids) noexcept
{
  return set_thread_affinity(t.native_handle(), cpu_ids);
}

bool set_thread_affinity(std::jthread& t, std::span<int const> cpu_ids) noexcept
{
  return set_thread_affinity(t.native_handle(), cpu_ids);
}

bool set_current_thread_affinity(std::span<int const> cpu_ids) noexcept
{
  return set_thread_affinity(::pthread_self(), cpu_ids);
}

std::vector<int> get_thread_affinity(pthread_t handle) noexcept
{
  cpu_set_t mask{};
  if (::pthread_getaffinity_np(handle, sizeof(mask), &mask) != 0) { return {}; }
  std::vector<int> ids;
  // A conservative upper bound so the vector never reallocates on the caller's
  // hot path; the actual count is capped by CPU_SETSIZE anyway.
  ids.reserve(static_cast<std::size_t>(CPU_COUNT(&mask)));
  for (int i = 0; i < CPU_SETSIZE; ++i) {
    if (CPU_ISSET(i, &mask)) { ids.push_back(i); }
  }
  return ids;
}

std::vector<int> get_current_thread_affinity() noexcept
{
  return get_thread_affinity(::pthread_self());
}

// ---------------------------------------------------------------------------
// scheduling policy and priority
// ---------------------------------------------------------------------------

namespace {

/// Whether @p priority is legal under @p policy: @c 0 for time-sharing,
/// @c [1, 99] for the real-time policies.  Checked before the syscall so a
/// misuse is a clean @c false rather than an @c EINVAL that says the same
/// thing more obscurely.
[[nodiscard]] bool priority_valid_for(scheduling_policy policy, int priority) noexcept
{
  switch (policy) {
    case scheduling_policy::other:
    case scheduling_policy::batch:
    case scheduling_policy::idle: return priority == 0;
    case scheduling_policy::fifo:
    case scheduling_policy::round_robin: return priority >= 1 && priority <= 99;
  }
  return false;
}

}  // namespace

bool set_thread_scheduling(pthread_t handle, scheduling_policy policy, int priority) noexcept
{
  if (!priority_valid_for(policy, priority)) { return false; }
  sched_param param{};
  param.sched_priority = priority;
  return ::pthread_setschedparam(handle, to_native(policy), &param) == 0;
}

bool set_thread_scheduling(std::thread& t, scheduling_policy policy, int priority) noexcept
{
  return set_thread_scheduling(t.native_handle(), policy, priority);
}

bool set_thread_scheduling(std::jthread& t, scheduling_policy policy, int priority) noexcept
{
  return set_thread_scheduling(t.native_handle(), policy, priority);
}

bool set_current_thread_scheduling(scheduling_policy policy, int priority) noexcept
{
  return set_thread_scheduling(::pthread_self(), policy, priority);
}

std::optional<scheduling_info> get_thread_scheduling(pthread_t handle) noexcept
{
  int native_policy = 0;
  sched_param param{};
  if (::pthread_getschedparam(handle, &native_policy, &param) != 0) { return std::nullopt; }
  auto const policy = from_native(native_policy);
  if (!policy) { return std::nullopt; }
  return scheduling_info{*policy, param.sched_priority};
}

std::optional<scheduling_info> get_current_thread_scheduling() noexcept
{
  return get_thread_scheduling(::pthread_self());
}

// ---------------------------------------------------------------------------
// niceness
// ---------------------------------------------------------------------------

bool set_current_thread_niceness(int nice_value) noexcept
{
  if (nice_value < -20 || nice_value > 19) { return false; }
  // Linux extends PRIO_PROCESS to name a TID as well as a PID, which is the
  // documented way to nice a single thread rather than the whole process.
  // gettid() has no glibc wrapper on older toolchains, so go through syscall.
  auto const tid = static_cast<::pid_t>(::syscall(SYS_gettid));
  errno          = 0;
  if (::setpriority(PRIO_PROCESS, static_cast<id_t>(tid), nice_value) != 0) { return false; }
  return true;
}

std::optional<int> get_current_thread_niceness() noexcept
{
  auto const tid = static_cast<::pid_t>(::syscall(SYS_gettid));
  // getpriority returns -1 both on error and for a legitimate nice value of -1,
  // so errno has to be reset up front and consulted after.
  errno         = 0;
  auto const nv = ::getpriority(PRIO_PROCESS, static_cast<id_t>(tid));
  if (nv == -1 && errno != 0) { return std::nullopt; }
  return nv;
}

// ---------------------------------------------------------------------------
// identification
// ---------------------------------------------------------------------------

std::uint64_t current_thread_id() noexcept
{
  return static_cast<std::uint64_t>(::syscall(SYS_gettid));
}

}  // namespace sirius::exec::thread_util
