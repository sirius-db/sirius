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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <random>

namespace sirius::io {

/**
 * Backend-neutral IO telemetry seam. Must stay free of telemetry-backend
 * dependencies (no generated bridge headers, no Rust types). With no sink
 * registered (the default) every emission point is a single untaken branch.
 */

/// 128-bit id, layout-compatible with the engine's UUID type (which lives in
/// a generated header this seam must not include). Nil (all zero) = unknown.
struct io_uuid {
  uint64_t high_bits{0};
  uint64_t low_bits{0};

  [[nodiscard]] constexpr bool is_nil() const noexcept { return high_bits == 0 && low_bits == 0; }
  constexpr bool operator==(const io_uuid&) const noexcept = default;
};

enum class io_phase : uint8_t { unknown, scan, prefetch, bind, open, list, head, footer };

/// Which code path served a leaf datasource read. Cache/stash HIT truth is a
/// join, not a field: a served-from-memory read has no backend_io_record with
/// its read_id.
enum class io_route : uint8_t { direct, cache };

enum class io_outcome : uint8_t { ok, error, cancelled };

/// Captured structurally at report time — never parsed from message strings.
enum class io_terminal_class : uint8_t {
  ok,
  http_error,
  transport_error,
  cuda_error,
  cancelled,
  shutdown,
};

/// Flows BY VALUE with each read; never shared by pointer into asynchronous
/// paths (the prefetch cache outlives the datasource that armed it).
struct io_attribution {
  io_uuid query_uuid{};
  io_uuid pipeline_uuid{};
  int32_t device_id{-1};
  io_phase phase{io_phase::unknown};
};

/// Exactly one per caller-visible datasource read, emitted at the leaf
/// implementations only (overload delegation must not double-count).
struct logical_io_record {
  io_uuid read_id{};
  io_attribution attribution{};
  io_route route{io_route::direct};
  uint64_t object_id{0};
  uint64_t offset{0};
  uint64_t bytes{0};
  uint64_t t_begin_ns{0};
  uint64_t t_end_ns{0};
  io_outcome outcome{io_outcome::ok};
};

/// Exactly one per request_manager, i.e. per actual backend access.
struct backend_io_record {
  io_uuid read_id{};
  io_attribution attribution{};
  uint64_t object_id{0};
  uint64_t bytes_requested{0};
  uint64_t bytes_delivered{0};
  uint32_t chunks_total{0};
  uint32_t chunks_completed{0};
  uint32_t retries{0};
  io_terminal_class terminal{io_terminal_class::ok};
  uint64_t t_create_ns{0};
  uint64_t t_complete_ns{0};
};

/// Consumer seam. Implementations must be thread-safe (calls arrive from
/// caller, reactor, and prefetch-cache threads) and may not block or throw.
class io_telemetry_sink {
 public:
  virtual ~io_telemetry_sink() = default;

  virtual void on_logical_read(const logical_io_record& record) noexcept = 0;
  virtual void on_backend_read(const backend_io_record& record) noexcept = 0;
};

/// Joins a backend record to its logical read. Built only when a sink is
/// active; pointer arguments are valid only for the duration of the call.
struct io_read_context {
  io_attribution attribution{};
  io_uuid read_id{};
  uint64_t object_id{0};
};

/// Process-unique read id (random high half, counter low half) — deliberately
/// not the engine's generated UUID type; joins are process-local.
namespace detail {
/// std::random_device may throw; fall back to clock/address entropy rather
/// than terminating from a noexcept context.
[[nodiscard]] inline uint64_t io_telemetry_entropy() noexcept
{
  try {
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32) ^ rd();
  } catch (...) {
    static const int anchor = 0;
    return static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()) ^
           reinterpret_cast<uintptr_t>(&anchor);
  }
}
}  // namespace detail

[[nodiscard]] inline io_uuid make_io_read_id() noexcept
{
  static const uint64_t hi = detail::io_telemetry_entropy() ^ 0x5152'4553'544c'4f47ULL;
  static std::atomic<uint64_t> counter{1};
  return {hi, counter.fetch_add(1, std::memory_order_relaxed)};
}

/// Session-salted 64-bit object identity; the salt is minted once per process
/// and never recorded. Centralized so every emitter derives it identically.
[[nodiscard]] inline uint64_t make_io_object_identity(const char* data, size_t len) noexcept
{
  static const uint64_t salt = detail::io_telemetry_entropy() ^ 0x4f42'4a49'4453'414cULL;
  uint64_t h                 = 0xcbf29ce484222325ULL ^ salt;
  for (size_t i = 0; i < len; ++i) {
    h ^= static_cast<unsigned char>(data[i]);
    h *= 0x100000001b3ULL;
  }
  return h ^ (salt << 1);
}

[[nodiscard]] inline uint64_t io_now_ns() noexcept
{
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now().time_since_epoch())
                                 .count());
}

}  // namespace sirius::io
