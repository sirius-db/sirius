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

#include "spill_context.hpp"

#include "compression_device_pool.hpp"

#include <cudf/utilities/error.hpp>

#include <rmm/error.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <stdexcept>

namespace sirius::compression {

namespace {
thread_local const spill_context* t_current_spill_context = nullptr;

std::atomic<bool> g_spill_enabled{false};

// Set when an allocation OOMs, cleared when the downgrade monitor sees pressure
// fall below downgrade_stop_fraction. Compressing costs device memory at exactly
// the moment there is none: the encode allocates working buffers while a spill is
// under way, so under pressure it competes with the query's own allocations.
std::atomic<bool> g_spill_suppressed{false};
std::atomic<std::uint32_t> g_explore_beam_width{20};
std::atomic<std::size_t> g_explore_max_bytes{256ULL * 1024 * 1024};
std::atomic<double> g_max_compressed_fraction{0.75};
std::atomic<std::uint64_t> g_replan_after_uses{128};
std::atomic<std::uint32_t> g_error_tolerance{3};
std::atomic<double> g_replan_change_threshold{0.20};
std::atomic<std::size_t> g_explore_sample_rows{65536};
std::atomic<std::size_t> g_spill_min_batch_bytes{64ULL * 1024 * 1024};
std::atomic<bool> g_spill_release_columns_early{false};
std::atomic<double> g_encode_reserve_fraction{0.5};
std::atomic<double> g_encode_min_headroom_fraction{0.10};
std::atomic<bool> g_encode_strict_reservation{false};
}  // namespace

void set_spill_encode_strict_reservation(bool strict) noexcept
{
  g_encode_strict_reservation.store(strict, std::memory_order_relaxed);
}

bool spill_encode_strict_reservation() noexcept
{
  return g_encode_strict_reservation.load(std::memory_order_relaxed);
}

const spill_context* current_spill_context() noexcept { return t_current_spill_context; }

void set_spill_compression_settings(bool enabled,
                                    std::uint32_t explore_beam_width,
                                    std::size_t explore_max_bytes,
                                    double max_compressed_fraction,
                                    std::uint64_t replan_after_uses,
                                    std::uint32_t error_tolerance,
                                    double replan_change_threshold,
                                    std::size_t explore_sample_rows,
                                    std::size_t min_batch_bytes,
                                    bool release_columns_early,
                                    double encode_reserve_fraction,
                                    double encode_min_headroom_fraction) noexcept
{
  g_spill_enabled.store(enabled, std::memory_order_relaxed);
  g_explore_beam_width.store(explore_beam_width, std::memory_order_relaxed);
  g_explore_max_bytes.store(explore_max_bytes, std::memory_order_relaxed);
  g_max_compressed_fraction.store(max_compressed_fraction, std::memory_order_relaxed);
  g_replan_after_uses.store(replan_after_uses, std::memory_order_relaxed);
  g_error_tolerance.store(error_tolerance, std::memory_order_relaxed);
  g_replan_change_threshold.store(replan_change_threshold, std::memory_order_relaxed);
  g_explore_sample_rows.store(explore_sample_rows, std::memory_order_relaxed);
  g_spill_min_batch_bytes.store(min_batch_bytes, std::memory_order_relaxed);
  g_spill_release_columns_early.store(release_columns_early, std::memory_order_relaxed);
  g_encode_reserve_fraction.store(encode_reserve_fraction, std::memory_order_relaxed);
  g_encode_min_headroom_fraction.store(encode_min_headroom_fraction, std::memory_order_relaxed);
}

namespace {

/// steady_clock milliseconds of the last physical device OOM; 0 = never.
std::atomic<std::int64_t> g_last_physical_oom_ms{0};

std::atomic<std::uint64_t> g_spill_skipped_pressure{0};
std::atomic<std::uint64_t> g_spill_skipped_reservation{0};
std::atomic<std::uint64_t> g_spill_fell_back{0};
std::atomic<std::uint64_t> g_in_place_fell_back{0};
std::atomic<std::uint64_t> g_output_fell_back{0};

std::int64_t now_ms() noexcept
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

}  // namespace

void note_physical_device_oom() noexcept
{
  // Never 0, which means "never happened".
  g_last_physical_oom_ms.store(std::max<std::int64_t>(1, now_ms()), std::memory_order_relaxed);
}

void clear_physical_device_oom_for_testing() noexcept
{
  g_last_physical_oom_ms.store(0, std::memory_order_relaxed);
}

const char* spill_compression_pressure_skip_reason() noexcept
{
  if (compression_device_pool_enabled()) {
    const auto arena = compression_device_pool_bytes();
    if (arena > 0 && static_cast<double>(compression_device_pool_used_bytes()) >=
                       kArenaSkipFraction * static_cast<double>(arena)) {
      return "compression arena nearly full";
    }
  }
  const auto last = g_last_physical_oom_ms.load(std::memory_order_relaxed);
  if (last != 0 && now_ms() - last < static_cast<std::int64_t>(kPhysicalOomSkipWindowMs)) {
    return "recent physical device OOM";
  }
  return nullptr;
}

void note_compression_fallback(compression_fallback_kind kind) noexcept
{
  switch (kind) {
    case compression_fallback_kind::spill_skipped_pressure:
      g_spill_skipped_pressure.fetch_add(1, std::memory_order_relaxed);
      break;
    case compression_fallback_kind::spill_skipped_reservation:
      g_spill_skipped_reservation.fetch_add(1, std::memory_order_relaxed);
      break;
    case compression_fallback_kind::spill_fell_back:
      g_spill_fell_back.fetch_add(1, std::memory_order_relaxed);
      break;
    case compression_fallback_kind::in_place:
      g_in_place_fell_back.fetch_add(1, std::memory_order_relaxed);
      break;
    case compression_fallback_kind::output:
      g_output_fell_back.fetch_add(1, std::memory_order_relaxed);
      break;
  }
}

compression_fallback_counters read_compression_fallback_counters() noexcept
{
  return {g_spill_skipped_pressure.load(std::memory_order_relaxed),
          g_spill_skipped_reservation.load(std::memory_order_relaxed),
          g_spill_fell_back.load(std::memory_order_relaxed),
          g_in_place_fell_back.load(std::memory_order_relaxed),
          g_output_fell_back.load(std::memory_order_relaxed)};
}

namespace testing {

namespace {
std::atomic<int> g_fault_kind{0};
std::atomic<std::uint32_t> g_fault_remaining{0};

/// Consume one armed fault of @p kind; false when none is armed.
bool take_fault(encode_fault kind) noexcept
{
  if (g_fault_kind.load(std::memory_order_relaxed) != static_cast<int>(kind)) { return false; }
  auto remaining = g_fault_remaining.load(std::memory_order_relaxed);
  while (remaining > 0) {
    if (g_fault_remaining.compare_exchange_weak(
          remaining, remaining - 1, std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}
}  // namespace

void inject_encode_fault(encode_fault kind, std::uint32_t count) noexcept
{
  g_fault_remaining.store(0, std::memory_order_relaxed);
  g_fault_kind.store(static_cast<int>(kind), std::memory_order_relaxed);
  g_fault_remaining.store(kind == encode_fault::none ? 0 : count, std::memory_order_relaxed);
}

void maybe_throw_encode_fault()
{
  if (take_fault(encode_fault::out_of_memory)) {
    throw rmm::out_of_memory("injected encode fault: out of memory");
  }
  if (take_fault(encode_fault::runtime_error)) {
    throw std::runtime_error("injected encode fault: runtime error");
  }
  if (take_fault(encode_fault::non_std_exception)) { throw 42; }
}

void maybe_throw_plan_fault()
{
  if (take_fault(encode_fault::plan_failure)) { throw std::runtime_error("injected plan fault"); }
}

void maybe_throw_decode_fault()
{
  if (take_fault(encode_fault::decode_cuda_oom)) {
    throw cudf::cuda_error("injected decode fault: cudaErrorMemoryAllocation",
                           cudaErrorMemoryAllocation);
  }
  if (take_fault(encode_fault::decode_corrupt)) {
    throw std::runtime_error("injected decode fault: corrupt payload");
  }
}

}  // namespace testing

bool spill_compression_enabled() noexcept
{
  return g_spill_enabled.load(std::memory_order_relaxed) &&
         !g_spill_suppressed.load(std::memory_order_relaxed);
}

void set_spill_compression_suppressed(bool suppressed) noexcept
{
  g_spill_suppressed.store(suppressed, std::memory_order_relaxed);
}

bool spill_compression_suppressed() noexcept
{
  return g_spill_suppressed.load(std::memory_order_relaxed);
}

spill_context make_spill_context(const cucascade::shared_data_repository* repo) noexcept
{
  return spill_context{
    .repo                         = repo,
    .explore_beam_width           = g_explore_beam_width.load(std::memory_order_relaxed),
    .explore_max_bytes            = g_explore_max_bytes.load(std::memory_order_relaxed),
    .max_compressed_fraction      = g_max_compressed_fraction.load(std::memory_order_relaxed),
    .replan_after_uses            = g_replan_after_uses.load(std::memory_order_relaxed),
    .error_tolerance              = g_error_tolerance.load(std::memory_order_relaxed),
    .replan_change_threshold      = g_replan_change_threshold.load(std::memory_order_relaxed),
    .explore_sample_rows          = g_explore_sample_rows.load(std::memory_order_relaxed),
    .min_batch_bytes              = g_spill_min_batch_bytes.load(std::memory_order_relaxed),
    .release_columns_early        = g_spill_release_columns_early.load(std::memory_order_relaxed),
    .encode_reserve_fraction      = g_encode_reserve_fraction.load(std::memory_order_relaxed),
    .encode_min_headroom_fraction = g_encode_min_headroom_fraction.load(std::memory_order_relaxed),
    .encode_strict_reservation    = g_encode_strict_reservation.load(std::memory_order_relaxed),
  };
}

scoped_spill_context::scoped_spill_context(const spill_context& ctx) noexcept
  : _previous(t_current_spill_context)
{
  t_current_spill_context = &ctx;
}

scoped_spill_context::~scoped_spill_context() { t_current_spill_context = _previous; }

// ── Task-output compression ──────────────────────────────────────────────────

namespace {
thread_local const output_compression_context* t_current_output_context = nullptr;

std::atomic<bool> g_output_enabled{false};
std::atomic<double> g_output_min_ratio{3.0};
std::atomic<double> g_output_min_compress_gbps{250.0};
std::atomic<double> g_output_min_decompress_gbps{250.0};
std::atomic<double> g_output_max_compressed_fraction{0.75};
std::atomic<std::size_t> g_output_min_batch_bytes{64ULL * 1024 * 1024};
std::atomic<bool> g_device_downgrade_enabled{false};
}  // namespace

const output_compression_context* current_output_compression_context() noexcept
{
  return t_current_output_context;
}

void set_output_compression_settings(bool enabled,
                                     double min_ratio,
                                     double min_compress_gbps,
                                     double min_decompress_gbps,
                                     double max_compressed_fraction,
                                     std::size_t min_batch_bytes,
                                     bool enable_device_downgrade) noexcept
{
  g_output_enabled.store(enabled, std::memory_order_relaxed);
  g_output_min_ratio.store(min_ratio, std::memory_order_relaxed);
  g_output_min_compress_gbps.store(min_compress_gbps, std::memory_order_relaxed);
  g_output_min_decompress_gbps.store(min_decompress_gbps, std::memory_order_relaxed);
  g_output_max_compressed_fraction.store(max_compressed_fraction, std::memory_order_relaxed);
  g_output_min_batch_bytes.store(min_batch_bytes, std::memory_order_relaxed);
  g_device_downgrade_enabled.store(enable_device_downgrade, std::memory_order_relaxed);
}

bool output_compression_enabled() noexcept
{
  return g_output_enabled.load(std::memory_order_relaxed);
}

bool device_compression_downgrade_enabled() noexcept
{
  return g_device_downgrade_enabled.load(std::memory_order_relaxed);
}

plan_register::plan_quality_gate output_compression_gate() noexcept
{
  return plan_register::plan_quality_gate{
    .min_ratio           = g_output_min_ratio.load(std::memory_order_relaxed),
    .min_compress_gbps   = g_output_min_compress_gbps.load(std::memory_order_relaxed),
    .min_decompress_gbps = g_output_min_decompress_gbps.load(std::memory_order_relaxed),
  };
}

output_compression_context make_output_compression_context(
  const cucascade::shared_data_repository* repo) noexcept
{
  return output_compression_context{
    .repo                    = repo,
    .max_compressed_fraction = g_output_max_compressed_fraction.load(std::memory_order_relaxed),
    .min_ratio               = g_output_min_ratio.load(std::memory_order_relaxed),
    .min_batch_bytes         = g_output_min_batch_bytes.load(std::memory_order_relaxed),
  };
}

scoped_output_compression_context::scoped_output_compression_context(
  const output_compression_context& ctx) noexcept
  : _previous(t_current_output_context)
{
  t_current_output_context = &ctx;
}

scoped_output_compression_context::~scoped_output_compression_context()
{
  t_current_output_context = _previous;
}

}  // namespace sirius::compression
