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

#include <atomic>

namespace sirius::compression {

namespace {
thread_local const spill_context* t_current_spill_context = nullptr;

std::atomic<bool> g_spill_enabled{false};
std::atomic<std::uint32_t> g_explore_beam_width{20};
std::atomic<std::size_t> g_explore_max_bytes{256ULL * 1024 * 1024};
std::atomic<double> g_max_compressed_fraction{0.75};
std::atomic<std::uint64_t> g_replan_after_uses{128};
std::atomic<std::uint32_t> g_error_tolerance{3};
std::atomic<double> g_replan_change_threshold{0.20};
std::atomic<std::size_t> g_explore_sample_rows{65536};
}  // namespace

const spill_context* current_spill_context() noexcept { return t_current_spill_context; }

void set_spill_compression_settings(bool enabled,
                                    std::uint32_t explore_beam_width,
                                    std::size_t explore_max_bytes,
                                    double max_compressed_fraction,
                                    std::uint64_t replan_after_uses,
                                    std::uint32_t error_tolerance,
                                    double replan_change_threshold,
                                    std::size_t explore_sample_rows) noexcept
{
  g_spill_enabled.store(enabled, std::memory_order_relaxed);
  g_explore_beam_width.store(explore_beam_width, std::memory_order_relaxed);
  g_explore_max_bytes.store(explore_max_bytes, std::memory_order_relaxed);
  g_max_compressed_fraction.store(max_compressed_fraction, std::memory_order_relaxed);
  g_replan_after_uses.store(replan_after_uses, std::memory_order_relaxed);
  g_error_tolerance.store(error_tolerance, std::memory_order_relaxed);
  g_replan_change_threshold.store(replan_change_threshold, std::memory_order_relaxed);
  g_explore_sample_rows.store(explore_sample_rows, std::memory_order_relaxed);
}

bool spill_compression_enabled() noexcept
{
  return g_spill_enabled.load(std::memory_order_relaxed);
}

spill_context make_spill_context(const cucascade::shared_data_repository* repo) noexcept
{
  return spill_context{
    .repo                    = repo,
    .explore_beam_width      = g_explore_beam_width.load(std::memory_order_relaxed),
    .explore_max_bytes       = g_explore_max_bytes.load(std::memory_order_relaxed),
    .max_compressed_fraction = g_max_compressed_fraction.load(std::memory_order_relaxed),
    .replan_after_uses       = g_replan_after_uses.load(std::memory_order_relaxed),
    .error_tolerance         = g_error_tolerance.load(std::memory_order_relaxed),
    .replan_change_threshold = g_replan_change_threshold.load(std::memory_order_relaxed),
    .explore_sample_rows     = g_explore_sample_rows.load(std::memory_order_relaxed),
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
