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
}  // namespace

const spill_context* current_spill_context() noexcept { return t_current_spill_context; }

void set_spill_compression_settings(bool enabled,
                                    std::uint32_t explore_beam_width,
                                    std::size_t explore_max_bytes,
                                    double max_compressed_fraction,
                                    std::uint64_t replan_after_uses) noexcept
{
  g_spill_enabled.store(enabled, std::memory_order_relaxed);
  g_explore_beam_width.store(explore_beam_width, std::memory_order_relaxed);
  g_explore_max_bytes.store(explore_max_bytes, std::memory_order_relaxed);
  g_max_compressed_fraction.store(max_compressed_fraction, std::memory_order_relaxed);
  g_replan_after_uses.store(replan_after_uses, std::memory_order_relaxed);
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
  };
}

scoped_spill_context::scoped_spill_context(const spill_context& ctx) noexcept
  : _previous(t_current_spill_context)
{
  t_current_spill_context = &ctx;
}

scoped_spill_context::~scoped_spill_context() { t_current_spill_context = _previous; }

}  // namespace sirius::compression
