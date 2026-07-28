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

#include <cstddef>
#include <cstdint>

namespace cucascade {
class data_repository;
using shared_data_repository = data_repository;
}  // namespace cucascade

namespace sirius::compression {

/**
 * @brief Per-thread context handed to the spill-compression converters.
 *
 * cuCascade's converter signature is fixed —
 * `(source, target_memory_space, stream, reservation)` — so there is no way to
 * pass the originating query-graph edge through it. The spill path needs that
 * edge to key the plan register (one Simpatico plan per operator output port).
 *
 * `convertible_data_batch::convert()` installs this context for the duration of
 * a single `convert_to<compressed_*_representation>()` call, which runs under
 * the batch's exclusive (mutable) lock on the calling downgrade-pool thread.
 * The converter reads it back via @ref current_spill_context.
 */
struct spill_context {
  /// Query-graph edge this batch came from; the plan_register key.
  const cucascade::shared_data_repository* repo{nullptr};

  /// Beam width for the first-spill explorer on this edge.
  std::uint32_t explore_beam_width{20};

  /// Per-column byte cap for the first-spill explorer.
  std::size_t explore_max_bytes{256ULL * 1024 * 1024};

  /// Discard the compressed form when it exceeds this fraction of the original
  /// device footprint (the batch is then spilled uncompressed).
  double max_compressed_fraction{0.75};

  /// Re-explore this edge once its cached plan has been used this many times
  /// (0 = never). Also re-tests an edge previously judged not worth compressing.
  std::uint64_t replan_after_uses{128};
};

/// The calling thread's active spill context, or nullptr when none is installed.
[[nodiscard]] const spill_context* current_spill_context() noexcept;

// ── Process-global settings ──────────────────────────────────────────────────
//
// The spill path runs inside cuCascade converters, which have no access to a
// SiriusContext, so the relevant compression_config fields are mirrored into
// process-global state at context initialization (same pattern as
// set_decompress_column_threads).

/// Mirror the spill-compression fields of compression_config into global state.
void set_spill_compression_settings(bool enabled,
                                    std::uint32_t explore_beam_width,
                                    std::size_t explore_max_bytes,
                                    double max_compressed_fraction,
                                    std::uint64_t replan_after_uses) noexcept;

/// Whether spill compression is enabled process-wide.
[[nodiscard]] bool spill_compression_enabled() noexcept;

/// Build a spill_context for @p repo from the process-global settings.
[[nodiscard]] spill_context make_spill_context(
  const cucascade::shared_data_repository* repo) noexcept;

/**
 * @brief RAII guard installing a @ref spill_context for the calling thread.
 *
 * Nesting is not expected (a downgrade thread converts one batch at a time);
 * the guard restores the previous context on destruction regardless.
 */
class scoped_spill_context {
 public:
  explicit scoped_spill_context(const spill_context& ctx) noexcept;
  ~scoped_spill_context();

  scoped_spill_context(const scoped_spill_context&)            = delete;
  scoped_spill_context& operator=(const scoped_spill_context&) = delete;
  scoped_spill_context(scoped_spill_context&&)                 = delete;
  scoped_spill_context& operator=(scoped_spill_context&&)      = delete;

 private:
  const spill_context* _previous;
};

}  // namespace sirius::compression
