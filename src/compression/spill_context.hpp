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

#include "plan_register.hpp"

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

  /// Consecutive compression errors to absorb before writing an edge off, so a
  /// transient failure under memory pressure is not mistaken for a verdict.
  std::uint32_t error_tolerance{3};

  /// Relative change in a column's ratio or throughputs below which a
  /// re-explored plan is treated as equivalent and the cached plan kept.
  double replan_change_threshold{0.20};

  /// Row prefix the explorer runs on (0 = whole column). Bounds the beam
  /// search's allocation, which otherwise fails under the memory pressure that
  /// caused the spill in the first place.
  std::size_t explore_sample_rows{65536};

  /// Batches smaller than this spill raw. Compression costs a roughly fixed
  /// amount per batch, so below some size it cannot repay the setup however good
  /// the ratio. Mirrors output_compression_context::min_batch_bytes; the spill
  /// path needs its own because it cannot choose its batch sizes.
  std::size_t min_batch_bytes{64ULL * 1024 * 1024};

  /// Free each source column as it is encoded; see
  /// sirius_config.hpp::spill_release_columns_early for the trade-off.
  bool release_columns_early{false};

  /// Fraction of a batch's uncompressed size to reserve on the device for the
  /// encode's working memory, when there is no compression arena.
  ///
  /// With an arena the encode allocates from a pool carved off the device at
  /// startup, outside cuCascade's accounting entirely. Without one it allocates
  /// from the query's own pool — during a downgrade, i.e. exactly when that pool
  /// is exhausted — and it does so *unreserved*, so it can push the pool past
  /// what reservations promised and surface as an OOM in an unrelated operator.
  /// Reserving first makes the demand visible and, when it cannot be met, lets
  /// the batch decline to an uncompressed spill instead of destabilising the
  /// query.
  ///
  /// A fraction rather than the full size because the encode's peak is the
  /// compressed output (input/ratio, ~0.2 at the 5x this workload sees) plus
  /// codec scratch of the same order — not another whole copy of the batch.
  /// Reserving 1.0 would be safe but would almost never be grantable under the
  /// pressure that triggered the spill, silently disabling compression.
  double encode_reserve_fraction{0.5};
};

/// The calling thread's active spill context, or nullptr when none is installed.
[[nodiscard]] const spill_context* current_spill_context() noexcept;

/**
 * @brief Per-thread context handed to the task-output compression converter.
 *
 * Same reason as @ref spill_context — the converter signature cannot carry the
 * edge — but a distinct type because the two paths differ in kind: the spill
 * path compresses because it must and will explore to find a plan, while this
 * one compresses only when lineage already offers a measured plan good enough to
 * be worth the GPU time.
 *
 * Installed by the operator sink for the duration of one
 * `convert_to<compressed_device_representation>()` call.
 */
struct output_compression_context {
  /// Query-graph edge this batch is being published to; the plan_register key.
  const cucascade::shared_data_repository* repo{nullptr};

  /// Discard the compressed form when it exceeds this fraction of the original
  /// device footprint (the batch is then published uncompressed).
  double max_compressed_fraction{0.75};

  /// Thresholds a column's plan must clear, and that its *achieved* ratio is
  /// re-checked against on the first batch.
  double min_ratio{3.0};

  /// Smallest batch worth compressing; below this the fixed per-batch cost
  /// cannot be repaid. See compression_config for the measurement.
  std::size_t min_batch_bytes{64ULL * 1024 * 1024};
};

/// The calling thread's active output-compression context, or nullptr.
[[nodiscard]] const output_compression_context* current_output_compression_context() noexcept;

/// Mirror the output-compression fields of compression_config into global state.
void set_output_compression_settings(bool enabled,
                                     double min_ratio,
                                     double min_compress_gbps,
                                     double min_decompress_gbps,
                                     double max_compressed_fraction,
                                     std::size_t min_batch_bytes,
                                     bool enable_device_downgrade) noexcept;

/// Whether eager task-output compression (the sink-time hook) is enabled.
[[nodiscard]] bool output_compression_enabled() noexcept;

/// Whether the downgrade executor may compress in place on the device.
[[nodiscard]] bool device_compression_downgrade_enabled() noexcept;

/// The configured gate for admitting a column's offline plan.
[[nodiscard]] plan_register::plan_quality_gate output_compression_gate() noexcept;

/// Build an output_compression_context for @p repo from the global settings.
[[nodiscard]] output_compression_context make_output_compression_context(
  const cucascade::shared_data_repository* repo) noexcept;

/// RAII guard installing an @ref output_compression_context for the calling thread.
class scoped_output_compression_context {
 public:
  explicit scoped_output_compression_context(const output_compression_context& ctx) noexcept;
  ~scoped_output_compression_context();

  scoped_output_compression_context(const scoped_output_compression_context&)            = delete;
  scoped_output_compression_context& operator=(const scoped_output_compression_context&) = delete;
  scoped_output_compression_context(scoped_output_compression_context&&)                 = delete;
  scoped_output_compression_context& operator=(scoped_output_compression_context&&)      = delete;

 private:
  const output_compression_context* _previous;
};

// ── Process-global settings ──────────────────────────────────────────────────
//
// The spill path runs inside cuCascade converters, which have no access to a
// SiriusContext, so the relevant compression_config fields are mirrored into
// process-global state at context initialization.

/// Mirror the spill-compression fields of compression_config into global state.
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
                                    double encode_reserve_fraction) noexcept;

/// Whether spill compression is enabled process-wide *and* not currently
/// suppressed. This is the predicate the spill path consults.
[[nodiscard]] bool spill_compression_enabled() noexcept;

/**
 * @brief Suppress spill compression without changing the configured setting.
 *
 * Compression needs device memory to encode, and a spill happens precisely when
 * there is none left, so under pressure the encode competes with the query's own
 * allocations. Suppressing it lets the spill proceed raw and frees memory sooner.
 *
 * Set by the OOM policy when an allocation fails; cleared by the downgrade
 * monitor once the space is no longer above its downgrade trigger. The
 * trigger/stop hysteresis is what keeps this from flapping — suppression holds
 * for the whole pressure episode rather than toggling per allocation. Per-query
 * state is dropped independently by plan_register::clear_spill_state().
 */
void set_spill_compression_suppressed(bool suppressed) noexcept;

/// Whether compression is currently suppressed by memory pressure.
[[nodiscard]] bool spill_compression_suppressed() noexcept;

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
