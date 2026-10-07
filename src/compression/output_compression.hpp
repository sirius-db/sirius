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

#include <rmm/cuda_stream_view.hpp>

namespace cucascade {
class data_batch;
class data_repository;
using shared_data_repository = data_repository;
}  // namespace cucascade

namespace sirius::op {
enum class MemoryBarrierType;
}  // namespace sirius::op

namespace sirius::compression {

/**
 * @brief Compress a finished task's output batch in place, if it is worth it.
 *
 * Called from the operator sink, once per output batch, just before the batch is
 * published to its consumers. Running here means the batch is still exclusively
 * ours — it has not been added to any repository, so no consumer task can have
 * subscribed to it and `try_to_mutable()` is uncontended. The batch stays on the
 * GPU and stays usable; it is simply held in a smaller form until a consumer
 * materializes it, at which point the existing compressed_device_representation
 * → gpu_table_representation converter decodes it.
 *
 * **Only FULL-barrier edges are compressed.** Compression pays for itself only
 * while the data sits resident. On a PIPELINE edge the consumer processes batches
 * as they arrive and on a PARTIAL edge it drains incrementally, so in both cases a
 * task follows almost immediately and we would decompress what we just
 * compressed — strictly a loss. A FULL barrier means the consumer cannot start
 * until the whole upstream pipeline finishes, so the batch accumulates in the
 * repository; that accumulation is also what drives spilling, which is exactly
 * the footprint worth shrinking.
 *
 * A column is compressed only when @p repo's column lineage reaches a base-table
 * plan whose *measured* ratio and throughputs clear the configured gate; columns
 * without one are stored raw inside the same compressed table. When no column
 * qualifies — the common case — this costs a single register lookup.
 *
 * Never throws: any failure (no plan, not enough saving, allocation failure, the
 * batch not being idle or GPU-resident) leaves the batch exactly as it was and
 * returns false, so publication proceeds uncompressed.
 *
 * @return true when the batch is now held compressed.
 */
bool try_compress_output_batch(cucascade::data_batch& batch,
                               const cucascade::shared_data_repository* repo,
                               op::MemoryBarrierType barrier,
                               rmm::cuda_stream_view stream);

// ── Compress-in-place as a downgrade target ──────────────────────────────────
//
// A third option for the downgrade executor, alongside spilling to HOST and
// DISK: compress the batch where it already is. The data stays on the GPU and
// stays usable — it is just held smaller — so this avoids the D2H copy and the
// later readback entirely.
//
// It is only worth doing when the *set* of candidates can actually satisfy the
// request. Compressing frees `size * (1 - 1/ratio)` per batch, so for candidate
// bytes C and a request R the set needs a ratio of at least C / (C - R); C <= R
// cannot be satisfied at any ratio. Predicting that up front matters because,
// unlike a spill, a compression that turns out not to pay cannot be undone
// cheaply — the batch has to be decompressed to be used again.

/// What compressing @p batch in place is expected to cost and free.
struct device_compression_estimate {
  std::size_t current_bytes{0};    ///< the batch's device footprint now
  std::size_t predicted_freed{0};  ///< bytes compressing is expected to release
  double predicted_ratio{1.0};     ///< whole-batch ratio implied by the plans
  bool viable{false};              ///< false when nothing about it qualifies

  [[nodiscard]] bool operator!() const { return !viable; }
};

/**
 * @brief Predict the saving from compressing @p batch in place, without doing it.
 *
 * Uses the offline plan metrics reached through @p repo's column lineage, so it
 * costs a register lookup and no GPU work. Columns whose plan does not clear the
 * gate are counted as incompressible (ratio 1.0), which makes the estimate
 * conservative: it is the saving from the columns we would actually compress.
 *
 * The per-column ratios are combined assuming equal column footprints, since the
 * real per-column byte split is not known without inspecting the batch. That is
 * an approximation — a batch whose one qualifying column is also its largest will
 * beat the estimate, and vice versa.
 */
[[nodiscard]] device_compression_estimate estimate_device_compression(
  const cucascade::data_batch& batch, const cucascade::shared_data_repository* repo);

/**
 * @brief Compress @p batch in place on the device. Returns bytes freed (0 on
 *        failure, leaving the batch untouched).
 *
 * Applies none of @ref try_compress_output_batch's gates — no barrier check, no
 * minimum batch size. The caller is the downgrade executor, which has already
 * established both that the memory is needed and that this candidate set can
 * supply it; a size gate exists only to avoid speculative work, and this is not
 * speculative.
 */
[[nodiscard]] std::size_t compress_in_place_for_downgrade(
  cucascade::data_batch& batch,
  const cucascade::shared_data_repository* repo,
  rmm::cuda_stream_view stream);

}  // namespace sirius::compression
