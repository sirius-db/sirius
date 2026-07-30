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

#include "compression/output_compression.hpp"

#include "compression/compressed_representation.hpp"
#include "compression/plan_register.hpp"
#include "compression/spill_context.hpp"
#include "data/sirius_converter_registry.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <exception>

namespace sirius::compression {

bool try_compress_output_batch(cucascade::data_batch& batch,
                               const cucascade::shared_data_repository* repo,
                               op::MemoryBarrierType barrier,
                               rmm::cuda_stream_view stream)
{
  if (!output_compression_enabled() || repo == nullptr) { return false; }

  // Only where the data will actually sit. A PIPELINE or PARTIAL consumer starts
  // on these batches almost immediately, so compressing them just buys an
  // immediate decompress. See the header for the full reasoning.
  if (barrier != op::MemoryBarrierType::FULL) { return false; }

  try {
    // Non-blocking: a batch a consumer already holds is not ours to rewrite, and
    // publication must not stall waiting for one.
    auto mut_opt = batch.try_to_mutable();
    if (!mut_opt) { return false; }
    auto& mut = *mut_opt;

    auto* space = const_cast<cucascade::memory::memory_space*>(mut.get_memory_space());
    if (space == nullptr || space->get_tier() != cucascade::memory::Tier::GPU) { return false; }

    auto const* data = mut.get_data();
    if (data == nullptr) { return false; }
    // Must be an uncompressed GPU table: the GPU tier also holds
    // compressed_device_representation, and re-compressing one is both wrong and
    // a no-op we should not pay for.
    auto const* gpu_rep = dynamic_cast<const cucascade::gpu_table_representation*>(data);
    if (gpu_rep == nullptr) { return false; }

    const std::size_t data_size = data->get_size_in_bytes();
    if (data_size == 0) { return false; }

    // Amortization gate. Compressing costs a roughly fixed ~3 ms per batch
    // (per-column, per-plan-node stream syncs plus blob staging), measured on the
    // SF100 sweep at ~1-2% of the codecs' rated throughput — so the cost is
    // almost entirely independent of how much data the batch holds. A small
    // batch cannot repay it however well it compresses.
    if (data_size < make_output_compression_context(repo).min_batch_bytes) { return false; }

    const auto num_columns = static_cast<std::size_t>(gpu_rep->get_table_view().num_columns());
    if (num_columns == 0) { return false; }

    // Cheap pre-check before reserving anything: an edge where no column's plan
    // clears the gate is the common case and must cost nothing but a lookup.
    // decide_output_plan caches this per edge, so the lineage walk runs once.
    if (!plan_register::global()
           .decide_output_plan(repo, num_columns, output_compression_gate())
           .has_value()) {
      return false;
    }

    // Compressing allocates the compressed form while the source table is still
    // live, so reserve the source's size as the upper bound on that transient.
    auto reservation = space->make_reservation_or_null(data_size);
    if (!reservation) { return false; }

    const auto ctx = make_output_compression_context(repo);
    scoped_output_compression_context guard(ctx);
    mut.convert_to<compressed_device_representation>(
      sirius::converter_registry::get(), *reservation, stream);
    return true;
  } catch (const std::exception& e) {
    // Declining is routine here — below the size threshold, a plan that did not
    // deliver, or an allocation failure — and it is never fatal: convert_to only
    // installs the new representation after the converter returns, so the batch
    // is untouched and publication proceeds uncompressed.
    SIRIUS_LOG_DEBUG(
      "[output_compression] declined for repo={} ({})", static_cast<const void*>(repo), e.what());
    return false;
  }
}

namespace {

/// The uncompressed GPU table inside @p batch, or nullptr when it is not one
/// (already compressed, or spilled off the GPU).
const cucascade::gpu_table_representation* as_gpu_table(const cucascade::idata_representation* data)
{
  return dynamic_cast<const cucascade::gpu_table_representation*>(data);
}

}  // namespace

device_compression_estimate estimate_device_compression(
  const cucascade::data_batch& batch, const cucascade::shared_data_repository* repo)
{
  device_compression_estimate est;
  if (!device_compression_downgrade_enabled() || repo == nullptr) { return est; }

  // Non-blocking, and this is load-bearing. The downgrade thread calls this
  // while pricing every candidate; a blocking to_read_only() here waits on a
  // batch a pipeline thread holds exclusively, while that pipeline thread is
  // itself waiting on this downgrade to free memory — a deadlock, observed as
  // 12 downgrade requests completed in 420 s against thousands in the baseline.
  // A batch we cannot inspect is simply not a candidate.
  auto ro_opt = const_cast<cucascade::data_batch&>(batch).try_to_read_only();
  if (!ro_opt) { return est; }
  auto& ro         = *ro_opt;
  auto const* data = ro.get_data();
  auto const* gpu  = as_gpu_table(data);
  if (gpu == nullptr) { return est; }

  const auto num_columns = static_cast<std::size_t>(gpu->get_table_view().num_columns());
  if (num_columns == 0) { return est; }
  est.current_bytes = data->get_size_in_bytes();
  if (est.current_bytes == 0) { return est; }

  const auto picked =
    plan_register::global().select_output_plans(repo, num_columns, output_compression_gate());
  if (picked.empty()) { return est; }

  // Equal-footprint approximation: a qualifying column contributes 1/ratio of its
  // share, a non-qualifying one contributes its whole share.
  double compressed_share = static_cast<double>(num_columns - picked.size());
  for (auto const& p : picked) {
    const double r = p.metrics.compression_ratio;
    compressed_share += (r > 1.0) ? 1.0 / r : 1.0;
  }
  if (compressed_share <= 0.0) { return est; }

  est.predicted_ratio = static_cast<double>(num_columns) / compressed_share;
  if (est.predicted_ratio <= 1.0) { return est; }

  est.predicted_freed = static_cast<std::size_t>(static_cast<double>(est.current_bytes) *
                                                 (1.0 - 1.0 / est.predicted_ratio));
  est.viable          = est.predicted_freed > 0;
  return est;
}

std::size_t compress_in_place_for_downgrade(cucascade::data_batch& batch,
                                            const cucascade::shared_data_repository* repo,
                                            rmm::cuda_stream_view stream)
{
  if (!device_compression_downgrade_enabled() || repo == nullptr) { return 0; }

  try {
    auto mut_opt = batch.try_to_mutable();
    if (!mut_opt) {
      SIRIUS_LOG_DEBUG("[output_compression] in-place skip: batch not exclusively lockable");
      return 0;
    }
    auto& mut = *mut_opt;

    auto* space = const_cast<cucascade::memory::memory_space*>(mut.get_memory_space());
    if (space == nullptr || space->get_tier() != cucascade::memory::Tier::GPU) {
      SIRIUS_LOG_DEBUG("[output_compression] in-place skip: not GPU-tier");
      return 0;
    }

    auto const* data = mut.get_data();
    if (as_gpu_table(data) == nullptr) {
      SIRIUS_LOG_DEBUG("[output_compression] in-place skip: not a plain gpu_table");
      return 0;
    }
    const std::size_t before = data->get_size_in_bytes();
    if (before == 0) {
      SIRIUS_LOG_DEBUG("[output_compression] in-place skip: zero bytes");
      return 0;
    }

    // Reserve the *compressed* footprint we expect to produce, not the source
    // size. This runs on a GPU that is by definition short of memory — that is
    // why a downgrade was requested — so reserving `before` fails essentially
    // always, which is exactly what the first run showed: 0 of 17 candidates
    // compressed, every one of them blocked on the reservation.
    //
    // The compressed output is what the new representation keeps; the codec's
    // own scratch is transient and is covered by the headroom the downgrade
    // trigger leaves. Estimated conservatively, and a failed reservation still
    // just declines.
    const auto est = estimate_device_compression(batch, repo);
    const std::size_t reserve_bytes =
      est.viable && est.predicted_ratio > 1.0
        ? static_cast<std::size_t>(static_cast<double>(before) / est.predicted_ratio)
        : before;
    auto reservation = space->make_reservation_or_null(reserve_bytes);
    if (!reservation) {
      SIRIUS_LOG_DEBUG("[output_compression] in-place skip: reservation of {}B failed (batch {}B)",
                       reserve_bytes,
                       before);
      return 0;
    }

    const auto ctx = make_output_compression_context(repo);
    scoped_output_compression_context guard(ctx);
    mut.convert_to<compressed_device_representation>(
      sirius::converter_registry::get(), *reservation, stream);

    auto const* after_rep   = mut.get_data();
    const std::size_t after = after_rep ? after_rep->get_size_in_bytes() : before;
    const std::size_t freed = (after < before) ? before - after : 0;
    SIRIUS_LOG_DEBUG(
      "[output_compression] downgrade compressed in place repo={} {}B -> {}B (freed {}B)",
      static_cast<const void*>(repo),
      before,
      after,
      freed);
    return freed;
  } catch (const std::exception& e) {
    SIRIUS_LOG_DEBUG("[output_compression] in-place downgrade declined for repo={} ({})",
                     static_cast<const void*>(repo),
                     e.what());
    return 0;
  }
}

}  // namespace sirius::compression
