/*
 * Copyright 2025, Sirius Contributors.
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

#include "compression/simpatico_compressed_representation.hpp"
#include "memory/size_arithmetic.hpp"
#include "telemetry/data_batch_probe.hpp"

#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace sirius {

/**
 * @brief Global atomic counter for generating unique data batch IDs.
 *
 * This provides a simple way to generate unique IDs for data batches
 * without requiring a data_repository_manager instance.
 */
inline std::atomic<uint64_t> g_next_batch_id{0};

/**
 * @brief Generate a unique data batch ID.
 */
inline uint64_t get_next_batch_id() { return g_next_batch_id++; }

/**
 * @brief Get a cudf::table_view from a read-only data_batch accessor.
 *
 * Assumes the underlying data_batch contains a gpu_table_representation.
 * The caller MUST already hold the read_only_data_batch (shared lock) — this
 * helper deliberately does NOT internally call batch.to_read_only() because
 * doing so would let a misuse pattern hide a P1 self-deadlock (acquiring a
 * read lock on a batch that the caller is about to upgrade to mutable).
 *
 * @param batch The read-only accessor to extract the table view from.
 * @return cudf::table_view The underlying cudf table view.
 */
inline cudf::table_view get_cudf_table_view(const cucascade::read_only_data_batch& batch)
{
  auto* data = batch.get_data();
  if (data == nullptr) { throw std::runtime_error("data_batch has no data representation"); }
  return data->cast<cucascade::gpu_table_representation>().get_table_view();
}

/**
 * @brief Peak device bytes needed to materialize @p data on the GPU.
 *
 * For ordinary representations, only the logical destination lands on device,
 * so the peak equals the uncompressed size. A Simpatico representation must
 * also stage its physical payload before decompression produces that destination,
 * so both are alive simultaneously: physical_bytes + logical_bytes. Classification
 * is type-based rather than inferred from a compression ratio: equal-sized,
 * expanded, and projected Simpatico payloads still require staging. The estimate
 * saturates when the reported sizes exceed the range of std::size_t.
 */
inline std::size_t peak_materialization_bytes(const cucascade::idata_representation* data)
{
  if (data == nullptr) { return 0; }
  auto const logical_bytes = data->get_uncompressed_data_size_in_bytes();
  if (!is_simpatico_compressed_representation(data)) { return logical_bytes; }
  return memory::saturating_add(data->get_size_in_bytes(), logical_bytes);
}

/**
 * @brief Get a cudf::table_view from an idle data_batch (convenience overload).
 *
 * Acquires a temporary read-only lock, extracts the table_view, then releases the lock.
 *
 * @warning The returned table_view references GPU memory that is only guaranteed stable while a
 * read-only lock is held. Since this function releases the lock before returning, the view can
 * become dangling if another thread downgrades or mutates the batch concurrently. Only use this
 * overload in contexts where the caller has exclusive ownership of the batch (e.g., diagnostic
 * functions running synchronously within a pipeline task). Prefer the
 * get_cudf_table_view(const read_only_data_batch&) overload when the caller can hold the lock.
 *
 * @param batch The idle data batch to extract the table view from.
 * @return cudf::table_view The underlying cudf table view.
 */
// NOLINTNEXTLINE(readability-non-const-parameter) -- to_read_only() is non-const
inline cudf::table_view get_cudf_table_view(cucascade::data_batch& batch)
{
  auto ro    = batch.to_read_only();
  auto* data = ro.get_data();
  if (data == nullptr) { throw std::runtime_error("data_batch has no data representation"); }
  return data->cast<cucascade::gpu_table_representation>().get_table_view();
}

/**
 * @brief Create a shared_ptr<data_batch> from a cudf::table, recording the writer event.
 *
 * STREAM-LINEAGE: @p writer_stream is REQUIRED. Every data_batch carrying a
 * gpu_table_representation is born with a recorded writer event so
 * cucascade::convert_gpu_to_gpu() can call
 * cudaStreamWaitEvent(reader_stream, writer_event, 0) before peer-copying
 * source buffers. This closes the cross-mempool stream-ordered race in
 * multi-GPU runs.
 *
 * @param table The cudf table (will be moved from).
 * @param memory_space The memory space where the table resides.
 * @param writer_stream The stream on which @p table's data was last written.
 *                      MUST be the actual writer stream — passing the wrong
 *                      stream re-opens the race this contract closes.
 * @param telemetry_info Telemetry context threaded into the batch's quent probe so the new batch
 *                       is linked into the query's telemetry lineage. Pass the producing
 *                       operator's batch_telemetry(); pass a default-constructed value only when
 *                       no lineage is available (e.g. tests).
 * @return std::shared_ptr<cucascade::data_batch> The new data batch.
 */
inline std::shared_ptr<cucascade::data_batch> make_data_batch(
  cudf::table&& table,
  cucascade::memory::memory_space& memory_space,
  rmm::cuda_stream_view writer_stream,
  const telemetry::batch_telemetry_info& telemetry_info)
{
  auto gpu_repr = std::make_unique<cucascade::gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), memory_space, writer_stream);
  const auto batch_id = get_next_batch_id();
  return cucascade::data_batch::make(
    batch_id,
    std::move(gpu_repr),
    telemetry::quent_data_batch_probe::create(telemetry_info, batch_id));
}

/**
 * @brief Create a shared_ptr<data_batch> from a unique_ptr<cudf::table>, recording the writer
 * event.
 *
 * @copydoc make_data_batch(cudf::table&&, cucascade::memory::memory_space&,
 *                          rmm::cuda_stream_view, const telemetry::batch_telemetry_info&)
 */
inline std::shared_ptr<cucascade::data_batch> make_data_batch(
  std::unique_ptr<cudf::table> table,
  cucascade::memory::memory_space& memory_space,
  rmm::cuda_stream_view writer_stream,
  const telemetry::batch_telemetry_info& telemetry_info)
{
  auto gpu_repr = std::make_unique<cucascade::gpu_table_representation>(
    std::move(table), memory_space, writer_stream);
  const auto batch_id = get_next_batch_id();
  return cucascade::data_batch::make(
    batch_id,
    std::move(gpu_repr),
    telemetry::quent_data_batch_probe::create(telemetry_info, batch_id));
}

/**
 * @brief Create a shared_ptr<data_batch> that owns a cudf::table_view via a type-erased owner.
 *
 * Wraps the gpu_table_representation owning_table_view ctor: the batch holds a non-owning
 * @p view whose underlying device memory is kept alive by @p owner (e.g. a read_only_data_batch
 * lock on a source batch, and/or a shared_ptr<cudf::table> of freshly-evaluated columns). The
 * owner must be copy-constructible (std::any requirement). Used by the projection operator to
 * return columns without copying passthrough (BOUND_REF) inputs.
 *
 * STREAM-LINEAGE: @p writer_stream must be a stream that is ordered after every write to the
 * memory referenced by @p view (the caller is responsible for inserting any cudaStreamWaitEvent
 * needed to establish that ordering before calling this helper).
 *
 * @tparam Owner Copy-constructible type that keeps @p view's device memory alive.
 * @param view The table view to expose (data ownership lives in @p owner).
 * @param owner The owner keeping the viewed memory alive (moved/copied into std::any).
 * @param alloc_size Allocation size in bytes attributed to this batch.
 * @param memory_space The memory space where the viewed data resides.
 * @param writer_stream Stream ordered after the writes that produced @p view's data.
 * @param telemetry_info Telemetry context threaded into the batch's quent probe so the new batch
 *                       is linked into the query's telemetry lineage. Pass the producing
 *                       operator's batch_telemetry(); pass a default-constructed value only when
 *                       no lineage is available (e.g. tests).
 */
template <typename Owner>
inline std::shared_ptr<cucascade::data_batch> make_data_batch_from_view(
  cudf::table_view view,
  Owner&& owner,
  std::size_t alloc_size,
  cucascade::memory::memory_space& memory_space,
  rmm::cuda_stream_view writer_stream,
  const telemetry::batch_telemetry_info& telemetry_info)
{
  auto gpu_repr = std::make_unique<cucascade::gpu_table_representation>(
    view, std::forward<Owner>(owner), alloc_size, memory_space, writer_stream);
  const auto batch_id = get_next_batch_id();
  return cucascade::data_batch::make(
    batch_id,
    std::move(gpu_repr),
    telemetry::quent_data_batch_probe::create(telemetry_info, batch_id));
}

}  // namespace sirius
