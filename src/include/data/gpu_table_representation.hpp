/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Vendored from cuCascade (b4abc2d:include/data/gpu_table_representation.hpp) when
// cuCascade dropped its libcudf dependency (NVIDIA/cuCascade#142). This cuDF-backed representation
// now lives in Sirius under namespace sirius. The writer-event methods, formerly plain methods
// reached via dynamic_cast, are now overrides of cucascade::idata_representation virtuals.

#pragma once

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <cucascade/data/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <any>
#include <cstddef>
#include <memory>
#include <variant>

namespace sirius {

using cucascade::idata_representation;

/**
 * @brief Data representation for a table being stored in GPU memory.
 *
 * This class currently represents a table just as a cuDF table along with the allocation where the
 * cudf's table data actually resides. The primary purpose for this is that the table can be
 * directly passed to cuDF APIs for processing without any additional copying while the underlying
 * memory is still owned/tracked by our memory allocator.
 *
 * TODO: Once the GPU memory resource is implemented, replace the allocation type from
 * IAllocatedMemory to the concrete type returned by the GPU memory allocator.
 */
class gpu_table_representation : public idata_representation {
 public:
  /**
   * @brief Construct a new gpu_table_representation object.
   *
   * STREAM-LINEAGE: every gpu_table_representation must be born with a recorded
   * writer event so cross-stream / cross-device readers (notably
   * representation_converters.cpp's convert_gpu_to_gpu()) can establish ordering
   * via cudaStreamWaitEvent. The constructor calls record_writer_event(@p
   * writer_stream) automatically — passing a default-constructed
   * cuda_stream_view records no event (legacy, only acceptable for paths whose
   * data was never produced on any stream).
   *
   * @param table Unique pointer to the cuDF table with the data (ownership is transferred)
   * @param memory_space The memory space where the GPU table resides
   * @param writer_stream The stream on which @p table's data was last written.
   *                      MUST be the actual writer stream — passing the wrong
   *                      stream re-introduces the race this contract closes.
   */
  gpu_table_representation(std::unique_ptr<cudf::table> table,
                           cucascade::memory::memory_space& memory_space,
                           rmm::cuda_stream_view writer_stream);

  /**
   * @brief Construct a new gpu_table_representation object from a cudf::table_view.
   *
   * STREAM-LINEAGE: writer_stream is REQUIRED — must be the stream on which the
   * underlying data of @p table_view was last written. See the simple-table ctor
   * docstring for the writer_stream contract.
   *
   * @param table_view View of the cuDF table (data ownership lives in @p owner)
   * @tparam Owner The type of the owner of the cuDF table (e.g., a specific operator or component)
   * @param owner Owner of the underlying data (transferred via std::any storage)
   * @param alloc_size Allocation size in bytes for the data
   * @param memory_space The memory space where the GPU table resides
   * @param writer_stream The stream on which @p table_view's data was last written.
   */
  template <typename Owner>
  gpu_table_representation(cudf::table_view table_view,
                           Owner&& owner,
                           std::size_t alloc_size,
                           cucascade::memory::memory_space& memory_space,
                           rmm::cuda_stream_view writer_stream);

  /**
   * @brief Destructor — destroys the writer-event if one was recorded.
   *
   * STREAM-LINEAGE: events recorded on a writer stream via record_writer_event() are
   * owned by the representation and released on destruction.
   */
  ~gpu_table_representation() override;

  // Non-copyable / non-movable: the representation owns a cudaEvent_t handle whose
  // lifetime must be unique. Move semantics could be added but are not needed by
  // any in-tree caller.
  gpu_table_representation(const gpu_table_representation&)            = delete;
  gpu_table_representation(gpu_table_representation&&)                 = delete;
  gpu_table_representation& operator=(const gpu_table_representation&) = delete;
  gpu_table_representation& operator=(gpu_table_representation&&)      = delete;

  /**
   * @brief Get the size of the data representation in bytes
   *
   * @return std::size_t The number of bytes used to store this representation
   */
  std::size_t get_size_in_bytes() const override;

  /**
   * @copydoc idata_representation::get_logical_data_size_in_bytes
   */
  std::size_t get_uncompressed_data_size_in_bytes() const override;

  /**
   * @brief Create a deep copy of this GPU table representation.
   *
   * The cloned representation will have its own copy of the underlying cuDF table,
   * residing in the same memory space as the original.
   *
   * @param stream CUDA stream for memory operations
   * @return std::unique_ptr<idata_representation> A new gpu_table_representation with copied data
   */
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) override;

  /**
   * @brief Get the underlying cuDF table view
   *
   * @return cudf::table_view A view of the cuDF table
   */
  cudf::table_view get_table_view() const;

  /**
   * @brief Release ownership of the underlying cuDF table
   *
   * After calling this method, this representation no longer owns the table.
   *
   * @param stream CUDA stream (used to materialize the table from a view path before release)
   * @return std::unique_ptr<cudf::table> The cuDF table
   */
  std::unique_ptr<cudf::table> release_table(rmm::cuda_stream_view stream);

  /**
   * @brief Record a CUDA event on @p writer_stream and store it as the writer event.
   *
   * STREAM-LINEAGE: any cross-stream / cross-device reader of this representation's
   * memory must wait on this event before issuing reads. Concretely,
   * representation_converters.cpp's convert_gpu_to_gpu() will call
   * cudaStreamWaitEvent(reader_stream, get_writer_event(), 0) before peer-copying
   * source buffers.
   *
   * Calling this multiple times overwrites the previously recorded event (the
   * representation owns a single writer event handle that is reused). Passing a
   * default-constructed cuda_stream_view records no event and clears any prior one.
   *
   * @param writer_stream The stream on which the most recent writes to this
   *                      representation's memory were enqueued.
   */
  void record_writer_event(rmm::cuda_stream_view writer_stream) override;

  /**
   * @brief Get the writer event recorded by record_writer_event(), or nullptr if none.
   *
   * Readers that cross stream / device boundaries must call cudaStreamWaitEvent on
   * this event (when non-null) before reading the underlying memory. When this
   * returns nullptr, callers should fall back to a coarser sync (e.g.
   * cudaDeviceSynchronize on the source device) — this is the legacy behavior
   * preserved for representations constructed by code paths that have not yet
   * been migrated to record writer events.
   *
   * @return cudaEvent_t The writer event, or nullptr if none has been recorded.
   */
  [[nodiscard]] cudaEvent_t get_writer_event() const override;

 private:
  struct owning_table_view {
    std::any owner;  ///< The owner of the cuDF table
    std::size_t alloc_size{0};
    cudf::table_view view;  ///< A view of the owned table for easy access
  };

  std::variant<std::unique_ptr<cudf::table>, owning_table_view>
    _table;  ///< cudf::table is the underlying representation of the data

  /// Lazily-created CUDA event recording the completion of the most recent
  /// writer-stream work that produced this representation. Null until the first
  /// call to record_writer_event().
  cudaEvent_t _writer_event{nullptr};
};

template <typename Owner>
gpu_table_representation::gpu_table_representation(cudf::table_view table_view,
                                                   Owner&& owner,
                                                   std::size_t alloc_size,
                                                   cucascade::memory::memory_space& memory_space,
                                                   rmm::cuda_stream_view writer_stream)
  : idata_representation(memory_space),
    _table(
      owning_table_view{std::make_any<Owner>(std::forward<Owner>(owner)), alloc_size, table_view})
{
  // STREAM-LINEAGE: record writer event so cross-stream/cross-device readers
  // can establish ordering via cudaStreamWaitEvent.
  if (writer_stream.value() != nullptr) { record_writer_event(writer_stream); }
}

}  // namespace sirius
