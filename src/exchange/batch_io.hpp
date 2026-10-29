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

#include "exchange/wire.hpp"

#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace cucascade::memory {
class memory_space;
}

namespace sirius::exchange {

/**
 * @brief A GPU batch being sent: its header and where each stream buffer lives.
 *
 * Holds the batch read-only until destroyed, so it cannot spill while frames are copied out of
 * it. Destroy it only once every copy_range() it enqueued has completed.
 */
class outgoing_batch {
 public:
  /// Describes @p batch, which must be on the GPU; a sliced table is copied first, on @p stream.
  /// Copies enqueued later on @p stream are ordered after the batch's writer.
  /// @return nullopt for a batch with no rows.
  /// @throws sirius::invalid_input_exception when the batch is not on the GPU, or has a column
  ///         neither fixed-width nor STRING.
  [[nodiscard]] static std::optional<outgoing_batch> make(
    std::shared_ptr<cucascade::data_batch> batch, rmm::cuda_stream_view stream);

  [[nodiscard]] batch_header const& header() const noexcept { return _header; }
  [[nodiscard]] std::uint64_t stream_bytes() const noexcept { return _header.stream_bytes; }

  /// Enqueues on @p stream the copies of stream range [@p begin, @p end) into device memory
  /// @p dst; padding between buffers is left as is.
  void copy_range(std::size_t begin,
                  std::size_t end,
                  void* dst,
                  rmm::cuda_stream_view stream) const;

 private:
  outgoing_batch(std::shared_ptr<cucascade::data_batch> batch,
                 cucascade::read_only_data_batch read_only);

  std::shared_ptr<cucascade::data_batch> _batch;
  cucascade::read_only_data_batch _read_only;
  std::unique_ptr<cudf::table> _unsliced;
  batch_header _header;
  std::vector<stream_buffer> _buffers;
  std::vector<void const*> _sources;
};

/**
 * @brief A received batch landing: GPU buffers within a reservation, else host blocks.
 *
 * Choosing a tier never waits: land() takes a reservation only if one is free now.
 */
class incoming_batch {
 public:
  /// Reserves room for a batch described by @p header in @p gpu, else in @p host (which may be
  /// null). Allocations are ordered on @p stream.
  /// @return nullptr when neither tier has room now.
  [[nodiscard]] static std::unique_ptr<incoming_batch> land(batch_header header,
                                                            cucascade::memory::memory_space& gpu,
                                                            cucascade::memory::memory_space* host,
                                                            rmm::cuda_stream_view stream);

  [[nodiscard]] cucascade::memory::Tier tier() const noexcept { return _tier; }
  [[nodiscard]] batch_header const& header() const noexcept { return _header; }

  /// Enqueues on @p stream the copies of stream range [@p begin, @p begin + @p size) from device
  /// memory @p src into the landing.
  /// @throws sirius::invalid_input_exception when the range is outside the stream.
  void copy_in(std::size_t begin, void const* src, std::size_t size, rmm::cuda_stream_view stream);

  /// The landed batch. Every copy_in() must have completed: a host batch is read by the CPU.
  /// The landing is empty afterwards.
  [[nodiscard]] std::shared_ptr<cucascade::data_batch> seal(rmm::cuda_stream_view stream);

 private:
  incoming_batch(batch_header header, cucascade::memory::memory_space& space);

  batch_header _header;
  std::vector<stream_buffer> _buffers;
  cucascade::memory::memory_space* _space;
  cucascade::memory::Tier _tier{cucascade::memory::Tier::GPU};
  std::vector<rmm::device_buffer> _gpu_buffers;
  cucascade::memory::fixed_size_host_memory_resource::fixed_multiple_blocks_allocation _host_blocks;
};

}  // namespace sirius::exchange
