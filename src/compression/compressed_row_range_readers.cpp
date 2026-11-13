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

#include "compression/compressed_row_range_readers.hpp"

#include "compression/compressed_disk_representation.hpp"
#include "compression/compressed_payload_sources.hpp"
#include "compression/compressed_representation.hpp"

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <cucascade/error.hpp>
#include <cucascade/memory/memory_reservation.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

namespace sirius::compression {

namespace {

using cucascade::row_range_unsupported_error;

/// The payload source of a reader state, tagged with the batch it reads.
///
/// compressed_reader_state has no slot for the batch's identity, so it travels with the
/// source. `storage` addresses the batch's shared backing store (device blob, pinned blob,
/// file). The wrapped source owns a share of that store, so the address cannot be reused by
/// another batch while this state lives.
class tagged_payload_source final : public simpatico::payload_source {
 public:
  tagged_payload_source(std::shared_ptr<simpatico::payload_source> source,
                        std::type_index type,
                        const void* storage)
    : _source(std::move(source)), _type(type), _storage(storage)
  {
  }

  void read_to_device(std::uint64_t offset,
                      std::uint64_t size,
                      void* dst,
                      rmm::cuda_stream_view stream) override
  {
    _source->read_to_device(offset, size, dst, stream);
  }

  [[nodiscard]] const std::byte* device_base() const noexcept override
  {
    return _source->device_base();
  }

  [[nodiscard]] bool reads(std::type_index type, const void* storage) const noexcept
  {
    return _type == type && _storage == storage;
  }

 private:
  std::shared_ptr<simpatico::payload_source> _source;
  std::type_index _type;
  const void* _storage;
};

/// A projection or a filtered view decodes to other rows and columns than the stored chunk.
template <typename Representation>
void require_whole_chunk(const Representation& representation)
{
  if (representation.selected_indices().has_value() || representation.pushdown_scan() != nullptr ||
      representation.visibility_mask().has_mask()) {
    throw row_range_unsupported_error(
      "compressed row-range read: the batch is a projection or a filtered view of its chunk");
  }
}

const void* backing_store(const compressed_device_representation& representation)
{
  require_whole_chunk(representation);
  if (!representation.has_table()) {
    throw row_range_unsupported_error(
      "compressed row-range read: the compressed device batch has no table");
  }
  return &representation.table();
}

const void* backing_store(const compressed_host_representation& representation)
{
  require_whole_chunk(representation);
  return &representation.payload();
}

const void* backing_store(const compressed_disk_representation& representation)
{
  return &representation.path();
}

/// The state in @p slot if it reads @p representation, else a new one stored in @p slot.
template <typename Representation>
compressed_reader_state& state_for(const Representation& representation,
                                   std::unique_ptr<cucascade::row_range_reader_state>& slot)
{
  auto const type     = std::type_index(typeid(Representation));
  auto const* storage = backing_store(representation);
  if (auto* state = dynamic_cast<compressed_reader_state*>(slot.get())) {
    auto const* tagged = dynamic_cast<const tagged_payload_source*>(state->payload.get());
    if (tagged != nullptr && tagged->reads(type, storage) && state->description != nullptr) {
      return *state;
    }
  }
  auto state         = std::make_unique<compressed_reader_state>();
  state->description = describe(representation);
  state->payload =
    std::make_shared<tagged_payload_source>(make_payload_source(representation), type, storage);
  state->cursors.resize(state->description->columns.size());
  auto& ref = *state;
  slot      = std::move(state);
  return ref;
}

/// Largest row count whose bytes fit @p capacity when @p reserved_rows more rows also need room.
std::size_t rows_fitting(std::size_t capacity, std::size_t bytes_per_row, std::size_t reserved_rows)
{
  auto const rows = capacity / bytes_per_row;
  return rows > reserved_rows ? rows - reserved_rows : 0;
}

/// One requested column, validated.
struct requested_column {
  const simpatico::hpln_column_description* source{nullptr};
  simpatico::compressed_column_cursor* cursor{nullptr};
  simpatico::column_write_target target;
  bool is_string{false};
  std::size_t width{0};  ///< bytes per row; 0 for STRING
};

template <typename Representation>
cucascade::row_range_result read_rows_compressed(
  const cucascade::idata_representation& source,
  const cucascade::row_range_request& request,
  std::span<cucascade::column_destination> dst,
  ::cuda::stream_ref stream,
  cucascade::memory::reservation* scratch,
  std::unique_ptr<cucascade::row_range_reader_state>* state_slot)
{
  auto const* representation = dynamic_cast<const Representation*>(&source);
  if (representation == nullptr) {
    throw std::invalid_argument(
      "compressed row-range read: the source is not the registered representation");
  }
  if (dst.size() != request.columns.size()) {
    throw std::invalid_argument("compressed row-range read: " + std::to_string(dst.size()) +
                                " destinations for " + std::to_string(request.columns.size()) +
                                " requested columns");
  }
  auto const num_rows = static_cast<std::size_t>(representation->num_rows());
  if (request.row_begin > num_rows) {
    throw std::out_of_range("compressed row-range read: row_begin " +
                            std::to_string(request.row_begin) + " is past the source's " +
                            std::to_string(num_rows) + " rows");
  }

  // A caller that keeps no state gets one for this call only.
  std::unique_ptr<cucascade::row_range_reader_state> call_state;
  auto& state = state_for(*representation, state_slot != nullptr ? *state_slot : call_state);

  rmm::cuda_stream_view const stream_view{stream.get()};
  auto const mr = scratch != nullptr ? scratch->get_memory_resource()
                                     : rmm::mr::get_current_device_resource_ref();

  // ---- validate every requested column and bound k by the slot capacities (rule 1) ----
  constexpr auto max_size = static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max());
  std::size_t k           = std::min(request.max_rows, num_rows - request.row_begin);
  std::vector<requested_column> columns(dst.size());
  for (std::size_t i = 0; i < dst.size(); ++i) {
    auto const index  = request.columns[i];
    auto const column = "compressed row-range read: requested column " + std::to_string(i);
    if (index >= state.description->columns.size()) {
      throw std::out_of_range(column + " has index " + std::to_string(index) + ", the source has " +
                              std::to_string(state.description->columns.size()) + " columns");
    }
    auto const& description = state.description->columns[index];
    auto const& d           = dst[i];
    auto const type         = description.dtype;
    bool const is_string    = type.id() == cudf::type_id::STRING;
    if (!is_string && !cudf::is_fixed_width(type)) {
      throw row_range_unsupported_error(column + " has an unsupported type (id " +
                                        std::to_string(static_cast<std::int32_t>(type.id())) + ")");
    }
    if (d.type_id != static_cast<std::int32_t>(type.id())) {
      throw std::invalid_argument(column + ": destination type id " + std::to_string(d.type_id) +
                                  " is not the source's " +
                                  std::to_string(static_cast<std::int32_t>(type.id())));
    }
    if (static_cast<std::size_t>(description.num_rows) != num_rows) {
      throw row_range_unsupported_error(column + " has " + std::to_string(description.num_rows) +
                                        " rows, the batch " + std::to_string(num_rows));
    }
    if (d.dst_row_offset > max_size) {
      throw std::invalid_argument(column + ": dst_row_offset " + std::to_string(d.dst_row_offset) +
                                  " exceeds a column's row limit");
    }
    k = std::min(k, max_size - d.dst_row_offset);

    auto& out     = columns[i];
    out.source    = &description;
    out.is_string = is_string;
    if (is_string) {
      if (d.offsets.data == nullptr) {
        throw std::invalid_argument(column + ": STRING destination has no offsets slot");
      }
      if (d.offset_width != 4 && d.offset_width != 8) {
        throw std::invalid_argument(column + ": STRING destination offset_width " +
                                    std::to_string(d.offset_width) + " is not 4 or 8");
      }
      // k + 1 offsets are written. The chars budget is applied by the decode itself.
      k = std::min(k, rows_fitting(d.offsets.capacity_bytes, d.offset_width, d.dst_row_offset + 1));
      if (d.dst_byte_offset > d.data.capacity_bytes) { k = 0; }
    } else {
      if (d.data.data == nullptr) {
        throw std::invalid_argument(column + ": fixed-width destination has no data slot");
      }
      out.width = cudf::size_of(type);
      k         = std::min(k, rows_fitting(d.data.capacity_bytes, out.width, d.dst_row_offset));
    }
    if (d.null_mask.data != nullptr) {
      auto const mask_rows = d.null_mask.capacity_bytes / sizeof(cudf::bitmask_type) * 32;
      k = std::min(k, mask_rows > d.dst_row_offset ? mask_rows - d.dst_row_offset : 0);
    } else if (description.null_count > 0) {
      throw row_range_unsupported_error(column + " has nulls and the destination no null mask");
    }

    auto& cursor = state.cursors[index];
    if (cursor == nullptr) {
      try {
        cursor = std::make_unique<simpatico::compressed_column_cursor>(
          state.description, index, state.payload, stream_view, mr);
      } catch (const simpatico::unsupported_chunk_decode& e) {
        throw row_range_unsupported_error(column + ": " + e.what());
      }
    }
    if (!cursor->supported()) {
      throw row_range_unsupported_error(column + " is compressed with a plan that cannot be " +
                                        "decoded by row range");
    }
    out.cursor                 = cursor.get();
    out.target.data            = d.data.data;
    out.target.data_capacity   = d.data.capacity_bytes;
    out.target.null_mask       = static_cast<cudf::bitmask_type*>(d.null_mask.data);
    out.target.offsets         = d.offsets.data;
    out.target.offset_width    = d.offset_width;
    out.target.dst_row_offset  = static_cast<cudf::size_type>(d.dst_row_offset);
    out.target.dst_byte_offset = d.dst_byte_offset;
  }

  cucascade::row_range_result result;
  result.data_bytes_written.assign(dst.size(), 0);

  // ---- decode ----
  auto const chunk_rows  = static_cast<std::size_t>(simpatico::kChunkRows);
  auto const chunk_begin = static_cast<cudf::size_type>(request.row_begin / chunk_rows);
  auto const skip_head   = static_cast<cudf::size_type>(request.row_begin % chunk_rows);
  auto const decode      = [&](requested_column& column, std::size_t take) {
    auto const chunk_end =
      static_cast<cudf::size_type>((request.row_begin + take + chunk_rows - 1) / chunk_rows);
    try {
      return static_cast<std::size_t>(
        simpatico::decompress_column_chunks(*column.cursor,
                                            chunk_begin,
                                            chunk_end,
                                            skip_head,
                                            static_cast<cudf::size_type>(take),
                                            column.target,
                                            stream_view));
    } catch (const simpatico::unsupported_chunk_decode& e) {
      throw row_range_unsupported_error(std::string{"compressed row-range read: "} + e.what());
    }
  };

  // STRING columns first: only the decode knows how many rows of chars fit. A column that
  // delivers fewer rows lowers k for every column after it. The STRING columns before it have
  // then written past the final k, but their first k rows and k + 1 offsets are what a read of
  // k rows writes, so they are not decoded again.
  for (auto& column : columns) {
    if (k == 0) { break; }
    if (column.is_string) { k = std::min(k, decode(column, k)); }
  }
  if (k > 0) {
    for (std::size_t i = 0; i < columns.size(); ++i) {
      auto& column = columns[i];
      if (column.is_string) { continue; }
      if (decode(column, k) != k) {
        throw std::logic_error("compressed row-range read: a fixed-width column was cut short");
      }
      result.data_bytes_written[i] = k * column.width;
    }

    // The chars a STRING column delivered: its rebased offset at row k, read back from the
    // destination (one transfer per STRING column, one wait for all of them).
    std::vector<std::int64_t> end_offsets(columns.size(), 0);
    std::vector<std::int32_t> end_offsets32(columns.size(), 0);
    bool any_string = false;
    for (std::size_t i = 0; i < columns.size(); ++i) {
      auto const& column = columns[i];
      if (!column.is_string) { continue; }
      any_string       = true;
      auto const width = static_cast<std::size_t>(column.target.offset_width);
      auto const* from =
        static_cast<const std::byte*>(column.target.offsets) + (dst[i].dst_row_offset + k) * width;
      void* const to =
        width == 4 ? static_cast<void*>(&end_offsets32[i]) : static_cast<void*>(&end_offsets[i]);
      CUCASCADE_CUDA_TRY(cudaMemcpyAsync(to, from, width, cudaMemcpyDeviceToHost, stream.get()));
    }
    if (any_string) {
      CUCASCADE_CUDA_TRY(cudaStreamSynchronize(stream.get()));
      for (std::size_t i = 0; i < columns.size(); ++i) {
        auto const& column = columns[i];
        if (!column.is_string) { continue; }
        auto const end =
          column.target.offset_width == 4 ? std::int64_t{end_offsets32[i]} : end_offsets[i];
        result.data_bytes_written[i] =
          static_cast<std::size_t>(end) - column.target.dst_byte_offset;
      }
    }
  }

  result.rows_written     = k;
  result.source_exhausted = request.row_begin + k == num_rows;
  return result;
}

template <typename Representation>
void register_reader(cucascade::row_range_reader_registry& registry)
{
  registry.register_reader<Representation>(&read_rows_compressed<Representation>);
}

}  // namespace

void register_compressed_row_range_readers(cucascade::row_range_reader_registry& registry)
{
  register_reader<compressed_device_representation>(registry);
  register_reader<compressed_host_representation>(registry);
  register_reader<compressed_disk_representation>(registry);
}

}  // namespace sirius::compression
