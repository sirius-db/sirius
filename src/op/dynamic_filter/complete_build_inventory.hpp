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

#include "op/dynamic_filter/build_input_certification.hpp"

#include <cudf/types.hpp>

#include <cuda/cmath>

#include <cucascade/memory/common.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace cucascade {
class data_batch;
}  // namespace cucascade

namespace sirius::op {

class build_arrival_ledger;

/**
 * @brief One top-level column of one arriving build batch
 */
struct column_facts {
  cudf::data_type carrier;                ///< The type the column arrived at
  std::optional<std::size_t> null_count;  ///< Its null rows, when its representation records them

  [[nodiscard]] bool operator==(column_facts const&) const = default;
};

/**
 * @brief What `build_arrival_ledger` reads from one arriving batch without touching device data
 */
struct arrival_facts {
  std::uint64_t rows;                 ///< The batch's rows
  std::vector<column_facts> columns;  ///< Its top-level columns, by ordinal
  cucascade::memory::Tier tier;       ///< The tier the batch was read from
};

/**
 * @brief Reads @p batch's rows and per-column carriers and null counts, without waiting for its
 * data lock
 *
 * A GPU table is read through its column views, whose null counts are already on the host. A host
 * or disk table is read from its column metadata, where a negative null count means unknown.
 *
 * @return The facts, `construction_decline::compressed_arrival` for a compressed representation, or
 * `construction_decline::unreadable_arrival` for a batch that is locked, empty of data, or of any
 * other representation
 */
[[nodiscard]] std::variant<arrival_facts, construction_decline> read_arrival(
  cucascade::data_batch& batch);

/**
 * @brief What one build column looks like across every batch of an inventory
 */
struct key_column_facts {
  /** @brief Distinct carriers in batch-ID order; empty when the column is absent from some batch */
  std::vector<cudf::data_type> carriers;
  /** @brief Total null rows; empty when the column is absent or some batch's count is unknown */
  std::optional<std::size_t> null_rows;
};

/**
 * @brief What an inventory's batches look like as a whole
 */
struct build_input_facts {
  std::size_t total_rows;                 ///< Rows across every batch
  std::size_t max_batch_rows;             ///< Rows of the largest batch
  std::vector<key_column_facts> columns;  ///< By build ordinal; as wide as the widest batch
};

/**
 * @brief The certified set of original batches received through one complete build input.
 *
 * `dynamic_filter_publication_session` accumulates a Bloom filter against this inventory: every
 * listed batch must contribute exactly once before the filter can be published. Batch IDs are
 * sorted and unique and carry their exact row counts and, per top-level column, the carrier and
 * null count they arrived with. Only `build_arrival_ledger::certify` creates an inventory, and
 * neither construction nor inspection performs GPU work.
 */
class complete_build_inventory final {
 public:
  struct batch_entry {
    std::uint64_t batch_id;
    std::uint64_t rows;

    [[nodiscard]] bool operator==(batch_entry const&) const = default;
  };

  /**
   * @brief One recorded batch: its entry and its columns
   */
  struct recorded_batch {
    batch_entry entry;
    std::vector<column_facts> columns;
  };

  complete_build_inventory(complete_build_inventory const&)                = delete;
  complete_build_inventory& operator=(complete_build_inventory const&)     = delete;
  complete_build_inventory(complete_build_inventory&&) noexcept            = default;
  complete_build_inventory& operator=(complete_build_inventory&&) noexcept = default;

  [[nodiscard]] std::span<batch_entry const> batches() const noexcept { return _batches; }
  /**
   * @brief Returns the type observed at @p ordinal in every recorded batch, or no value when
   * missing or inconsistent.
   */
  [[nodiscard]] std::optional<cudf::data_type> consistent_type_at(
    std::size_t ordinal) const noexcept;
  [[nodiscard]] std::size_t total_rows() const noexcept { return _total_rows; }
  [[nodiscard]] batch_entry const* find(std::uint64_t batch_id) const noexcept;

  /**
   * @brief The columns the batch at @p batch_index of `batches()` arrived with
   */
  [[nodiscard]] std::span<column_facts const> batch_columns(std::size_t batch_index) const noexcept;

  /**
   * @brief Summarizes every column across the batches, for construction policy
   */
  [[nodiscard]] build_input_facts facts() const;

 private:
  friend class build_arrival_ledger;

  /**
   * @brief Sorts @p batches by ID and builds the inventory.
   *
   * @return nullopt for no batches, duplicate IDs, or a row total that overflows
   */
  [[nodiscard]] static std::optional<complete_build_inventory> try_create(
    std::vector<recorded_batch> batches);

  complete_build_inventory(std::vector<batch_entry> batches,
                           std::vector<std::vector<column_facts>> batch_columns,
                           std::vector<std::optional<cudf::data_type>> schema,
                           std::size_t total_rows);

  std::vector<batch_entry> _batches;
  std::vector<std::vector<column_facts>> _batch_columns;  // Parallel to _batches
  std::vector<std::optional<cudf::data_type>> _schema;    // The consistent carrier per ordinal
  std::size_t _total_rows;
};

/**
 * @brief A certified inventory, or why the input could not be certified
 */
using certified_build_input = std::variant<complete_build_inventory, construction_decline>;

/**
 * @brief Records the metadata of every batch pushed into one build publication boundary's FULL
 * input.
 *
 * PARTITION or BUILD_FILTER records each arrival from `on_input_batch_pushed`, before the batch
 * becomes poppable, and certifies the ledger once its source pipeline has finished. The ledger
 * records `(id, rows)` and each top-level column's carrier and null count (`read_arrival`) from a
 * GPU, host, or disk table. A compressed or unreadable arrival closes the ledger with that reason,
 * and it then cannot certify; a missing or changed column only makes that ordinal unavailable to
 * key admission. Payload child layouts do not affect certification. After such an arrival,
 * `abandon()`, or a failed `certify()`, arrivals are ignored.
 *
 * After `certify()` succeeded, an arrival that could add keys throws `std::logic_error`: a FULL
 * barrier received a batch after its source pipeline finished. A readable GPU, host, or disk
 * arrival without rows cannot add keys and is ignored.
 */
class build_arrival_ledger final {
 public:
  /**
   * @brief The repository facts `certify()` checks the ledger against.
   */
  struct certification {
    std::size_t repository_batches;  ///< Batches currently queued in the input repository
  };

  /**
   * @brief Records one arriving batch without waiting for its data lock or accessing device data.
   *
   * @throw std::logic_error if the ledger was already certified and @p batch is not a readable
   * batch without rows
   */
  void record(cucascade::data_batch& batch);

  /**
   * @brief Certifies the recorded batches as the complete build input, at most once.
   *
   * Transitions RECORDING to CERTIFIED on success, or to CLOSED on failure (compressed, unreadable,
   * or inconsistent batch).
   *
   * @pre No further batch can arrive: the input port is FULL and its source pipeline has finished.
   * @return The inventory iff the ledger is still recording and recorded exactly
   * `facts.repository_batches` batches; otherwise the arrival's reason that closed the ledger, or
   * `construction_decline::inventory_mismatch`. Arrivals are ignored afterwards.
   */
  [[nodiscard]] certified_build_input certify(certification facts) noexcept;

  /**
   * @brief Stops recording and frees the entries; later arrivals are ignored. Transitions RECORDING
   * to CLOSED. After this call, `certify()` returns nullopt.
   */
  void abandon() noexcept;

 private:
  /**
   * @brief State machine:
   * - RECORDING: The ledger is actively recording incoming batches.
   *   -> CLOSED: The ledger has been closed, either by abandoning or certifying.
   *   -> CERTIFIED: The ledger has been certified with a complete inventory.
   */
  enum class status : std::uint8_t { RECORDING, CLOSED, CERTIFIED };

  /**
   * @brief Stops recording and frees the entries. The caller holds `_mutex`. Transitions RECORDING
   * to CLOSED.
   */
  void close() noexcept;

  /**
   * @brief Closes the ledger because of an arrival it cannot accept. The caller holds `_mutex`.
   */
  void close(construction_decline reason) noexcept;

  std::mutex _mutex;
  std::vector<complete_build_inventory::recorded_batch> _entries;
  std::optional<construction_decline> _closed_by;  // The arrival that closed the ledger, if any
  status _status{status::RECORDING};
};

namespace detail {

/**
 * @brief The Bloom geometry shared by every key and every GPU of one accumulation.
 *
 * Every per-GPU partial array and every published replica has `raw_bytes` of storage. Publication
 * moves each array in chunks of `chunk_bytes`; the last chunk of an array may be shorter.
 */
struct accumulated_bloom_geometry {
  static constexpr std::size_t k_min_transfer_chunk_bytes = std::size_t{2} << 20;
  static constexpr std::size_t k_max_transfer_chunk_bytes = std::size_t{8} << 20;

  /**
   * @brief Bloom filter blocks per key.
   */
  std::size_t blocks;
  /**
   * @brief Bytes of one key's filter array, a whole number of blocks.
   */
  std::size_t raw_bytes;
  /**
   * @brief `raw_bytes` rounded up to the allocation alignment and summed over the active keys: the
   * footprint on one GPU.
   */
  std::size_t arrays_bytes;
  /**
   * @brief Transfer chunk: a multiple of the allocation alignment and at most `raw_bytes` rounded
   * up to that alignment.
   */
  std::size_t chunk_bytes;

  /**
   * @brief Number of chunks per key array.
   */
  [[nodiscard]] std::size_t chunks_per_key() const noexcept
  {
    return ::cuda::ceil_div(raw_bytes, chunk_bytes);
  }

  /**
   * @brief Sizes one Bloom array for @p total_rows keys.
   *
   * @return nullopt when there are no active keys, the arithmetic overflows, or `arrays_bytes`
   * exceeds @p cap
   */
  [[nodiscard]] static std::optional<accumulated_bloom_geometry> try_create(
    std::size_t total_rows, std::size_t active_keys, std::uint64_t cap) noexcept;
};

/**
 * @brief The transfer chunk for one array of @p raw_bytes: an eighth of the array, clamped to
 * `accumulated_bloom_geometry`'s chunk bounds, rounded up to the allocation alignment, and at most
 * the aligned array.
 *
 * @pre @p raw_bytes rounded up to the allocation alignment does not overflow
 */
[[nodiscard]] std::size_t accumulated_bloom_chunk_bytes(std::size_t raw_bytes) noexcept;

}  // namespace detail
}  // namespace sirius::op
