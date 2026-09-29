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

#include <cudf/table/table_view.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace cucascade {
class data_batch;
}  // namespace cucascade

namespace sirius::op {

class build_arrival_ledger;

/**
 * @brief The certified set of original batches that one build PARTITION received through its FULL
 * input.
 *
 * `dynamic_filter_publication_session` accumulates a Bloom filter against this inventory: every
 * listed batch must contribute exactly once before the filter can be published. Batch IDs are
 * sorted and unique and carry their exact row counts; the schema describes every batch. Only
 * `build_arrival_ledger::certify` creates an inventory, and neither construction nor inspection
 * performs GPU work.
 */
class complete_build_inventory final {
 public:
  struct column_schema {
    cudf::data_type type;
    std::vector<column_schema> children;

    [[nodiscard]] static column_schema describe(cudf::column_view const& column);
    [[nodiscard]] bool matches(cudf::column_view const& column) const noexcept;
    [[nodiscard]] bool operator==(column_schema const&) const = default;
  };

  struct batch_entry {
    std::uint64_t batch_id;
    std::uint64_t rows;

    [[nodiscard]] bool operator==(batch_entry const&) const = default;
  };

  complete_build_inventory(complete_build_inventory const&)            = delete;
  complete_build_inventory& operator=(complete_build_inventory const&) = delete;
  complete_build_inventory(complete_build_inventory&& other) noexcept;
  complete_build_inventory& operator=(complete_build_inventory&& other) noexcept;

  [[nodiscard]] bool valid() const noexcept { return _partition_count > 1 && !_batches.empty(); }
  [[nodiscard]] std::span<batch_entry const> batches() const noexcept { return _batches; }
  [[nodiscard]] std::span<column_schema const> schema() const noexcept { return _schema; }
  [[nodiscard]] bool schema_matches(cudf::table_view const& table) const noexcept;
  [[nodiscard]] std::size_t total_rows() const noexcept { return _total_rows; }
  [[nodiscard]] std::size_t partition_count() const noexcept { return _partition_count; }
  [[nodiscard]] batch_entry const* find(std::uint64_t batch_id) const noexcept;

 private:
  friend class build_arrival_ledger;

  /**
   * @brief Sorts @p batches and builds the inventory.
   *
   * @return nullopt for fewer than two partitions, no batches, duplicate IDs, or a row total that
   * overflows
   */
  [[nodiscard]] static std::optional<complete_build_inventory> try_create(
    std::vector<batch_entry> batches,
    std::vector<column_schema> schema,
    std::size_t partition_count);

  complete_build_inventory(std::vector<batch_entry> batches,
                           std::vector<column_schema> schema,
                           std::size_t total_rows,
                           std::size_t partition_count);

  std::vector<batch_entry> _batches;
  std::vector<column_schema> _schema;
  std::size_t _total_rows;
  std::size_t _partition_count;
};

/**
 * @brief Records the metadata of every batch pushed into one build PARTITION's FULL input.
 *
 * `sirius_physical_partition` records each arrival from `on_input_batch_pushed`, before the batch
 * becomes poppable, and certifies the ledger once its source pipeline has finished. The ledger
 * records `(id, rows)` and the first batch's column schema; an arrival that cannot be read without
 * blocking, is not GPU resident, or changes the schema poisons it. After `abandon()` or a failed
 * `certify()`, arrivals are ignored.
 *
 * After `certify()` succeeded, an arrival that could add keys throws `std::logic_error`: a FULL
 * barrier received a batch after its source pipeline finished. A readable GPU arrival without rows
 * cannot add keys and is ignored. The rule relies on a finished source pipeline pushing no rows.
 * One path finishes a pipeline without draining its source:
 * `sirius_pipeline::update_pipeline_status` finishes it once an operator's limit is exhausted, and
 * tasks created for it afterwards push nothing because `sirius_physical_streaming_limit::execute`
 * emits no batch after its limit is exhausted.
 */
class build_arrival_ledger final {
 public:
  /**
   * @brief The repository facts `certify()` checks the ledger against.
   */
  struct certification {
    std::size_t repository_batches;  ///< Batches currently queued in the input repository
    std::size_t partition_count;     ///< Physical partitions of the build PARTITION
  };

  /**
   * @brief Records one arriving batch. Never blocks and never touches device memory.
   *
   * @throw std::logic_error if the ledger was already certified and @p batch is not a readable GPU
   * batch without rows
   */
  void record(cucascade::data_batch& batch);

  /**
   * @brief Certifies the recorded batches as the complete build input, at most once.
   *
   * @pre No further batch can arrive: the input port is FULL and its source pipeline has finished.
   * @return The inventory iff the ledger is not poisoned and recorded exactly
   * `facts.repository_batches` batches; otherwise nullopt, after which arrivals are ignored
   */
  [[nodiscard]] std::optional<complete_build_inventory> certify(certification facts) noexcept;

  /**
   * @brief Stops recording and frees the entries; later arrivals are ignored.
   */
  void abandon() noexcept;

 private:
  enum class status : std::uint8_t { recording, poisoned, closed, certified };

  std::mutex _mutex;
  std::vector<complete_build_inventory::batch_entry> _entries;
  std::optional<std::vector<complete_build_inventory::column_schema>> _schema;
  status _status{status::recording};
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
   * @brief Bytes of one key's filter array, a multiple of 32.
   */
  std::size_t raw_bytes;
  /**
   * @brief `raw_bytes` rounded up to the allocation alignment.
   */
  std::size_t aligned_bytes;
  /**
   * @brief `aligned_bytes` summed over the active keys: the footprint on one GPU.
   */
  std::size_t arrays_bytes;
  /**
   * @brief Transfer chunk: a multiple of the allocation alignment and at most `aligned_bytes`.
   */
  std::size_t chunk_bytes;

  /**
   * @brief Number of chunks per key array.
   */
  [[nodiscard]] std::size_t chunks_per_key() const noexcept
  {
    return (raw_bytes + chunk_bytes - 1) / chunk_bytes;
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

}  // namespace detail
}  // namespace sirius::op
