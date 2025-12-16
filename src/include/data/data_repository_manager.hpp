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

#include "data/data_repository.hpp"
#include "data_batch.hpp"
#include "gpu_physical_operator.hpp"
#include "helper/helper.hpp"

#include <map>
#include <string>

namespace sirius {

/**
 * @brief Key type for identifying a unique operator-port combination.
 *
 * Uses std::string instead of string_view to ensure the key owns its data,
 * avoiding lifetime issues and allowing use with standard comparison.
 */
struct operator_port_key {
  ::duckdb::GPUPhysicalOperator* op;
  std::string port_id;

  bool operator==(const operator_port_key& other) const
  {
    return op == other.op && port_id == other.port_id;
  }

  bool operator<(const operator_port_key& other) const
  {
    if (op != other.op) return op < other.op;
    return port_id < other.port_id;
  }
};

/**
 * @brief Central manager for coordinating data repositories across multiple pipelines.
 *
 * data_repository_manager serves as the top-level coordinator for data management in the
 * Sirius system. It maintains a collection of idata_repository instances, each associated
 * with a specific pipeline, and provides centralized services for:
 *
 * - Repository lifecycle management (creation, access, cleanup)
 * - Cross-pipeline data batch coordination
 * - Unique batch ID generation
 * - Global eviction and memory management policies
 *
 * Architecture:
 * ```
 * data_repository_manager
 * ├── Pipeline 1 → idata_repository (FIFO/LRU/Priority)
 * ├── Pipeline 2 → idata_repository (FIFO/LRU/Priority)
 * └── Pipeline N → idata_repository (FIFO/LRU/Priority)
 * ```
 *
 * The manager abstracts the complexity of multi-pipeline data management and provides
 * a unified interface for higher-level components like the GPU executor and memory manager.
 *
 * @note All operations are thread-safe and can be called concurrently from multiple
 *       pipeline execution threads.
 */
class data_repository_manager {
  // Friend declaration to allow data_batch_view destructor to call delete_data_batch
  friend class data_batch_view;

 public:
  /**
   * @brief Default constructor - initializes empty repository manager.
   */
  data_repository_manager() = default;

  /**
   * @brief Register a new data repository for the specified pipeline.
   *
   * Associates a data repository implementation with a pipeline ID. Each pipeline
   * can have exactly one repository, and attempting to add a repository for an
   * existing pipeline will replace the previous one.
   *
   * @param op The GPUPhysicalOperator associated with the repository
   * @param repository Unique pointer to the repository implementation (ownership transferred)
   *
   * @note Thread-safe operation
   */
  void add_new_repository(::duckdb::GPUPhysicalOperator* op,
                          std::string_view port_id,
                          sirius::unique_ptr<idata_repository> repository);

  /**
   * @brief Add a new data_batch to the holder.
   *
   * This method stores the actual data_batch object in the manager's holder.
   * data_batch_views reference these batches.
   *
   * @param batch The data_batch to add (ownership transferred)
   * @param ops The GPUPhysicalOperators whose repositories will receive views of this batch
   *
   * @note Thread-safe operation
   */
  void add_new_data_batch(
    sirius::unique_ptr<data_batch> batch,
    sirius::vector<std::pair<::duckdb::GPUPhysicalOperator*, std::string_view>> ops);

  /**
   * @brief Get direct access to a pipeline's repository for advanced operations.
   *
   * Provides direct access to the underlying repository implementation, allowing
   * for repository-specific operations that aren't covered by the common interface.
   *
   * @param op the GPUPhysicalOperator whose repository is requested
   * @return sirius::unique_ptr<idata_repository>& Reference to the repository
   *
   * @throws std::out_of_range If no repository exists for the specified pipeline
   * @note Thread-safe for read access, but modifications should use the repository's own thread
   * safety
   */
  sirius::unique_ptr<idata_repository>& get_repository(::duckdb::GPUPhysicalOperator* op,
                                                       std::string_view port_id);

  /**
   * @brief Generate a globally unique data batch identifier.
   *
   * Returns a monotonically increasing ID that's unique across all pipelines
   * and repositories managed by this instance. Used to ensure data batches
   * can be uniquely identified for debugging, tracking, and cross-reference purposes.
   *
   * @return uint64_t A unique batch ID
   *
   * @note Thread-safe atomic operation with no contention
   */
  uint64_t get_next_data_batch_id();

  /**
   * @brief Get N batches from the manager where those batches reside in this memory space provided
   * until the amount to downgrade is reached or there are no more batches in that memory space to
   * downgrade.
   *
   * @param memory_space_id The memory space id to get the data batches from
   * @param amount_to_downgrade The amount of data in bytes to downgrade
   * @return std::vector<std::unique_ptr<data_batch>> A vector of data batches that are to be
   * downgraded
   */
  std::vector<std::unique_ptr<data_batch>> get_data_batches_for_downgrade(
    sirius::memory::memory_space_id memory_space_id, size_t amount_to_downgrade);

 private:
  /**
   * @brief Delete a data batch from the manager.
   *
   * This method is private and can only be called by data_batch_view destructor
   * when the last view to a batch is destroyed. This enforces proper lifecycle
   * management through RAII and reference counting.
   *
   * @param batch_id The ID of the data batch to delete
   *
   * @note Thread-safe operation
   * @note Private method - only accessible via friend class data_batch_view
   */
  void delete_data_batch(size_t batch_id);

  mutex _mutex;  ///< Mutex for thread-safe access to holder
  sirius::atomic<uint64_t> _next_data_batch_id =
    0;  ///< Atomic counter for generating unique data batch identifiers
  std::map<operator_port_key, sirius::unique_ptr<idata_repository>>
    _repositories;  ///< Map of pipeline ID to idata_repository (uses std::map for O(log n) lookups
                    ///< without needing a hash function)
  unordered_map<size_t, sirius::unique_ptr<data_batch>> _data_batches;
};

}  // namespace sirius
