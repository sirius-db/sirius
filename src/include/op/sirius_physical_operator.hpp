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

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/enums/operator_result_type.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/optimizer/join_order/join_node.hpp"
#include "helper/logical_type.hpp"
#include "helper/types.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "sirius/exception.hpp"

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>

#include <atomic>
#include <list>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace sirius {

namespace op {
class sirius_physical_operator;
}  // namespace op

namespace pipeline {
class sirius_pipeline;
class sirius_pipeline_build_state;
class sirius_meta_pipeline;
}  // namespace pipeline
namespace op {

enum class TaskCreationHint { WAITING_FOR_INPUT_DATA, READY };

enum class MemoryBarrierType { PIPELINE, PARTIAL, FULL };

struct task_creation_hint {
  TaskCreationHint hint{TaskCreationHint::WAITING_FOR_INPUT_DATA};
  sirius_physical_operator* producer{nullptr};
};

/**
 * @brief Generic base class for operator input/output data.
 *
 * This is an intentionally minimal base class with no opinion on what the
 * data should be. Derived classes define the concrete data representation.
 */
class operator_data {
 public:
  operator_data()          = default;
  virtual ~operator_data() = default;

  operator_data(const operator_data&)            = default;
  operator_data& operator=(const operator_data&) = default;
  operator_data(operator_data&&)                 = default;
  operator_data& operator=(operator_data&&)      = default;
};

/**
 * @brief Operator data carrying a collection of data batches for pipeline execution.
 *
 * This is the standard data container used by pipeline operators. It wraps a
 * vector of data batches that flow between operators in a pipeline.
 */
class pipelineable_operator_data : public operator_data {
 public:
  pipelineable_operator_data() = default;
  explicit pipelineable_operator_data(
    std::vector<std::shared_ptr<::cucascade::data_batch>> data_batches)
    : _data_batches(std::move(data_batches))
  {
  }

  /**
   * @brief Get the data batches.
   * @return Const reference to vector of data batch pointers
   */
  [[nodiscard]] const std::vector<std::shared_ptr<::cucascade::data_batch>>& get_data_batches()
    const
  {
    return _data_batches;
  }

  /**
   * @brief Move the data batches out of this container, leaving it empty.
   * @return Vector of data batch pointers (moved out).
   */
  std::vector<std::shared_ptr<::cucascade::data_batch>> release_data_batches()
  {
    return std::move(_data_batches);
  }

  /**
   * @brief Lock all data batches for processing in the requested memory space.
   *
   * Iterates over all batches and locks (or converts then locks) each one into the
   * requested memory space. Returns the processing handles that keep the batches locked
   * until they go out of scope.
   *
   * Returns std::nullopt if any batch fails to lock (triggers a retry/reschedule).
   * Propagates rmm::out_of_memory so the caller can record metrics and reschedule.
   *
   * @param requested_memory_space  Target memory space; may be nullptr to use each batch's
   *                                current space.
   * @param stream                  CUDA stream used for any data-movement kernels.
   * @return Processing handles for all batches, or std::nullopt on lock failure.
   */
  virtual std::optional<std::vector<::cucascade::read_only_data_batch>>
  prepare_for_processing(const ::cucascade::memory::memory_space* requested_memory_space,
                         rmm::cuda_stream_view stream);

 private:
  std::vector<std::shared_ptr<::cucascade::data_batch>> _data_batches;
};

/**
 * @brief Operator data with partition index for partitioned pipeline execution.
 *
 * Extends pipelineable_operator_data to include partition index information,
 * used by partition-aware operators (partition, concat, etc.).
 */
class partitioned_operator_data : public pipelineable_operator_data {
 public:
  partitioned_operator_data() = default;
  partitioned_operator_data(std::vector<std::shared_ptr<::cucascade::data_batch>> data_batches,
                            std::size_t partition_idx)
    : pipelineable_operator_data(std::move(data_batches)), _partition_idx(partition_idx)
  {
  }

  /**
   * @brief Get the partition index.
   * @return Partition index
   */
  [[nodiscard]] std::size_t get_partition_idx() const { return _partition_idx; }

 private:
  std::size_t _partition_idx = 0;
};

/**
 * @brief Operator data carrying a collection of read-only locked data batches.
 *
 * This is the RAII accessor-based sibling of pipelineable_operator_data. It stores
 * already-locked read_only_data_batch objects (from the new 3-class cucascade API),
 * ensuring that the shared lock is held for the lifetime of this object.
 *
 * Move-only — copying is not permitted because read_only_data_batch is move-only.
 */
class read_only_pipelineable_operator_data : public operator_data {
 public:
  read_only_pipelineable_operator_data() = default;
  explicit read_only_pipelineable_operator_data(
    std::vector<::cucascade::read_only_data_batch> batches)
    : _read_only_batches(std::move(batches))
  {
  }

  ~read_only_pipelineable_operator_data() override = default;

  // Move-only — read_only_data_batch is move-only
  read_only_pipelineable_operator_data(read_only_pipelineable_operator_data&&)            = default;
  read_only_pipelineable_operator_data& operator=(read_only_pipelineable_operator_data&&) = default;
  read_only_pipelineable_operator_data(const read_only_pipelineable_operator_data&)       = delete;
  read_only_pipelineable_operator_data& operator=(
    const read_only_pipelineable_operator_data&) = delete;

  /**
   * @brief Get the locked read-only data batches.
   * @return Const reference to vector of read_only_data_batch accessors
   */
  [[nodiscard]] const std::vector<::cucascade::read_only_data_batch>& get_read_only_batches()
    const
  {
    return _read_only_batches;
  }

  /**
   * @brief Move the read-only batches out of this container, leaving it empty.
   * @return Vector of read_only_data_batch accessors (moved out).
   */
  std::vector<::cucascade::read_only_data_batch> release_read_only_batches()
  {
    return std::move(_read_only_batches);
  }

 private:
  std::vector<::cucascade::read_only_data_batch> _read_only_batches;
};

/**
 * @brief Operator data with partition index for partitioned pipeline execution, using
 * read-only locked batches.
 *
 * Extends read_only_pipelineable_operator_data to include partition index information,
 * mirroring the partitioned_operator_data -> pipelineable_operator_data relationship
 * but using the new RAII read_only_data_batch accessor type.
 *
 * Move-only — read_only_data_batch is move-only.
 */
class read_only_partitioned_operator_data : public read_only_pipelineable_operator_data {
 public:
  read_only_partitioned_operator_data() = default;
  read_only_partitioned_operator_data(std::vector<::cucascade::read_only_data_batch> batches,
                                      std::size_t partition_idx)
    : read_only_pipelineable_operator_data(std::move(batches)), _partition_idx(partition_idx)
  {
  }

  ~read_only_partitioned_operator_data() override = default;

  // Move-only
  read_only_partitioned_operator_data(read_only_partitioned_operator_data&&)            = default;
  read_only_partitioned_operator_data& operator=(read_only_partitioned_operator_data&&) = default;
  read_only_partitioned_operator_data(const read_only_partitioned_operator_data&)       = delete;
  read_only_partitioned_operator_data& operator=(
    const read_only_partitioned_operator_data&) = delete;

  /**
   * @brief Get the partition index.
   * @return Partition index
   */
  [[nodiscard]] std::size_t get_partition_idx() const { return _partition_idx; }

 private:
  std::size_t _partition_idx = 0;
};

//! sirius_physical_operator is the base class of the physical operators present in the
//! execution plan
class sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::INVALID;
  //! Static counter for generating unique operator IDs
  static inline std::atomic<size_t> next_operator_id{0};

 public:
  sirius_physical_operator(SiriusPhysicalOperatorType type,
                           duckdb::vector<sirius::logical_type> types,
                           std::size_t estimated_cardinality)
    : type(type),
      types(std::move(types)),
      estimated_cardinality(estimated_cardinality),
      operator_id(next_operator_id++)
  {
  }
  sirius_physical_operator() : operator_id(next_operator_id++) {}
  virtual ~sirius_physical_operator() {}

  //! The physical operator type
  SiriusPhysicalOperatorType type;
  //! The set of children of the operator
  duckdb::vector<duckdb::unique_ptr<sirius_physical_operator>> children;
  //! The types returned by this physical operator
  duckdb::vector<sirius::logical_type> types;
  //! The estimated cardinality of this physical operator
  std::size_t estimated_cardinality;
  //! The unique ID of this operator (auto-incremented at creation)
  size_t operator_id;

  //! Lock for concurrent access to operator state
  std::mutex lock;

 public:
  virtual std::string get_name() const;

  virtual std::string params_to_string() const { return ""; }

  virtual std::string to_string() const;

  void print() const;

  virtual duckdb::vector<duckdb::const_reference<sirius_physical_operator>> get_children() const;

  //! Return a vector of the types that will be returned by this operator
  const duckdb::vector<sirius::logical_type>& get_types() const { return types; }

  //! Get the unique operator ID
  size_t get_operator_id() const { return operator_id; }

  virtual bool equals(const sirius_physical_operator& other) const { return false; }

  virtual void verify();

 public:
  // Operator interface
  virtual std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                                 rmm::cuda_stream_view stream);

  //! The influence the operator has on order (insertion order means no influence)
  virtual sirius::OrderPreservationType operator_order() const
  {
    return sirius::OrderPreservationType::INSERTION_ORDER;
  }

 public:
  // Source interface
  virtual bool is_source() const { return false; }

  //! The type of order emitted by the operator (as a source)
  virtual sirius::OrderPreservationType source_order() const
  {
    return sirius::OrderPreservationType::INSERTION_ORDER;
  }

 public:
  // Sink interface
  virtual void sink(const operator_data& input_data, rmm::cuda_stream_view stream);

  virtual bool is_sink() const { return false; }

  //! Whether or not the sink operator depends on the order of the input chunks
  //! If this is set to true, we cannot do things like caching intermediate vectors
  virtual bool sink_order_dependent() const { return false; }

 public:
  // Pipeline construction
  virtual duckdb::vector<duckdb::const_reference<sirius_physical_operator>> get_sources() const;

  //! Build the pipelines for the operator
  virtual void build_pipelines(pipeline::sirius_pipeline& current,
                               pipeline::sirius_meta_pipeline& meta_pipeline);

  //! Called when the pipeline this operator belongs to finishes. Override to release resources.
  virtual void finalize_operator() {}

 public:
  template <class TARGET>
  TARGET& Cast()
  {
    // TODO(amin) this is buggy code
    if (TARGET::TYPE != SiriusPhysicalOperatorType::INVALID && type != TARGET::TYPE) {
      throw internal_exception(
        "Failed to cast physical operator to type - physical operator type mismatch");
    }
    return reinterpret_cast<TARGET&>(*this);
  }

  template <class TARGET>
  const TARGET& Cast() const
  {
    if (TARGET::TYPE != SiriusPhysicalOperatorType::INVALID && type != TARGET::TYPE) {
      throw internal_exception(
        "Failed to cast physical operator to type - physical operator type mismatch");
    }
    return reinterpret_cast<const TARGET&>(*this);
  }

  struct port {
    MemoryBarrierType type;
    ::cucascade::shared_data_repository* repo;
    duckdb::shared_ptr<pipeline::sirius_pipeline> src_pipeline;
    duckdb::shared_ptr<pipeline::sirius_pipeline> dest_pipeline;
  };

  /// Describes a downstream operator's port to which data is pushed
  struct next_port_info {
    //! The downstream operator that receives data batches from this sink
    sirius_physical_operator* next_operator;
    //! The port name on the downstream operator to push data into
    std::string_view next_operator_port_name;
  };

  // source pipeline pushed to repo of the ports
  void push_data_batch(std::string_view port_id, std::shared_ptr<::cucascade::data_batch> batch);
  //! Add a port to the operator
  void add_port(std::string_view port_id, std::unique_ptr<port> p);
  //! Get a port from the operator
  port* get_port(std::string_view port_id);
  //! Get all ports from the operator
  std::vector<std::string_view> get_port_ids();
  //! Check if the source pipeline is finished
  bool is_source_pipeline_finished();
  //! Returns true if any FULL-barrier port has src_pipeline == src
  bool has_full_barrier_from(const pipeline::sirius_pipeline* src) const;
  //! Add a next port after sink
  void add_next_port_after_sink(next_port_info port_info);
  //! Get the next ports after sink
  std::vector<sirius_physical_operator::next_port_info>& get_next_port_after_sink();

  //! Get the next task hint
  virtual std::optional<task_creation_hint> get_next_task_hint();

  /// \brief check if there are more tasks to create
  /// \note not necessarily ready to create at the moment
  /// the function is called
  virtual bool can_create_more_tasks() const
  {
    // WSM TODO implement this
    throw std::runtime_error("can_create_more_tasks not implemented for operator " + get_name());
    return true;
  }

  /// \brief check if all tasks have been processed
  virtual bool has_processed_all_tasks() const
  {
    // WSM TODO implement this
    throw std::runtime_error("has_processed_all_tasks not implemented for operator " + get_name());
    return true;
  }

  /// \brief check if this operator has exhausted its limit, allowing the pipeline to finish early
  virtual bool is_limit_exhausted() const { return false; }

  //! Get the input batch
  virtual std::unique_ptr<operator_data> get_next_task_input_data();
  //! Check if all ports are empty
  [[nodiscard]] virtual bool all_ports_empty();
  //! Check if the pipeline is finished
  bool check_pipeline_finished();

  //! Get pipeline
  duckdb::shared_ptr<pipeline::sirius_pipeline> get_pipeline() const noexcept;

  void set_pipeline(duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline);

 protected:
  duckdb::shared_ptr<pipeline::sirius_pipeline> _pipeline;
  //! Lookup map: port name -> raw pointer into _ports_list (never owns)
  std::unordered_map<std::string, port*> ports;
  //! Ownership container for ports, kept sorted by src_pipeline->get_pipeline_id().
  //! std::list is used intentionally: its nodes have stable addresses, so raw pointers
  //! in `ports` are never invalidated by insertions.
  std::list<std::unique_ptr<port>> _ports_list;
  //! The next operators to be executed after this operator when it is used as a sink
  std::vector<sirius_physical_operator::next_port_info> next_port_after_sink;
};

}  // namespace op
}  // namespace sirius
