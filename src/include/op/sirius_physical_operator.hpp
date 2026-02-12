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
#include "duckdb/common/enums/order_preservation_type.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/physical_operator_states.hpp"
#include "duckdb/optimizer/join_order/join_node.hpp"
#include "helper/types.hpp"
#include "op/sirius_physical_operator_type.hpp"

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>

#include <atomic>
#include <optional>
#include <string_view>

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
 * @brief Container for operator data batches.
 *
 * Wraps a collection of data batches that can be passed between operators.
 */
class operator_data {
 public:
  operator_data() = default;
  explicit operator_data(std::vector<std::shared_ptr<::cucascade::data_batch>> data_batches)
    : _data_batches(std::move(data_batches))
  {
  }

  virtual ~operator_data() = default;

  /**
   * @brief Get mutable data batches.
   * @return Mutable reference to vector of data batch pointers
   */
  [[nodiscard]] const std::vector<std::shared_ptr<::cucascade::data_batch>>& get_data_batches()
    const
  {
    return _data_batches;
  }

 private:
  std::vector<std::shared_ptr<::cucascade::data_batch>> _data_batches;
};

/**
 * @brief Container for partitioned operator data.
 *
 * Extends operator_data to include partition index information.
 */
class partitioned_operator_data : public operator_data {
 public:
  partitioned_operator_data() = default;
  partitioned_operator_data(std::vector<std::shared_ptr<::cucascade::data_batch>> data_batches,
                            std::size_t partition_idx)
    : operator_data(std::move(data_batches)), _partition_idx(partition_idx)
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

//! sirius_physical_operator is the base class of the physical operators present in the
//! execution plan
class sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::INVALID;
  //! Static counter for generating unique operator IDs
  static inline std::atomic<size_t> next_operator_id{0};

 public:
  sirius_physical_operator(SiriusPhysicalOperatorType type,
                           duckdb::vector<duckdb::LogicalType> types,
                           duckdb::idx_t estimated_cardinality)
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
  duckdb::vector<duckdb::LogicalType> types;
  //! The estimated cardinality of this physical operator
  duckdb::idx_t estimated_cardinality;
  //! The unique ID of this operator (auto-incremented at creation)
  size_t operator_id;

  //! The global sink state of this operator
  duckdb::unique_ptr<duckdb::GlobalSinkState> sink_state;
  //! The global state of this operator
  duckdb::unique_ptr<duckdb::GlobalOperatorState> op_state;
  //! Lock for (re)setting any of the operator states
  std::mutex lock;

 public:
  virtual std::string get_name() const;

  virtual std::string params_to_string() const { return ""; }

  virtual std::string to_string() const;

  void print() const;

  virtual duckdb::vector<duckdb::const_reference<sirius_physical_operator>> get_children() const;

  //! Return a vector of the types that will be returned by this operator
  const duckdb::vector<duckdb::LogicalType>& get_types() const { return types; }

  //! Get the unique operator ID
  size_t get_operator_id() const { return operator_id; }

  virtual bool equals(const sirius_physical_operator& other) const { return false; }

  virtual void verify();

 public:
  // Operator interface
  virtual duckdb::unique_ptr<duckdb::OperatorState> get_operator_state(
    duckdb::ExecutionContext& context) const;

  virtual duckdb::unique_ptr<duckdb::GlobalOperatorState> get_global_operator_state(
    duckdb::ClientContext& context) const;

  virtual operator_data execute(const operator_data& input_data, rmm::cuda_stream_view stream);

  //! The influence the operator has on order (insertion order means no influence)
  virtual duckdb::OrderPreservationType operator_order() const
  {
    return duckdb::OrderPreservationType::INSERTION_ORDER;
  }

 public:
  // Source interface
  virtual duckdb::unique_ptr<duckdb::LocalSourceState> get_local_source_state(
    duckdb::ExecutionContext& context, duckdb::GlobalSourceState& gstate) const;

  virtual duckdb::unique_ptr<duckdb::GlobalSourceState> get_global_source_state(
    duckdb::ClientContext& context) const;

  virtual bool is_source() const { return false; }

  //! The type of order emitted by the operator (as a source)
  virtual duckdb::OrderPreservationType source_order() const
  {
    return duckdb::OrderPreservationType::INSERTION_ORDER;
  }

 public:
  // Sink interface
  virtual duckdb::unique_ptr<duckdb::LocalSinkState> get_local_sink_state(
    duckdb::ExecutionContext& context) const;

  virtual duckdb::unique_ptr<duckdb::GlobalSinkState> get_global_sink_state(
    duckdb::ClientContext& context) const;

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

 public:
  template <class TARGET>
  TARGET& Cast()
  {
    // TODO(amin) this is buggy code
    if (TARGET::TYPE != SiriusPhysicalOperatorType::INVALID && type != TARGET::TYPE) {
      throw duckdb::InternalException(
        "Failed to cast physical operator to type - physical operator type mismatch");
    }
    return reinterpret_cast<TARGET&>(*this);
  }

  template <class TARGET>
  const TARGET& Cast() const
  {
    if (TARGET::TYPE != SiriusPhysicalOperatorType::INVALID && type != TARGET::TYPE) {
      throw duckdb::InternalException(
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
  //! Add a next port after sink
  void add_next_port_after_sink(
    std::pair<sirius_physical_operator*, std::string_view> port_locator);
  //! Get the next ports after sink
  std::vector<std::pair<sirius_physical_operator*, std::string_view>>& get_next_port_after_sink();

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

  //! Get the input batch
  virtual std::optional<operator_data> get_next_task_input_data();
  //! Check if all ports are empty
  bool all_ports_empty();
  //! Check if the pipeline is finished
  bool check_pipeline_finished();

  //! Get pipeline
  duckdb::shared_ptr<pipeline::sirius_pipeline> get_pipeline() const noexcept;

  void set_pipeline(duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline);

 protected:
  duckdb::shared_ptr<pipeline::sirius_pipeline> _pipeline;
  //! The ports of the operator
  std::unordered_map<std::string, std::unique_ptr<port>> ports;
  //! The next operators to be executed after this operator when it is used as a sink
  std::vector<std::pair<sirius_physical_operator*, std::string_view>> next_port_after_sink;
};

}  // namespace op
}  // namespace sirius
