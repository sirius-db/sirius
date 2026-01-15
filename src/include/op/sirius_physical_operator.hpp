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
#include "duckdb/common/enums/physical_operator_type.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/physical_operator_states.hpp"
#include "duckdb/optimizer/join_order/join_node.hpp"
#include "helper/types.hpp"

#include <data/data_batch.hpp>
#include <data/data_repository.hpp>

namespace duckdb {
class GPUExecutor;
}  // namespace duckdb

namespace sirius {

namespace op {
class sirius_physical_operator;
}  // namespace op

namespace pipeline {
class sirius_pipeline;
class sirius_pipeline_build_state;
class sirius_meta_pipeline;
}  // namespace pipeline

namespace creator {
class task_creator;
using task_hint = std::variant<std::monostate,
                               op::sirius_physical_operator*,
                               duckdb::shared_ptr<pipeline::sirius_pipeline>>;
}  // namespace creator

namespace op {
enum class memory_barrier_type { PIPELINE, PARTIAL, FULL };

//! sirius_physical_operator is the base class of the physical operators present in the
//! execution plan
class sirius_physical_operator {
 public:
  static constexpr const duckdb::PhysicalOperatorType TYPE = duckdb::PhysicalOperatorType::INVALID;

 public:
  sirius_physical_operator(duckdb::PhysicalOperatorType type,
                           duckdb::vector<duckdb::LogicalType> types,
                           duckdb::idx_t estimated_cardinality)
    : type(type), types(std::move(types)), estimated_cardinality(estimated_cardinality)
  {
  }
  sirius_physical_operator() = default;

  virtual ~sirius_physical_operator() {}

  //! The physical operator type
  duckdb::PhysicalOperatorType type;
  //! The set of children of the operator
  duckdb::vector<duckdb::unique_ptr<sirius_physical_operator>> children;
  //! The types returned by this physical operator
  duckdb::vector<duckdb::LogicalType> types;
  //! The estimated cardinality of this physical operator
  duckdb::idx_t estimated_cardinality;

  //! The global sink state of this operator
  duckdb::unique_ptr<duckdb::GlobalSinkState> sink_state;
  //! The global state of this operator
  duckdb::unique_ptr<duckdb::GlobalOperatorState> op_state;
  //! Lock for (re)setting any of the operator states
  std::mutex lock;

 public:
  virtual std::string get_name() const;
  // virtual std::string params_to_string() const {
  // 	return "";
  // }
  // virtual std::string to_string() const;
  // void print() const;
  virtual duckdb::vector<duckdb::const_reference<sirius_physical_operator>> get_children() const;

  //! Return a vector of the types that will be returned by this operator
  const duckdb::vector<duckdb::LogicalType>& get_types() const { return types; }

  virtual bool equals(const sirius_physical_operator& other) const { return false; }

  virtual void verify();

 public:
  // Operator interface
  virtual duckdb::unique_ptr<duckdb::OperatorState> get_operator_state(
    duckdb::ExecutionContext& context) const;

  virtual duckdb::unique_ptr<duckdb::GlobalOperatorState> get_global_operator_state(
    duckdb::ClientContext& context) const;

  virtual ::std::vector<::std::shared_ptr<::cucascade::data_batch>> execute(
    const ::std::vector<::std::shared_ptr<::cucascade::data_batch>>& input_batches);

  virtual ::std::vector<::std::shared_ptr<::cucascade::data_batch>> sink_execute(
    const ::std::vector<::std::shared_ptr<::cucascade::data_batch>>& input_batches);

  virtual bool parallel_operator() const { return false; }

  virtual bool requires_final_execute() const { return false; }

  //! The influence the operator has on order (insertion order means no influence)
  virtual duckdb::OrderPreservationType operator_order() const
  {
    return duckdb::OrderPreservationType::INSERTION_ORDER;
  }

 public:
  virtual duckdb::unique_ptr<duckdb::LocalSourceState> get_local_source_state(
    duckdb::ExecutionContext& context, duckdb::GlobalSourceState& gstate) const;
  virtual duckdb::unique_ptr<duckdb::GlobalSourceState> get_global_source_state(
    duckdb::ClientContext& context) const;

  virtual bool is_source() const { return false; }

  virtual bool parallel_source() const { return false; }

  //! The type of order emitted by the operator (as a source)
  virtual duckdb::OrderPreservationType source_order() const
  {
    return duckdb::OrderPreservationType::INSERTION_ORDER;
  }

 public:
  virtual duckdb::unique_ptr<duckdb::LocalSinkState> get_local_sink_state(
    duckdb::ExecutionContext& context) const;

  virtual duckdb::unique_ptr<duckdb::GlobalSinkState> get_global_sink_state(
    duckdb::ClientContext& context) const;

  virtual bool is_sink() const { return false; }

  virtual bool parallel_sink() const { return false; }

  virtual bool requires_batch_index() const { return false; }

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
    if (TARGET::TYPE != duckdb::PhysicalOperatorType::INVALID && type != TARGET::TYPE) {
      throw duckdb::InternalException(
        "Failed to cast physical operator to type - physical operator type mismatch");
    }
    return reinterpret_cast<TARGET&>(*this);
  }

  template <class TARGET>
  const TARGET& Cast() const
  {
    if (TARGET::TYPE != duckdb::PhysicalOperatorType::INVALID && type != TARGET::TYPE) {
      throw duckdb::InternalException(
        "Failed to cast physical operator to type - physical operator type mismatch");
    }
    return reinterpret_cast<const TARGET&>(*this);
  }

  struct port {
    memory_barrier_type type;
    ::cucascade::shared_data_repository* repo;
    duckdb::shared_ptr<pipeline::sirius_pipeline> src_pipeline;
    duckdb::shared_ptr<pipeline::sirius_pipeline> dest_pipeline;
  };

  // source pipeline pushed to repo of the ports
  void push_data_batch(std::string_view port_id, std::shared_ptr<::cucascade::data_batch> batch);
  //! Add a port to the operator
  void add_port(std::string_view port_id, duckdb::unique_ptr<port> p);
  //! Get a port from the operator
  port* get_port(std::string_view port_id);
  //! Check if the source pipeline is finished
  bool is_source_pipeline_finished();
  //! Add a next port after sink
  void add_next_port_after_sink(
    std::pair<sirius_physical_operator*, std::string_view> port_locator);
  //! Get the next ports after sink
  duckdb::vector<std::pair<sirius_physical_operator*, std::string_view>>&
  get_next_port_after_sink();
  //! Get the next task hint
  // virtual ::sirius::creator::task_hint get_next_task_hint();
  //! Get the input batch
  std::vector<::std::shared_ptr<::cucascade::data_batch>> get_input_batch();
  //! Check if all ports are empty
  bool all_ports_empty();
  //! Check if the pipeline is finished
  bool check_pipeline_finished();
  //! Set the creator of the task
  void set_creator(::sirius::creator::task_creator* creator);

 private:
  //! The ports of the operator
  duckdb::unordered_map<std::string, duckdb::unique_ptr<port>> ports;
  //! The next operators to be executed after this operator when it is used as a sink
  duckdb::vector<std::pair<sirius_physical_operator*, std::string_view>> next_port_after_sink;
  //! The creator of the task
  ::sirius::creator::task_creator* creator;
};

}  // namespace op
}  // namespace sirius
