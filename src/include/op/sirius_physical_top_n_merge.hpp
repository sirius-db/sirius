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

#include "duckdb/planner/bound_query_node.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_top_n.hpp"

namespace duckdb {
struct DynamicFilterData;
}  // namespace duckdb

namespace sirius {
namespace planner {
class sirius_physical_plan_generator;
}  // namespace planner
namespace op {

//! Represents a physical ordering of the data. Note that this will not change
//! the data but only add a selection vector.
class sirius_physical_top_n_merge : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::MERGE_TOP_N;

 public:
  sirius_physical_top_n_merge(sirius_physical_top_n* top_n);

  sirius_physical_top_n_merge(duckdb::vector<sirius::logical_type> types_p,
                              duckdb::vector<duckdb::BoundOrderByNode> orders,
                              std::size_t limit,
                              std::size_t offset,
                              duckdb::shared_ptr<duckdb::DynamicFilterData> dynamic_filter,
                              std::size_t estimated_cardinality);

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  std::size_t limit;
  std::size_t offset;
  //! Dynamic table filter (if any)
  duckdb::shared_ptr<duckdb::DynamicFilterData> dynamic_filter;

  sirius_physical_operator* child_op;
  sirius_physical_operator* get_child_op() const { return child_op; }

 public:
  bool is_source() const override { return true; }
  sirius::OrderPreservationType source_order() const override
  {
    return sirius::OrderPreservationType::FIXED_ORDER;
  }

 public:
  bool is_sink() const override { return true; }

  //! Whether this merge joins its downstream pipeline. Set only during planning (see the friend
  //! below); read-only thereafter to honor the no-mutation-after-planning contract.
  [[nodiscard]] bool fuse_into_parent() const noexcept { return _fuse_into_parent; }

  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  std::unique_ptr<operator_data> get_next_task_input_data() override;

 private:
  //! The plan-generator is the sole writer of the fusion flag, and only at plan time.
  friend class sirius::planner::sirius_physical_plan_generator;
  void set_fuse_into_parent(bool fuse) noexcept { _fuse_into_parent = fuse; }

  bool _fuse_into_parent = false;
};

}  // namespace op
}  // namespace sirius
