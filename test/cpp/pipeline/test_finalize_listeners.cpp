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

#include "helper/type_conversions.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <catch.hpp>
#include <duckdb.hpp>

#include <memory>
#include <vector>

using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;

namespace {

// The base operator is abstract; this adds only the order decision every concrete operator makes.
class listener_op : public sirius_physical_operator {
 public:
  using sirius_physical_operator::sirius_physical_operator;
  [[nodiscard]] sirius::OrderPreservationType output_order_preservation() const override
  {
    return sirius::OrderPreservationType::NO_ORDER;
  }
};

}  // namespace

TEST_CASE("finalize listeners run once, after the operator finalizes, when its pipeline finishes",
          "[pipeline][finalize_listener]")
{
  auto const types =
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
  listener_op producer(SiriusPhysicalOperatorType::PROJECTION, types, 0);
  auto pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(
    sirius::pipeline::pipeline_build_context{nullptr});
  sirius::pipeline::sirius_pipeline_build_state build_state;
  build_state.set_pipeline_source(*pipeline, producer);
  build_state.set_pipeline_sink(*pipeline, &producer, 0);
  build_state.add_pipeline_operator(*pipeline, producer);

  std::vector<bool> seen_finalized;
  producer.add_finalize_listener([&] { seen_finalized.push_back(producer.finalized.load()); });
  producer.add_finalize_listener([&] { seen_finalized.push_back(true); });

  pipeline->mark_task_created();
  pipeline->update_pipeline_status(false);
  // A task still runs: the pipeline is not finished, and nothing has been called.
  REQUIRE_FALSE(pipeline->is_pipeline_finished());
  CHECK(seen_finalized.empty());

  pipeline->mark_task_completed();
  pipeline->update_pipeline_status(false);
  REQUIRE(pipeline->is_pipeline_finished());
  // Both listeners ran, in order, after finalized was set.
  CHECK(seen_finalized == std::vector<bool>{true, true});

  // The pipeline finishes once, so the listeners run once.
  pipeline->update_pipeline_status(false);
  CHECK(seen_finalized.size() == 2);
}

TEST_CASE("an operator with no finalize listener finalizes as before",
          "[pipeline][finalize_listener]")
{
  auto const types =
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
  listener_op op(SiriusPhysicalOperatorType::PROJECTION, types, 0);
  REQUIRE_FALSE(op.finalized.load());
  op.finalize_operator();
  CHECK(op.finalized.load());
}
