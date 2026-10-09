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

#include "catch.hpp"
#include "operator/operator_test_utils.hpp"
#include "telemetry/batch_telemetry.hpp"
#include "utils/sirius_test_env.hpp"
#include "utils/telemetry_utils.hpp"

#include <cucascade/data/data_repository.hpp>

TEST_CASE("ending one query preserves another query's shared-batch telemetry",
          "[batch_query_ownership]")
{
  for (auto* env : {sirius::test::g_shared_env,
                    sirius::test::g_integration_env,
                    sirius::test::g_integration_env_2gpu})
    if (env && env->is_active()) env->pause();
  auto manager   = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu      = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  auto context   = sirius::test::make_test_telemetry_context();
  auto& registry = sirius::telemetry::batch_telemetry_registry::instance();
  registry.install(context, *manager);
  struct reset {
    sirius::telemetry::batch_telemetry_registry& registry;
    ~reset() { registry.uninstall(); }
  } cleanup{registry};
  auto q1 = sirius::make_query_id(501), q2 = sirius::make_query_id(502);
  cucascade::shared_data_repository a, b;
  uuid::UUID pipeline_a{1, 1}, pipeline_b{1, 2}, task_b{2, 2};
  registry.register_consumer_port(&a, pipeline_a, {3, 1}, q1);
  registry.register_consumer_port(&b, pipeline_b, {3, 2}, q2);
  auto shared_batch = sirius::test::operator_utils::make_numeric_batch(
    *gpu, std::vector<int32_t>{1, 2, 3}, cudf::type_id::INT32);
  registry.on_published(shared_batch, &a, sirius::telemetry::batch_origin::operator_output);
  registry.on_published(shared_batch, &b, sirius::telemetry::batch_origin::operator_output);
  REQUIRE(registry.records_for_query(q1).placements == 1);
  REQUIRE(registry.records_for_query(q2).placements == 1);
  registry.on_query_end(q1);
  CHECK(registry.records_for_query(q1).ports == 0);
  CHECK(registry.records_for_query(q1).placements == 0);
  CHECK(registry.records_for_query(q2).ports == 1);
  CHECK(registry.records_for_query(q2).placements == 1);
  registry.on_packaged(shared_batch, pipeline_b, task_b, q2);
  registry.on_processing(shared_batch, task_b);
  registry.on_consumed(shared_batch->get_batch_id(), task_b);
  CHECK(registry.records_for_query(q2).placements == 0);
  registry.on_published(shared_batch, &b, sirius::telemetry::batch_origin::operator_output);
  CHECK(registry.records_for_query(q2).placements == 1);
  registry.on_query_end(q2);
  CHECK(registry.records_for_query(q2).ports == 0);
  CHECK(registry.records_for_query(q2).placements == 0);
}
