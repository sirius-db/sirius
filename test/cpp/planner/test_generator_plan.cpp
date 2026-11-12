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

#include "catch.hpp"
#include "exec/stream_plan_bindings.hpp"
#include "helper/type_conversions.hpp"
#include "op/scan/generator/generate_function.hpp"
#include "op/scan/table_scan/bound_read_view.hpp"
#include "op/scan/table_scan/scan_contract.hpp"
#include "op/sirius_physical_generator.hpp"
#include "planner/connector_registry.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "transparent/read_view_registry.hpp"
#include "utils/sirius_test_env.hpp"

#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/planner/operator/logical_get.hpp>

#include <cstdint>
#include <string>
#include <variant>

namespace {

using sirius::op::SiriusPhysicalOperatorType;
using sirius::op::scan::generator_identity;
using sirius::op::scan::source_kind;
using sirius::op::scan::generator::column_spec;
using sirius::op::scan::generator::distribution;
using sirius::op::scan::generator::generator_spec;
using sirius::op::scan::generator::value_type;

constexpr std::uint64_t kNumBatches   = 4;
constexpr std::uint64_t kRowsPerBatch = 1000;

std::string generate_call(std::uint64_t seed = 42)
{
  return "sirius_generate(num_batches := 4, rows_per_batch := 1000, seed := " +
         std::to_string(seed) +
         ", columns := ["
         "{name: 'k1', type: 'BIGINT', dist: 'uniform', p1: 0, p2: 1000, cardinality: 0, "
         "null_fraction: 0.0}, "
         "{name: 's', type: 'VARCHAR', dist: 'uniform', p1: 8, p2: 16, cardinality: 0, "
         "null_fraction: 0.25}])";
}

generator_spec expected_spec(std::uint64_t seed = 42)
{
  generator_spec spec;
  spec.num_batches    = kNumBatches;
  spec.rows_per_batch = kRowsPerBatch;
  spec.seed           = seed;
  spec.columns = {column_spec{"k1", value_type::int64, distribution::uniform, 0, 1000, 0, 0.0},
                  column_spec{"s", value_type::varchar, distribution::uniform, 8, 16, 0, 0.25}};
  return spec;
}

struct plan_test_generator : sirius::planner::sirius_physical_plan_generator {
  using sirius_physical_plan_generator::create_plan;
  using sirius_physical_plan_generator::read_views;
  using sirius_physical_plan_generator::sirius_physical_plan_generator;

  // The boundary create_plan(unique_ptr) stops at before lineage and scan lowering.
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> plan_tree(duckdb::LogicalOperator& root)
  {
    root.ResolveOperatorTypes();
    duckdb::ColumnBindingResolver resolver;
    resolver.VisitOperator(root);
    return create_plan(root);
  }
};

// Plans without executing: gpu_execution is off, so the connection only binds and optimizes.
struct generator_plan_fixture {
  duckdb::Connection con;

  generator_plan_fixture() : con(shared_connection())
  {
    REQUIRE_FALSE(con.Query("SET gpu_execution=false")->HasError());
    REQUIRE_FALSE(con.Query("BEGIN")->HasError());
  }
  ~generator_plan_fixture() { con.Query("ROLLBACK"); }

  duckdb::unique_ptr<duckdb::LogicalOperator> logical(std::string const& sql)
  {
    auto plan = con.ExtractPlan(sql);
    REQUIRE(plan);
    return plan;
  }

 private:
  static duckdb::Connection shared_connection()
  {
    REQUIRE(sirius::test::g_shared_env);
    return sirius::test::g_shared_env->make_connection();
  }
};

duckdb::LogicalGet& generate_get(duckdb::LogicalOperator& node)
{
  auto find = [](auto&& self, duckdb::LogicalOperator& op) -> duckdb::LogicalGet* {
    if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
      return &op.Cast<duckdb::LogicalGet>();
    }
    for (auto& child : op.children) {
      if (auto* get = self(self, *child)) { return get; }
    }
    return nullptr;
  };
  auto* get = find(find, node);
  REQUIRE(get);
  REQUIRE(get->function.GetName() == sirius::op::scan::generator::kGenerateFunctionName);
  return *get;
}

// The single leaf of a linear plan, with the operator types on the way down to it.
sirius::op::sirius_physical_operator& leaf_of(sirius::op::sirius_physical_operator& root,
                                              std::vector<SiriusPhysicalOperatorType>& path)
{
  auto* node = &root;
  while (!node->children.empty()) {
    path.push_back(node->type);
    REQUIRE(node->children.size() == 1);
    node = node->children.front().get();
  }
  return *node;
}

duckdb::unique_ptr<duckdb::FunctionData> forged_bind(duckdb::ClientContext&,
                                                     duckdb::TableFunctionBindInput&,
                                                     duckdb::vector<duckdb::LogicalType>&,
                                                     duckdb::vector<duckdb::Identifier>&)
{
  return nullptr;
}

void forged_scan(duckdb::ClientContext&, duckdb::TableFunctionInput&, duckdb::DataChunk&) {}

}  // namespace

TEST_CASE("sirius_generate plans to a GENERATOR source carrying the bound spec",
          "[planner][generator_plan][shared_context]")
{
  generator_plan_fixture fixture;
  auto logical = fixture.logical("SELECT * FROM " + generate_call());
  plan_test_generator generator(*fixture.con.context);
  auto physical = generator.plan_tree(*logical);
  REQUIRE(physical);

  std::vector<SiriusPhysicalOperatorType> path;
  auto& leaf = leaf_of(*physical, path);
  REQUIRE(leaf.type == SiriusPhysicalOperatorType::GENERATOR);
  auto const& source = leaf.Cast<sirius::op::sirius_physical_generator>();
  CHECK(source.is_source());
  CHECK(source.spec() == expected_spec());
  CHECK(source.estimated_cardinality == kNumBatches * kRowsPerBatch);
  REQUIRE(source.types.size() == 2);

  // The leaf has one contract, recorded as supported under the generator's own profile.
  REQUIRE(generator.read_views);
  REQUIRE(generator.read_views->entries().size() == 1);
  auto const& record = generator.read_views->entries().front();
  CHECK(record.contract.materializer.kind == sirius::op::scan::materializer_kind::stream);
  CHECK(record.contract.materializer.profile == "sirius.generate.v1");
  CHECK(record.eligibility.verdict == sirius::op::scan::eligibility_verdict::supported);
  REQUIRE(record.contract.view);
  REQUIRE(record.contract.view->identity);
  CHECK(record.contract.view->identity->source.kind == source_kind::generator);
}

TEST_CASE("sirius_generate keeps the projection above the GENERATOR source",
          "[planner][generator_plan][shared_context]")
{
  generator_plan_fixture fixture;
  auto logical = fixture.logical("SELECT k1 FROM " + generate_call() + " ORDER BY k1");
  // The binder lists only the referenced column; nothing is pushed into the function.
  auto const& get = generate_get(*logical);
  REQUIRE(get.GetColumnIds().size() == 1);
  CHECK(get.projection_ids.empty());

  plan_test_generator generator(*fixture.con.context);
  auto physical = generator.plan_tree(*logical);
  REQUIRE(physical);

  std::vector<SiriusPhysicalOperatorType> path;
  auto& leaf = leaf_of(*physical, path);
  REQUIRE(leaf.type == SiriusPhysicalOperatorType::GENERATOR);
  // The source still produces every column of the spec.
  auto const& source = leaf.Cast<sirius::op::sirius_physical_generator>();
  CHECK(source.spec() == expected_spec());
  CHECK(source.types.size() == 2);
  CHECK(source.estimated_cardinality == kNumBatches * kRowsPerBatch);

  REQUIRE_FALSE(path.empty());
  CHECK(path.back() == SiriusPhysicalOperatorType::PROJECTION);
  bool sorted = false;
  for (auto type : path) {
    sorted |= type == SiriusPhysicalOperatorType::ORDER_BY;
  }
  CHECK(sorted);
}

TEST_CASE("sirius_generate selects reordered and later columns above the source",
          "[planner][generator_plan][shared_context]")
{
  generator_plan_fixture fixture;
  plan_test_generator generator(*fixture.con.context);
  for (auto const* select : {"SELECT s FROM ", "SELECT s, k1 FROM "}) {
    CAPTURE(select);
    auto logical  = fixture.logical(select + generate_call());
    auto physical = generator.create_plan(generate_get(*logical));
    REQUIRE(physical);
    REQUIRE(physical->type == SiriusPhysicalOperatorType::PROJECTION);
    REQUIRE(physical->types.size() == generate_get(*logical).GetColumnIds().size());
    CHECK(physical->types.front() == sirius::from_duckdb(duckdb::LogicalType::VARCHAR));
    REQUIRE(physical->children.size() == 1);
    auto const& child = *physical->children.front();
    REQUIRE(child.type == SiriusPhysicalOperatorType::GENERATOR);
    CHECK(child.types.size() == 2);
    CHECK(child.Cast<sirius::op::sirius_physical_generator>().spec() == expected_spec());
  }
}

TEST_CASE("The connector registry admits only the trusted sirius_generate definition",
          "[planner][generator_plan][shared_context]")
{
  using sirius::planner::lookup_connector;
  using sirius::planner::lookup_connector_classified;
  using sirius::planner::lookup_decline;

  generator_plan_fixture fixture;
  auto& context = *fixture.con.context;
  auto logical  = fixture.logical("SELECT * FROM " + generate_call());
  auto& get     = generate_get(*logical);

  auto const* entry = lookup_connector(get, context);
  REQUIRE(entry);
  CHECK(entry->function_name == "sirius_generate");
  CHECK(entry->kind == source_kind::generator);
  CHECK(entry->registry_profile == "sirius.generate.v1");
  CHECK(entry->bind_data_matches(get.bind_data.get()));
  CHECK(entry->lower == nullptr);
  CHECK(entry->certify == nullptr);
  CHECK_FALSE(entry->permits_cpu_replay);

  std::size_t registered = 0;
  for (auto const& connector : sirius::planner::registered_connectors()) {
    registered += connector.function_name == "sirius_generate";
  }
  CHECK(registered == 1);

  // Same name, different callbacks: the comparison against the trusted reference refuses it.
  auto const original = get.function;
  get.function.bind   = forged_bind;
  CHECK(lookup_connector_classified(get, context).decline == lookup_decline::callback_mismatch);
  CHECK_FALSE(lookup_connector(get, context));
  get.function          = original;
  get.function.function = forged_scan;
  CHECK(lookup_connector_classified(get, context).decline == lookup_decline::callback_mismatch);
  get.function = original;
  REQUIRE(lookup_connector(get, context));

  // Same name and callbacks, another function's bind data.
  auto const generated = std::move(get.bind_data);
  get.bind_data        = duckdb::make_uniq<sirius::exec::stream_source_bind_data>(1);
  CHECK_FALSE(entry->bind_data_matches(get.bind_data.get()));
  CHECK(lookup_connector_classified(get, context).decline == lookup_decline::bind_data_mismatch);
  plan_test_generator generator(context);
  REQUIRE_THROWS_WITH(generator.create_plan(get),
                      Catch::Matchers::ContainsSubstring(
                        "Table function 'sirius_generate' is not supported in Sirius"));
  get.bind_data.reset();
  CHECK_FALSE(lookup_connector(get, context));
}

TEST_CASE("A sirius_generate read view is identified by its spec",
          "[planner][generator_plan][shared_context]")
{
  generator_plan_fixture fixture;
  auto& context = *fixture.con.context;
  auto capture  = [&](std::string const& sql) {
    auto logical = fixture.logical(sql);
    return sirius::op::scan::capture_bound_read_view(generate_get(*logical), context);
  };

  auto const view = capture("SELECT * FROM " + generate_call(42));
  REQUIRE(view.identity);
  auto const& identity = *view.identity;
  CHECK(identity.source.function_name == "sirius_generate");
  CHECK(identity.source.kind == source_kind::generator);
  CHECK(identity.source.registry_profile == "sirius.generate.v1");
  CHECK(identity.bound_names == duckdb::vector<std::string>{"k1", "s"});
  CHECK(identity.bound_types == duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT,
                                                                    duckdb::LogicalType::VARCHAR});
  REQUIRE(std::holds_alternative<generator_identity>(identity.data_view));
  auto const& generated = std::get<generator_identity>(identity.data_view);
  CHECK(generated.num_batches == kNumBatches);
  CHECK(generated.rows_per_batch == kRowsPerBatch);
  CHECK(generated.seed == 42);
  CHECK_FALSE(view.replay_policy.permits_cpu_replay);
  CHECK(view.replay_policy.reason == "generator");
  CHECK_FALSE(view.provider);

  // Same spec, bound again: the same identity.
  auto const again = capture("SELECT k1 FROM " + generate_call(42) + " ORDER BY k1");
  CHECK(again.identity->fingerprint == identity.fingerprint);
  CHECK(sirius::op::scan::canonical_read_view_text(again) ==
        sirius::op::scan::canonical_read_view_text(view));
  CHECK(std::get<generator_identity>(again.identity->data_view).spec_hash == generated.spec_hash);

  // A different seed is different data.
  auto const reseeded = capture("SELECT * FROM " + generate_call(43));
  CHECK_FALSE(reseeded.identity->fingerprint == identity.fingerprint);
  CHECK(std::get<generator_identity>(reseeded.identity->data_view).spec_hash !=
        generated.spec_hash);

  // So is a different column parameter under the same seed and batch shape.
  auto const reshaped = capture(
    "SELECT * FROM sirius_generate(num_batches := 4, rows_per_batch := 1000, seed := 42, "
    "columns := ["
    "{name: 'k1', type: 'BIGINT', dist: 'uniform', p1: 0, p2: 2000, cardinality: 0, "
    "null_fraction: 0.0}, "
    "{name: 's', type: 'VARCHAR', dist: 'uniform', p1: 8, p2: 16, cardinality: 0, "
    "null_fraction: 0.25}])");
  CHECK_FALSE(reshaped.identity->fingerprint == identity.fingerprint);
  CHECK(std::get<generator_identity>(reshaped.identity->data_view).spec_hash !=
        generated.spec_hash);
}
