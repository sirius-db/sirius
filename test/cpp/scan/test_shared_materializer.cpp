/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "io/kvikio/kvikio_context.hpp"
#include "op/scan/detail/read_retirement.hpp"
#include "op/scan/parquet_scan_core.hpp"
#include "op/scan/shared_materializer.hpp"
#include "sirius_context.hpp"
#include "util/env_guard.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/scalar/scalar.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <duckdb/common/multi_file/multi_file_reader.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/table_filter_test_utils.hpp>

using namespace sirius::op::scan;
namespace {
template <class T>
std::unique_ptr<cudf::column> upload(std::vector<T> const& values,
                                     cudf::type_id type,
                                     ::cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  auto result = cudf::make_numeric_column(
    cudf::data_type{type}, values.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  if (!values.empty()) {
    REQUIRE(cudaMemcpyAsync(result->mutable_view().template data<T>(),
                            values.data(),
                            values.size() * sizeof(T),
                            cudaMemcpyHostToDevice,
                            stream.get()) == cudaSuccess);
    stream.sync();
  }
  return result;
}
std::shared_ptr<static_predicate_catalogue const> catalogue()
{
  std::vector<sirius::op::table_filter_conjunct> terms;
  for (std::size_t i = 0; i < 2; ++i)
    terms.push_back(
      {i,
       i,
       duckdb::BoundComparisonExpression::Create(
         duckdb::ExpressionType::COMPARE_GREATERTHAN,
         duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, i),
         duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(100)))});
  return std::make_shared<static_predicate_catalogue const>(77, 1, std::move(terms));
}
struct recorder : step_observer {
  std::vector<std::size_t> terms;
  std::vector<std::string> files;
  std::vector<step_identity> identities;
  int assemblies      = 0;
  bool throw_callback = false;
  void on_visibility(step_identity const& id, applied_visibility const& binding) override
  {
    identities.push_back(id);
    files.push_back(binding.file);
    if (throw_callback) throw std::runtime_error("observer");
  }
  void on_residual(step_identity const& id, std::size_t term) override
  {
    identities.push_back(id);
    terms.push_back(term);
  }
  void on_assemble(step_identity const& id) override
  {
    identities.push_back(id);
    ++assemblies;
  }
};
struct scenario {
  prepared_split split;
  materializer_inputs inputs;
  std::shared_ptr<materialization_program> program = std::make_shared<materialization_program>();
  scenario()
  {
    auto plan              = std::make_shared<scan_plan>();
    plan->data_columns     = {{0, "a"}, {1, "b"}};
    plan->output_layout    = {{scan_plan::output_entry::DATA, 1}};
    split.plan             = plan;
    split.attempt          = 3;
    split.logical_input    = 29;
    auto domain            = std::make_shared<lineage_domain>();
    domain->files          = {{"A", 5}};
    domain->file_runs      = {{0, 0, 5}};
    split.input_domain     = domain;
    program->contract      = 77;
    program->generation    = 1;
    program->catalogue     = catalogue();
    program->predicates    = 2;
    program->input_types   = std::vector<cudf::data_type>(2, cudf::data_type{cudf::type_id::INT32});
    program->output_types  = {cudf::data_type{cudf::type_id::INT32}};
    program->input_layout  = std::make_shared<int const>(1);
    program->output_layout = std::make_shared<int const>(2);
    auto deletes           = std::make_shared<iceberg_delete_set const>(
      "A", std::make_shared<std::vector<int64_t> const>(std::initializer_list<int64_t>{2}));
    inputs.visibility = {{{"A", deletes}, deletes}};
    program->required_visibility =
      std::make_shared<visibility_bindings const>(visibility_bindings{{"A", deletes}});
    inputs.mapping = {
      program, {}, std::make_shared<mapping_certification const>(*program, plan, domain)};
  }
  decode_result decode(::cuda::stream_ref stream, rmm::device_async_resource_ref mr) const
  {
    auto a        = upload<int32_t>({1, 101, 102, 103, 104}, cudf::type_id::INT32, stream, mr);
    auto b        = upload<int32_t>({1000, 5, 200, 300, 50}, cudf::type_id::INT32, stream, mr);
    auto ordinals = upload<int64_t>({0, 1, 2, 3, 4}, cudf::type_id::INT64, stream, mr);
    cudf::numeric_scalar<int32_t> cutoff(100, true, stream, mr);
    auto mask     = cudf::binary_operation(a->view(),
                                       cutoff,
                                       cudf::binary_operator::GREATER,
                                       cudf::data_type{cudf::type_id::BOOL8},
                                       stream,
                                       mr);
    auto selected = sirius::apply_retention_mask(
      cudf::table_view{{a->view(), b->view(), ordinals->view()}}, mask->view(), stream, mr);
    stream.sync();
    auto columns = selected->release();
    decode_result result;
    result.lineage         = {row_position_state::per_row, split.input_domain};
    result.lineage.per_row = std::move(columns.back());
    columns.pop_back();
    // Re-evaluating the already enforced term would incorrectly remove every row.
    cudf::numeric_scalar<int32_t> changed(0, true, stream, mr);
    columns[0]         = cudf::make_column_from_scalar(changed, columns[0]->size(), stream, mr);
    result.batch.table = owning_table_view{std::make_unique<cudf::table>(std::move(columns))};
    scan_execution_record facts;
    facts.producer                = execution_record_producer::own_decoder;
    facts.authoritative           = true;
    facts.contract                = 77;
    facts.generation              = 1;
    facts.logical_schema_complete = true;
    facts.layout                  = program->input_layout;
    facts.layout_width            = 2;
    facts.predicates              = {{0, sirius::op::predicate_stage::enforced, {}}};
    facts.positions               = row_position_state::per_row;
    result.batch.execution        = facts;
    return result;
  }
};
}  // namespace

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Replacement decoder shares visibility residual and assembly with production",
                 "[scan][shared_materializer][integration]")
{
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  auto mr = rmm::device_async_resource_ref(space->get_default_allocator());
  scenario test;
  sirius::util::env_guard test_options("SIRIUS_ENABLE_TEST_OPTIONS", "1");
  recorder observer;
  scoped_step_observer observe(observer);
  SECTION("real remaining work") {}
  SECTION("observer exceptions cannot alter work") { observer.throw_callback = true; }
  auto output = materialize_shared(
    test.split,
    test.inputs,
    [&](auto const&, auto const&, auto const&, auto s, auto resource) {
      return test.decode(s, resource);
    },
    *space,
    stream.view(),
    mr);
  REQUIRE(output.batch.table.num_rows() == 1);
  int32_t value = 0;
  REQUIRE(cudaMemcpyAsync(&value,
                          output.batch.table.view().column(0).data<int32_t>(),
                          sizeof(value),
                          cudaMemcpyDeviceToHost,
                          stream.value()) == cudaSuccess);
  stream.synchronize();
  CHECK(value == 300);
  CHECK(output.lineage.ordinals_on(stream.view()) == std::vector<int64_t>{3});
  CHECK(observer.files == std::vector<std::string>{"A"});
  CHECK(observer.terms == std::vector<std::size_t>{1});
  CHECK(observer.assemblies == 1);
  for (auto const& id : observer.identities) {
    CHECK(id.contract == 77);
    CHECK(id.attempt == 3);
    CHECK(id.logical_input == 29);
  }
  // Feed the completed result through the same entry: no semantic step runs twice.
  auto completed = std::make_shared<decode_result>(std::move(output));
  auto again     = materialize_shared(
    test.split,
    test.inputs,
    [&](auto const&, auto const&, auto const&, auto, auto) { return std::move(*completed); },
    *space,
    stream.view(),
    mr);
  CHECK(again.batch.table.num_rows() == 1);
  CHECK(observer.files.size() == 1);
  CHECK(observer.terms.size() == 1);
  CHECK(observer.assemblies == 1);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Shared decoder refuses malformed facts before applying remaining work",
                 "[scan][shared_materializer][integration]")
{
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  auto mr = rmm::device_async_resource_ref(space->get_default_allocator());
  scenario test;
  std::function<void(decode_result&)> corrupt;
  SECTION("unknown predicate")
  {
    corrupt = [](auto& r) { r.batch.execution->predicates[0].conjunct = 9; };
  }
  SECTION("wrong same-width layout")
  {
    corrupt = [](auto& r) { r.batch.execution->layout = std::make_shared<int const>(1); };
  }
  SECTION("wrong reported width")
  {
    corrupt = [](auto& r) { r.batch.execution->layout_width = 1; };
  }
  SECTION("wrong domain identity")
  {
    corrupt = [](auto& r) {
      auto domain               = std::make_shared<lineage_domain>(*r.lineage.domain);
      domain->files[0].identity = "B";
      r.lineage.domain          = domain;
    };
  }
  SECTION("lost positions")
  {
    corrupt = [](auto& r) {
      r.lineage.state = row_position_state::lost;
      r.lineage.per_row.reset();
      r.batch.execution->positions = row_position_state::lost;
    };
  }
  SECTION("false identity")
  {
    corrupt = [](auto& r) {
      r.lineage.state = row_position_state::identity;
      r.lineage.per_row.reset();
      r.batch.execution->positions = row_position_state::identity;
    };
  }
  SECTION("wrong delete set")
  {
    corrupt = [](auto& r) {
      r.batch.execution->visibility_applied = {{"A", std::make_shared<int const>(0)}};
    };
  }
  SECTION("missing deletion")
  {
    corrupt = [](auto& r) { r.batch.execution->visibility_complete = true; };
  }
  REQUIRE_THROWS(materialize_shared(
    test.split,
    test.inputs,
    [&](auto const&, auto const&, auto const&, auto s, auto resource) {
      auto result = test.decode(s, resource);
      corrupt(result);
      return result;
    },
    *space,
    stream.view(),
    mr));
}

TEST_CASE_METHOD(
  sirius::test::GpuExecutionFixture,
  "Shared entry validates certification cancellation and empty visibility before work",
  "[scan][shared_materializer][integration]")
{
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  auto mr = rmm::device_async_resource_ref(space->get_default_allocator());
  scenario test;
  int calls        = 0;
  decode_fn decode = [&](auto const&, auto const&, auto const&, auto s, auto resource) {
    ++calls;
    return test.decode(s, resource);
  };
  SECTION("uncertified input never invokes decoder")
  {
    test.inputs.mapping.evidence.reset();
    REQUIRE_THROWS(materialize_shared(test.split, test.inputs, decode, *space, stream, mr));
    CHECK(calls == 0);
  }
  SECTION("a foreign plan cannot reuse certification")
  {
    test.split.plan = std::make_shared<scan_plan>(*test.split.plan);
    REQUIRE_THROWS(materialize_shared(test.split, test.inputs, decode, *space, stream, mr));
    CHECK(calls == 0);
  }
  SECTION("cancelled attempt does not decode")
  {
    std::stop_source stop;
    stop.request_stop();
    test.split.stop = stop.get_token();
    REQUIRE_THROWS(materialize_shared(test.split, test.inputs, decode, *space, stream, mr));
    CHECK(calls == 0);
  }
  SECTION("empty deletion requires no positions")
  {
    auto empty             = std::make_shared<iceberg_delete_set const>("A");
    test.inputs.visibility = {{{"A", empty}, empty}};
    test.program->required_visibility =
      std::make_shared<visibility_bindings const>(visibility_bindings{{"A", empty}});
    test.split.input_domain.reset();
    auto result = materialize_shared(
      test.split,
      test.inputs,
      [&](auto const&, auto const&, auto const&, auto s, auto resource) {
        auto value                       = test.decode(s, resource);
        value.lineage                    = {};
        value.batch.execution->positions = row_position_state::none;
        return value;
      },
      *space,
      stream,
      mr);
    CHECK(result.batch.table.num_rows() == 2);
    CHECK(result.lineage.state == row_position_state::none);
  }
  SECTION("allocation failure after residual submission is retired")
  {
    sirius::op::scan::detail::read_retirement_test_hooks hooks(
      sirius::op::scan::detail::read_checkpoint::PAYLOAD_GATHER);
    REQUIRE_THROWS_AS(materialize_shared(test.split, test.inputs, decode, *space, stream, mr),
                      rmm::out_of_memory);
    CHECK(hooks.checkpoint_reached);
    CHECK(cudaStreamQuery(stream.value()) == cudaSuccess);
  }
}

TEST_CASE_METHOD(
  sirius::test::GpuExecutionFixture,
  "Parquet decoder override uses certified preparation and deferred production finish",
  "[scan][shared_materializer][integration]")
{
  bool const hive = GENERATE(false, true);
  sirius::test::scratch_dir dir("decoder_override");
  std::filesystem::create_directories(dir.path() / "part=7");
  auto path = dir.file("part=7/input.parquet");
  run_ok("COPY (SELECT * FROM (VALUES (1,1000),(101,5),(102,200),(103,300),(104,50)) t(a,b)) TO " +
         dir.file_literal("part=7/input.parquet") + " (FORMAT PARQUET)");
  auto info                 = std::make_unique<parquet_ingestible_table_info>();
  info->resolved_file_paths = {path};
  info->names               = {"a", "b", "part"};
  info->returned_types.assign(3, sirius::logical_type::make(sirius::type_id::INTEGER));
  info->column_ids             = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  info->projection_ids         = {1, 0};
  info->projected_output_count = 1;
  if (hive) {
    info->column_ids.emplace_back(2);
    info->projection_ids         = {1, 2, 0};
    info->projected_output_count = 2;
    info->partition_indices      = {duckdb::HivePartitioningIndex("7", 2)};
  }
  info->table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  for (int i = 0; i < 2; ++i)
    info->table_filters->PushFilter(
      duckdb::ProjectionIndex(i),
      sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_GREATERTHAN,
                                    duckdb::Value::INTEGER(100)));
  parquet_scan_core core(std::move(info));
  auto ioctx   = std::make_shared<sirius::io::kvikio_context>();
  auto prepare = core.next_split_provider(
    [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; });
  REQUIRE(prepare);
  auto coalescer = core.create_batch_coalescer();
  auto splits    = coalescer->push(prepare({}));
  for (auto& split : coalescer->flush())
    splits.push_back(std::move(split));
  REQUIRE(splits.size() == 1);
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  int calls = 0;
  core.set_decode_override([&](prepared_split const& split,
                               materializer_inputs const& inputs,
                               auto const&,
                               auto s,
                               auto mr) {
    ++calls;
    scenario source;
    auto result                                 = source.decode(s, mr);
    result.lineage.domain                       = split.input_domain;
    result.batch.execution->contract            = inputs.mapping.program->contract;
    result.batch.execution->layout              = inputs.mapping.program->input_layout;
    result.batch.execution->visibility_complete = true;
    auto columns                                = result.batch.table.release(s, mr)->release();
    std::size_t a_slot                          = 0;
    if (split.plan->data_columns.front().name == "b") {
      std::swap(columns[0], columns[1]);
      a_slot = 1;
    }
    result.batch.table = owning_table_view{std::make_unique<cudf::table>(std::move(columns))};
    for (std::size_t id = 0; id < inputs.mapping.program->catalogue->size(); ++id) {
      auto slots = inputs.mapping.program->catalogue->columns(id);
      if (slots.size() == 1 && slots.front() == a_slot)
        result.batch.execution->predicates = {{id, sirius::op::predicate_stage::enforced, {}}};
    }
    return result;
  });
  auto batch = core.materialize_metadata_to_table(*splits.front(), *space, stream, false, nullptr);
  auto output =
    core.post_filter_and_project(std::move(batch), *space, stream, false, nullptr, nullptr, {});
  REQUIRE(output->num_rows() == 2);
  REQUIRE(output->num_columns() == (hive ? 2 : 1));
  std::vector<int32_t> values(2);
  REQUIRE(cudaMemcpyAsync(values.data(),
                          output->view().column(0).data<int32_t>(),
                          2 * sizeof(int32_t),
                          cudaMemcpyDeviceToHost,
                          stream.value()) == cudaSuccess);
  stream.synchronize();
  CHECK(values == std::vector<int32_t>{200, 300});
  CHECK(calls == 1);
  auto& physical = dynamic_cast<parquet_split_info&>(*splits.front());
  physical.set_contract_payload(physical.contract_id(), {}, {});
  REQUIRE_THROWS(core.materialize_metadata_to_table(physical, *space, stream, false, nullptr));
  CHECK(calls == 1);
}
