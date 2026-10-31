/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "io/kvikio/kvikio_context.hpp"
#include "op/scan/detail/read_retirement.hpp"
#include "op/scan/iceberg_metadata_reader.hpp"
#include "op/scan/parquet_scan_core.hpp"
#include "op/scan/shared_materializer.hpp"
#include "pipeline/completion_handler.hpp"
#include "sirius_context.hpp"
#include "util/env_guard.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/scalar/scalar.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <duckdb/common/multi_file/multi_file_reader.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/scan_resource_test_utils.hpp>
#include <utils/table_filter_test_utils.hpp>
#include <utils/tracking_kvikio_context.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <numeric>
#include <thread>

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
template <class T>
std::vector<T> download(cudf::column_view column, ::cuda::stream_ref stream)
{
  std::vector<T> result(column.size());
  if (!result.empty()) {
    REQUIRE(cudaMemcpyAsync(result.data(),
                            column.data<T>(),
                            result.size() * sizeof(T),
                            cudaMemcpyDeviceToHost,
                            stream.get()) == cudaSuccess);
    stream.sync();
  }
  return result;
}
std::shared_ptr<static_predicate_catalogue const> catalogue(bool answers = false)
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
  return std::make_shared<static_predicate_catalogue const>(
    77,
    1,
    std::move(terms),
    answers ? std::unordered_set<std::size_t>{1} : std::unordered_set<std::size_t>{});
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
    static_cast<::cuda::stream_ref>(stream),
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
  CHECK(output.lineage.ordinals_on(static_cast<::cuda::stream_ref>(stream)) ==
        std::vector<int64_t>{3});
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
    static_cast<::cuda::stream_ref>(stream),
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
  std::optional<std::string> expected_error;
  SECTION("assembled output cannot use an unfinished input mapping")
  {
    expected_error = "decoder assembled output before logical normalization";
    corrupt        = [&](auto& r) {
      r.batch.execution->projection_complete     = true;
      r.batch.execution->whole_static_complete   = true;
      r.batch.execution->logical_schema_complete = false;
      r.batch.execution->layout                  = test.program->output_layout;
    };
  }
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
  SECTION("duplicate ordinals")
  {
    corrupt = [&](auto& r) {
      r.lineage.per_row = upload<int64_t>({1, 1, 3, 4}, cudf::type_id::INT64, stream, mr);
    };
  }
  SECTION("out-of-range ordinals")
  {
    corrupt = [&](auto& r) {
      r.lineage.per_row = upload<int64_t>({1, 2, 3, 5}, cudf::type_id::INT64, stream, mr);
    };
  }
  SECTION("wrong generation")
  {
    corrupt = [](auto& r) { ++r.batch.execution->generation; };
  }
  SECTION("answer in a value column")
  {
    corrupt = [](auto& r) {
      r.batch.execution->predicates = {{1, sirius::op::predicate_stage::answer, 1}};
    };
  }
  SECTION("dropped residual input")
  {
    corrupt = [&](auto& r) {
      auto cols = r.batch.table.release(stream, mr)->release();
      cols.pop_back();
      r.batch.table = owning_table_view{std::make_unique<cudf::table>(std::move(cols))};
      r.batch.execution->layout_width = 1;
    };
  }
  SECTION("missing deletion")
  {
    corrupt = [](auto& r) { r.batch.execution->visibility_complete = true; };
  }
  auto run = [&] {
    return materialize_shared(
      test.split,
      test.inputs,
      [&](auto const&, auto const&, auto const&, auto s, auto resource) {
        auto result = test.decode(s, resource);
        corrupt(result);
        return result;
      },
      *space,
      static_cast<::cuda::stream_ref>(stream),
      mr);
  };
  if (expected_error)
    REQUIRE_THROWS_WITH(run(), Catch::Matchers::ContainsSubstring(*expected_error));
  else
    REQUIRE_THROWS(run());
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
  SECTION("a foreign generation cannot reuse certification")
  {
    ++test.program->generation;
    REQUIRE_THROWS(materialize_shared(test.split, test.inputs, decode, *space, stream, mr));
    CHECK(calls == 0);
  }
  SECTION("wrong visibility file is rejected before decode")
  {
    test.inputs.visibility.front().binding.file = "B";
    REQUIRE_THROWS(materialize_shared(test.split, test.inputs, decode, *space, stream, mr));
    CHECK(calls == 0);
  }
  SECTION("another delete set cannot reuse its binding")
  {
    test.inputs.visibility.front().owner = std::make_shared<iceberg_delete_set const>("A");
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
  SECTION("cancellation during decode retires submitted work before publishing")
  {
    test.inputs.phase = GENERATE(materialization_phase::complete, materialization_phase::decoded);
    bool const needs_mapping = GENERATE(false, true);
    CAPTURE(test.inputs.phase, needs_mapping);
    if (needs_mapping) {
      auto plan = std::make_shared<scan_plan>(*test.split.plan);
      plan->virtual_columns.push_back({999,
                                       "row",
                                       sirius::logical_type::make(sirius::type_id::BIGINT),
                                       scan_plan::parquet_virtual_column_kind::FILE_ROW_NUMBER,
                                       2});
      test.program->input_types->push_back(cudf::data_type{cudf::type_id::INT64});
      test.split.plan = plan;
      test.inputs.mapping.evidence =
        std::make_shared<mapping_certification const>(*test.program, plan, test.split.input_domain);
    }
    std::stop_source stop;
    test.split.stop = stop.get_token();
    sirius::test::tracked_scan_resource tracked{mr, stream};
    REQUIRE_THROWS_WITH(materialize_shared(
                          test.split,
                          test.inputs,
                          [&](auto const&, auto const&, auto const&, auto s, auto resource) {
                            auto decoded = test.decode(s, resource);
                            decoded.batch.execution->logical_schema_complete = !needs_mapping;
                            stop.request_stop();
                            return decoded;
                          },
                          *space,
                          stream,
                          rmm::device_async_resource_ref{tracked}),
                        Catch::Matchers::ContainsSubstring("cancelled"));
    CHECK(cudaStreamQuery(stream.value()) == cudaSuccess);
    CHECK(tracked.state->live == 0);
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
  bool const hive      = GENERATE(false, true);
  bool const positions = GENERATE(false, true);
  bool const mapped    = GENERATE(false, true);
  bool const repeated  = !hive && !mapped ? GENERATE(false, true) : false;
  sirius::test::scratch_dir dir("decoder_override");
  std::filesystem::create_directories(dir.path() / "part=7");
  auto path = dir.file("part=7/input.parquet");
  run_ok(
    "COPY (SELECT * FROM (VALUES (1,1000),(101,5),(102,200),(103,300),(104,50)) t(pa,pb)) TO " +
    dir.file_literal("part=7/input.parquet") + " (FORMAT PARQUET, FIELD_IDS {pa: 1, pb: 2})");
  auto info                 = std::make_unique<parquet_ingestible_table_info>();
  info->resolved_file_paths = {path};
  if (repeated) info->resolved_file_paths.push_back(path);
  info->names = {mapped ? "a" : "pa", mapped ? "b" : "pb", "part"};
  info->returned_types.assign(3, sirius::logical_type::make(sirius::type_id::INTEGER));
  info->column_ids             = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  info->projection_ids         = {1, 0};
  info->projected_output_count = 1;
  if (mapped) {
    info->contract_id                = 77;
    info->returned_types[0]          = sirius::logical_type::make(sirius::type_id::BIGINT);
    info->field_mapping_requirements = std::vector<field_requirement>{
      {1, duckdb::LogicalType::BIGINT, {}, true}, {2, duckdb::LogicalType::INTEGER, {}, false}};
    info->prepare_field_mapping = [](auto metadata, auto identity, auto program, auto fields) {
      return std::make_shared<field_id_mapping const>(field_id_mapping::compile(
        std::move(metadata), identity, std::move(program), std::move(fields)));
    };
  }
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
  auto ioctx    = std::make_shared<sirius::test::tracking_kvikio_context>();
  auto resolver = [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; };
  auto prepare  = core.next_split_provider(resolver);
  REQUIRE(prepare);
  auto coalescer = core.create_batch_coalescer();
  auto splits    = coalescer->push(prepare({}));
  if (repeated) {
    auto next = core.next_split_provider(resolver);
    REQUIRE(next);
    for (auto& split : coalescer->push(next({})))
      splits.push_back(std::move(split));
  }
  for (auto& split : coalescer->flush())
    splits.push_back(std::move(split));
  REQUIRE(splits.size() == 1);
  auto const prepared_reads = ioctx->counts->calls.load();
  auto context              = sirius::test::get_registered_sirius_context(*con);
  auto* space               = context->get_memory_manager().get_memory_space(
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
    ioctx->metadata_store().clear();  // Retained approvals must survive eviction during decode.
    auto result                      = source.decode(s, mr);
    result.lineage.domain            = split.input_domain;
    result.batch.execution->contract = inputs.mapping.program->contract;
    result.batch.execution->layout   = inputs.mapping.program->input_layout;
    result.batch.execution->visibility_complete =
      false;  // No file deletes: shared entry completes it.
    result.batch.execution->logical_schema_complete = !mapped;
    if (repeated) {
      REQUIRE(split.input_domain->files.size() == 1);
      REQUIRE(split.input_domain->file_runs.size() == 2);
      auto payload = result.batch.table.release(s, mr);
      auto twice =
        cudf::concatenate(std::vector<cudf::table_view>{payload->view(), payload->view()}, s, mr);
      s.sync();
      result.batch.table = owning_table_view{std::move(twice)};
      result.lineage.per_row =
        upload<int64_t>({1, 2, 3, 4, 6, 7, 8, 9}, cudf::type_id::INT64, s, mr);
    }
    if (!positions) {
      result.lineage                    = {};
      result.batch.execution->positions = row_position_state::none;
    }
    auto columns       = result.batch.table.release(s, mr)->release();
    std::size_t a_slot = 0;
    if (split.plan->data_columns.front().name == "b" ||
        split.plan->data_columns.front().name == "pb") {
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
  REQUIRE(output->num_rows() == (repeated ? 4 : 2));
  REQUIRE(output->num_columns() == (hive ? 2 : 1));
  std::vector<int32_t> values(output->num_rows());
  REQUIRE(cudaMemcpyAsync(values.data(),
                          output->view().column(0).data<int32_t>(),
                          values.size() * sizeof(int32_t),
                          cudaMemcpyDeviceToHost,
                          stream.value()) == cudaSuccess);
  stream.synchronize();
  CHECK(values ==
        (repeated ? std::vector<int32_t>{200, 300, 200, 300} : std::vector<int32_t>{200, 300}));
  CHECK(calls == 1);
  CHECK(ioctx->counts->calls.load() == prepared_reads);
  if (mapped && positions && !hive) {
    sirius::test::tracked_scan_resource tracked(space->get_default_allocator(), stream);
    cucascade::memory::memory_space counted(cucascade::memory::gpu_memory_space_config{
      .device_id       = space->get_device_id(),
      .memory_capacity = space->get_max_memory(),
      .mr_factory_fn   = [&](int, std::size_t) {
        return cuda::mr::any_resource<cuda::mr::device_accessible>{tracked};
      }});
    auto run = [&] {
      auto decoded =
        core.materialize_metadata_to_table(*splits.front(), counted, stream, false, nullptr);
      return core.post_filter_and_project(
        std::move(decoded), counted, stream, false, nullptr, nullptr, {});
    };
    {
      auto checked = run();
      CHECK(checked->num_rows() == 2);
    }
    stream.synchronize();
    auto attempts = tracked.state->attempts;
    REQUIRE(attempts > 0);
    for (std::size_t fail = 0; fail < attempts; ++fail) {
      CAPTURE(fail, attempts);
      tracked.state->attempts = 0;
      tracked.state->fail_at  = fail;
      REQUIRE_THROWS(run());
      stream.synchronize();
      CHECK(tracked.state->live == 0);
      CHECK_FALSE(tracked.state->wrong_stream);
    }
  }
  auto const completed_calls = calls;
  auto& physical             = dynamic_cast<parquet_split_info&>(*splits.front());
  if (mapped) {
    auto original = physical.rg_slices.front().file_metadata;
    physical.rg_slices.front().file_metadata =
      std::make_shared<cudf::io::parquet::FileMetaData const>(*original);
    REQUIRE_THROWS(core.materialize_metadata_to_table(physical, *space, stream, false, nullptr));
    CHECK(calls == completed_calls);
    CHECK(ioctx->counts->calls.load() == prepared_reads);
    physical.rg_slices.front().file_metadata = original;
  }
  physical.set_contract_payload(physical.contract_id(), {}, {});
  REQUIRE_THROWS(core.materialize_metadata_to_table(physical, *space, stream, false, nullptr));
  CHECK(calls == completed_calls);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Shared decoder completes visibility and predicate states independently",
                 "[scan][shared_materializer][integration]")
{
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  auto mr = rmm::device_async_resource_ref(space->get_default_allocator());
  scenario test;
  sirius::util::env_guard opt("SIRIUS_ENABLE_TEST_OPTIONS", "1");
  recorder observed;
  scoped_step_observer scope(observed);
  std::vector<int32_t> expected{300};
  int mode = 0;
  SECTION("no visibility consumer and no lineage")
  {
    mode     = 1;
    expected = {200, 300};
    test.inputs.visibility.clear();
    test.program->required_visibility.reset();
  }
  SECTION("zero rows need no lineage")
  {
    mode = 2;
    expected.clear();
  }
  SECTION("all rows deleted")
  {
    expected.clear();
    auto all = std::make_shared<iceberg_delete_set const>(
      "A",
      std::make_shared<std::vector<int64_t> const>(std::initializer_list<int64_t>{1, 2, 3, 4}));
    test.inputs.visibility = {{{"A", all}, all}};
    test.program->required_visibility =
      std::make_shared<visibility_bindings const>(visibility_bindings{{"A", all}});
  }
  SECTION("answered condition is evaluated from BOOL8")
  {
    mode                    = 3;
    test.program->catalogue = catalogue(true);
  }
  SECTION("an enforced answer must not be evaluated twice")
  {
    mode                    = 4;
    test.program->catalogue = catalogue(true);
  }
  SECTION("pruning is not static enforcement")
  {
    mode = 5;
    expected.clear();
  }
  SECTION("requested coverage is not static enforcement")
  {
    mode = 6;
    expected.clear();
    test.inputs.coverage = {test.program->catalogue, {0, 1}, true};
  }
  if (mode == 3 || mode == 4) {
    auto plan           = std::make_shared<scan_plan>(*test.split.plan);
    plan->output_layout = {{scan_plan::output_entry::DATA, 0}};
    test.split.plan     = plan;
    test.inputs.mapping.evidence =
      std::make_shared<mapping_certification const>(*test.program, plan, test.split.input_domain);
  }
  auto result = materialize_shared(
    test.split,
    test.inputs,
    [&](auto const&, auto const&, auto const&, auto s, auto resource) {
      auto decoded = test.decode(s, resource);
      if (mode == 1 || mode == 2) {
        decoded.lineage                    = {};
        decoded.batch.execution->positions = row_position_state::none;
      }
      if (mode == 2) {
        std::vector<std::unique_ptr<cudf::column>> columns;
        for (int i = 0; i < 2; ++i)
          columns.push_back(cudf::make_empty_column(cudf::type_id::INT32));
        decoded.batch.table = owning_table_view{std::make_unique<cudf::table>(std::move(columns))};
      }
      if (mode == 3 || mode == 4) {
        auto columns = decoded.batch.table.release(s, resource)->release();
        cudf::numeric_scalar<int32_t> cutoff(100, true, s, resource);
        auto answer = cudf::binary_operation(columns[1]->view(),
                                             cutoff,
                                             cudf::binary_operator::GREATER,
                                             cudf::data_type{cudf::type_id::BOOL8},
                                             s,
                                             resource);
        if (mode == 4) {
          std::vector<cudf::column_view> views{
            columns[0]->view(), answer->view(), decoded.lineage.per_row->view()};
          auto selected =
            sirius::apply_retention_mask(cudf::table_view{views}, answer->view(), s, resource);
          s.sync();
          columns                 = selected->release();
          decoded.lineage.per_row = std::move(columns.back());
          columns.pop_back();
          // Completed BOOL8 answers are kept for layout validation but are never re-applied.
          columns[1] = upload<int8_t>({0, 0}, cudf::type_id::BOOL8, s, resource);
        } else
          columns[1] = std::move(answer);
        decoded.batch.table = owning_table_view{std::make_unique<cudf::table>(std::move(columns))};
        decoded.batch.execution->predicates.push_back(
          {1,
           mode == 3 ? sirius::op::predicate_stage::answer : sirius::op::predicate_stage::enforced,
           1});
      }
      if (mode == 5)
        decoded.batch.execution->predicates[0].stage = sirius::op::predicate_stage::pruning_only;
      if (mode == 6) decoded.batch.execution->predicates.clear();
      return decoded;
    },
    *space,
    stream,
    mr);
  if (mode == 3 || mode == 4) expected = {0};
  CHECK(download<int32_t>(result.batch.table.view().column(0), stream) == expected);
  CHECK(result.batch.execution->visibility_complete);
  CHECK(result.batch.execution->whole_static_complete);
  CHECK(result.batch.execution->projection_complete);
  CHECK(observed.assemblies == 1);
  if (mode == 4)
    CHECK(observed.terms.empty());
  else if (mode != 5 && mode != 6)
    CHECK(observed.terms == std::vector<std::size_t>{1});
  else
    CHECK(observed.terms == std::vector<std::size_t>{0, 1});
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Shared visibility skips an applied file and deletes remaining file rows",
                 "[scan][shared_materializer][integration]")
{
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  auto mr = rmm::device_async_resource_ref(space->get_default_allocator());
  scenario test;
  auto domain = std::make_shared<lineage_domain>(*test.split.input_domain);
  domain->files.push_back({"B", 2});
  domain->file_runs.push_back({1, 0, 2});
  test.split.input_domain = domain;
  auto second             = std::make_shared<iceberg_delete_set const>(
    "B", std::make_shared<std::vector<int64_t> const>(std::initializer_list<int64_t>{0}));
  test.inputs.visibility.push_back({{"B", second}, second});
  auto required = *test.program->required_visibility;
  required.push_back({"B", second});
  test.program->required_visibility = std::make_shared<visibility_bindings const>(required);
  sirius::util::env_guard opt("SIRIUS_ENABLE_TEST_OPTIONS", "1");
  recorder observed;
  scoped_step_observer scope(observed);
  auto result = materialize_shared(
    test.split,
    test.inputs,
    [&](auto const&, auto const&, auto const&, auto s, auto resource) {
      auto decoded = test.decode(s, resource);
      std::vector<std::unique_ptr<cudf::column>> columns;
      columns.push_back(upload<int32_t>({0, 0, 0, 0, 0, 0}, cudf::type_id::INT32, s, resource));
      columns.push_back(
        upload<int32_t>({5, 200, 300, 50, 500, 600}, cudf::type_id::INT32, s, resource));
      decoded.batch.table = owning_table_view{std::make_unique<cudf::table>(std::move(columns))};
      decoded.lineage.per_row =
        upload<int64_t>({1, 2, 3, 4, 5, 6}, cudf::type_id::INT64, s, resource);
      apply_lineage_visibility(
        decoded.batch.table,
        decoded.lineage,
        [&](auto const& file) -> std::span<int64_t const> {
          return file == "A" ? test.inputs.visibility[0].owner->positions
                             : std::span<int64_t const>{};
        },
        s,
        resource);
      decoded.batch.execution->visibility_applied = {required[0]};
      return decoded;
    },
    *space,
    stream,
    mr);
  CHECK(download<int32_t>(result.batch.table.view().column(0), stream) ==
        std::vector<int32_t>{300, 600});
  CHECK(result.lineage.ordinals_on(stream) == std::vector<int64_t>{3, 6});
  CHECK(observed.files == std::vector<std::string>{"B"});
  CHECK(observed.terms == std::vector<std::size_t>{1});
  CHECK(result.batch.execution->visibility_applied.size() == 2);
}

TEST_CASE("Scan observers preserve nested scopes and wait for in-flight callbacks",
          "[scan][shared_observer]")
{
  sirius::util::env_guard opt("SIRIUS_ENABLE_TEST_OPTIONS", "1");
  auto owner = std::make_shared<int const>(1);
  visibility_bindings required{{"A", owner}};
  auto apply = [&](uint64_t attempt, uint64_t input) {
    visibility_bindings applied;
    bool complete = false;
    apply_materialization_visibility(required, applied, complete, [](auto) {}, 77, attempt, input);
    REQUIRE(complete);
    REQUIRE(applied.size() == 1);
  };
  SECTION("non-LIFO removal does not restore a dead scope")
  {
    recorder first, second;
    auto a = std::make_unique<scoped_step_observer>(first);
    apply(1, 11);
    auto b = std::make_unique<scoped_step_observer>(second);
    apply(2, 12);
    a.reset();
    apply(3, 13);
    b.reset();
    apply(4, 14);
    CHECK(first.files.size() == 1);
    CHECK(second.files.size() == 2);
    CHECK(second.identities[0].attempt == 2);
    CHECK(second.identities[1].logical_input == 13);
  }
  SECTION("removal waits for a callback on another thread")
  {
    struct blocking : step_observer {
      std::promise<void> entered;
      std::shared_future<void> release;
      explicit blocking(std::shared_future<void> release) : release(std::move(release)) {}
      void on_visibility(step_identity const&, applied_visibility const&) override
      {
        entered.set_value();
        release.wait();
      }
    };
    std::promise<void> release;
    blocking observer(release.get_future().share());
    auto scope   = std::make_unique<scoped_step_observer>(observer);
    auto entered = observer.entered.get_future();
    std::atomic<bool> worker_ok{false};
    std::thread worker([&] {
      visibility_bindings applied;
      bool complete = false;
      apply_materialization_visibility(required, applied, complete, [](auto) {}, 77, 1, 1);
      worker_ok = complete && applied.size() == 1;
    });
    entered.wait();
    auto removal = std::async(std::launch::async, [&] { scope.reset(); });
    CHECK(removal.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
    release.set_value();
    worker.join();
    removal.get();
    CHECK(worker_ok.load());
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Shared materialization retires every failing allocation on its task stream",
                 "[scan][shared_materializer][integration]")
{
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  sirius::test::tracked_scan_resource resource(space->get_default_allocator(), stream);
  rmm::device_async_resource_ref mr{resource};
  scenario test;
  auto plan           = std::make_shared<scan_plan>(*test.split.plan);
  bool const virtuals = GENERATE(false, true);
  if (virtuals) {
    plan->virtual_columns.push_back({999,
                                     "row",
                                     sirius::logical_type::make(sirius::type_id::BIGINT),
                                     scan_plan::parquet_virtual_column_kind::FILE_ROW_NUMBER,
                                     2});
    test.program->input_types->push_back(cudf::data_type{cudf::type_id::INT64});
  }
  plan->output_layout = {{scan_plan::output_entry::DATA, 1}, {scan_plan::output_entry::DATA, 1}};
  test.split.plan     = plan;
  test.program->output_types.push_back(cudf::data_type{cudf::type_id::INT32});
  test.inputs.mapping.evidence =
    std::make_shared<mapping_certification const>(*test.program, plan, test.split.input_domain);
  auto run = [&] {
    return materialize_shared(
      test.split,
      test.inputs,
      [&](auto const&, auto const&, auto const&, auto s, auto memory) {
        auto decoded                                     = test.decode(s, memory);
        decoded.batch.execution->logical_schema_complete = !virtuals;
        return decoded;
      },
      *space,
      stream,
      mr);
  };
  {
    auto result = run();
    CHECK(result.batch.table.num_rows() == 1);
  }
  stream.synchronize();
  auto const attempts = resource.state->attempts;
  REQUIRE(attempts > 0);
  CHECK(resource.state->live == 0);
  CHECK_FALSE(resource.state->wrong_stream);
  // Includes decoder buffers in addition to the shared workspace estimate.
  CHECK(resource.state->peak <=
        shared_materialization_workspace_bytes(5, 5 * (virtuals ? 16 : 8), 1, 1) +
          5 * 2 * sizeof(int32_t));
  for (std::size_t fail = 0; fail < attempts; ++fail) {
    CAPTURE(fail, attempts);
    resource.state->attempts = 0;
    resource.state->fail_at  = fail;
    REQUIRE_THROWS(run());
    stream.synchronize();
    CHECK(resource.state->live == 0);
    CHECK_FALSE(resource.state->wrong_stream);
  }
  resource.state->attempts = 0;
  resource.state->fail_at  = std::numeric_limits<std::size_t>::max();
  {
    sirius::op::scan::detail::read_retirement_test_hooks hooks(
      sirius::op::scan::detail::read_checkpoint::BATCH, true);
    REQUIRE_THROWS_AS(run(), sirius::unjoined_gpu_work);
    CHECK(hooks.retained.has_value());
    CHECK(resource.state->live > 0);
    stream.synchronize();
  }
  CHECK(resource.state->live == 0);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Shared decoder synthesizes virtual predicate inputs before evaluation",
                 "[scan][shared_materializer][integration]")
{
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  auto mr = rmm::device_async_resource_ref(space->get_default_allocator());
  scenario test;
  auto plan = std::make_shared<scan_plan>(*test.split.plan);
  plan->virtual_columns.push_back({999,
                                   "row",
                                   sirius::logical_type::make(sirius::type_id::BIGINT),
                                   scan_plan::parquet_virtual_column_kind::FILE_ROW_NUMBER,
                                   2});
  plan->output_layout = {{scan_plan::output_entry::DATA, 2}, {scan_plan::output_entry::DATA, 1}};
  test.split.plan     = plan;
  test.program->input_types->push_back(cudf::data_type{cudf::type_id::INT64});
  test.program->output_types = {cudf::data_type{cudf::type_id::INT64},
                                cudf::data_type{cudf::type_id::INT32}};
  std::vector<sirius::op::table_filter_conjunct> terms;
  for (std::size_t id = 0; id < 3; ++id) {
    auto const slot = id == 1 ? 2 : id == 2 ? 1 : 0;
    auto const type = id == 1 ? duckdb::LogicalType::BIGINT : duckdb::LogicalType::INTEGER;
    auto value      = id == 1 ? duckdb::Value::BIGINT(2) : duckdb::Value::INTEGER(100);
    terms.push_back({id,
                     static_cast<std::size_t>(slot),
                     duckdb::BoundComparisonExpression::Create(
                       duckdb::ExpressionType::COMPARE_GREATERTHAN,
                       duckdb::make_uniq<duckdb::BoundReferenceExpression>(type, slot),
                       duckdb::make_uniq<duckdb::BoundConstantExpression>(value))});
  }
  test.program->catalogue =
    std::make_shared<static_predicate_catalogue const>(77, 1, std::move(terms));
  test.program->predicates = 3;
  test.inputs.mapping.evidence =
    std::make_shared<mapping_certification const>(*test.program, plan, test.split.input_domain);
  test.inputs.expression_policy.strategy =
    GENERATE(sirius::expression_evaluator_strategy::MATERIALIZE,
             sirius::expression_evaluator_strategy::AST_INTERPRET,
             sirius::expression_evaluator_strategy::AST_JIT);
  test.inputs.expression_policy.regex_jit = false;
  auto output                             = materialize_shared(
    test.split,
    test.inputs,
    [&](auto const&, auto const&, auto const&, auto s, auto resource) {
      auto decoded                                     = test.decode(s, resource);
      decoded.batch.execution->logical_schema_complete = false;
      return decoded;
    },
    *space,
    stream,
    mr);
  REQUIRE(output.batch.table.n_columns() == 2);
  CHECK(download<int64_t>(output.batch.table.view().column(0), stream) == std::vector<int64_t>{3});
  CHECK(download<int32_t>(output.batch.table.view().column(1), stream) ==
        std::vector<int32_t>{300});
  CHECK(output.lineage.ordinals_on(stream) == std::vector<int64_t>{3});
  CHECK(output.lineage.resolve(3) == std::pair<std::size_t, int64_t>{0, 3});
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Independent oracle detects plausible decoder omissions and false coverage",
                 "[scan][shared_materializer][integration]")
{
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  auto mr = rmm::device_async_resource_ref(space->get_default_allocator());
  scenario test;
  bool const omission = GENERATE(false, true);
  // CPU oracle starts from the original input, independently of reported coverage.
  std::vector<int32_t> oracle;
  std::vector<int32_t> a{1, 101, 102, 103, 104}, b{1000, 5, 200, 300, 50};
  for (std::size_t row = 0; row < a.size(); ++row)
    if (a[row] > 100 && b[row] > 100 && row != 2) oracle.push_back(b[row]);
  auto output = materialize_shared(
    test.split,
    test.inputs,
    [&](auto const&, auto const&, auto const&, auto s, auto resource) {
      auto decoded = test.decode(s, resource);
      if (omission) {
        // A structurally valid subset silently omits the only valid final row.
        auto kept     = upload<int64_t>({0, 1, 3}, cudf::type_id::INT64, s, resource);
        auto selected = cudf::gather(decoded.batch.table.view(),
                                     kept->view(),
                                     cudf::out_of_bounds_policy::DONT_CHECK,
                                     s,
                                     resource);
        auto lineage  = decoded.lineage.select(kept->view(), s, resource);
        s.sync();
        decoded.batch.table = owning_table_view{std::move(selected)};
        decoded.lineage     = std::move(lineage);
      } else {
        decoded.batch.execution->whole_static_complete = true;
      }
      return decoded;
    },
    *space,
    stream,
    mr);
  CHECK(download<int32_t>(output.batch.table.view().column(0), stream) != oracle);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Shared LIKE cache and observer identities survive separate attempts",
                 "[scan][shared_materializer][integration]")
{
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  auto mr = rmm::device_async_resource_ref(space->get_default_allocator());
  scenario test;
  auto domain               = std::make_shared<lineage_domain>(*test.split.input_domain);
  domain->files[0].identity = "AA";
  test.split.input_domain   = domain;
  auto deleted              = std::make_shared<iceberg_delete_set const>(
    "AA", std::make_shared<std::vector<int64_t> const>(std::initializer_list<int64_t>{2}));
  test.inputs.visibility = {{{"AA", deleted}, deleted}};
  test.program->required_visibility =
    std::make_shared<visibility_bindings const>(visibility_bindings{{"AA", deleted}});
  auto plan = std::make_shared<scan_plan>(*test.split.plan);
  plan->virtual_columns.push_back({999,
                                   "filename",
                                   sirius::logical_type::make(sirius::type_id::VARCHAR),
                                   scan_plan::parquet_virtual_column_kind::FILENAME,
                                   2});
  test.split.plan = plan;
  test.program->input_types->push_back(cudf::data_type{cudf::type_id::STRING});
  std::vector<sirius::op::table_filter_conjunct> terms;
  for (std::size_t id = 0; id < 2; ++id)
    terms.push_back(
      {id,
       id,
       duckdb::BoundComparisonExpression::Create(
         duckdb::ExpressionType::COMPARE_GREATERTHAN,
         duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, id),
         duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(100)))});
  auto like = duckdb::make_uniq<duckdb::BoundFunctionExpression>(
    duckdb::BoundScalarFunction(
      duckdb::ScalarFunction("~~",
                             {duckdb::LogicalType::VARCHAR, duckdb::LogicalType::VARCHAR},
                             duckdb::LogicalType::BOOLEAN,
                             nullptr)),
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{},
    nullptr);
  like->GetChildrenMutable().push_back(
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::VARCHAR, 2));
  like->GetChildrenMutable().push_back(
    duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value("%A%A%")));
  terms.push_back({2, 2, std::move(like)});
  test.program->catalogue =
    std::make_shared<static_predicate_catalogue const>(77, 1, std::move(terms));
  test.program->predicates = 3;
  test.inputs.mapping.evidence =
    std::make_shared<mapping_certification const>(*test.program, plan, domain);
  auto cache                     = std::make_shared<sirius::like_multiliteral_cache>();
  test.inputs.like_swar_fastpath = true;
  test.inputs.like_cache         = cache;
  test.inputs.expression_policy  = {sirius::expression_evaluator_strategy::AST_JIT, false};
  sirius::util::env_guard opt("SIRIUS_ENABLE_TEST_OPTIONS", "1");
  recorder observed;
  scoped_step_observer scope(observed);
  for (uint64_t attempt : {3, 4}) {
    test.split.attempt       = attempt;
    test.split.logical_input = 26 + attempt;
    auto output              = materialize_shared(
      test.split,
      test.inputs,
      [&](auto const&, auto const&, auto const&, auto s, auto resource) {
        auto decoded                                     = test.decode(s, resource);
        decoded.batch.execution->logical_schema_complete = false;
        return decoded;
      },
      *space,
      stream,
      mr);
    CHECK(download<int32_t>(output.batch.table.view().column(0), stream) ==
          std::vector<int32_t>{300});
  }
  CHECK(cache->classification_count_for_testing() == 1);
  CHECK(observed.files == std::vector<std::string>{"AA", "AA"});
  CHECK(observed.terms == std::vector<std::size_t>{1, 2, 1, 2});
  REQUIRE(observed.identities.size() == 8);
  for (std::size_t i = 0; i < observed.identities.size(); ++i) {
    CHECK(observed.identities[i].contract == 77);
    CHECK(observed.identities[i].attempt == (i < 4 ? 3 : 4));
    CHECK(observed.identities[i].logical_input == (i < 4 ? 29 : 30));
  }
}

TEST_CASE_METHOD(
  sirius::test::GpuExecutionFixture,
  "Shared legacy completion binds the delivered output and preserves visibility gates",
  "[scan][shared_materializer][integration]")
{
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  auto mr = rmm::device_async_resource_ref(space->get_default_allocator());
  scenario test;
  test.inputs.validation         = validation_policy::legacy_iceberg;
  test.inputs.diagnostic_context = "legacy-file";
  test.split.input_domain.reset();
  bool const nonempty  = GENERATE(false, true);
  bool const assembled = GENERATE(false, true);
  bool const deferred  = GENERATE(false, true);
  if (!nonempty) {
    auto empty             = std::make_shared<iceberg_delete_set const>("A");
    test.inputs.visibility = {{{"A", empty}, empty}};
    test.program->required_visibility =
      std::make_shared<visibility_bindings const>(visibility_bindings{{"A", empty}});
  }
  if (deferred) test.inputs.phase = materialization_phase::decoded;
  sirius::util::env_guard opt("SIRIUS_ENABLE_TEST_OPTIONS", "1");
  recorder observed;
  scoped_step_observer scope(observed);
  auto run = [&] {
    return materialize_shared(
      test.split,
      test.inputs,
      [&](auto const&, auto const&, auto const&, auto s, auto resource) {
        // The legacy producer already enforced both predicates. Changed values
        // would remove all rows if the shared consumer re-evaluated them.
        std::vector<std::unique_ptr<cudf::column>> columns;
        if (!assembled)
          columns.push_back(upload<int32_t>({0, 0}, cudf::type_id::INT32, s, resource));
        columns.push_back(upload<int32_t>({200, 300}, cudf::type_id::INT32, s, resource));
        decode_result result;
        result.batch.table = owning_table_view{std::make_unique<cudf::table>(std::move(columns))};
        result.batch.state =
          assembled ? filter_state::ROW_FILTERED_AND_PROJECTED : filter_state::ROW_FILTERED;
        return result;
      },
      *space,
      stream,
      mr);
  };
  if (nonempty && !deferred) {
    REQUIRE_THROWS_AS(run(), sirius::internal_exception);
    REQUIRE_THROWS_WITH(
      run(),
      Catch::Matchers::ContainsSubstring(
        "decode returned a filtered table, so row positions no longer identify file rows"));
    CHECK(observed.files.empty());
    CHECK(observed.terms.empty());
    CHECK(observed.assemblies == 0);
  } else {
    auto result = run();
    REQUIRE(result.batch.execution);
    CHECK(result.batch.execution->projection_complete == (assembled || !deferred));
    CHECK(result.batch.execution->layout ==
          (assembled || !deferred ? test.program->output_layout : test.program->input_layout));
    CHECK(result.batch.execution->visibility_complete == !deferred);
    CHECK(observed.terms.empty());
    CHECK(observed.assemblies == (!assembled && !deferred ? 1 : 0));
    if (!deferred) {
      CHECK(download<int32_t>(result.batch.table.view().column(0), stream) ==
            std::vector<int32_t>{200, 300});
      CHECK(result.lineage.state == row_position_state::none);
    }
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Deferred replacement success retires borrowed staging before returning",
                 "[scan][shared_materializer][decoder_boundary][integration]")
{
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  auto mr = rmm::device_async_resource_ref(space->get_default_allocator());
  scenario test;
  test.inputs.phase = materialization_phase::decoded;
  test.inputs.visibility.clear();
  test.program->required_visibility.reset();
  auto staging = std::shared_ptr<cudf::column const>(
    upload<int32_t>({901, 902, 903, 904}, cudf::type_id::INT32, stream, mr));
  test.inputs.mapping.evidence =
    std::make_shared<mapping_certification const>(*test.program, test.split.plan, staging);
  std::atomic<bool> copied{false};
  auto const mode        = GENERATE(0, 1, 2);  // success, unsafe retirement, cancelled during copy
  bool const failed_join = mode == 1;
  std::stop_source stop;
  test.split.stop = stop.get_token();
  struct completion_state {
    std::atomic<bool>* copied;
    std::stop_source* stop;
    bool cancel;
  } completion{&copied, &stop, mode == 2};
  detail::read_retirement_test_hooks hooks(
    failed_join ? std::optional{detail::read_checkpoint::BATCH} : std::nullopt, failed_join);
  auto run = [&] {
    return materialize_shared(
      test.split,
      test.inputs,
      [&](auto const&, auto const&, auto const&, auto s, auto resource) {
        auto decoded                       = test.decode(s, resource);
        decoded.lineage                    = {};
        decoded.batch.execution->positions = row_position_state::none;
        REQUIRE(cudaLaunchHostFunc(
                  s.get(),
                  [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); },
                  nullptr) == cudaSuccess);
        REQUIRE(cudaMemcpyAsync(const_cast<int32_t*>(
                                  decoded.batch.table.view().column(0).template data<int32_t>()),
                                staging->view().data<int32_t>(),
                                4 * sizeof(int32_t),
                                cudaMemcpyDeviceToDevice,
                                s.get()) == cudaSuccess);
        REQUIRE(cudaLaunchHostFunc(
                  s.get(),
                  [](void* p) {
                    auto& done = *static_cast<completion_state*>(p);
                    done.copied->store(true);
                    if (done.cancel) done.stop->request_stop();
                  },
                  &completion) == cudaSuccess);
        return decoded;
      },
      *space,
      stream,
      mr);
  };
  if (failed_join) {
    CHECK_THROWS_AS(run(), sirius::unjoined_gpu_work);
    CHECK(hooks.retained.has_value());
    CHECK(copied.load());
  } else if (mode == 2) {
    CHECK_THROWS_WITH(run(), Catch::Matchers::ContainsSubstring("cancelled"));
    CHECK(copied.load());
  } else {
    auto output = run();
    CHECK(copied.load());
    CHECK(cudaStreamQuery(stream.value()) == cudaSuccess);
    CHECK(download<int32_t>(output.batch.table.view().column(0), stream) ==
          std::vector<int32_t>{901, 902, 903, 904});
  }
  stream.synchronize();
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Replacement results release unused positions or preserve selected lineage",
                 "[scan][shared_materializer][decoder_boundary][integration]")
{
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  auto mr = rmm::device_async_resource_ref(space->get_default_allocator());
  scenario test;
  bool const visibility_done = GENERATE(false, true);
  if (!visibility_done) {
    test.inputs.visibility.clear();
    test.program->required_visibility.reset();
  }
  test.split.input_domain.reset();
  test.inputs.needs_lineage = GENERATE(false, true);
  auto output               = materialize_shared(
    test.split,
    test.inputs,
    [&](auto const&, auto const&, auto const&, auto s, auto resource) {
      decode_result decoded;
      std::vector<int32_t> a(100, 101), b(100, 0);
      std::fill_n(b.begin(), 5, 200);
      std::vector<std::unique_ptr<cudf::column>> columns;
      columns.push_back(upload<int32_t>(a, cudf::type_id::INT32, s, resource));
      columns.push_back(upload<int32_t>(b, cudf::type_id::INT32, s, resource));
      decoded.batch.table = owning_table_view{std::make_unique<cudf::table>(std::move(columns))};
      decoded.batch.row_positions = std::make_unique<gpu_row_positions>();
      std::vector<uint64_t> rows(100);
      std::iota(rows.begin(), rows.end(), uint64_t{0});
      if (visibility_done)
        for (std::size_t i = 2; i < rows.size(); ++i)
          ++rows[i];
      decoded.batch.row_positions->rows =
        upload<uint64_t>(rows, cudf::type_id::UINT64, s, resource);
      auto file_rows = uint64_t{visibility_done ? 101u : 100u};
      decoded.batch.row_positions->sources.push_back({"A", file_rows, {{0, file_rows}}});
      if (visibility_done) {
        scan_execution_record facts;
        facts.producer                = execution_record_producer::own_decoder;
        facts.contract                = 77;
        facts.generation              = 1;
        facts.authoritative           = true;
        facts.logical_schema_complete = true;
        facts.layout                  = test.program->input_layout;
        facts.layout_width            = 2;
        facts.visibility_complete     = true;
        facts.visibility_applied      = *test.program->required_visibility;
        facts.positions               = row_position_state::per_row;
        decoded.batch.execution       = std::move(facts);
      }
      return decoded;
    },
    *space,
    stream,
    mr);
  CHECK(output.batch.table.num_rows() == 5);
  CHECK_FALSE(output.batch.row_positions);
  if (test.inputs.needs_lineage) {
    output.lineage.validate_on(5, stream, mr);
    CHECK(output.lineage.ordinals_on(stream) == (visibility_done
                                                   ? std::vector<int64_t>{0, 1, 3, 4, 5}
                                                   : std::vector<int64_t>{0, 1, 2, 3, 4}));
  } else {
    CHECK(output.lineage.state == row_position_state::none);
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Production decoder override validates types row groups and attempt context",
                 "[scan][shared_materializer][decoder_boundary][integration]")
{
  auto mode = std::string(GENERATE(
    "valid", "wrong_input_type", "wrong_output_type", "reversed_rg", "repeated_rg", "cancelled"));
  sirius::test::scratch_dir dir("decoder_validation");
  auto path = dir.file("input.parquet");
  run_ok("COPY (SELECT i::BIGINT id FROM range(4096) t(i)) TO " +
         dir.file_literal("input.parquet") + " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
  auto info                    = std::make_unique<parquet_ingestible_table_info>();
  info->resolved_file_paths    = {path};
  info->names                  = {"id"};
  info->returned_types         = {sirius::logical_type::make(sirius::type_id::BIGINT)};
  info->column_ids             = {duckdb::ColumnIndex(0)};
  info->projected_output_count = 1;
  parquet_scan_core core(std::move(info));
  auto ioctx     = std::make_shared<sirius::io::kvikio_context>();
  auto prepare   = core.next_split_provider([ioctx](std::string_view) { return ioctx; });
  auto coalescer = core.create_batch_coalescer();
  auto splits    = coalescer->push(prepare({}));
  for (auto& split : coalescer->flush())
    splits.push_back(std::move(split));
  REQUIRE(splits.size() == 1);
  auto& physical = dynamic_cast<parquet_split_info&>(*splits.front());
  REQUIRE(physical.rg_slices.front().row_group_indices == std::vector<int>{0, 1});
  bool const shared_metadata = GENERATE(false, true);
  std::shared_ptr<scan_info> retained_input;
  if (shared_metadata) retained_input = std::move(splits.front());
  if (mode == "reversed_rg") physical.rg_slices.front().row_group_indices = {1, 0};
  if (mode == "repeated_rg") physical.rg_slices.front().row_group_indices = {1, 1};
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  std::stop_source stop;
  if (mode == "cancelled") stop.request_stop();
  int calls = 0;
  core.set_decode_override(
    [&](auto const& split, auto const& inputs, auto const&, auto s, auto mr) {
      ++calls;
      CHECK(bool(split.info) == shared_metadata);
      if (shared_metadata) {
        CHECK(split.slices.data() == physical.rg_slices.data());
        CHECK(split.dependencies.data() == physical.dependencies().data());
      } else {
        CHECK(split.metadata_owner);
      }
      CHECK(split.attempt == static_cast<uint64_t>(calls + 6));
      CHECK(split.logical_input == 123);
      CHECK(split.stop.stop_possible());
      decode_result decoded;
      std::vector<std::unique_ptr<cudf::column>> columns;
      if (mode.starts_with("wrong"))
        columns.push_back(upload<int32_t>({99}, cudf::type_id::INT32, s, mr));
      else
        columns.push_back(upload<int64_t>({99}, cudf::type_id::INT64, s, mr));
      decoded.batch.table = owning_table_view{std::make_unique<cudf::table>(std::move(columns))};
      scan_execution_record facts;
      facts.producer                = execution_record_producer::own_decoder;
      facts.contract                = inputs.mapping.program->contract;
      facts.generation              = inputs.mapping.program->generation;
      facts.authoritative           = true;
      facts.logical_schema_complete = true;
      facts.projection_complete     = mode == "wrong_output_type";
      facts.whole_static_complete   = true;
      facts.layout            = facts.projection_complete ? inputs.mapping.program->output_layout
                                                          : inputs.mapping.program->input_layout;
      facts.layout_width      = 1;
      decoded.batch.execution = facts;
      return decoded;
    });
  auto run = [&](uint64_t attempt) {
    return core.materialize_metadata_to_table(
      physical, *space, stream, false, nullptr, {}, {}, {}, {attempt, 123, stop.get_token()});
  };
  if (mode == "valid") {
    for (uint64_t attempt : {7, 8}) {
      auto output =
        core.post_filter_and_project(run(attempt), *space, stream, false, nullptr, nullptr, {});
      CHECK(output->view().column(0).type().id() == cudf::type_id::INT64);
    }
    CHECK(calls == 2);
  } else {
    REQUIRE_THROWS(run(7));
    CHECK(calls == (mode.starts_with("wrong") ? 1 : 0));
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Shared certification rejects another files normalization before decode",
                 "[scan][shared_materializer][decoder_boundary][integration]")
{
  sirius::test::scratch_dir dir("mapping_swap");
  run_ok("COPY (SELECT 1::INTEGER id) TO " + dir.file_literal("a.parquet") +
         " (FORMAT PARQUET, FIELD_IDS {id: 1})");
  run_ok("COPY (SELECT 99::INTEGER x) TO " + dir.file_literal("b.parquet") +
         " (FORMAT PARQUET, FIELD_IDS {x: 2})");
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  scenario test;
  auto plan           = std::make_shared<scan_plan>();
  plan->data_columns  = {{0, "x"}};
  plan->output_layout = {{scan_plan::output_entry::DATA, 0}};
  test.split.plan     = plan;
  test.inputs.visibility.clear();
  test.program->required_visibility.reset();
  test.program->catalogue = std::make_shared<static_predicate_catalogue const>(
    77, 1, std::vector<sirius::op::table_filter_conjunct>{});
  test.program->predicates       = 0;
  test.inputs.coverage.catalogue = test.program->catalogue;
  test.program->input_types = std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT32}};
  auto mapping              = [&](char const* file) {
    auto source = context->get_scan_manager().create_datasource(dir.file(file));
    auto footer =
      resolve_parquet_metadata_with_identity(
        *source, 77, dir.file(file), cudf::io::parquet_reader_options::builder().build())
        .artifact;
    return std::make_shared<field_id_mapping const>(
      field_id_mapping::compile(footer,
                                footer->built_for_identity,
                                *test.program,
                                std::vector<field_requirement>{
                                  {2, duckdb::LogicalType::INTEGER, duckdb::Value::INTEGER(7)}}));
  };
  auto a                  = mapping("a.parquet");
  auto b                  = mapping("b.parquet");
  auto domain             = std::make_shared<lineage_domain>();
  domain->files           = {{dir.file("b.parquet"), 1}};
  domain->file_runs       = {{0, 0, 1}};
  test.split.input_domain = domain;
  test.inputs.mapping.evidence =
    std::make_shared<mapping_certification const>(*test.program, plan, b, b);
  test.inputs.mapping.normalization = a;
  int calls                         = 0;
  CHECK_THROWS_WITH(materialize_shared(
                      test.split,
                      test.inputs,
                      [&](auto const&, auto const&, auto const&, auto, auto) {
                        ++calls;
                        return decode_result{};
                      },
                      *space,
                      stream,
                      rmm::device_async_resource_ref(space->get_default_allocator())),
                    Catch::Matchers::ContainsSubstring("certification binding mismatch"));
  CHECK(calls == 0);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Shared decoder applies real eager V2 signed positions within input ranges",
                 "[scan][shared_materializer][decoder_boundary][eager_delete][integration]")
{
  sirius::test::scratch_dir dir("lineage_eager");
  auto path             = dir.file("input.parquet");
  bool const duplicates = GENERATE(false, true);
  run_ok("COPY (SELECT i::INTEGER id FROM range(20) t(i)) TO " + dir.file_literal("input.parquet") +
         " (FORMAT PARQUET)");
  run_ok("COPY (SELECT " + dir.file_literal("input.parquet") +
         " file_path, pos FROM (VALUES (-1::BIGINT), (11::BIGINT)" +
         (duplicates ? ",(11::BIGINT)" : "") + ") t(pos)) TO " +
         dir.file_literal("deletes.parquet") + " (FORMAT PARQUET)");
  auto ioctx = std::make_shared<sirius::io::kvikio_context>();
  iceberg_delete_discovery discovery;
  discovery.positional_delete_files = {dir.file("deletes.parquet")};
  auto payload = load_delete_payload(*con->context, "lineage-eager", ioctx.get(), {}, discovery);
  REQUIRE(payload);
  REQUIRE(payload->positional_deletes.at(path) ==
          (duplicates ? std::vector<int64_t>{-1, 11, 11} : std::vector<int64_t>{-1, 11}));
  scenario test;
  auto plan               = std::make_shared<scan_plan>();
  plan->data_columns      = {{0, "id"}};
  plan->output_layout     = {{scan_plan::output_entry::DATA, 0}};
  test.split.plan         = plan;
  auto domain             = std::make_shared<lineage_domain>();
  bool const disjoint     = GENERATE(false, true);
  domain->files           = {{path, 20}};
  domain->file_runs       = {{0, disjoint ? 12 : 0, disjoint ? 8 : 20}};
  test.split.input_domain = domain;
  test.program->catalogue = std::make_shared<static_predicate_catalogue const>(
    77, 1, std::vector<sirius::op::table_filter_conjunct>{});
  test.program->predicates  = 0;
  test.program->input_types = std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT32}};
  test.inputs.coverage.catalogue = test.program->catalogue;
  auto deletes                   = std::make_shared<iceberg_delete_set const>(
    path, std::make_shared<std::vector<int64_t> const>(payload->positional_deletes.at(path)));
  test.inputs.visibility = {{{path, deletes}, deletes}};
  test.program->required_visibility =
    std::make_shared<visibility_bindings const>(visibility_bindings{{path, deletes}});
  test.inputs.mapping.evidence =
    std::make_shared<mapping_certification const>(*test.program, plan, domain);
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  auto output = materialize_shared(
    test.split,
    test.inputs,
    [&](auto const&, auto const&, auto const&, auto s, auto mr) {
      decode_result decoded;
      decoded.lineage = {row_position_state::identity, domain};
      std::vector<int32_t> data(disjoint ? 8 : 20);
      std::iota(data.begin(), data.end(), disjoint ? 12 : 0);
      std::vector<std::unique_ptr<cudf::column>> columns;
      columns.push_back(upload<int32_t>(data, cudf::type_id::INT32, s, mr));
      decoded.batch.table = owning_table_view{std::make_unique<cudf::table>(std::move(columns))};
      return decoded;
    },
    *space,
    stream,
    rmm::device_async_resource_ref(space->get_default_allocator()));
  std::vector<int32_t> expected;
  for (int32_t i = disjoint ? 12 : 0; i < 20; ++i)
    if (i != 11) expected.push_back(i);
  CHECK(download<int32_t>(output.batch.table.view().column(0), stream) == expected);
  CHECK(output.batch.execution->visibility_complete);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Production OOM retry reaches shared steps with a new attempt identity",
                 "[scan][shared_materializer][decoder_boundary][integration]")
{
  sirius::test::scratch_dir dir("scan_attempt");
  run_ok("SET gpu_execution=false");
  run_ok("COPY (SELECT i::BIGINT id FROM range(128) t(i)) TO " + dir.file_literal("input.parquet") +
         " (FORMAT PARQUET)");
  sirius::util::env_guard opt("SIRIUS_ENABLE_TEST_OPTIONS", "1");
  struct attempts_observer : step_observer {
    std::atomic<bool> retried{false};
    std::atomic<uint64_t> input{0};
    void on_assemble(step_identity const& id) override
    {
      if (id.attempt > 1) {
        retried.store(true);
        input.store(id.logical_input);
      }
    }
  } observed;
  scoped_step_observer scope(observed);
  sirius::test::scoped_setting backoff(*con, "sirius_test_gpu_task_retry_backoff_ms", 0);
  sirius::test::scoped_setting injection(*con, "sirius_test_inject_gpu_task_oom", 1);
  run_ok("SET gpu_execution=true");
  auto result =
    con->Query("SELECT sum(id) FROM read_parquet(" + dir.file_literal("input.parquet") + ")");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());
  CHECK(result->Collection().GetValue(0, 0).ToString() == "8128");
  CHECK(observed.retried.load());
  CHECK(observed.input.load() != 0);
}

TEST_CASE("Query terminal errors stop in-flight decoder tokens",
          "[scan][shared_materializer][decoder_boundary]")
{
  sirius::pipeline::completion_handler completion;
  auto token = completion.execution_stop_token();
  CHECK(token.stop_possible());
  CHECK_FALSE(token.stop_requested());
  completion.report_error(std::make_exception_ptr(std::runtime_error("cancel scan")));
  CHECK(token.stop_requested());
}

TEST_CASE("Range clipping preserves strict deferred bounds and signed eager bounds",
          "[scan][shared_materializer][decoder_boundary]")
{
  auto values =
    std::make_shared<std::vector<int64_t> const>(std::initializer_list<int64_t>{-1, 11, 21});
  iceberg_delete_set eager("A", values);
  CHECK_NOTHROW(eager.validate_file_bounds(20));
  iceberg_delete_set deferred("A");
  std::array<int64_t, 2> outside{11, 21};
  deferred.positions = outside;
  CHECK_THROWS(deferred.validate_file_bounds(20));
  deferred.positions = std::span<int64_t const>(outside).first(1);
  CHECK_NOTHROW(deferred.validate_file_bounds(20));
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Repeated input domains reuse selected file deletes within workspace budget",
                 "[scan][shared_materializer][decoder_boundary][integration]")
{
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto* space  = context->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  auto upstream = rmm::device_async_resource_ref(space->get_default_allocator());
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(upload<int32_t>({42}, cudf::type_id::INT32, stream, upstream));
  owning_table_view table{std::make_unique<cudf::table>(std::move(columns))};
  auto domain   = std::make_shared<lineage_domain>();
  domain->files = {{"A", 1024}};
  domain->file_runs.assign(128, lineage_domain::file_run{0, 0, 1024});
  row_lineage lineage{row_position_state::runs, domain, {{0, 1}}, {}};
  std::vector<int64_t> deletes(1024);
  std::iota(deletes.begin(), deletes.end(), int64_t{0});
  sirius::test::tracked_scan_resource tracked{upstream, stream};
  apply_lineage_visibility(
    table,
    lineage,
    [&](std::string const&) -> std::span<int64_t const> { return deletes; },
    stream,
    rmm::device_async_resource_ref(tracked));
  CHECK(table.num_rows() == 0);
  CHECK(tracked.state->peak <= shared_materialization_workspace_bytes(1, 4, 1024, 128));
  CHECK_FALSE(tracked.state->wrong_stream);
}
