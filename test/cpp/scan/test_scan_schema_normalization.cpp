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

// Schema contract sirius_gpu_scan_operator::execute() holds every split to.
//
// A resident split is normalized against normalization_targets() even with no plan sidecar,
// because a chunk pinned while narrowing was on must restore to its native carrier. Two shapes
// cannot be normalized at all, and each one means the table this scan materialized is not the
// table its output types describe:
//
//   - a column count that disagrees with the target list, which leaves every later stage
//     indexing columns that are not the ones it named;
//   - without a sidecar, a carrier that is not a narrower form of the native type, which no
//     restoring cast can turn into the declared type.
//
// Both throw here rather than reaching a consumer, since a batch that disagrees with its own
// declared schema surfaces far from the scan that produced it.

#include "operator/operator_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <compression/compressed_scan.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <helper/type_conversions.hpp>
#include <io/io_context.hpp>
#include <op/scan/decoded_batch_representation.hpp>
#include <op/scan/gpu_ingestible.hpp>
#include <op/scan/gpu_ingestible_types.hpp>
#include <op/scan/sirius_gpu_scan_operator.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <op/sirius_physical_operator.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct test_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu_space;
  cucascade::memory::memory_space* host_space;
  rmm::cuda_stream conv_stream;

  test_env()
    : mgr(sirius::test::operator_utils::initialize_memory_manager()),
      gpu_space(mgr->get_memory_space(cucascade::memory::Tier::GPU, 0)),
      host_space(mgr->get_memory_space(cucascade::memory::Tier::HOST, 0)),
      conv_stream()
  {
  }

  ::cuda::stream_ref stream() { return conv_stream; }
};

test_env& env()
{
  sirius::test::operator_utils::ensure_converter_registry();
  static test_env e;
  return e;
}

std::unique_ptr<cudf::column> make_column(cucascade::memory::memory_space& space,
                                          cudf::data_type type,
                                          std::size_t rows)
{
  auto mr     = sirius::test::operator_utils::get_resource_ref(space);
  auto stream = sirius::test::operator_utils::default_stream();
  return cudf::make_numeric_column(
    type, static_cast<cudf::size_type>(rows), cudf::mask_state::UNALLOCATED, stream, mr);
}

std::shared_ptr<cucascade::data_batch> make_host_resident_batch(
  test_env& e, const std::vector<std::vector<int32_t>>& values_by_column)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(values_by_column.size());
  for (auto const& values : values_by_column) {
    auto column = make_column(*e.gpu_space, cudf::data_type{cudf::type_id::INT32}, values.size());
    cudaMemcpyAsync(column->mutable_view().data<int32_t>(),
                    values.data(),
                    sizeof(int32_t) * values.size(),
                    cudaMemcpyHostToDevice,
                    e.stream().get());
    columns.push_back(std::move(column));
  }

  cucascade::gpu_table_representation gpu_repr(
    std::make_unique<cudf::table>(std::move(columns)), *e.gpu_space, e.stream());
  auto host_repr = sirius::converter_registry::get().convert<cucascade::host_data_representation>(
    gpu_repr, e.host_space, e.stream());
  e.stream().sync();
  return cucascade::data_batch::make(sirius::get_next_batch_id(), std::move(host_repr));
}

/// Cached chunk standing in for a pinned split: its own contents never reach the assertions,
/// since only is_resident() routes execute() down the cached branch.
std::shared_ptr<cucascade::data_batch> make_resident_batch(test_env& e, std::size_t rows)
{
  std::shared_ptr<cudf::column> col =
    make_column(*e.gpu_space, cudf::data_type{cudf::type_id::INT64}, rows);
  std::vector<std::shared_ptr<cudf::column>> columns{col};
  std::vector<cudf::column_view> views{col->view()};
  auto const alloc_size = col->alloc_size();
  auto repr =
    std::make_unique<cucascade::gpu_table_representation>(cudf::table_view(views),
                                                          std::move(columns),
                                                          alloc_size,
                                                          *e.gpu_space,
                                                          ::cuda::stream_ref{cudaStream_t{}});
  return cucascade::data_batch::make(sirius::get_next_batch_id(), std::move(repr));
}

class stub_table_info final : public sirius::op::scan::ingestible_table_info {
 public:
  [[nodiscard]] std::span<std::string const> column_names() const override { return {}; }
  [[nodiscard]] std::span<std::string const> file_paths() const override { return {}; }
  [[nodiscard]] std::string display_name() const override { return "<stub>"; }
};

/// Hands execute() a caller-chosen table as the post-filter result, which is the seam where a
/// materialized shape that disagrees with the scan's output types can be injected. A resident
/// split reads its rows from the cached batch, so every metadata entry point stays unreachable.
class stub_ingestible final : public sirius::op::scan::gpu_ingestible {
 public:
  using table_factory = std::function<std::unique_ptr<cudf::table>()>;

  /// @p leading_identity and @p output_prefix_width are what the stub reports about the assembly
  /// its post_filter_and_project stands in for.
  explicit stub_ingestible(table_factory produce,
                           bool leading_identity                          = false,
                           std::optional<std::size_t> output_prefix_width = std::nullopt)
    : _produce(std::move(produce)),
      _leading_identity(leading_identity),
      _output_prefix_width(output_prefix_width)
  {
  }

  std::unique_ptr<cudf::table> post_filter_and_project(
    sirius::op::scan::filtered_table&&,
    const cucascade::memory::memory_space&,
    ::cuda::stream_ref,
    bool,
    std::shared_ptr<const sirius::like_multiliteral_cache>,
    std::unique_ptr<cudf::column>*,
    std::span<std::size_t const> /*elided*/) override
  {
    return _produce();
  }

  std::unique_ptr<sirius::op::scan::batch_coalescer> create_batch_coalescer() const override
  {
    return nullptr;
  }

  [[nodiscard]] bool has_processed_all_metadata() const override { return true; }

  metadata_scan_task_t next_split_provider(sirius::io::ioctx_resolver) override { return {}; }

  sirius::op::scan::filtered_table materialize_metadata_to_table(
    const sirius::op::scan::scan_info&,
    const cucascade::memory::memory_space&,
    ::cuda::stream_ref,
    bool,
    std::shared_ptr<const sirius::like_multiliteral_cache>) override
  {
    throw std::logic_error("stub_ingestible: a resident split never decodes scan metadata");
  }

  [[nodiscard]] const sirius::op::scan::ingestible_table_info& table_info() const noexcept override
  {
    return _info;
  }

  [[nodiscard]] std::vector<std::size_t> materialized_column_order() const override { return {}; }

  [[nodiscard]] bool output_assembly_is_leading_identity() const noexcept override
  {
    return _leading_identity;
  }

  [[nodiscard]] std::optional<std::size_t> output_prefix_width() const noexcept override
  {
    return _output_prefix_width;
  }

 private:
  table_factory _produce;
  bool _leading_identity;
  std::optional<std::size_t> _output_prefix_width;
  stub_table_info _info;
};

/// Scan declaring a single BIGINT output and no plan sidecar, so normalization holds its table
/// to exactly one INT64 column.
sirius::op::scan::sirius_gpu_scan_operator make_bigint_scan(
  std::shared_ptr<sirius::op::scan::gpu_ingestible> ingestible)
{
  return sirius::op::scan::sirius_gpu_scan_operator{
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT}),
    /*estimated_cardinality=*/0,
    std::move(ingestible),
    /*contract_id=*/1};
}

constexpr std::size_t kRows = 8;

void admit_fixture_resident_input(sirius::op::scan::scan_operator_input& input)
{
  sirius::op::scan::pin_validation validation;
  validation.identity = validation.layout = validation.structure = {true, true};
  input.resident_contract_id                                     = 1;
  input.resident_validation                                      = validation;
}

}  // namespace

TEST_CASE("an ingestible cannot report survivors until it says so", "[scan][late_mat]")
{
  // A late-materialization rowid over a FILTERED scan is built from the surviving row positions,
  // and only an ingestible that populates the out-parameter has them. Accepting the parameter
  // proves nothing: duckdb_native_gpu_ingestible takes it and filters with a plain select, so a
  // scan served by it must be refused at install. The default is therefore false, and an
  // implementation opts in only once it actually writes the positions.
  stub_ingestible stub([] { return std::unique_ptr<cudf::table>{}; });
  auto const& as_base = static_cast<sirius::op::scan::gpu_ingestible const&>(stub);
  REQUIRE_FALSE(as_base.can_report_survivors());
  // The unfiltered default too — the pair is what the install gate consults.
  REQUIRE_FALSE(as_base.has_row_filter());
}

TEST_CASE("scan construction rejects an incomplete native carrier schema",
          "[scan_normalization][gpu_scan]")
{
  duckdb::vector<sirius::logical_type> types;
  types.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
  types.push_back(sirius::logical_type::make_decimal(4, 2));

  REQUIRE_THROWS_WITH(
    sirius::op::scan::sirius_gpu_scan_operator(std::move(types), 0, nullptr, /*contract_id=*/1),
    Catch::Matchers::ContainsSubstring(
      "output column 1 (DECIMAL(4,2)) has no native cuDF carrier"));
}

TEST_CASE("scan execute rejects a materialized column count its output types do not describe",
          "[scan_normalization][gpu_scan]")
{
  auto& e    = env();
  auto batch = make_resident_batch(e, kRows);

  auto ingestible = std::make_shared<stub_ingestible>([&e] {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(make_column(*e.gpu_space, cudf::data_type{cudf::type_id::INT64}, kRows));
    columns.push_back(make_column(*e.gpu_space, cudf::data_type{cudf::type_id::INT64}, kRows));
    return std::make_unique<cudf::table>(std::move(columns));
  });
  auto scan       = make_bigint_scan(ingestible);

  sirius::op::scan::scan_operator_input input(batch);
  input.gpu_memory_space = e.gpu_space;
  admit_fixture_resident_input(input);
  REQUIRE(input.is_resident());

  REQUIRE_THROWS_WITH(scan.execute(input, e.stream()),
                      Catch::Matchers::ContainsSubstring("output schema width mismatch"));
}

TEST_CASE("scan execute rechecks resident query freshness",
          "[scan_normalization][gpu_scan][certificate]")
{
  auto& e    = env();
  auto batch = make_resident_batch(e, kRows);
  auto scan  = make_bigint_scan(
    std::make_shared<stub_ingestible>([] { return std::make_unique<cudf::table>(); }));
  auto input              = std::make_unique<sirius::op::scan::scan_operator_input>(batch);
  input->gpu_memory_space = e.gpu_space;
  admit_fixture_resident_input(*input);
  input->resident_validation->query_token = 17;
  scan.set_query_validation(18, {});

  REQUIRE_THROWS_AS(scan.execute(*input, e.stream()), sirius::op::scan::certificate_incomplete);
}

TEST_CASE("scan execute rejects a native carrier no restoring cast can reach its output type",
          "[scan_normalization][gpu_scan]")
{
  auto& e    = env();
  auto batch = make_resident_batch(e, kRows);

  // FLOAT64 is neither the declared INT64 nor a narrower carrier of it, so the restoring cast
  // this scan would otherwise apply has nothing it can legally do.
  auto ingestible = std::make_shared<stub_ingestible>([&e] {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(make_column(*e.gpu_space, cudf::data_type{cudf::type_id::FLOAT64}, kRows));
    return std::make_unique<cudf::table>(std::move(columns));
  });
  auto scan       = make_bigint_scan(ingestible);

  sirius::op::scan::scan_operator_input input(batch);
  input.gpu_memory_space = e.gpu_space;
  admit_fixture_resident_input(input);
  REQUIRE(input.is_resident());

  REQUIRE_THROWS_WITH(scan.execute(input, e.stream()),
                      Catch::Matchers::ContainsSubstring("native schema carrier mismatch"));
}

TEST_CASE("scan execute restores a narrowed resident carrier to its native output type",
          "[scan_normalization][gpu_scan]")
{
  auto& e    = env();
  auto batch = make_resident_batch(e, kRows);

  // The shape the guards above exist to let through: a chunk stored narrow while narrowing was
  // on, restoring to the native carrier its output type declares.
  auto ingestible = std::make_shared<stub_ingestible>([&e] {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(make_column(*e.gpu_space, cudf::data_type{cudf::type_id::INT32}, kRows));
    return std::make_unique<cudf::table>(std::move(columns));
  });
  auto scan       = make_bigint_scan(ingestible);

  sirius::op::scan::scan_operator_input input(batch);
  input.gpu_memory_space = e.gpu_space;
  admit_fixture_resident_input(input);
  REQUIRE(input.is_resident());

  auto output        = scan.execute(input, e.stream());
  auto* pipelineable = dynamic_cast<const sirius::op::pipelineable_operator_data*>(output.get());
  REQUIRE(pipelineable != nullptr);
  auto const& batches = pipelineable->get_data_batches();
  REQUIRE(batches.size() == 1);

  auto restored = batches[0]->to_read_only();
  auto view     = sirius::get_cudf_table_view(restored);
  REQUIRE(view.num_columns() == 1);
  REQUIRE(view.column(0).type().id() == cudf::type_id::INT64);
  REQUIRE(view.num_rows() == static_cast<cudf::size_type>(kRows));
}

TEST_CASE("scan execute transactionally restores a fresh cached conversion",
          "[scan_normalization][gpu_scan][transactional_steal]")
{
  auto& e = env();
  std::vector<int32_t> const narrow_values{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int32_t> const unchanged_values{11, 12, 13, 14, 15, 16, 17, 18};
  auto batch = make_host_resident_batch(e, {narrow_values, unchanged_values});

  bool post_filter_reached = false;
  auto ingestible = std::make_shared<stub_ingestible>([&]() -> std::unique_ptr<cudf::table> {
    post_filter_reached = true;
    throw std::logic_error("transactional scan must bypass post_filter_and_project");
  });
  duckdb::vector<duckdb::LogicalType> logical_types;
  logical_types.push_back(duckdb::LogicalType::BIGINT);
  logical_types.push_back(duckdb::LogicalType::INTEGER);
  sirius::op::scan::sirius_gpu_scan_operator scan{sirius::from_duckdb_vec(logical_types),
                                                  /*estimated_cardinality=*/0,
                                                  ingestible,
                                                  /*contract_id=*/1};

  sirius::op::scan::scan_operator_input input(batch);
  admit_fixture_resident_input(input);
  input.needs_carrier_conversion = true;
  input.prepare_for_processing(e.gpu_space, e.stream());
  REQUIRE(input.converted_table_steal_pending);
  REQUIRE(input.stolen_table_bytes > 0);

  const void* narrow_source_data;
  const void* unchanged_source_data;
  {
    auto ro          = batch->to_read_only();
    auto source_view = sirius::get_cudf_table_view(ro);
    REQUIRE(source_view.num_columns() == 2);
    narrow_source_data    = source_view.column(0).data<int32_t>();
    unchanged_source_data = source_view.column(1).data<int32_t>();
  }

  auto output        = scan.execute(input, e.stream());
  auto* pipelineable = dynamic_cast<const sirius::op::pipelineable_operator_data*>(output.get());
  REQUIRE(pipelineable != nullptr);
  REQUIRE_FALSE(post_filter_reached);
  REQUIRE_FALSE(input.converted_table_steal_pending);
  REQUIRE(input.stolen_table_consumed);
  {
    auto ro = batch->to_read_only();
    REQUIRE(ro.get_data()->get_size_in_bytes() == 0);
  }

  auto const& batches = pipelineable->get_data_batches();
  REQUIRE(batches.size() == 1);
  auto restored = batches[0]->to_read_only();
  auto view     = sirius::get_cudf_table_view(restored);
  REQUIRE(view.num_columns() == 2);
  REQUIRE(view.column(0).type().id() == cudf::type_id::INT64);
  REQUIRE(view.column(1).type().id() == cudf::type_id::INT32);
  REQUIRE(static_cast<const void*>(view.column(0).data<int64_t>()) != narrow_source_data);
  REQUIRE(static_cast<const void*>(view.column(1).data<int32_t>()) == unchanged_source_data);

  e.stream().sync();
  std::vector<int64_t> restored_values(narrow_values.size());
  std::vector<int32_t> moved_values(unchanged_values.size());
  cudaMemcpy(restored_values.data(),
             view.column(0).data<int64_t>(),
             sizeof(int64_t) * restored_values.size(),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(moved_values.data(),
             view.column(1).data<int32_t>(),
             sizeof(int32_t) * moved_values.size(),
             cudaMemcpyDeviceToHost);
  REQUIRE(restored_values == std::vector<int64_t>(narrow_values.begin(), narrow_values.end()));
  REQUIRE(moved_values == unchanged_values);
}

TEST_CASE("scan execute steals a decode-filtered batch that dropped its pure-filter columns",
          "[scan_normalization][gpu_scan][transactional_steal]")
{
  auto& e = env();
  std::vector<int32_t> const narrow_values{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int32_t> const unchanged_values{11, 12, 13, 14, 15, 16, 17, 18};
  std::vector<int32_t> const filter_values{21, 22, 23, 24, 25, 26, 27, 28};

  // The scan reads the first two of the materialized columns, so a decode that applied its whole
  // filter may leave the third behind; post_filter_and_project stands in for the assembly that
  // keeps those two.
  bool post_filter_reached   = false;
  auto const assemble_output = [&]() -> std::unique_ptr<cudf::table> {
    post_filter_reached = true;
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(
      make_column(*e.gpu_space, cudf::data_type{cudf::type_id::INT32}, narrow_values.size()));
    columns.push_back(
      make_column(*e.gpu_space, cudf::data_type{cudf::type_id::INT32}, narrow_values.size()));
    return std::make_unique<cudf::table>(std::move(columns));
  };
  auto const scan_over = [&](bool leading_identity) {
    duckdb::vector<duckdb::LogicalType> logical_types;
    logical_types.push_back(duckdb::LogicalType::BIGINT);
    logical_types.push_back(duckdb::LogicalType::INTEGER);
    return sirius::op::scan::sirius_gpu_scan_operator{
      sirius::from_duckdb_vec(logical_types),
      /*estimated_cardinality=*/0,
      std::make_shared<stub_ingestible>(
        assemble_output, leading_identity, /*output_prefix_width=*/std::size_t{2}),
      /*contract_id=*/1};
  };
  auto scan = scan_over(/*leading_identity=*/true);

  // Stamped before prepare as the decode outcome would stamp them: a plain host conversion leaves
  // them untouched.
  auto const decode_filtered_input = [](std::shared_ptr<cucascade::data_batch> batch,
                                        bool dropped) {
    auto input = std::make_unique<sirius::op::scan::scan_operator_input>(std::move(batch));
    input->row_filter_pending                   = true;
    input->pushdown_row_filtered                = true;
    input->pushdown_filter_only_columns_dropped = dropped;
    input->needs_carrier_conversion             = true;
    return input;
  };

  SECTION("a batch of exactly the output columns is stolen without a copy")
  {
    auto batch = make_host_resident_batch(e, {narrow_values, unchanged_values});
    auto input = decode_filtered_input(batch, /*dropped=*/true);
    input->prepare_for_processing(e.gpu_space, e.stream());
    REQUIRE(input->converted_table_steal_pending);

    const void* unchanged_source_data;
    {
      auto ro               = batch->to_read_only();
      unchanged_source_data = sirius::get_cudf_table_view(ro).column(1).data<int32_t>();
    }

    auto output        = scan.execute(*input, e.stream());
    auto* pipelineable = dynamic_cast<const sirius::op::pipelineable_operator_data*>(output.get());
    REQUIRE(pipelineable != nullptr);
    REQUIRE_FALSE(post_filter_reached);
    REQUIRE(input->stolen_table_consumed);

    auto const& batches = pipelineable->get_data_batches();
    REQUIRE(batches.size() == 1);
    auto restored = batches[0]->to_read_only();
    auto view     = sirius::get_cudf_table_view(restored);
    REQUIRE(view.num_columns() == 2);
    REQUIRE(view.column(0).type().id() == cudf::type_id::INT64);
    REQUIRE(view.column(1).type().id() == cudf::type_id::INT32);
    REQUIRE(static_cast<const void*>(view.column(1).data<int32_t>()) == unchanged_source_data);

    e.stream().sync();
    std::vector<int64_t> restored_values(narrow_values.size());
    std::vector<int32_t> moved_values(unchanged_values.size());
    cudaMemcpy(restored_values.data(),
               view.column(0).data<int64_t>(),
               sizeof(int64_t) * restored_values.size(),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(moved_values.data(),
               view.column(1).data<int32_t>(),
               sizeof(int32_t) * moved_values.size(),
               cudaMemcpyDeviceToHost);
    REQUIRE(restored_values == std::vector<int64_t>(narrow_values.begin(), narrow_values.end()));
    REQUIRE(moved_values == unchanged_values);
  }

  SECTION("a batch that kept its pure-filter column takes the assembly path")
  {
    auto batch = make_host_resident_batch(e, {narrow_values, unchanged_values, filter_values});
    auto input = decode_filtered_input(batch, /*dropped=*/false);
    input->prepare_for_processing(e.gpu_space, e.stream());
    REQUIRE(input->converted_table_steal_pending);

    auto output = scan.execute(*input, e.stream());
    REQUIRE(output != nullptr);
    REQUIRE(post_filter_reached);
    REQUIRE_FALSE(input->stolen_table_consumed);
  }

  SECTION(
    "a batch without its pure-filter column whose output is not a leading identity is assembled")
  {
    auto batch = make_host_resident_batch(e, {narrow_values, unchanged_values});
    auto input = decode_filtered_input(batch, /*dropped=*/true);
    input->prepare_for_processing(e.gpu_space, e.stream());
    REQUIRE(input->converted_table_steal_pending);

    auto reordering_scan = scan_over(/*leading_identity=*/false);
    auto output          = reordering_scan.execute(*input, e.stream());
    REQUIRE(output != nullptr);
    REQUIRE(post_filter_reached);
    REQUIRE_FALSE(input->stolen_table_consumed);
  }

  SECTION("an empty batch of exactly the output columns is stolen")
  {
    auto batch = make_host_resident_batch(e, {std::vector<int32_t>{}, std::vector<int32_t>{}});
    auto input = decode_filtered_input(batch, /*dropped=*/true);
    input->prepare_for_processing(e.gpu_space, e.stream());
    REQUIRE(input->converted_table_steal_pending);

    auto output        = scan.execute(*input, e.stream());
    auto* pipelineable = dynamic_cast<const sirius::op::pipelineable_operator_data*>(output.get());
    REQUIRE(pipelineable != nullptr);
    REQUIRE_FALSE(post_filter_reached);
    REQUIRE(input->stolen_table_consumed);
    auto restored = pipelineable->get_data_batches().at(0)->to_read_only();
    auto view     = sirius::get_cudf_table_view(restored);
    REQUIRE(view.num_columns() == 2);
    REQUIRE(view.num_rows() == 0);
    REQUIRE(view.column(0).type().id() == cudf::type_id::INT64);
    REQUIRE(view.column(1).type().id() == cudf::type_id::INT32);
  }
}

TEST_CASE("materialize refuses a batch without pure-filter columns it cannot vouch for",
          "[scan_normalization][gpu_scan]")
{
  auto& e         = env();
  auto const none = [] { return std::unique_ptr<cudf::table>{}; };

  // One materialized column, while the stub's output reads two.
  auto batch = make_resident_batch(e, kRows);
  sirius::op::scan::scan_operator_input input(batch);
  input.gpu_memory_space                     = e.gpu_space;
  input.pushdown_filter_only_columns_dropped = true;

  SECTION("dropped without the whole filter applied")
  {
    // The width matches, so only the missing filter can be what refuses it.
    stub_ingestible ingestible(none, true, std::size_t{1});
    REQUIRE_THROWS_WITH(
      ingestible.materialize_table(input, e.stream()),
      Catch::Matchers::ContainsSubstring("without applying the scan's whole filter"));
  }

  SECTION("dropped to a width the output does not read")
  {
    input.pushdown_row_filtered = true;
    stub_ingestible ingestible(none, true, std::size_t{2});
    REQUIRE_THROWS_WITH(ingestible.materialize_table(input, e.stream()),
                        Catch::Matchers::ContainsSubstring("1 column(s), but the output reads 2"));
  }

  SECTION("a stolen table dropped to a width the output does not read")
  {
    input.pushdown_row_filtered = true;
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(make_column(*e.gpu_space, cudf::data_type{cudf::type_id::INT64}, kRows));
    input.stolen_table = std::make_unique<cudf::table>(std::move(columns));
    stub_ingestible ingestible(none, true, std::size_t{2});
    REQUIRE_THROWS_WITH(ingestible.materialize_table(input, e.stream()),
                        Catch::Matchers::ContainsSubstring("1 column(s), but the output reads 2"));
    REQUIRE_FALSE(input.stolen_table_consumed);
  }

  SECTION("dropped for an ingestible that promises no prefix")
  {
    input.pushdown_row_filtered = true;
    stub_ingestible ingestible(none);
    REQUIRE_THROWS_WITH(
      ingestible.materialize_table(input, e.stream()),
      Catch::Matchers::ContainsSubstring("but the output reads an unknown number"));
  }
}

TEST_CASE("prepare stamps the outcome of a decode another converter already ran",
          "[scan_normalization][gpu_scan]")
{
  auto& e = env();

  // The batch reaches the scan already decoded on the GPU (by the memory prefetcher, say), so
  // prepare converts nothing; the decode's outcome binds the split all the same.
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(make_column(*e.gpu_space, cudf::data_type{cudf::type_id::INT64}, kRows));
  sirius::pushdown_outcome outcome;
  outcome.row_filtered                = true;
  outcome.filter_only_columns_dropped = true;
  outcome.predicates_enforced         = true;
  auto batch                          = cucascade::data_batch::make(
    sirius::get_next_batch_id(),
    std::make_unique<sirius::decompression_pushdown_batch_representation>(
      std::make_unique<cudf::table>(std::move(columns)), *e.gpu_space, e.stream(), outcome));

  sirius::op::scan::scan_operator_input input(batch);
  input.prepare_for_processing(e.gpu_space, e.stream());

  CHECK(input.pushdown_row_filtered);
  CHECK(input.pushdown_filter_only_columns_dropped);
  CHECK(input.pushdown_predicates_enforced);
  CHECK(input.stolen_table == nullptr);
  CHECK_FALSE(input.converted_table_steal_pending);
}
