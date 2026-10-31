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
#include "utils/test_validation_utility.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <compression/compressed_scan.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <helper/type_conversions.hpp>
#include <io/io_context.hpp>
#include <late_mat/column_origin.hpp>
#include <late_mat/defer_directive.hpp>
#include <late_mat/port_materialize.hpp>
#include <op/scan/decoded_batch_representation.hpp>
#include <op/scan/detail/read_retirement.hpp>
#include <op/scan/gpu_ingestible.hpp>
#include <op/scan/gpu_ingestible_types.hpp>
#include <op/scan/iceberg_gpu_ingestible.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/scan_output_operator_data.hpp>
#include <op/scan/sirius_gpu_scan_operator.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <op/scan/sirius_physical_dynamic_filter.hpp>
#include <op/sirius_physical_operator.hpp>
#include <pipeline/sirius_pipeline.hpp>
#include <planner/late_mat_plan_pass.hpp>
#include <scan_manager/load_balancing_scan_batch_coalescer.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <scan_manager/split_connector.hpp>
#include <sirius/exception.hpp>
#include <utils/table_filter_test_utils.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
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
  /// its post_filter_and_project stands in for; @p materialized_width is how many leading columns
  /// it materializes, so a nonzero width with a leading identity makes scan_decode_bindings yield
  /// bindings for this ingestible.
  explicit stub_ingestible(table_factory produce,
                           bool leading_identity                          = false,
                           std::optional<std::size_t> output_prefix_width = std::nullopt,
                           std::size_t materialized_width                 = 0)
    : _produce(std::move(produce)),
      _leading_identity(leading_identity),
      _output_prefix_width(output_prefix_width),
      _materialized_width(materialized_width)
  {
  }

  /// Makes post_filter_and_project report every produced row as a survivor of a residual filter.
  void report_survivors() noexcept { _reports_survivors = true; }

  [[nodiscard]] bool can_report_survivors() const noexcept override { return _reports_survivors; }

  std::optional<sirius::expression_query_policy> observed_policy;

  std::unique_ptr<cudf::table> post_filter_and_project(
    sirius::op::scan::filtered_table&&,
    const cucascade::memory::memory_space&,
    ::cuda::stream_ref stream,
    bool,
    std::shared_ptr<const sirius::like_multiliteral_cache>,
    std::unique_ptr<cudf::column>* survivors,
    std::span<std::size_t const> /*elided*/,
    sirius::expression_query_policy policy) override
  {
    observed_policy = policy;
    auto table      = _produce();
    if (_reports_survivors && survivors != nullptr) {
      *survivors = cudf::sequence(
        table->num_rows(), cudf::numeric_scalar<std::int32_t>(0, true, stream), stream);
    }
    return table;
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
    std::shared_ptr<const sirius::like_multiliteral_cache>,
    sirius::expression_query_policy /*policy*/,
    sirius::op::scan::scan_execution_context /*execution*/) override
  {
    throw std::logic_error("stub_ingestible: a resident split never decodes scan metadata");
  }

  [[nodiscard]] const sirius::op::scan::ingestible_table_info& table_info() const noexcept override
  {
    return _info;
  }

  [[nodiscard]] std::vector<std::size_t> materialized_column_order() const override
  {
    std::vector<std::size_t> order(_materialized_width);
    std::iota(order.begin(), order.end(), std::size_t{0});
    return order;
  }

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
  std::size_t _materialized_width;
  bool _reports_survivors = false;
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

/// Serves one batch, then reports exhaustion.
class queued_provider final : public sirius::scan_manager::databatch_provider {
 public:
  batch next;
  batch get_next_batch() override { return std::exchange(next, {}); }
};

/// Filter identity for receipts; nothing here executes it.
class identity_only_filter final : public sirius::op::sirius_dynamic_filter {
 public:
  [[nodiscard]] sirius::op::sirius_dynamic_filter_kind kind() const override
  {
    return sirius::op::sirius_dynamic_filter_kind::IN_LIST;
  }
};

/// Membership filter, replicated on every device, whose every probe declines as a probe of an
/// incompatible type does
class declining_filter final : public sirius::op::sirius_dynamic_filter,
                               public sirius::op::sirius_mask_applicable {
 public:
  [[nodiscard]] sirius::op::sirius_dynamic_filter_kind kind() const override
  {
    return sirius::op::sirius_dynamic_filter_kind::IN_LIST;
  }
  [[nodiscard]] bool is_available_on_device(int) const noexcept override { return true; }
  using sirius::op::sirius_mask_applicable::compute_mask;
  [[nodiscard]] std::unique_ptr<cudf::column> compute_mask(
    cudf::column_view const&,
    std::uint32_t const*,
    int,
    ::cuda::stream_ref,
    rmm::device_async_resource_ref) const override
  {
    return nullptr;
  }
};

/**
 * @brief A two-column BIGINT scan with a consumer and a leading-identity ingestible, and its
 * endpoint, which restores the scan's deferral.
 *
 * The ingestible elides the deferred column at position 1, so it produces only column 0, with
 * `produced_rows` rows.
 */
struct deferring_scan_fixture {
  static constexpr std::size_t kDeferred = 1;

  test_env& e               = env();
  std::size_t produced_rows = kRows;
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> channel =
    std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::shared_ptr<stub_ingestible> ingestible = std::make_shared<stub_ingestible>(
    [this] {
      std::vector<std::unique_ptr<cudf::column>> columns;
      columns.push_back(
        make_column(*e.gpu_space, cudf::data_type{cudf::type_id::INT64}, produced_rows));
      return std::make_unique<cudf::table>(std::move(columns));
    },
    /*leading_identity=*/true,
    /*output_prefix_width=*/std::nullopt,
    /*materialized_width=*/2);
  std::shared_ptr<sirius::op::scan::dynamic_filter_consumer> consumer =
    std::make_shared<sirius::op::scan::dynamic_filter_consumer>(
      channel,
      sirius::op::scan::consumer_config{},
      sirius::op::scan::scan_decode_bindings(*ingestible, 2));
  std::shared_ptr<sirius::late_mat::pin_entry_handle> handle =
    std::make_shared<sirius::late_mat::pin_entry_handle>("deferring_scan", 1);
  sirius::op::scan::sirius_gpu_scan_operator scan{
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT,
                                                                duckdb::LogicalType::BIGINT}),
    0,
    ingestible,
    1,
    nullptr,
    nullptr,
    consumer};
  /// The endpoint, which is also the port that restores the deferral.
  sirius::op::scan::sirius_physical_dynamic_filter port{scan.types, kRows, consumer};

  [[nodiscard]] sirius::late_mat::column_origin origin() const
  {
    sirius::late_mat::column_origin o;
    o.handle     = handle;
    o.column_pos = kDeferred;
    o.generation = handle->generation();
    return o;
  }

  [[nodiscard]] bool install_deferral()
  {
    std::vector<cudf::data_type> const schema(2, cudf::data_type{cudf::type_id::INT64});
    return sirius::planner::install_deferral(
      scan,
      port,
      sirius::late_mat::make_defer_pair(schema, {kDeferred}, schema, {kDeferred}, {origin()}));
  }

  /// A resident split of `produced_rows` rows whose origin covers one pinned chunk of `kRows` rows.
  [[nodiscard]] std::unique_ptr<sirius::op::scan::scan_operator_input> split()
  {
    auto input = std::make_unique<sirius::op::scan::scan_operator_input>(
      make_resident_batch(e, produced_rows));
    admit_fixture_resident_input(*input);
    input->gpu_memory_space = e.gpu_space;
    auto origin             = std::make_shared<sirius::late_mat::scan_batch_origin>();
    origin->columns =
      std::make_shared<std::vector<sirius::late_mat::column_origin>>(2, this->origin());
    origin->range = sirius::late_mat::row_range{0, static_cast<std::int64_t>(kRows)};
    input->origin = std::move(origin);
    return input;
  }
};

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

TEST_CASE("scan emits a receipt with its output identity without training the consumer",
          "[scan_normalization][consumer_integration][scan_receipt]")
{
  using namespace sirius::op::scan;
  auto& e      = env();
  auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto consumer =
    std::make_shared<dynamic_filter_consumer>(channel, consumer_config{}, std::vector<binding>{});
  auto ingestible = std::make_shared<stub_ingestible>([&e] {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(make_column(*e.gpu_space, cudf::data_type{cudf::type_id::INT64}, kRows));
    return std::make_unique<cudf::table>(std::move(columns));
  });
  sirius_gpu_scan_operator scan{
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT}),
    0,
    ingestible,
    1,
    nullptr,
    nullptr,
    consumer};
  REQUIRE(scan.consumer() == consumer);
  REQUIRE(consumer->decode_bindings().empty());
  scan_operator_input input(make_resident_batch(e, kRows));
  admit_fixture_resident_input(input);
  input.gpu_memory_space     = e.gpu_space;
  auto output                = scan.execute(input, e.stream());
  auto const* receipt_output = dynamic_cast<scan_output_operator_data const*>(output.get());
  REQUIRE(receipt_output != nullptr);
  REQUIRE(receipt_output->get_data_batches().size() == 1);
  REQUIRE(receipt_output->receipt().endpoint == channel);
  REQUIRE(receipt_output->receipt().original_batch_id ==
          receipt_output->get_data_batches().front()->get_batch_id());
  REQUIRE(receipt_output->receipt().applied.empty());
  REQUIRE(receipt_output->receipt().decode_attached.empty());
  REQUIRE_FALSE(sirius::op::scan::detail::gate_test_access::applicable(
    sirius::op::scan::detail::consumer_test_access::gate(*consumer), channel->snapshot()));
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

TEST_CASE("scan execute releases restored cast sources without waiting on the host",
          "[scan_normalization][gpu_scan]")
{
  auto& e = env();
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  ::cuda::stream_ref const scan_stream{stream.value()};
  auto batch      = make_resident_batch(e, kRows);
  auto ingestible = std::make_shared<stub_ingestible>([&] {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                static_cast<cudf::size_type>(kRows),
                                                cudf::mask_state::UNALLOCATED,
                                                scan_stream,
                                                e.gpu_space->get_default_allocator()));
    return std::make_unique<cudf::table>(std::move(columns));
  });
  auto scan       = make_bigint_scan(ingestible);
  sirius::op::scan::scan_operator_input input(batch);
  input.gpu_memory_space = e.gpu_space;
  admit_fixture_resident_input(input);
  // Load the kernels execute launches first: loading a kernel's module on its first launch waits
  // for the held stream.
  {
    sirius::op::scan::scan_operator_input warmup(make_resident_batch(e, kRows));
    warmup.gpu_memory_space = e.gpu_space;
    admit_fixture_resident_input(warmup);
    (void)scan.execute(warmup, scan_stream);
    stream.synchronize();
  }

  // Hold the stream until the host releases it. The hold is bounded so that a host wait inside
  // execute ends with the hold expired, not with a hung test.
  std::atomic<bool> released{false};
  auto const hold = [](void* flag) {
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!static_cast<std::atomic<bool>*>(flag)->load() &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
  };
  REQUIRE(cudaLaunchHostFunc(stream.value(), hold, &released) == cudaSuccess);
  auto output = scan.execute(input, scan_stream);
  CHECK(cudaStreamQuery(stream.value()) == cudaErrorNotReady);
  released = true;
  stream.synchronize();

  auto const& batches =
    dynamic_cast<const sirius::op::pipelineable_operator_data&>(*output).get_data_batches();
  REQUIRE(batches.size() == 1);
  auto restored = batches[0]->to_read_only();
  REQUIRE(sirius::get_cudf_table_view(restored).column(0).type().id() == cudf::type_id::INT64);
}

TEST_CASE("scan execute frees each restored cast source before casting the next",
          "[scan_normalization][gpu_scan]")
{
  constexpr std::size_t kColumns  = 3;
  constexpr std::size_t kCastRows = std::size_t{1} << 16;
  auto& e                         = env();
  rmm::mr::statistics_resource_adaptor allocations{
    cuda::mr::any_resource<cuda::mr::device_accessible>{e.gpu_space->get_default_allocator()}};
  cucascade::memory::memory_space counted_space(cucascade::memory::gpu_memory_space_config{
    .device_id       = e.gpu_space->get_device_id(),
    .memory_capacity = e.gpu_space->get_max_memory(),
    .mr_factory_fn   = [&](int, std::size_t) {
      return cuda::mr::any_resource<cuda::mr::device_accessible>{allocations};
    }});
  auto ingestible = std::make_shared<stub_ingestible>([&] {
    std::vector<std::unique_ptr<cudf::column>> columns;
    for (std::size_t column = 0; column < kColumns; ++column) {
      columns.push_back(
        make_column(counted_space, cudf::data_type{cudf::type_id::INT32}, kCastRows));
    }
    return std::make_unique<cudf::table>(std::move(columns));
  });
  sirius::op::scan::sirius_gpu_scan_operator scan{
    sirius::from_duckdb_vec(
      duckdb::vector<duckdb::LogicalType>(kColumns, duckdb::LogicalType::BIGINT)),
    /*estimated_cardinality=*/0,
    ingestible,
    /*contract_id=*/1};
  sirius::op::scan::scan_operator_input input(make_resident_batch(e, kRows));
  input.gpu_memory_space = &counted_space;
  admit_fixture_resident_input(input);

  auto output = scan.execute(input, e.stream());
  e.stream().sync();
  auto const restored = dynamic_cast<sirius::op::pipelineable_operator_data const&>(*output)
                          .get_data_batches()
                          .at(0)
                          ->to_read_only();
  auto const view = sirius::get_cudf_table_view(restored);
  REQUIRE(view.num_columns() == static_cast<cudf::size_type>(kColumns));
  for (auto const& column : view) {
    REQUIRE(column.type().id() == cudf::type_id::INT64);
  }
  // Each cast replaces and frees its source before the next cast allocates, so the peak holds every
  // cast and one source rather than every source.
  auto const source_bytes = kCastRows * sizeof(std::int32_t);
  auto const cast_bytes   = kColumns * kCastRows * sizeof(std::int64_t);
  auto const peak         = static_cast<std::size_t>(allocations.get_bytes_counter().peak);
  CHECK(peak >= cast_bytes + source_bytes);
  CHECK(peak < cast_bytes + 2 * source_bytes);
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
    admit_fixture_resident_input(*input);
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

TEST_CASE("a deferring scan withholds its consumer from decode attachment",
          "[late_mat][dynamic_filter][consumer_integration]")
{
  deferring_scan_fixture f;
  REQUIRE(f.consumer->decode_bindings().size() == 2);
  REQUIRE(f.scan.decode_consumer() == f.consumer);

  SECTION("a deferral with a port") { REQUIRE(f.install_deferral()); }
  SECTION("a count deferral")
  {
    REQUIRE(sirius::planner::install_count_deferral(
      f.scan, sirius::late_mat::deferred_scan_output{{deferring_scan_fixture::kDeferred}}));
  }
  REQUIRE(f.scan.decode_consumer() == nullptr);
  REQUIRE(f.scan.consumer() == f.consumer);

  queued_provider provider;
  provider.next.data              = make_resident_batch(f.e, kRows);
  provider.contract_id            = 1;
  provider.validation.identity    = provider.validation.layout =
    provider.validation.structure = {true, true};
  sirius::scan_manager::load_balancing_scan_batch_coalescer::drain_cached_provider(
    provider, f.scan.get_split_connector(), std::stop_token{}, false, 1, 0);
  auto handed_out   = f.scan.get_next_task_input_data();
  auto const* split = dynamic_cast<sirius::op::scan::scan_operator_input const*>(handed_out.get());
  REQUIRE(split != nullptr);
  REQUIRE(split->consumer == nullptr);
}

TEST_CASE("a deferring scan names a decode that removed rows instead of mis-addressing them",
          "[late_mat][dynamic_filter][consumer_integration]")
{
  deferring_scan_fixture f;
  f.produced_rows = kRows / 2;
  REQUIRE(f.install_deferral());
  auto split = f.split();
  // Each section's message names what that decode reported, so neither passes on the other's.
  std::string reported;
  SECTION("the decode applied the static filter")
  {
    split->pushdown_row_filtered = true;
    reported                     = "(static filter applied: true, membership probes offered: 0)";
  }
  SECTION("the decode was offered membership probes")
  {
    split->decode_attached = {{std::make_shared<identity_only_filter>(), 0}};
    reported               = "(static filter applied: false, membership probes offered: 1)";
  }
  REQUIRE_THROWS_WITH(
    f.scan.execute(*split, f.e.stream()),
    Catch::Matchers::ContainsSubstring(
      "the decode removed rows of a scan whose deferral addresses whole chunks") &&
      Catch::Matchers::ContainsSubstring(reported));
}

TEST_CASE("a deferring scan checks the decoded rows before trusting a residual filter's survivors",
          "[late_mat][dynamic_filter][consumer_integration]")
{
  deferring_scan_fixture f;
  f.ingestible->report_survivors();
  REQUIRE(f.install_deferral());
  SECTION("the decode kept the whole chunk")
  {
    // The survivors address the chunk's rows, so they alone do not fail the scan.
    auto split = f.split();
    REQUIRE_NOTHROW(f.scan.execute(*split, f.e.stream()));
  }
  SECTION("the decode removed rows")
  {
    // Survivor positions would index the decoded rows rather than the chunk's, so producing them
    // must not excuse the decode.
    f.produced_rows        = kRows / 2;
    auto split             = f.split();
    split->decode_attached = {{std::make_shared<identity_only_filter>(), 0}};
    REQUIRE_THROWS_WITH(
      f.scan.execute(*split, f.e.stream()),
      Catch::Matchers::ContainsSubstring(
        "the decode removed rows of a scan whose deferral addresses whole chunks"));
  }
}

TEST_CASE("a deferring scan keeps decode history at deferred ordinals",
          "[late_mat][dynamic_filter][consumer_integration][scan_receipt]")
{
  using sirius::op::scan::applied_entry;
  deferring_scan_fixture f;
  REQUIRE(f.install_deferral());
  auto const deferred_filter = std::make_shared<identity_only_filter>();
  auto const kept_filter     = std::make_shared<identity_only_filter>();
  auto const outside_filter  = std::make_shared<identity_only_filter>();
  auto split                 = f.split();
  split->decode_attached     = {
    {deferred_filter, deferring_scan_fixture::kDeferred}, {kept_filter, 0}, {outside_filter, 5}};
  auto output = f.scan.execute(*split, f.e.stream());
  auto const* scan_output =
    dynamic_cast<sirius::op::scan::scan_output_operator_data const*>(output.get());
  REQUIRE(scan_output != nullptr);
  REQUIRE(scan_output->receipt().decode_attached ==
          std::vector<applied_entry>{{deferred_filter, deferring_scan_fixture::kDeferred},
                                     {kept_filter, 0}});
  auto lease = scan_output->get_data_batches().front()->to_read_only();
  auto view  = sirius::get_cudf_table_view(lease);
  REQUIRE(view.num_columns() == 2);
  REQUIRE(view.column(deferring_scan_fixture::kDeferred).type().id() == cudf::type_id::UINT64);
}

TEST_CASE("decode history at a deferred ordinal keeps the endpoint from sampling restored rows",
          "[late_mat][dynamic_filter][consumer_integration]")
{
  using sirius::op::scan::detail::consumer_test_access;
  using sirius::op::scan::detail::gate_test_access;
  deferring_scan_fixture f;
  auto const stream = f.e.stream();
  auto const mr     = f.e.gpu_space->get_default_allocator();

  // The pin the deferral restores from: its column 1 holds 100 + row.
  sirius::scan_manager::pinned_entry entry;
  entry.tier     = cucascade::memory::Tier::GPU;
  entry.num_rows = kRows;
  std::vector<std::int64_t> values(kRows);
  std::iota(values.begin(), values.end(), std::int64_t{100});
  for (auto const* name : {"a", "b"}) {
    entry.cache_info.names.emplace_back(name);
    std::shared_ptr<cudf::column> column =
      make_column(*f.e.gpu_space, cudf::data_type{cudf::type_id::INT64}, kRows);
    cudaMemcpyAsync(column->mutable_view().data<std::int64_t>(),
                    values.data(),
                    values.size() * sizeof(std::int64_t),
                    cudaMemcpyHostToDevice,
                    stream.get());
    entry.data_batches_by_column.emplace(name, std::vector<std::shared_ptr<cudf::column>>{column});
  }
  stream.sync();
  auto const entry_owner = std::shared_ptr<sirius::scan_manager::pinned_entry const>(
    &entry, [](sirius::scan_manager::pinned_entry const*) {});
  f.handle->set_entry(entry_owner);

  // F keeps every row: the decode already conditioned these rows on it.
  auto producer   = f.channel->register_producer({deferring_scan_fixture::kDeferred});
  auto const keys = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, kRows, cudf::mask_state::UNALLOCATED, stream, mr);
  cudaMemcpyAsync(keys->mutable_view().data<std::int64_t>(),
                  values.data(),
                  values.size() * sizeof(std::int64_t),
                  cudaMemcpyHostToDevice,
                  stream.get());
  auto const filter =
    std::make_shared<sirius::op::sirius_dynamic_small_in_list_filter>(keys->view(), stream, mr);
  stream.sync();
  REQUIRE(producer.push_filter(deferring_scan_fixture::kDeferred, filter));
  REQUIRE(f.install_deferral());

  auto split             = f.split();
  split->decode_attached = {{filter, deferring_scan_fixture::kDeferred}};
  auto scanned           = f.scan.execute(*split, stream);
  auto const& riding     = dynamic_cast<sirius::op::pipelineable_operator_data const&>(*scanned);

  // What materialize_deferred_input does before the endpoint runs.
  auto const leases = riding.get_read_only_batches();
  auto const view   = sirius::get_cudf_table_view(leases.front());
  REQUIRE(sirius::late_mat::port_directive_matches(f.port.port_directive(), view));
  auto restored = sirius::make_data_batch(
    sirius::late_mat::materialize_at_port(f.port.port_directive(), view, stream, mr),
    *f.e.gpu_space,
    stream,
    {});
  auto input = riding.with_batches_preserving_rows({std::move(restored)});
  input->prepare_for_processing(f.e.gpu_space, stream);
  auto output = f.port.execute(*input, stream);
  auto const filtered =
    dynamic_cast<sirius::op::pipelineable_operator_data const&>(*output).get_read_only_batches();
  REQUIRE(sirius::get_cudf_table_view(filtered.front()).num_rows() ==
          static_cast<cudf::size_type>(kRows));

  auto& gate = consumer_test_access::gate(*f.consumer);
  REQUIRE(gate_test_access::needs_combined_sample(gate, f.channel->snapshot()));
  REQUIRE_FALSE(
    gate_test_access::filter_keep_ratio(gate, {filter.get(), deferring_scan_fixture::kDeferred}, 1)
      .has_value());
}

namespace {

namespace scan = sirius::op::scan;

std::unique_ptr<scan::parquet_ingestible_table_info> resident_parquet_info(
  std::vector<std::size_t> const& columns,
  std::size_t output_arity,
  std::optional<std::size_t> positive_filter = std::nullopt)
{
  auto info   = std::make_unique<scan::parquet_ingestible_table_info>();
  info->names = {"key", "payload", "keep"};
  for (std::size_t i = 0; i < info->names.size(); ++i) {
    info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
  }
  for (auto column : columns) {
    info->column_ids.emplace_back(column);
    info->projection_ids.push_back(info->projection_ids.size());
  }
  info->projected_output_count = output_arity;
  if (positive_filter) {
    info->table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
    info->table_filters->PushFilter(
      duckdb::ProjectionIndex(*positive_filter),
      sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_GREATERTHAN,
                                    duckdb::Value::BIGINT(0)));
  }
  return info;
}

template <typename T>
std::unique_ptr<cudf::column> resident_column(
  std::vector<T> const& values,
  cudf::type_id type,
  std::optional<cudf::size_type> null_row = std::nullopt)
{
  auto& e     = env();
  auto column = cudf::make_numeric_column(
    cudf::data_type{type},
    static_cast<cudf::size_type>(values.size()),
    null_row ? cudf::mask_state::ALL_VALID : cudf::mask_state::UNALLOCATED,
    e.stream(),
    e.gpu_space->get_default_allocator());
  if (!values.empty()) {
    REQUIRE(cudaMemcpyAsync(column->mutable_view().data<T>(),
                            values.data(),
                            values.size() * sizeof(T),
                            cudaMemcpyHostToDevice,
                            e.stream().get()) == cudaSuccess);
  }
  if (null_row) {
    cudf::set_null_mask(
      column->mutable_view().null_mask(), *null_row, *null_row + 1, false, e.stream());
    column->set_null_count(1);
  }
  e.stream().sync();
  return column;
}

std::shared_ptr<cucascade::data_batch> resident_columns(
  std::vector<std::unique_ptr<cudf::column>> columns)
{
  auto& e = env();
  return sirius::make_data_batch(
    std::make_unique<cudf::table>(std::move(columns)), *e.gpu_space, e.stream(), {});
}

template <typename T>
void require_resident_values(cudf::column_view column,
                             cudf::type_id type,
                             std::vector<T> const& expected,
                             std::vector<char> validity = {})
{
  REQUIRE(column.type().id() == type);
  REQUIRE(column.size() == static_cast<cudf::size_type>(expected.size()));
  if (validity.empty()) { validity.assign(expected.size(), 1); }
  REQUIRE(sirius::test::detail::column_row_validity(column, env().stream()) == validity);
  auto const values =
    sirius::test::detail::copy_device_values<T>(column.data<T>(), expected.size(), env().stream());
  for (std::size_t row = 0; row < expected.size(); ++row) {
    CAPTURE(row);
    if (validity[row]) { CHECK(values[row] == expected[row]); }
  }
}

struct resident_parquet_fixture {
  test_env& e = env();
  sirius::op::dynamic_filter_stats stats;
  std::shared_ptr<scan::parquet_gpu_ingestible> ingestible;
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> channel =
    std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  std::shared_ptr<scan::dynamic_filter_consumer> consumer;
  scan::sirius_gpu_scan_operator op;

  resident_parquet_fixture(std::unique_ptr<scan::parquet_ingestible_table_info> info,
                           std::size_t output_arity,
                           bool cached              = true,
                           duckdb::LogicalType type = duckdb::LogicalType::BIGINT,
                           bool force_counts        = true,
                           bool fused_membership    = true)
    : resident_parquet_fixture(std::move(info),
                               duckdb::vector<duckdb::LogicalType>(output_arity, type),
                               cached,
                               force_counts,
                               fused_membership)
  {
  }

  resident_parquet_fixture(std::unique_ptr<scan::parquet_ingestible_table_info> info,
                           duckdb::vector<duckdb::LogicalType> const& output_types,
                           bool cached           = true,
                           bool force_counts     = true,
                           bool fused_membership = true)
    : ingestible(scan::make_ingestible(std::move(info))),
      consumer(std::make_shared<scan::dynamic_filter_consumer>(
        channel,
        scan::consumer_config{.mode = scan::dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY,
                              .force_counts     = force_counts,
                              .cached_placement = cached,
                              .fused_membership = fused_membership,
                              .stats            = &stats},
        scan::scan_decode_bindings(*ingestible, output_types.size()))),
      op(sirius::from_duckdb_vec(output_types), 0, ingestible, 1, nullptr, nullptr, consumer)
  {
  }

  auto publish(std::size_t ordinal, std::vector<std::int64_t> const& values)
  {
    auto keys   = resident_column(values, cudf::type_id::INT64);
    auto filter = std::make_shared<sirius::op::sirius_dynamic_small_in_list_filter>(
      keys->view(), e.stream(), e.gpu_space->get_default_allocator());
    e.stream().sync();
    auto producer = channel->register_producer({ordinal});
    REQUIRE(producer.push_filter(ordinal, filter));
    return filter;
  }

  auto split(std::shared_ptr<cucascade::data_batch> const& batch, bool conversion = true)
  {
    auto input = std::make_unique<scan::scan_operator_input>(batch);
    admit_fixture_resident_input(*input);
    input->row_filter_pending       = ingestible->has_row_filter();
    input->needs_carrier_conversion = conversion;
    input->cached_view_possible     = op.cached_view_possible();
    input->prepare_for_processing(e.gpu_space, e.stream());
    return input;
  }
};

}  // namespace

TEST_CASE("resident Parquet filters a shared narrow projection before restoring duplicate outputs",
          "[scan_normalization][cached_scan][cached_projection]")
{
  auto const fused = GENERATE(false, true);
  bool cached      = true;
  SECTION("cached placement") {}
  SECTION("placement disabled") { cached = false; }
  resident_parquet_fixture f(
    resident_parquet_info({1, 0, 1, 2}, 3, 3), 3, cached, duckdb::LogicalType::BIGINT, true, fused);
  f.publish(1, {41, 83, 29, 53});
  REQUIRE(f.ingestible->materialized_column_order() == std::vector<std::size_t>{1, 0, 2});
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(resident_column<std::int32_t>(
    {901, 302, 703, 404, 805, 106, 607, 208}, cudf::type_id::INT32, 3));
  columns.push_back(
    resident_column<std::int32_t>({41, 17, 83, 29, 61, 11, 97, 53}, cudf::type_id::INT32));
  columns.push_back(resident_column<std::int32_t>({1, 1, 0, 1, 1, 0, 1, 1}, cudf::type_id::INT32));
  auto source                                          = resident_columns(std::move(columns));
  std::weak_ptr<cucascade::data_batch> source_lifetime = source;
  auto input                                           = f.split(source);
  REQUIRE_FALSE(input->stolen_table);
  REQUIRE_FALSE(input->converted_table_steal_pending);
  {
    auto materialized = f.ingestible->materialize_table(*input, f.e.stream());
    auto layout       = f.ingestible->view_layout(materialized, {});
    REQUIRE(layout);
    CHECK(layout->output_columns == std::vector<cudf::size_type>{0, 1, 0});
    CHECK(layout->projected_columns == layout->output_columns);
    REQUIRE(layout->residual);
  }
  auto scanned = f.op.execute(*input, f.e.stream());
  {
    auto source_lease = source->to_read_only();
    auto view         = sirius::get_cudf_table_view(source_lease);
    REQUIRE(view.num_columns() == 3);
    require_resident_values<std::int32_t>(view.column(0),
                                          cudf::type_id::INT32,
                                          {901, 302, 703, 404, 805, 106, 607, 208},
                                          {1, 1, 1, 0, 1, 1, 1, 1});
    require_resident_values<std::int32_t>(
      view.column(1), cudf::type_id::INT32, {41, 17, 83, 29, 61, 11, 97, 53});
    auto leases =
      dynamic_cast<sirius::op::pipelineable_operator_data const&>(*scanned).get_read_only_batches();
    REQUIRE(sirius::get_cudf_table_view(leases.at(0)).num_rows() == (cached ? 3 : 6));
  }
  input.reset();
  source.reset();
  REQUIRE(source_lifetime.expired());
  scan::sirius_physical_dynamic_filter endpoint(f.op.types, kRows, f.consumer);
  scanned->prepare_for_processing(f.e.gpu_space, f.e.stream());
  auto result = endpoint.execute(*scanned, f.e.stream());
  auto leases =
    dynamic_cast<sirius::op::pipelineable_operator_data const&>(*result).get_read_only_batches();
  auto view = sirius::get_cudf_table_view(leases.at(0));
  REQUIRE(view.num_columns() == 3);
  require_resident_values<std::int64_t>(
    view.column(0), cudf::type_id::INT64, {901, 0, 208}, {1, 0, 1});
  require_resident_values<std::int64_t>(view.column(1), cudf::type_id::INT64, {41, 29, 53});
  require_resident_values<std::int64_t>(
    view.column(2), cudf::type_id::INT64, {901, 0, 208}, {1, 0, 1});
  CHECK(view.column(0).head() != view.column(2).head());
  auto const counts = f.stats.snapshot();
  CHECK(counts.cached_placements == (cached ? 1 : 0));
  CHECK((counts.cached_payload_bytes > 0) == cached);
  CHECK(counts.cached_disabled == (cached ? 0 : 1));
  CHECK(counts.cached_bypassed_transfer == 0);
}

TEST_CASE("resident Parquet checkpoint whose probes all decline serves the ordinary projection",
          "[scan_normalization][cached_scan]")
{
  resident_parquet_fixture f(resident_parquet_info({0, 1}, 2), 2);
  auto producer = f.channel->register_producer({0});
  REQUIRE(producer.push_filter(0, std::make_shared<declining_filter>()));
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(resident_column<std::int64_t>({41, 29, 53}, cudf::type_id::INT64));
  columns.push_back(resident_column<std::int64_t>({901, 404, 208}, cudf::type_id::INT64));
  auto source         = resident_columns(std::move(columns));
  auto input          = f.split(source, false);
  auto output         = f.op.execute(*input, f.e.stream());
  auto const& scanned = dynamic_cast<scan::scan_output_operator_data const&>(*output);
  CHECK(scanned.receipt().applied.empty());
  auto leases     = scanned.get_read_only_batches();
  auto const view = sirius::get_cudf_table_view(leases.at(0));
  REQUIRE(view.num_columns() == 2);
  require_resident_values<std::int64_t>(view.column(0), cudf::type_id::INT64, {41, 29, 53});
  require_resident_values<std::int64_t>(view.column(1), cudf::type_id::INT64, {901, 404, 208});
  // The program ran and selected nothing, so the placement declined for want of membership.
  auto const counts = f.stats.snapshot();
  CHECK(counts.cached_placements == 0);
  CHECK(counts.cached_declined_no_membership == 1);
  CHECK(counts.cached_declined_no_layout == 0);
  CHECK(counts.applications_skipped_other_device == 0);
}

TEST_CASE("resident Parquet residual without a probing membership filter counts no placement",
          "[scan_normalization][cached_scan]")
{
  resident_parquet_fixture f(resident_parquet_info({0, 1, 2}, 2, 2), 2);
  auto producer = f.channel->register_producer({0});
  // Prepare selects no membership step, so the ordinary residual path filters the batch.
  SECTION("no published filter can probe the batch")
  {
    REQUIRE(producer.push_filter(0, std::make_shared<identity_only_filter>()));
  }
  // The checkpoint runs, but only the residual selects rows.
  SECTION("every membership probe declines")
  {
    REQUIRE(producer.push_filter(0, std::make_shared<declining_filter>()));
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(resident_column<std::int64_t>({41, 29, 53}, cudf::type_id::INT64));
  columns.push_back(resident_column<std::int64_t>({901, 404, 208}, cudf::type_id::INT64));
  columns.push_back(resident_column<std::int64_t>({1, 0, 1}, cudf::type_id::INT64));
  auto source         = resident_columns(std::move(columns));
  auto input          = f.split(source, false);
  auto output         = f.op.execute(*input, f.e.stream());
  auto const& scanned = dynamic_cast<scan::scan_output_operator_data const&>(*output);
  CHECK(scanned.receipt().applied.empty());
  auto leases     = scanned.get_read_only_batches();
  auto const view = sirius::get_cudf_table_view(leases.at(0));
  REQUIRE(view.num_columns() == 2);
  require_resident_values<std::int64_t>(view.column(0), cudf::type_id::INT64, {41, 53});
  require_resident_values<std::int64_t>(view.column(1), cudf::type_id::INT64, {901, 208});
  auto const counts = f.stats.snapshot();
  CHECK(counts.cached_declined_no_membership == 1);
  CHECK(counts.cached_placements == 0);
  CHECK(counts.cached_payload_bytes == 0);
  CHECK(counts.cached_declined_no_layout == 0);
}

TEST_CASE("resident Parquet checkpoint sizes its skip policy by the projected row width",
          "[scan_normalization][cached_scan]")
{
  // Eight stored BIGINT columns take 64 bytes a row, but the two the scan projects take 16, the row
  // width the checkpoint's gather moves. At that narrow width a filter measured to keep 55% is
  // skipped.
  constexpr std::size_t kStored = 8;
  auto info                     = std::make_unique<scan::parquet_ingestible_table_info>();
  for (std::size_t column = 0; column < kStored; ++column) {
    info->names.push_back("c" + std::to_string(column));
    info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
    info->column_ids.emplace_back(column);
    info->projection_ids.push_back(column);
  }
  info->projected_output_count = 2;
  resident_parquet_fixture f(std::move(info), 2);
  auto filter = f.publish(0, {41, 53});
  scan::detail::consumer_test_access::gate(*f.consumer)
    .record_filter_keep_ratio({filter.get(), 0}, 0.55, 1);
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (std::size_t column = 0; column < kStored; ++column) {
    columns.push_back(resident_column<std::int64_t>({41, 29, 53}, cudf::type_id::INT64));
  }
  auto source = resident_columns(std::move(columns));
  auto input  = f.split(source, false);
  auto output = f.op.execute(*input, f.e.stream());
  auto leases =
    dynamic_cast<scan::scan_output_operator_data const&>(*output).get_read_only_batches();
  CHECK(sirius::get_cudf_table_view(leases.at(0)).num_rows() == 3);
  auto const counts = f.stats.snapshot();
  CHECK(counts.cached_placements == 0);
  CHECK(counts.cached_declined_no_membership == 1);
}

TEST_CASE("resident Parquet checkpoint counts a program it cannot prepare as missing membership",
          "[scan_normalization][cached_scan]")
{
  resident_parquet_fixture f(resident_parquet_info({0, 1}, 2), 2);
  SECTION("no filter published") {}
  SECTION("gate disabled until the channel grows")
  {
    (void)f.publish(0, {41, 53});
    scan::detail::consumer_test_access::gate(*f.consumer)
      .record_keep_ratio(10, 10, f.channel->snapshot().generation());
  }
  SECTION("no published filter can probe the batch")
  {
    auto producer = f.channel->register_producer({0});
    REQUIRE(producer.push_filter(0, std::make_shared<identity_only_filter>()));
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(resident_column<std::int64_t>({41, 29, 53}, cudf::type_id::INT64));
  columns.push_back(resident_column<std::int64_t>({901, 404, 208}, cudf::type_id::INT64));
  auto source = resident_columns(std::move(columns));
  auto input  = f.split(source, false);
  auto output = f.op.execute(*input, f.e.stream());
  auto leases =
    dynamic_cast<scan::scan_output_operator_data const&>(*output).get_read_only_batches();
  CHECK(sirius::get_cudf_table_view(leases.at(0)).num_rows() == 3);
  auto const counts = f.stats.snapshot();
  CHECK(counts.cached_placements == 0);
  CHECK(counts.cached_disabled == 0);
  CHECK(counts.cached_declined_no_membership == 1);
}

TEST_CASE("resident Parquet split on another device passes the checkpoint without a decline",
          "[scan_normalization][cached_scan]")
{
  int devices = 0;
  int current = -1;
  REQUIRE(cudaGetDeviceCount(&devices) == cudaSuccess);
  REQUIRE(cudaGetDevice(&current) == cudaSuccess);
  if (devices < 2) { SKIP("naming another device needs a second visible GPU"); }
  resident_parquet_fixture f(resident_parquet_info({0, 1}, 2), 2);
  auto producer = f.channel->register_producer({0});
  REQUIRE(producer.push_filter(0, std::make_shared<declining_filter>()));
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(resident_column<std::int64_t>({41, 29, 53}, cudf::type_id::INT64));
  columns.push_back(resident_column<std::int64_t>({901, 404, 208}, cudf::type_id::INT64));
  auto source = resident_columns(std::move(columns));
  auto input  = f.split(source, false);
  // Names the other GPU but allocates on this one, so only the device the program is prepared for
  // differs.
  cucascade::memory::memory_space elsewhere(cucascade::memory::gpu_memory_space_config{
    .device_id       = current == 0 ? 1 : 0,
    .memory_capacity = f.e.gpu_space->get_max_memory(),
    .mr_factory_fn   = [&](int, std::size_t) {
      return cuda::mr::any_resource<cuda::mr::device_accessible>{
        f.e.gpu_space->get_default_allocator()};
    }});
  input->gpu_memory_space = &elsewhere;
  auto output             = f.op.execute(*input, f.e.stream());
  auto const& scanned     = dynamic_cast<scan::scan_output_operator_data const&>(*output);
  CHECK(scanned.receipt().applied.empty());
  auto leases = scanned.get_read_only_batches();
  CHECK(sirius::get_cudf_table_view(leases.at(0)).num_rows() == 3);
  auto const counts = f.stats.snapshot();
  CHECK(counts.applications_skipped_other_device == 1);
  CHECK(counts.cached_placements == 0);
  CHECK(counts.cached_declined_no_membership == 0);
}

TEST_CASE("resident Parquet accepts proven dropped filter columns and rejects missing proof",
          "[scan_normalization][cached_scan][cached_layout]")
{
  resident_parquet_fixture f(resident_parquet_info({0, 1, 2}, 2, 2), 2);
  f.publish(0, {41, 53});
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(resident_column<std::int32_t>({41, 29, 53}, cudf::type_id::INT32));
  columns.push_back(resident_column<std::int32_t>({901, 404, 208}, cudf::type_id::INT32));
  auto source                                 = resident_columns(std::move(columns));
  auto input                                  = f.split(source);
  input->pushdown_filter_only_columns_dropped = true;
  SECTION("missing full-filter proof fails before cached application")
  {
    REQUIRE_THROWS_WITH(
      f.op.execute(*input, f.e.stream()),
      Catch::Matchers::ContainsSubstring("without applying the scan's whole filter"));
    CHECK(f.stats.snapshot().cached_placements == 0);
  }
  SECTION("full-filter proof admits the surviving output prefix")
  {
    input->pushdown_row_filtered = true;
    auto output                  = f.op.execute(*input, f.e.stream());
    auto leases =
      dynamic_cast<sirius::op::pipelineable_operator_data const&>(*output).get_read_only_batches();
    auto view = sirius::get_cudf_table_view(leases.at(0));
    REQUIRE(view.num_columns() == 2);
    require_resident_values<std::int64_t>(view.column(0), cudf::type_id::INT64, {41, 53});
    require_resident_values<std::int64_t>(view.column(1), cudf::type_id::INT64, {901, 208});
    CHECK(f.stats.snapshot().cached_placements == 1);
    auto original = source->to_read_only();
    require_resident_values<std::int32_t>(
      sirius::get_cudf_table_view(original).column(0), cudf::type_id::INT32, {41, 29, 53});
  }
}

TEST_CASE("resident Parquet uses a BOOL8 pure-filter answer only as a residual",
          "[scan_normalization][cached_scan][cached_layout]")
{
  // The residual is nullable, which the ordinary fold and the fused rounds resolve separately.
  auto const fused = GENERATE(false, true);
  resident_parquet_fixture f(
    resident_parquet_info({0, 1}, 1, 1), 1, true, duckdb::LogicalType::BIGINT, true, fused);
  f.publish(0, {41, 29, 53});
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(resident_column<std::int32_t>({41, 29, 53, 17}, cudf::type_id::INT32));
  columns.push_back(resident_column<std::int8_t>({1, 0, 1, 1}, cudf::type_id::BOOL8, 2));
  auto source                       = resident_columns(std::move(columns));
  auto input                        = f.split(source);
  input->pushdown_predicate_columns = {1};
  {
    auto materialized = f.ingestible->materialize_table(*input, f.e.stream());
    auto layout       = f.ingestible->view_layout(materialized, {});
    REQUIRE(layout);
    CHECK(layout->output_columns == std::vector<cudf::size_type>{0});
    REQUIRE(layout->residual);
    auto projecting_predicate = scan::make_ingestible(resident_parquet_info({0, 1}, 2, 1));
    CHECK_FALSE(projecting_predicate->view_layout(materialized, {}));
  }
  auto output = f.op.execute(*input, f.e.stream());
  auto leases =
    dynamic_cast<sirius::op::pipelineable_operator_data const&>(*output).get_read_only_batches();
  auto view = sirius::get_cudf_table_view(leases.at(0));
  REQUIRE(view.num_columns() == 1);
  require_resident_values<std::int64_t>(view.column(0), cudf::type_id::INT64, {41});
  CHECK(f.stats.snapshot().cached_placements == 1);
  auto original = source->to_read_only();
  require_resident_values<std::int8_t>(sirius::get_cudf_table_view(original).column(1),
                                       cudf::type_id::BOOL8,
                                       {1, 0, 1, 1},
                                       {1, 1, 0, 1});
}

TEST_CASE("resident Parquet declines layouts without a borrowed DATA proof",
          "[scan_normalization][cached_scan][cached_layout]")
{
  auto info        = resident_parquet_info({0}, 1);
  bool iceberg     = false;
  bool unknown     = false;
  bool zero_output = false;
  // Only a missing column is a per-batch decline; every other layout declines for the whole plan.
  bool plan_admits = false;
  SECTION("partition output") { info->partition_indices = {duckdb::HivePartitioningIndex("7", 0)}; }
  SECTION("virtual filename output")
  {
    info->returned_types[0] = sirius::logical_type::make(sirius::type_id::VARCHAR);
    info->virtual_columns.push_back({0,
                                     "filename",
                                     info->returned_types[0],
                                     scan::scan_plan::parquet_virtual_column_kind::FILENAME});
  }
  SECTION("Iceberg does not inherit Parquet admission") { iceberg = true; }
  SECTION("an unknown ingestible has no default layout") { unknown = true; }
  SECTION("missing materialized column")
  {
    info        = resident_parquet_info({0, 1}, 2);
    plan_admits = true;
  }
  SECTION("zero outputs retain a row-count carrier on the ordinary path")
  {
    info        = resident_parquet_info({0}, 0);
    zero_output = true;
  }
  std::shared_ptr<scan::gpu_ingestible> ingestible;
  if (iceberg) {
    auto ice                    = std::make_unique<scan::iceberg_ingestible_table_info>();
    ice->names                  = {"key"};
    ice->returned_types         = {sirius::logical_type::make(sirius::type_id::BIGINT)};
    ice->column_ids             = {duckdb::ColumnIndex(0)};
    ice->projected_output_count = 1;
    ice->delete_data            = std::make_shared<scan::IcebergDeleteData>();
    ingestible                  = std::make_shared<scan::iceberg_gpu_ingestible>(std::move(ice));
  } else if (unknown) {
    ingestible = std::make_shared<stub_ingestible>([] { return std::make_unique<cudf::table>(); });
  } else {
    ingestible = scan::make_ingestible(std::move(info));
  }
  auto& e = env();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(resident_column<std::int32_t>({41, 53}, cudf::type_id::INT32));
  auto source = resident_columns(std::move(columns));
  scan::scan_operator_input input(source);
  input.gpu_memory_space = e.gpu_space;
  auto materialized      = ingestible->materialize_table(input, e.stream());
  CHECK(ingestible->can_serve_cached_view() == plan_admits);
  CHECK_FALSE(ingestible->view_layout(materialized, {}));
  REQUIRE(materialized.table.num_rows() == 2);
  if (zero_output) {
    auto output = ingestible->post_filter_and_project(
      std::move(materialized), *e.gpu_space, e.stream(), false, nullptr, nullptr, {});
    REQUIRE(output->num_columns() == 1);
    require_resident_values<std::int32_t>(output->view().column(0), cudf::type_id::INT32, {41, 53});
  }
}

TEST_CASE("Parquet projection moves unique owned outputs and copies only kept repeats",
          "[scan_normalization][cached_projection]")
{
  auto& e         = env();
  auto ingestible = scan::make_ingestible(resident_parquet_info({1, 0, 1}, 3));
  REQUIRE(ingestible->materialized_column_order() == std::vector<std::size_t>{1, 0});
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(resident_column<std::int64_t>({901, 404, 208}, cudf::type_id::INT64));
  columns.push_back(resident_column<std::int64_t>({41, 29, 53}, cudf::type_id::INT64));
  auto const payload = columns[0]->view().head();
  auto const key     = columns[1]->view().head();
  scan::filtered_table input{
    scan::owning_table_view{std::make_unique<cudf::table>(std::move(columns))},
    scan::filter_state::UNFILTERED};
  std::vector<std::size_t> elided;
  SECTION("a kept repeat is a separate copy") {}
  SECTION("an elided repeat is never copied") { elided = {2}; }
  rmm::mr::statistics_resource_adaptor allocations{
    cuda::mr::any_resource<cuda::mr::device_accessible>{e.gpu_space->get_default_allocator()}};
  cucascade::memory::memory_space counted_space(cucascade::memory::gpu_memory_space_config{
    .device_id       = e.gpu_space->get_device_id(),
    .memory_capacity = e.gpu_space->get_max_memory(),
    .mr_factory_fn   = [&](int, std::size_t) {
      return cuda::mr::any_resource<cuda::mr::device_accessible>{allocations};
    }});
  auto output = ingestible->post_filter_and_project(
    std::move(input), counted_space, e.stream(), false, nullptr, nullptr, elided);
  // Unique outputs move, so the kept repeat's copy is the projection's only allocation.
  CHECK(allocations.get_allocations_counter().total == (elided.empty() ? 1 : 0));
  auto const view = output->view();
  REQUIRE(view.num_columns() == (elided.empty() ? 3 : 2));
  CHECK(view.column(0).head() == payload);
  CHECK(view.column(1).head() == key);
  require_resident_values<std::int64_t>(view.column(0), cudf::type_id::INT64, {901, 404, 208});
  require_resident_values<std::int64_t>(view.column(1), cudf::type_id::INT64, {41, 29, 53});
  if (elided.empty()) {
    CHECK(view.column(2).head() != payload);
    require_resident_values<std::int64_t>(view.column(2), cudf::type_id::INT64, {901, 404, 208});
  }
}

TEST_CASE("fresh HOST conversion refused by the transfer branch uses cached Parquet placement",
          "[scan_normalization][cached_scan][cached_transfer]")
{
  resident_parquet_fixture f(resident_parquet_info({0, 1, 2}, 2, 2), 2);
  f.publish(0, {41, 53});
  auto source = make_host_resident_batch(f.e, {{41, 29, 53}, {901, 404, 208}, {1, 1, 1}});
  scan::scan_operator_input input(source);
  admit_fixture_resident_input(input);
  input.needs_carrier_conversion = true;
  input.row_filter_pending       = true;
  input.pushdown_row_filtered    = true;
  input.cached_view_possible     = f.op.cached_view_possible();
  input.prepare_for_processing(f.e.gpu_space, f.e.stream());
  REQUIRE(input.converted_table_steal_pending);
  REQUIRE_FALSE(input.stolen_table);
  auto output = f.op.execute(input, f.e.stream());
  REQUIRE_FALSE(input.stolen_table_consumed);
  auto leases =
    dynamic_cast<sirius::op::pipelineable_operator_data const&>(*output).get_read_only_batches();
  auto view = sirius::get_cudf_table_view(leases.at(0));
  REQUIRE(view.num_columns() == 2);
  require_resident_values<std::int64_t>(view.column(0), cudf::type_id::INT64, {41, 53});
  require_resident_values<std::int64_t>(view.column(1), cudf::type_id::INT64, {901, 208});
  auto original      = source->to_read_only();
  auto original_view = sirius::get_cudf_table_view(original);
  REQUIRE(original_view.num_columns() == 3);
  require_resident_values<std::int32_t>(
    original_view.column(0), cudf::type_id::INT32, {41, 29, 53});
  require_resident_values<std::int32_t>(
    original_view.column(1), cudf::type_id::INT32, {901, 404, 208});
  auto counts = f.stats.snapshot();
  CHECK(counts.cached_placements == 1);
  CHECK(counts.cached_payload_bytes > 0);
  CHECK(counts.cached_bypassed_transfer == 0);
}

TEST_CASE("owned Parquet transfers bypass cached placement and remain filterable at the endpoint",
          "[scan_normalization][cached_scan][cached_transfer]")
{
  bool transactional = false;
  SECTION("direct owned table") {}
  SECTION("transactional carrier restoration") { transactional = true; }
  auto const logical = transactional ? duckdb::LogicalType::BIGINT : duckdb::LogicalType::INTEGER;
  resident_parquet_fixture f(resident_parquet_info({0, 1}, 2), 2, true, logical);
  f.publish(0, {41, 53});
  auto source = make_host_resident_batch(f.e, {{41, 29, 53}, {901, 404, 208}});
  auto input  = f.split(source, transactional);
  REQUIRE(input->converted_table_steal_pending == transactional);
  REQUIRE(static_cast<bool>(input->stolen_table) == !transactional);
  auto scanned = f.op.execute(*input, f.e.stream());
  REQUIRE(input->stolen_table_consumed);
  REQUIRE_FALSE(input->converted_table_steal_pending);
  {
    auto original = source->to_read_only();
    CHECK(original.get_data()->get_size_in_bytes() == 0);
    auto leases =
      dynamic_cast<sirius::op::pipelineable_operator_data const&>(*scanned).get_read_only_batches();
    REQUIRE(sirius::get_cudf_table_view(leases.at(0)).num_rows() == 3);
  }
  input.reset();
  source.reset();
  scan::sirius_physical_dynamic_filter endpoint(f.op.types, kRows, f.consumer);
  scanned->prepare_for_processing(f.e.gpu_space, f.e.stream());
  auto output = endpoint.execute(*scanned, f.e.stream());
  auto leases =
    dynamic_cast<sirius::op::pipelineable_operator_data const&>(*output).get_read_only_batches();
  auto view = sirius::get_cudf_table_view(leases.at(0));
  REQUIRE(view.num_columns() == 2);
  if (transactional) {
    require_resident_values<std::int64_t>(view.column(0), cudf::type_id::INT64, {41, 53});
    require_resident_values<std::int64_t>(view.column(1), cudf::type_id::INT64, {901, 208});
  } else {
    require_resident_values<std::int32_t>(view.column(0), cudf::type_id::INT32, {41, 53});
    require_resident_values<std::int32_t>(view.column(1), cudf::type_id::INT32, {901, 208});
  }
  auto counts = f.stats.snapshot();
  CHECK(counts.cached_bypassed_transfer == 1);
  CHECK(counts.cached_placements == 0);
  CHECK(counts.cached_payload_bytes == 0);
}

TEST_CASE("cached Parquet deferral validates original rows and restores source-relative survivors",
          "[scan_normalization][cached_scan][cached_late_mat][assembly_fault][late_mat]")
{
  // Positions and retained assembly owners must not depend on how the membership mask was formed.
  auto const fused = GENERATE(false, true);
  resident_parquet_fixture f(
    resident_parquet_info({0, 1, 2}, 2, 2), 2, true, duckdb::LogicalType::BIGINT, true, fused);
  auto filter             = f.publish(0, {41, 83, 29, 53});
  auto handle             = std::make_shared<sirius::late_mat::pin_entry_handle>("resident", 1);
  auto entry              = std::make_shared<sirius::scan_manager::pinned_entry>();
  entry->tier             = cucascade::memory::Tier::GPU;
  entry->num_rows         = 11;
  entry->cache_info.names = {"key", "payload"};
  std::shared_ptr<cudf::column> pinned_keys =
    resident_column<std::int64_t>({5, 7, 9, 41, 17, 83, 29, 61, 11, 97, 53}, cudf::type_id::INT64);
  std::shared_ptr<cudf::column> pinned_payload = resident_column<std::int64_t>(
    {999, 888, 777, 901, 302, 703, 404, 805, 106, 607, 208}, cudf::type_id::INT64);
  entry->data_batches_by_column.emplace("key", std::vector{pinned_keys});
  entry->data_batches_by_column.emplace("payload", std::vector{pinned_payload});
  handle->set_entry(entry);
  sirius::late_mat::column_origin origin;
  origin.handle     = handle;
  origin.column_pos = 1;
  origin.generation = handle->generation();
  scan::sirius_physical_dynamic_filter port(f.op.types, kRows, f.consumer);
  std::vector<cudf::data_type> const schema(2, cudf::data_type{cudf::type_id::INT64});
  REQUIRE(sirius::planner::install_deferral(
    f.op, port, sirius::late_mat::make_defer_pair(schema, {1}, schema, {1}, {origin})));
  REQUIRE_FALSE(f.op.decode_consumer());
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
    resident_column<std::int32_t>({41, 17, 83, 29, 61, 11, 97, 53}, cudf::type_id::INT32));
  columns.push_back(
    resident_column<std::int32_t>({901, 302, 703, 404, 805, 106, 607, 208}, cudf::type_id::INT32));
  columns.push_back(resident_column<std::int32_t>({1, 1, 0, 1, 1, 0, 1, 1}, cudf::type_id::INT32));
  auto source           = resident_columns(std::move(columns));
  auto input            = f.split(source);
  auto batch_origin     = std::make_shared<sirius::late_mat::scan_batch_origin>();
  batch_origin->columns = std::make_shared<std::vector<sirius::late_mat::column_origin>>(2, origin);
  batch_origin->range   = {3, static_cast<std::int64_t>(kRows)};
  input->origin         = batch_origin;
  SECTION("missing origin fails before residual submission")
  {
    input->origin.reset();
    scan::detail::read_retirement_test_hooks fault(scan::detail::read_checkpoint::RESIDUAL);
    REQUIRE_THROWS_WITH(f.op.execute(*input, f.e.stream()),
                        Catch::Matchers::ContainsSubstring("this split carries no origin"));
    CHECK(f.stats.snapshot().cached_placements == 0);
  }
  SECTION("wrong original row count fails before residual submission")
  {
    batch_origin->range.rows = kRows + 1;
    scan::detail::read_retirement_test_hooks fault(scan::detail::read_checkpoint::RESIDUAL);
    REQUIRE_THROWS_WITH(f.op.execute(*input, f.e.stream()),
                        Catch::Matchers::ContainsSubstring("the decode removed rows"));
    CHECK(f.stats.snapshot().cached_position_selections == 0);
  }
  SECTION("assembly failure keeps survivor and cast storage until checked retirement")
  {
    using scan::detail::read_checkpoint;
    auto const point     = GENERATE(read_checkpoint::SUBSTITUTION, read_checkpoint::NORMALIZATION);
    auto const fail_join = GENERATE(false, true);
    CAPTURE(point, fail_join);
    auto& gate = scan::detail::consumer_test_access::gate(*f.consumer);
    std::unique_ptr<sirius::op::operator_data> failed_output;
    {
      scan::detail::read_retirement_test_hooks fault(point, fail_join);
      if (fail_join) {
        REQUIRE_THROWS_AS(failed_output = f.op.execute(*input, f.e.stream()),
                          sirius::unjoined_gpu_work);
        REQUIRE(fault.retained.has_value());
        // The retained state answers only as the scan assembly's own type.
        REQUIRE_FALSE(scan::detail::consumer_test_access::retained_cached_storage(fault));
        auto const retained = scan::detail::scan_operator_test_access::retained_assembly(fault);
        REQUIRE(retained);
        auto const& storage = *retained;
        REQUIRE(storage.output != nullptr);
        REQUIRE(storage.positions != nullptr);
        REQUIRE(storage.output->num_columns() == 2);
        require_resident_values<std::int32_t>(
          storage.positions->view(), cudf::type_id::INT32, {0, 3, 7});
        auto const view = storage.output->view();
        require_resident_values<std::uint64_t>(view.column(1), cudf::type_id::UINT64, {3, 6, 10});
        // Normalization frees each cast source in stream order, so only the casts remain.
        if (point == read_checkpoint::SUBSTITUTION) {
          require_resident_values<std::int32_t>(view.column(0), cudf::type_id::INT32, {41, 29, 53});
        } else {
          require_resident_values<std::int64_t>(view.column(0), cudf::type_id::INT64, {41, 29, 53});
        }
      } else {
        REQUIRE_THROWS_AS(failed_output = f.op.execute(*input, f.e.stream()), rmm::out_of_memory);
        CHECK_FALSE(fault.retained.has_value());
      }
      REQUIRE(failed_output == nullptr);
      REQUIRE(scan::detail::gate_test_access::needs_combined_sample(gate, f.channel->snapshot()));
      REQUIRE_FALSE(scan::detail::gate_test_access::filter_keep_ratio(gate, {filter.get(), 0}, 1));
      auto const counts = f.stats.snapshot();
      CHECK(counts.cached_placements == 0);
      CHECK(counts.cached_payload_bytes == 0);
      CHECK(counts.cached_position_selections == 0);
      CHECK_FALSE(f.op.total_source_output_bytes());
      CHECK_FALSE(input->stolen_table_consumed);
      auto original   = source->to_read_only();
      auto const view = sirius::get_cudf_table_view(original);
      REQUIRE(view.num_columns() == 3);
      require_resident_values<std::int32_t>(
        view.column(0), cudf::type_id::INT32, {41, 17, 83, 29, 61, 11, 97, 53});
      require_resident_values<std::int32_t>(
        view.column(1), cudf::type_id::INT32, {901, 302, 703, 404, 805, 106, 607, 208});
    }
    // The fault hook physically joined before retaining storage; retry after simulated failed
    // completion is test-only.
    input->prepare_for_processing(f.e.gpu_space, f.e.stream());
    auto output         = f.op.execute(*input, f.e.stream());
    auto const& scanned = dynamic_cast<scan::scan_output_operator_data const&>(*output);
    REQUIRE(scanned.receipt().applied.size() == 1);
    auto leases     = scanned.get_read_only_batches();
    auto const view = sirius::get_cudf_table_view(leases.at(0));
    REQUIRE(view.num_columns() == 2);
    require_resident_values<std::int64_t>(view.column(0), cudf::type_id::INT64, {41, 29, 53});
    require_resident_values<std::uint64_t>(view.column(1), cudf::type_id::UINT64, {3, 6, 10});
    auto restored = sirius::late_mat::materialize_at_port(
      port.port_directive(), view, f.e.stream(), f.e.gpu_space->get_default_allocator());
    require_resident_values<std::int64_t>(
      restored->view().column(1), cudf::type_id::INT64, {901, 404, 208});
    auto const counts = f.stats.snapshot();
    CHECK(counts.cached_placements == 1);
    CHECK(counts.cached_position_selections == 1);
  }
  SECTION("first and last survivors restore distinct values after source destruction")
  {
    auto output = f.op.execute(*input, f.e.stream());
    input.reset();
    source.reset();
    auto leases =
      dynamic_cast<sirius::op::pipelineable_operator_data const&>(*output).get_read_only_batches();
    auto view = sirius::get_cudf_table_view(leases.at(0));
    REQUIRE(view.num_columns() == 2);
    require_resident_values<std::int64_t>(view.column(0), cudf::type_id::INT64, {41, 29, 53});
    require_resident_values<std::uint64_t>(view.column(1), cudf::type_id::UINT64, {3, 6, 10});
    REQUIRE(sirius::late_mat::port_directive_matches(port.port_directive(), view));
    auto restored = sirius::late_mat::materialize_at_port(
      port.port_directive(), view, f.e.stream(), f.e.gpu_space->get_default_allocator());
    REQUIRE(restored->num_columns() == 2);
    require_resident_values<std::int64_t>(
      restored->view().column(0), cudf::type_id::INT64, {41, 29, 53});
    require_resident_values<std::int64_t>(
      restored->view().column(1), cudf::type_id::INT64, {901, 404, 208});
    auto counts = f.stats.snapshot();
    CHECK(counts.cached_placements == 1);
    CHECK(counts.cached_position_selections == 1);
  }
}

TEST_CASE(
  "cached Parquet assembly OOM preserves source and defers observations until retry succeeds",
  "[scan_normalization][cached_scan][assembly_fault]")
{
  using scan::detail::read_checkpoint;
  auto checkpoint = read_checkpoint::NORMALIZATION;
  SECTION("normalization") {}
  SECTION("batch construction") { checkpoint = read_checkpoint::BATCH; }
  SECTION("payload construction") { checkpoint = read_checkpoint::PAYLOAD; }
  resident_parquet_fixture f(resident_parquet_info({0}, 1), 1);
  auto filter = f.publish(0, {41, 53});
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(resident_column<std::int32_t>({41, 29, 53}, cudf::type_id::INT32));
  auto source = resident_columns(std::move(columns));
  auto input  = f.split(source);
  {
    scan::detail::read_retirement_test_hooks fault(checkpoint);
    REQUIRE_THROWS_AS(f.op.execute(*input, f.e.stream()), rmm::out_of_memory);
  }
  auto& gate = scan::detail::consumer_test_access::gate(*f.consumer);
  REQUIRE(scan::detail::gate_test_access::needs_combined_sample(gate, f.channel->snapshot()));
  REQUIRE_FALSE(scan::detail::gate_test_access::filter_keep_ratio(gate, {filter.get(), 0}, 1));
  CHECK(f.stats.snapshot().cached_placements == 0);
  {
    auto lease = source->to_read_only();
    require_resident_values<std::int32_t>(
      sirius::get_cudf_table_view(lease).column(0), cudf::type_id::INT32, {41, 29, 53});
  }
  input->prepare_for_processing(f.e.gpu_space, f.e.stream());
  auto output = f.op.execute(*input, f.e.stream());
  auto leases =
    dynamic_cast<sirius::op::pipelineable_operator_data const&>(*output).get_read_only_batches();
  auto view = sirius::get_cudf_table_view(leases.at(0));
  REQUIRE(view.num_columns() == 1);
  require_resident_values<std::int64_t>(view.column(0), cudf::type_id::INT64, {41, 53});
  auto const ratio = scan::detail::gate_test_access::filter_keep_ratio(gate, {filter.get(), 0}, 1);
  REQUIRE(ratio);
  CHECK(*ratio == Catch::Approx(2.0 / 3.0));
  CHECK(f.stats.snapshot().cached_placements == 1);
}

TEST_CASE("cached Parquet restores narrowed DATE and decimal payload survivors with nulls",
          "[scan_normalization][cached_scan][cached_carriers]")
{
  bool decimal = false;
  SECTION("DATE restores epoch days from INT16") {}
  SECTION("decimal restores DECIMAL32 without changing scale") { decimal = true; }
  auto info               = resident_parquet_info({0, 1}, 2);
  info->returned_types[1] = decimal ? sirius::logical_type::make_decimal(18, 2)
                                    : sirius::logical_type::make(sirius::type_id::DATE);
  auto const logical = decimal ? duckdb::LogicalType::DECIMAL(18, 2) : duckdb::LogicalType::DATE;
  resident_parquet_fixture f(
    std::move(info), duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT, logical});
  f.publish(0, {41, 53, 17});
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(resident_column<std::int32_t>({41, 29, 53, 17}, cudf::type_id::INT32));
  if (decimal) {
    std::vector<std::int32_t> const values{12345, -67890, 24680, -999};
    auto payload = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::DECIMAL32, -2},
                                                 static_cast<cudf::size_type>(values.size()),
                                                 cudf::mask_state::ALL_VALID,
                                                 f.e.stream(),
                                                 f.e.gpu_space->get_default_allocator());
    REQUIRE(cudaMemcpyAsync(payload->mutable_view().data<std::int32_t>(),
                            values.data(),
                            values.size() * sizeof(std::int32_t),
                            cudaMemcpyHostToDevice,
                            f.e.stream().get()) == cudaSuccess);
    cudf::set_null_mask(payload->mutable_view().null_mask(), 2, 3, false, f.e.stream());
    payload->set_null_count(1);
    f.e.stream().sync();
    columns.push_back(std::move(payload));
  } else {
    columns.push_back(
      resident_column<std::int16_t>({-120, 730, 23000, -8000}, cudf::type_id::INT16, 2));
  }
  auto source = resident_columns(std::move(columns));
  auto input  = f.split(source);
  REQUIRE(input->needs_carrier_conversion);
  REQUIRE_FALSE(input->stolen_table);
  REQUIRE_FALSE(input->converted_table_steal_pending);
  auto output = f.op.execute(*input, f.e.stream());
  auto leases =
    dynamic_cast<scan::scan_output_operator_data const&>(*output).get_read_only_batches();
  auto const view = sirius::get_cudf_table_view(leases.at(0));
  REQUIRE(view.num_columns() == 2);
  require_resident_values<std::int64_t>(view.column(0), cudf::type_id::INT64, {41, 53, 17});
  auto original            = source->to_read_only();
  auto const original_view = sirius::get_cudf_table_view(original);
  REQUIRE(original_view.num_columns() == 2);
  require_resident_values<std::int32_t>(
    original_view.column(0), cudf::type_id::INT32, {41, 29, 53, 17});
  if (decimal) {
    REQUIRE(view.column(1).type().scale() == -2);
    require_resident_values<std::int64_t>(
      view.column(1), cudf::type_id::DECIMAL64, {12345, 0, -999}, {1, 0, 1});
    REQUIRE(original_view.column(1).type().scale() == -2);
    require_resident_values<std::int32_t>(original_view.column(1),
                                          cudf::type_id::DECIMAL32,
                                          {12345, -67890, 24680, -999},
                                          {1, 1, 0, 1});
  } else {
    require_resident_values<std::int32_t>(
      view.column(1), cudf::type_id::TIMESTAMP_DAYS, {-120, 0, -8000}, {1, 0, 1});
    require_resident_values<std::int16_t>(
      original_view.column(1), cudf::type_id::INT16, {-120, 730, 23000, -8000}, {1, 1, 0, 1});
  }
  auto const counts = f.stats.snapshot();
  CHECK(counts.cached_placements == 1);
  CHECK(counts.cached_payload_bytes > 0);
  CHECK(counts.cached_bypassed_transfer == 0);
}

TEST_CASE("scan forwards its pipeline expression policy to residual filtering",
          "[scan_normalization][gpu_scan][query_options]")
{
  auto& e         = env();
  auto batch      = make_resident_batch(e, kRows);
  auto ingestible = std::make_shared<stub_ingestible>([&e] {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(make_column(*e.gpu_space, cudf::data_type{cudf::type_id::INT64}, kRows));
    return std::make_unique<cudf::table>(std::move(columns));
  });
  for (auto strategy : {sirius::expression_evaluator_strategy::MATERIALIZE,
                        sirius::expression_evaluator_strategy::AST_JIT}) {
    auto scan = make_bigint_scan(ingestible);
    sirius::operator_params options;
    options.expression_strategy = strategy;
    options.enable_regex_jit    = strategy != sirius::expression_evaluator_strategy::MATERIALIZE;
    sirius::pipeline::pipeline_build_context ctx{
      nullptr, true, 1, std::make_shared<const sirius::operator_params>(options)};
    scan.set_pipeline(std::make_shared<sirius::pipeline::sirius_pipeline>(ctx));
    sirius::op::scan::scan_operator_input input(batch);
    input.gpu_memory_space = e.gpu_space;
    admit_fixture_resident_input(input);
    ingestible->observed_policy.reset();
    auto result = scan.execute(input, e.stream());
    REQUIRE(ingestible->observed_policy.has_value());
    CHECK(ingestible->observed_policy->strategy == strategy);
    CHECK(ingestible->observed_policy->regex_jit == options.enable_regex_jit);
    e.stream().sync();
  }
}

TEST_CASE("resident and stolen Parquet facts preserve actual legacy decisions",
          "[scan_normalization][execution_record][cached_scan]")
{
  auto& e           = env();
  bool const stolen = GENERATE(false, true);
  int const mode    = GENERATE(0, 1, 2, 3, 4, 5, 6, 7);
  CAPTURE(stolen, mode);
  auto info = resident_parquet_info({0, 1}, 1, mode ? std::optional<std::size_t>{1} : std::nullopt);
  bool const answered = mode == 3 || mode == 4 || mode == 5;
  if (answered) {
    // Dictionary equality answers are offered only for pure-filter string columns.
    info->returned_types[1] = sirius::logical_type::make(sirius::type_id::VARCHAR);
    info->table_filters     = duckdb::make_uniq<duckdb::TableFilterSet>();
    info->table_filters->PushFilter(
      duckdb::ProjectionIndex(1),
      sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_EQUAL, duckdb::Value("keep")));
  }
  auto counters            = std::make_shared<scan::physical_check_counters>();
  counters->track_units    = true;
  info->profiles->counters = counters;
  std::vector<bool> decisions;
  counters->scan_execution_for_testing = [&](bool authority, auto const&) {
    decisions.push_back(authority);
  };
  auto ingestible      = scan::make_ingestible(std::move(info));
  bool const complete  = mode == 2 || mode == 7;
  bool const qualified = mode != 5 && mode != 6;
  std::vector<std::int64_t> expected{41, 29, 53};
  if (mode == 1 || mode == 5 || mode == 6) expected = {41, 53};
  if (mode == 3) expected = {41};
  for (bool authority : {false, true}) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(resident_column<std::int64_t>({41, 29, 53}, cudf::type_id::INT64));
    if (mode != 7) {
      if (answered)
        columns.push_back(resident_column<std::int8_t>(
          mode == 4 ? std::vector<std::int8_t>{0, 0, 0} : std::vector<std::int8_t>{1, 0, 1},
          cudf::type_id::BOOL8,
          mode == 3 ? std::optional<cudf::size_type>{2} : std::nullopt));
      else
        columns.push_back(resident_column<std::int64_t>(
          complete ? std::vector<std::int64_t>{0, 0, 0} : std::vector<std::int64_t>{1, 0, 1},
          cudf::type_id::INT64));
    }
    auto table = std::make_unique<cudf::table>(std::move(columns));
    auto source =
      sirius::make_data_batch(std::make_unique<cudf::table>(
                                table->view(), e.stream(), e.gpu_space->get_default_allocator()),
                              *e.gpu_space,
                              e.stream(),
                              {});
    scan::scan_operator_input input(source);
    input.gpu_memory_space                     = e.gpu_space;
    input.execution                            = {4, 45, {}};
    input.pushdown_row_filtered                = complete;
    input.pushdown_filter_only_columns_dropped = mode == 7;
    if (answered)
      input.pushdown_predicate_columns =
        mode == 5 ? std::vector<std::size_t>{1, 1} : std::vector<std::size_t>{1};
    if (mode == 6) input.pushdown_predicate_columns = {99};
    input.pushdown_predicates_enforced = mode == 4;
    if (stolen) input.stolen_table = std::move(table);
    auto batch = ingestible->materialize_table(input, e.stream());
    REQUIRE(batch.execution);
    auto const& facts = *batch.execution;
    CHECK(facts.producer == scan::execution_record_producer::resident);
    CHECK(batch.attempt == 4);
    CHECK(batch.logical_input == 45);
    CHECK(facts.authoritative == qualified);
    CHECK(facts.whole_static_complete == (mode == 0 || complete));
    CHECK_FALSE(facts.projection_complete);
    CHECK(facts.visibility_applied.empty());
    CHECK_FALSE(batch.row_positions);
    CHECK_FALSE(batch.lineage_record);
    if (answered) {
      REQUIRE(facts.predicates.size() == 1);
      CHECK(facts.predicates.front().answer_slot == 1);
      CHECK(facts.predicates.front().stage == (mode == 4 ? sirius::op::predicate_stage::enforced
                                                         : sirius::op::predicate_stage::answer));
    }
    // Compare the real producer boundary and the legacy consumer using independent batches.
    batch.execution->authoritative = authority && qualified;
    auto output                    = ingestible->post_filter_and_project(
      std::move(batch), *e.gpu_space, e.stream(), false, nullptr, nullptr, {});
    REQUIRE(output->num_columns() == 1);
    require_resident_values<std::int64_t>(output->view().column(0), cudf::type_id::INT64, expected);
    CHECK(input.stolen_table_consumed == stolen);
    if (stolen) REQUIRE_THROWS(ingestible->materialize_table(input, e.stream()));
  }
  CHECK(decisions == std::vector<bool>{false, qualified});
}

TEST_CASE("Parquet consumes actual decompression outcome before shared remaining work",
          "[scan_normalization][execution_record][cached_scan]")
{
  auto& e             = env();
  int const mode      = GENERATE(0, 1, 2, 3);
  auto info           = resident_parquet_info({0, 1}, 1, 1);
  bool const answered = mode >= 2;
  if (answered) {
    info->returned_types[1] = sirius::logical_type::make(sirius::type_id::VARCHAR);
    info->table_filters     = duckdb::make_uniq<duckdb::TableFilterSet>();
    info->table_filters->PushFilter(
      duckdb::ProjectionIndex(1),
      sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_EQUAL, duckdb::Value("keep")));
  }
  auto ingestible                   = scan::make_ingestible(std::move(info));
  std::vector<std::int64_t> payload = mode == 1 || mode == 3
                                        ? std::vector<std::int64_t>{41, 53}
                                        : std::vector<std::int64_t>{41, 29, 53};
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(resident_column(payload, cudf::type_id::INT64));
  if (mode == 0) columns.push_back(resident_column<std::int64_t>({1, 0, 1}, cudf::type_id::INT64));
  if (answered)
    columns.push_back(resident_column<std::int8_t>(
      mode == 3 ? std::vector<std::int8_t>{0, 0} : std::vector<std::int8_t>{1, 0, 1},
      cudf::type_id::BOOL8,
      mode == 2 ? std::optional<cudf::size_type>{2} : std::nullopt));
  sirius::pushdown_outcome outcome;
  outcome.row_filtered                = mode == 1;
  outcome.filter_only_columns_dropped = mode == 1;
  outcome.predicates_enforced         = mode == 3;
  if (answered) outcome.predicate_columns = {1};
  auto source = cucascade::data_batch::make(
    sirius::get_next_batch_id(),
    std::make_unique<sirius::decompression_pushdown_batch_representation>(
      std::make_unique<cudf::table>(std::move(columns)), *e.gpu_space, e.stream(), outcome));
  scan::scan_operator_input input(source);
  input.prepare_for_processing(e.gpu_space, e.stream());
  CHECK(input.pushdown_row_filtered == outcome.row_filtered);
  CHECK(input.pushdown_predicate_columns == outcome.predicate_columns);
  CHECK(input.pushdown_predicates_enforced == outcome.predicates_enforced);
  auto decoded = ingestible->materialize_table(input, e.stream());
  REQUIRE(decoded.execution);
  CHECK(decoded.execution->authoritative);
  CHECK(decoded.execution->producer == scan::execution_record_producer::resident);
  CHECK(decoded.execution->whole_static_complete == (mode == 1));
  CHECK(decoded.execution->visibility_applied.empty());
  if (answered) {
    REQUIRE(decoded.execution->predicates.size() == 1);
    CHECK(
      decoded.execution->predicates.front().stage ==
      (mode == 3 ? sirius::op::predicate_stage::enforced : sirius::op::predicate_stage::answer));
  }
  auto output = ingestible->post_filter_and_project(
    std::move(decoded), *e.gpu_space, e.stream(), false, nullptr, nullptr, {});
  require_resident_values<std::int64_t>(
    output->view().column(0),
    cudf::type_id::INT64,
    mode == 2 ? std::vector<std::int64_t>{41} : std::vector<std::int64_t>{41, 53});
}
