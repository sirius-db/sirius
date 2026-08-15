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

#include <catch.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/data_batch_utils.hpp>
#include <helper/type_conversions.hpp>
#include <io/io_context.hpp>
#include <op/scan/gpu_ingestible.hpp>
#include <op/scan/gpu_ingestible_types.hpp>
#include <op/scan/owning_table_view.hpp>
#include <op/scan/sirius_gpu_scan_operator.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <op/sirius_physical_operator.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct test_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu_space;
  rmm::cuda_stream conv_stream;

  test_env()
    : mgr(sirius::test::operator_utils::initialize_memory_manager()),
      gpu_space(mgr->get_memory_space(cucascade::memory::Tier::GPU, 0)),
      conv_stream()
  {
  }

  rmm::cuda_stream_view stream() { return conv_stream.view(); }
};

test_env& env()
{
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

/// Cached chunk standing in for a pinned split: its own contents never reach the assertions,
/// since only is_resident() routes execute() down the cached branch.
std::shared_ptr<cucascade::data_batch> make_resident_batch(test_env& e, std::size_t rows)
{
  std::shared_ptr<cudf::column> col =
    make_column(*e.gpu_space, cudf::data_type{cudf::type_id::INT64}, rows);
  std::vector<std::shared_ptr<cudf::column>> columns{col};
  std::vector<cudf::column_view> views{col->view()};
  auto const alloc_size = col->alloc_size();
  auto repr             = std::make_unique<cucascade::gpu_table_representation>(
    cudf::table_view(views), std::move(columns), alloc_size, *e.gpu_space, rmm::cuda_stream_view{});
  return cucascade::data_batch::make(sirius::get_next_batch_id(), std::move(repr));
}

class stub_table_info final : public sirius::op::scan::ingestible_table_info {
 public:
  [[nodiscard]] std::span<std::string const> column_names() const override { return {}; }
  [[nodiscard]] std::span<std::string const> file_paths() const override { return {}; }
};

/// Hands execute() a caller-chosen table as the post-filter result, which is the seam where a
/// materialized shape that disagrees with the scan's output types can be injected. A resident
/// split reads its rows from the cached batch, so every metadata entry point stays unreachable.
class stub_ingestible final : public sirius::op::scan::gpu_ingestible {
 public:
  using table_factory = std::function<std::unique_ptr<cudf::table>()>;

  explicit stub_ingestible(table_factory produce) : _produce(std::move(produce)) {}

  sirius::op::scan::owning_table_view post_filter_and_project(
    sirius::op::scan::filtered_table&&,
    const cucascade::memory::memory_space&,
    rmm::cuda_stream_view) override
  {
    return sirius::op::scan::owning_table_view{_produce()};
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
    rmm::cuda_stream_view) override
  {
    throw std::logic_error("stub_ingestible: a resident split never decodes scan metadata");
  }

  [[nodiscard]] const sirius::op::scan::ingestible_table_info& table_info() const noexcept override
  {
    return _info;
  }

  [[nodiscard]] std::vector<std::size_t> materialized_column_order() const override { return {}; }

 private:
  table_factory _produce;
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
    std::move(ingestible)};
}

constexpr std::size_t kRows = 8;

}  // namespace

TEST_CASE("scan construction rejects an incomplete native carrier schema",
          "[scan_normalization][gpu_scan]")
{
  duckdb::vector<sirius::logical_type> types;
  types.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
  types.push_back(sirius::logical_type::make_decimal(4, 2));

  REQUIRE_THROWS_WITH(sirius::op::scan::sirius_gpu_scan_operator(std::move(types), 0, nullptr),
                      Catch::Contains("output column 1 (DECIMAL(4,2)) has no native cuDF carrier"));
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
  REQUIRE(input.is_resident());

  REQUIRE_THROWS_WITH(scan.execute(input, e.stream()),
                      Catch::Contains("output schema width mismatch"));
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
  REQUIRE(input.is_resident());

  REQUIRE_THROWS_WITH(scan.execute(input, e.stream()),
                      Catch::Contains("native schema carrier mismatch"));
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
