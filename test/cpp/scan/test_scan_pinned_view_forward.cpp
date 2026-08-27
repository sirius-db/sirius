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

// Verifies that unfiltered GPU-pinned scans forward matching columns without copying, cast only
// mismatched carriers, and keep the pinned storage alive for the output batch's lifetime.

#include "operator/operator_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/data_batch_utils.hpp>
#include <data/reclaim_ledger.hpp>
#include <helper/type_conversions.hpp>
#include <io/io_context.hpp>
#include <op/scan/gpu_ingestible.hpp>
#include <op/scan/gpu_ingestible_types.hpp>
#include <op/scan/owning_table_view.hpp>
#include <op/scan/sirius_gpu_scan_operator.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <op/sirius_physical_operator.hpp>

#include <array>
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

using sirius::test::operator_utils::copy_column_to_host;

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

constexpr cudf::data_type kInt64{cudf::type_id::INT64};
constexpr cudf::data_type kInt32{cudf::type_id::INT32};

// Owned device column of `type` filled with `values`.
template <typename T>
std::unique_ptr<cudf::column> make_owned_column(cucascade::memory::memory_space& space,
                                                cudf::data_type type,
                                                std::vector<T> const& values)
{
  auto mr     = sirius::test::operator_utils::get_resource_ref(space);
  auto stream = sirius::test::operator_utils::default_stream();
  auto col    = cudf::make_numeric_column(
    type, static_cast<cudf::size_type>(values.size()), cudf::mask_state::UNALLOCATED, stream, mr);
  cudaMemcpy(col->mutable_view().head<T>(),
             values.data(),
             sizeof(T) * values.size(),
             cudaMemcpyHostToDevice);
  return col;
}

// Shared device column, matching the storage form of a pinned cache entry.
template <typename T>
std::shared_ptr<cudf::column> make_pinned_column(cucascade::memory::memory_space& space,
                                                 cudf::data_type type,
                                                 std::vector<T> const& values)
{
  return std::shared_ptr<cudf::column>(make_owned_column(space, type, values));
}

// Wrapper batch in the shape used for a raw GPU pin: a view over shared column storage.
std::shared_ptr<cucascade::data_batch> make_pin_shaped_batch(
  test_env& e, std::vector<std::shared_ptr<cudf::column>> columns)
{
  std::vector<cudf::column_view> views;
  views.reserve(columns.size());
  std::size_t alloc_size = 0;
  for (auto const& col : columns) {
    views.push_back(col->view());
    alloc_size += col->alloc_size();
  }
  auto repr = std::make_unique<cucascade::gpu_table_representation>(
    cudf::table_view(views), std::move(columns), alloc_size, *e.gpu_space, rmm::cuda_stream_view{});
  return cucascade::data_batch::make(sirius::get_next_batch_id(), std::move(repr));
}

class stub_table_info final : public sirius::op::scan::ingestible_table_info {
 public:
  [[nodiscard]] std::span<std::string const> column_names() const override { return {}; }
  [[nodiscard]] std::span<std::string const> file_paths() const override { return {}; }
};

// Resident-only ingestible that forwards the materialized handle or runs a caller-supplied hook.
// Metadata entry points are unreachable because resident splits read the cached batch.
class stub_ingestible final : public sirius::op::scan::gpu_ingestible {
 public:
  using post_filter_hook =
    std::function<sirius::op::scan::owning_table_view(sirius::op::scan::filtered_table&&)>;

  stub_ingestible() = default;
  explicit stub_ingestible(post_filter_hook hook) : _hook(std::move(hook)) {}

  sirius::op::scan::owning_table_view post_filter_and_project(
    sirius::op::scan::filtered_table&& input,
    const cucascade::memory::memory_space&,
    rmm::cuda_stream_view,
    bool,
    std::shared_ptr<const sirius::like_multiliteral_cache>,
    std::unique_ptr<cudf::column>*,
    std::span<std::size_t const> /*elided*/) override
  {
    if (_hook) { return _hook(std::move(input)); }
    return std::move(input.table);
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
    rmm::cuda_stream_view,
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

 private:
  post_filter_hook _hook;
  stub_table_info _info;
};

// Scan with `n_outputs` BIGINT columns and no sidecar; non-INT64 resident carriers must cast.
sirius::op::scan::sirius_gpu_scan_operator make_bigint_scan(
  std::shared_ptr<sirius::op::scan::gpu_ingestible> ingestible, std::size_t n_outputs = 1)
{
  duckdb::vector<duckdb::LogicalType> types;
  for (std::size_t i = 0; i < n_outputs; ++i) {
    types.push_back(duckdb::LogicalType::BIGINT);
  }
  return sirius::op::scan::sirius_gpu_scan_operator{
    sirius::from_duckdb_vec(types), /*estimated_cardinality=*/0, std::move(ingestible)};
}

// Drive execute() over one resident split and return its output batch.
std::shared_ptr<cucascade::data_batch> run_scan(test_env& e,
                                                std::shared_ptr<stub_ingestible> ingestible,
                                                std::shared_ptr<cucascade::data_batch> batch,
                                                std::size_t n_outputs = 1)
{
  auto scan = make_bigint_scan(std::move(ingestible), n_outputs);
  sirius::op::scan::scan_operator_input input(std::move(batch));
  REQUIRE(input.is_resident());
  input.prepare_for_processing(e.gpu_space, e.stream());

  auto output        = scan.execute(input, e.stream());
  auto* pipelineable = dynamic_cast<const sirius::op::pipelineable_operator_data*>(output.get());
  REQUIRE(pipelineable != nullptr);
  auto const& batches = pipelineable->get_data_batches();
  REQUIRE(batches.size() == 1);
  return batches[0];
}

// Drop all input-side owners after the scan, then verify that the output batch alone keeps the
// pinned columns alive and releases them when destroyed.
void check_pinned_columns_survive_unpin(test_env& e,
                                        std::vector<std::shared_ptr<cudf::column>> columns,
                                        std::vector<std::vector<std::int64_t>> const& expected)
{
  std::shared_ptr<cucascade::data_batch> out;
  {
    auto wrapper = make_pin_shaped_batch(e, columns);
    out          = run_scan(e, std::make_shared<stub_ingestible>(), wrapper, columns.size());
  }  // split, wrapper handle, and operator data are gone: unpin
  e.stream().synchronize();

  {
    auto ro   = out->to_read_only();
    auto view = sirius::get_cudf_table_view(ro);
    for (std::size_t i = 0; i < expected.size(); ++i) {
      REQUIRE(copy_column_to_host<std::int64_t>(view.column(static_cast<cudf::size_type>(i))) ==
              expected[i]);
    }
  }

  // The output batch's owner still holds the wrapper batch, hence every pinned column.
  for (auto const& column : columns) {
    REQUIRE(column.use_count() > 1);
  }
  out.reset();
  for (auto const& column : columns) {
    REQUIRE(column.use_count() == 1);
  }
}

}  // namespace

TEST_CASE("a pinned unfiltered split forwards as a zero-copy view",
          "[pinned_view_forward][gpu_scan]")
{
  auto& e = env();
  std::vector<std::int64_t> const values{1, 2, 3, 4, 5, 6, 7, 8};
  auto pinned             = make_pinned_column<std::int64_t>(*e.gpu_space, kInt64, values);
  auto const* pinned_head = pinned->view().head<std::int64_t>();

  auto out = run_scan(e, std::make_shared<stub_ingestible>(), make_pin_shaped_batch(e, {pinned}));

  auto ro         = out->to_read_only();
  auto const& rep = ro.get_data()->cast<cucascade::gpu_table_representation>();
  auto view       = rep.get_table_view();
  REQUIRE(view.num_columns() == 1);
  REQUIRE(view.column(0).head<std::int64_t>() == pinned_head);  // no D2D copy
  REQUIRE(rep.get_size_in_bytes() > 0);
  REQUIRE(sirius::reclaimable_size_in_bytes(ro) == 0);  // pinned storage outlives the batch
  REQUIRE(rep.get_writer_event() != nullptr);

  e.stream().synchronize();
  REQUIRE(copy_column_to_host<std::int64_t>(view.column(0)) == values);
}

TEST_CASE("a projected subset of a pinned split stays a view", "[pinned_view_forward][gpu_scan]")
{
  auto& e = env();
  std::vector<std::int64_t> const kept{11, 12, 13, 14};
  std::vector<std::int64_t> const dropped{91, 92, 93, 94};
  auto kept_col           = make_pinned_column<std::int64_t>(*e.gpu_space, kInt64, kept);
  auto dropped_col        = make_pinned_column<std::int64_t>(*e.gpu_space, kInt64, dropped);
  auto const* pinned_head = kept_col->view().head<std::int64_t>();

  // Index-only projection of the pinned view, the shape post_filter_and_project produces for a
  // pinned column-superset scan.
  auto ingestible = std::make_shared<stub_ingestible>([](sirius::op::scan::filtered_table&& in) {
    std::array<std::size_t, 1> keep{0};
    in.table.select_columns(keep);
    return std::move(in.table);
  });
  auto out        = run_scan(e, ingestible, make_pin_shaped_batch(e, {kept_col, dropped_col}));

  auto ro   = out->to_read_only();
  auto view = sirius::get_cudf_table_view(ro);
  REQUIRE(view.num_columns() == 1);
  REQUIRE(view.column(0).head<std::int64_t>() == pinned_head);

  e.stream().synchronize();
  REQUIRE(copy_column_to_host<std::int64_t>(view.column(0)) == kept);
}

TEST_CASE("an owned post-filter result keeps the materializing path",
          "[pinned_view_forward][gpu_scan]")
{
  auto& e = env();
  std::vector<std::int64_t> const pinned_values{1, 2, 3, 4};
  std::vector<std::int64_t> const fresh_values{5, 6, 7, 8};
  auto pinned             = make_pinned_column<std::int64_t>(*e.gpu_space, kInt64, pinned_values);
  auto const* pinned_head = pinned->view().head<std::int64_t>();

  // A row filter gathers into a fresh owned table; release_view must disengage for it.
  auto ingestible =
    std::make_shared<stub_ingestible>([&e, &fresh_values](sirius::op::scan::filtered_table&&) {
      std::vector<std::unique_ptr<cudf::column>> columns;
      columns.push_back(make_owned_column<std::int64_t>(*e.gpu_space, kInt64, fresh_values));
      return sirius::op::scan::owning_table_view{std::make_unique<cudf::table>(std::move(columns))};
    });
  auto out = run_scan(e, ingestible, make_pin_shaped_batch(e, {pinned}));

  auto ro   = out->to_read_only();
  auto view = sirius::get_cudf_table_view(ro);
  REQUIRE(view.num_columns() == 1);
  REQUIRE(view.column(0).head<std::int64_t>() != pinned_head);
  // Owned output declares nothing: it defaults to full reclaimability.
  REQUIRE(sirius::reclaim_ledger::instance().declared_bytes(ro.get_batch_id()) == std::nullopt);

  e.stream().synchronize();
  REQUIRE(copy_column_to_host<std::int64_t>(view.column(0)) == fresh_values);
}

TEST_CASE("a fully narrowed pinned split emits freshly restored columns",
          "[pinned_view_forward][gpu_scan]")
{
  auto& e = env();
  std::vector<std::int32_t> const values{21, 22, 23, 24};
  auto pinned             = make_pinned_column<std::int32_t>(*e.gpu_space, kInt32, values);
  auto const* pinned_head = pinned->view().head<std::int32_t>();

  auto out = run_scan(e, std::make_shared<stub_ingestible>(), make_pin_shaped_batch(e, {pinned}));

  auto ro   = out->to_read_only();
  auto view = sirius::get_cudf_table_view(ro);
  REQUIRE(view.num_columns() == 1);
  REQUIRE(view.column(0).type().id() == cudf::type_id::INT64);
  REQUIRE(static_cast<void const*>(view.column(0).head()) != static_cast<void const*>(pinned_head));
  // Every byte of the output is a freshly cast column, so all of it is reclaimable.
  REQUIRE(sirius::reclaimable_size_in_bytes(ro) == ro.get_data()->get_size_in_bytes());

  e.stream().synchronize();
  REQUIRE(copy_column_to_host<std::int64_t>(view.column(0)) ==
          std::vector<std::int64_t>{21, 22, 23, 24});
}

TEST_CASE("a mixed narrowed-carrier split casts only the mismatched column",
          "[pinned_view_forward][gpu_scan]")
{
  auto& e = env();
  std::vector<std::int32_t> const narrow_values{41, 42, 43, 44};
  std::vector<std::int64_t> const native_values{51, 52, 53, 54};
  auto narrow_col         = make_pinned_column<std::int32_t>(*e.gpu_space, kInt32, narrow_values);
  auto native_col         = make_pinned_column<std::int64_t>(*e.gpu_space, kInt64, native_values);
  auto const* narrow_head = narrow_col->view().head<std::int32_t>();
  auto const* native_head = native_col->view().head<std::int64_t>();

  auto out = run_scan(e,
                      std::make_shared<stub_ingestible>(),
                      make_pin_shaped_batch(e, {narrow_col, native_col}),
                      /*n_outputs=*/2);

  auto ro         = out->to_read_only();
  auto const& rep = ro.get_data()->cast<cucascade::gpu_table_representation>();
  auto view       = rep.get_table_view();
  REQUIRE(view.num_columns() == 2);
  // The narrowed column restored into a fresh buffer; the native column is view-forwarded.
  REQUIRE(view.column(0).type().id() == cudf::type_id::INT64);
  REQUIRE(static_cast<void const*>(view.column(0).head()) != static_cast<void const*>(narrow_head));
  REQUIRE(view.column(1).head<std::int64_t>() == native_head);
  REQUIRE(rep.get_size_in_bytes() > 0);
  // Only the freshly cast column is reclaimable; the forwarded column's storage stays pinned.
  REQUIRE(sirius::reclaimable_size_in_bytes(ro) > 0);
  REQUIRE(sirius::reclaimable_size_in_bytes(ro) < rep.get_size_in_bytes());
  REQUIRE(rep.get_writer_event() != nullptr);

  e.stream().synchronize();
  REQUIRE(copy_column_to_host<std::int64_t>(view.column(0)) ==
          std::vector<std::int64_t>{41, 42, 43, 44});
  REQUIRE(copy_column_to_host<std::int64_t>(view.column(1)) == native_values);
}

TEST_CASE("the forwarded owner keeps pinned columns alive past unpin",
          "[pinned_view_forward][gpu_scan]")
{
  auto& e = env();
  std::vector<std::shared_ptr<cudf::column>> columns;
  columns.push_back(make_pinned_column<std::int64_t>(*e.gpu_space, kInt64, {31, 32, 33, 34}));
  check_pinned_columns_survive_unpin(e, std::move(columns), {{31, 32, 33, 34}});
}

TEST_CASE("a mixed emission's composite owner keeps pinned columns alive past unpin",
          "[pinned_view_forward][gpu_scan]")
{
  auto& e = env();
  std::vector<std::shared_ptr<cudf::column>> columns;
  columns.push_back(make_pinned_column<std::int32_t>(*e.gpu_space, kInt32, {61, 62, 63, 64}));
  columns.push_back(make_pinned_column<std::int64_t>(*e.gpu_space, kInt64, {71, 72, 73, 74}));
  check_pinned_columns_survive_unpin(e, std::move(columns), {{61, 62, 63, 64}, {71, 72, 73, 74}});
}

TEST_CASE("an all-cast emission still carries the surrendered owner past unpin",
          "[pinned_view_forward][gpu_scan]")
{
  auto& e = env();
  std::vector<std::shared_ptr<cudf::column>> columns;
  columns.push_back(make_pinned_column<std::int32_t>(*e.gpu_space, kInt32, {81, 82, 83, 84}));
  check_pinned_columns_survive_unpin(e, std::move(columns), {{81, 82, 83, 84}});
}
