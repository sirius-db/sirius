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

// Unit tests for the equality-delete machinery: the anti-join mask kernel, the group build, the
// filter that probes it, and the pipeline's extra-column strip.
//
// These exist because NO SQL test can reach this code. read_iceberg_delete_data() refuses live
// equality-delete entries and the plan gate declines those tables, so every fixture asserts CPU
// fallback. An inverted keep/discard mask, a wrong strict-inequality in applies_to, or a strip
// that cuts the wrong end would all stay green until the route is switched on -- which is the
// one moment the fixtures were built to protect. Driving the components directly is the only way
// to hold them to their contract in the meantime.

#include "op/scan/iceberg_delete_filter.hpp"
#include "op/scan/iceberg_equality_delete_mask.hpp"
#include "op/scan/iceberg_metadata_reader.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/join/join.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <catch.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace sirius::op::scan;

namespace {

rmm::cuda_stream_view test_stream() { return cudf::get_default_stream(); }

rmm::device_async_resource_ref test_mr() { return cudf::get_current_device_resource_ref(); }

/// An INT32 column from host values; `nulls` marks which entries are null.
std::unique_ptr<cudf::column> int32_column(std::vector<int32_t> const& values,
                                           std::vector<bool> const& nulls = {})
{
  auto stream = test_stream();
  auto col    = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32},
                                           static_cast<cudf::size_type>(values.size()),
                                           cudf::mask_state::UNALLOCATED,
                                           stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(col->mutable_view().data<int32_t>(),
                                values.data(),
                                values.size() * sizeof(int32_t),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  stream.synchronize();
  if (!nulls.empty()) {
    REQUIRE(nulls.size() == values.size());
    // Built on the host and copied whole: sized with bitmask_allocation_size_bytes because a
    // bit-tight buffer crashes cudf's deep copies.
    auto const bytes =
      cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(values.size()));
    std::vector<cudf::bitmask_type> host_mask(bytes / sizeof(cudf::bitmask_type), ~0u);
    cudf::size_type null_count = 0;
    for (std::size_t i = 0; i < nulls.size(); ++i) {
      if (!nulls[i]) { continue; }
      host_mask[i / 32] &= ~(cudf::bitmask_type{1} << (i % 32));
      ++null_count;
    }
    if (null_count > 0) {
      rmm::device_buffer mask{host_mask.data(), bytes, stream};
      stream.synchronize();
      col->set_null_mask(std::move(mask), null_count);
    }
  }
  return col;
}

std::vector<int32_t> to_host(cudf::column_view const& col)
{
  std::vector<int32_t> out(static_cast<std::size_t>(col.size()));
  auto stream = test_stream();
  CUDF_CUDA_TRY(cudaMemcpyAsync(out.data(),
                                col.data<int32_t>(),
                                out.size() * sizeof(int32_t),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();
  return out;
}

std::vector<uint8_t> mask_to_host(cudf::column_view const& col)
{
  std::vector<uint8_t> out(static_cast<std::size_t>(col.size()));
  auto stream = test_stream();
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    out.data(), col.data<uint8_t>(), out.size(), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  return out;
}

rmm::device_uvector<cudf::size_type> device_indices(std::vector<cudf::size_type> const& host)
{
  auto stream = test_stream();
  rmm::device_uvector<cudf::size_type> dev(host.size(), stream, test_mr());
  CUDF_CUDA_TRY(cudaMemcpyAsync(dev.data(),
                                host.data(),
                                host.size() * sizeof(cudf::size_type),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  stream.synchronize();
  return dev;
}

/// A filter that records how many times it ran and hands the batch straight back.
class counting_filter : public iceberg_delete_filter {
 public:
  std::unique_ptr<cudf::table> apply(std::unique_ptr<cudf::table> tbl,
                                     batch_layout,
                                     rmm::cuda_stream_view,
                                     rmm::device_async_resource_ref) override
  {
    ++calls;
    return tbl;
  }
  int calls = 0;
};

}  // namespace

//===----------------------------------------------------------------------===//
// make_anti_join_mask
//===----------------------------------------------------------------------===//

TEST_CASE("iceberg equality delete - anti-join mask keeps unmatched rows", "[iceberg][scan]")
{
  // left_join() writes JoinNoMatch for a probe row that found no delete row. That row SURVIVES.
  // Any other value means the row matched a delete and must go. Inverting this deletes exactly
  // the rows the table keeps and keeps the ones it deletes, with no error anywhere.
  auto const indices = device_indices({cudf::JoinNoMatch, 0, cudf::JoinNoMatch, 7});

  auto mask = make_anti_join_mask(indices, 4, test_stream(), test_mr());
  REQUIRE(mask->size() == 4);
  REQUIRE(mask->type().id() == cudf::type_id::BOOL8);
  REQUIRE(mask_to_host(mask->view()) == std::vector<uint8_t>{1, 0, 1, 0});
}

TEST_CASE("iceberg equality delete - anti-join mask honours n_rows", "[iceberg][scan]")
{
  // The probe result can be longer than the batch; only the first n_rows entries are the batch's.
  auto const indices = device_indices({cudf::JoinNoMatch, 3, cudf::JoinNoMatch});

  auto mask = make_anti_join_mask(indices, 2, test_stream(), test_mr());
  REQUIRE(mask->size() == 2);
  REQUIRE(mask_to_host(mask->view()) == std::vector<uint8_t>{1, 0});
}

//===----------------------------------------------------------------------===//
// build_equality_group + equality_delete_filter
//===----------------------------------------------------------------------===//

TEST_CASE("iceberg equality delete - group deduplicates across delete files", "[iceberg][scan]")
{
  std::vector<std::unique_ptr<cudf::column>> a;
  a.push_back(int32_column({1, 2}));
  cudf::table file_a{std::move(a)};

  std::vector<std::unique_ptr<cudf::column>> b;
  b.push_back(int32_column({2, 3}));
  cudf::table file_b{std::move(b)};

  auto group =
    build_equality_group({"id"}, {std::optional<int32_t>{1}}, {file_a.view(), file_b.view()});

  // {1,2} ∪ {2,3} = three distinct keys, not four rows.
  REQUIRE(group.delete_table->num_rows() == 3);
  REQUIRE(group.key_names == std::vector<std::string>{"id"});
}

TEST_CASE("iceberg equality delete - filter removes exactly the matching rows", "[iceberg][scan]")
{
  std::vector<std::unique_ptr<cudf::column>> del;
  del.push_back(int32_column({2, 4}));
  cudf::table delete_file{std::move(del)};

  auto data = std::make_shared<IcebergDeleteData>();
  data->equality_delete_groups.push_back(
    build_equality_group({"id"}, {std::optional<int32_t>{1}}, {delete_file.view()}));
  // sequence_number 0 means "applies to every data file", which keeps this case about the mask.
  data->equality_delete_groups.back().sequence_number = 0;

  std::vector<std::unique_ptr<cudf::column>> batch_cols;
  batch_cols.push_back(int32_column({1, 2, 3, 4, 5}));
  auto batch = std::make_unique<cudf::table>(std::move(batch_cols));

  equality_delete_filter filter{data, 0, {0}};
  std::vector<batch_row_run> runs{{"data-0.parquet", 0, 0, 5}};

  auto out = filter.apply(std::move(batch), batch_layout{runs}, test_stream(), test_mr());
  REQUIRE(out->num_rows() == 3);
  REQUIRE(to_host(out->get_column(0).view()) == std::vector<int32_t>{1, 3, 5});
}

TEST_CASE("iceberg equality delete - a NULL key deletes the NULL rows", "[iceberg][scan]")
{
  // Iceberg equality deletes match with null-equal semantics: a delete row whose key is NULL
  // removes data rows whose key is NULL. cudf's default is null_equality::UNEQUAL, so this is
  // the one place the group build's explicit choice shows up.
  std::vector<std::unique_ptr<cudf::column>> del;
  del.push_back(int32_column({0}, {true}));  // single NULL key
  cudf::table delete_file{std::move(del)};

  auto data = std::make_shared<IcebergDeleteData>();
  data->equality_delete_groups.push_back(
    build_equality_group({"id"}, {std::optional<int32_t>{1}}, {delete_file.view()}));
  data->equality_delete_groups.back().sequence_number = 0;

  std::vector<std::unique_ptr<cudf::column>> batch_cols;
  batch_cols.push_back(int32_column({1, 0, 3}, {false, true, false}));
  auto batch = std::make_unique<cudf::table>(std::move(batch_cols));

  equality_delete_filter filter{data, 0, {0}};
  std::vector<batch_row_run> runs{{"data-0.parquet", 0, 0, 3}};

  auto out = filter.apply(std::move(batch), batch_layout{runs}, test_stream(), test_mr());
  REQUIRE(out->num_rows() == 2);
  REQUIRE(to_host(out->get_column(0).view()) == std::vector<int32_t>{1, 3});
}

TEST_CASE("iceberg equality delete - missing key column throws", "[iceberg][scan]")
{
  // The key columns are force-projected into the scan. If that widening never happened, the
  // batch is short and returning it unchanged would hand back rows the table deleted.
  std::vector<std::unique_ptr<cudf::column>> del;
  del.push_back(int32_column({1}));
  cudf::table delete_file{std::move(del)};

  auto data = std::make_shared<IcebergDeleteData>();
  data->equality_delete_groups.push_back(
    build_equality_group({"id"}, {std::optional<int32_t>{1}}, {delete_file.view()}));
  data->equality_delete_groups.back().sequence_number = 0;

  std::vector<std::unique_ptr<cudf::column>> batch_cols;
  batch_cols.push_back(int32_column({1, 2}));
  auto batch = std::make_unique<cudf::table>(std::move(batch_cols));

  equality_delete_filter filter{data, 0, {3}};  // index past the end of the batch
  std::vector<batch_row_run> runs{{"data-0.parquet", 0, 0, 2}};

  REQUIRE_THROWS_AS(filter.apply(std::move(batch), batch_layout{runs}, test_stream(), test_mr()),
                    std::invalid_argument);
}

//===----------------------------------------------------------------------===//
// iceberg_delete_pipeline extra-column strip
//===----------------------------------------------------------------------===//

TEST_CASE("iceberg delete pipeline - strips appended key columns from the tail", "[iceberg][scan]")
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(int32_column({1, 2}));    // projected
  cols.push_back(int32_column({10, 20}));  // projected
  cols.push_back(int32_column({99, 99}));  // appended key
  auto batch = std::make_unique<cudf::table>(std::move(cols));

  auto counter = std::make_shared<counting_filter>();
  iceberg_delete_pipeline pipeline;
  pipeline.add_filter(counter);
  pipeline.set_extra_column_count(1);

  std::vector<batch_row_run> runs{{"data-0.parquet", 0, 0, 2}};
  auto out = pipeline.apply(std::move(batch), batch_layout{runs}, test_stream(), test_mr());

  REQUIRE(counter->calls == 1);
  REQUIRE(out->num_columns() == 2);
  // The tail is what gets cut: the surviving columns must be the first two, in order.
  REQUIRE(to_host(out->get_column(0).view()) == std::vector<int32_t>{1, 2});
  REQUIRE(to_host(out->get_column(1).view()) == std::vector<int32_t>{10, 20});
}

TEST_CASE("iceberg delete pipeline - refuses a batch that is all appended keys", "[iceberg][scan]")
{
  // Nothing left after the strip means the pipeline's extra-column count and the projection the
  // scan produced disagree. Skipping the strip would return the key columns as query output.
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(int32_column({1, 2}));
  auto batch = std::make_unique<cudf::table>(std::move(cols));

  iceberg_delete_pipeline pipeline;
  pipeline.add_filter(std::make_shared<counting_filter>());
  pipeline.set_extra_column_count(1);

  std::vector<batch_row_run> runs{{"data-0.parquet", 0, 0, 2}};
  REQUIRE_THROWS_AS(pipeline.apply(std::move(batch), batch_layout{runs}, test_stream(), test_mr()),
                    std::invalid_argument);
}
