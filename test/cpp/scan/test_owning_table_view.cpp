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

#include "op/scan/owning_table_view.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace sirius::op::scan;

namespace {

rmm::cuda_stream_view test_stream() { return cudf::get_default_stream(); }
rmm::device_async_resource_ref test_mr() { return cudf::get_current_device_resource_ref(); }

// Build an INT32 column whose single value tags the column for identification.
std::unique_ptr<cudf::column> tagged_column(std::int32_t tag)
{
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 1, cudf::mask_state::UNALLOCATED, test_stream());
  cudaMemcpy(col->mutable_view().data<std::int32_t>(), &tag, sizeof(tag), cudaMemcpyHostToDevice);
  return col;
}

// Build a table with one INT32 column per tag.
std::unique_ptr<cudf::table> tagged_table(std::vector<std::int32_t> const& tags)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(tags.size());
  for (auto t : tags) {
    cols.push_back(tagged_column(t));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

// Read back the single tag value of each column of a view.
std::vector<std::int32_t> read_tags(cudf::table_view view)
{
  std::vector<std::int32_t> out(static_cast<std::size_t>(view.num_columns()));
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    cudaMemcpy(&out[static_cast<std::size_t>(i)],
               view.column(i).data<std::int32_t>(),
               sizeof(std::int32_t),
               cudaMemcpyDeviceToHost);
  }
  return out;
}

// The device data pointers of each column of a view (column identity).
std::vector<void const*> data_ptrs(cudf::table_view view)
{
  std::vector<void const*> out(static_cast<std::size_t>(view.num_columns()));
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    out[static_cast<std::size_t>(i)] = view.column(i).head();
  }
  return out;
}

// A keep-alive owner that does NOT satisfy no_alloc_materializable (no
// dereferenceable release()), forcing the copying materialization path.
struct table_keepalive {
  std::unique_ptr<cudf::table> table;
};

}  // namespace

TEST_CASE("owning_table_view empty state", "[owning_table_view]")
{
  owning_table_view handle;
  REQUIRE_FALSE(static_cast<bool>(handle));
  REQUIRE(handle.n_columns() == 0);
  REQUIRE(handle.view().num_columns() == 0);

  // Mutating an empty handle is a no-op and leaves it empty.
  std::array<std::size_t, 1> drop{0};
  handle.drop_columns(drop);
  REQUIRE_FALSE(static_cast<bool>(handle));

  // release() on an empty handle yields nullptr.
  REQUIRE(handle.release(test_stream(), test_mr()) == nullptr);
}

TEST_CASE("owning_table_view from owned table exposes the table", "[owning_table_view]")
{
  owning_table_view handle{tagged_table({10, 20, 30})};
  REQUIRE(static_cast<bool>(handle));
  REQUIRE(handle.n_columns() == 3);
  REQUIRE(read_tags(handle.view()) == std::vector<std::int32_t>{10, 20, 30});
}

TEST_CASE("owning_table_view num_rows / column / column_types accessors", "[owning_table_view]")
{
  // 4 columns, 1 row each.
  owning_table_view handle{tagged_table({10, 20, 30, 40})};

  REQUIRE(handle.num_rows() == 1);
  REQUIRE(handle.column_types().size() == 4);
  for (auto const& t : handle.column_types()) {
    REQUIRE(t.id() == cudf::type_id::INT32);
  }

  // column() returns the column at the current position; row count is
  // unaffected by dropping columns.
  std::array<std::size_t, 1> drop{0};
  handle.drop_columns(drop);  // -> 20, 30, 40
  REQUIRE(handle.num_rows() == 1);
  REQUIRE(handle.column_types().size() == 3);

  std::int32_t tag = 0;
  cudaMemcpy(&tag, handle.column(0).data<std::int32_t>(), sizeof(tag), cudaMemcpyDeviceToHost);
  REQUIRE(tag == 20);

  REQUIRE_THROWS_AS(handle.column(3), std::out_of_range);

  // Empty handle: zero rows, no columns.
  owning_table_view empty;
  REQUIRE(empty.num_rows() == 0);
  REQUIRE(empty.column_types().empty());
  REQUIRE_THROWS_AS(empty.column(0), std::out_of_range);
}

TEST_CASE("owning_table_view drop + materialize moves buffers (no copy)", "[owning_table_view]")
{
  auto table = tagged_table({10, 20, 30});
  // Capture the original device pointers before handing the table over.
  auto original_ptrs = data_ptrs(table->view());

  owning_table_view handle{std::move(table)};

  // Drop the middle column — pure view manipulation, no allocation.
  std::array<std::size_t, 1> drop{1};
  handle.drop_columns(drop);
  REQUIRE(handle.n_columns() == 2);
  REQUIRE(read_tags(handle.view()) == std::vector<std::int32_t>{10, 30});

  // Materialize back to an owned table: surviving columns keep their buffers.
  auto result = handle.release(test_stream(), test_mr());
  REQUIRE(result != nullptr);
  REQUIRE(read_tags(result->view()) == std::vector<std::int32_t>{10, 30});

  auto result_ptrs = data_ptrs(result->view());
  REQUIRE(result_ptrs[0] == original_ptrs[0]);  // column 10 buffer preserved
  REQUIRE(result_ptrs[1] == original_ptrs[2]);  // column 30 buffer preserved

  // Handle is now empty.
  REQUIRE_FALSE(static_cast<bool>(handle));
  REQUIRE(handle.release(test_stream(), test_mr()) == nullptr);
}

TEST_CASE("owning_table_view reorder swaps columns", "[owning_table_view]")
{
  owning_table_view handle{tagged_table({10, 20, 30})};

  // Swap positions 0 and 2.
  std::array<std::pair<std::size_t, std::size_t>, 1> swaps{{{0, 2}}};
  handle.reorder_columns(swaps);
  REQUIRE(read_tags(handle.view()) == std::vector<std::int32_t>{30, 20, 10});

  auto result = handle.release(test_stream(), test_mr());
  REQUIRE(read_tags(result->view()) == std::vector<std::int32_t>{30, 20, 10});
}

TEST_CASE("owning_table_view reorder then drop compose on current positions", "[owning_table_view]")
{
  owning_table_view handle{tagged_table({10, 20, 30, 40})};

  std::array<std::pair<std::size_t, std::size_t>, 1> swaps{{{0, 3}}};
  handle.reorder_columns(swaps);  // -> 40, 20, 30, 10
  std::array<std::size_t, 2> drop{1, 2};
  handle.drop_columns(drop);  // drop current positions 1 and 2 -> 40, 10
  REQUIRE(handle.n_columns() == 2);
  REQUIRE(read_tags(handle.view()) == std::vector<std::int32_t>{40, 10});
}

TEST_CASE("owning_table_view select_columns projects and reorders (no copy)", "[owning_table_view]")
{
  auto table         = tagged_table({10, 20, 30, 40});
  auto original_ptrs = data_ptrs(table->view());

  owning_table_view handle{std::move(table)};

  // Project down to columns 3 and 0, in that order.
  std::array<std::size_t, 2> keep{3, 0};
  handle.select_columns(keep);
  REQUIRE(handle.n_columns() == 2);
  REQUIRE(read_tags(handle.view()) == std::vector<std::int32_t>{40, 10});

  // select_columns composes on current positions: now keep only position 1 (10).
  std::array<std::size_t, 1> keep2{1};
  handle.select_columns(keep2);
  REQUIRE(read_tags(handle.view()) == std::vector<std::int32_t>{10});

  // Zero-alloc: the surviving buffer is the original column 10 buffer.
  auto result = handle.release(test_stream(), test_mr());
  REQUIRE(data_ptrs(result->view())[0] == original_ptrs[0]);
}

TEST_CASE("owning_table_view select_columns rejects bad positions", "[owning_table_view]")
{
  owning_table_view handle{tagged_table({10, 20, 30})};

  std::array<std::size_t, 2> out_of_range{0, 3};
  REQUIRE_THROWS_AS(handle.select_columns(out_of_range), std::out_of_range);

  std::array<std::size_t, 2> duplicate{1, 1};
  REQUIRE_THROWS_AS(handle.select_columns(duplicate), std::invalid_argument);

  // Unchanged after rejected operations.
  REQUIRE(handle.n_columns() == 3);
  REQUIRE(read_tags(handle.view()) == std::vector<std::int32_t>{10, 20, 30});
}

TEST_CASE("owning_table_view rejects out-of-range positions", "[owning_table_view]")
{
  owning_table_view handle{tagged_table({10, 20, 30})};

  std::array<std::size_t, 1> bad_drop{3};
  REQUIRE_THROWS_AS(handle.drop_columns(bad_drop), std::out_of_range);

  std::array<std::pair<std::size_t, std::size_t>, 1> bad_swap{{{0, 3}}};
  REQUIRE_THROWS_AS(handle.reorder_columns(bad_swap), std::out_of_range);

  // The handle is unchanged after the rejected operations.
  REQUIRE(handle.n_columns() == 3);
  REQUIRE(read_tags(handle.view()) == std::vector<std::int32_t>{10, 20, 30});
}

TEST_CASE("owning_table_view generic owner materializes by copy", "[owning_table_view]")
{
  auto table         = tagged_table({10, 20, 30});
  auto original_ptrs = data_ptrs(table->view());
  auto base_view     = table->view();

  // Type-erase a non-no_alloc owner; materialization must copy.
  owning_table_view handle{table_keepalive{std::move(table)}, base_view};
  REQUIRE(handle.n_columns() == 3);

  std::array<std::size_t, 1> drop{1};
  handle.drop_columns(drop);
  REQUIRE(read_tags(handle.view()) == std::vector<std::int32_t>{10, 30});

  auto result = handle.release(test_stream(), test_mr());
  REQUIRE(read_tags(result->view()) == std::vector<std::int32_t>{10, 30});

  // Copying path: the materialized buffers differ from the originals.
  auto result_ptrs = data_ptrs(result->view());
  REQUIRE(result_ptrs[0] != original_ptrs[0]);
  REQUIRE(result_ptrs[1] != original_ptrs[2]);
}
