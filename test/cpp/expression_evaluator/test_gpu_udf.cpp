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

#include "ast_test_support.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <expression/ast/gpu_function_call.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <sirius/udf.hpp>

#include <memory>
#include <string_view>
#include <vector>

namespace {
using namespace sirius::expr_test;

auto const integer = sirius::logical_type::make(sirius::type_id::INTEGER);

// A distinct resource identity proves dispatch forwards the caller's allocator.
class test_resource {
 public:
  rmm::device_async_resource_ref upstream = cudf::get_current_device_resource_ref();
  std::shared_ptr<int> allocations        = std::make_shared<int>(0);

  void* allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    ++*allocations;
    return upstream.allocate(stream, bytes, alignment);
  }
  void deallocate(::cuda::stream_ref stream,
                  void* pointer,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    upstream.deallocate(stream, pointer, bytes, alignment);
  }
  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    ++*allocations;
    return upstream.allocate_sync(bytes, alignment);
  }
  void deallocate_sync(void* pointer, std::size_t bytes, std::size_t alignment) noexcept
  {
    upstream.deallocate_sync(pointer, bytes, alignment);
  }
  bool operator==(test_resource const& other) const noexcept
  {
    return allocations == other.allocations;
  }
  friend void get_property(test_resource const&, ::cuda::mr::device_accessible) noexcept {}
};

std::unique_ptr<cudf::table> make_input(::cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr,
                                        bool nullable = false,
                                        bool empty    = false)
{
  std::vector<int32_t> const values{3, -4, 8, 0, 11};
  cudf::bitmask_type const mask = 0b10101;
  auto const rows               = empty ? 0 : static_cast<cudf::size_type>(values.size());
  auto column                   = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, rows, cudf::mask_state::UNALLOCATED, stream, mr);
  if (rows != 0) {
    REQUIRE(cudaMemcpyAsync(column->mutable_view().data<int32_t>(),
                            values.data(),
                            rows * sizeof(int32_t),
                            cudaMemcpyHostToDevice,
                            stream.get()) == cudaSuccess);
  }
  if (nullable && rows != 0) {
    auto device_mask = cudf::create_null_mask(rows, cudf::mask_state::UNINITIALIZED, stream, mr);
    REQUIRE(cudaMemcpyAsync(
              device_mask.data(), &mask, sizeof(mask), cudaMemcpyHostToDevice, stream.get()) ==
            cudaSuccess);
    column->set_null_mask(std::move(device_mask), 2);
  }
  // Host upload sources must outlive the async copies.
  REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<ast_node> make_call(sirius::gpu_scalar_callback callback,
                                    bool propagate_nulls,
                                    bool with_literal = false)
{
  std::vector<std::unique_ptr<ast_node>> arguments;
  arguments.push_back(make_ref_typed(0, integer));
  if (with_literal) { arguments.push_back(make_int_const(6)); }
  auto implementation =
    std::make_shared<sirius::gpu_scalar_function>(sirius::gpu_scalar_function{std::move(callback)});
  return std::make_unique<ast_node>(
    sirius::ast::gpu_function_call{implementation, std::move(arguments), integer, propagate_nulls});
}

auto strategy()
{
  return GENERATE(sirius::expression_evaluator_strategy::MATERIALIZE,
                  sirius::expression_evaluator_strategy::AST_INTERPRET,
                  sirius::expression_evaluator_strategy::AST_JIT);
}
}  // namespace

TEST_CASE("GPU UDF uses task stream and allocator and broadcasts scalar arguments",
          "[gpu_udf][expression_evaluator]")
{
  auto const selected_strategy = strategy();
  rmm::cuda_stream owned_stream;
  ::cuda::stream_ref const stream{owned_stream.value()};
  test_resource resource;
  rmm::device_async_resource_ref mr{resource};
  auto input = make_input(stream, mr);
  std::vector<std::unique_ptr<cudf::column>> narrowed_columns;
  narrowed_columns.push_back(
    cudf::cast(input->view().column(0), cudf::data_type{cudf::type_id::INT8}, stream, mr));
  input                         = std::make_unique<cudf::table>(std::move(narrowed_columns));
  auto const allocations_before = *resource.allocations;
  bool invoked                  = false;
  auto call                     = make_call(
    [&](std::span<cudf::column_view const> args,
        cudf::size_type rows,
        ::cuda::stream_ref task_stream,
        rmm::device_async_resource_ref task_mr) {
      invoked = true;
      CHECK(task_stream.get() == stream.get());
      CHECK(task_mr == mr);
      REQUIRE(args.size() == 2);
      CHECK(args[0].size() == rows);
      CHECK(args[1].size() == rows);
      CHECK(args[0].type().id() == cudf::type_id::INT32);
      return cudf::binary_operation(args[0],
                                    args[1],
                                    cudf::binary_operator::ADD,
                                    cudf::data_type{cudf::type_id::INT32},
                                    task_stream,
                                    task_mr);
    },
    true,
    true);
  std::vector<std::unique_ptr<ast_node>> outer_args;
  outer_args.push_back(std::move(call));
  outer_args.push_back(make_int_const(2));
  auto product = make_func(sirius::function_id::mul, std::move(outer_args), integer);
  std::vector<std::unique_ptr<ast_node>> sum_args;
  sum_args.push_back(std::move(product));
  sum_args.push_back(make_int_const(1));
  // Two outer operations cross the default AST threshold, exercising the
  // native call as a materialization boundary under both AST strategies.
  auto expression = make_func(sirius::function_id::add, std::move(sum_args), integer);
  sirius::expression_evaluator evaluator{
    *expression, mr, stream, sirius::expression_query_policy{selected_strategy}};
  auto result = evaluator.evaluate(input->view());
  REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);
  REQUIRE(invoked);
  CHECK(*resource.allocations > allocations_before);
  REQUIRE(copy_column_to_host<int32_t>(result->view().column(0)) ==
          std::vector<int32_t>{19, 5, 29, 13, 35});
  // Results own their memory rather than borrowing the callback's argument columns.
  input.reset();
  CHECK(copy_column_to_host<int32_t>(result->view().column(0))[0] == 19);

  // Reuse this evaluator with a smaller sliced batch and a different NULL mask.
  // A stale restoration cache or native temporary would retain the first five rows.
  auto second_input = make_input(stream, mr, true);
  auto second_narrowed =
    cudf::cast(second_input->view().column(0), cudf::data_type{cudf::type_id::INT8}, stream, mr);
  auto second_slice  = cudf::slice(second_narrowed->view(), {1, 4}, stream);
  auto second_result = evaluator.evaluate(cudf::table_view{{second_slice[0]}});
  REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);
  REQUIRE(second_result->num_rows() == 3);
  CHECK(copy_valids_to_host(second_result->view().column(0)) ==
        std::vector<bool>{false, true, false});
  CHECK(copy_column_to_host<int32_t>(second_result->view().column(0))[1] == 29);
  // Re-evaluation cannot invalidate an already returned owning result.
  CHECK(copy_column_to_host<int32_t>(result->view().column(0)) ==
        std::vector<int32_t>{19, 5, 29, 13, 35});
}

TEST_CASE("GPU UDF default and special NULL policies differ", "[gpu_udf][expression_evaluator]")
{
  auto const selected_strategy    = strategy();
  auto const callback_valid       = GENERATE(true, false);
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto mr                         = cudf::get_current_device_resource_ref();
  auto input                      = make_input(stream, mr, true);
  for (bool propagate : {true, false}) {
    auto expression = make_call(
      [callback_valid](std::span<cudf::column_view const>,
                       cudf::size_type rows,
                       ::cuda::stream_ref task_stream,
                       rmm::device_async_resource_ref task_mr) {
        cudf::numeric_scalar<int32_t> const value{17, callback_valid, task_stream, task_mr};
        return cudf::make_column_from_scalar(value, rows, task_stream, task_mr);
      },
      propagate);
    sirius::expression_evaluator evaluator{
      *expression, mr, stream, sirius::expression_query_policy{selected_strategy}};
    auto result = evaluator.evaluate(input->view());
    REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);
    CHECK(copy_valids_to_host(result->view().column(0)) ==
          (!callback_valid ? std::vector<bool>(5, false)
           : propagate     ? std::vector<bool>{true, false, true, false, true}
                           : std::vector<bool>(5, true)));
  }
}

TEST_CASE("GPU UDF string NULL policies preserve validity and remove invalid payloads",
          "[gpu_udf][expression_evaluator]")
{
  auto const selected_strategy    = strategy();
  auto const callback_valid       = GENERATE(true, false);
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto mr                         = cudf::get_current_device_resource_ref();
  auto input                      = make_input(stream, mr, true);
  auto const varchar              = sirius::logical_type::make(sirius::type_id::VARCHAR);
  for (bool propagate : {true, false}) {
    auto implementation = std::make_shared<sirius::gpu_scalar_function>(
      sirius::gpu_scalar_function{[callback_valid](std::span<cudf::column_view const>,
                                                   cudf::size_type rows,
                                                   ::cuda::stream_ref task_stream,
                                                   rmm::device_async_resource_ref task_mr) {
        // A valid callback result has payload at every row, including input NULLs.
        // Default NULL handling must remove those bytes after adding its mask.
        cudf::string_scalar const text{
          std::string_view{"tag"}, callback_valid, task_stream, task_mr};
        return cudf::make_column_from_scalar(text, rows, task_stream, task_mr);
      }});
    std::vector<std::unique_ptr<ast_node>> arguments;
    arguments.push_back(make_ref_typed(0, integer));
    ast_node const expression{
      sirius::ast::gpu_function_call{implementation, std::move(arguments), varchar, propagate}};
    sirius::expression_evaluator evaluator{
      expression, mr, stream, sirius::expression_query_policy{selected_strategy}};
    auto result = evaluator.evaluate(input->view());
    REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);
    auto const column = result->view().column(0);
    auto const valid  = !callback_valid ? std::vector<bool>(5, false)
                        : propagate     ? std::vector<bool>{true, false, true, false, true}
                                        : std::vector<bool>(5, true);
    CHECK(copy_valids_to_host(column) == valid);
    CHECK_FALSE(cudf::has_nonempty_nulls(column, stream));
    CHECK(cudf::strings_column_view{column}.chars_size(stream) == (!callback_valid ? 0
                                                                   : propagate     ? 9
                                                                                   : 15));
    std::vector<std::string> expected;
    for (bool present : valid) {
      expected.emplace_back(present ? "tag" : "");
    }
    CHECK(copy_string_column_to_host(column) == expected);
  }
}

TEST_CASE("GPU UDF rejects invalid callback results", "[gpu_udf][expression_evaluator]")
{
  auto const invalid_result       = GENERATE(0, 1, 2);
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto mr                         = cudf::get_current_device_resource_ref();
  auto input                      = make_input(stream, mr);
  auto expression                 = make_call(
    [invalid_result](std::span<cudf::column_view const>,
                     cudf::size_type rows,
                     ::cuda::stream_ref task_stream,
                     rmm::device_async_resource_ref task_mr) -> std::unique_ptr<cudf::column> {
      if (invalid_result == 0) { return nullptr; }
      auto const type = invalid_result == 1 ? cudf::type_id::INT64 : cudf::type_id::INT32;
      return cudf::make_numeric_column(cudf::data_type{type},
                                       invalid_result == 2 ? rows + 1 : rows,
                                       cudf::mask_state::UNALLOCATED,
                                       task_stream,
                                       task_mr);
    },
    true);
  sirius::expression_evaluator evaluator{*expression, mr, stream};
  REQUIRE_THROWS(evaluator.evaluate(input->view()));
}

TEST_CASE("GPU UDF accepts empty batches and sliced column views",
          "[gpu_udf][expression_evaluator]")
{
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto mr                         = cudf::get_current_device_resource_ref();
  auto expression                 = make_call(
    [](std::span<cudf::column_view const> args,
       cudf::size_type,
       ::cuda::stream_ref task_stream,
       rmm::device_async_resource_ref task_mr) {
      return std::make_unique<cudf::column>(args[0], task_stream, task_mr);
    },
    true);
  auto empty = make_input(stream, mr, false, true);
  sirius::expression_evaluator empty_evaluator{*expression, mr, stream};
  auto empty_result = empty_evaluator.evaluate(empty->view());
  CHECK(empty_result->num_rows() == 0);
  CHECK(empty_result->view().column(0).type().id() == cudf::type_id::INT32);

  auto input = make_input(stream, mr);
  auto slice = cudf::slice(input->view().column(0), {1, 4}, stream);
  sirius::expression_evaluator sliced_evaluator{*expression, mr, stream};
  auto result = sliced_evaluator.evaluate(cudf::table_view{{slice[0]}});
  REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);
  CHECK(copy_column_to_host<int32_t>(result->view().column(0)) == std::vector<int32_t>{-4, 8, 0});

  auto nullable_input = make_input(stream, mr, true);
  auto nullable_slice = cudf::slice(nullable_input->view().column(0), {1, 4}, stream);
  sirius::expression_evaluator nullable_evaluator{*expression, mr, stream};
  auto nullable_result = nullable_evaluator.evaluate(cudf::table_view{{nullable_slice[0]}});
  REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);
  CHECK(copy_valids_to_host(nullable_result->view().column(0)) ==
        std::vector<bool>{false, true, false});
}
