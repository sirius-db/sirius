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

// Host launcher contract of round_floating_point: empty inputs and unsupported types. Values are
// covered against DuckDB in test/cpp/integration/test_gpu_execution_round.cpp.

#include "catch.hpp"

// sirius
#include <expression_evaluator/round_floating_point.hpp>
#include <sirius/exception.hpp>

// cudf
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

TEST_CASE("round_floating_point returns an empty column of the input type",
          "[expression_evaluator][round]")
{
  cuda::stream_ref const stream = cudf::get_default_stream();
  auto mr                       = cudf::get_current_device_resource_ref();
  auto const type_id            = GENERATE(cudf::type_id::FLOAT32, cudf::type_id::FLOAT64);
  auto const input              = cudf::make_numeric_column(
    cudf::data_type{type_id}, 0, cudf::mask_state::UNALLOCATED, stream, mr);
  auto const result = sirius::round_floating_point(input->view(), 2, stream, mr);
  REQUIRE(result->type().id() == type_id);
  REQUIRE(result->size() == 0);
}

TEST_CASE("round_floating_point rejects non-floating-point input", "[expression_evaluator][round]")
{
  cuda::stream_ref const stream = cudf::get_default_stream();
  auto mr                       = cudf::get_current_device_resource_ref();
  auto const input              = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, 3, cudf::mask_state::UNALLOCATED, stream, mr);
  REQUIRE_THROWS_AS(sirius::round_floating_point(input->view(), 2, stream, mr),
                    sirius::invalid_input_exception);
}
