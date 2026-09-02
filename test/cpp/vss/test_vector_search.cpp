/*
 * Copyright 2025, Sirius Contributors.
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

// test
#include <catch.hpp>

// sirius
#include <helper/logical_type.hpp>
#include <vss/vector_search_internal.hpp>

// cudf
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <vector>

namespace {

using sirius::logical_type;
using sirius::type_id;
using sirius::vss::make_empty_vss_output;

logical_type int_type() { return logical_type::make(type_id::INTEGER); }

// A Sirius-style ARRAY<FLOAT>[dim] type, so the LIST mapping can be asserted on
// the empty output.
logical_type float_array_type(uint32_t dim)
{
  return logical_type::make_array(logical_type::make(type_id::FLOAT), dim);
}

}  // namespace

// The empty output is built from the catalog types the bind step resolves, so it
// works even when the pinned table has no data batches to read a type from.
TEST_CASE("make_empty_vss_output builds the result schema from catalog types", "[vss]")
{
  SECTION("output columns keep their types; FLOAT32 distance appended; zero rows")
  {
    auto out = make_empty_vss_output({int_type(), float_array_type(4)});
    REQUIRE(out->num_columns() == 3);
    REQUIRE(out->num_rows() == 0);
    REQUIRE(out->get_column(0).type().id() == cudf::type_id::INT32);
    REQUIRE(out->get_column(1).type().id() == cudf::type_id::LIST);
    REQUIRE(out->get_column(2).type().id() == cudf::type_id::FLOAT32);
  }

  SECTION("output column order is honored, distance always last")
  {
    auto out = make_empty_vss_output({float_array_type(4), int_type()});
    REQUIRE(out->num_columns() == 3);
    REQUIRE(out->get_column(0).type().id() == cudf::type_id::LIST);
    REQUIRE(out->get_column(1).type().id() == cudf::type_id::INT32);
    REQUIRE(out->get_column(2).type().id() == cudf::type_id::FLOAT32);
  }

  SECTION("no output columns yields just the distance column")
  {
    auto out = make_empty_vss_output({});
    REQUIRE(out->num_columns() == 1);
    REQUIRE(out->num_rows() == 0);
    REQUIRE(out->get_column(0).type().id() == cudf::type_id::FLOAT32);
  }
}
