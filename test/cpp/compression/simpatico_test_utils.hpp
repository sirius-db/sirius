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

#pragma once

// Shorthands the simpatico tests use everywhere, over the shared test helpers.

#include "operator/operator_test_utils.hpp"
#include "operator/operator_type_traits.hpp"
#include "utils/data_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <cstdint>
#include <memory>
#include <vector>

inline std::unique_ptr<cudf::column> int32_column(std::vector<std::int32_t> const& values)
{
  return sirius::test::vector_to_cudf_column<
    sirius::test::operator_utils::gpu_type_traits<std::int32_t>>(values);
}

inline std::vector<std::int32_t> read_back(cudf::column_view const& v)
{
  return sirius::test::operator_utils::copy_column_to_host<std::int32_t>(v);
}
