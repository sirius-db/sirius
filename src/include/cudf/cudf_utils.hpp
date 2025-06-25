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

#pragma once

#include <cudf/table/table.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>

#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/round.hpp>
#include <cudf/unary.hpp>
#include <cudf/ast/expressions.hpp>

#include "duckdb/common/exception.hpp"

namespace duckdb {

inline bool IsCudfTypeDecimal(const cudf::data_type& type) {
  return type.id() == cudf::type_id::DECIMAL32 ||
         type.id() == cudf::type_id::DECIMAL64 ||
         type.id() == cudf::type_id::DECIMAL128;
}

inline int GetCudfDecimalTypeSize(const cudf::data_type& type) {
  if (type.id() == cudf::type_id::DECIMAL32) {
    return sizeof(int32_t);
  }
  if (type.id() == cudf::type_id::DECIMAL64) {
    return sizeof(int64_t);
  }
  if (type.id() == cudf::type_id::DECIMAL128) {
    return sizeof(__int128_t);
  }
  throw InternalException("Non decimal cudf type called in `GetCudfDecimalTypeSize`: %d",
                          static_cast<int>(type.id()));
}

}
