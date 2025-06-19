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
