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

#include "duckdb/common/constants.hpp"

namespace sirius::op {

//===--------------------------------------------------------------------===//
// Physical Operator Types
//===--------------------------------------------------------------------===//
enum class SiriusPhysicalOperatorType : uint8_t {
  INVALID,
  ORDER_BY,
  LIMIT,
  STREAMING_LIMIT,
  LIMIT_PERCENT,
  TOP_N,
  WINDOW,
  UNNEST,
  UNGROUPED_AGGREGATE,
  HASH_GROUP_BY,
  PERFECT_HASH_GROUP_BY,
  PARTITIONED_AGGREGATE,
  FILTER,
  PROJECTION,
  COPY_TO_FILE,
  BATCH_COPY_TO_FILE,
  RESERVOIR_SAMPLE,
  STREAMING_SAMPLE,
  STREAMING_WINDOW,
  PIVOT,
  COPY_DATABASE,

  // -----------------------------
  // Scans
  // -----------------------------
  TABLE_SCAN,
  DUMMY_SCAN,
  COLUMN_DATA_SCAN,
  CHUNK_SCAN,
  RECURSIVE_CTE_SCAN,
  RECURSIVE_RECURRING_CTE_SCAN,
  CTE_SCAN,
  DELIM_SCAN,
  EXPRESSION_SCAN,
  POSITIONAL_SCAN,
  // -----------------------------
  // Joins
  // -----------------------------
  BLOCKWISE_NL_JOIN,
  NESTED_LOOP_JOIN,
  HASH_JOIN,
  CROSS_PRODUCT,
  PIECEWISE_MERGE_JOIN,
  IE_JOIN,
  LEFT_DELIM_JOIN,
  RIGHT_DELIM_JOIN,
  POSITIONAL_JOIN,
  ASOF_JOIN,
  // -----------------------------
  // SetOps
  // -----------------------------
  UNION,
  RECURSIVE_CTE,
  RECURSIVE_KEY_CTE,
  CTE,

  // -----------------------------
  // Updates
  // -----------------------------
  INSERT,
  BATCH_INSERT,
  DELETE_OPERATOR,
  UPDATE,
  MERGE_INTO,

  // -----------------------------
  // Schema
  // -----------------------------
  CREATE_TABLE,
  CREATE_TABLE_AS,
  BATCH_CREATE_TABLE_AS,
  CREATE_INDEX,
  ALTER,
  CREATE_SEQUENCE,
  CREATE_VIEW,
  CREATE_SCHEMA,
  CREATE_MACRO,
  DROP,
  PRAGMA,
  TRANSACTION,
  CREATE_TYPE,
  ATTACH,
  DETACH,

  // -----------------------------
  // Helpers
  // -----------------------------
  EXPLAIN,
  EXPLAIN_ANALYZE,
  EMPTY_RESULT,
  EXECUTE,
  PREPARE,
  VACUUM,
  EXPORT,
  SET,
  SET_VARIABLE,
  LOAD,
  INOUT_FUNCTION,
  RESULT_COLLECTOR,
  RESET,
  EXTENSION,
  VERIFY_VECTOR,
  UPDATE_EXTENSIONS,

  // -----------------------------
  // Secret
  // -----------------------------
  CREATE_SECRET,

  // -----------------------------
  // Sirius Specific Operators
  // -----------------------------
  PARTITION,
  CONCAT,
  MERGE_SORT,
  MERGE_GROUP_BY,
  MERGE_TOP_N,
  MERGE_AGGREGATE,
  SORT_PARTITION,
  SORT_SAMPLE,
  DUCKDB_SCAN,
  PARQUET_SCAN
};

std::string SiriusPhysicalOperatorToString(SiriusPhysicalOperatorType type);

}  // namespace sirius::op
