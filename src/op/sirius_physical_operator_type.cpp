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

#include "op/sirius_physical_operator_type.hpp"

namespace sirius::op {

// LCOV_EXCL_START
std::string SiriusPhysicalOperatorToString(SiriusPhysicalOperatorType type)
{
  switch (type) {
    case SiriusPhysicalOperatorType::TABLE_SCAN: return "TABLE_SCAN";
    case SiriusPhysicalOperatorType::DUMMY_SCAN: return "DUMMY_SCAN";
    case SiriusPhysicalOperatorType::CHUNK_SCAN: return "CHUNK_SCAN";
    case SiriusPhysicalOperatorType::COLUMN_DATA_SCAN: return "COLUMN_DATA_SCAN";
    case SiriusPhysicalOperatorType::DELIM_SCAN: return "DELIM_SCAN";
    case SiriusPhysicalOperatorType::ORDER_BY: return "ORDER_BY";
    case SiriusPhysicalOperatorType::LIMIT: return "LIMIT";
    case SiriusPhysicalOperatorType::LIMIT_PERCENT: return "LIMIT_PERCENT";
    case SiriusPhysicalOperatorType::STREAMING_LIMIT: return "STREAMING_LIMIT";
    case SiriusPhysicalOperatorType::RESERVOIR_SAMPLE: return "RESERVOIR_SAMPLE";
    case SiriusPhysicalOperatorType::STREAMING_SAMPLE: return "STREAMING_SAMPLE";
    case SiriusPhysicalOperatorType::TOP_N: return "TOP_N";
    case SiriusPhysicalOperatorType::WINDOW: return "WINDOW";
    case SiriusPhysicalOperatorType::STREAMING_WINDOW: return "STREAMING_WINDOW";
    case SiriusPhysicalOperatorType::UNNEST: return "UNNEST";
    case SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE: return "UNGROUPED_AGGREGATE";
    case SiriusPhysicalOperatorType::HASH_GROUP_BY: return "HASH_GROUP_BY";
    case SiriusPhysicalOperatorType::PERFECT_HASH_GROUP_BY: return "PERFECT_HASH_GROUP_BY";
    case SiriusPhysicalOperatorType::PARTITIONED_AGGREGATE: return "PARTITIONED_AGGREGATE";
    case SiriusPhysicalOperatorType::FILTER: return "FILTER";
    case SiriusPhysicalOperatorType::PROJECTION: return "PROJECTION";
    case SiriusPhysicalOperatorType::COPY_TO_FILE: return "COPY_TO_FILE";
    case SiriusPhysicalOperatorType::BATCH_COPY_TO_FILE: return "BATCH_COPY_TO_FILE";
    case SiriusPhysicalOperatorType::LEFT_DELIM_JOIN: return "LEFT_DELIM_JOIN";
    case SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN: return "RIGHT_DELIM_JOIN";
    case SiriusPhysicalOperatorType::BLOCKWISE_NL_JOIN: return "BLOCKWISE_NL_JOIN";
    case SiriusPhysicalOperatorType::NESTED_LOOP_JOIN: return "NESTED_LOOP_JOIN";
    case SiriusPhysicalOperatorType::HASH_JOIN: return "HASH_JOIN";
    case SiriusPhysicalOperatorType::PIECEWISE_MERGE_JOIN: return "PIECEWISE_MERGE_JOIN";
    case SiriusPhysicalOperatorType::IE_JOIN: return "IE_JOIN";
    case SiriusPhysicalOperatorType::ASOF_JOIN: return "ASOF_JOIN";
    case SiriusPhysicalOperatorType::CROSS_PRODUCT: return "CROSS_PRODUCT";
    case SiriusPhysicalOperatorType::POSITIONAL_JOIN: return "POSITIONAL_JOIN";
    case SiriusPhysicalOperatorType::POSITIONAL_SCAN: return "POSITIONAL_SCAN";
    case SiriusPhysicalOperatorType::UNION: return "UNION";
    case SiriusPhysicalOperatorType::INSERT: return "INSERT";
    case SiriusPhysicalOperatorType::BATCH_INSERT: return "BATCH_INSERT";
    case SiriusPhysicalOperatorType::DELETE_OPERATOR: return "DELETE";
    case SiriusPhysicalOperatorType::UPDATE: return "UPDATE";
    case SiriusPhysicalOperatorType::MERGE_INTO: return "MERGE_INTO";
    case SiriusPhysicalOperatorType::EMPTY_RESULT: return "EMPTY_RESULT";
    case SiriusPhysicalOperatorType::CREATE_TABLE: return "CREATE_TABLE";
    case SiriusPhysicalOperatorType::CREATE_TABLE_AS: return "CREATE_TABLE_AS";
    case SiriusPhysicalOperatorType::BATCH_CREATE_TABLE_AS: return "BATCH_CREATE_TABLE_AS";
    case SiriusPhysicalOperatorType::CREATE_INDEX: return "CREATE_INDEX";
    case SiriusPhysicalOperatorType::EXPLAIN: return "EXPLAIN";
    case SiriusPhysicalOperatorType::EXPLAIN_ANALYZE: return "EXPLAIN_ANALYZE";
    case SiriusPhysicalOperatorType::EXECUTE: return "EXECUTE";
    case SiriusPhysicalOperatorType::VACUUM: return "VACUUM";
    case SiriusPhysicalOperatorType::RECURSIVE_CTE: return "REC_CTE";
    case SiriusPhysicalOperatorType::RECURSIVE_KEY_CTE: return "REC_KEY_CTE";
    case SiriusPhysicalOperatorType::CTE: return "CTE";
    case SiriusPhysicalOperatorType::RECURSIVE_CTE_SCAN: return "REC_CTE_SCAN";
    case SiriusPhysicalOperatorType::RECURSIVE_RECURRING_CTE_SCAN: return "REC_REC_CTE_SCAN";
    case SiriusPhysicalOperatorType::CTE_SCAN: return "CTE_SCAN";
    case SiriusPhysicalOperatorType::EXPRESSION_SCAN: return "EXPRESSION_SCAN";
    case SiriusPhysicalOperatorType::ALTER: return "ALTER";
    case SiriusPhysicalOperatorType::CREATE_SEQUENCE: return "CREATE_SEQUENCE";
    case SiriusPhysicalOperatorType::CREATE_VIEW: return "CREATE_VIEW";
    case SiriusPhysicalOperatorType::CREATE_SCHEMA: return "CREATE_SCHEMA";
    case SiriusPhysicalOperatorType::CREATE_MACRO: return "CREATE_MACRO";
    case SiriusPhysicalOperatorType::CREATE_SECRET: return "CREATE_SECRET";
    case SiriusPhysicalOperatorType::DROP: return "DROP";
    case SiriusPhysicalOperatorType::PRAGMA: return "PRAGMA";
    case SiriusPhysicalOperatorType::TRANSACTION: return "TRANSACTION";
    case SiriusPhysicalOperatorType::PREPARE: return "PREPARE";
    case SiriusPhysicalOperatorType::EXPORT: return "EXPORT";
    case SiriusPhysicalOperatorType::SET: return "SET";
    case SiriusPhysicalOperatorType::SET_VARIABLE: return "SET_VARIABLE";
    case SiriusPhysicalOperatorType::RESET: return "RESET";
    case SiriusPhysicalOperatorType::LOAD: return "LOAD";
    case SiriusPhysicalOperatorType::INOUT_FUNCTION: return "INOUT_FUNCTION";
    case SiriusPhysicalOperatorType::CREATE_TYPE: return "CREATE_TYPE";
    case SiriusPhysicalOperatorType::ATTACH: return "ATTACH";
    case SiriusPhysicalOperatorType::DETACH: return "DETACH";
    case SiriusPhysicalOperatorType::RESULT_COLLECTOR: return "RESULT_COLLECTOR";
    case SiriusPhysicalOperatorType::EXTENSION: return "EXTENSION";
    case SiriusPhysicalOperatorType::PIVOT: return "PIVOT";
    case SiriusPhysicalOperatorType::COPY_DATABASE: return "COPY_DATABASE";
    case SiriusPhysicalOperatorType::VERIFY_VECTOR: return "VERIFY_VECTOR";
    case SiriusPhysicalOperatorType::UPDATE_EXTENSIONS: return "UPDATE_EXTENSIONS";
    case SiriusPhysicalOperatorType::PARTITION: return "PARTITION";
    case SiriusPhysicalOperatorType::CONCAT: return "CONCAT";
    case SiriusPhysicalOperatorType::MERGE_SORT: return "MERGE_SORT";
    case SiriusPhysicalOperatorType::MERGE_GROUP_BY: return "MERGE_GROUP_BY";
    case SiriusPhysicalOperatorType::MERGE_TOP_N: return "MERGE_TOP_N";
    case SiriusPhysicalOperatorType::MERGE_AGGREGATE: return "MERGE_AGGREGATE";
    case SiriusPhysicalOperatorType::SORT_PARTITION: return "SORT_PARTITION";
    case SiriusPhysicalOperatorType::SORT_SAMPLE: return "SORT_SAMPLE";
    case SiriusPhysicalOperatorType::DUCKDB_SCAN: return "DUCKDB_SCAN";
    case SiriusPhysicalOperatorType::INVALID: break;
  }
  return "INVALID";
}
// LCOV_EXCL_STOP

}  // namespace sirius::op
