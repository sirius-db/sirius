#include "gpu_physical_hash_join.hpp"

namespace duckdb {

// GPUPhysicalHashJoin::GPUPhysicalHashJoin(PhysicalOperator op) {
// };

GPUPhysicalHashJoin::GPUPhysicalHashJoin(LogicalOperator &op, unique_ptr<GPUPhysicalOperator> left,
                                   unique_ptr<GPUPhysicalOperator> right, vector<JoinCondition> cond, JoinType join_type,
                                   const vector<idx_t> &left_projection_map, const vector<idx_t> &right_projection_map,
                                   vector<LogicalType> delim_types, idx_t estimated_cardinality) {};

} // namespace duckdb