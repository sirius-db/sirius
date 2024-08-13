#include "operator/gpu_physical_grouped_aggregate.hpp"
#include "duckdb/main/client_context.hpp"

namespace duckdb {

// GPUPhysicalGroupedAggregate::GPUPhysicalGroupedAggregate(PhysicalOperator op) {
// };

GPUPhysicalGroupedAggregate::GPUPhysicalGroupedAggregate(ClientContext &context, vector<LogicalType> types,
                                             vector<unique_ptr<Expression>> expressions, idx_t estimated_cardinality)
    : GPUPhysicalGroupedAggregate(context, std::move(types), std::move(expressions), {}, estimated_cardinality) {
}

GPUPhysicalGroupedAggregate::GPUPhysicalGroupedAggregate(ClientContext &context, vector<LogicalType> types,
                                             vector<unique_ptr<Expression>> expressions,
                                             vector<unique_ptr<Expression>> groups_p, idx_t estimated_cardinality)
    : GPUPhysicalGroupedAggregate(context, std::move(types), std::move(expressions), std::move(groups_p), {}, {},
                            estimated_cardinality) {
}

GPUPhysicalGroupedAggregate::GPUPhysicalGroupedAggregate(ClientContext &context, vector<LogicalType> types,
                                             vector<unique_ptr<Expression>> expressions,
                                             vector<unique_ptr<Expression>> groups_p,
                                             vector<GroupingSet> grouping_sets_p,
                                             vector<unsafe_vector<idx_t>> grouping_functions_p,
                                             idx_t estimated_cardinality)
    : GPUPhysicalOperator(PhysicalOperatorType::HASH_GROUP_BY, std::move(types), estimated_cardinality),
      grouping_sets(std::move(grouping_sets_p)) {
}

} // namespace duckdb