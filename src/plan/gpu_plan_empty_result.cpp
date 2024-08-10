#include "duckdb/execution/operator/scan/physical_empty_result.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_empty_result.hpp"

#include "gpu_physical_empty_result.hpp"
#include "gpu_physical_plan_generator.hpp"

namespace duckdb {

unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::CreatePlan(LogicalEmptyResult &op) {
	D_ASSERT(op.children.size() == 0);
	return make_uniq<GPUPhysicalEmptyResult>(op.types, op.estimated_cardinality);
}

} // namespace duckdb
