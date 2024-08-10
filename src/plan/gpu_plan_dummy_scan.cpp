#include "duckdb/execution/operator/scan/physical_dummy_scan.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_dummy_scan.hpp"

#include "gpu_physical_dummy_scan.hpp"
#include "gpu_physical_plan_generator.hpp"

namespace duckdb {

unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::CreatePlan(LogicalDummyScan &op) {
	D_ASSERT(op.children.size() == 0);
	return make_uniq<GPUPhysicalDummyScan>(op.types, op.estimated_cardinality);
}

} // namespace duckdb
