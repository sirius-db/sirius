#include "duckdb/planner/operator/logical_set_operation.hpp"
#include "gpu_physical_plan_generator.hpp"
#include "operator/gpu_physical_union.hpp"

namespace duckdb {
unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::CreatePlan(LogicalSetOperation& op)
{
  if (op.type != LogicalOperatorType::LOGICAL_UNION) {
    throw NotImplementedException("Only UNION ALL supported on GPU");
  }
  auto left     = CreatePlan(*op.children[0]);
  auto right    = CreatePlan(*op.children[1]);
  auto union_op = make_uniq<GPUPhysicalUnion>(op.types, op.estimated_cardinality);
  union_op->children.push_back(std::move(left));
  union_op->children.push_back(std::move(right));
  return std::move(union_op);
}
}  // namespace duckdb
