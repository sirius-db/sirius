#include "gpu_physical_plan_generator.hpp"

#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/query_profiler.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator/list.hpp"
#include "duckdb/execution/operator/helper/physical_verify_vector.hpp"

namespace duckdb {

// class DependencyExtractor : public LogicalOperatorVisitor {
// public:
// 	explicit DependencyExtractor(LogicalDependencyList &dependencies) : dependencies(dependencies) {
// 	}

// protected:
// 	unique_ptr<Expression> VisitReplace(BoundFunctionExpression &expr, unique_ptr<Expression> *expr_ptr) override {
// 		// extract dependencies from the bound function expression
// 		if (expr.function.dependency) {
// 			expr.function.dependency(expr, dependencies);
// 		}
// 		return nullptr;
// 	}

// private:
// 	LogicalDependencyList &dependencies;
// };

GPUPhysicalPlanGenerator::GPUPhysicalPlanGenerator(ClientContext &context, GPUContext& gpu_context) : 
	context(context), gpu_context(gpu_context) {
}

GPUPhysicalPlanGenerator::~GPUPhysicalPlanGenerator() {
}

unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::CreatePlan(unique_ptr<LogicalOperator> op) {
	auto &profiler = QueryProfiler::Get(context);

	// first resolve column references
	profiler.StartPhase(MetricsType::PHYSICAL_PLANNER_COLUMN_BINDING);
	ColumnBindingResolver resolver;
	resolver.VisitOperator(*op);
	profiler.EndPhase();

	// now resolve types of all the operators
	profiler.StartPhase(MetricsType::PHYSICAL_PLANNER_RESOLVE_TYPES);
	op->ResolveOperatorTypes();
	profiler.EndPhase();

	// extract dependencies from the logical plan
	// DependencyExtractor extractor(dependencies);
	// extractor.VisitOperator(*op);

	// then create the main physical plan
	profiler.StartPhase(MetricsType::PHYSICAL_PLANNER_CREATE_PLAN);
	auto plan = CreatePlan(*op);
	profiler.EndPhase();

	plan->Verify();
	return plan;
}

unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::CreatePlan(LogicalOperator &op) {
	op.estimated_cardinality = op.EstimateCardinality(context);
	unique_ptr<GPUPhysicalOperator> plan = nullptr;

	switch (op.type) {
	case LogicalOperatorType::LOGICAL_GET:
		plan = CreatePlan(op.Cast<LogicalGet>());
		break;
	case LogicalOperatorType::LOGICAL_PROJECTION:
		plan = CreatePlan(op.Cast<LogicalProjection>());
		break;
	case LogicalOperatorType::LOGICAL_EMPTY_RESULT:
		plan = CreatePlan(op.Cast<LogicalEmptyResult>());
		break;
	case LogicalOperatorType::LOGICAL_FILTER:
		plan = CreatePlan(op.Cast<LogicalFilter>());
		break;
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		plan = CreatePlan(op.Cast<LogicalAggregate>());
		break;
	case LogicalOperatorType::LOGICAL_WINDOW:
		throw NotImplementedException("Window not supported");
		// plan = CreatePlan(op.Cast<LogicalWindow>());
		break;
	case LogicalOperatorType::LOGICAL_UNNEST:
		throw NotImplementedException("Unnest not supported");
		// plan = CreatePlan(op.Cast<LogicalUnnest>());
		break;
	case LogicalOperatorType::LOGICAL_LIMIT:
		plan = CreatePlan(op.Cast<LogicalLimit>());
		break;
	case LogicalOperatorType::LOGICAL_SAMPLE:
		throw NotImplementedException("Sample not supported");
		// plan = CreatePlan(op.Cast<LogicalSample>());
		break;
	case LogicalOperatorType::LOGICAL_ORDER_BY:
		plan = CreatePlan(op.Cast<LogicalOrder>());
		break;
	case LogicalOperatorType::LOGICAL_TOP_N:
		// throw NotImplementedException("Top N not supported");
		plan = CreatePlan(op.Cast<LogicalTopN>());
		break;
	case LogicalOperatorType::LOGICAL_COPY_TO_FILE:
		throw NotImplementedException("Copy to file not supported");
		// plan = CreatePlan(op.Cast<LogicalCopyToFile>());
		break;
	case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
		plan = CreatePlan(op.Cast<LogicalDummyScan>());
		break;
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
		throw NotImplementedException("Any join not supported");
		// plan = CreatePlan(op.Cast<LogicalAnyJoin>());
		break;
	case LogicalOperatorType::LOGICAL_ASOF_JOIN:
		throw NotImplementedException("Asof join not supported");
		break;
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		plan = CreatePlan(op.Cast<LogicalComparisonJoin>());
		break;
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		throw NotImplementedException("Cross product not supported");
		// plan = CreatePlan(op.Cast<LogicalCrossProduct>());
		break;
	case LogicalOperatorType::LOGICAL_POSITIONAL_JOIN:
		throw NotImplementedException("Positional join not supported");
		// plan = CreatePlan(op.Cast<LogicalPositionalJoin>());
		break;
	case LogicalOperatorType::LOGICAL_UNION:
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_INTERSECT:
		throw NotImplementedException("Set operation not supported");
		// plan = CreatePlan(op.Cast<LogicalSetOperation>());
		break;
	case LogicalOperatorType::LOGICAL_INSERT:
		throw NotImplementedException("Insert not supported");
		// plan = CreatePlan(op.Cast<LogicalInsert>());
		break;
	case LogicalOperatorType::LOGICAL_DELETE:
		throw NotImplementedException("Delete not supported");
		// plan = CreatePlan(op.Cast<LogicalDelete>());
		break;
	case LogicalOperatorType::LOGICAL_CHUNK_GET:
		plan = CreatePlan(op.Cast<LogicalColumnDataGet>());
		break;
	case LogicalOperatorType::LOGICAL_DELIM_GET:
		plan = CreatePlan(op.Cast<LogicalDelimGet>());
		break;
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
		// throw NotImplementedException("Expression get not supported");
		plan = CreatePlan(op.Cast<LogicalExpressionGet>());
		break;
	case LogicalOperatorType::LOGICAL_UPDATE:
		throw NotImplementedException("Update not supported");
		// plan = CreatePlan(op.Cast<LogicalUpdate>());
		break;
	case LogicalOperatorType::LOGICAL_CREATE_TABLE:
		throw NotImplementedException("Create table not supported");
		// plan = CreatePlan(op.Cast<LogicalCreateTable>());
		break;
	case LogicalOperatorType::LOGICAL_CREATE_INDEX:
		throw NotImplementedException("Create index not supported");
		// plan = CreatePlan(op.Cast<LogicalCreateIndex>());
		break;
	case LogicalOperatorType::LOGICAL_CREATE_SECRET:
		throw NotImplementedException("Create secret not supported");
		// plan = CreatePlan(op.Cast<LogicalCreateSecret>());
		break;
	case LogicalOperatorType::LOGICAL_EXPLAIN:
		throw NotImplementedException("Explain not supported");
		// plan = CreatePlan(op.Cast<LogicalExplain>());
		break;
	case LogicalOperatorType::LOGICAL_DISTINCT:
		throw NotImplementedException("Distinct not supported");
		// plan = CreatePlan(op.Cast<LogicalDistinct>());
		break;
	case LogicalOperatorType::LOGICAL_PREPARE:
		throw NotImplementedException("Prepare not supported");
		// plan = CreatePlan(op.Cast<LogicalPrepare>());
		break;
	case LogicalOperatorType::LOGICAL_EXECUTE:
		throw NotImplementedException("Execute not supported");
		// plan = CreatePlan(op.Cast<LogicalExecute>());
		break;
	case LogicalOperatorType::LOGICAL_CREATE_VIEW:
	case LogicalOperatorType::LOGICAL_CREATE_SEQUENCE:
	case LogicalOperatorType::LOGICAL_CREATE_SCHEMA:
	case LogicalOperatorType::LOGICAL_CREATE_MACRO:
	case LogicalOperatorType::LOGICAL_CREATE_TYPE:
		throw NotImplementedException("Create not supported");
		// plan = CreatePlan(op.Cast<LogicalCreate>());
		break;
	case LogicalOperatorType::LOGICAL_PRAGMA:
		throw NotImplementedException("Pragma not supported");
		// plan = CreatePlan(op.Cast<LogicalPragma>());
		break;
	case LogicalOperatorType::LOGICAL_VACUUM:
		throw NotImplementedException("Vacuum not supported");
		// plan = CreatePlan(op.Cast<LogicalVacuum>());
		break;
	case LogicalOperatorType::LOGICAL_TRANSACTION:
	case LogicalOperatorType::LOGICAL_ALTER:
	case LogicalOperatorType::LOGICAL_DROP:
	case LogicalOperatorType::LOGICAL_LOAD:
	case LogicalOperatorType::LOGICAL_ATTACH:
	case LogicalOperatorType::LOGICAL_DETACH:
		throw NotImplementedException("Simple not supported");
		// plan = CreatePlan(op.Cast<LogicalSimple>());
		break;
	case LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
		throw NotImplementedException("Recursive CTE not supported");
		// plan = CreatePlan(op.Cast<LogicalRecursiveCTE>());
		break;
	case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
		plan = CreatePlan(op.Cast<LogicalMaterializedCTE>());
		break;
	case LogicalOperatorType::LOGICAL_CTE_REF:
		plan = CreatePlan(op.Cast<LogicalCTERef>());
		break;
	case LogicalOperatorType::LOGICAL_EXPORT:
		throw NotImplementedException("Export not supported");
		// plan = CreatePlan(op.Cast<LogicalExport>());
		break;
	case LogicalOperatorType::LOGICAL_SET:
		throw NotImplementedException("Set not supported");
		// plan = CreatePlan(op.Cast<LogicalSet>());
		break;
	case LogicalOperatorType::LOGICAL_RESET:
		throw NotImplementedException("Reset not supported");
		// plan = CreatePlan(op.Cast<LogicalReset>());
		break;
	case LogicalOperatorType::LOGICAL_PIVOT:
		throw NotImplementedException("Pivot not supported");
		// plan = CreatePlan(op.Cast<LogicalPivot>());
		break;
	case LogicalOperatorType::LOGICAL_COPY_DATABASE:
		throw NotImplementedException("Copy database not supported");
		// plan = CreatePlan(op.Cast<LogicalCopyDatabase>());
		break;
	case LogicalOperatorType::LOGICAL_UPDATE_EXTENSIONS:
		throw NotImplementedException("Update extensions not supported");
		// plan = CreatePlan(op.Cast<LogicalSimple>());
		break;
	case LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
		throw NotImplementedException("Extension operator not supported");
		// plan = op.Cast<LogicalExtensionOperator>().CreatePlan(context, *this);

		// if (!plan) {
		// 	throw InternalException("Missing GPUPhysicalOperator for Extension Operator");
		// }
		break;
	case LogicalOperatorType::LOGICAL_JOIN:
	case LogicalOperatorType::LOGICAL_DEPENDENT_JOIN:
	case LogicalOperatorType::LOGICAL_INVALID: {
		throw NotImplementedException("Unimplemented logical operator type!");
	}
	default:
		throw NotImplementedException("Unimplemented logical operator type");
	}
	if (!plan) {
		throw InternalException("Physical plan generator - no plan generated");
	}

	plan->estimated_cardinality = op.estimated_cardinality;
#ifdef DUCKDB_VERIFY_VECTOR_OPERATOR
	auto verify = make_uniq<PhysicalVerifyVector>(std::move(plan));
	plan = std::move(verify);
#endif

	return plan;
}

} // namespace duckdb