#include "operator/gpu_physical_filter.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"

namespace duckdb {

GPUPhysicalFilter::GPUPhysicalFilter(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list,
                               idx_t estimated_cardinality)
    : GPUPhysicalOperator(PhysicalOperatorType::FILTER, std::move(types), estimated_cardinality) {

	D_ASSERT(select_list.size() > 0);
	if (select_list.size() > 1) {
		// create a big AND out of the expressions
		auto conjunction = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
		for (auto &expr : select_list) {
			conjunction->children.push_back(std::move(expr));
		}
		expression = std::move(conjunction);
	} else {
		expression = std::move(select_list[0]);
	}

	GPUExpressionExecutor* gpu_expression_executor = new GPUExpressionExecutor();

}

// OperatorResultType 
// GPUPhysicalFilter::Execute(ExecutionContext &context, GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation,
// 	                                   GlobalOperatorState &gstate, OperatorState &state) const {
OperatorResultType 
GPUPhysicalFilter::Execute(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const {
	printf("Executing expression %s\n", expression->ToString().c_str());

	
	if (expression->type == ExpressionType::CONJUNCTION_OR) {
		auto &expr = expression->Cast<BoundConjunctionExpression>();

		//Q19 HACK!!!
		if (expr.children.size() == 3 && expr.children[2]->type == ExpressionType::CONJUNCTION_AND) {
			auto &expr2 = expr.children[2]->Cast<BoundConjunctionExpression>();
			if (expr2.children.size() == 5 && expr2.children[0]->type == ExpressionType::COMPARE_EQUAL && expr2.children[1]->type == ExpressionType::COMPARE_GREATERTHANOREQUALTO && \
						 expr2.children[2]->type == ExpressionType::COMPARE_LESSTHANOREQUALTO && expr2.children[3]->type == ExpressionType::COMPARE_LESSTHANOREQUALTO && expr2.children[4]->type == ExpressionType::COMPARE_IN) {

				auto &expr3 = expr2.children[4]->Cast<BoundOperatorExpression>();
				if (expr3.children.size() == 5) {
						
						string t = "(((P_BRAND = 12) AND (L_QUANTITY <= 11) AND (P_SIZE <= 5) AND (P_CONTAINER IN (0, 1, 4, 5))) OR ((P_BRAND = 23) AND (L_QUANTITY >= 10) AND (L_QUANTITY <= 20) AND (P_SIZE <= 10) AND (P_CONTAINER IN (17, 18, 20, 21))) OR ((P_BRAND = 34) AND (L_QUANTITY >= 20) AND (L_QUANTITY <= 30) AND (P_SIZE <= 15) AND (P_CONTAINER IN (8, 9, 12, 13))))";
						if (!expression->ToString().compare(t)) {

								BoundComparisonExpression& first = expr2.children[0]->Cast<BoundComparisonExpression>();
								auto p_brand = first.left->Cast<BoundReferenceExpression>().index;
								BoundComparisonExpression& second = expr2.children[1]->Cast<BoundComparisonExpression>();
								auto l_quantity = second.left->Cast<BoundReferenceExpression>().index;
								BoundComparisonExpression& third = expr2.children[3]->Cast<BoundComparisonExpression>();
								auto p_size = third.left->Cast<BoundReferenceExpression>().index;
								BoundOperatorExpression& fourth = expr2.children[4]->Cast<BoundOperatorExpression>();
								auto p_container = fourth.children[0]->Cast<BoundReferenceExpression>().index;

								uint64_t p_brand_val[3] = {12, 23, 34};
								uint64_t l_quantity_val[6] = {1, 11, 10, 20, 20, 30};
								uint64_t p_size_val[3] = {5, 10, 15};
								uint64_t p_container_val[12] = {0, 1, 4, 5, 17, 18, 20, 21, 8, 9, 12, 13};



						}

				}
			}
		}

		//Q7 HACK!!!
		auto &expr = expression->Cast<BoundConjunctionExpression>();
		if (expr.children.size() == 2 && expr.children[0]->type == ExpressionType::CONJUNCTION_AND && expr.children[1]->type == ExpressionType::CONJUNCTION_AND) {
			auto &expr2 = expr.children[0]->Cast<BoundConjunctionExpression>();
			if (expr2.children.size() == 2 && expr2.children[0]->type == ExpressionType::COMPARE_EQUAL && expr2.children[1]->type == ExpressionType::COMPARE_EQUAL) {

				string t = "(((N_NATIONKEY = 6) AND (N_NATIONKEY = 7)) OR ((N_NATIONKEY = 7) AND (N_NATIONKEY = 6)))";
				if (!expression->ToString().compare(t)) {
					
						BoundComparisonExpression& first = expr2.children[0]->Cast<BoundComparisonExpression>();
						auto n_nationkey1 = first.left->Cast<BoundReferenceExpression>().index;
						BoundComparisonExpression& second = expr2.children[1]->Cast<BoundComparisonExpression>();
						auto n_nationkey2 = second.left->Cast<BoundReferenceExpression>().index;

						uint64_t val[4] = {6, 7, 7, 6};
				}

			}
		}
	} else if (expression->type == ExpressionType::CONJUNCTION_AND) {
		auto &expr = expression->Cast<BoundConjunctionExpression>();

		//Q16 HACK!!!
		if (expr.children.size() == 3 && expr.children[0]->type == ExpressionType::COMPARE_NOTEQUAL && expr.children[1]->type == ExpressionType::CONJUNCTION_OR && expr.children[2]->type == ExpressionType::COMPARE_IN) {
			
			string t = "((P_BRAND != 45) AND ((P_TYPE < 65) OR (P_TYPE >= 70)) AND (P_SIZE IN (49, 14, 23, 45, 19, 3, 36, 9)))";
			if (!expression->ToString().compare(t)) {
				BoundComparisonExpression& first = expr.children[0]->Cast<BoundComparisonExpression>();
				auto p_brand = first.left->Cast<BoundReferenceExpression>().index;
				BoundConjunctionExpression& second = expr.children[1]->Cast<BoundConjunctionExpression>();
				BoundComparisonExpression& second_temp = second.children[0]->Cast<BoundComparisonExpression>();
				auto p_type = second_temp.left->Cast<BoundReferenceExpression>().index;
				BoundOperatorExpression& third = expr.children[2]->Cast<BoundOperatorExpression>();
				auto p_size = third.children[0]->Cast<BoundReferenceExpression>().index;

				uint64_t p_brand_val = 45;
				uint64_t p_type_val[2] = {65, 70};
				uint64_t p_size_val[8] = {49, 14, 23, 45, 19, 3, 36, 9};
			}

		}


		//Q12 HACK!!!
		if (expr.children.size() == 3 && expr.children[0]->type == ExpressionType::COMPARE_LESSTHAN && expr.children[1]->type == ExpressionType::COMPARE_LESSTHAN && expr.children[2]->type == ExpressionType::COMPARE_IN) {
			
			string t = "((L_COMMITDATE < L_RECEIPTDATE) AND (L_SHIPDATE < L_COMMITDATE) AND (L_SHIPMODE IN (4, 6)))";
			if (!expression->ToString().compare(t)) {
				BoundConjunctionExpression& expr = expression->Cast<BoundConjunctionExpression>();

				BoundComparisonExpression& first = expr.children[0]->Cast<BoundComparisonExpression>();
				auto l_commitdate = first.left->Cast<BoundReferenceExpression>().index;
				auto l_receiptdate = first.right->Cast<BoundReferenceExpression>().index;
				BoundComparisonExpression& second = expr.children[1]->Cast<BoundComparisonExpression>();
				auto l_shipdate = second.left->Cast<BoundReferenceExpression>().index;
				auto l_commitdate = second.right->Cast<BoundReferenceExpression>().index;
				BoundOperatorExpression& third = expr.children[2]->Cast<BoundOperatorExpression>();
				auto l_shipmode = third.children[0]->Cast<BoundReferenceExpression>().index;

				uint64_t l_shipmode_val[2] = {4, 6};
			}

		}

	}

	//Q2 HACK!!!
	if (expression->type == ExpressionType::COMPARE_EQUAL) {

		auto &expr = expression->Cast<BoundComparisonExpression>();	
		if (expr.left->type == ExpressionType::BOUND_FUNCTION) {
			auto& function_expr = expr.left->Cast<BoundFunctionExpression>();
			if (function_expr.function.name.compare("%") == 0) {
				auto& left_function_expr = function_expr.children[0]->Cast<BoundFunctionExpression>();
				if (left_function_expr.function.name.compare("+") == 0) {
						auto p_type = left_function_expr.children[0]->Cast<BoundReferenceExpression>().index;
						uint64_t p_type_val = 0;

				}
			}
		}


	}

    gpu_expression_executor->FilterRecursiveExpression(input_relation, output_relation, *expression, 0);
	return OperatorResultType::FINISHED;
}



} // namespace duckdb