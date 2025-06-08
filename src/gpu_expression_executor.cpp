#include "gpu_expression_executor.hpp"
#include "operator/gpu_materialize.hpp"
#include "operator/gpu_physical_strings_matching.hpp"
#include "operator/gpu_physical_substring.hpp"
#include "log/logging.hpp"

namespace duckdb {

template <typename T>
shared_ptr<GPUColumn>
GPUExpressionExecutor::ResolveTypeBinaryConstantExpression(shared_ptr<GPUColumn> column, T constant, GPUBufferManager* gpuBufferManager, string function_name) {
    T* a = reinterpret_cast<T*> (column->data_wrapper.data);
    size_t size = column->column_length;
    T* c = gpuBufferManager->customCudaMalloc<T>(size, 0, 0);
    if (function_name.compare("+") == 0) {
        binaryConstantExpression<T>(a, constant, c, size, 0);
    } else if (function_name.compare("-") == 0) {
        binaryConstantExpression<T>(a, constant, c, size, 1);
    } else if (function_name.compare("*") == 0) {
        binaryConstantExpression<T>(a, constant, c, size, 2);
    } else if (function_name.compare("/") == 0) {
        binaryConstantExpression<T>(a, constant, c, size, 3);
    } else if (function_name.compare("//") == 0) {
        binaryConstantExpression<T>(a, constant, c, size, 3);
    } else {
        throw NotImplementedException("Function name not supported");
    }
    shared_ptr<GPUColumn> result = make_shared_ptr<GPUColumn>(size, column->data_wrapper.type, reinterpret_cast<uint8_t*>(c));
    return result;
}

shared_ptr<GPUColumn>
GPUExpressionExecutor::HandleBinaryConstantExpression(shared_ptr<GPUColumn> column, BoundConstantExpression& expr, GPUBufferManager* gpuBufferManager, string function_name) {
    switch(column->data_wrapper.type.id()) {
      case GPUColumnTypeId::INT32: {
        int constant = expr.value.GetValue<int>();
        return ResolveTypeBinaryConstantExpression<int>(column, constant, gpuBufferManager, function_name);
      } case GPUColumnTypeId::INT64: {
        uint64_t constant = expr.value.GetValue<uint64_t>();
        return ResolveTypeBinaryConstantExpression<uint64_t>(column, constant, gpuBufferManager, function_name);
      } case GPUColumnTypeId::FLOAT32: {
        float constant = expr.value.GetValue<float>();
        return ResolveTypeBinaryConstantExpression<float>(column, constant, gpuBufferManager, function_name);
      } case GPUColumnTypeId::FLOAT64: {
        double constant = expr.value.GetValue<double>();
        return ResolveTypeBinaryConstantExpression<double>(column, constant, gpuBufferManager, function_name);
      } default:
        throw NotImplementedException("Unsupported sirius column type in `HandleBinaryConstantExpression`: %d",
                                      static_cast<int>(column->data_wrapper.type.id()));
    }
}

template <typename T>
shared_ptr<GPUColumn>
GPUExpressionExecutor::ResolveTypeBinaryExpression(shared_ptr<GPUColumn> column1, shared_ptr<GPUColumn> column2, GPUBufferManager* gpuBufferManager, string function_name) {
    T* a = reinterpret_cast<T*> (column1->data_wrapper.data);
    T* b = reinterpret_cast<T*> (column2->data_wrapper.data);
    size_t size = column1->column_length;
    T* c = gpuBufferManager->customCudaMalloc<T>(size, 0, 0);
    if (function_name.compare("+") == 0) {
        binaryExpression<T>(a, b, c, size, 0);
    } else if (function_name.compare("-") == 0) {
        binaryExpression<T>(a, b, c, size, 1);
    } else if (function_name.compare("*") == 0) {
        binaryExpression<T>(a, b, c, size, 2);
    } else if (function_name.compare("/") == 0) {
        binaryExpression<T>(a, b, c, size, 3);
    } else {
        throw NotImplementedException("Function name not supported");
    }
    shared_ptr<GPUColumn> result = make_shared_ptr<GPUColumn>(size, column1->data_wrapper.type, reinterpret_cast<uint8_t*>(c));
    return result;
}

shared_ptr<GPUColumn>
GPUExpressionExecutor::HandleBinaryExpression(shared_ptr<GPUColumn> column1, shared_ptr<GPUColumn> column2, GPUBufferManager* gpuBufferManager, string function_name) {
    switch(column1->data_wrapper.type.id()) {
      case GPUColumnTypeId::INT32:
        return ResolveTypeBinaryExpression<int>(column1, column2, gpuBufferManager, function_name);
      case GPUColumnTypeId::INT64:
        return ResolveTypeBinaryExpression<uint64_t>(column1, column2, gpuBufferManager, function_name);
      case GPUColumnTypeId::FLOAT32:
        return ResolveTypeBinaryExpression<float>(column1, column2, gpuBufferManager, function_name);
      case GPUColumnTypeId::FLOAT64:
        return ResolveTypeBinaryExpression<double>(column1, column2, gpuBufferManager, function_name);
      default:
        throw NotImplementedException("Unsupported sirius column type in `HandleBinaryExpression`: %d",
                                      static_cast<int>(column1->data_wrapper.type.id()));
    }
}

template <typename T>
void 
GPUExpressionExecutor::ResolveTypeComparisonConstantExpression (shared_ptr<GPUColumn> column, BoundConstantExpression& expr1, BoundConstantExpression& expr2, uint64_t* &count, uint64_t* & row_ids, ExpressionType expression_type) {
    T* a = reinterpret_cast<T*> (column->data_wrapper.data);
    size_t size = column->column_length;
    T b = expr1.value.GetValue<T>();
    T c;
    switch (expression_type) {
      case ExpressionType::COMPARE_EQUAL:
        comparisonConstantExpression<T>(a, b, c, row_ids, count, size, 0);
        break;
      case ExpressionType::COMPARE_NOTEQUAL:
        comparisonConstantExpression<T>(a, b, c, row_ids, count, size, 1);
        break;
      case ExpressionType::COMPARE_GREATERTHAN:
        comparisonConstantExpression<T>(a, b, c, row_ids, count, size, 2);
        break;
      case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
        comparisonConstantExpression<T>(a, b, c, row_ids, count, size, 3);
        break;
      case ExpressionType::COMPARE_LESSTHAN:
        comparisonConstantExpression<T>(a, b, c, row_ids, count, size, 4);
        break;
      case ExpressionType::COMPARE_LESSTHANOREQUALTO:
        comparisonConstantExpression<T>(a, b, c, row_ids, count, size, 5);
        break;
      case ExpressionType::COMPARE_BETWEEN:
        c = expr2.value.GetValue<T>();
        comparisonConstantExpression<T>(a, b, c, row_ids, count, size, 6);
        break;
      case ExpressionType::COMPARE_NOT_BETWEEN:
        c = expr2.value.GetValue<T>();
        comparisonConstantExpression<T>(a, b, c, row_ids, count, size, 7);
        break;
      default:
        throw NotImplementedException("Comparison type not supported");
    }
}

void 
GPUExpressionExecutor::HandleComparisonConstantExpression(shared_ptr<GPUColumn> column, BoundConstantExpression& expr1, BoundConstantExpression& expr2, uint64_t* &count, uint64_t* &row_ids, ExpressionType expression_type) {
    switch(column->data_wrapper.type.id()) {
      case GPUColumnTypeId::INT32:
        ResolveTypeComparisonConstantExpression<int>(column, expr1, expr2, count, row_ids, expression_type);
        break;
      case GPUColumnTypeId::INT64:
        ResolveTypeComparisonConstantExpression<uint64_t>(column, expr1, expr2, count, row_ids, expression_type);
        break;
      case GPUColumnTypeId::FLOAT32:
        ResolveTypeComparisonConstantExpression<float>(column, expr1, expr2, count, row_ids, expression_type);
        break;
      case GPUColumnTypeId::FLOAT64:
        ResolveTypeComparisonConstantExpression<double>(column, expr1, expr2, count, row_ids, expression_type);
        break;
      case GPUColumnTypeId::BOOLEAN:
        ResolveTypeComparisonConstantExpression<uint8_t>(column, expr1, expr2, count, row_ids, expression_type);
        break;
      default:
        throw NotImplementedException("Unsupported sirius column type in `HandleComparisonConstantExpression`: %d",
                                      static_cast<int>(column->data_wrapper.type.id()));
    }
}

template <typename T>
void 
GPUExpressionExecutor::ResolveTypeComparisonExpression (shared_ptr<GPUColumn> column1, shared_ptr<GPUColumn> column2, uint64_t* &count, uint64_t* & row_ids, ExpressionType expression_type) {
    T* a = reinterpret_cast<T*> (column1->data_wrapper.data);
    T* b = reinterpret_cast<T*> (column2->data_wrapper.data);
    size_t size = column1->column_length;
    switch (expression_type) {
      case ExpressionType::COMPARE_EQUAL:
        comparisonExpression<T>(a, b, row_ids, count, size, 0);
        break;
      case ExpressionType::COMPARE_NOTEQUAL:
        comparisonExpression<T>(a, b, row_ids, count, size, 1);
        break;
      case ExpressionType::COMPARE_GREATERTHAN:
        comparisonExpression<T>(a, b, row_ids, count, size, 2);
        break;
      case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
        comparisonExpression<T>(a, b, row_ids, count, size, 3);
        break;
      case ExpressionType::COMPARE_LESSTHAN:
        comparisonExpression<T>(a, b, row_ids, count, size, 4);
        break;
      case ExpressionType::COMPARE_LESSTHANOREQUALTO:
        comparisonExpression<T>(a, b, row_ids, count, size, 5);
        break;
      default:
        throw NotImplementedException("Comparison type not supported");
    }
}

void 
GPUExpressionExecutor::HandleComparisonExpression(shared_ptr<GPUColumn> column1, shared_ptr<GPUColumn> column2, uint64_t* &count, uint64_t* &row_ids, ExpressionType expression_type) {
    switch(column1->data_wrapper.type.id()) {
      case GPUColumnTypeId::INT32:
        ResolveTypeComparisonExpression<int>(column1, column2, count, row_ids, expression_type);
        break;
      case GPUColumnTypeId::INT64:
        ResolveTypeComparisonExpression<uint64_t>(column1, column2, count, row_ids, expression_type);
        break;
      case GPUColumnTypeId::FLOAT32:
        ResolveTypeComparisonExpression<float>(column1, column2, count, row_ids, expression_type);
        break;
      case GPUColumnTypeId::FLOAT64:
        ResolveTypeComparisonExpression<double>(column1, column2, count, row_ids, expression_type);
        break;
      default:
        throw NotImplementedException("Unsupported sirius column type in `HandleComparisonExpression`: %d",
                                      static_cast<int>(column1->data_wrapper.type.id()));
    }
}

shared_ptr<GPUColumn>
GPUExpressionExecutor::HandleRoundExpression(shared_ptr<GPUColumn> column, int decimal_places) {
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    switch(column->data_wrapper.type.id()) {
      case GPUColumnTypeId::FLOAT32: {
        float* a = reinterpret_cast<float*> (column->data_wrapper.data);
        size_t size = column->column_length;
        float* c = gpuBufferManager->customCudaMalloc<float>(size, 0, 0);
        floatRoundExpression(a, c, decimal_places, size);
        shared_ptr<GPUColumn> result = make_shared_ptr<GPUColumn>(size, column->data_wrapper.type, reinterpret_cast<uint8_t*>(c));
        return result;
      } case GPUColumnTypeId::FLOAT64: {
        double* a = reinterpret_cast<double*> (column->data_wrapper.data);
        size_t size = column->column_length;
        double* c = gpuBufferManager->customCudaMalloc<double>(size, 0, 0);
        doubleRoundExpression(a, c, decimal_places, size);
        shared_ptr<GPUColumn> result = make_shared_ptr<GPUColumn>(size, column->data_wrapper.type, reinterpret_cast<uint8_t*>(c));
        return result;
      } default:
        throw NotImplementedException("Unsupported sirius column type in `HandleRoundExpression`: %d",
                                      static_cast<int>(column->data_wrapper.type.id()));
    }
}

void 
GPUExpressionExecutor::FilterRecursiveExpression(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, Expression& expr, int depth) {
    
    bool is_specific_filter = HandlingSpecificFilter(input_relation, output_relation, expr);
    if (is_specific_filter) return;
    
    SIRIUS_LOG_DEBUG("Expression class {}", ExpressionClassToString(expr.expression_class));
    uint64_t* comparison_idx = nullptr;
    uint64_t* count = nullptr;

    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

	  switch (expr.expression_class) {
          case ExpressionClass::BOUND_BETWEEN: {
                auto &bound_between = expr.Cast<BoundBetweenExpression>();
                SIRIUS_LOG_DEBUG("Executing between expression");
                // FilterRecursiveExpression(input_relation, output_relation, *(bound_between.input), depth + 1);
                // FilterRecursiveExpression(input_relation, output_relation, *(bound_between.lower), depth + 1);
                // FilterRecursiveExpression(input_relation, output_relation, *(bound_between.upper), depth + 1);
                auto &bound_ref = bound_between.input->Cast<BoundReferenceExpression>();
                auto &bound_lower = bound_between.lower->Cast<BoundConstantExpression>();
                auto &bound_upper = bound_between.upper->Cast<BoundConstantExpression>();
                size_t size;

                shared_ptr<GPUColumn> materialized_column = HandleMaterializeExpression(input_relation.columns[bound_ref.index], bound_ref, gpuBufferManager);
                // count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
                HandleComparisonConstantExpression(materialized_column, bound_lower, bound_upper, count, comparison_idx, bound_between.type);
                // if (count[0] == 0) throw NotImplementedException("No match found");
            break;
            } case ExpressionClass::BOUND_REF: {
                auto &bound_ref = expr.Cast<BoundReferenceExpression>();
                SIRIUS_LOG_DEBUG("Executing bound reference expression");
                SIRIUS_LOG_DEBUG("Reading column index {}", bound_ref.index);
                input_relation.checkLateMaterialization(bound_ref.index);
                // SIRIUS_LOG_DEBUG("output_relation.columns.size() {}", output_relation.columns.size());
                // SIRIUS_LOG_DEBUG("input_relation.columns.size() {}", input_relation.columns.size());
                // SIRIUS_LOG_DEBUG("bound_ref.index {}", bound_ref.index);
                output_relation.columns[bound_ref.index] = input_relation.columns[bound_ref.index];
                output_relation.columns[bound_ref.index]->row_ids = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
                SIRIUS_LOG_DEBUG("Overwrite row ids with column {}", bound_ref.index);
            break;
          } case ExpressionClass::BOUND_CASE: {
                throw NotImplementedException("Case expression not supported");
                auto &bound_case = expr.Cast<BoundCaseExpression>();
                SIRIUS_LOG_DEBUG("Executing case expression");
                for (idx_t i = 0; i < bound_case.case_checks.size(); i++) {
                    FilterRecursiveExpression(input_relation, output_relation, *(bound_case.case_checks[i].when_expr), depth + 1);
                    FilterRecursiveExpression(input_relation, output_relation, *(bound_case.case_checks[i].then_expr), depth + 1);
                }
                FilterRecursiveExpression(input_relation, output_relation, *(bound_case.else_expr), depth + 1);
            break;
          } case ExpressionClass::BOUND_CAST: {
                throw NotImplementedException("Cast expression not supported");
                auto &bound_cast = expr.Cast<BoundCastExpression>();
                SIRIUS_LOG_DEBUG("Executing cast expression");
                FilterRecursiveExpression(input_relation, output_relation, *(bound_cast.child), depth + 1);
            break;
          } case ExpressionClass::BOUND_COMPARISON: {
                auto &bound_comparison = expr.Cast<BoundComparisonExpression>();
                SIRIUS_LOG_DEBUG("Executing comparison expression");
                // FilterRecursiveExpression(input_relation, output_relation, *(bound_comparison.left), depth + 1);
                // FilterRecursiveExpression(input_relation, output_relation, *(bound_comparison.right), depth + 1);

                if (bound_comparison.right->expression_class == ExpressionClass::BOUND_CONSTANT) {
                    auto &bound_ref1 = bound_comparison.left->Cast<BoundReferenceExpression>();
                    auto &bound_ref2 = bound_comparison.right->Cast<BoundConstantExpression>();
                    size_t size;

                    if (input_relation.columns[bound_ref1.index]->data_wrapper.data == nullptr) {
                        SIRIUS_LOG_DEBUG("Column is null");
                        count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
                        count[0] = 0;
                    } else {
                        shared_ptr<GPUColumn> materialized_column = HandleMaterializeExpression(input_relation.columns[bound_ref1.index], bound_ref1, gpuBufferManager);
                        // count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
                        HandleComparisonConstantExpression(materialized_column, bound_ref2, bound_ref2, count, comparison_idx, bound_comparison.type);
                        // if (count[0] == 0) throw NotImplementedException("No match found");
                    }
                } else if (bound_comparison.right->expression_class == ExpressionClass::BOUND_REF) {
                    auto &bound_ref1 = bound_comparison.left->Cast<BoundReferenceExpression>();
                    auto &bound_ref2 = bound_comparison.right->Cast<BoundReferenceExpression>();
                    size_t size;

                    if (input_relation.columns[bound_ref1.index]->data_wrapper.data == nullptr || input_relation.columns[bound_ref2.index]->data_wrapper.data == nullptr) {
                        SIRIUS_LOG_DEBUG("Column is null");
                        count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
                        count[0] = 0;
                    } else {
                        shared_ptr<GPUColumn> materialized_column1 = HandleMaterializeExpression(input_relation.columns[bound_ref1.index], bound_ref1, gpuBufferManager);
                        shared_ptr<GPUColumn> materialized_column2 = HandleMaterializeExpression(input_relation.columns[bound_ref2.index], bound_ref2, gpuBufferManager);
                        // count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
                        HandleComparisonExpression(materialized_column1, materialized_column2, count, comparison_idx, bound_comparison.type);  
                        // if (count[0] == 0) throw NotImplementedException("No match found");   
                    }       
                } else {
                  throw NotImplementedException("Comparison expression not supported");
                }
            break;
          } case ExpressionClass::BOUND_CONJUNCTION: {
                throw NotImplementedException("Conjunction expression not supported");
                auto &bound_conjunction = expr.Cast<BoundConjunctionExpression>();
                SIRIUS_LOG_DEBUG("Executing conjunction expression");
                for (auto &child : bound_conjunction.children) {
                    FilterRecursiveExpression(input_relation, output_relation, *child, depth + 1);
                }
            break;
          } case ExpressionClass::BOUND_CONSTANT: {
                SIRIUS_LOG_DEBUG("Reading value {}", expr.Cast<BoundConstantExpression>().value.ToString());
            break;
          } case ExpressionClass::BOUND_FUNCTION: {
                SIRIUS_LOG_DEBUG("Executing function expression");
                auto &bound_function = expr.Cast<BoundFunctionExpression>();
                // for (auto &child : bound_function.children) {
                //     FilterRecursiveExpression(input_relation, output_relation, *child, depth + 1);
                // }
                std::string bound_function_name = bound_function.ToString();
                if (bound_function.children[0]->type != ExpressionType::BOUND_REF) {
                  throw NotImplementedException("Contains function not supported");
                }
                SIRIUS_LOG_DEBUG("Function name {}", bound_function_name);
                auto& bound_ref = bound_function.children[0]->Cast<BoundReferenceExpression>();
                auto& bound_const = bound_function.children[1]->Cast<BoundConstantExpression>();
                std::string match_str = bound_const.value.ToString();
                if (input_relation.columns[bound_ref.index]->data_wrapper.data == nullptr) {
                    SIRIUS_LOG_DEBUG("Column is null");
                    count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
                    count[0] = 0;
                } else {
                  // count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
                  shared_ptr<GPUColumn> materialized_column = HandleMaterializeExpression(input_relation.columns[bound_ref.index], bound_ref, gpuBufferManager);
                  if(bound_function_name.find("prefix") != std::string::npos) {
                    HandlePrefixMatching(materialized_column, match_str, comparison_idx, count, 0);
                  } else if(bound_function_name.find("contains") != std::string::npos) {
                    HandleStringMatching(materialized_column, match_str, comparison_idx, count, 0);
                  } else if(bound_function_name.find("!~~") != std::string::npos) {
                    HandleMultiStringMatching(materialized_column, match_str, comparison_idx, count, 1);
                  } else if(bound_function_name.find("~~") != std::string::npos) {
                    HandleMultiStringMatching(materialized_column, match_str, comparison_idx, count, 0);
                  } else {
                    throw NotImplementedException("Function not supported");
                  }
                  // if (count[0] == 0) throw NotImplementedException("No match found");
                }
            break;
          } case ExpressionClass::BOUND_OPERATOR: {
                SIRIUS_LOG_DEBUG("Executing IN or NOT expression");
                auto &bound_operator = expr.Cast<BoundOperatorExpression>();
                // for (auto &child : bound_operator.children) {
                //     FilterRecursiveExpression(input_relation, output_relation, *child, depth + 1);
                // }
                if (bound_operator.type == ExpressionType::OPERATOR_NOT) {
                    SIRIUS_LOG_DEBUG("Executing NOT expression");
                    // if children is a bound reference
                    if (bound_operator.children[0]->expression_class == ExpressionClass::BOUND_REF) {
                      auto &bound_ref = bound_operator.children[0]->Cast<BoundReferenceExpression>();
                      if (input_relation.columns[bound_ref.index]->data_wrapper.data == nullptr) {
                          SIRIUS_LOG_DEBUG("Column is null");
                          count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
                          count[0] = 0;
                      } else {
                        shared_ptr<GPUColumn> materialized_column = HandleMaterializeExpression(input_relation.columns[bound_ref.index], bound_ref, gpuBufferManager);
                        Value one = Value::BOOLEAN(1);
                        BoundConstantExpression bound_constant_expr = BoundConstantExpression(one);
                        // count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
                        HandleComparisonConstantExpression(materialized_column, bound_constant_expr, bound_constant_expr, count, comparison_idx, ExpressionType::COMPARE_NOTEQUAL);
                        // if (count[0] == 0) throw NotImplementedException("No match found"); 
                      }
                    // if children is a bound function (contains or prefix)
                    } else if (bound_operator.children[0]->expression_class == ExpressionClass::BOUND_FUNCTION) {
                      auto& bound_function = bound_operator.children[0]->Cast<BoundFunctionExpression>();
                      std::string bound_function_name = bound_function.ToString();
                      auto& bound_ref = bound_function.children[0]->Cast<BoundReferenceExpression>();
                      auto& bound_const = bound_function.children[1]->Cast<BoundConstantExpression>();
                      std::string match_str = bound_const.value.ToString();
                      if (input_relation.columns[bound_ref.index]->data_wrapper.data == nullptr) {
                          SIRIUS_LOG_DEBUG("Column is null");
                          count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
                          count[0] = 0;
                      } else {
                        // count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
                        shared_ptr<GPUColumn> materialized_column = HandleMaterializeExpression(input_relation.columns[bound_ref.index], bound_ref, gpuBufferManager);
                        if(bound_function_name.find("prefix") != std::string::npos) {
                            HandlePrefixMatching(materialized_column, match_str, comparison_idx, count, 1);
                        } else if(bound_function_name.find("contains") != std::string::npos) {
                            HandleStringMatching(materialized_column, match_str, comparison_idx, count, 1);
                        } 
                        // if (count[0] == 0) throw NotImplementedException("No match found");
                      }
                    }
                } else {
                  throw NotImplementedException("Other BOUND_OPERATOR expression not supported");
                }
            break;
          } case ExpressionClass::BOUND_PARAMETER: {
            throw NotImplementedException("Parameter expression is not supported");
            break;
          } default: {
            throw InternalException("Attempting to execute expression of unknown type!");
          }
    }

    if (depth == 0) {
        SIRIUS_LOG_DEBUG("Writing filter result");
        HandleMaterializeRowIDs(input_relation, output_relation, count[0], comparison_idx, gpuBufferManager, true);
    }
}


void 
GPUExpressionExecutor::ProjectionRecursiveExpression(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, Expression& expr, int output_idx, int depth) {
    
    bool is_specific_projection = HandlingSpecificProjection(input_relation, output_relation, expr, output_idx);
    if (is_specific_projection) return;

    shared_ptr<GPUColumn> result;

    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    switch (expr.expression_class) {
        case ExpressionClass::BOUND_BETWEEN: {
              throw NotImplementedException("Between expression not supported");
              auto &bound_between = expr.Cast<BoundBetweenExpression>();
              SIRIUS_LOG_DEBUG("Executing between expression");
              ProjectionRecursiveExpression(input_relation, output_relation, *(bound_between.input), output_idx, depth + 1);
              ProjectionRecursiveExpression(input_relation, output_relation, *(bound_between.lower), output_idx, depth + 1);
              ProjectionRecursiveExpression(input_relation, output_relation, *(bound_between.upper), output_idx, depth + 1);
              break;
        } case ExpressionClass::BOUND_REF: {
              auto &bound_ref = expr.Cast<BoundReferenceExpression>();
              SIRIUS_LOG_DEBUG("Executing bound reference expression");
              SIRIUS_LOG_DEBUG("Reading column index {}", bound_ref.index);
              input_relation.checkLateMaterialization(bound_ref.index);
              if (depth == 0) {
                projected_columns.push_back(bound_ref.index);
              }
              break;
        } case ExpressionClass::BOUND_CASE: {
              throw NotImplementedException("Case expression not supported");
              auto &bound_case = expr.Cast<BoundCaseExpression>();
              SIRIUS_LOG_DEBUG("Executing case expression");
              for (idx_t i = 0; i < bound_case.case_checks.size(); i++) {
                  ProjectionRecursiveExpression(input_relation, output_relation, *(bound_case.case_checks[i].when_expr), output_idx, depth + 1);
                  ProjectionRecursiveExpression(input_relation, output_relation, *(bound_case.case_checks[i].then_expr), output_idx, depth + 1);
              }
              ProjectionRecursiveExpression(input_relation, output_relation, *(bound_case.else_expr), output_idx, depth + 1);
              break;
        } case ExpressionClass::BOUND_CAST: {
              throw NotImplementedException("Cast expression not supported");
              auto &bound_cast = expr.Cast<BoundCastExpression>();
              SIRIUS_LOG_DEBUG("Executing cast expression");
              ProjectionRecursiveExpression(input_relation, output_relation, *(bound_cast.child), output_idx, depth + 1);
              break;
        } case ExpressionClass::BOUND_COMPARISON: {
              throw NotImplementedException("Comparison expression not supported");
              auto &bound_comparison = expr.Cast<BoundComparisonExpression>();
              SIRIUS_LOG_DEBUG("Executing comparison expression");
              ProjectionRecursiveExpression(input_relation, output_relation, *(bound_comparison.left), output_idx, depth + 1);
              ProjectionRecursiveExpression(input_relation, output_relation, *(bound_comparison.right), output_idx, depth + 1);
              break;
        } case ExpressionClass::BOUND_CONJUNCTION: {
              throw NotImplementedException("Conjunction expression not supported");
              auto &bound_conjunction = expr.Cast<BoundConjunctionExpression>();
              SIRIUS_LOG_DEBUG("Executing conjunction expression");
              for (auto &child : bound_conjunction.children) {
                  ProjectionRecursiveExpression(input_relation, output_relation, *child, output_idx, depth + 1);
              }
              break;
        } case ExpressionClass::BOUND_CONSTANT: {
              SIRIUS_LOG_DEBUG("Reading value {}", expr.Cast<BoundConstantExpression>().value.ToString());
              break;
        } case ExpressionClass::BOUND_FUNCTION: {
              SIRIUS_LOG_DEBUG("Executing function expression");
              auto &bound_function = expr.Cast<BoundFunctionExpression>();
              SIRIUS_LOG_DEBUG("Function name %s\n", bound_function.function.name);
              if((bound_function.ToString().find("substr") != std::string::npos) || (bound_function.ToString().find("substring") != std::string::npos)) {
                  auto &bound_ref1 = bound_function.children[0]->Cast<BoundReferenceExpression>();
                  auto &bound_ref2 = bound_function.children[1]->Cast<BoundConstantExpression>();
                  auto &bound_ref3 = bound_function.children[2]->Cast<BoundConstantExpression>();
                  shared_ptr<GPUColumn> input_column = input_relation.columns[bound_ref1.index];
                  uint64_t start_idx = bound_ref2.value.GetValue<uint64_t>();
                  if (start_idx < 1) throw InvalidInputException("Start index should be greater than 0");
                  uint64_t length = bound_ref3.value.GetValue<uint64_t>();
                  shared_ptr<GPUColumn> materialized_column = HandleMaterializeExpression(input_column, bound_ref1, gpuBufferManager);
                  result = HandleSubString(materialized_column, start_idx, length);
              } else if (bound_function.ToString().find("round") != std::string::npos) {
                  auto &bound_ref = bound_function.children[0]->Cast<BoundReferenceExpression>();
                  auto &bound_const = bound_function.children[1]->Cast<BoundConstantExpression>();
                  int decimal_places = bound_const.value.GetValue<int>();
                  shared_ptr<GPUColumn> materialized_column = HandleMaterializeExpression(input_relation.columns[bound_ref.index], bound_ref, gpuBufferManager);
                  result = HandleRoundExpression(materialized_column, decimal_places);
              } else if (bound_function.children[1]->expression_class == ExpressionClass::BOUND_CONSTANT) {
                  auto &bound_ref1 = bound_function.children[0]->Cast<BoundReferenceExpression>();
                  auto &bound_ref2 = bound_function.children[1]->Cast<BoundConstantExpression>();
                  shared_ptr<GPUColumn> materialized_column = HandleMaterializeExpression(input_relation.columns[bound_ref1.index], bound_ref1, gpuBufferManager);
                  result = HandleBinaryConstantExpression(materialized_column, bound_ref2, gpuBufferManager, bound_function.function.name);
              } else if (bound_function.children[1]->expression_class == ExpressionClass::BOUND_REF) {
                  auto &bound_ref1 = bound_function.children[0]->Cast<BoundReferenceExpression>();
                  auto &bound_ref2 = bound_function.children[1]->Cast<BoundReferenceExpression>();
                  shared_ptr<GPUColumn> materialized_column1 = HandleMaterializeExpression(input_relation.columns[bound_ref1.index], bound_ref1, gpuBufferManager);
                  shared_ptr<GPUColumn> materialized_column2 = HandleMaterializeExpression(input_relation.columns[bound_ref2.index], bound_ref2, gpuBufferManager);
                  result = HandleBinaryExpression(materialized_column1, materialized_column2, gpuBufferManager, bound_function.function.name);            
              } else {
                throw NotImplementedException("Function expression not supported");
              }
              break;
        } case ExpressionClass::BOUND_OPERATOR: {
              throw NotImplementedException("Operator expression is not supported");
              break;
        } case ExpressionClass::BOUND_PARAMETER: {
              throw NotImplementedException("Parameter expression is not supported");
              break;
        } default: {
              throw InternalException("Attempting to execute expression of unknown type!");
        }
    }
    SIRIUS_LOG_DEBUG("Writing projection result to idx {}", output_idx);
    if (depth == 0) {
        if (expr.expression_class == ExpressionClass::BOUND_REF) {
            output_relation.columns[output_idx] = input_relation.columns[expr.Cast<BoundReferenceExpression>().index];
        } else {
            if (result) {
                output_relation.columns[output_idx] = result;
                output_relation.columns[output_idx]->row_ids = nullptr;
                output_relation.columns[output_idx]->row_id_count = 0;
            } else {
                output_relation.columns[output_idx] = make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::INT32), nullptr);
            }
        }
    }

}


} // namespace duckdb