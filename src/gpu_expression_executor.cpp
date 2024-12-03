#include "gpu_expression_executor.hpp"

namespace duckdb {

template <typename T>
GPUColumn*
GPUExpressionExecutor::ResolveTypeBinaryConstantExpression(GPUColumn* column, BoundConstantExpression& expr, GPUBufferManager* gpuBufferManager, string function_name) {
    T* a = reinterpret_cast<T*> (column->data_wrapper.data);
    size_t size = column->column_length;
    uint64_t b = expr.value.GetValue<uint64_t>();
    T* c = gpuBufferManager->customCudaMalloc<T>(size, 0, 0);
    if (function_name.compare("+") == 0) {
        binaryConstantExpression<T>(a, b, c, size, 0);
    } else if (function_name.compare("-") == 0) {
        binaryConstantExpression<T>(a, b, c, size, 1);
    } else if (function_name.compare("*") == 0) {
        binaryConstantExpression<T>(a, b, c, size, 2);
    } else if (function_name.compare("/") == 0) {
        binaryConstantExpression<T>(a, b, c, size, 3);
    } else {
        throw NotImplementedException("Function name not supported");
    }
    GPUColumn* result = new GPUColumn(size, column->data_wrapper.type, reinterpret_cast<uint8_t*>(c));
    return result;
}

GPUColumn*
GPUExpressionExecutor::HandleBinaryConstantExpression(GPUColumn* column, BoundConstantExpression& expr, GPUBufferManager* gpuBufferManager, string function_name) {
    switch(column->data_wrapper.type) {
      case ColumnType::INT32:
        return ResolveTypeBinaryConstantExpression<int>(column, expr, gpuBufferManager, function_name);
      case ColumnType::INT64:
        return ResolveTypeBinaryConstantExpression<uint64_t>(column, expr, gpuBufferManager, function_name);
      case ColumnType::FLOAT32:
        return ResolveTypeBinaryConstantExpression<float>(column, expr, gpuBufferManager, function_name);
      case ColumnType::FLOAT64:
        return ResolveTypeBinaryConstantExpression<double>(column, expr, gpuBufferManager, function_name);
      default:
        throw NotImplementedException("HandleBinaryConstantExpression Unsupported column type");
    }
}

template <typename T>
GPUColumn*
GPUExpressionExecutor::ResolveTypeBinaryExpression(GPUColumn* column1, GPUColumn* column2, GPUBufferManager* gpuBufferManager, string function_name) {
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
    GPUColumn* result = new GPUColumn(size, column1->data_wrapper.type, reinterpret_cast<uint8_t*>(c));
    return result;
}

GPUColumn*
GPUExpressionExecutor::HandleBinaryExpression(GPUColumn* column1, GPUColumn* column2, GPUBufferManager* gpuBufferManager, string function_name) {
    switch(column1->data_wrapper.type) {
      case ColumnType::INT32:
        return ResolveTypeBinaryExpression<int>(column1, column2, gpuBufferManager, function_name);
      case ColumnType::INT64:
        return ResolveTypeBinaryExpression<uint64_t>(column1, column2, gpuBufferManager, function_name);
      case ColumnType::FLOAT32:
        return ResolveTypeBinaryExpression<float>(column1, column2, gpuBufferManager, function_name);
      case ColumnType::FLOAT64:
        return ResolveTypeBinaryExpression<double>(column1, column2, gpuBufferManager, function_name);
      default:
        throw NotImplementedException("HandleBinaryExpression Unsupported column type");
    }
}

template <typename T>
void 
GPUExpressionExecutor::ResolveTypeComparisonConstantExpression (GPUColumn* column, BoundConstantExpression& expr1, BoundConstantExpression& expr2, uint64_t* &count, uint64_t* & row_ids, ExpressionType expression_type) {
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
GPUExpressionExecutor::HandleComparisonConstantExpression(GPUColumn* column, BoundConstantExpression& expr1, BoundConstantExpression& expr2, uint64_t* &count, uint64_t* &row_ids, ExpressionType expression_type) {
    switch(column->data_wrapper.type) {
      case ColumnType::INT32:
        ResolveTypeComparisonConstantExpression<int>(column, expr1, expr2, count, row_ids, expression_type);
        break;
      case ColumnType::INT64:
        ResolveTypeComparisonConstantExpression<uint64_t>(column, expr1, expr2, count, row_ids, expression_type);
        break;
      case ColumnType::FLOAT32:
        ResolveTypeComparisonConstantExpression<float>(column, expr1, expr2, count, row_ids, expression_type);
        break;
      case ColumnType::FLOAT64:
        ResolveTypeComparisonConstantExpression<double>(column, expr1, expr2, count, row_ids, expression_type);
        break;
      default:
        throw NotImplementedException("HandleComparisonConstantExpression Unsupported column type");
    }
}

template <typename T>
void 
GPUExpressionExecutor::ResolveTypeComparisonExpression (GPUColumn* column1, GPUColumn* column2, uint64_t* &count, uint64_t* & row_ids, ExpressionType expression_type) {
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
GPUExpressionExecutor::HandleComparisonExpression(GPUColumn* column1, GPUColumn* column2, uint64_t* &count, uint64_t* &row_ids, ExpressionType expression_type) {
    switch(column1->data_wrapper.type) {
      case ColumnType::INT32:
        ResolveTypeComparisonExpression<int>(column1, column2, count, row_ids, expression_type);
        break;
      case ColumnType::INT64:
        ResolveTypeComparisonExpression<uint64_t>(column1, column2, count, row_ids, expression_type);
        break;
      case ColumnType::FLOAT32:
        ResolveTypeComparisonExpression<float>(column1, column2, count, row_ids, expression_type);
        break;
      case ColumnType::FLOAT64:
        ResolveTypeComparisonExpression<double>(column1, column2, count, row_ids, expression_type);
        break;
      default:
        throw NotImplementedException("HandleComparisonExpression Unsupported column type");
    }
}

template <typename T>
GPUColumn* 
GPUExpressionExecutor::ResolveTypeMaterializeExpression(GPUColumn* column, BoundReferenceExpression& bound_ref, GPUBufferManager* gpuBufferManager) {
    // GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    size_t size;
    T* a;
    if (column->row_ids != nullptr) {
        T* temp = reinterpret_cast<T*> (column->data_wrapper.data);
        uint64_t* row_ids_input = reinterpret_cast<uint64_t*> (column->row_ids);
        a = gpuBufferManager->customCudaMalloc<T>(column->row_id_count, 0, 0);
        materializeExpression<T>(temp, a, row_ids_input, column->row_id_count);
        size = column->row_id_count;
    } else {
        a = reinterpret_cast<T*> (column->data_wrapper.data);
        size = column->column_length;
    }
    GPUColumn* result = new GPUColumn(size, column->data_wrapper.type, reinterpret_cast<uint8_t*>(a));
    // printf("size %ld\n", size);
    return result;
}

GPUColumn* 
GPUExpressionExecutor::HandleMaterializeExpression(GPUColumn* column, BoundReferenceExpression& bound_ref, GPUBufferManager* gpuBufferManager) {
    switch(column->data_wrapper.type) {
        case ColumnType::INT32:
            return ResolveTypeMaterializeExpression<int>(column, bound_ref, gpuBufferManager);
        case ColumnType::INT64:
            return ResolveTypeMaterializeExpression<uint64_t>(column, bound_ref, gpuBufferManager);
        case ColumnType::FLOAT32:
            return ResolveTypeMaterializeExpression<float>(column, bound_ref, gpuBufferManager);
        case ColumnType::FLOAT64:
            return ResolveTypeMaterializeExpression<double>(column, bound_ref, gpuBufferManager);
        default:
            throw NotImplementedException("HandleMaterializeExpression Unsupported column type");
    }
}

// Code from https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
std::vector<std::string> string_split(std::string s, std::string delimiter) {
    std::vector<std::string> tokens;
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
      token = s.substr(0, pos);
      if(token.length() > 0) {
        tokens.push_back(token);
      }
      s.erase(0, pos + delimiter.length());
    }

    if(s.length() > 0) {
      tokens.push_back(s);
    }

    return tokens;
}

void 
GPUExpressionExecutor::FilterRecursiveExpression(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, Expression& expr, int depth) {
    bool is_specific_filter = HandlingSpecificFilter(input_relation, output_relation, expr);
    if (is_specific_filter) return;
    
    printf("FilterRecursiveExpression Expression class %d at depth %d\n", expr.expression_class);
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    uint64_t* comparison_idx = nullptr;
    uint64_t* count = gpuBufferManager->customCudaHostAlloc<uint64_t>(sizeof(uint64_t));

	switch (expr.expression_class) {
	case ExpressionClass::BOUND_BETWEEN: {
        auto &bound_between = expr.Cast<BoundBetweenExpression>();
        printf("Executing between expression\n");
        // FilterRecursiveExpression(input_relation, output_relation, *(bound_between.input), depth + 1);
        // FilterRecursiveExpression(input_relation, output_relation, *(bound_between.lower), depth + 1);
        // FilterRecursiveExpression(input_relation, output_relation, *(bound_between.upper), depth + 1);
        auto &bound_ref = bound_between.input->Cast<BoundReferenceExpression>();
        auto &bound_lower = bound_between.lower->Cast<BoundConstantExpression>();
        auto &bound_upper = bound_between.upper->Cast<BoundConstantExpression>();
        size_t size;

        GPUColumn* materialized_column = HandleMaterializeExpression(input_relation.columns[bound_ref.index], bound_ref, gpuBufferManager);
        HandleComparisonConstantExpression(materialized_column, bound_lower, bound_upper, count, comparison_idx, bound_between.type);
        if (count[0] == 0) throw NotImplementedException("No match found");
		break;
    } case ExpressionClass::BOUND_REF: {
        auto &bound_ref = expr.Cast<BoundReferenceExpression>();
        printf("Executing bound reference expression\n");
        GPUColumn* src_column = input_relation.columns[bound_ref.index];
        printf("Reading column index %ld with column name %s\n", bound_ref.index, src_column->name.c_str());
		    input_relation.checkLateMaterialization(bound_ref.index);
        printf("output_relation.columns.size() %ld\n", output_relation.columns.size());
        printf("input_relation.columns.size() %ld\n", input_relation.columns.size());
        printf("bound_ref.index %ld\n", bound_ref.index);
        output_relation.columns[bound_ref.index] = input_relation.columns[bound_ref.index];
        output_relation.columns[bound_ref.index]->row_ids = new uint64_t[1];
        printf("Overwrite row ids with column %ld\n", bound_ref.index);
		break;
	} case ExpressionClass::BOUND_CASE: {
        auto &bound_case = expr.Cast<BoundCaseExpression>();
        printf("Executing case expression\n");
        for (idx_t i = 0; i < bound_case.case_checks.size(); i++) {
            FilterRecursiveExpression(input_relation, output_relation, *(bound_case.case_checks[i].when_expr), depth + 1);
            FilterRecursiveExpression(input_relation, output_relation, *(bound_case.case_checks[i].then_expr), depth + 1);
        }
        FilterRecursiveExpression(input_relation, output_relation, *(bound_case.else_expr), depth + 1);
		break;
	} case ExpressionClass::BOUND_CAST: {
        auto &bound_cast = expr.Cast<BoundCastExpression>();
        printf("Executing cast expression\n");
        FilterRecursiveExpression(input_relation, output_relation, *(bound_cast.child), depth + 1);
		break;
	} case ExpressionClass::BOUND_COMPARISON: {
        auto &bound_comparison = expr.Cast<BoundComparisonExpression>();
        printf("Executing comparison expression\n");
        // FilterRecursiveExpression(input_relation, output_relation, *(bound_comparison.left), depth + 1);
        // FilterRecursiveExpression(input_relation, output_relation, *(bound_comparison.right), depth + 1);

        if (bound_comparison.right->expression_class == ExpressionClass::BOUND_CONSTANT) {
            auto &bound_ref1 = bound_comparison.left->Cast<BoundReferenceExpression>();
            auto &bound_ref2 = bound_comparison.right->Cast<BoundConstantExpression>();
            size_t size;

            GPUColumn* materialized_column = HandleMaterializeExpression(input_relation.columns[bound_ref1.index], bound_ref1, gpuBufferManager);
            HandleComparisonConstantExpression(materialized_column, bound_ref2, bound_ref2, count, comparison_idx, bound_comparison.type);
        } else if (bound_comparison.right->expression_class == ExpressionClass::BOUND_REF) {
            auto &bound_ref1 = bound_comparison.left->Cast<BoundReferenceExpression>();
            auto &bound_ref2 = bound_comparison.right->Cast<BoundReferenceExpression>();
            size_t size;

            GPUColumn* materialized_column1 = HandleMaterializeExpression(input_relation.columns[bound_ref1.index], bound_ref1, gpuBufferManager);
            GPUColumn* materialized_column2 = HandleMaterializeExpression(input_relation.columns[bound_ref2.index], bound_ref2, gpuBufferManager);
            HandleComparisonExpression(materialized_column1, materialized_column2, count, comparison_idx, bound_comparison.type);            
        }
        if (count[0] == 0) throw NotImplementedException("No match found");
		break;
	} case ExpressionClass::BOUND_CONJUNCTION: {
        auto &bound_conjunction = expr.Cast<BoundConjunctionExpression>();
		printf("Executing conjunction expression\n");
        for (auto &child : bound_conjunction.children) {
            FilterRecursiveExpression(input_relation, output_relation, *child, depth + 1);
        }
		break;
	} case ExpressionClass::BOUND_CONSTANT: {
      std::string search_string = expr.Cast<BoundConstantExpression>().value.ToString();
      printf("Calling string matching at depth of %d for str %s\n", depth, search_string.c_str());
		  break;
	} case ExpressionClass::BOUND_FUNCTION: {
        auto &bound_function = expr.Cast<BoundFunctionExpression>();
        std::string bound_function_name = bound_function.ToString();
        std::cout << "Got bound function " << bound_function_name << std::endl;
        if(bound_function_name.find("contains") != std::string::npos) {
          // Get the column
          GPUColumn* curr_column = input_relation.columns[0];
          std::cout << "Running single term contains function with is row ids null of " << (curr_column->row_ids == nullptr) << std::endl;

          // Get the match string from the other child
          auto& bound_const = (*bound_function.children[1]).Cast<BoundConstantExpression>();
          std::string match_str = bound_const.value.ToString();
          std::cout << "Got match str of " << match_str << " at depth " << depth << std::endl;
          
          // Get the input column
          if(curr_column->row_id_count == 0) {
            curr_column->row_id_count = curr_column->data_wrapper.num_strings;
          }
          comparison_idx = StringMatching(curr_column, match_str, count);
        } else if(bound_function_name.find("~~") != std::string::npos) {
          // Get the column
          GPUColumn* curr_column = input_relation.columns[0];
          std::cout << "Running multi term match function with is row ids null of " << (curr_column->row_ids == nullptr) << std::endl;

          // Get the match string from the other child
          auto& bound_const = (*bound_function.children[1]).Cast<BoundConstantExpression>();
          std::string match_str = bound_const.value.ToString();
          std::vector<std::string> match_terms = string_split(match_str, "%");
          std::cout << "Got match str of " << match_str << " at depth " << depth << std::endl;
          
          // Get the input column
          if(curr_column->row_id_count == 0) {
            curr_column->row_id_count = curr_column->data_wrapper.num_strings;
          }      

          comparison_idx = MultiStringMatching(curr_column, match_terms, count);
        } else {
          std::cout << "Running non contains bound function" << std::endl;
          for (auto &child : bound_function.children) {
            printf("Executing function expression child\n");
            FilterRecursiveExpression(input_relation, output_relation, *child, depth + 1);
          }
        }
		break;
	} case ExpressionClass::BOUND_OPERATOR: {
        printf("Executing IN expression\n");
        auto &bound_operator = expr.Cast<BoundOperatorExpression>();
        for (auto &child : bound_operator.children) {
            FilterRecursiveExpression(input_relation, output_relation, *child, depth + 1);
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
        printf("Writing filter result\n");
        for (int i = 0; i < input_relation.columns.size(); i++) {
            output_relation.columns[i] = new GPUColumn(*input_relation.columns[i]);
            if (comparison_idx) {
                if (input_relation.columns[i]->row_ids == nullptr) {
                  std::cout << "Doing a direct copy of the row ids" << std::endl;  
                  output_relation.columns[i]->row_ids = comparison_idx;
                } else {
                  std::cout << "Calling materializeExpression with " << count[0] << " records and existing count of " << input_relation.columns[i]->row_id_count << std::endl;
                  uint64_t* row_ids_input = reinterpret_cast<uint64_t*> (input_relation.columns[i]->row_ids);
                  uint64_t* new_row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(count[0], 0, 0);
                  materializeExpression<uint64_t>(row_ids_input, new_row_ids, comparison_idx, count[0]);
                  output_relation.columns[i]->row_ids = new_row_ids;
                }
            }
            if(count) output_relation.columns[i]->row_id_count = count[0];

            std::cout << "Wrote output relation with " << output_relation.columns[i]->row_id_count << " rows and not null of ";
            std::cout << (output_relation.columns[i]->row_ids != nullptr) << std::endl;
        }
    }
}


void 
GPUExpressionExecutor::ProjectionRecursiveExpression(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, Expression& expr, int output_idx, int depth) {
    bool is_specific_projection = HandlingSpecificProjection(input_relation, output_relation, expr, output_idx);
    if (is_specific_projection) return;

    GPUColumn* result;

    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

	switch (expr.expression_class) {
	case ExpressionClass::BOUND_BETWEEN: {
        auto &bound_between = expr.Cast<BoundBetweenExpression>();
        printf("Executing between expression\n");
        ProjectionRecursiveExpression(input_relation, output_relation, *(bound_between.input), output_idx, depth + 1);
        ProjectionRecursiveExpression(input_relation, output_relation, *(bound_between.lower), output_idx, depth + 1);
        ProjectionRecursiveExpression(input_relation, output_relation, *(bound_between.upper), output_idx, depth + 1);
		break;
	} case ExpressionClass::BOUND_REF: {
        auto &bound_ref = expr.Cast<BoundReferenceExpression>();
        printf("Executing bound reference expression\n");
        printf("Reading column index %ld\n", bound_ref.index);
		input_relation.checkLateMaterialization(bound_ref.index);
		break;
	} case ExpressionClass::BOUND_CASE: {
        auto &bound_case = expr.Cast<BoundCaseExpression>();
        printf("Executing case expression\n");
        for (idx_t i = 0; i < bound_case.case_checks.size(); i++) {
            ProjectionRecursiveExpression(input_relation, output_relation, *(bound_case.case_checks[i].when_expr), output_idx, depth + 1);
            ProjectionRecursiveExpression(input_relation, output_relation, *(bound_case.case_checks[i].then_expr), output_idx, depth + 1);
        }
        ProjectionRecursiveExpression(input_relation, output_relation, *(bound_case.else_expr), output_idx, depth + 1);
		break;
	} case ExpressionClass::BOUND_CAST: {
        auto &bound_cast = expr.Cast<BoundCastExpression>();
        printf("Executing cast expression\n");
        ProjectionRecursiveExpression(input_relation, output_relation, *(bound_cast.child), output_idx, depth + 1);
		break;
	} case ExpressionClass::BOUND_COMPARISON: {
        auto &bound_comparison = expr.Cast<BoundComparisonExpression>();
        printf("Executing comparison expression\n");
        ProjectionRecursiveExpression(input_relation, output_relation, *(bound_comparison.left), output_idx, depth + 1);
        ProjectionRecursiveExpression(input_relation, output_relation, *(bound_comparison.right), output_idx, depth + 1);
		break;
	} case ExpressionClass::BOUND_CONJUNCTION: {
        auto &bound_conjunction = expr.Cast<BoundConjunctionExpression>();
		printf("Executing conjunction expression\n");
        for (auto &child : bound_conjunction.children) {
            ProjectionRecursiveExpression(input_relation, output_relation, *child, output_idx, depth + 1);
        }
		break;
	} case ExpressionClass::BOUND_CONSTANT: {
        printf("Reading value %s\n", expr.Cast<BoundConstantExpression>().value.ToString().c_str());
		break;
	} case ExpressionClass::BOUND_FUNCTION: {
        printf("Executing function expression\n");
        auto &bound_function = expr.Cast<BoundFunctionExpression>();
        // for (auto &child : bound_function.children) {
        //     ProjectionRecursiveExpression(input_relation, output_relation, *child, output_idx, depth + 1);
        // }

        if (bound_function.children[1]->expression_class == ExpressionClass::BOUND_CONSTANT) {
            auto &bound_ref1 = bound_function.children[0]->Cast<BoundReferenceExpression>();
            auto &bound_ref2 = bound_function.children[1]->Cast<BoundConstantExpression>();
            GPUColumn* materialized_column = HandleMaterializeExpression(input_relation.columns[bound_ref1.index], bound_ref1, gpuBufferManager);
            result = HandleBinaryConstantExpression(materialized_column, bound_ref2, gpuBufferManager, bound_function.function.name);
        } else if (bound_function.children[1]->expression_class == ExpressionClass::BOUND_REF) {
            auto &bound_ref1 = bound_function.children[0]->Cast<BoundReferenceExpression>();
            auto &bound_ref2 = bound_function.children[1]->Cast<BoundReferenceExpression>();
            GPUColumn* materialized_column1 = HandleMaterializeExpression(input_relation.columns[bound_ref1.index], bound_ref1, gpuBufferManager);
            GPUColumn* materialized_column2 = HandleMaterializeExpression(input_relation.columns[bound_ref2.index], bound_ref2, gpuBufferManager);
            result = HandleBinaryExpression(materialized_column1, materialized_column2, gpuBufferManager, bound_function.function.name);            
        }

        // auto &bound_ref1 = bound_function.children[0]->Cast<BoundReferenceExpression>();
        // auto &bound_ref2 = bound_function.children[1]->Cast<BoundReferenceExpression>();
        // size_t size = input_relation.columns[bound_ref1.index]->column_length;
        // double* ptr_double = gpuBufferManager->customCudaMalloc<double>(size, 0, 0);

        // double* a, *b;
        // if (input_relation.checkLateMaterialization(bound_ref1.index)) {
        //     double* temp = reinterpret_cast<double*> (input_relation.columns[bound_ref1.index]->data_wrapper.data);
        //     uint64_t* row_ids_input = reinterpret_cast<uint64_t*> (input_relation.columns[bound_ref1.index]->row_ids);
        //     a = gpuBufferManager->customCudaMalloc<double>(input_relation.columns[bound_ref1.index]->row_id_count, 0, 0);
        //     materializeExpression<double>(temp, a, row_ids_input, input_relation.columns[bound_ref1.index]->row_id_count);
        // } else {
        //     a = reinterpret_cast<double*> (input_relation.columns[bound_ref1.index]->data_wrapper.data);
        // }

        // if (input_relation.checkLateMaterialization(bound_ref2.index)) {
        //     double* temp = reinterpret_cast<double*> (input_relation.columns[bound_ref2.index]->data_wrapper.data);
        //     uint64_t* row_ids_input = reinterpret_cast<uint64_t*> (input_relation.columns[bound_ref2.index]->row_ids);
        //     b = gpuBufferManager->customCudaMalloc<double>(input_relation.columns[bound_ref2.index]->row_id_count, 0, 0);
        //     materializeExpression<double>(temp, b, row_ids_input, input_relation.columns[bound_ref2.index]->row_id_count);
        //     size = input_relation.columns[bound_ref2.index]->row_id_count;
        // } else {
        //     b = reinterpret_cast<double*> (input_relation.columns[bound_ref2.index]->data_wrapper.data);
        //     size = input_relation.columns[bound_ref2.index]->column_length;
        // }
        // binaryExpression<double>(a, b, ptr_double, (uint64_t) size, 0);
        // result = new GPUColumn(size, ColumnType::FLOAT64, reinterpret_cast<uint8_t*>(ptr_double));
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
    printf("Writing projection result to idx %ld\n", output_idx);
    if (depth == 0) {
        if (expr.expression_class == ExpressionClass::BOUND_REF) {
            output_relation.columns[output_idx] = input_relation.columns[expr.Cast<BoundReferenceExpression>().index];
            // printf("size = %ld\n", output_relation.columns[output_idx]->column_length);
            // printf("row ids count = %ld\n", output_relation.columns[output_idx]->row_id_count);
        } else {
            uint8_t* fake_data = new uint8_t[1];
            if (result) {
                output_relation.columns[output_idx] = result;
                // output_relation.length = result->column_length;
                output_relation.columns[output_idx]->row_ids = nullptr;
                output_relation.columns[output_idx]->row_id_count = 0;
                // uint8_t* host_data = new uint8_t[output_relation.length * 8];
                // callCudaMemcpyDeviceToHost<uint8_t>(host_data, output_relation.columns[output_idx]->data_wrapper.data, output_relation.length * 8, 0);
            } else {
                output_relation.columns[output_idx] = new GPUColumn(1, ColumnType::INT32, fake_data);
            }
        }
    }
}


} // namespace duckdb