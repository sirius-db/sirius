#include "cudf/cudf_utils.hpp"
#include "gpu_physical_hash_join.hpp"
#include "gpu_buffer_manager.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

namespace duckdb {

void cudf_probe(vector<shared_ptr<GPUColumn>>& probe_keys, cudf::hash_join* hash_table, int num_keys, uint64_t*& row_ids_left, uint64_t*& row_ids_right, uint64_t*& count)
{
    GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());
    if (probe_keys[0]->column_length == 0) {
        printf("N is 0\n");
        count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
        count[0] = 0;
        return;
    }

    cudf::set_current_device_resource(gpuBufferManager->mr);
    printf("CUDF probe\n");

    std::vector<cudf::column_view> probe_keys_cudf;

    for (int key = 0; key < num_keys; key++) {
        probe_keys_cudf.push_back(probe_keys[key]->convertToCudfColumn());
    }

    auto probe_table = cudf::table_view(probe_keys_cudf);

    auto result = hash_table->inner_join(probe_table);
    
    auto result_count = result.first->size();
    rmm::device_buffer row_ids_left_buffer = result.first->release();
    rmm::device_buffer row_ids_right_buffer = result.second->release();

    row_ids_left = convertInt32ToUInt64(reinterpret_cast<int32_t*>(row_ids_left_buffer.data()), result_count);
    row_ids_right = convertInt32ToUInt64(reinterpret_cast<int32_t*>(row_ids_right_buffer.data()), result_count);

    gpuBufferManager->rmm_stored_buffers.push_back(std::make_unique<rmm::device_buffer>(std::move(row_ids_left_buffer)));
    gpuBufferManager->rmm_stored_buffers.push_back(std::make_unique<rmm::device_buffer>(std::move(row_ids_right_buffer)));

    count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
    count[0] = result_count;
}

void cudf_build(vector<shared_ptr<GPUColumn>>& build_columns, cudf::hash_join*& hash_table, int num_keys) {

    if (build_columns[0]->column_length == 0) {
        printf("N is 0\n");
        return;
    }

    printf("CUDF build\n");

    GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());
    cudf::set_current_device_resource(gpuBufferManager->mr);

    std::vector<cudf::column_view> build_keys_cudf;
    for (int key = 0; key < num_keys; key++) {
        build_keys_cudf.push_back(build_columns[key]->convertToCudfColumn());
    }

    auto build_table = cudf::table_view(build_keys_cudf);
    hash_table = new cudf::hash_join(build_table, cudf::null_equality::EQUAL);
}

void cudf_mixed_join(vector<shared_ptr<GPUColumn>>& probe_columns, vector<shared_ptr<GPUColumn>>& build_columns, const vector<JoinCondition>& conditions, JoinType join_type, uint64_t*& row_ids_left, uint64_t*& row_ids_right, uint64_t*& count) {
    
    GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());
    if (build_columns[0]->column_length == 0 || probe_columns[0]->column_length == 0) {
        printf("N is 0\n");
        count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
        count[0] = 0;
        return;
    }

    printf("CUDF mixed join\n");

    cudf::set_current_device_resource(gpuBufferManager->mr);

    std::vector<cudf::column_view> probe_equal_columns;
    std::vector<cudf::column_view> build_equal_columns;
    std::vector<cudf::column_view> probe_conditional_columns;
    std::vector<cudf::column_view> build_conditional_columns;

    std::vector<cudf::ast::operation> cudf_exprs;
    for (int cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
        if (conditions[cond_idx].comparison == ExpressionType::COMPARE_EQUAL || conditions[cond_idx].comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
            probe_equal_columns.push_back(probe_columns[cond_idx]->convertToCudfColumn());
            build_equal_columns.push_back(build_columns[cond_idx]->convertToCudfColumn());
        } else {
            if (conditions[cond_idx].comparison == ExpressionType::COMPARE_NOTEQUAL || conditions[cond_idx].comparison == ExpressionType::COMPARE_DISTINCT_FROM) {
                probe_conditional_columns.push_back(probe_columns[cond_idx]->convertToCudfColumn());
                build_conditional_columns.push_back(build_columns[cond_idx]->convertToCudfColumn());

                auto not_equal_op = cudf::ast::ast_operator::NOT_EQUAL;
                auto left_ref = cudf::ast::column_reference(probe_conditional_columns.size() - 1, cudf::ast::table_reference::LEFT);
                auto right_ref = cudf::ast::column_reference(build_conditional_columns.size() - 1, cudf::ast::table_reference::RIGHT);
                auto cudf_expr = cudf::ast::operation(not_equal_op, left_ref, right_ref);
                cudf_exprs.push_back(cudf_expr);
            } else {
                throw NotImplementedException("Unsupported comparison type");
            }
        }
    }

    //merge cudf_exprs into a single expression
    cudf::ast::operation final_expr = cudf_exprs[0];
    for (int expr_idx = 1; expr_idx < cudf_exprs.size(); expr_idx++) {
        final_expr = cudf::ast::operation(cudf::ast::ast_operator::BITWISE_AND, final_expr, cudf_exprs[expr_idx]);
    }

    auto probe_equal_table = cudf::table_view(probe_equal_columns);
    auto build_equal_table = cudf::table_view(build_equal_columns);
    auto probe_conditional_table = cudf::table_view(probe_conditional_columns);
    auto build_conditional_table = cudf::table_view(build_conditional_columns);

    auto result = cudf::mixed_inner_join(probe_equal_table, build_equal_table, probe_conditional_table, build_conditional_table, final_expr);

    auto result_count = result.first->size();
    rmm::device_buffer row_ids_left_buffer = result.first->release();
    rmm::device_buffer row_ids_right_buffer = result.second->release();

    row_ids_left = convertInt32ToUInt64(reinterpret_cast<int32_t*>(row_ids_left_buffer.data()), result_count);
    row_ids_right = convertInt32ToUInt64(reinterpret_cast<int32_t*>(row_ids_right_buffer.data()), result_count);

    gpuBufferManager->rmm_stored_buffers.push_back(std::make_unique<rmm::device_buffer>(std::move(row_ids_left_buffer)));
    gpuBufferManager->rmm_stored_buffers.push_back(std::make_unique<rmm::device_buffer>(std::move(row_ids_right_buffer)));

    count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
    count[0] = result_count;
}

} //namespace duckdb