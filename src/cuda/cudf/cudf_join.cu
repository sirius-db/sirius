#include "cudf/cudf_utils.hpp"
#include "../operator/cuda_helper.cuh"
#include "gpu_physical_hash_join.hpp"
#include "gpu_buffer_manager.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "log/logging.hpp"

namespace duckdb {

void cudf_hash_inner_join(vector<shared_ptr<GPUColumn>>& probe_keys, vector<shared_ptr<GPUColumn>>& build_keys, int num_keys, uint64_t*& row_ids_left, uint64_t*& row_ids_right, uint64_t*& count) {
    GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());
    if (build_keys[0]->column_length == 0 || probe_keys[0]->column_length == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
        count[0] = 0;
        return;
    }

    SIRIUS_LOG_DEBUG("CUDF hash inner join");
    SIRIUS_LOG_DEBUG("Build Input size: {}", build_keys[0]->column_length);
    SIRIUS_LOG_DEBUG("Probe Input size: {}", probe_keys[0]->column_length);
    SETUP_TIMING();
    START_TIMER();

    cudf::set_current_device_resource(gpuBufferManager->mr);

    std::vector<cudf::column_view> build_keys_cudf;
    for (int key = 0; key < num_keys; key++) {
        build_keys_cudf.push_back(build_keys[key]->convertToCudfColumn());
    }

    auto build_table = cudf::table_view(build_keys_cudf);
    auto hash_table = cudf::hash_join(build_table, cudf::null_equality::EQUAL);

    std::vector<cudf::column_view> probe_keys_cudf;
    for (int key = 0; key < num_keys; key++) {
        probe_keys_cudf.push_back(probe_keys[key]->convertToCudfColumn());
    }

    auto probe_table = cudf::table_view(probe_keys_cudf);
    auto result = hash_table.inner_join(probe_table);
    
    auto result_count = result.first->size();
    rmm::device_buffer row_ids_left_buffer = result.first->release();
    rmm::device_buffer row_ids_right_buffer = result.second->release();

    row_ids_left = convertInt32ToUInt64(reinterpret_cast<int32_t*>(row_ids_left_buffer.data()), result_count);
    row_ids_right = convertInt32ToUInt64(reinterpret_cast<int32_t*>(row_ids_right_buffer.data()), result_count);

    gpuBufferManager->rmm_stored_buffers.push_back(std::make_unique<rmm::device_buffer>(std::move(row_ids_left_buffer)));
    gpuBufferManager->rmm_stored_buffers.push_back(std::make_unique<rmm::device_buffer>(std::move(row_ids_right_buffer)));

    count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
    count[0] = result_count;

    STOP_TIMER();
    SIRIUS_LOG_DEBUG("CUDF Inner join result count: {}", count[0]);
}

void cudf_mixed_or_conditional_inner_join(vector<shared_ptr<GPUColumn>>& probe_columns, vector<shared_ptr<GPUColumn>>& build_columns, const vector<JoinCondition>& conditions, JoinType join_type, uint64_t*& row_ids_left, uint64_t*& row_ids_right, uint64_t*& count) {
    
    GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());
    if (build_columns[0]->column_length == 0 || probe_columns[0]->column_length == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
        count[0] = 0;
        return;
    }

    SIRIUS_LOG_DEBUG("CUDF mixed or conditional inner join");
    SIRIUS_LOG_DEBUG("Build Input size: {}", build_columns[0]->column_length);
    SIRIUS_LOG_DEBUG("Probe Input size: {}", probe_columns[0]->column_length);
    SETUP_TIMING();
    START_TIMER();

    cudf::set_current_device_resource(gpuBufferManager->mr);

    std::vector<cudf::column_view> probe_equal_columns;
    std::vector<cudf::column_view> build_equal_columns;
    std::vector<cudf::column_view> probe_conditional_columns;
    std::vector<cudf::column_view> build_conditional_columns;

    std::vector<cudf::ast::operation> cudf_exprs;
    std::vector<cudf::ast::column_reference> owned_left_column_references;
    std::vector<cudf::ast::column_reference> owned_right_column_references;
    for (int cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
        switch (conditions[cond_idx].comparison) {
            case ExpressionType::COMPARE_EQUAL:
            case ExpressionType::COMPARE_NOT_DISTINCT_FROM: {
                probe_equal_columns.push_back(probe_columns[cond_idx]->convertToCudfColumn());
                build_equal_columns.push_back(build_columns[cond_idx]->convertToCudfColumn());
                break;
            }
            case ExpressionType::COMPARE_NOTEQUAL:
            case ExpressionType::COMPARE_DISTINCT_FROM:
            case ExpressionType::COMPARE_LESSTHAN:
            case ExpressionType::COMPARE_GREATERTHAN:
            case ExpressionType::COMPARE_LESSTHANOREQUALTO:
            case ExpressionType::COMPARE_GREATERTHANOREQUALTO: {
                probe_conditional_columns.push_back(probe_columns[cond_idx]->convertToCudfColumn());
                build_conditional_columns.push_back(build_columns[cond_idx]->convertToCudfColumn());
                cudf::ast::ast_operator compare_op;
                switch (conditions[cond_idx].comparison) {
                    case ExpressionType::COMPARE_NOTEQUAL:
                    case ExpressionType::COMPARE_DISTINCT_FROM:
                        compare_op = cudf::ast::ast_operator::NOT_EQUAL;
                        break;
                    case ExpressionType::COMPARE_LESSTHAN:
                        compare_op = cudf::ast::ast_operator::LESS;
                        break;
                    case ExpressionType::COMPARE_GREATERTHAN:
                        compare_op = cudf::ast::ast_operator::GREATER;
                        break;
                    case ExpressionType::COMPARE_LESSTHANOREQUALTO:
                        compare_op = cudf::ast::ast_operator::LESS_EQUAL;
                        break;
                    case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
                        compare_op = cudf::ast::ast_operator::GREATER_EQUAL;
                        break;
                }
                owned_left_column_references.emplace_back(probe_conditional_columns.size() - 1, cudf::ast::table_reference::LEFT);
                owned_right_column_references.emplace_back(build_conditional_columns.size() - 1, cudf::ast::table_reference::RIGHT);
                cudf_exprs.emplace_back(compare_op, owned_left_column_references.back(), owned_right_column_references.back());
                break;
            }
            default:
                throw NotImplementedException("Unsupported comparison type in `cudf_mixed_join`: %d",
                                              static_cast<int>(conditions[cond_idx].comparison));
        }
    }

    // merge cudf_exprs into a single expression, it's guaranteed from caller that `cudf_exprs` is not empty
    std::vector<cudf::ast::operation> intermediate_cudf_exprs;
    intermediate_cudf_exprs.push_back(std::move(cudf_exprs[0]));
    for (int expr_idx = 1; expr_idx < cudf_exprs.size(); expr_idx++) {
        intermediate_cudf_exprs.emplace_back(
            cudf::ast::ast_operator::BITWISE_AND, intermediate_cudf_exprs.back(), cudf_exprs[expr_idx]);
    }
    const auto& final_expr = intermediate_cudf_exprs.back();

    auto probe_equal_table = cudf::table_view(probe_equal_columns);
    auto build_equal_table = cudf::table_view(build_equal_columns);
    auto probe_conditional_table = cudf::table_view(probe_conditional_columns);
    auto build_conditional_table = cudf::table_view(build_conditional_columns);

    std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
              std::unique_ptr<rmm::device_uvector<cudf::size_type>>> result;
    if (probe_equal_table.num_columns() == 0) {
        result = cudf::conditional_inner_join(probe_conditional_table, build_conditional_table, final_expr);
    } else {
        result = cudf::mixed_inner_join(probe_equal_table, build_equal_table, probe_conditional_table, build_conditional_table, final_expr);
    }

    auto result_count = result.first->size();
    rmm::device_buffer row_ids_left_buffer = result.first->release();
    rmm::device_buffer row_ids_right_buffer = result.second->release();

    row_ids_left = convertInt32ToUInt64(reinterpret_cast<int32_t*>(row_ids_left_buffer.data()), result_count);
    row_ids_right = convertInt32ToUInt64(reinterpret_cast<int32_t*>(row_ids_right_buffer.data()), result_count);

    gpuBufferManager->rmm_stored_buffers.push_back(std::make_unique<rmm::device_buffer>(std::move(row_ids_left_buffer)));
    gpuBufferManager->rmm_stored_buffers.push_back(std::make_unique<rmm::device_buffer>(std::move(row_ids_right_buffer)));

    count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
    count[0] = result_count;

    STOP_TIMER();
    SIRIUS_LOG_DEBUG("CUDF Mixed join result count: {}", count[0]);
}

} //namespace duckdb