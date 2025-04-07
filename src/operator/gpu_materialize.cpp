#include "operator/gpu_materialize.hpp"

namespace duckdb
{

    template <typename T>
    GPUColumn *
    ResolveTypeMaterializeExpression(GPUColumn *column, BoundReferenceExpression &bound_ref, GPUBufferManager *gpuBufferManager)
    {
        printf("TESTING\n");
        size_t size;
        printf("A\n");
        T *a;
        printf("B\n");
        if (column->data_wrapper.data == nullptr)
        {
            return new GPUColumn(column->column_length, column->data_wrapper.type, nullptr);
        }
        printf("C\n");
        if (column->row_ids != nullptr)
        {
            printf("D\n");
            T *temp = reinterpret_cast<T *>(column->data_wrapper.data);
            printf("E\n");
            uint64_t *row_ids_input = reinterpret_cast<uint64_t *>(column->row_ids);
            printf("F\n");
            a = gpuBufferManager->customCudaMalloc<T>(column->row_id_count, 0, 0);
            printf("G\n");
            materializeExpression<T>(temp, a, row_ids_input, column->row_id_count);
            printf("H\n");
            size = column->row_id_count;
            printf("I\n");
        }
        else
        {
            printf("J\n");
            a = reinterpret_cast<T *>(column->data_wrapper.data);
            printf("K\n");
            size = column->column_length;
            printf("L\n");
        }
        printf("M\n");
        GPUColumn *result = new GPUColumn(size, column->data_wrapper.type, reinterpret_cast<uint8_t *>(a));
        printf("N\n");
        result->is_unique = column->is_unique;
        printf("O\n");
        printf("Result: %d\n", result->data_wrapper.type);
        return result;
    }

    GPUColumn *
    ResolveTypeMaterializeString(GPUColumn *column, BoundReferenceExpression &bound_ref, GPUBufferManager *gpuBufferManager)
    {
        size_t size;
        uint8_t *a;
        uint64_t *result_offset;
        uint64_t *new_num_bytes;
        if (column->data_wrapper.data == nullptr)
        {
            return new GPUColumn(column->column_length, column->data_wrapper.type, nullptr, nullptr, column->data_wrapper.num_bytes, column->data_wrapper.is_string_data);
        }
        if (column->row_ids != nullptr)
        {
            // Late materalize the input relationship
            uint8_t *data = column->data_wrapper.data;
            uint64_t *offset = column->data_wrapper.offset;
            uint64_t *row_ids = column->row_ids;
            size = column->row_id_count;
            materializeString(data, offset, a, result_offset, row_ids, new_num_bytes, size);
        }
        else
        {
            a = column->data_wrapper.data;
            result_offset = column->data_wrapper.offset;
            size = column->column_length;
            new_num_bytes = new uint64_t[1];
            new_num_bytes[0] = column->data_wrapper.num_bytes;
        }
        // printf("Materialized string column with size %ld\n", new_num_bytes[0]);
        GPUColumn *result = new GPUColumn(size, column->data_wrapper.type, a, result_offset, new_num_bytes[0], column->data_wrapper.is_string_data);
        result->is_unique = column->is_unique;
        return result;
    }

    GPUColumn *
    HandleMaterializeExpression(GPUColumn *column, BoundReferenceExpression &bound_ref, GPUBufferManager *gpuBufferManager)
    {
        printf("SFDKLSDJFL\n");
        printf("Type %d\n", column->data_wrapper.type);
        switch (column->data_wrapper.type)
        {
        case ColumnType::INT32:
            printf("INT32\n");
            return ResolveTypeMaterializeExpression<int>(column, bound_ref, gpuBufferManager);
        case ColumnType::INT64:
            printf("INT64\n");
            return ResolveTypeMaterializeExpression<uint64_t>(column, bound_ref, gpuBufferManager);
        case ColumnType::FLOAT32:
            printf("FLOAT32\n");
            return ResolveTypeMaterializeExpression<float>(column, bound_ref, gpuBufferManager);
        case ColumnType::FLOAT64:
            printf("FLOAT64\n");
            return ResolveTypeMaterializeExpression<double>(column, bound_ref, gpuBufferManager);
        case ColumnType::BOOLEAN:
            printf("BOOLEAN\n");
            return ResolveTypeMaterializeExpression<uint8_t>(column, bound_ref, gpuBufferManager);
        case ColumnType::VARCHAR:
            printf("VARCHAR\n");
            return ResolveTypeMaterializeString(column, bound_ref, gpuBufferManager);
        default:
            throw NotImplementedException("Unsupported column type ksldfjsdlk");
        }
    }

    void
    HandleMaterializeRowIDs(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation, uint64_t count, uint64_t *row_ids, GPUBufferManager *gpuBufferManager, bool maintain_unique)
    {
        vector<uint64_t *> new_row_ids;
        vector<uint64_t *> prev_row_ids;
        for (int i = 0; i < input_relation.columns.size(); i++)
        {
            printf("Passing column idx %d in input relation to idx %d in output relation\n", i, i);
            if (count == 0)
            {
                output_relation.columns[i] = new GPUColumn(0, input_relation.columns[i]->data_wrapper.type, input_relation.columns[i]->data_wrapper.data,
                                                           input_relation.columns[i]->data_wrapper.offset, 0, input_relation.columns[i]->data_wrapper.is_string_data);
                output_relation.columns[i]->row_id_count = 0;
                if (maintain_unique)
                {
                    output_relation.columns[i]->is_unique = input_relation.columns[i]->is_unique;
                }
                else
                {
                    output_relation.columns[i]->is_unique = false;
                }
                continue;
            }
            output_relation.columns[i] = new GPUColumn(input_relation.columns[i]->column_length, input_relation.columns[i]->data_wrapper.type, input_relation.columns[i]->data_wrapper.data,
                                                       input_relation.columns[i]->data_wrapper.offset, input_relation.columns[i]->data_wrapper.num_bytes, input_relation.columns[i]->data_wrapper.is_string_data);
            if (maintain_unique)
            {
                output_relation.columns[i]->is_unique = input_relation.columns[i]->is_unique;
            }
            else
            {
                output_relation.columns[i]->is_unique = false;
            }
            if (row_ids)
            {
                if (input_relation.columns[i]->row_ids == nullptr)
                {
                    output_relation.columns[i]->row_ids = row_ids;
                }
                else
                {
                    auto it = find(prev_row_ids.begin(), prev_row_ids.end(), input_relation.columns[i]->row_ids);
                    if (it != prev_row_ids.end())
                    {
                        auto idx = it - prev_row_ids.begin();
                        output_relation.columns[i]->row_ids = new_row_ids[idx];
                    }
                    else
                    {
                        uint64_t *temp_prev_row_ids = reinterpret_cast<uint64_t *>(input_relation.columns[i]->row_ids);
                        uint64_t *temp_new_row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(count, 0, 0);
                        materializeExpression<uint64_t>(temp_prev_row_ids, temp_new_row_ids, row_ids, count);
                        output_relation.columns[i]->row_ids = temp_new_row_ids;
                        new_row_ids.push_back(temp_new_row_ids);
                        prev_row_ids.push_back(temp_prev_row_ids);
                    }
                }
            }
            output_relation.columns[i]->row_id_count = count;
        }
    }

    void
    HandleMaterializeRowIDsRHS(GPUIntermediateRelation &hash_table_result, GPUIntermediateRelation &output_relation,
                               vector<idx_t> rhs_output_columns, size_t offset, uint64_t count, uint64_t *row_ids, GPUBufferManager *gpuBufferManager, bool maintain_unique)
    {
        vector<uint64_t *> new_row_ids;
        vector<uint64_t *> prev_row_ids;
        for (idx_t i = 0; i < rhs_output_columns.size(); i++)
        {
            const auto rhs_col = rhs_output_columns[i];
            printf("Passing column idx %d from hash table to idx %d in output relation\n", rhs_col, offset + i);
            if (count == 0)
            {
                output_relation.columns[offset + i] = new GPUColumn(0, hash_table_result.columns[rhs_col]->data_wrapper.type, hash_table_result.columns[rhs_col]->data_wrapper.data,
                                                                    hash_table_result.columns[rhs_col]->data_wrapper.offset, 0, hash_table_result.columns[rhs_col]->data_wrapper.is_string_data);
                output_relation.columns[offset + i]->row_id_count = 0;
                if (maintain_unique)
                {
                    output_relation.columns[offset + i]->is_unique = hash_table_result.columns[rhs_col]->is_unique;
                }
                else
                {
                    output_relation.columns[offset + i]->is_unique = false;
                }
                continue;
            }
            output_relation.columns[offset + i] = new GPUColumn(hash_table_result.columns[rhs_col]->column_length, hash_table_result.columns[rhs_col]->data_wrapper.type, hash_table_result.columns[rhs_col]->data_wrapper.data,
                                                                hash_table_result.columns[rhs_col]->data_wrapper.offset, hash_table_result.columns[rhs_col]->data_wrapper.num_bytes, hash_table_result.columns[rhs_col]->data_wrapper.is_string_data);
            if (maintain_unique)
            {
                output_relation.columns[offset + i]->is_unique = hash_table_result.columns[rhs_col]->is_unique;
            }
            else
            {
                output_relation.columns[offset + i]->is_unique = false;
            }
            if (row_ids)
            {
                if (hash_table_result.columns[rhs_col]->row_ids == nullptr)
                {
                    output_relation.columns[offset + i]->row_ids = row_ids;
                }
                else
                {
                    auto it = find(prev_row_ids.begin(), prev_row_ids.end(), hash_table_result.columns[rhs_col]->row_ids);
                    if (it != prev_row_ids.end())
                    {
                        auto idx = it - prev_row_ids.begin();
                        output_relation.columns[offset + i]->row_ids = new_row_ids[idx];
                    }
                    else
                    {
                        uint64_t *temp_prev_row_ids = reinterpret_cast<uint64_t *>(hash_table_result.columns[rhs_col]->row_ids);
                        uint64_t *temp_new_row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(count, 0, 0);
                        materializeExpression<uint64_t>(temp_prev_row_ids, temp_new_row_ids, row_ids, count);
                        output_relation.columns[offset + i]->row_ids = temp_new_row_ids;
                        new_row_ids.push_back(temp_new_row_ids);
                        prev_row_ids.push_back(temp_prev_row_ids);
                    }
                }
            }
            output_relation.columns[offset + i]->row_id_count = count;
        }
    }

} // namespace duckdb