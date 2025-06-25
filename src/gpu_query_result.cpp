/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gpu_query_result.hpp"
#include "duckdb/common/to_string.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/box_renderer.hpp"

namespace duckdb {

void GPUResultCollection::SetCapacity(size_t capacity) { 
    data_chunks = new DataChunk[capacity];
}

void GPUResultCollection::AddChunk(DataChunk& chunk) {
    num_rows += chunk.size();
    data_chunks[write_idx].Move(chunk);
    write_idx += 1;
}

unique_ptr<DataChunk> GPUResultCollection::GetNext() {
    // We have returned all of the values then return the empty result
    if(read_idx >= write_idx) {
        return nullptr;
    }

    // Create a result that references the value in the buffer
    DataChunk& return_chunk = data_chunks[read_idx];
    unique_ptr<DataChunk> result_value = make_uniq<DataChunk>();

    result_value->InitializeEmpty(return_chunk.GetTypes());
    result_value->SetCardinality(return_chunk.size());
    for(int col = 0; col < return_chunk.data.size(); col++) {
        result_value->data[col].Reference(return_chunk.data[col]);
    }
    read_idx += 1;
    num_rows -= result_value->size();

    return result_value;
}

GPUQueryResult::GPUQueryResult(StatementType statement_type, StatementProperties properties,
    vector<string> names, vector<LogicalType> types, ClientProperties client_properties, unique_ptr<GPUResultCollection> result_collection)
    : QueryResult(QueryResultType::MATERIALIZED_RESULT, statement_type, std::move(properties), types,
std::move(names), std::move(client_properties)), result_collection(std::move(result_collection)) {

}

string GPUQueryResult::ToString() {
    std::string result("GPUQueryResult");
    return result;
}

string GPUQueryResult::ToBox(ClientContext &context, const BoxRendererConfig &config) {
    std::string result("BoxedGPUQueryResult");
    return result;
}

idx_t GPUQueryResult::RowCount() const {
    idx_t row_count = (idx_t) result_collection->size();
    return row_count;
}

unique_ptr<DataChunk> GPUQueryResult::Fetch() {
	return FetchRaw();
}

Value GPUQueryResult::GetValue(idx_t column, idx_t index) {
    throw InternalException("GetValue not implemented for GPUQueryResult");
}

unique_ptr<DataChunk> GPUQueryResult::FetchRaw() {
    return result_collection->GetNext();
}

}