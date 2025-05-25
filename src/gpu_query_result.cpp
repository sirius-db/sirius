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
    std::cout << "AddChunk: After writing chunk to idx " << write_idx << " got num rows of " << num_rows << std::endl;
    write_idx += 1;
}

unique_ptr<DataChunk> GPUResultCollection::GetNext() {
    // We have returned all of the values then return the empty result
    unique_ptr<DataChunk> result_value = make_uniq<DataChunk>();
    if(read_idx >= write_idx) {
        return nullptr;
    }

    // Create a result that references the value in the buffer
    DataChunk& return_chunk = data_chunks[read_idx];
    result_value->Reference(return_chunk);
    read_idx += 1;

    std::cout << "GPU Result Collection: Returning chunk with row count of " << result_value->size() << " for idx " << read_idx << std::endl;
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
    std::cout << "GPU Query Result: Returning row count of " << row_count << std::endl;
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