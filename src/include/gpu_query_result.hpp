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

#pragma once

#include "helper/common.h"
#include "utils.hpp"
#include "duckdb/common/winapi.hpp"
#include "duckdb/main/query_result.hpp"

namespace duckdb {

class GPUResultCollection { 
public: 
	GPUResultCollection() : read_idx(0), write_idx(0), num_rows(0), data_chunks(nullptr) {

	}

	void SetCapacity(size_t capacity);
	void AddChunk(DataChunk& chunk);
	unique_ptr<DataChunk> GetNext();

	size_t size() {
		return num_rows;
	}

	~GPUResultCollection() {
		if(data_chunks != nullptr) {
			delete[] data_chunks;
		}
	}

	DataChunk* data_chunks;
	size_t num_rows;
	size_t write_idx;
	size_t read_idx;
};

// The reason we need to implement our own QueryResult is that duckdb's MaterializedQueryResult
// in constructed by passing in a ColumnDataCollection which makes a copy of the data when we try
// to Append DataChunks to it. By implement our own basic QueryResult we can bypass this unncessary
// copy
class GPUQueryResult : public QueryResult {
public:
    DUCKDB_API GPUQueryResult(StatementType statement_type, StatementProperties properties, vector<string> names, 
		vector<LogicalType> types, ClientProperties client_properties, unique_ptr<GPUResultCollection> result_collection);
    
        //! Fetches a DataChunk from the query result.
	//! This will consume the result (i.e. the result can only be scanned once with this function)
	DUCKDB_API unique_ptr<DataChunk> Fetch() override;
	DUCKDB_API unique_ptr<DataChunk> FetchRaw() override;
	//! Converts the QueryResult to a string
	DUCKDB_API string ToString() override;
	DUCKDB_API string ToBox(ClientContext &context, const BoxRendererConfig &config) override;

	//! Gets the (index) value of the (column index) column.
	//! Note: this is very slow. Scanning over the underlying collection is much faster.
	DUCKDB_API Value GetValue(idx_t column, idx_t index);

	DUCKDB_API idx_t RowCount() const;

private:
    // The actual chunks we want to return
    unique_ptr<GPUResultCollection> result_collection;
};

}