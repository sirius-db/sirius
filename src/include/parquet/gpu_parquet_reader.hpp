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

#include "gpu_columns.hpp"
#include "duckdb/common/common.hpp"

namespace duckdb {

struct GPUParquetReaderOptions {
	string file_path;
	vector<string> projected_columns;
	//! Optional list of row groups to read. Empty -> read all
	vector<int> row_groups;
};

class GPUParquetReader {
public:
	static shared_ptr<GPUIntermediateRelation> ReadFile(const string &file_path);
	static shared_ptr<GPUIntermediateRelation> ReadFile(const string &file_path, const vector<string> &projected_columns);
	static shared_ptr<GPUIntermediateRelation> ReadFile(const GPUParquetReaderOptions &options);
};

} // namespace duckdb

