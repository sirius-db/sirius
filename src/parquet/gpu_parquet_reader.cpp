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

#include "parquet/gpu_parquet_reader.hpp"

#include <algorithm>
#include <filesystem>

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/string_util.hpp"
#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"

namespace duckdb {

namespace {

string NormalizeTableName(const string &file_path) {
	namespace fs = std::filesystem;
	string candidate;
	try {
		fs::path path_obj(file_path);
		candidate = path_obj.filename().string();
	} catch (...) {
		candidate = file_path;
	}
	if (candidate.empty()) {
		candidate = file_path;
	}
	return StringUtil::Upper(candidate);
}

string NormalizeColumnName(const cudf::io::table_metadata &metadata, idx_t column_idx) {
	string candidate;
	if (column_idx < metadata.schema_info.size()) {
		candidate = metadata.schema_info[column_idx].name;
	}
	if (candidate.empty()) {
		candidate = "COLUMN_" + to_string(column_idx);
	}
	return StringUtil::Upper(candidate);
}

} // namespace

shared_ptr<GPUIntermediateRelation> GPUParquetReader::ReadFile(const string &file_path) {
	GPUParquetReaderOptions options;
	options.file_path = file_path;
	return ReadFile(options);
}

shared_ptr<GPUIntermediateRelation> GPUParquetReader::ReadFile(const string &file_path, const vector<string> &projected_columns) {
	GPUParquetReaderOptions options;
	options.file_path = file_path;
	options.projected_columns = projected_columns;
	return ReadFile(options);
}

shared_ptr<GPUIntermediateRelation> GPUParquetReader::ReadFile(const GPUParquetReaderOptions &options) {
	if (options.file_path.empty()) {
		throw InvalidInputException("GPUParquetReader requires a non-empty file path");
	}

	SIRIUS_LOG_INFO("GPUParquetReader reading file {}", options.file_path);
	auto source = cudf::io::source_info(options.file_path);
	auto builder = cudf::io::parquet_reader_options::builder(source);
	if (!options.projected_columns.empty()) {
		builder.columns(options.projected_columns);
	}
	if (!options.row_groups.empty()) {
		std::vector<std::vector<cudf::size_type>> row_group_filters;
		row_group_filters.emplace_back();
		row_group_filters.back().reserve(options.row_groups.size());
		for (auto rg : options.row_groups) {
			row_group_filters.back().push_back(static_cast<cudf::size_type>(rg));
		}
		builder.row_groups(std::move(row_group_filters));
	}
	auto reader_options = builder.build();

	auto table_with_metadata = cudf::io::read_parquet(reader_options);
	auto cudf_table = std::move(table_with_metadata.tbl);
	auto metadata = std::move(table_with_metadata.metadata);

	if (!cudf_table) {
		throw IOException("Failed to read Parquet file: %s", options.file_path);
	}

	auto num_columns = static_cast<idx_t>(cudf_table->num_columns());
	auto &gpu_buffer_manager = GPUBufferManager::GetInstance();
	auto gpu_relation = make_shared_ptr<GPUIntermediateRelation>(num_columns);
	gpu_relation->names = NormalizeTableName(options.file_path);

	auto released_columns = cudf_table->release();
	for (idx_t col_idx = 0; col_idx < num_columns; col_idx++) {
		auto &column_ptr = released_columns[col_idx];
		if (!column_ptr) {
			throw IOException("Missing column index %lld while reading %s", col_idx, options.file_path);
		}
		auto gpu_column = make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::INT32), nullptr, nullptr);
		gpu_column->setFromCudfColumn(*column_ptr, false, nullptr, 0, &gpu_buffer_manager);
		gpu_relation->columns[col_idx] = std::move(gpu_column);
		gpu_relation->column_names[col_idx] = NormalizeColumnName(metadata, col_idx);
	}

	SIRIUS_LOG_INFO("GPUParquetReader loaded {} columns from {}", num_columns, options.file_path);
	return gpu_relation;
}

} // namespace duckdb

