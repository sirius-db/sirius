/*
 * Copyright 2026, Sirius Contributors.
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

#include "storage/ducklake_multi_file_list.hpp"
#include "storage/ducklake_multi_file_reader.hpp"
#include "storage/ducklake_transaction.hpp"

#include <cxxabi.h>
#include <sys/types.h>

#include <memory>
#include <string>

namespace sirius::ducklake {

// Only the pinned Linux provider exports these private C++ types and operations.
struct provider_api {
  static std::shared_ptr<const provider_api> resolve(duckdb::DatabaseInstance& db);
  bool matches_loaded_file() const;

  duckdb::DuckLakeFunctionInfo const* function_info(duckdb::TableFunctionInfo const*) const;
  duckdb::DuckLakeMultiFileReader const* reader(duckdb::MultiFileReader const*) const;
  duckdb::DuckLakeMultiFileList const* file_list(duckdb::MultiFileList const*) const;

  decltype(&duckdb::DuckLakeFunctions::GetDuckLakeScanFunction) scan_function;
  decltype(&duckdb::DuckLakeMultiFileReader::CreateInstance) create_reader;
  decltype(&duckdb::DuckLakeMultiFileList::GetFiles) get_files;
  decltype(&duckdb::DuckLakeMultiFileList::SetScanColumns) set_scan_columns;
  decltype(&duckdb::DuckLakeTransaction::HasLocalDeleteForFile) has_local_delete;

 private:
  std::shared_ptr<void> module;
  std::string path;
  dev_t device;
  ino_t inode;
  __cxxabiv1::__class_type_info const* info_type;
  __cxxabiv1::__class_type_info const* reader_type;
  __cxxabiv1::__class_type_info const* list_type;
};
}  // namespace sirius::ducklake
