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

#include "transparent/parquet_footer_statistics.hpp"

#include <duckdb/common/file_system.hpp>
#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <duckdb/common/multi_file/table_function_multi_file.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/storage/external_file_cache/external_file_cache.hpp>
#include <duckdb/storage/statistics/numeric_stats.hpp>
#include <log/logging.hpp>
#include <parquet_multi_file_info.hpp>
#include <parquet_reader.hpp>

#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sirius::transparent {

namespace {

struct file_statistics {
  duckdb::FileMetadata file;
  std::vector<std::pair<duckdb::Identifier, duckdb::unique_ptr<duckdb::BaseStatistics>>> columns;
};

/// Whether \p cached still describes \p current. A modification time is trusted the way DuckDB's
/// own file caches trust it, only once it is old enough that a coarse clock cannot hide a rewrite.
bool unchanged(duckdb::FileMetadata const& cached, duckdb::FileMetadata const& current)
{
  return cached.file_size == current.file_size && cached.file_id == current.file_id &&
         cached.device_id == current.device_id &&
         duckdb::ExternalFileCache::IsValid(/*validate=*/true,
                                            cached.version_tag,
                                            cached.last_modification_time,
                                            current.version_tag,
                                            current.last_modification_time);
}

std::mutex cache_mutex;
std::unordered_map<std::string, std::shared_ptr<file_statistics const>> cache;

std::shared_ptr<file_statistics const> statistics_of(duckdb::ClientContext& context,
                                                     duckdb::OpenFileInfo const& file,
                                                     duckdb::ParquetOptions const& options)
{
  auto& fs           = duckdb::FileSystem::GetFileSystem(context);
  auto const current = fs.OpenFile(file, duckdb::FileFlags::FILE_FLAGS_READ)->Stats();
  {
    std::lock_guard guard(cache_mutex);
    auto found = cache.find(file.path);
    if (found != cache.end() && unchanged(found->second->file, current)) { return found->second; }
  }
  auto result  = std::make_shared<file_statistics>();
  result->file = current;
  duckdb::ParquetReader reader(context, file, options);
  for (auto const& column : reader.columns) {
    auto stats = reader.GetStatistics(context, column.name);
    if (stats) { result->columns.emplace_back(column.name, std::move(stats)); }
  }
  std::lock_guard guard(cache_mutex);
  cache[file.path] = result;
  return result;
}

duckdb::BaseStatistics const* find(file_statistics const& file, duckdb::Identifier const& name)
{
  for (auto const& [column, stats] : file.columns) {
    if (column == name) { return stats.get(); }
  }
  return nullptr;
}

std::string quoted(std::string const& name)
{
  std::string result = "\"";
  for (char c : name) {
    if (c == '"') { result += '"'; }
    result += c;
  }
  return result + '"';
}

/// The `column_statistics` value describing \p bind's columns across \p files, or NULL when the
/// files say nothing the option can carry.
duckdb::Value merged_statistics(duckdb::MultiFileBindData const& bind,
                                std::vector<std::shared_ptr<file_statistics const>> const& files)
{
  auto const text = duckdb::LogicalType::VARCHAR;
  std::vector<duckdb::Value> keys, values;
  for (duckdb::idx_t i = 0; i < bind.names.size(); ++i) {
    if (bind.IsGeneratedColumn(i)) { continue; }
    duckdb::unique_ptr<duckdb::BaseStatistics> merged;
    for (auto const& file : files) {
      auto const* stats = find(*file, bind.names[i]);
      if (!stats || stats->GetType() != bind.types[i]) {
        merged = nullptr;
        break;
      }
      if (merged) {
        merged->Merge(*stats);
      } else {
        merged = stats->ToUnique();
      }
    }
    if (!merged) { continue; }
    std::vector<duckdb::Value> names, given;
    // Without a null count DuckDB keeps the column's NULLs unknown; a zero proves there are none.
    if (!merged->CanHaveNull()) {
      names.emplace_back("null_count");
      given.emplace_back("0");
    }
    if (merged->GetStatsType() == duckdb::StatisticsType::NUMERIC_STATS &&
        duckdb::NumericStats::HasMinMax(*merged)) {
      names.emplace_back("min");
      given.emplace_back(duckdb::NumericStats::Min(*merged).ToString());
      names.emplace_back("max");
      given.emplace_back(duckdb::NumericStats::Max(*merged).ToString());
    }
    if (names.empty()) { continue; }
    keys.emplace_back(quoted(bind.names[i].GetIdentifierName()));
    values.push_back(duckdb::Value::MAP(text, text, std::move(names), std::move(given)));
  }
  if (keys.empty()) { return duckdb::Value(); }
  return duckdb::Value::MAP(
    text, duckdb::LogicalType::MAP(text, text), std::move(keys), std::move(values));
}

bool add_statistics(duckdb::ClientContext& context, duckdb::LogicalGet& get)
{
  auto const& name = get.function.GetName().GetIdentifierName();
  if (name != "read_parquet" && name != "parquet_scan") { return false; }
  auto* bind = dynamic_cast<duckdb::MultiFileBindData*>(get.bind_data.get());
  if (!bind || !bind->file_list || !bind->initial_reader) { return false; }
  auto& options = bind->file_options;
  // DuckDB reads the statistics itself for one file or with union_by_name; a schema or bound
  // schema renames columns, so the footers' names need not be the scan's.
  if (bind->file_list->GetExpandResult() != duckdb::FileExpandResult::MULTIPLE_FILES ||
      options.union_by_name || !options.column_statistics.IsNull() || !options.schema.empty() ||
      !options.bound_schema.empty()) {
    return false;
  }
  auto const* reader =
    dynamic_cast<duckdb::TableFunctionFileReader const*>(bind->initial_reader.get());
  if (!reader || !reader->bind_data) { return false; }
  auto const& parquet_options = duckdb::ParquetScanFunction::GetFileOptions(*reader->bind_data);
  auto const all_files        = bind->file_list->GetAllFiles();
  std::vector<std::shared_ptr<file_statistics const>> files;
  files.reserve(all_files.size());
  for (auto const& file : all_files) {
    if (duckdb::FileSystem::IsRemoteFile(file.path)) { return false; }
    files.push_back(statistics_of(context, file, parquet_options));
  }
  auto statistics = merged_statistics(*bind, files);
  if (statistics.IsNull()) { return false; }
  options.column_statistics = std::move(statistics);
  options.VerifyColumnStatistics(bind->names, bind->types);
  return true;
}

void visit(duckdb::ClientContext& context, duckdb::LogicalOperator& op, std::size_t& added)
{
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
    auto& get = op.Cast<duckdb::LogicalGet>();
    try {
      if (add_statistics(context, get)) { ++added; }
    } catch (std::exception const& e) {
      if (auto* bind = dynamic_cast<duckdb::MultiFileBindData*>(get.bind_data.get())) {
        bind->file_options.column_statistics = duckdb::Value();
      }
      SIRIUS_LOG_DEBUG("No footer statistics for a Parquet scan: {}", e.what());
    }
  }
  for (auto& child : op.children) {
    visit(context, *child, added);
  }
}

}  // namespace

std::size_t add_parquet_footer_statistics(duckdb::ClientContext& context,
                                          duckdb::LogicalOperator& plan)
{
  std::size_t added = 0;
  visit(context, plan, added);
  return added;
}

std::size_t read_parquet_footer_statistics(duckdb::ClientContext& context,
                                           std::vector<std::string> const& paths)
{
  duckdb::ParquetOptions const options(context);
  std::size_t cached = 0;
  for (auto const& path : paths) {
    if (duckdb::FileSystem::IsRemoteFile(path)) { continue; }
    statistics_of(context, duckdb::OpenFileInfo(path), options);
    ++cached;
  }
  return cached;
}

}  // namespace sirius::transparent
