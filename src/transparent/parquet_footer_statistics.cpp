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

#include "op/scan/table_scan/bound_read_view.hpp"

#include <duckdb/common/file_system.hpp>
#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <duckdb/common/multi_file/table_function_multi_file.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/parallel/task_executor.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/storage/external_file_cache/external_file_cache.hpp>
#include <duckdb/storage/statistics/numeric_stats.hpp>
#include <log/logging.hpp>
#include <parquet_multi_file_info.hpp>
#include <parquet_reader.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sirius::transparent {

namespace {

struct file_statistics {
  std::vector<std::pair<duckdb::Identifier, duckdb::unique_ptr<duckdb::BaseStatistics>>> columns;
};

bool is_strong_etag(std::string_view tag)
{
  if (tag.size() < 2 || tag.front() != '"' || tag.back() != '"') { return false; }
  return std::ranges::none_of(tag.substr(1, tag.size() - 2),
                              [](unsigned char c) { return c == '"' || c < 0x21 || c == 0x7F; });
}

/// Text that is equal for two sightings of \p path only if they saw the same bytes, or nullopt when
/// \p metadata cannot promise that. A remote file needs a strong ETag: a weak one allows different
/// bytes, and a remote modification time may be absent or coarse. A local file's version tag holds
/// its device, inode, size and modification time, trusted the way DuckDB's own file caches trust
/// it: only once the time is old enough that a coarse clock cannot hide a rewrite.
std::optional<std::string> version_of(std::string const& path, duckdb::FileMetadata const& metadata)
{
  if (metadata.file_size < 0 || metadata.version_tag.empty()) { return std::nullopt; }
  if (duckdb::FileSystem::IsRemoteFile(path)) {
    if (!is_strong_etag(metadata.version_tag)) { return std::nullopt; }
  } else if (!duckdb::ExternalFileCache::IsValid(/*validate=*/true,
                                                 /*cached_version_tag=*/"",
                                                 metadata.last_modification_time,
                                                 /*current_version_tag=*/"",
                                                 metadata.last_modification_time)) {
    return std::nullopt;
  }
  return std::to_string(metadata.file_size) + ':' + metadata.version_tag;
}

/// The metadata the glob that listed \p file attached to it, if it attached a version tag. Both
/// DuckDB's local file system and Sirius's S3 file system attach one, so validating a cached entry
/// usually costs no request.
std::optional<duckdb::FileMetadata> listed_metadata(duckdb::OpenFileInfo const& file)
{
  if (!file.extended_info) { return std::nullopt; }
  auto const& listed = file.extended_info->options;
  auto const size    = listed.find("file_size");
  auto const tag     = listed.find("etag");
  if (size == listed.end() || size->second.IsNull() || tag == listed.end() ||
      tag->second.IsNull()) {
    return std::nullopt;
  }
  duckdb::FileMetadata metadata;
  metadata.file_size   = size->second.GetValue<int64_t>();
  metadata.version_tag = duckdb::StringValue::Get(tag->second);
  if (auto const modified = listed.find("last_modified");
      modified != listed.end() && !modified->second.IsNull()) {
    metadata.last_modification_time = modified->second.GetValue<duckdb::timestamp_t>();
  }
  return metadata;
}

struct cache_entry {
  std::string version;
  std::shared_ptr<file_statistics const> statistics;
};

// Keyed by the reader options as well as the path: options such as can_have_nan change the
// statistics a footer reads as without changing the column's type.
std::mutex cache_mutex;
std::unordered_map<std::string, cache_entry> cache;

std::string cache_key(std::string const& options_identity, std::string const& path)
{
  auto key = options_identity;
  key += '\0';
  key += path;
  return key;
}

std::shared_ptr<file_statistics const> cached(std::string const& key,
                                              std::optional<std::string> const& version)
{
  if (!version) { return nullptr; }
  std::lock_guard guard(cache_mutex);
  auto found = cache.find(key);
  if (found == cache.end() || found->second.version != *version) { return nullptr; }
  return found->second.statistics;
}

/// The statistics of \p file, read from its footer unless the cache holds them for the version
/// \p listed names.
std::shared_ptr<file_statistics const> statistics_of(
  duckdb::ClientContext& context,
  duckdb::OpenFileInfo const& file,
  duckdb::ParquetOptions const& options,
  std::string const& key,
  std::optional<duckdb::FileMetadata> const& listed)
{
  auto version = listed ? version_of(file.path, *listed) : std::nullopt;
  if (!listed) {
    // A local file's metadata is one stat; opening it would cost an open and a close.
    auto& fs            = duckdb::FileSystem::GetFileSystem(context);
    auto const metadata = fs.GetStatsIfExists(file);
    if (metadata) { version = version_of(file.path, *metadata); }
    if (auto hit = cached(key, version)) { return hit; }
  }
  auto result = std::make_shared<file_statistics>();
  duckdb::ParquetReader reader(context, file, options);
  for (auto const& column : reader.columns) {
    auto stats = reader.GetStatistics(context, column.name);
    if (stats) { result->columns.emplace_back(column.name, std::move(stats)); }
  }
  // The version describes the file as listed or opened before this read, so a rewrite between the
  // two leaves an entry that the next sighting of the new version misses.
  if (version) {
    std::lock_guard guard(cache_mutex);
    cache[key] = {std::move(*version), result};
  }
  return result;
}

class footer_read_task : public duckdb::BaseExecutorTask {
 public:
  footer_read_task(duckdb::TaskExecutor& executor, std::function<void()> read)
    : duckdb::BaseExecutorTask(executor), read_(std::move(read))
  {
  }

  void ExecuteTask() override { read_(); }

 private:
  std::function<void()> read_;
};

/// statistics_of for each of \p files, in order. Footers that must be read are read on DuckDB's
/// worker threads, since each may cost an object-store request. Rethrows the first failure.
std::vector<std::shared_ptr<file_statistics const>> statistics_of_all(
  duckdb::ClientContext& context,
  std::vector<duckdb::OpenFileInfo> const& files,
  duckdb::ParquetOptions const& options)
{
  std::vector<std::shared_ptr<file_statistics const>> result(files.size());
  auto const options_identity = op::scan::parquet_options_identity(options);
  std::vector<std::size_t> unresolved;
  std::vector<std::string> keys(files.size());
  std::vector<std::optional<duckdb::FileMetadata>> listed(files.size());
  auto& fs = duckdb::FileSystem::GetFileSystem(context);
  for (std::size_t i = 0; i < files.size(); ++i) {
    keys[i]   = cache_key(options_identity, files[i].path);
    listed[i] = listed_metadata(files[i]);
    if (listed[i]) {
      result[i] = cached(keys[i], version_of(files[i].path, *listed[i]));
    } else if (!duckdb::FileSystem::IsRemoteFile(files[i].path)) {
      // A local file costs one stat to check, less than scheduling a task for it; only a miss,
      // which reads a footer, goes to the executor.
      if (auto const metadata = fs.GetStatsIfExists(files[i])) {
        result[i] = cached(keys[i], version_of(files[i].path, *metadata));
      }
    }
    if (!result[i]) { unresolved.push_back(i); }
  }
  auto read = [&](std::size_t i) {
    result[i] = statistics_of(context, files[i], options, keys[i], listed[i]);
  };
  if (unresolved.size() == 1) {
    read(unresolved.front());
  } else if (!unresolved.empty()) {
    duckdb::TaskExecutor executor(context);
    for (auto i : unresolved) {
      executor.ScheduleTask(duckdb::make_uniq<footer_read_task>(executor, [&read, i] { read(i); }));
    }
    executor.WorkOnTasks();
  }
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
  auto const files            = statistics_of_all(context, all_files, parquet_options);
  auto statistics             = merged_statistics(*bind, files);
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
  std::vector<duckdb::OpenFileInfo> const files(paths.begin(), paths.end());
  return statistics_of_all(context, files, options).size();
}

}  // namespace sirius::transparent
