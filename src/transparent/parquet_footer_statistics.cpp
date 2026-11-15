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
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sirius::transparent {

namespace {

std::atomic<std::size_t> live_records{0};
std::atomic<std::uint64_t> next_record_id{1};
std::atomic<std::uint64_t> footers_read{0};

/// What one footer says. Every read makes a record with an id of its own, so equal ids mean the
/// same statistics.
struct file_statistics {
  file_statistics() : id(next_record_id.fetch_add(1, std::memory_order_relaxed))
  {
    live_records.fetch_add(1, std::memory_order_relaxed);
  }
  ~file_statistics() { live_records.fetch_sub(1, std::memory_order_relaxed); }
  file_statistics(file_statistics const&)            = delete;
  file_statistics& operator=(file_statistics const&) = delete;

  std::uint64_t id;
  std::size_t num_rows = 0;
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
  std::list<std::string const*>::iterator position;
};

/// The merged, verified statistics a scan was last given, with the ids of the per-file records they
/// were merged from. The entry fits a scan only while the cache answers each of its files with the
/// same record. It holds ids, not the records, so it keeps none of them alive.
struct merge_entry {
  std::vector<std::uint64_t> files;
  decltype(duckdb::MultiFileBindData::names) names;
  decltype(duckdb::MultiFileBindData::types) types;
  duckdb::Value statistics;
};

// One lock for the three caches: what each may keep is bounded together with the others.
std::mutex cache_mutex;
detail::cache_capacities capacities = detail::default_cache_capacities;
// Keyed by the reader options as well as the path: options such as can_have_nan change the
// statistics a footer reads as without changing the column's type.
std::unordered_map<std::string, cache_entry> cache;
/// The keys of the cache, least recently used first.
std::list<std::string const*> cache_order;
/// Keyed by the paths the scan was given, so a glob that gains files replaces its entry instead of
/// adding one; the entry itself says whether it fits the scan at hand.
std::unordered_map<std::string, merge_entry> merges;
/// The file ids the merged entries hold together.
std::size_t merge_references = 0;
/// The footer row total of one scan of one planning. The plan and the copies made of it share it.
struct row_total {
  std::size_t rows;
  std::uint64_t sequence;
  std::size_t members = 0;
  bool withdrawn      = false;
};

struct row_count_entry {
  duckdb::weak_ptr<duckdb::MultiFileList> owner;
  std::shared_ptr<row_total> total;
};

/// Keyed by the file list of a scan's bind data, which a plan owns and each plan copy has anew, so
/// a total answers only for the plan it was published or copied for. The weak owner tells a list
/// that is gone from a new one at its address.
std::unordered_map<duckdb::MultiFileList const*, row_count_entry> row_counts;
std::uint64_t next_row_total_sequence = 0;
/// The totals some live entry still answers with; bounded like the merged statistics.
std::size_t live_row_totals = 0;

void evict_statistics_locked()
{
  while (cache.size() > capacities.statistics) {
    auto const* key = cache_order.front();
    cache_order.pop_front();
    cache.erase(cache.find(*key));
  }
}

void erase_merge_locked(std::unordered_map<std::string, merge_entry>::iterator entry)
{
  merge_references -= entry->second.files.size();
  merges.erase(entry);
}

void evict_merges_locked()
{
  while (!merges.empty() &&
         (merges.size() > capacities.merges || merge_references > capacities.merge_references)) {
    erase_merge_locked(merges.begin());
  }
}

void erase_row_count_locked(
  std::unordered_map<duckdb::MultiFileList const*, row_count_entry>::iterator entry)
{
  auto& total = *entry->second.total;
  if (--total.members == 0 && !total.withdrawn) { --live_row_totals; }
  row_counts.erase(entry);
}

/// Drops the entries whose plan is gone or whose total was withdrawn.
void purge_row_counts_locked()
{
  for (auto entry = row_counts.begin(); entry != row_counts.end();) {
    auto const stale = entry++;
    if (stale->second.total->withdrawn || stale->second.owner.expired()) {
      erase_row_count_locked(stale);
    }
  }
}

/// Withdraws the oldest totals until at most \p limit remain; their plans then have no row count.
void withdraw_row_totals_locked(std::size_t limit)
{
  while (live_row_totals > limit) {
    row_total* oldest = nullptr;
    for (auto const& [list, entry] : row_counts) {
      auto& total = *entry.total;
      if (!total.withdrawn && (!oldest || total.sequence < oldest->sequence)) { oldest = &total; }
    }
    if (!oldest) {
      live_row_totals = 0;
      return;
    }
    oldest->withdrawn = true;
    --live_row_totals;
  }
}

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
  cache_order.splice(cache_order.end(), cache_order, found->second.position);
  return found->second.statistics;
}

/// Keeps \p statistics as the answer for \p key at \p version, in place of an older version, and
/// drops the least recently used records beyond the capacity.
void remember(std::string const& key,
              std::string version,
              std::shared_ptr<file_statistics const> statistics)
{
  std::lock_guard guard(cache_mutex);
  if (capacities.statistics == 0) { return; }
  auto const [entry, inserted] = cache.try_emplace(key);
  if (inserted) {
    try {
      entry->second.position = cache_order.insert(cache_order.end(), &entry->first);
    } catch (...) {
      cache.erase(entry);
      throw;
    }
  } else {
    cache_order.splice(cache_order.end(), cache_order, entry->second.position);
  }
  entry->second.version    = std::move(version);
  entry->second.statistics = std::move(statistics);
  evict_statistics_locked();
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
  footers_read.fetch_add(1, std::memory_order_relaxed);
  duckdb::ParquetReader reader(context, file, options);
  result->num_rows = reader.NumRows();
  for (auto const& column : reader.columns) {
    auto stats = reader.GetStatistics(context, column.name);
    if (stats) { result->columns.emplace_back(column.name, std::move(stats)); }
  }
  // The version describes the file as listed or opened before this read, so a rewrite between the
  // two leaves an entry that the next sighting of the new version misses.
  if (version) { remember(key, std::move(*version), result); }
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

/// The files of one scan on the way to their statistics: what the cache answered, and what is left
/// to read.
struct scan_statistics {
  std::vector<std::shared_ptr<file_statistics const>> files;
  std::vector<std::string> keys;
  std::vector<std::optional<duckdb::FileMetadata>> listed;
  std::vector<std::size_t> unresolved;
  std::size_t remote_reads = 0;
  std::size_t local_reads  = 0;
};

/// The cached statistics of \p files, in order, without reading a footer. The rest are counted as
/// reads. A remote file the listing did not describe is one of them even if it is cached: learning
/// its version takes a request, which is what the count is there to bound.
scan_statistics cached_statistics(duckdb::ClientContext& context,
                                  std::vector<duckdb::OpenFileInfo> const& files,
                                  duckdb::ParquetOptions const& options)
{
  scan_statistics result;
  result.files.resize(files.size());
  result.keys.resize(files.size());
  result.listed.resize(files.size());
  auto const options_identity = op::scan::parquet_options_identity(options);
  auto& fs                    = duckdb::FileSystem::GetFileSystem(context);
  for (std::size_t i = 0; i < files.size(); ++i) {
    bool const remote = duckdb::FileSystem::IsRemoteFile(files[i].path);
    result.keys[i]    = cache_key(options_identity, files[i].path);
    result.listed[i]  = listed_metadata(files[i]);
    if (result.listed[i]) {
      result.files[i] = cached(result.keys[i], version_of(files[i].path, *result.listed[i]));
    } else if (!remote) {
      // A local file costs one stat to check, less than scheduling a task for it; only a miss,
      // which reads a footer, goes to the executor.
      if (auto const metadata = fs.GetStatsIfExists(files[i])) {
        result.files[i] = cached(result.keys[i], version_of(files[i].path, *metadata));
      }
    }
    if (!result.files[i]) {
      result.unresolved.push_back(i);
      ++(remote ? result.remote_reads : result.local_reads);
    }
  }
  return result;
}

/// statistics_of for each file \p scan has not resolved. Footers that must be read are read on
/// DuckDB's worker threads, since each may cost an object-store request. Rethrows the first
/// failure.
void read_unresolved(duckdb::ClientContext& context,
                     std::vector<duckdb::OpenFileInfo> const& files,
                     duckdb::ParquetOptions const& options,
                     scan_statistics& scan)
{
  auto read = [&](std::size_t i) {
    scan.files[i] = statistics_of(context, files[i], options, scan.keys[i], scan.listed[i]);
  };
  if (scan.unresolved.size() == 1) {
    read(scan.unresolved.front());
  } else if (!scan.unresolved.empty()) {
    duckdb::TaskExecutor executor(context);
    for (auto i : scan.unresolved) {
      executor.ScheduleTask(duckdb::make_uniq<footer_read_task>(executor, [&read, i] { read(i); }));
    }
    executor.WorkOnTasks();
  }
  scan.unresolved.clear();
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

std::string merge_cache_key(duckdb::LogicalGet const& get)
{
  std::string key;
  for (auto const& parameter : get.parameters) {
    key += parameter.ToString();
    key += '\0';
  }
  return key;
}

std::optional<duckdb::Value> cached_merge(std::string const& key,
                                          scan_statistics const& scan,
                                          duckdb::MultiFileBindData const& bind)
{
  std::lock_guard guard(cache_mutex);
  auto const found = merges.find(key);
  if (found == merges.end() || found->second.names != bind.names ||
      found->second.types != bind.types ||
      !std::ranges::equal(
        found->second.files, scan.files, {}, {}, [](auto const& file) { return file->id; })) {
    return std::nullopt;
  }
  return found->second.statistics;
}

/// Keeps \p statistics for the next planning of the scan, if it can answer one: every record must
/// be the one the cache holds for its file, and the entry must fit beside the others.
void remember_merge(std::string const& key,
                    scan_statistics const& scan,
                    duckdb::MultiFileBindData const& bind,
                    duckdb::Value const& statistics)
{
  std::lock_guard guard(cache_mutex);
  if (auto const old = merges.find(key); old != merges.end()) { erase_merge_locked(old); }
  if (capacities.merges == 0 || scan.files.size() > capacities.merge_references) { return; }
  for (std::size_t i = 0; i < scan.files.size(); ++i) {
    auto const found = cache.find(scan.keys[i]);
    if (found == cache.end() || found->second.statistics != scan.files[i]) { return; }
  }
  // A miss only costs a merge, so any entry may make room for a new set of paths.
  while (!merges.empty() && (merges.size() >= capacities.merges ||
                             merge_references + scan.files.size() > capacities.merge_references)) {
    erase_merge_locked(merges.begin());
  }
  merge_entry entry{{}, bind.names, bind.types, statistics};
  entry.files.reserve(scan.files.size());
  for (auto const& file : scan.files) {
    entry.files.push_back(file->id);
  }
  merges.emplace(key, std::move(entry));
  merge_references += scan.files.size();
}

bool within(std::int64_t limit, std::size_t reads)
{
  return limit < 0 || reads <= static_cast<std::size_t>(limit);
}

bool is_parquet_scan(duckdb::LogicalGet const& get)
{
  auto const& name = get.function.GetName().GetIdentifierName();
  return name == "read_parquet" || name == "parquet_scan";
}

duckdb::shared_ptr<duckdb::MultiFileList> const* file_list_of(duckdb::LogicalGet const& get)
{
  if (!is_parquet_scan(get)) { return nullptr; }
  auto const* bind = dynamic_cast<duckdb::MultiFileBindData const*>(get.bind_data.get());
  return bind && bind->file_list ? &bind->file_list : nullptr;
}

bool answers(row_count_entry const& entry, duckdb::MultiFileList const* list)
{
  return !entry.total->withdrawn && entry.owner.lock().get() == list;
}

void forget_row_count(duckdb::MultiFileList const* list)
{
  std::lock_guard guard(cache_mutex);
  if (auto const found = row_counts.find(list); found != row_counts.end()) {
    erase_row_count_locked(found);
  }
}

void remember_row_count(duckdb::shared_ptr<duckdb::MultiFileList> const& list, std::size_t rows)
{
  std::lock_guard guard(cache_mutex);
  if (auto const found = row_counts.find(list.get()); found != row_counts.end()) {
    erase_row_count_locked(found);
  }
  purge_row_counts_locked();
  if (capacities.merges == 0) { return; }
  withdraw_row_totals_locked(capacities.merges - 1);
  auto total       = std::make_shared<row_total>(row_total{rows, next_row_total_sequence});
  auto const entry = row_counts.try_emplace(list.get()).first;
  // Nothing below throws: the entry and the counters change together.
  entry->second.owner          = duckdb::weak_ptr<duckdb::MultiFileList>(list);
  entry->second.total          = std::move(total);
  entry->second.total->members = 1;
  ++next_row_total_sequence;
  ++live_row_totals;
}

bool add_statistics(duckdb::ClientContext& context,
                    duckdb::LogicalGet& get,
                    footer_read_limits const& limits)
{
  if (!is_parquet_scan(get)) { return false; }
  auto* bind = dynamic_cast<duckdb::MultiFileBindData*>(get.bind_data.get());
  if (!bind || !bind->file_list) { return false; }
  // Unknown until this planning has covered every file of the scan.
  forget_row_count(bind->file_list.get());
  if (!bind->initial_reader) { return false; }
  auto& options = bind->file_options;
  // DuckDB reads the statistics itself for one file or with union_by_name; a schema or bound
  // schema renames columns, so the footers' names need not be the scan's. A one-file scan still
  // gets its footer read, for the row count.
  auto const expand      = bind->file_list->GetExpandResult();
  bool const single_file = expand == duckdb::FileExpandResult::SINGLE_FILE;
  if ((expand != duckdb::FileExpandResult::MULTIPLE_FILES && !single_file) ||
      options.union_by_name || (!single_file && !options.column_statistics.IsNull()) ||
      !options.schema.empty() || !options.bound_schema.empty()) {
    return false;
  }
  auto const* reader =
    dynamic_cast<duckdb::TableFunctionFileReader const*>(bind->initial_reader.get());
  if (!reader || !reader->bind_data) { return false; }
  auto const& parquet_options = duckdb::ParquetScanFunction::GetFileOptions(*reader->bind_data);
  auto const all_files        = bind->file_list->GetAllFiles();
  auto scan                   = cached_statistics(context, all_files, parquet_options);
  // All or nothing: the statistics of some files say nothing true about the scan, and the
  // optimizer takes what it is given as fact.
  if (!within(limits.remote, scan.remote_reads) || !within(limits.local, scan.local_reads)) {
    SIRIUS_LOG_DEBUG(
      "No footer statistics for a Parquet scan of {} files: {} remote and {} local footers to "
      "read, limits {} and {}",
      all_files.size(),
      scan.remote_reads,
      scan.local_reads,
      limits.remote,
      limits.local);
    return false;
  }
  read_unresolved(context, all_files, parquet_options, scan);
  std::size_t total_rows = 0;
  for (auto const& file : scan.files) {
    total_rows += file->num_rows;
  }
  remember_row_count(bind->file_list, total_rows);
  if (single_file) { return false; }
  auto const scan_key = merge_cache_key(get);
  if (auto const hit = cached_merge(scan_key, scan, *bind)) {
    options.column_statistics = *hit;
  } else {
    auto statistics = merged_statistics(*bind, scan.files);
    if (statistics.IsNull()) { return false; }
    options.column_statistics = std::move(statistics);
    options.VerifyColumnStatistics(bind->names, bind->types);
    remember_merge(scan_key, scan, *bind, options.column_statistics);
  }
  return true;
}

void visit(duckdb::ClientContext& context,
           duckdb::LogicalOperator& op,
           footer_read_limits const& limits,
           std::size_t& added)
{
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
    auto& get           = op.Cast<duckdb::LogicalGet>();
    auto* bind          = dynamic_cast<duckdb::MultiFileBindData*>(get.bind_data.get());
    bool const supplied = bind && !bind->file_options.column_statistics.IsNull();
    try {
      if (add_statistics(context, get, limits)) { ++added; }
    } catch (std::exception const& e) {
      // Withdraw what this call installed; statistics the caller supplied stay.
      if (bind && !supplied) { bind->file_options.column_statistics = duckdb::Value(); }
      if (bind && bind->file_list) { forget_row_count(bind->file_list.get()); }
      SIRIUS_LOG_DEBUG("No footer statistics for a Parquet scan: {}", e.what());
    }
  }
  for (auto& child : op.children) {
    visit(context, *child, limits, added);
  }
}

}  // namespace

std::size_t add_parquet_footer_statistics(duckdb::ClientContext& context,
                                          duckdb::LogicalOperator& plan)
{
  std::size_t added = 0;
  visit(context, plan, footer_read_limits_of(context), added);
  return added;
}

std::optional<std::size_t> parquet_scan_row_count(duckdb::LogicalGet const& get)
{
  auto const* list = file_list_of(get);
  if (!list) { return std::nullopt; }
  std::lock_guard guard(cache_mutex);
  auto const found = row_counts.find(list->get());
  if (found == row_counts.end() || !answers(found->second, list->get())) { return std::nullopt; }
  return found->second.total->rows;
}

void forget_parquet_scan_row_counts(duckdb::LogicalOperator const& plan)
{
  if (plan.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
    if (auto const* list = file_list_of(plan.Cast<duckdb::LogicalGet>())) {
      forget_row_count(list->get());
    }
  }
  for (auto const& child : plan.children) {
    forget_parquet_scan_row_counts(*child);
  }
}

void copy_parquet_scan_row_count(duckdb::LogicalGet const& from, duckdb::LogicalGet const& to)
{
  auto const* source = file_list_of(from);
  auto const* target = file_list_of(to);
  if (!source || !target || source->get() == target->get()) { return; }
  try {
    std::lock_guard guard(cache_mutex);
    auto const found = row_counts.find(source->get());
    if (found == row_counts.end() || !answers(found->second, source->get())) { return; }
    auto total = found->second.total;
    if (auto const old = row_counts.find(target->get()); old != row_counts.end()) {
      erase_row_count_locked(old);
    }
    // Copies are made per execution; entries of plans that are gone leave here, not only at the
    // next planning.
    if (row_counts.size() > 4 * capacities.merges + 16) { purge_row_counts_locked(); }
    auto const entry    = row_counts.try_emplace(target->get()).first;
    entry->second.owner = duckdb::weak_ptr<duckdb::MultiFileList>(*target);
    entry->second.total = std::move(total);
    ++entry->second.total->members;
  } catch (std::bad_alloc const&) {
    // The copy plans without the count.
  }
}

std::size_t read_parquet_footer_statistics(duckdb::ClientContext& context,
                                           std::vector<std::string> const& paths)
{
  duckdb::ParquetOptions const options(context);
  std::vector<duckdb::OpenFileInfo> const files(paths.begin(), paths.end());
  auto scan = cached_statistics(context, files, options);
  read_unresolved(context, files, options, scan);
  return scan.files.size();
}

std::size_t detail::merged_statistics_cache_size()
{
  std::lock_guard guard(cache_mutex);
  return merges.size();
}

std::size_t detail::row_count_cache_size()
{
  std::lock_guard guard(cache_mutex);
  purge_row_counts_locked();
  return live_row_totals;
}

std::size_t detail::statistics_cache_size()
{
  std::lock_guard guard(cache_mutex);
  return cache.size();
}

std::size_t detail::merged_reference_count()
{
  std::lock_guard guard(cache_mutex);
  return merge_references;
}

std::size_t detail::live_statistics_records()
{
  return live_records.load(std::memory_order_relaxed);
}

std::uint64_t detail::footer_read_count() { return footers_read.load(std::memory_order_relaxed); }

detail::cache_capacities detail::set_cache_capacities_for_testing(cache_capacities replacement)
{
  std::lock_guard guard(cache_mutex);
  auto const previous = capacities;
  capacities          = replacement;
  evict_statistics_locked();
  evict_merges_locked();
  purge_row_counts_locked();
  withdraw_row_totals_locked(capacities.merges);
  return previous;
}

void detail::clear_caches_for_testing()
{
  std::lock_guard guard(cache_mutex);
  cache.clear();
  cache_order.clear();
  merges.clear();
  merge_references = 0;
  row_counts.clear();
  live_row_totals = 0;
}

}  // namespace sirius::transparent
