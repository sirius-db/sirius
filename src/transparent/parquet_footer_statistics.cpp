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

#include "io/s3/sirius_httpfs.hpp"
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
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sirius::transparent {

namespace {

std::atomic<std::size_t> live_records{0};
std::atomic<std::uint64_t> next_record_id{1};
std::atomic<std::uint64_t> footers_read{0};
std::atomic<std::uint64_t> files_inspected{0};

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

/// Whether \p path names a remote file. DuckDB's test compares lowercase scheme prefixes, so alone
/// it takes the `S3://` spelling that Sirius's S3 file system serves for a local file.
bool is_remote(std::string const& path)
{
  return io::s3::can_handle_s3_path(path) || duckdb::FileSystem::IsRemoteFile(path);
}

/// Text that is equal for two sightings of \p path only if they saw the same bytes, or nullopt when
/// \p metadata cannot promise that. A remote file needs a strong ETag: a weak one allows different
/// bytes, and a remote modification time may be absent or coarse. A local file's version tag holds
/// its device, inode, size and modification time, trusted the way DuckDB's own file caches trust
/// it: only once the time is old enough that a coarse clock cannot hide a rewrite.
std::optional<std::string> version_of(std::string const& path, duckdb::FileMetadata const& metadata)
{
  if (metadata.file_size < 0 || metadata.version_tag.empty()) { return std::nullopt; }
  if (is_remote(path)) {
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

/// Per column of a scan's bind: how many values lie from the smallest to the largest the column
/// holds across the scan's files, where the footers give an integer range.
using value_span_list = std::vector<std::optional<std::uint64_t>>;

/// What the footers of a scan's files say of its columns together.
struct merged_facts {
  /// The `column_statistics` value, or NULL when the files say nothing the option can carry.
  duckdb::Value statistics;
  std::shared_ptr<value_span_list const> value_spans;
};

/// The merged, verified statistics a scan was last given, with the ids of the per-file records they
/// were merged from. The entry fits a scan only while the cache answers each of its files with the
/// same record. It holds ids, not the records, so it keeps none of them alive.
struct merge_entry {
  std::vector<std::uint64_t> files;
  decltype(duckdb::MultiFileBindData::names) names;
  decltype(duckdb::MultiFileBindData::types) types;
  merged_facts facts;
};

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
/// The footer row total of one scan, kept for the plan that holds the scan's file list.
struct row_count_entry {
  duckdb::weak_ptr<duckdb::MultiFileList> owner;
  std::size_t rows;
  /// Groups copied counts for the test accessor.
  std::uint64_t serial;
  /// The value spans of the scan's columns, shared with the copies of its plan. Null until the
  /// planning that counted the rows has merged the columns.
  std::shared_ptr<value_span_list const> value_spans;
};

/// Keyed by the file list of a scan's bind data, which a plan owns and each plan copy has anew, so
/// a total answers only for the plan it was published or copied for. The weak owner tells a list
/// that is gone from a new one at its address.
std::unordered_map<duckdb::MultiFileList const*, row_count_entry> row_counts;
std::uint64_t next_row_count_serial             = 0;
constexpr std::size_t row_count_sweep_allowance = 64;
std::size_t row_count_sweep_at                  = row_count_sweep_allowance;

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

/// Drops the entries whose file list is gone. Nothing tells the table when a list goes, so it is
/// swept when it reaches twice its size after the last sweep plus the allowance: the insertions
/// since then pay for the sweep.
void sweep_row_counts_locked()
{
  std::erase_if(row_counts, [](auto const& entry) { return entry.second.owner.expired(); });
  row_count_sweep_at = 2 * row_counts.size() + row_count_sweep_allowance;
}

void keep_row_count_locked(duckdb::shared_ptr<duckdb::MultiFileList> const& list,
                           std::size_t rows,
                           std::uint64_t serial,
                           std::shared_ptr<value_span_list const> value_spans = nullptr)
{
  if (row_counts.size() >= row_count_sweep_at) { sweep_row_counts_locked(); }
  row_counts.insert_or_assign(
    list.get(),
    row_count_entry{
      duckdb::weak_ptr<duckdb::MultiFileList>(list), rows, serial, std::move(value_spans)});
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
  /// The files looked at: all of them unless the walk stopped over the local limit.
  std::size_t inspected = 0;
};

/// Whether no per-file record is dropped for capacity, so that the order of use decides nothing.
bool statistics_cache_unbounded()
{
  std::lock_guard guard(cache_mutex);
  return capacities.statistics == detail::unbounded;
}

bool within(std::int64_t limit, std::size_t reads)
{
  return limit < 0 || reads <= static_cast<std::size_t>(limit);
}

/// The cached statistics of \p files, in order, without reading a footer. The rest are counted as
/// reads. A remote file the listing did not describe is one of them even if it is cached: learning
/// its version takes a request, which is what the count is there to bound. Stops at the first
/// local miss that exceeds \p local_limit: the scan is over that limit whatever follows, and this
/// avoids a stat for each remaining unlisted local file. With a bounded per-file cache the walk
/// goes on to the last file instead, because a hit also marks its record as recently used and the
/// records that stay decide what a later scan has to read.
scan_statistics cached_statistics(duckdb::ClientContext& context,
                                  std::vector<duckdb::OpenFileInfo> const& files,
                                  duckdb::ParquetOptions const& options,
                                  std::int64_t local_limit = -1)
{
  scan_statistics result;
  result.files.resize(files.size());
  result.keys.resize(files.size());
  result.listed.resize(files.size());
  auto const options_identity = op::scan::parquet_options_identity(options);
  auto& fs                    = duckdb::FileSystem::GetFileSystem(context);
  bool const stop_over_local  = local_limit >= 0 && statistics_cache_unbounded();
  for (std::size_t i = 0; i < files.size(); ++i) {
    ++result.inspected;
    bool const remote = is_remote(files[i].path);
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
      if (stop_over_local && !remote && !within(local_limit, result.local_reads)) { break; }
    }
  }
  files_inspected.fetch_add(result.inspected, std::memory_order_relaxed);
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

/// The statistics of each of \p files, in order, reading the footers the cache does not answer.
/// Not limited: for the callers that process every file.
std::vector<std::shared_ptr<file_statistics const>> statistics_of_all(
  duckdb::ClientContext& context,
  std::vector<duckdb::OpenFileInfo> const& files,
  duckdb::ParquetOptions const& options)
{
  auto scan = cached_statistics(context, files, options);
  read_unresolved(context, files, options, scan);
  return std::move(scan.files);
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

/// The statistics of \p bind's column \p i across \p files, or null when it is generated, or a
/// file lacks them or types the column differently.
duckdb::unique_ptr<duckdb::BaseStatistics> merged_column(
  duckdb::MultiFileBindData const& bind,
  std::vector<std::shared_ptr<file_statistics const>> const& files,
  duckdb::idx_t i)
{
  if (bind.IsGeneratedColumn(i)) { return nullptr; }
  duckdb::unique_ptr<duckdb::BaseStatistics> merged;
  for (auto const& file : files) {
    auto const* stats = find(*file, bind.names[i]);
    if (!stats || stats->GetType() != bind.types[i]) { return nullptr; }
    if (merged) {
      merged->Merge(*stats);
    } else {
      merged = stats->ToUnique();
    }
  }
  return merged;
}

/// The integer a statistics bound holds, or nullopt for a type that is not stored as an integer
/// of up to 64 bits. A DECIMAL gives its unscaled value and a DATE its day number, which count
/// values as well as the values themselves do.
std::optional<__int128> integer_of(duckdb::Value const& value)
{
  if (value.IsNull()) { return std::nullopt; }
  switch (value.type().InternalType()) {
    case duckdb::PhysicalType::INT8: return value.GetValueUnsafe<int8_t>();
    case duckdb::PhysicalType::INT16: return value.GetValueUnsafe<int16_t>();
    case duckdb::PhysicalType::INT32: return value.GetValueUnsafe<int32_t>();
    case duckdb::PhysicalType::INT64: return value.GetValueUnsafe<int64_t>();
    case duckdb::PhysicalType::UINT8: return value.GetValueUnsafe<uint8_t>();
    case duckdb::PhysicalType::UINT16: return value.GetValueUnsafe<uint16_t>();
    case duckdb::PhysicalType::UINT32: return value.GetValueUnsafe<uint32_t>();
    case duckdb::PhysicalType::UINT64: return value.GetValueUnsafe<uint64_t>();
    default: return std::nullopt;
  }
}

/// How many values lie from the smallest to the largest \p merged holds, or nullopt when it gives
/// no integer range.
std::optional<std::uint64_t> value_span(duckdb::BaseStatistics const& merged)
{
  if (merged.GetStatsType() != duckdb::StatisticsType::NUMERIC_STATS ||
      !duckdb::NumericStats::HasMinMax(merged)) {
    return std::nullopt;
  }
  auto const low  = integer_of(duckdb::NumericStats::Min(merged));
  auto const high = integer_of(duckdb::NumericStats::Max(merged));
  if (!low || !high || *high < *low) { return std::nullopt; }
  auto const span = *high - *low + 1;
  if (span > static_cast<__int128>(std::numeric_limits<std::uint64_t>::max())) {
    return std::nullopt;
  }
  return static_cast<std::uint64_t>(span);
}

/// What the footers of \p files say of \p bind's columns together: the `column_statistics` value
/// describing them, NULL when they say nothing the option can carry, and each column's value span.
/// Both come from one merge of each column, so the spans cost a scan no pass of their own.
merged_facts merged_statistics(duckdb::MultiFileBindData const& bind,
                               std::vector<std::shared_ptr<file_statistics const>> const& files)
{
  auto const text = duckdb::LogicalType::VARCHAR;
  value_span_list spans(bind.names.size());
  std::vector<duckdb::Value> keys, values;
  for (duckdb::idx_t i = 0; i < bind.names.size(); ++i) {
    auto const merged = merged_column(bind, files, i);
    if (!merged) { continue; }
    spans[i] = value_span(*merged);
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
  auto value_spans = std::make_shared<value_span_list const>(std::move(spans));
  if (keys.empty()) { return {duckdb::Value(), std::move(value_spans)}; }
  return {duckdb::Value::MAP(
            text, duckdb::LogicalType::MAP(text, text), std::move(keys), std::move(values)),
          std::move(value_spans)};
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

std::optional<merged_facts> cached_merge(std::string const& key,
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
  return found->second.facts;
}

/// Keeps \p facts for the next planning of the scan, if it can answer one: every record must
/// be the one the cache holds for its file, and the entry must fit beside the others.
void remember_merge(std::string const& key,
                    scan_statistics const& scan,
                    duckdb::MultiFileBindData const& bind,
                    merged_facts const& facts)
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
  merge_entry entry{{}, bind.names, bind.types, facts};
  entry.files.reserve(scan.files.size());
  for (auto const& file : scan.files) {
    entry.files.push_back(file->id);
  }
  merges.emplace(key, std::move(entry));
  merge_references += scan.files.size();
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
  return entry.owner.lock().get() == list;
}

void forget_row_count(duckdb::MultiFileList const* list)
{
  std::lock_guard guard(cache_mutex);
  row_counts.erase(list);
}

void remember_row_count(duckdb::shared_ptr<duckdb::MultiFileList> const& list, std::size_t rows)
{
  std::lock_guard guard(cache_mutex);
  keep_row_count_locked(list, rows, next_row_count_serial++);
}

/// Adds the value spans of its scan's columns to the row count kept for \p list. They belong to
/// the same planning as the count, and go when it goes.
void remember_value_spans(duckdb::MultiFileList const* list,
                          std::shared_ptr<value_span_list const> value_spans)
{
  std::lock_guard guard(cache_mutex);
  auto const found = row_counts.find(list);
  if (found != row_counts.end() && answers(found->second, list)) {
    found->second.value_spans = std::move(value_spans);
  }
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
  auto scan = cached_statistics(context, all_files, parquet_options, limits.local);
  // All or nothing: the statistics of some files say nothing true about the scan, and the
  // optimizer takes what it is given as fact.
  if (!within(limits.remote, scan.remote_reads) || !within(limits.local, scan.local_reads)) {
    SIRIUS_LOG_DEBUG(
      "No footer statistics or row count for a Parquet scan of {} files: {} remote and {} local "
      "misses in the {} files inspected, limits {} and {}",
      all_files.size(),
      scan.remote_reads,
      scan.local_reads,
      scan.inspected,
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
  if (single_file) {
    // DuckDB has this file's statistics itself; only its value spans are kept.
    value_span_list spans(bind->names.size());
    for (duckdb::idx_t i = 0; i < spans.size(); ++i) {
      if (auto const column = merged_column(*bind, scan.files, i)) {
        spans[i] = value_span(*column);
      }
    }
    remember_value_spans(bind->file_list.get(),
                         std::make_shared<value_span_list const>(std::move(spans)));
    return false;
  }
  auto const scan_key = merge_cache_key(get);
  if (auto hit = cached_merge(scan_key, scan, *bind)) {
    remember_value_spans(bind->file_list.get(), std::move(hit->value_spans));
    options.column_statistics = std::move(hit->statistics);
  } else {
    auto merged = merged_statistics(*bind, scan.files);
    // Files that say nothing the option can carry give no range either.
    if (merged.statistics.IsNull()) { return false; }
    remember_value_spans(bind->file_list.get(), merged.value_spans);
    options.column_statistics = merged.statistics;
    options.VerifyColumnStatistics(bind->names, bind->types);
    remember_merge(scan_key, scan, *bind, merged);
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
      SIRIUS_LOG_DEBUG("No footer statistics or row count for a Parquet scan: {}", e.what());
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
  {
    std::lock_guard guard(cache_mutex);
    auto const found = row_counts.find(list->get());
    if (found != row_counts.end() && answers(found->second, list->get())) {
      return found->second.rows;
    }
  }
  sirius::log::log_noexcept([&] {
    SIRIUS_LOG_DEBUG("No footer row count for the Parquet scan at table index {}",
                     get.table_index.index);
  });
  return std::nullopt;
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
    auto const rows   = found->second.rows;
    auto const serial = found->second.serial;
    auto value_spans  = found->second.value_spans;
    keep_row_count_locked(*target, rows, serial, std::move(value_spans));
  } catch (std::bad_alloc const&) {
    // The copy plans without the count.
  }
}

std::optional<std::uint64_t> parquet_scan_value_span(duckdb::LogicalGet const& get,
                                                     std::size_t column)
{
  auto const* list = file_list_of(get);
  if (!list) { return std::nullopt; }
  std::lock_guard guard(cache_mutex);
  auto const found = row_counts.find(list->get());
  if (found == row_counts.end() || !answers(found->second, list->get()) ||
      !found->second.value_spans || column >= found->second.value_spans->size()) {
    return std::nullopt;
  }
  return (*found->second.value_spans)[column];
}

std::size_t read_parquet_footer_statistics(duckdb::ClientContext& context,
                                           std::vector<std::string> const& paths)
{
  duckdb::ParquetOptions const options(context);
  std::vector<duckdb::OpenFileInfo> const files(paths.begin(), paths.end());
  return statistics_of_all(context, files, options).size();
}

std::size_t parquet_files_row_count(duckdb::ClientContext& context,
                                    std::vector<std::string> const& paths)
{
  duckdb::ParquetOptions const options(context);
  std::vector<duckdb::OpenFileInfo> const files(paths.begin(), paths.end());
  std::size_t rows = 0;
  for (auto const& file : statistics_of_all(context, files, options)) {
    rows += file->num_rows;
  }
  return rows;
}

std::size_t detail::merged_statistics_cache_size()
{
  std::lock_guard guard(cache_mutex);
  return merges.size();
}

std::size_t detail::row_count_cache_size()
{
  std::lock_guard guard(cache_mutex);
  sweep_row_counts_locked();
  std::unordered_set<std::uint64_t> totals;
  for (auto const& [list, entry] : row_counts) {
    totals.insert(entry.serial);
  }
  return totals.size();
}

std::size_t detail::row_count_association_count()
{
  std::lock_guard guard(cache_mutex);
  return row_counts.size();
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

std::uint64_t detail::inspected_file_count()
{
  return files_inspected.load(std::memory_order_relaxed);
}

detail::cache_capacities detail::set_cache_capacities_for_testing(cache_capacities replacement)
{
  std::lock_guard guard(cache_mutex);
  auto const previous = capacities;
  capacities          = replacement;
  evict_statistics_locked();
  evict_merges_locked();
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
  row_count_sweep_at = row_count_sweep_allowance;
}

}  // namespace sirius::transparent
