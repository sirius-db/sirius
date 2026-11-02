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

#include "op/scan/parquet_delete_reader.hpp"

#include <duckdb/common/exception.hpp>
#include <duckdb/common/types/data_chunk.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/parallel/task_executor.hpp>
#include <parquet_multi_file_info.hpp>
#include <parquet_reader.hpp>

#include <atomic>
#include <exception>
#include <mutex>
#include <unordered_set>

namespace sirius::op::scan {
namespace {
class delete_scan_task : public duckdb::BaseExecutorTask {
 public:
  delete_scan_task(duckdb::TaskExecutor& executor, std::function<void()> run)
    : BaseExecutorTask(executor), run(std::move(run))
  {
  }
  void ExecuteTask() override { run(); }

 private:
  std::function<void()> run;
};
}  // namespace

void read_parquet_delete_batches(duckdb::ClientContext& context,
                                 duckdb::OpenFileInfo file,
                                 parquet_delete_read_options const& options,
                                 parquet_delete_bind const& bind,
                                 std::function<parquet_delete_consumer()> const& make_consumer)
{
  using namespace duckdb;
  auto check_stop = [&] {
    context.InterruptCheck();
    if (options.check_stop) options.check_stop();
  };
  check_stop();
  // Preserve caller-supplied open options, but never accept stale cached metadata.
  auto info = make_shared_ptr<ExtendedOpenFileInfo>();
  if (file.extended_info) info->options = file.extended_info->options;
  info->options["validate_external_file_cache"] = Value::BOOLEAN(true);
  file.extended_info                            = std::move(info);
  ParquetReader reader(context, std::move(file), ParquetOptions(context));
  auto& opened  = reader.GetHandle();
  auto identity = opened.GetFileHandle()->Stats();
  bool local = identity.file_type == FileType::FILE_TYPE_REGULAR && identity.device_id.IsValid() &&
               identity.file_id.IsValid();
  if (identity.file_size < 0 || opened.GetFileSize() != static_cast<uint64_t>(identity.file_size) ||
      (options.expected_size && *options.expected_size != opened.GetFileSize()))
    throw IOException("Parquet delete size differs from binding");
  auto verify = [&](CachingFileHandle& handle) {
    check_stop();
    auto current = handle.GetFileHandle()->Stats();
    // Remote files do not have POSIX device/inode identifiers. Compare the
    // metadata supplied by their filesystem on every handle, including the
    // separate scan handle, without changing that filesystem's admission or
    // credential rules. Conditional range requests remain the backend's job.
    if (current.file_size != identity.file_size ||
        current.last_modification_time != identity.last_modification_time ||
        current.version_tag != identity.version_tag ||
        handle.GetFileSize() != static_cast<uint64_t>(identity.file_size) ||
        handle.GetVersionTag() != identity.version_tag ||
        handle.GetLastModifiedTime() != identity.last_modification_time ||
        (local && (!current.device_id.IsValid() || !current.file_id.IsValid() ||
                   current.device_id.GetIndex() != identity.device_id.GetIndex() ||
                   current.file_id.GetIndex() != identity.file_id.GetIndex())))
      throw IOException("Parquet delete object changed during reading");
    if (!local && options.verify_remote) options.verify_remote(handle);
  };
  verify(opened);
  auto size = opened.GetFileSize();
  if (size < 12) throw IOException("Truncated Parquet delete file");
  uint8_t tail[8];
  opened.GetFileHandle()->Read(tail, 8, size - 8);
  verify(opened);
  uint64_t footer = uint64_t(tail[0]) | (uint64_t(tail[1]) << 8) | (uint64_t(tail[2]) << 16) |
                    (uint64_t(tail[3]) << 24);
  if (tail[4] != 'P' || tail[5] != 'A' || tail[6] != 'R' || tail[7] != '1' || footer > size - 12 ||
      (options.expected_footer_size && *options.expected_footer_size != footer))
    throw IOException("Parquet delete footer differs from binding");

  std::vector<parquet_delete_column> columns;
  for (auto const& column : reader.GetColumns())
    columns.push_back({column.name.GetIdentifierName(), column.type});
  auto projection = bind(columns, reader.NumRows());
  if (projection.empty()) throw InvalidInputException("Empty Parquet delete projection");
  duckdb::vector<duckdb::LogicalType> types;
  std::unordered_set<uint64_t> seen;
  for (auto index : projection) {
    if (index >= columns.size() || !seen.insert(index).second)
      throw InvalidInputException("Invalid Parquet delete projection");
    reader.column_indexes.emplace_back(index);
    reader.column_ids.emplace_back(index);
    types.push_back(columns[index].type);
  }
  ParquetReadGlobalState global(nullptr);
  auto& scheduler = TaskScheduler::GetScheduler(context);
  auto workers = std::max<idx_t>(1, std::min(reader.NumRowGroups(), scheduler.NumberOfThreads()));
  std::mutex claim_mutex;
  std::atomic<bool> failed{false};
  std::vector<uint64_t> rows(workers);
  auto scan_groups = [&](idx_t worker, parquet_delete_consumer const& consume) {
    ParquetReadLocalState local_state;
    DataChunk chunk;
    chunk.Initialize(context, types);
    auto& count = rows[worker];
    while (true) {
      check_stop();
      {
        std::lock_guard lock(claim_mutex);
        if (failed.load(std::memory_order_relaxed)) return;
        if (!reader.TryInitializeScan(context, global, local_state)) break;
      }
      reader.PrepareScan(context, global, local_state);
      verify(*local_state.scan_state.file_handle);
      auto io = reader.ScheduleIO(context, global, local_state);
      if (io.HasTasks()) io.ExecuteTasksSynchronously();
      if (io.GetResultType() != AsyncResultType::HAVE_MORE_OUTPUT)
        throw IOException("Parquet delete I/O did not complete");
      while (true) {
        if (failed.load(std::memory_order_relaxed)) return;
        verify(*local_state.scan_state.file_handle);
        chunk.Reset();
        auto scan = reader.Scan(context, global, local_state, chunk);
        if (scan.HasTasks()) {
          scan.ExecuteTasksSynchronously();
          continue;
        }
        auto status = scan.GetResultType();
        verify(*local_state.scan_state.file_handle);
        if (status != AsyncResultType::HAVE_MORE_OUTPUT && status != AsyncResultType::FINISHED)
          throw IOException("Parquet delete decoding did not complete");
        if (chunk.size() > reader.NumRows() - count)
          throw IOException("Parquet delete rows exceed footer count");
        count += chunk.size();
        if (chunk.size()) consume(chunk);
        if (status == AsyncResultType::FINISHED) break;
      }
    }
  };
  if (workers == 1) {
    scan_groups(0, make_consumer());
  } else {
    // Open the shared scan handle with the caller's transaction before dispatch.
    reader.PrepareReadAhead(context, global);
    std::exception_ptr first_error;
    TaskExecutor executor(context);
    duckdb::vector<duckdb::unique_ptr<BaseExecutorTask>> tasks;
    for (idx_t i = 0; i < workers; ++i) {
      tasks.push_back(make_uniq<delete_scan_task>(executor, [&, i, consume = make_consumer()] {
        try {
          scan_groups(i, consume);
        } catch (...) {
          std::lock_guard lock(claim_mutex);
          if (!first_error) first_error = std::current_exception();
          failed.store(true, std::memory_order_relaxed);
        }
      }));
    }
    executor.ScheduleTasks(std::move(tasks));
    executor.WorkOnTasks();
    if (first_error) std::rethrow_exception(first_error);
  }
  uint64_t count = 0;
  for (auto n : rows) {
    if (n > reader.NumRows() - count) throw IOException("Parquet delete rows exceed footer count");
    count += n;
  }
  verify(opened);
  if (count != reader.NumRows()) throw IOException("Incomplete Parquet delete footer coverage");
}
}  // namespace sirius::op::scan
