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
#include <duckdb/parallel/scan_read_ahead.hpp>
#include <duckdb/parallel/task_executor.hpp>
#include <parquet_multi_file_info.hpp>
#include <parquet_reader.hpp>

#include <atomic>
#include <exception>
#include <mutex>
#include <vector>
#include <unordered_set>

namespace sirius::op::scan {
namespace {
// DuckDB's async pool transports ErrorData. Keep the original exception (including
// S3 version information) until both the decoding and I/O tasks have drained.
struct delete_read_failure {
  std::atomic<bool> failed{false};
  std::once_flag first;
  std::exception_ptr error;

  void capture()
  {
    std::call_once(first, [&] { error = std::current_exception(); });
    failed.store(true, std::memory_order_release);
  }
};

class delete_prefetch_task : public duckdb::AsyncTask {
 public:
  delete_prefetch_task(duckdb::unique_ptr<duckdb::AsyncTask> task,
                       delete_read_failure& failure,
                       std::function<void()> const& check_stop)
    : task(std::move(task)), failure(failure), check_stop(check_stop)
  {
  }
  duckdb::idx_t GetIOSize() const override { return task->GetIOSize(); }
  void Execute() override
  {
    if (failure.failed.load(std::memory_order_relaxed)) return;
    try {
      check_stop();
      task->Execute();
    } catch (...) {
      failure.capture();
      throw;
    }
  }

 private:
  duckdb::unique_ptr<duckdb::AsyncTask> task;
  delete_read_failure& failure;
  std::function<void()> const& check_stop;
};

class delete_scan_task : public duckdb::BaseExecutorTask {
 public:
  delete_scan_task(duckdb::TaskExecutor& executor, std::function<void()> run)
    : BaseExecutorTask(executor), run(std::move(run))
  {
  }
  void ExecuteTask() override
  {
    auto execute = std::move(run);
    execute();
  }
  void Cancel() override { run = nullptr; }

 private:
  std::function<void()> run;
};
void scan_delete_row_groups(duckdb::ClientContext& context,
                            duckdb::ParquetReader& reader,
                            duckdb::vector<duckdb::LogicalType> const& types,
                            std::function<parquet_delete_consumer()> const& make_consumer,
                            std::function<void()> const& check_stop,
                            std::function<void(duckdb::CachingFileHandle&)> const& verify,
                            std::function<void()> const& after_decode_for_testing)
{
  using namespace duckdb;
  ParquetReadGlobalState global(nullptr);
  auto& scheduler = TaskScheduler::GetScheduler(context);
  auto workers = std::max<idx_t>(1, std::min(reader.NumRowGroups(), scheduler.NumberOfThreads()));
  std::mutex claim_mutex;
  delete_read_failure failure;
  std::vector<parquet_delete_consumer> consumers;
  duckdb::unique_ptr<ScanReadAhead> read_ahead;
  if (reader.NumRowGroups() > 1) {
    bool disable_prefetch = false;
    context.TryGetCurrentSetting("disable_parquet_prefetching", disable_prefetch);
    if (!disable_prefetch) read_ahead = ScanReadAhead::Create(context);
  }
  if (workers > 1 || read_ahead) reader.PrepareReadAhead(context, global);

  auto claim = [&](ParquetReadLocalState& state) {
    check_stop();
    std::lock_guard lock(claim_mutex);
    return !failure.failed.load(std::memory_order_relaxed) &&
           reader.TryInitializeScan(context, global, state);
  };
  auto schedule_io = [&](ParquetReadLocalState& state) {
    reader.PrepareScan(context, global, state);
    verify(*state.scan_state.file_handle);
    return reader.ScheduleIO(context, global, state);
  };
  using prefetch_job = ScanReadAheadJobWrapper<ParquetReadLocalState>;
  auto produce       = [&](duckdb::vector<duckdb::unique_ptr<AsyncTask>>& tasks)
    -> duckdb::unique_ptr<ScanReadAheadJob> {
    try {
      auto job = make_uniq<prefetch_job>();
      if (!claim(*job)) return nullptr;
      job->batch_index = job->group_index;
      auto io          = schedule_io(*job);
      if (io.HasTasks()) {
        tasks = io.ExtractAsyncTasks();
        for (auto& task : tasks)
          task = make_uniq<delete_prefetch_task>(std::move(task), failure, check_stop);
      } else if (io.GetResultType() != AsyncResultType::HAVE_MORE_OUTPUT) {
        throw IOException("Parquet delete I/O did not complete");
      }
      return std::move(job);
    } catch (...) {
      failure.capture();
      throw;
    }
  };
  std::atomic<uint64_t> admitted{0};
  const auto expected = reader.NumRows();
  auto scan_groups    = [&](parquet_delete_consumer const& consume) {
    ParquetReadLocalState serial_state;
    TableFunctionInput input(nullptr, nullptr, &global);
    DataChunk chunk;
    chunk.Initialize(context, types);
    while (!failure.failed.load(std::memory_order_relaxed)) {
      duckdb::unique_ptr<ScanReadAheadJob> job;
      auto* state = &serial_state;
      if (read_ahead) {
        // Synchronous table-function input waits for the claimed job's I/O.
        auto result = read_ahead->AcquireJob(context, input, produce, job);
        if (result == ScanReadAheadAcquire::EXHAUSTED) break;
        if (result != ScanReadAheadAcquire::ACQUIRED)
          throw InternalException("Synchronous Parquet delete scan unexpectedly parked");
        state = static_cast<prefetch_job*>(job.get());
      } else {
        if (!claim(*state)) break;
        auto io = schedule_io(*state);
        if (io.HasTasks()) io.ExecuteTasksSynchronously();
        if (io.GetResultType() != AsyncResultType::HAVE_MORE_OUTPUT)
          throw IOException("Parquet delete I/O did not complete");
      }
      while (true) {
        if (failure.failed.load(std::memory_order_relaxed)) return;
        verify(*state->scan_state.file_handle);
        chunk.Reset();
        auto scan = reader.Scan(context, global, *state, chunk);
        if (scan.HasTasks()) {
          scan.ExecuteTasksSynchronously();
          continue;
        }
        auto status = scan.GetResultType();
        verify(*state->scan_state.file_handle);
        if (status != AsyncResultType::HAVE_MORE_OUTPUT && status != AsyncResultType::FINISHED)
          throw IOException("Parquet delete decoding did not complete");
        if (chunk.size()) {
          if (after_decode_for_testing) after_decode_for_testing();
          auto count = admitted.load(std::memory_order_relaxed);
          do {
            if (chunk.size() > expected - count)
              throw IOException("Parquet delete rows exceed footer count");
          } while (!admitted.compare_exchange_weak(
            count, count + chunk.size(), std::memory_order_relaxed));
          if (failure.failed.load(std::memory_order_acquire)) return;
          consume(chunk);
        }
        if (status == AsyncResultType::FINISHED) break;
      }
    }
  };
  auto run = [&](parquet_delete_consumer const& consume) {
    try {
      scan_groups(consume);
    } catch (...) {
      failure.capture();
    }
  };
  std::exception_ptr executor_error;
  if (workers == 1) {
    consumers.push_back(make_consumer());
    run(consumers.back());
  } else {
    TaskExecutor executor(context);
    try {
      duckdb::vector<duckdb::unique_ptr<BaseExecutorTask>> tasks;
      for (idx_t i = 0; i < workers; ++i) {
        consumers.push_back(make_consumer());
        tasks.push_back(make_uniq<delete_scan_task>(executor, [&, i] { run(consumers[i]); }));
      }
      executor.ScheduleTasks(std::move(tasks));
      executor.WorkOnTasks();
    } catch (...) {
      executor_error = std::current_exception();
      failure.failed.store(true, std::memory_order_release);
      executor.CancelAndDrain();
    }
  }
  if (read_ahead) {
    read_ahead->CancelAndDrain();
    try {
      read_ahead->ThrowIfError();
    } catch (...) {
      if (!executor_error) executor_error = std::current_exception();
    }
  }
  if (failure.error) std::rethrow_exception(failure.error);
  if (executor_error) std::rethrow_exception(executor_error);
  verify(reader.GetHandle());
  if (admitted.load(std::memory_order_relaxed) != expected)
    throw IOException("Incomplete Parquet delete footer coverage");
}
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
  std::mutex stats_mutex;
  auto verify = [&](CachingFileHandle& handle) {
    check_stop();
    auto current = [&] {
      std::lock_guard lock(stats_mutex);
      return handle.GetFileHandle()->Stats();
    }();
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
  if (options.expected_footer_size) {
    uint8_t tail[8];
    // Check the bound size against the open object even if its cached trailer is reusable.
    opened.GetFileHandle()->Read(tail, sizeof(tail), size - sizeof(tail));
    verify(opened);
    uint64_t footer = uint64_t(tail[0]) | (uint64_t(tail[1]) << 8) | (uint64_t(tail[2]) << 16) |
                      (uint64_t(tail[3]) << 24);
    if (tail[4] != 'P' || tail[5] != 'A' || tail[6] != 'R' || tail[7] != '1' || footer == 0 ||
        footer > size - 12 || *options.expected_footer_size != footer)
      throw IOException("Parquet delete footer differs from binding");
  }

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
  scan_delete_row_groups(
    context, reader, types, make_consumer, check_stop, verify, options.after_decode_for_testing);
}
}  // namespace sirius::op::scan
