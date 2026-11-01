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
#include <parquet_multi_file_info.hpp>
#include <parquet_reader.hpp>

#include <stdexcept>
#include <unordered_set>

namespace sirius::op::scan {
void read_parquet_delete_batches(duckdb::ClientContext& context,
                                 duckdb::OpenFileInfo file,
                                 parquet_delete_read_options const& options,
                                 parquet_delete_bind const& bind,
                                 std::function<void(duckdb::DataChunk&)> const& consume)
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
  ParquetReadLocalState local_state;
  DataChunk chunk;
  chunk.Initialize(context, types);
  uint64_t count = 0;
  while (true) {
    check_stop();
    if (!reader.TryInitializeScan(context, global, local_state)) break;
    reader.PrepareScan(context, global, local_state);
    verify(*local_state.scan_state.file_handle);
    auto io = reader.ScheduleIO(context, global, local_state);
    if (io.HasTasks()) io.ExecuteTasksSynchronously();
    if (io.GetResultType() != AsyncResultType::HAVE_MORE_OUTPUT)
      throw IOException("Parquet delete I/O did not complete");
    while (true) {
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
  verify(opened);
  if (count != reader.NumRows()) throw IOException("Incomplete Parquet delete footer coverage");
}
}  // namespace sirius::op::scan
