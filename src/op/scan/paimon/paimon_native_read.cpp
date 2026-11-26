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

#include "op/scan/paimon/paimon_native_read.hpp"

#include <paimon/executor.h>
#include <paimon/fs/file_system.h>
#include <paimon/memory/memory_pool.h>
#include <paimon/read_context.h>
#include <paimon/scan_context.h>
#include <paimon/status.h>
#include <paimon/table/source/plan.h>
#include <paimon/table/source/table_read.h>
#include <paimon/table/source/table_scan.h>

#include <functional>
#include <set>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace sirius::op::scan {
namespace {
void check_stop(scan_manager::preparation_stop_token stop)
{
  if (stop.stop_requested())
    throw std::system_error(std::make_error_code(std::errc::operation_canceled));
}
template <class T>
T checked(paimon::Result<T>&& result)
{
  if (!result.ok()) detail::throw_paimon_status(result.status());
  return std::move(result).value();
}
class inline_executor final : public paimon::Executor {
 public:
  void Add(std::function<void()> task) override { task(); }
  void ShutdownNow() override {}
  uint32_t GetThreadNum() const override { return 1; }
};

struct read_owners {
  std::shared_ptr<void> modules;
  std::shared_ptr<paimon::MemoryPool> pool;
  std::shared_ptr<paimon::Executor> executor = std::make_shared<inline_executor>();
  std::unique_ptr<paimon::TableScan> scan;
  std::shared_ptr<paimon::Plan> plan;
  std::unique_ptr<paimon::TableRead> read;
};
}  // namespace
std::unique_ptr<paimon_batch_cursor> open_paimon_read(
  paimon_read_metadata const& metadata,
  std::vector<int32_t> const& ids,
  std::shared_ptr<paimon::MemoryPool> pool,
  std::shared_ptr<void> modules,
  int64_t max_rows,
  scan_manager::preparation_stop_token stop,
  std::shared_ptr<paimon::FileSystem> filesystem)
{
  check_stop(stop);
  if (!pool || !modules || max_rows <= 0 || ids.empty() || metadata.branch != "main" ||
      metadata.path.empty() || metadata.schema_json.empty() || metadata.schema_id < 0 ||
      (metadata.snapshot_id && *metadata.snapshot_id <= 0))
    throw std::invalid_argument("Paimon read requires retained main metadata, owners and limits");
  std::set<int32_t> unique;
  for (auto id : ids)
    if (id < 0 || !unique.insert(id).second)
      throw std::invalid_argument("Paimon read projection requires distinct stable field IDs");
  auto options = metadata.options;
  for (auto const& [key, value] : options) {
    if (key == "branch" && value != metadata.branch)
      throw std::invalid_argument("Paimon read options disagree with retained branch");
    if (!key.starts_with("scan.")) continue;
    if (key == "scan.mode" && value == "from-snapshot-full") continue;
    if (key == "scan.snapshot-id" && metadata.snapshot_id &&
        value == std::to_string(*metadata.snapshot_id))
      continue;
    throw std::invalid_argument("Paimon read options disagree with retained snapshot selection");
  }
  options["branch"]    = metadata.branch;
  options["scan.mode"] = "from-snapshot-full";
  if (metadata.snapshot_id) options["scan.snapshot-id"] = std::to_string(*metadata.snapshot_id);
  options["read.batch-size"]                    = std::to_string(max_rows);
  options["read.preparation-window"]            = "1";
  options["parquet.read.enable-pre-buffer"]     = "false";
  options["parquet.read.executor.thread-count"] = "0";
  auto owners                                   = std::make_shared<read_owners>();
  owners->modules                               = std::move(modules);
  owners->pool                                  = std::move(pool);
  if (metadata.snapshot_id) {
    paimon::ScanContextBuilder scan(metadata.path);
    if (filesystem) scan.WithFileSystem(filesystem);
    owners->scan = checked(paimon::TableScan::Create(checked(scan.SetOptions(options)
                                                               .SetTableSchema(metadata.schema_json)
                                                               .WithMemoryPool(owners->pool)
                                                               .WithExecutor(owners->executor)
                                                               .Finish())));
    owners->plan = checked(owners->scan->CreatePlan());
    if (owners->plan->SnapshotId() != metadata.snapshot_id)
      throw std::runtime_error("Paimon plan changed the retained snapshot");
  }
  check_stop(stop);
  paimon::ReadContextBuilder read(metadata.path);
  if (filesystem) read.WithFileSystem(filesystem);
  owners->read = checked(paimon::TableRead::Create(checked(read.WithBranch(metadata.branch)
                                                             .SetTableSchema(metadata.schema_json)
                                                             .SetOptions(options)
                                                             .SetReadFieldIds(ids)
                                                             .WithMemoryPool(owners->pool)
                                                             .WithExecutor(owners->executor)
                                                             .EnablePrefetch(false)
                                                             .EnableMultiThreadRowToBatch(false)
                                                             .EnableLateMaterializing(false)
                                                             .SetReadAheadCacheEnabled(false)
                                                             .EnablePredicateFilter(false)
                                                             .Finish())));
  check_stop(stop);
  // Initializing the retained schema and projection is required even for empty-at-bind.
  // Never plan "latest" for a null snapshot; later commits cannot make it nonempty.
  if (!metadata.snapshot_id || owners->plan->Splits().empty()) return {};
  auto reader = checked(owners->read->CreateReader(owners->plan->Splits()));
  try {
    check_stop(stop);
    return std::make_unique<paimon_batch_cursor>(std::move(reader), owners);
  } catch (...) {
    // Covers a stop or cursor-holder allocation failure before ownership transfer.
    if (reader) {
      try {
        reader->Close();
      } catch (...) {
      }
    }
    throw;
  }
}
}  // namespace sirius::op::scan
