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

#include "scan_manager/preparation_ledger.hpp"

#include <fcntl.h>
#include <paimon/executor.h>
#include <paimon/fs/file_system.h>
#include <paimon/memory/memory_pool.h>
#include <paimon/read_context.h>
#include <paimon/scan_context.h>
#include <paimon/table/source/plan.h>
#include <paimon/table/source/table_read.h>
#include <paimon/table/source/table_scan.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <filesystem>
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
  if (!result.ok()) throw std::runtime_error(result.status().ToString());
  return std::move(result).value();
}
class inline_executor final : public paimon::Executor {
 public:
  void Add(std::function<void()> task) override { task(); }
  void ShutdownNow() override {}
  uint32_t GetThreadNum() const override { return 1; }
};

// Only the stock, whole-file ReadFile operation uses this stream. The file is already
// open when its length is obtained and remains the source of the read after admission.
class snapshot_input_stream final : public paimon::InputStream {
 public:
  ~snapshot_input_stream() override
  {
    if (fd_ >= 0) ::close(fd_);
  }

  paimon::Status open(std::string const& path)
  {
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (fd_ < 0) {
      if (errno == ENOENT) return paimon::Status::NotExist("retained snapshot missing");
      return io_error("open");
    }
    struct stat info{};
    if (::fstat(fd_, &info) != 0) return io_error("fstat");
    if (!S_ISREG(info.st_mode) || info.st_size < 0)
      return paimon::Status::Invalid("retained snapshot must be a local regular file");
    length_ = info.st_size;
    return paimon::Status::OK();
  }

  paimon::Status Close() override
  {
    if (fd_ >= 0 && ::close(std::exchange(fd_, -1)) != 0) return io_error("close");
    return paimon::Status::OK();
  }
  paimon::Result<int64_t> Length() const override { return length_; }
  paimon::Result<int64_t> Read(char* data, int64_t size) override
  {
    if (size != length_) return paimon::Status::Invalid("retained snapshot read extent changed");
    int64_t total = 0;
    while (total < size) {
      auto count = ::read(fd_, data + total, size - total);
      if (count == 0) break;
      if (count < 0) {
        if (errno == EINTR) continue;
        return io_error("read");
      }
      total += count;
    }
    return total;
  }
  paimon::Status Seek(int64_t, paimon::SeekOrigin) override { return unsupported(); }
  paimon::Result<int64_t> GetPos() const override { return unsupported(); }
  paimon::Result<int64_t> Read(char*, int64_t, int64_t) override { return unsupported(); }
  void ReadAsync(char*, int64_t, int64_t, std::function<void(paimon::Status)>&& done) override
  {
    done(unsupported());
  }
  paimon::Result<std::string> GetUri() const override { return std::string{}; }

 private:
  static paimon::Status io_error(char const* operation)
  {
    auto const code = std::error_code(errno, std::generic_category());
    return paimon::Status::IOError(operation, " retained snapshot: ", code.message());
  }
  static paimon::Status unsupported()
  {
    return paimon::Status::NotImplemented(
      "retained snapshot stream only supports a whole-file read");
  }
  int fd_         = -1;
  int64_t length_ = 0;
};

class snapshot_budget_filesystem final : public paimon::FileSystem {
 public:
  snapshot_budget_filesystem(paimon_snapshot_input_budget const& budget, std::string path)
    : filesystem_(budget.filesystem), reserve_(budget.try_reserve), path_(std::move(path))
  {
  }

  paimon::Result<std::unique_ptr<paimon::InputStream>> Open(std::string const& path) const override
  {
    if (path != path_) return filesystem_->Open(path);
#if defined(__GLIBCXX__) && _GLIBCXX_USE_CXX11_ABI
    if (reservation_ || !reserve_)
      return paimon::Status::Invalid("retained snapshot input can only be opened once per plan");
    auto stream = std::make_unique<snapshot_input_stream>();
    auto status = stream->open(path);
    if (!status.ok()) return status;
    auto const length = static_cast<uint64_t>(stream->Length().value());
    auto const empty  = std::string{};
    if (length > empty.max_size()) return refuse("retained snapshot exceeds string capacity");
    // stock ReadFile resizes an initially empty libstdc++ string once, which can
    // grow to twice its inline capacity and allocates an extra terminator byte.
    auto const bytes = length <= empty.capacity()
                         ? uint64_t{0}
                         : std::max(length, uint64_t{2} * empty.capacity()) + 1;
    reservation_     = reserve_(bytes);
    if (!reservation_) return refuse("retained snapshot input exceeds available budget");
    return std::unique_ptr<paimon::InputStream>(std::move(stream));
#else
    return paimon::Status::NotImplemented(
      "retained snapshot admission requires libstdc++ C++11 ABI");
#endif
  }

  paimon::Result<std::unique_ptr<paimon::InputStream>> Open(
    paimon::FileStatus const& status) const override
  {
    auto path = status.GetPath();
    return path == path_ ? Open(path) : filesystem_->Open(status);
  }
  paimon::Result<paimon::FileStatus> GetFileStatus(std::string const& path) const override
  {
    return filesystem_->GetFileStatus(path);
  }
  paimon::Status ListDir(std::string const& path,
                         std::vector<paimon::BasicFileStatus>* files) const override
  {
    return filesystem_->ListDir(path, files);
  }
  paimon::Status ListFileStatus(std::string const& path,
                                std::vector<paimon::FileStatus>* files) const override
  {
    return filesystem_->ListFileStatus(path, files);
  }
  paimon::Result<bool> Exists(std::string const& path) const override
  {
    return filesystem_->Exists(path);
  }
  paimon::Result<std::unique_ptr<paimon::OutputStream>> Create(std::string const&,
                                                               bool) const override
  {
    return read_only();
  }
  paimon::Status Mkdirs(std::string const&) const override { return read_only(); }
  paimon::Status Rename(std::string const&, std::string const&) const override
  {
    return read_only();
  }
  paimon::Status Delete(std::string const&, bool) const override { return read_only(); }

  bool exceeded_budget() const { return exceeded_budget_; }

  void finish()
  {
    if (!reservation_)
      throw std::runtime_error("Paimon planning did not consume the admitted retained snapshot");
    reservation_.reset();
    reserve_ = {};
  }

 private:
  paimon::Status refuse(char const* message) const
  {
    exceeded_budget_ = true;
    return paimon::Status::CapacityError(message);
  }
  static paimon::Status read_only()
  {
    return paimon::Status::NotImplemented("scan filesystem is read-only");
  }

  std::shared_ptr<paimon::FileSystem> filesystem_;
  std::function<std::shared_ptr<void>(uint64_t)> reserve_;
  std::string path_;
  // The input stream closes before stock Paimon finishes parsing its string.
  // Inline execution lets us retire the lease only after CreatePlan unwinds.
  mutable std::shared_ptr<void> reservation_;
  mutable bool exceeded_budget_ = false;
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
std::unique_ptr<paimon_batch_cursor> open_paimon_read(paimon_read_metadata const& metadata,
                                                      std::vector<int32_t> const& ids,
                                                      std::shared_ptr<paimon::MemoryPool> pool,
                                                      std::shared_ptr<void> modules,
                                                      int64_t max_rows,
                                                      scan_manager::preparation_stop_token stop,
                                                      paimon_snapshot_input_budget const* budget)
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
  options["parquet.read.enable-pre-buffer"]     = "false";
  options["parquet.read.executor.thread-count"] = "0";
  auto owners                                   = std::make_shared<read_owners>();
  owners->modules                               = std::move(modules);
  owners->pool                                  = std::move(pool);
  if (metadata.snapshot_id) {
    paimon::ScanContextBuilder scan(metadata.path);
    std::shared_ptr<snapshot_budget_filesystem> snapshot_fs;
    if (budget) {
      auto path = std::filesystem::path(metadata.path);
      // System-table aliases can resolve to a different table root inside the SDK.
      if (!budget->filesystem || !budget->try_reserve || !path.is_absolute() ||
          path.lexically_normal().string() != metadata.path || !path.has_filename() ||
          metadata.path.find('\0') != std::string::npos ||
          path.filename().string().find('$') != std::string::npos ||
          path.parent_path().filename() == "sys")
        throw std::invalid_argument(
          "Paimon snapshot admission requires a normalized local data table path and budget");
      snapshot_fs = std::make_shared<snapshot_budget_filesystem>(
        *budget,
        (path / "snapshot" / ("snapshot-" + std::to_string(*metadata.snapshot_id))).string());
      scan.WithFileSystem(snapshot_fs);
    }
    owners->scan = checked(paimon::TableScan::Create(checked(scan.SetOptions(options)
                                                               .SetTableSchema(metadata.schema_json)
                                                               .WithMemoryPool(owners->pool)
                                                               .WithExecutor(owners->executor)
                                                               .Finish())));
    auto plan    = owners->scan->CreatePlan();
    if (snapshot_fs && snapshot_fs->exceeded_budget() && !plan.ok() &&
        plan.status().IsCapacityError())
      throw scan_manager::preparation_resource_error(plan.status().ToString(), true);
    owners->plan = checked(std::move(plan));
    if (snapshot_fs) snapshot_fs->finish();
    if (owners->plan->SnapshotId() != metadata.snapshot_id)
      throw std::runtime_error("Paimon plan changed the retained snapshot");
  }
  check_stop(stop);
  paimon::ReadContextBuilder read(metadata.path);
  if (metadata.snapshot_id && budget) read.WithFileSystem(budget->filesystem);
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
    return std::make_unique<paimon_batch_cursor>(std::move(reader), owners, max_rows);
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
