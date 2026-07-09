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

#include "io/s3/sirius_httpfs.hpp"

#include "io/io_context.hpp"
#include "io/sirius_datasource.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"

#include <duckdb/common/exception.hpp>
#include <duckdb/common/file_opener.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/main/client_context.hpp>

#include <cctype>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace sirius::io::s3 {

namespace {

constexpr std::string_view kScheme = "s3://";

/// FileHandle backed by a sirius_datasource resolved through the
/// scan_manager's create_datasource(path) seam — the datasource carries its
/// io backend, io_object and any cached metadata, and its host_read goes
/// through the prefetch-cache-integrated path. Holds shared ownership so the
/// backend outlives the handle. @c cursor_ only serves the sequential
/// @c Read/Seek bookkeeping; the parquet reader uses positional reads.
class sirius_httpfs_file_handle : public duckdb::FileHandle {
 public:
  sirius_httpfs_file_handle(duckdb::FileSystem& fs,
                            std::string path,
                            duckdb::FileOpenFlags flags,
                            std::shared_ptr<sirius::io::sirius_datasource> datasource)
    : duckdb::FileHandle(fs, std::move(path), flags), datasource_(std::move(datasource))
  {
  }

  void Close() override {}

  std::shared_ptr<sirius::io::sirius_datasource> datasource_;
  duckdb::idx_t cursor_{0};
};

sirius_httpfs_file_handle& as_httpfs_handle(duckdb::FileHandle& handle)
{
  return static_cast<sirius_httpfs_file_handle&>(handle);
}

}  // namespace

bool sirius_httpfs::CanHandleFile(const std::string& fpath)
{
  if (fpath.size() <= kScheme.size()) { return false; }
  for (std::size_t i = 0; i < kScheme.size(); ++i) {
    if (static_cast<char>(std::tolower(static_cast<unsigned char>(fpath[i]))) != kScheme[i]) {
      return false;
    }
  }
  // After "s3://" we need a non-empty bucket AND a non-empty key, i.e. a '/'
  // that is neither first (empty bucket) nor last (empty key). This rejects
  // "s3://bucket".
  auto const rest  = std::string_view{fpath}.substr(kScheme.size());
  auto const slash = rest.find('/');
  return slash != std::string_view::npos && slash != 0 && slash + 1 < rest.size();
}

duckdb::unique_ptr<duckdb::FileHandle> sirius_httpfs::OpenFile(
  const std::string& path,
  duckdb::FileOpenFlags flags,
  duckdb::optional_ptr<duckdb::FileOpener> opener)
{
  // Read-only filesystem: reject write opens (e.g. COPY ... TO 's3://…') before
  // resolving the connection, so callers get a clear error instead of failing
  // later on a HEAD of a not-yet-existing object.
  if (flags.OpenForWriting()) {
    throw duckdb::IOException("[sirius_httpfs] '" + path +
                              "' is read-only; S3 writes (COPY TO) are not supported");
  }
  // The ClientFileSystem (OpenerFileSystem) layer injects the connection's
  // FileOpener even though the parquet reader passes none.
  auto client = duckdb::FileOpener::TryGetClientContext(opener);
  if (!client) {
    throw std::runtime_error("[sirius_httpfs] no ClientContext while opening '" + path +
                             "'; S3 reads require a Sirius-enabled connection");
  }
  // Transparent S3 is GPU-only. The GPU scan reads column data through the routed
  // ioctx, so this FileSystem only ever serves the bind-time footer read for the
  // GPU path. If gpu_execution is off there is no GPU consumer, so opening here
  // would be a CPU read of s3:// — which Sirius does not support. Refuse with a
  // clear message instead of silently serving a CPU fallback.
  {
    duckdb::Value gpu_exec;
    auto have         = client->TryGetCurrentSetting("gpu_execution", gpu_exec);
    bool const gpu_on = have && !gpu_exec.IsNull() && gpu_exec.GetValue<bool>();
    if (!gpu_on) {
      throw duckdb::IOException("[sirius_httpfs] reading '" + path +
                                "' over S3 requires GPU execution: S3 is GPU-only and has no CPU "
                                "fallback; SET gpu_execution=true");
    }
  }
  auto sirius_ctx = client->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    throw std::runtime_error(
      "[sirius_httpfs] Sirius is not initialized on this connection while opening '" + path + "'");
  }
  // No S3 CPU fallback. This FileSystem is only meant to serve the bind-time
  // footer read for the transparent GPU path (the GPU scan then reads via the
  // routed ioctx, not here). A CPU-fallback replay active here means the GPU
  // plan failed and we are replaying on CPU (run_internal_cpu_fallback_query
  // wraps the replay in a CpuFallbackGuard) — e.g. gpu_execution('SELECT ...
  // FROM v_s3') whose view body reads s3:// and whose GPU plan failed. Refuse
  // the read so s3:// data is never served to a CPU plan, even when reached
  // indirectly through a view. Uses the narrow CpuFallbackGuard flag (not the
  // broad is_internal_query_active), so a legitimate internal s3:// read is not
  // blocked.
  if (sirius_ctx->is_cpu_fallback_active()) {
    throw duckdb::IOException(
      "[sirius_httpfs] reading '" + path +
      "' on a CPU execution path: S3 CPU fallback is not supported (S3 is GPU-only); "
      "Sirius has no CPU fallback for S3 data sources");
  }
  // Resolve through the scan_manager's datasource factory (the routed seam):
  // the returned sirius_datasource performs the HEAD and carries the backend;
  // HEAD failures (missing key / auth / network) propagate as exceptions for
  // DuckDB to surface at bind time.
  auto datasource = sirius_ctx->get_scan_manager().create_datasource(path);
  if (!datasource) {
    throw std::runtime_error("[sirius_httpfs] no S3 backend supports '" + path + "'");
  }
  return duckdb::make_uniq<sirius_httpfs_file_handle>(*this, path, flags, std::move(datasource));
}

void sirius_httpfs::Read(duckdb::FileHandle& handle,
                         void* buffer,
                         int64_t nr_bytes,
                         duckdb::idx_t location)
{
  if (nr_bytes < 0) {
    throw duckdb::IOException("[sirius_httpfs] negative read size on '" + handle.GetPath() + "'");
  }
  auto& h        = as_httpfs_handle(handle);
  auto const got = h.datasource_->host_read(static_cast<std::size_t>(location),
                                            static_cast<std::size_t>(nr_bytes),
                                            static_cast<std::uint8_t*>(buffer));
  // DuckDB's positional Read contract is read-exactly-or-throw; host_read
  // clips an EOF-crossing range to a short read, which would otherwise leave the
  // tail of `buffer` stale.
  if (got != static_cast<std::size_t>(nr_bytes)) {
    throw duckdb::IOException("[sirius_httpfs] short read on '" + handle.GetPath() +
                              "': requested " + std::to_string(nr_bytes) + " at " +
                              std::to_string(static_cast<std::uint64_t>(location)) + ", got " +
                              std::to_string(got));
  }
}

int64_t sirius_httpfs::Read(duckdb::FileHandle& handle, void* buffer, int64_t nr_bytes)
{
  if (nr_bytes < 0) {
    throw duckdb::IOException("[sirius_httpfs] negative read size on '" + handle.GetPath() + "'");
  }
  auto& h        = as_httpfs_handle(handle);
  auto const got = h.datasource_->host_read(static_cast<std::size_t>(h.cursor_),
                                            static_cast<std::size_t>(nr_bytes),
                                            static_cast<std::uint8_t*>(buffer));
  h.cursor_ += got;
  return static_cast<int64_t>(got);
}

int64_t sirius_httpfs::GetFileSize(duckdb::FileHandle& handle)
{
  return static_cast<int64_t>(as_httpfs_handle(handle).datasource_->size());
}

duckdb::timestamp_t sirius_httpfs::GetLastModifiedTime(duckdb::FileHandle& /*handle*/)
{
  return duckdb::timestamp_t(0);
}

duckdb::vector<duckdb::OpenFileInfo> sirius_httpfs::Glob(const std::string& path,
                                                         duckdb::FileOpener* /*opener*/)
{
  // No S3 LIST: reject glob/wildcard patterns with a clear error instead of
  // treating '*' as a literal key and failing later on object open.
  if (duckdb::FileSystem::HasGlob(path)) {
    throw duckdb::IOException(
      "[sirius_httpfs] glob/wildcard patterns are not supported for s3:// "
      "(no S3 LIST); specify an exact object key: '" +
      path + "'");
  }
  duckdb::vector<duckdb::OpenFileInfo> result;
  if (CanHandleFile(path)) { result.emplace_back(path); }
  return result;
}

void sirius_httpfs::Seek(duckdb::FileHandle& handle, duckdb::idx_t location)
{
  as_httpfs_handle(handle).cursor_ = location;
}

duckdb::idx_t sirius_httpfs::SeekPosition(duckdb::FileHandle& handle)
{
  return as_httpfs_handle(handle).cursor_;
}

}  // namespace sirius::io::s3
