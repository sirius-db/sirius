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

#include "io/s3/sirius_s3_filesystem.hpp"

#include "io/io_context.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"

#include <duckdb/common/exception.hpp>
#include <duckdb/common/file_opener.hpp>
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

/// FileHandle backed by a resolved s3_ioctx + io_object. Holds shared ownership
/// so the backend outlives the handle. @c cursor_ only serves the sequential
/// @c Read/Seek bookkeeping; the parquet reader uses positional reads.
class sirius_s3_file_handle : public duckdb::FileHandle {
 public:
  sirius_s3_file_handle(duckdb::FileSystem& fs,
                        std::string path,
                        duckdb::FileOpenFlags flags,
                        std::shared_ptr<sirius::io::sirius_ioctx> ioctx,
                        std::shared_ptr<sirius::io::sirius_io_object> object)
    : duckdb::FileHandle(fs, std::move(path), flags),
      ioctx_(std::move(ioctx)),
      object_(std::move(object))
  {
  }

  void Close() override {}

  std::shared_ptr<sirius::io::sirius_ioctx> ioctx_;
  std::shared_ptr<sirius::io::sirius_io_object> object_;
  duckdb::idx_t cursor_{0};
};

sirius_s3_file_handle& as_s3_handle(duckdb::FileHandle& handle)
{
  return static_cast<sirius_s3_file_handle&>(handle);
}

}  // namespace

bool sirius_s3_filesystem::CanHandleFile(const std::string& fpath)
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

duckdb::unique_ptr<duckdb::FileHandle> sirius_s3_filesystem::OpenFile(
  const std::string& path,
  duckdb::FileOpenFlags flags,
  duckdb::optional_ptr<duckdb::FileOpener> opener)
{
  // Read-only filesystem: reject write opens (e.g. COPY ... TO 's3://…') before
  // resolving the connection, so callers get a clear error instead of failing
  // later on a HEAD of a not-yet-existing object.
  if (flags.OpenForWriting()) {
    throw duckdb::IOException("[sirius_s3_filesystem] '" + path +
                              "' is read-only; S3 writes (COPY TO) are not supported");
  }
  // The ClientFileSystem (OpenerFileSystem) layer injects the connection's
  // FileOpener even though the parquet reader passes none (newplan §29.9).
  auto client = duckdb::FileOpener::TryGetClientContext(opener);
  if (!client) {
    throw std::runtime_error("[sirius_s3_filesystem] no ClientContext while opening '" + path +
                             "'; S3 reads require a Sirius-enabled connection");
  }
  auto sirius_ctx = client->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    throw std::runtime_error(
      "[sirius_s3_filesystem] Sirius is not initialized on this connection "
      "while opening '" +
      path + "'");
  }
  auto ioctx = sirius_ctx->get_scan_manager().io_ctx_shared_for(path);
  if (!ioctx) {
    throw std::runtime_error("[sirius_s3_filesystem] no S3 backend supports '" + path + "'");
  }
  auto object = ioctx->create_io_object(path);
  return duckdb::make_uniq<sirius_s3_file_handle>(
    *this, path, flags, std::move(ioctx), std::move(object));
}

void sirius_s3_filesystem::Read(duckdb::FileHandle& handle,
                                void* buffer,
                                int64_t nr_bytes,
                                duckdb::idx_t location)
{
  if (nr_bytes < 0) {
    throw duckdb::IOException("[sirius_s3_filesystem] negative read size on '" + handle.GetPath() +
                              "'");
  }
  auto& h        = as_s3_handle(handle);
  auto const got = h.ioctx_->host_read_io(*h.object_,
                                          static_cast<std::size_t>(location),
                                          static_cast<std::size_t>(nr_bytes),
                                          static_cast<std::uint8_t*>(buffer));
  // DuckDB's positional Read contract is read-exactly-or-throw; host_read_io
  // clips an EOF-crossing range to a short read, which would otherwise leave the
  // tail of `buffer` stale.
  if (got != static_cast<std::size_t>(nr_bytes)) {
    throw duckdb::IOException("[sirius_s3_filesystem] short read on '" + handle.GetPath() +
                              "': requested " + std::to_string(nr_bytes) + " at " +
                              std::to_string(static_cast<std::uint64_t>(location)) + ", got " +
                              std::to_string(got));
  }
}

int64_t sirius_s3_filesystem::Read(duckdb::FileHandle& handle, void* buffer, int64_t nr_bytes)
{
  auto& h        = as_s3_handle(handle);
  auto const got = h.ioctx_->host_read_io(*h.object_,
                                          static_cast<std::size_t>(h.cursor_),
                                          static_cast<std::size_t>(nr_bytes),
                                          static_cast<std::uint8_t*>(buffer));
  h.cursor_ += got;
  return static_cast<int64_t>(got);
}

int64_t sirius_s3_filesystem::GetFileSize(duckdb::FileHandle& handle)
{
  return static_cast<int64_t>(as_s3_handle(handle).object_->size());
}

duckdb::timestamp_t sirius_s3_filesystem::GetLastModifiedTime(duckdb::FileHandle& /*handle*/)
{
  return duckdb::timestamp_t(0);
}

duckdb::vector<duckdb::OpenFileInfo> sirius_s3_filesystem::Glob(const std::string& path,
                                                                duckdb::FileOpener* /*opener*/)
{
  // No S3 LIST: reject glob/wildcard patterns with a clear error instead of
  // treating '*' as a literal key and failing later on object open (§29.5).
  if (duckdb::FileSystem::HasGlob(path)) {
    throw duckdb::IOException(
      "[sirius_s3_filesystem] glob/wildcard patterns are not supported for s3:// "
      "(no S3 LIST); specify an exact object key: '" +
      path + "'");
  }
  duckdb::vector<duckdb::OpenFileInfo> result;
  if (CanHandleFile(path)) { result.emplace_back(path); }
  return result;
}

void sirius_s3_filesystem::Seek(duckdb::FileHandle& handle, duckdb::idx_t location)
{
  as_s3_handle(handle).cursor_ = location;
}

duckdb::idx_t sirius_s3_filesystem::SeekPosition(duckdb::FileHandle& handle)
{
  return as_s3_handle(handle).cursor_;
}

}  // namespace sirius::io::s3
