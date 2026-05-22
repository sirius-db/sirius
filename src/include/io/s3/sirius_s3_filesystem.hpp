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

#pragma once

#include <duckdb/common/file_system.hpp>
#include <duckdb/common/open_file_info.hpp>

#include <cstdint>
#include <string>

namespace sirius::io::s3 {

/// @brief DuckDB FileSystem subsystem that serves @c s3:// reads through
/// Sirius's per-connection @c s3_ioctx.
///
/// Registered DB-global on the VirtualFileSystem; the object itself is
/// **stateless** (holds no @c s3_ioctx). The per-connection backend is resolved
/// lazily in @ref OpenFile via the injected @c FileOpener ->
/// @c ClientContext -> @c SiriusContext: the @c ClientFileSystem
/// (@c OpenerFileSystem) layer supplies the opener even though the parquet
/// reader calls @c OpenFile with a null opener (see newplan.md §29.9). The
/// resolved backend + io_object are parked on the returned handle, so reads need
/// no opener.
///
/// Read-only: this backs DuckDB's CPU @c read_parquet over S3 — the S3
/// execution CPU fallback (Sirius GPU scan stays the primary path). The parquet
/// reader uses only positional @c Read, so no real cursor is required.
class sirius_s3_filesystem : public duckdb::FileSystem {
 public:
  sirius_s3_filesystem() = default;

  /// True only for a well-formed @c s3://<bucket>/<key> (case-insensitive
  /// scheme; rejects @c s3://bucket with no key, @c file://, local paths).
  bool CanHandleFile(const std::string& fpath) override;

  duckdb::unique_ptr<duckdb::FileHandle> OpenFile(
    const std::string& path,
    duckdb::FileOpenFlags flags,
    duckdb::optional_ptr<duckdb::FileOpener> opener = nullptr) override;

  void Read(duckdb::FileHandle& handle,
            void* buffer,
            int64_t nr_bytes,
            duckdb::idx_t location) override;
  int64_t Read(duckdb::FileHandle& handle, void* buffer, int64_t nr_bytes) override;

  int64_t GetFileSize(duckdb::FileHandle& handle) override;
  duckdb::timestamp_t GetLastModifiedTime(duckdb::FileHandle& handle) override;

  /// Concrete-object only: returns @c {path} when @ref CanHandleFile, else empty
  /// (no S3 LIST, so wildcards are out of scope — newplan.md §29.5).
  duckdb::vector<duckdb::OpenFileInfo> Glob(const std::string& path,
                                            duckdb::FileOpener* opener = nullptr) override;

  void Seek(duckdb::FileHandle& handle, duckdb::idx_t location) override;
  duckdb::idx_t SeekPosition(duckdb::FileHandle& handle) override;
  bool CanSeek() override { return true; }
  bool OnDiskFile(duckdb::FileHandle& /*handle*/) override { return false; }

  std::string GetName() const override { return "SiriusS3FileSystem"; }
};

}  // namespace sirius::io::s3
