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

/// @brief DuckDB FileSystem subsystem that lets DuckDB's native @c read_parquet
/// BIND an @c s3:// object through Sirius's per-connection REST io backend.
///
/// Purpose: make the transparent form work —
/// @code
///   SET gpu_execution = true;
///   SELECT ... FROM read_parquet('s3://bucket/file.parquet');
/// @endcode
/// without loading httpfs and without the @c sirius_read_parquet rewrite. DuckDB
/// binds @c read_parquet('s3://') by reading the parquet footer through this
/// FileSystem (positional reads via the routed @c rest_ioctx); the resulting
/// native @c read_parquet scan is then captured by the transparent optimizer
/// hook and executed on GPU, where the actual column data is read via the same
/// routed ioctx (NOT through this FileSystem).
///
/// Registered DB-global on the VirtualFileSystem; the object itself is
/// **stateless** (holds no ioctx). The per-connection backend is resolved
/// lazily in @ref OpenFile via the injected @c FileOpener -> @c ClientContext ->
/// @c SiriusContext: the @c ClientFileSystem (@c OpenerFileSystem) layer supplies
/// the opener even though the parquet reader calls @c OpenFile with a null opener.
///
/// **GPU-only / no CPU fallback:** @ref OpenFile refuses to open an @c s3:// path
/// unless @c gpu_execution is enabled on the connection. The GPU path reads via
/// the routed ioctx, so this FileSystem only ever serves the bind-time footer
/// read; it is never the CPU data path. Read-only: write opens are rejected.
class sirius_httpfs : public duckdb::FileSystem {
 public:
  sirius_httpfs() = default;

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

  /// Concrete-object only: returns @c {path} when @ref CanHandleFile, else empty.
  /// Rejects glob/wildcard patterns (no S3 LIST) with a clear error.
  duckdb::vector<duckdb::OpenFileInfo> Glob(const std::string& path,
                                            duckdb::FileOpener* opener = nullptr) override;

  void Seek(duckdb::FileHandle& handle, duckdb::idx_t location) override;
  duckdb::idx_t SeekPosition(duckdb::FileHandle& handle) override;
  bool CanSeek() override { return true; }
  bool OnDiskFile(duckdb::FileHandle& /*handle*/) override { return false; }

  std::string GetName() const override { return "SiriusHttpFS"; }
};

}  // namespace sirius::io::s3
