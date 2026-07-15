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

#include "io/s3/s3_list_parser.hpp"

#include <duckdb/common/file_system.hpp>
#include <duckdb/common/open_file_info.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace sirius::scan_manager {
class sirius_scan_manager;
}  // namespace sirius::scan_manager

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

  /// Extended open: when @p file carries a @c "file_size" option (a
  /// non-negative BIGINT — the LIST-provided size @ref Glob attaches), the
  /// datasource is opened through the parquet-footer-probe seam: ONE
  /// suffix-range GET resolves the size from Content-Range (no HEAD) and
  /// stashes the trailing bytes for the binder's footer reads. The size value
  /// itself only discriminates this extended open from the plain fallback —
  /// the probe re-derives the authoritative size from the response. A missing /
  /// wrong-typed / negative option silently falls back to the plain open path
  /// (HEAD, then ranged GETs) — third-party-populated @c OpenFileInfo must not
  /// break the open. Non-parquet files opened this way still read correctly;
  /// they just pay for a stashed tail that no footer read consumes.
  duckdb::unique_ptr<duckdb::FileHandle> OpenFileExtended(
    const duckdb::OpenFileInfo& file,
    duckdb::FileOpenFlags flags,
    duckdb::optional_ptr<duckdb::FileOpener> opener) override;

  void Read(duckdb::FileHandle& handle,
            void* buffer,
            int64_t nr_bytes,
            duckdb::idx_t location) override;
  int64_t Read(duckdb::FileHandle& handle, void* buffer, int64_t nr_bytes) override;

  int64_t GetFileSize(duckdb::FileHandle& handle) override;
  duckdb::timestamp_t GetLastModifiedTime(duckdb::FileHandle& handle) override;

  /// Exact key: returns @c {path} when @ref CanHandleFile, else empty. A
  /// glob/wildcard pattern expands via one paginated S3 LIST (through the
  /// connection's scan_manager — see @ref expand_glob), gated like @ref
  /// OpenFile: requires a resolvable @c ClientContext, @c gpu_execution
  /// enabled, and no active CPU-fallback replay. Each match carries its
  /// LIST-provided size in the extended info so the subsequent open needs no
  /// HEAD (one suffix-range GET instead — see @ref OpenFileExtended). Match
  /// paths embed the literal LIST key via @ref escape_s3_key_for_uri: keys
  /// containing @c #, @c ? or a bare @c % show the escaped form in the path
  /// text (and in the @c filename virtual column) but open the exact object.
  /// A matched key containing a percent-encoded sequence (@c % followed by
  /// two hex digits) throws instead — DuckDB URL-decodes hive partition
  /// values from the path once, so no single path text can serve both the
  /// open and hive semantics for such keys; full support is a tracked
  /// follow-up. Zero matches yield an empty vector (DuckDB raises its
  /// standard no-files error).
  duckdb::vector<duckdb::OpenFileInfo> Glob(const std::string& path,
                                            duckdb::FileOpener* opener = nullptr) override;

  void Seek(duckdb::FileHandle& handle, duckdb::idx_t location) override;
  duckdb::idx_t SeekPosition(duckdb::FileHandle& handle) override;
  bool CanSeek() override { return true; }
  bool OnDiskFile(duckdb::FileHandle& /*handle*/) override { return false; }

  std::string GetName() const override { return "SiriusHttpFS"; }

 protected:
  bool SupportsOpenFileExtended() const override { return true; }
};

/// Escape a literal S3 object key for embedding into an @c s3:// URI so the
/// later @c sirius::io::parse() of that URI restores the exact key bytes.
/// LIST returns keys as literal bytes, but the open path treats the produced
/// string as a URI — @c '#' starts a fragment, @c '?' starts a query, and
/// @c %xx percent-decodes — so exactly those three bytes are escaped
/// (@c '%'→"%25" first, then @c '#'→"%23", @c '?'→"%3F"). Every other byte,
/// including @c '/', @c '=' and space, stays literal: hive-partition path
/// text (@c col=a b/) must survive unchanged. Identity for keys without
/// @c %/#/?, i.e. every plain key round-trips byte-identically.
std::string escape_s3_key_for_uri(std::string_view key);

/// Expand an @c s3:// glob @p pattern into concrete object URIs via one
/// paginated ListObjectsV2 sweep, streamed page-by-page (peak memory = one
/// page + the matches). The LIST prefix is everything up to the last '/'
/// before the first wildcard, so the sweep is server-side-narrowed to the
/// table's prefix. Matching follows DuckDB's glob semantics per '/'-segment
/// (@c *, @c ?, @c […] never cross a segment; @c ** crosses). Matches carry
/// their LIST size in @c extended_info->options["file_size"]. Throws (never
/// truncates) once more than @p max_matches keys match — "narrow the glob
/// prefix"; wildcards in the bucket segment are rejected. @p max_matches unset
/// → the backend's configured cap (@c rest.list_max_matches).
duckdb::vector<duckdb::OpenFileInfo> expand_glob(
  std::string const& pattern,
  sirius::scan_manager::sirius_scan_manager& scan_manager,
  std::optional<std::size_t> max_matches = std::nullopt);

}  // namespace sirius::io::s3
