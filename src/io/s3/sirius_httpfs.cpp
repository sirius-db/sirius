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
#include <duckdb/function/scalar/string_common.hpp>
#include <duckdb/main/client_context.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

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
    : duckdb::FileHandle(fs, std::move(path), flags),
      datasource_(std::move(datasource)),
      version_tag_(datasource_->get_io_object().validation_tag())
  {
  }

  void Close() override {}

  std::shared_ptr<sirius::io::sirius_datasource> datasource_;
  std::string version_tag_;
  duckdb::idx_t cursor_{0};
};

sirius_httpfs_file_handle& as_httpfs_handle(duckdb::FileHandle& handle)
{
  return static_cast<sirius_httpfs_file_handle&>(handle);
}

/// Shared gate for every s3:// access through this FileSystem (open AND glob
/// expansion): resolve the connection's SiriusContext and enforce the GPU-only
/// contract — reject when gpu_execution is off or a CPU-fallback replay is
/// active. @p verb only shapes the error text.
duckdb::shared_ptr<duckdb::SiriusContext> resolve_gated_sirius_context(
  duckdb::optional_ptr<duckdb::FileOpener> opener, std::string const& path, char const* verb)
{
  // The ClientFileSystem (OpenerFileSystem) layer injects the connection's
  // FileOpener even though the parquet reader passes none.
  auto client = duckdb::FileOpener::TryGetClientContext(opener);
  if (!client) {
    throw std::runtime_error(std::string("[sirius_httpfs] no ClientContext while ") + verb + " '" +
                             path + "'; S3 reads require a Sirius-enabled connection");
  }
  auto sirius_ctx = client->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    throw std::runtime_error(std::string("[sirius_httpfs] Sirius is not initialized on this "
                                         "connection while ") +
                             verb + " '" + path + "'");
  }
  // Health is checked first so an unavailable runtime always surfaces the
  // stable unavailable error (a gpu_execution=false session must not mask it
  // with the gpu-off message), and the untrusted scan manager is never
  // consulted.
  if (sirius_ctx->get_runtime_health() == duckdb::SiriusContext::runtime_health::UNAVAILABLE) {
    sirius_ctx->throw_runtime_unavailable();
  }
  // Transparent S3 is GPU-only. If gpu_execution is off there is no GPU
  // consumer, so serving here would be a CPU read of s3:// — which Sirius does
  // not support. Refuse with a clear message instead of silently serving a CPU
  // fallback. Applies to glob expansion too: an expanded file list is only ever
  // consumed by a scan that would hit this same wall at open.
  {
    duckdb::Value gpu_exec;
    auto have         = client->TryGetCurrentSetting("gpu_execution", gpu_exec);
    bool const gpu_on = have && !gpu_exec.IsNull() && gpu_exec.GetValue<bool>();
    if (!gpu_on) {
      throw duckdb::IOException(std::string("[sirius_httpfs] ") + verb + " '" + path +
                                "' over S3 requires GPU execution: S3 is GPU-only and has no CPU "
                                "fallback; SET gpu_execution=true");
    }
  }
  // No S3 CPU fallback. A CPU-fallback replay active here means the GPU plan
  // failed and we are replaying on CPU (the replay is wrapped in a
  // CpuFallbackGuard). Refuse so s3:// data is never served to a CPU plan,
  // even when reached indirectly through a view. Uses the narrow
  // CpuFallbackGuard flag (not the broad is_internal_query_active), so a
  // legitimate internal s3:// read is not blocked. The flag is per-connection
  // (this connection's replay), so an unrelated connection's fallback does
  // not block this one's S3 access.
  auto conn_state = duckdb::get_sirius_connection_state(*client);
  if (conn_state && conn_state->is_cpu_fallback_active()) {
    throw duckdb::IOException(std::string("[sirius_httpfs] ") + verb + " '" + path +
                              "' on a CPU execution path: S3 CPU fallback is not supported (S3 is "
                              "GPU-only); Sirius has no CPU fallback for S3 data sources");
  }
  return sirius_ctx;
}

/// The LIST-provided size Glob attaches ("file_size", duckdb-httpfs convention).
/// Wrong type / null / negative → nullopt: a third-party-populated OpenFileInfo
/// must degrade to the plain (HEAD) open path, never break the open.
std::optional<std::uint64_t> extract_known_size(duckdb::OpenFileInfo const& file)
{
  if (!file.extended_info) { return std::nullopt; }
  auto const it = file.extended_info->options.find("file_size");
  if (it == file.extended_info->options.end()) { return std::nullopt; }
  auto const& value = it->second;
  if (value.IsNull() || value.type().id() != duckdb::LogicalTypeId::BIGINT) { return std::nullopt; }
  auto const size = value.GetValue<int64_t>();
  if (size < 0) { return std::nullopt; }
  return static_cast<std::uint64_t>(size);
}

/// Split @p s into its '/'-separated segments into @p out (cleared first). The
/// segment views alias @p s and exist only for glob matching. @p out is a
/// reused scratch buffer so a broad listing does not heap-allocate per key.
void split_segments(std::string_view s, std::vector<std::string_view>& out)
{
  out.clear();
  for (std::size_t b = 0;;) {
    auto const slash = s.find('/', b);
    if (slash == std::string_view::npos) {
      out.push_back(s.substr(b));
      return;
    }
    out.push_back(s.substr(b, slash - b));
    b = slash + 1;
  }
}

/// DuckDB glob semantics over a flat key: `*` / `?` / `[…]` match within one
/// '/'-segment (duckdb::Glob, the same matcher LocalFileSystem applies per
/// directory entry); `**` matches zero or more whole segments (the crawl).
/// Memoized on `(ki, pi)` — plain backtracking is exponential in the number of
/// `**` segments. @p memo is a reused scratch buffer sized `(keys.size()+1)*(pats.size()+1)`,
/// values -1 (unknown) / 0 (false) / 1 (true).
bool match_glob_segments(std::vector<std::string_view> const& keys,
                         std::size_t ki,
                         std::vector<std::string_view> const& pats,
                         std::size_t pi,
                         std::vector<std::int8_t>& memo)
{
  auto const idx = ki * (pats.size() + 1) + pi;
  if (memo[idx] != -1) { return memo[idx] != 0; }
  bool result = false;
  if (pi == pats.size()) {
    result = ki == keys.size();
  } else if (pats[pi] == "**") {
    for (std::size_t skip = ki; skip <= keys.size(); ++skip) {
      if (match_glob_segments(keys, skip, pats, pi + 1, memo)) {
        result = true;
        break;
      }
    }
  } else {
    result = ki < keys.size() &&
             duckdb::Glob(keys[ki].data(), keys[ki].size(), pats[pi].data(), pats[pi].size()) &&
             match_glob_segments(keys, ki + 1, pats, pi + 1, memo);
  }
  memo[idx] = static_cast<std::int8_t>(result ? 1 : 0);
  return result;
}

/// Fast path for a pattern with NO `**`: every pattern segment matches exactly
/// one key segment (`*`/`?`/`[…]` never cross a '/'), so the match reduces to a
/// length check plus a per-segment duckdb::Glob — no recursion, no memo. This is
/// the common glob (`root_*.parquet`, `nation_[ab].parquet`), so keeping it off
/// the memoized path avoids a per-listed-key memo write on a broad listing.
bool match_glob_no_crawl(std::vector<std::string_view> const& keys,
                         std::vector<std::string_view> const& pats)
{
  if (keys.size() != pats.size()) { return false; }
  for (std::size_t i = 0; i < keys.size(); ++i) {
    if (!duckdb::Glob(keys[i].data(), keys[i].size(), pats[i].data(), pats[i].size())) {
      return false;
    }
  }
  return true;
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
  auto sirius_ctx = resolve_gated_sirius_context(opener, path, "reading");
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

duckdb::unique_ptr<duckdb::FileHandle> sirius_httpfs::OpenFileExtended(
  const duckdb::OpenFileInfo& file,
  duckdb::FileOpenFlags flags,
  duckdb::optional_ptr<duckdb::FileOpener> opener)
{
  auto const known_size = extract_known_size(file);
  if (!known_size) { return OpenFile(file.path, flags, opener); }
  if (flags.OpenForWriting()) {
    throw duckdb::IOException("[sirius_httpfs] '" + file.path +
                              "' is read-only; S3 writes (COPY TO) are not supported");
  }
  auto sirius_ctx = resolve_gated_sirius_context(opener, file.path, "reading");
  // A parquet_footer_probe open: one suffix-range GET resolves the size (== the
  // LIST size that rode the glob expansion) and stashes the footer, so the
  // binder's footer reads are served locally (no HEAD, no separate footer GETs).
  auto datasource = sirius_ctx->get_scan_manager().create_datasource(
    file.path, sirius::io::open_hint::parquet_footer_probe);
  if (!datasource) {
    throw std::runtime_error("[sirius_httpfs] no S3 backend supports '" + file.path + "'");
  }
  return duckdb::make_uniq<sirius_httpfs_file_handle>(
    *this, file.path, flags, std::move(datasource));
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

std::string sirius_httpfs::GetVersionTag(duckdb::FileHandle& handle)
{
  return as_httpfs_handle(handle).version_tag_;
}

duckdb::vector<duckdb::OpenFileInfo> sirius_httpfs::Glob(const std::string& path,
                                                         duckdb::FileOpener* opener)
{
  if (!duckdb::FileSystem::HasGlob(path)) {
    duckdb::vector<duckdb::OpenFileInfo> result;
    if (CanHandleFile(path)) { result.emplace_back(path); }
    return result;
  }
  // Wildcard expansion needs the connection's backend (one paginated LIST) and
  // is gated exactly like OpenFile: expansion is metadata-only, but its file
  // list is only ever consumed by a GPU-only scan, so failing here gives the
  // clear error at the earliest point.
  auto sirius_ctx = resolve_gated_sirius_context(opener, path, "glob-expanding");
  return expand_glob(path, sirius_ctx->get_scan_manager());
}

namespace {

// True when @p key has a directory segment (not the final filename) that is a Hive
// partition (has '=') and also contains a literal '?'. DuckDB's Parse cancels
// partition candidacy on any '?' in a segment, so such a directory would be
// silently dropped from partitioning. An escaped '%3F' is safe.
bool key_has_literal_question_in_hive_segment(std::string_view key)
{
  std::size_t seg_start = 0;
  for (std::size_t i = 0; i < key.size(); ++i) {
    if (key[i] == '/') {  // close a directory segment; the trailing filename is skipped
      auto const seg = key.substr(seg_start, i - seg_start);
      if (seg.find('=') != std::string_view::npos && seg.find('?') != std::string_view::npos) {
        return true;
      }
      seg_start = i + 1;
    }
  }
  return false;
}

}  // namespace

duckdb::vector<duckdb::OpenFileInfo> expand_glob(
  std::string const& pattern,
  sirius::scan_manager::sirius_scan_manager& scan_manager,
  std::optional<std::size_t> max_matches)
{
  constexpr std::string_view k_scheme = "s3://";
  if (pattern.size() <= k_scheme.size()) {
    throw duckdb::IOException("[sirius_httpfs] malformed s3:// glob pattern: '" + pattern + "'");
  }
  auto const rest         = std::string_view{pattern}.substr(k_scheme.size());
  auto const bucket_slash = rest.find('/');
  auto const bucket       = rest.substr(0, bucket_slash);
  if (bucket.find_first_of("*?[") != std::string_view::npos) {
    throw duckdb::IOException(
      "[sirius_httpfs] wildcards in the bucket segment are not supported: '" + pattern + "'");
  }
  if (bucket.empty() || bucket_slash == std::string_view::npos) {
    throw duckdb::IOException("[sirius_httpfs] malformed s3:// glob pattern: '" + pattern + "'");
  }
  auto const key_pattern = rest.substr(bucket_slash + 1);

  // LIST prefix = everything up to the last '/' before the first wildcard —
  // ListObjectsV2 is prefix-indexed, so the sweep is server-side-narrowed to
  // the table's prefix before any keys flow.
  auto const first_wild = key_pattern.find_first_of("*?[");
  auto const last_slash = key_pattern.rfind('/', first_wild);
  auto const prefix     = last_slash == std::string_view::npos ? std::string_view{}
                                                               : key_pattern.substr(0, last_slash + 1);

  std::vector<std::string_view> pattern_segments;
  split_segments(key_pattern, pattern_segments);
  bool const has_crawl =
    std::find(pattern_segments.begin(), pattern_segments.end(), "**") != pattern_segments.end();
  std::string const list_uri  = "s3://" + std::string{bucket} + "/" + std::string{prefix};
  std::size_t const match_cap = max_matches.value_or(scan_manager.s3_list_max_matches(list_uri));

  // Stream the pages, keeping only matches: peak memory = one page (≤1000
  // entries) + the matched set, regardless of the prefix's population. The
  // match cap throws rather than truncating — a shortened file list would
  // silently change query results. `key_segments` / `memo` are reused across
  // keys so a broad listing does not heap-allocate per key; the `**` memo is
  // only touched when the pattern actually has a crawl segment.
  duckdb::vector<duckdb::OpenFileInfo> matches;
  std::vector<std::string_view> key_segments;
  std::vector<std::int8_t> memo;
  scan_manager.list_objects_paged(
    list_uri, /*page_size=*/1000, [&](sirius::io::rest::s3::list_objects_v2_page const& page) {
      for (auto const& entry : page.entries) {
        split_segments(entry.key, key_segments);
        bool matched;
        if (has_crawl) {
          memo.assign((key_segments.size() + 1) * (pattern_segments.size() + 1), -1);
          matched = match_glob_segments(key_segments, 0, pattern_segments, 0, memo);
        } else {
          matched = match_glob_no_crawl(key_segments, pattern_segments);
        }
        if (!matched) { continue; }
        // Fail loud on a literal '?' in a Hive partition segment (see above);
        // match-scoped, so unmatched keys under the prefix are unaffected.
        if (key_has_literal_question_in_hive_segment(entry.key)) {
          throw duckdb::IOException(
            "[sirius_httpfs] glob '" + pattern + "' matched S3 key '" + entry.key +
            "' with a literal '?' in a Hive partition segment; percent-encode it (%3F) or exclude "
            "it from the glob");
        }
        if (matches.size() >= match_cap) {
          throw duckdb::IOException("[sirius_httpfs] glob '" + pattern + "' matched more than " +
                                    std::to_string(match_cap) +
                                    " objects — narrow the glob prefix");
        }
        // Embed the raw LIST key verbatim: the signing layer RFC3986-encodes it and
        // uri_parser does not decode the s3 key, so the LIST key == the opened key.
        duckdb::OpenFileInfo info("s3://" + std::string{bucket} + "/" + std::string{entry.key});
        if (entry.size <= static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max())) {
          info.extended_info = duckdb::make_shared_ptr<duckdb::ExtendedOpenFileInfo>();
          info.extended_info->options["file_size"] =
            duckdb::Value::BIGINT(static_cast<int64_t>(entry.size));
        }
        matches.push_back(std::move(info));
      }
      return true;
    });

  std::sort(
    matches.begin(), matches.end(), [](auto const& a, auto const& b) { return a.path < b.path; });
  return matches;
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
