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

#pragma once

#include <cucascade/cudf/datasource.hpp>  // cucascade::io::{datasource, ioctx, open_hint}

#include <memory>
#include <string>
#include <string_view>

namespace sirius::io {

/**
 * @brief Strip a leading `file:` scheme (case-insensitive) so @p path can be
 *        handed to a local-file backend, normalizing the local path the way
 *        DuckDB does.
 *
 * Anything else — bare absolute paths, `s3://`, `gs://`, ... — is returned
 * unchanged, so this is safe to apply unconditionally at an I/O boundary.
 *
 * Iceberg manifests written by the Apache implementations record fully-qualified
 * URIs (`file:///abs/path/x.parquet`), while DuckDB's multi-file binder and our
 * own fixtures generally carry bare paths. The local backends only open bare
 * paths, so an un-stripped URI reaches `create_io_object` and throws
 * "unsupported path" — which surfaces as a RUNTIME GPU fallback, not a clean
 * plan-time decline.
 *
 * Why Sirius keeps its own: cuCascade's `cucascade::io::strip_file_scheme` is
 * duckdb-free and only strips the scheme (plus percent-decoding). This one parses
 * the `file:` URI with @c duckdb::Path, so the stripped local path is normalized
 * (empty and `.` segments dropped, `x/..` folded). Sirius's callers
 * (`normalize_path` in the scan manager, the iceberg/puffin readers, plan_get)
 * compare and key on these paths and rely on that normalization, so they keep
 * this implementation; cuCascade's is not a drop-in replacement.
 *
 * Never throws.
 */
[[nodiscard]] std::string strip_file_scheme(std::string_view path);

/**
 * @brief Open a datasource for @p path on @p io_ctx after normalizing it with
 *        @ref strip_file_scheme.
 *
 * Every Sirius open goes through here rather than @c cucascade::io::open_datasource
 * directly. The old @c ioctx::open_datasource member normalized with duckdb::Path
 * semantics (this file's @ref strip_file_scheme); cuCascade's free function applies
 * only its own scheme strip, which does not fold `.`/`..` or empty segments. The
 * io_object's @c raw_file_cache_id keys the prefetching cache (fs_cache) and the
 * metadata_store, so a path must be canonical the same way the scan manager's
 * @c normalize_path() makes it, or one file is cached under two keys and a
 * metadata_store lookup by the normalized path misses.
 *
 * @throws Whatever @c cucascade::io::open_datasource throws (unsupported or
 *         unreachable path, null @p io_ctx).
 */
[[nodiscard]] std::unique_ptr<cucascade::io::datasource> open_datasource(
  std::shared_ptr<cucascade::io::ioctx> io_ctx, std::string path);

/// As above, forwarding @p hint to the backend (e.g. @c open_hint::parquet_footer_probe).
[[nodiscard]] std::unique_ptr<cucascade::io::datasource> open_datasource(
  std::shared_ptr<cucascade::io::ioctx> io_ctx, std::string path, cucascade::io::open_hint hint);

}  // namespace sirius::io
