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

}  // namespace sirius::io
