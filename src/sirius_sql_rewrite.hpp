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

#include <string>

namespace sirius {

/// Rewrite Sirius-owned remote parquet calls in a SQL query string.
///
/// Scans @p query with SQL-token awareness — single-quoted string literals,
/// double-quoted identifiers, and `--` line / `/* */` block comments are
/// skipped — and rewrites every `read_parquet('s3://…')` table-function call
/// to `sirius_read_parquet('s3://…')`. A call is rewritten only when its first
/// argument is a single-quoted string literal whose value begins with the
/// `s3://` scheme. The function-name match is case-insensitive and
/// word-bounded, so adjacent functions (`parquet_scan`, `read_parquet_mr`,
/// `parquet_metadata`) and local-path arguments (`/tmp/…`, `file://…`) are
/// left untouched.
///
/// This is the entry point that lets `gpu_execution` route `s3://` parquet
/// reads through Sirius's own bind (`sirius_read_parquet`) instead of DuckDB's
/// native `read_parquet`, which has no S3 filesystem.
std::string rewrite_sirius_owned_remote_parquet_calls(std::string const& query);

/// True iff @p query contains at least one Sirius-owned remote parquet call —
/// `read_parquet('s3://…')` as a real, rewritable call (comment / quote /
/// word-boundary aware, case-insensitive). Equivalent to "the rewrite would
/// change this query". Used to decide that a failed GPU query has no CPU
/// fallback: S3 parquet is read only on the GPU path, and DuckDB's CPU reader
/// has no S3 filesystem, so an s3:// query cannot fall back to CPU.
bool references_sirius_owned_s3_parquet(std::string const& query);

}  // namespace sirius
