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

// `COPY (SELECT ...) TO 'x.hpln' (FORMAT simpatico)` -- the writer side of the .hpln source.
//
// Until this existed a .hpln could only be produced by calling sirius::write_tables_to_hpln from
// C++, so nobody could make one without building Sirius. The file this writes is the same
// container read_simpatico() reads: logical types, per-group zone maps and a chunk directory,
// produced by the same writer -- this is the SQL surface over it, not a second writer.

#include <duckdb/function/copy_function.hpp>

#include <cstddef>

namespace sirius::compression {

/// Rows per file chunk when `chunk_rows` is not given.
///
/// A chunk is the unit a scan walks as one split and the unit pruning drops whole, so it wants to
/// be large enough that the per-chunk metadata is negligible and small enough that dropping one
/// is worth something. 1 Mi rows is both, and it is a whole multiple of the 8192-row zone-map
/// group and of simpatico's 1024-row decode chunk, so groups never straddle a chunk boundary.
inline constexpr std::size_t kDefaultHplnChunkRows = 1024UL * 1024UL;

/// The `simpatico` copy function, ready for Catalog::CreateCopyFunction.
[[nodiscard]] duckdb::CopyFunction make_simpatico_copy_function();

}  // namespace sirius::compression
