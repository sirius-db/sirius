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

// Ingesting a .hpln file as a pinned compressed chunk.
//
// A pin today reads parquet, materializes it on the GPU, and compresses it on the way into the
// cache -- which is why pinning dominates the SF1000 wall clock (~151 s) while the queries it
// serves take ~9 s. A file that is ALREADY in the pinned representation needs none of that: the
// payload lands in pinned host blocks byte-for-byte as stored, so ingest is an I/O copy rather
// than a decode plus a re-compress. Nothing is decoded here and no GPU is touched.
//
// The served-from side then needs no new code at all: a compressed_host_representation built this
// way goes through the same converter as a pinned one, including the range-skipped fetch
// (CHUNK_SKIPPING_PLAN.md 6.5).

#include "compressed_representation.hpp"

#include <api/compressed_table_io.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace sirius {

/// A .hpln file staged into pinned host memory, plus what its header says is in it.
struct ingested_hpln {
  std::shared_ptr<pinned_compressed_blob> blob;
  simpatico::hpln_schema schema;
};

/// Read @p path into pinned host memory belonging to @p host_space.
///
/// The header is located by reading a speculative prefix and growing it if the parse reports
/// truncation -- .hpln carries no length prefix or footer, so its extent cannot be known without
/// parsing (see CHUNK_SKIPPING_PLAN.md 7.5). Throws std::runtime_error on a missing, truncated or
/// malformed file, since a partially ingested table must never become a pinned entry.
[[nodiscard]] ingested_hpln read_hpln_into_pinned(std::string const& path,
                                                  cucascade::memory::memory_space& host_space);

}  // namespace sirius
