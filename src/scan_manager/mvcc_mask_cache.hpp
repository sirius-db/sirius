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

#include "scan_manager/mvcc_chunk_mask.hpp"

#include <duckdb/common/typedefs.hpp>

#include <mutex>

namespace sirius::scan_manager {

/**
 * @brief The snapshot facts deciding whether a keep-mask set can cross queries. Captured
 *        per mask-job request, just before the masks are built or probed.
 */
struct mvcc_mask_snapshot_key {
  duckdb::transaction_t last_commit{0};
  duckdb::transaction_t start_time{0};
  bool changes_made{false};
};

/**
 * @brief True when a mask set built under @p built may enter the version cache.
 *
 * Both conditions are required:
 *  - the builder made no writes, since a transaction sees its own uncommitted deletes;
 *  - its snapshot covers every commit existing at build (start_time >= last_commit), else
 *    the set misses commits in (start_time, last_commit] a later query would see under an
 *    unchanged last_commit.
 *
 * Exact because DuckDB assigns start times and commit ids from one counter under the lock
 * that publishes @c last_commit: a concurrent commit gets an id above the builder's
 * snapshot and increments last_commit before any transaction observing it begins.
 */
[[nodiscard]] inline bool mvcc_mask_cache_publishable(mvcc_mask_snapshot_key const& built)
{
  return !built.changes_made && built.start_time >= built.last_commit;
}

/**
 * @brief True when a set built under @p built serves @p query exactly: last_commit
 *        unchanged, the query's snapshot covers it, and the query is writer-free.
 */
[[nodiscard]] inline bool mvcc_mask_cache_reusable(mvcc_mask_snapshot_key const& built,
                                                   mvcc_mask_snapshot_key const& query)
{
  return mvcc_mask_cache_publishable(built) && !query.changes_made &&
         query.last_commit == built.last_commit && query.start_time >= built.last_commit;
}

/**
 * @brief Per-entry cache of the last keep-mask set, keyed by the snapshot it was built
 *        under.
 *
 * Single-version: publishing under a new key replaces the set, so at most one table
 * version of pinned mask memory is retained per entry. Probe and publish are short copies
 * under @ref mutex; word storage is shared, never duplicated.
 */
struct mvcc_mask_version_cache {
  std::mutex mutex;
  bool valid{false};
  mvcc_mask_snapshot_key built;
  mvcc_chunk_mask_set masks;
};

}  // namespace sirius::scan_manager
