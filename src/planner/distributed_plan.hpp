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

#include "exchange/fabric.hpp"
#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace cucascade::memory {
class memory_space;
}

namespace sirius::exchange {
class exchange_service;
}

namespace sirius::exec {
class query_lifecycle_registry;
}

namespace duckdb {
class ClientContext;
}

namespace sirius::op {
class sirius_physical_exchange_sink;
class sirius_physical_hash_join;
class sirius_physical_sort_sample;
class sirius_physical_table_scan;
}  // namespace sirius::op

namespace sirius::op::scan {
class parquet_ingestible_table_info;
}

namespace sirius {
struct operator_params;
}

namespace sirius::planner {

struct connector;

/**
 * @brief Rewrites a plan, after insert_gpu_pipeline_operators, to run on every rank of
 * @p service, each over its own share of the scanned files.
 *
 * Exchanges go where the single-process plan already partitions; "The plan rewrite" in
 * docs/super-sirius/distributed-sirius.md lists every plan shape and its exchange. Rank 0's
 * result is the query's; the other ranks' are empty, or meaningless after a gather.
 *
 * Every rank must plan the same query: before returning, this blocks until every rank of
 * @p service has planned it, comparing the plans' shapes in an all-gather.
 *
 * @param query This query's number, the same on every rank: its exchanges' stream ids.
 * @param lifecycle Gates the exchange sources' landing into their ports' repositories.
 * @throws sirius::not_implemented_exception on an operator distributed execution does not
 *         support, rather than a wrong answer; this rank throws before the all-gather.
 * @throws sirius::invalid_input_exception when another rank planned a different shape, on every
 *         rank.
 */
void distribute_plan(duckdb::unique_ptr<op::sirius_physical_operator>& plan,
                     std::shared_ptr<exchange::exchange_service> service,
                     cucascade::memory::memory_space& gpu,
                     std::uint64_t query,
                     exec::query_lifecycle_registry& lifecycle);

/// Bytes `[start, start + length)` of file `file` (an index into the table's files).
struct file_byte_range {
  std::size_t file     = 0;
  std::uint64_t start  = 0;
  std::uint64_t length = 0;

  bool operator==(file_byte_range const&) const = default;
};

/**
 * @brief The byte ranges rank @p rank of @p world scans, of files of @p file_bytes bytes each.
 *
 * The files are laid end to end and cut into @p world equal spans; rank r gets span r, as one
 * range per file it touches, in file order. The spans tile every file exactly, so with row groups
 * owned by the range holding their start offset (`parquet_byte_range.hpp`), every row group is
 * read by exactly one rank. A rank may get no row group at all, e.g. of a one-row-group table.
 */
[[nodiscard]] std::vector<file_byte_range> byte_ranges_for_rank(
  std::span<std::uint64_t const> file_bytes, std::size_t rank, std::size_t world);

/**
 * @brief Sets @p info's files to this rank's share of @p files, or to all of them when
 * @p op_params is not distributed.
 *
 * Every rank binds every file of a table. Local files are split by byte range
 * (`byte_ranges_for_rank`), so a rank may get no range at all; remote ones, which take no byte
 * ranges, are dealt whole: files rank, rank + world, ...
 *
 * @throws sirius::not_implemented_exception for a remote table of fewer files than ranks, or a
 *         scan that reads `filename` or `file_index`, which each rank would number from 0.
 */
void assign_files_to_rank(op::scan::parquet_ingestible_table_info& info,
                          std::vector<std::string> files,
                          operator_params const& op_params);

/**
 * @brief Refuses, in distributed mode, a scan that @p source would lower to a read no rank split:
 * a DuckDB table, `sirius_read_parquet` of one URI, or a Parquet scan whose plan carries its own
 * byte ranges. Each rank would read all of it, and the query's answer would count it once per rank.
 *
 * The plan generator calls it once per scan, before lowering, only when distributed; the other
 * refusals (`filename`/`file_index`, too few remote files) need the bound files and stay in
 * assign_files_to_rank.
 *
 * @throws sirius::not_implemented_exception naming what cannot be split.
 */
void check_distributed_scan(op::sirius_physical_table_scan const& scan,
                            connector const& source,
                            duckdb::ClientContext& context);

/// Has @p join merge its dynamic filters across ranks with @p merge, its Blooms sized for
/// @p bloom_keys keys, and close its publication when its build CONCAT (`children[1]`) finalizes:
/// a peer's merge waits on this rank's part, and that peer's probe may be what this rank's probe
/// waits on, so the join's own finalize is too late.
void install_filter_merge(op::sirius_physical_hash_join& join,
                          std::shared_ptr<op::cross_rank_filter_merge> merge,
                          std::size_t bloom_keys);

/// Has @p sample take its boundaries from every rank's sample, all-gathered on @p stream, and
/// give the range exchange's boundaries to @p sink, the EXCHANGE_SINK after it, in whose order
/// the samples are merged. @p sink must outlive @p sample's boundary task.
void install_sort_boundaries(op::sirius_physical_sort_sample& sample,
                             op::sirius_physical_exchange_sink& sink,
                             std::shared_ptr<exchange::exchange_service> service,
                             exchange::stream_id stream,
                             cucascade::memory::memory_space& gpu);

}  // namespace sirius::planner
