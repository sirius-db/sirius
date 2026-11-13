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

#include <api/chunk_decode.hpp>
#include <cucascade/data/row_range_read.hpp>

#include <memory>
#include <vector>

namespace sirius::compression {

/**
 * @brief Reader state for one (caller, compressed batch).
 *
 * Created by the reader on the first read and handed back on later reads of the same batch, so
 * the parsed header, the payload source and each column's cursor (decoder state, grow-only
 * scratch, dictionary keys) are built once. The payload bytes a read stages on the device are
 * freed when that read returns; only a dictionary's keys stay with their cursor.
 *
 * Which batch the state reads is recorded in `payload`: the reader wraps the source of
 * make_payload_source in a file-local type tagged with the batch's representation type and
 * backing store, and replaces a state whose payload is not such a wrapper or tags another batch
 * or tier. The wrapper also records the stream and the memory resource of the cursors.
 *
 * One stream and one memory resource per set of cursors. The cursors allocate device memory on
 * the stream of the read that built them, from that read's resource: `scratch`'s
 * get_memory_resource(), or the current device resource when the read has no scratch. A read on
 * another stream, or with another resource, first waits for the device work of the earlier reads
 * and destroys every cursor; they are rebuilt as columns are requested. Destroying the state
 * waits for the last read too.
 *
 * Keep a state for one task, not across tasks. A caller that reads on a new stream per task (the
 * windowed merger) gets nothing from an older state but its parsed header: the first read of the
 * task blocks until the previous task's reads have run and rebuilds every cursor (their prefix
 * loads and dictionary keys included).
 *
 * The state owns a share of the batch's backing store (through `payload`, see
 * compressed_payload_sources.hpp), on top of the device memory of its cursors. While the state
 * lives, a device blob or a pinned host blob stays allocated even after the batch was downgraded
 * to another tier or dropped, and that memory is in no reservation: a downgrade of the batch
 * frees nothing until the state is destroyed or handed to a read of another batch or tier. Drop
 * the state when the task that read with it ends, and before waiting for memory that a downgrade
 * of its batch is meant to free.
 *
 * Lifetime: every memory resource a read of this state used must outlive the state (the cursors
 * free their memory through it, when they are dropped or with the state), and the stream of the
 * last read must still be a live stream when the next read starts and when the state is
 * destroyed. A scratch reservation need not outlive the read it was passed to; its memory space
 * must.
 *
 * Not thread-safe: one read at a time per state.
 */
struct compressed_reader_state final : cucascade::row_range_reader_state {
  std::shared_ptr<const simpatico::hpln_description> description;
  std::shared_ptr<simpatico::payload_source> payload;
  /// One cursor per source column (index = source column), created lazily; null until first used.
  std::vector<std::unique_ptr<simpatico::compressed_column_cursor>> cursors;
};

/**
 * @brief Registers row-range readers for compressed_device_representation,
 * compressed_host_representation and compressed_disk_representation.
 *
 * The readers decode only the 1024-row chunks covering the requested rows. A column whose plan
 * is not chunk-decodable, a projection or a filtered view of a chunk (selected columns, a
 * pushdown scan or a visibility mask), a device batch without a table and a device batch on
 * another GPU than the read's stream make the reader throw cucascade::row_range_unsupported_error,
 * so the caller falls back to a whole-batch decode.
 *
 * Exception to the reader contract of cucascade/data/row_range_read.hpp (rules 3 and 4: only
 * `k + 1` offsets are written and nothing at or after row `dst_row_offset + k` is touched). The
 * number of rows whose chars fit a STRING slot is known only once that column is decoded, and
 * the STRING columns are decoded in request order, each with the row count the columns before it
 * left. When a read requests several STRING columns and a later one delivers fewer rows `k` than
 * an earlier one was decoded with, the earlier columns are not decoded again: their first `k`
 * rows, `k + 1` offsets and `k` mask bits are exactly what a read of `k` rows writes, but their
 * slots also hold offsets, chars and mask bits of rows past `k` (never past the slot capacities
 * or the rows the read was asked for). `rows_written` and `data_bytes_written` report `k` rows
 * for every column. A caller that appends is unaffected, since its next read overwrites those
 * positions; a caller that needs the tail of a destination preserved must request one STRING
 * column per read. Fixed-width columns, and a read with at most one STRING column, follow the
 * contract exactly.
 *
 * Device memory (decode state and staged payload) is allocated on the read's stream from
 * `scratch->get_memory_resource()` when the read has a scratch reservation, the same resource
 * the built-in readers take their temporaries from; without one, from the current device
 * resource. See compressed_reader_state for the rules a state kept across reads follows.
 */
void register_compressed_row_range_readers(cucascade::row_range_reader_registry& registry);

}  // namespace sirius::compression
