/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

// File contract:
//   Walks DuckDB's per-column segment tree for a caller-supplied range of
//   row groups and returns a `column_scan_result` describing each segment's
//   on-disk byte layout (block id, offset, size, compression type) plus
//   pinned `BufferHandle`s that keep those blocks resident while the
//   downstream GPU decoder runs.
//
//   Read-only paths only. Walks the *data* segment tree; the validity tree
//   walk is deferred to PR E (gates on a `StandardColumnData::GetValidityData`
//   accessor that is not in upstream DuckDB 1.5.2). Until E lands,
//   `column_scan_result.validity` is left empty and `has_nulls` stays false —
//   E's viability check refuses any column with potential nulls so the gap
//   is plan-time-correct, not silently-wrong.

#include "op/scan/direct_block_scan.hpp"

#include <log/logging.hpp>

// duckdb
#include <duckdb/common/types/validity_mask.hpp>
#include <duckdb/common/types/vector.hpp>
#include <duckdb/function/compression_function.hpp>
#include <duckdb/storage/block_manager.hpp>
#include <duckdb/storage/buffer/buffer_handle.hpp>
#include <duckdb/storage/buffer_manager.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>
#include <duckdb/storage/statistics/numeric_stats.hpp>
#include <duckdb/storage/statistics/string_stats.hpp>
#include <duckdb/storage/storage_manager.hpp>
#include <duckdb/storage/table/column_data.hpp>
#include <duckdb/storage/table/column_segment.hpp>
#include <duckdb/storage/table/column_segment_tree.hpp>
#include <duckdb/storage/table/row_group.hpp>
#include <duckdb/storage/table/segment_tree.hpp>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {

template <typename T>
static void extract_typed_constant(const duckdb::BaseStatistics& stats, uint8_t* dest)
{
  auto v = duckdb::NumericStats::GetMin<T>(stats);
  std::memcpy(dest, &v, sizeof(v));
}

// Returns true on success (16-byte constant_data populated). Returns false
// for any PhysicalType this function doesn't know how to extract (e.g.
// VARCHAR — the value can exceed 16 bytes; INTERVAL — packed struct
// without a NumericStats::GetMin overload). Caller must NOT report
// COMPRESSION_CONSTANT downstream when this returns false; otherwise the
// downstream decoder will read uninitialized bytes.
[[nodiscard]] static bool extract_constant_from_stats(const duckdb::ColumnSegment& segment,
                                                      uint8_t* dest)
{
  auto& stats = segment.stats.statistics;
  switch (segment.type.InternalType()) {
    case duckdb::PhysicalType::BOOL:
    case duckdb::PhysicalType::INT8: extract_typed_constant<int8_t>(stats, dest); return true;
    case duckdb::PhysicalType::INT16: extract_typed_constant<int16_t>(stats, dest); return true;
    case duckdb::PhysicalType::INT32: extract_typed_constant<int32_t>(stats, dest); return true;
    case duckdb::PhysicalType::INT64: extract_typed_constant<int64_t>(stats, dest); return true;
    case duckdb::PhysicalType::UINT8: extract_typed_constant<uint8_t>(stats, dest); return true;
    case duckdb::PhysicalType::UINT16: extract_typed_constant<uint16_t>(stats, dest); return true;
    case duckdb::PhysicalType::UINT32: extract_typed_constant<uint32_t>(stats, dest); return true;
    case duckdb::PhysicalType::UINT64: extract_typed_constant<uint64_t>(stats, dest); return true;
    case duckdb::PhysicalType::FLOAT: extract_typed_constant<float>(stats, dest); return true;
    case duckdb::PhysicalType::DOUBLE: extract_typed_constant<double>(stats, dest); return true;
    case duckdb::PhysicalType::INT128:
      extract_typed_constant<duckdb::hugeint_t>(stats, dest);
      return true;
    default:
      // Telemetry: per feedback_dead_code_audit.md instrumentation pattern.
      // If this fires in CI, we know which physical type to add support for
      // (or refuse explicitly at viability in PR E).
      SIRIUS_LOG_WARN(
        "[direct_block_scan] CONSTANT segment with unsupported physical type {} — "
        "returning unsupported. Caller must NOT mark this segment COMPRESSION_CONSTANT.",
        static_cast<int>(segment.type.InternalType()));
      std::memset(dest, 0, 16);
      return false;
  }
}

//===----------------------------------------------------------------------===//
// Direct mmap — pure-extension atomic fast path
//
// For read-only databases we can mmap the .duckdb file once and compute
// block byte offsets directly, bypassing BufferManager::Pin()'s global
// mutex. On parallel scan tasks (4-8 threads × hundreds of blocks each)
// the mutex is the dominant cost; mmap drops it.
//
// Layout:
//   g_mmap_last : atomic<mmap_file_state*>
//     Hot-path single-slot cache. Lock-free acquire load. Single-DB
//     workloads (TPC-H, ClickBench) hit this for every parallel scan task.
//
//   g_mmap_cache : map<BlockManager*, unique_ptr<mmap_file_state>>
//     Full registry guarded by g_mmap_mutex. Touched on cold miss (first
//     query) or when switching between attached databases.
//
// ABA safety: BlockManager* alone is not a stable identity across
// ATTACH/DETACH cycles (DuckDB may allocate a new BlockManager at the
// same heap address as a freed one). Each cached state captures
// (st_dev, st_ino, path) at mmap time; every cache hit (fast and slow)
// re-stat()s the path and validates the file identity. Mismatches
// trigger eviction + remap. Cost: ~1µs stat per
// `try_get_mmap_for_table` call (once per (column-range × row-group) —
// negligible vs the BufferManager mutex it bypasses).
//
// Correctness: per-state layout (alloc_size, hdr_size, block_start) is
// discovered once per BlockManager via std::call_once against a real
// pinned reference block (cross-checks the mmap'd bytes match Pin()).
// Keying the layout per-BlockManager (not globally) prevents a second
// attached DB with a different block size from silently inheriting the
// first file's layout.
//
// SIRIUS_DISABLE_MMAP=1 forces Pin() for benchmarking / debugging.
//===----------------------------------------------------------------------===//

struct block_layout_info {
  size_t alloc_size  = 0;
  size_t hdr_size    = 0;
  size_t block_start = 0;
  bool valid         = false;
};

struct mmap_file_state {
  duckdb::BlockManager* owner = nullptr;
  uint8_t* base               = nullptr;
  size_t size                 = 0;
  // Stable file identity captured at mmap time — used to detect
  // BlockManager* address reuse across ATTACH/DETACH cycles. See ABA
  // safety note in file-level comment.
  std::string path;
  dev_t dev = 0;
  ino_t ino = 0;
  // Block layout is keyed per-BlockManager — see file-level comment.
  std::once_flag layout_once;
  block_layout_info layout;

  mmap_file_state()                                  = default;
  mmap_file_state(const mmap_file_state&)            = delete;
  mmap_file_state(mmap_file_state&&)                 = delete;
  mmap_file_state& operator=(const mmap_file_state&) = delete;
  mmap_file_state& operator=(mmap_file_state&&)      = delete;
  ~mmap_file_state()
  {
    // Releases the mapping if construction completed but the state was
    // never inserted into g_mmap_cache (e.g. emplace threw bad_alloc).
    // For cached entries, this fires only at process exit, where the
    // kernel would tear down the mapping anyway — but it costs nothing
    // and keeps the partial-init failure path leak-free.
    if (base) { ::munmap(base, size); }
  }
};

static std::atomic<mmap_file_state*> g_mmap_last{nullptr};
static std::mutex g_mmap_mutex;
static std::unordered_map<duckdb::BlockManager*, std::unique_ptr<mmap_file_state>> g_mmap_cache;

// Returns true if the cached state still describes the same on-disk file —
// i.e. (st_dev, st_ino) at the cached path matches what we captured at
// mmap time. Returns false if the file has been replaced (e.g. ATTACH
// cycle reusing the same path with a different file) or if the file is
// gone.
[[nodiscard]] static bool validate_file_identity(const mmap_file_state& state)
{
  struct stat st;
  if (::stat(state.path.c_str(), &st) != 0) { return false; }
  return st.st_dev == state.dev && st.st_ino == state.ino;
}

static mmap_file_state* try_get_mmap_for_table(duckdb::DataTable& storage)
{
  if (const char* env = std::getenv("SIRIUS_DISABLE_MMAP")) {
    if (env[0] != '0') { return nullptr; }
  }

  duckdb::BlockManager* bm = nullptr;
  try {
    bm = &storage.GetAttached().GetStorageManager().GetBlockManager();
  } catch (...) {
    return nullptr;
  }
  if (!bm) { return nullptr; }

  // Fast path: lock-free pointer match + stable-identity validation.
  // The stat() catches BlockManager* address reuse across ATTACH/DETACH
  // (the cached state's owner pointer is still bitwise equal to bm but
  // the underlying file has changed).
  auto* last = g_mmap_last.load(std::memory_order_acquire);
  if (last && last->owner == bm && validate_file_identity(*last)) { return last; }

  // Slow path: register/init/revalidate under mutex.
  std::lock_guard<std::mutex> lock(g_mmap_mutex);
  auto it = g_mmap_cache.find(bm);
  if (it != g_mmap_cache.end()) {
    if (validate_file_identity(*it->second)) {
      g_mmap_last.store(it->second.get(), std::memory_order_release);
      return it->second.get();
    }
    // Stale entry — file at this path has changed (or BlockManager*
    // address was reused). Drop the cached mapping; remap below.
    SIRIUS_LOG_INFO(
      "[direct_block_scan] cached mmap for {} stale (file replaced or BlockManager* reused) "
      "— evicting",
      it->second->path);
    g_mmap_cache.erase(it);
  }

  std::string path;
  try {
    path = storage.GetAttached().GetStorageManager().GetDBPath();
  } catch (...) {
    return nullptr;
  }
  if (path.empty()) { return nullptr; }

  // Allocate the state holder before mmap so that any later allocation
  // failure (e.g. unordered_map::emplace bad_alloc) leaves cleanup to
  // the unique_ptr's destructor — which munmaps via mmap_file_state's
  // destructor since `base` is set. Without this ordering, a throw
  // between mmap and the cache insert would leak the mapping.
  auto state   = std::make_unique<mmap_file_state>();
  state->owner = bm;
  state->path  = path;

  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) { return nullptr; }

  struct stat st;
  if (::fstat(fd, &st) < 0) {
    ::close(fd);
    return nullptr;
  }

  auto file_size = static_cast<size_t>(st.st_size);
  void* mapped   = ::mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
  ::close(fd);
  if (mapped == MAP_FAILED) { return nullptr; }

  state->base = static_cast<uint8_t*>(mapped);
  state->size = file_size;
  state->dev  = st.st_dev;
  state->ino  = st.st_ino;

  auto* state_ptr = state.get();
  g_mmap_cache.emplace(bm, std::move(state));
  g_mmap_last.store(state_ptr, std::memory_order_release);

  SIRIUS_LOG_INFO("[direct_block_scan] mmap'd {} ({:.1f} MB) — atomic fast path",
                  path,
                  static_cast<double>(file_size) / 1e6);
  return state_ptr;
}

static void discover_block_layout(duckdb::ColumnSegment& segment,
                                  duckdb::BufferManager& buffer_manager,
                                  mmap_file_state& mmap_state)
{
  if (!segment.block) { return; }

  auto& layout      = mmap_state.layout;
  layout.alloc_size = segment.block->GetBlockAllocSize();
  layout.hdr_size   = segment.block->GetBlockHeaderSize();
  if (layout.alloc_size == 0) { return; }

  auto pin_handle = buffer_manager.Pin(segment.block);
  auto block_id   = segment.GetBlockId();
  auto block_off  = segment.GetBlockOffset();
  auto* pin_ptr   = pin_handle.Ptr() + block_off;

  // SingleFileBlockManager::BLOCK_START is FILE_HEADER_SIZE * 3 = 4096 * 3.
  constexpr size_t CANDIDATE_BLOCK_START = 4096 * 3;
  size_t byte_offset = CANDIDATE_BLOCK_START + static_cast<size_t>(block_id) * layout.alloc_size +
                       layout.hdr_size + block_off;

  if (byte_offset + 64 > mmap_state.size) { return; }
  auto* mmap_ptr = mmap_state.base + byte_offset;

  if (std::memcmp(mmap_ptr, pin_ptr, std::min<size_t>(64, segment.SegmentSize())) != 0) {
    SIRIUS_LOG_ERROR("[direct_block_scan] mmap validation FAILED — disabling mmap bypass.");
    return;
  }

  layout.block_start = CANDIDATE_BLOCK_START;
  layout.valid       = true;
  SIRIUS_LOG_INFO("[direct_block_scan] block layout ok: alloc={} hdr={} start={}",
                  layout.alloc_size,
                  layout.hdr_size,
                  layout.block_start);
}

// Returns the mmap'd address for a segment's data, or nullptr if the
// requested [data_file_offset, data_file_offset + segment_size) range
// would extend past the mapped file (truncated / corrupt file, or an
// unexpected layout). Callers must fall back to BufferManager::Pin on
// nullptr.
//
// `out_block_offset` is set to (hdr_size + block_offset) so that
// `data_ptr - out_block_offset` points to the start of the on-disk
// block (header + data). The downstream bulk H2D step copies
// `alloc_size` bytes from `(data_ptr - block_offset)` and must land
// within the block boundary — see the file-level note on path-dependent
// `block_offset` semantics.
static inline uint8_t* mmap_block_data_ptr(const mmap_file_state& mmap_state,
                                           duckdb::block_id_t block_id,
                                           size_t block_offset,
                                           size_t segment_size,
                                           size_t& out_block_offset)
{
  const auto& layout       = mmap_state.layout;
  size_t block_file_offset = layout.block_start + static_cast<size_t>(block_id) * layout.alloc_size;
  size_t data_file_offset  = block_file_offset + layout.hdr_size + block_offset;
  // Bounds-check the FULL segment range, not just the first byte.
  // Downstream decoders read `segment_size` bytes starting at the
  // returned pointer. A truncated / corrupt file or a stale block
  // layout could otherwise OOB into unmapped memory and SIGSEGV.
  if (data_file_offset > mmap_state.size || segment_size > mmap_state.size - data_file_offset) {
    return nullptr;
  }
  out_block_offset = layout.hdr_size + block_offset;
  return mmap_state.base + data_file_offset;
}

//===----------------------------------------------------------------------===//
// Segment tree scan
//
// Walks the segment tree of a single ColumnData, producing a
// `direct_block_scan_result` whose `segments` vector has one entry per
// segment. For each persistent segment with a backing block, take the mmap
// fast path when available (zero-copy direct pointer into the mmap'd file)
// and fall back to BufferManager::Pin when mmap is unavailable, gated off,
// or layout validation failed.
//
// COMPRESSION_CONSTANT segments are blockless — the value lives in segment
// stats. We extract it into segment_info::constant_data (16 bytes inline)
// and leave data_ptr null; the downstream decoder fills the column with the
// constant.
//===----------------------------------------------------------------------===//

static direct_block_scan_result scan_segment_tree(duckdb::ColumnData& col_data,
                                                  duckdb::BufferManager& buffer_manager,
                                                  bool* has_nulls_out,
                                                  mmap_file_state* mmap_state)
{
  direct_block_scan_result result;
  auto& seg_tree = col_data.GetSegmentTree();
  auto seg_node  = seg_tree.GetRootSegment();

  while (seg_node) {
    auto& segment = seg_node->GetNode();
    direct_block_scan_result::segment_info seg_info;
    seg_info.row_count = segment.count.load();

    if (has_nulls_out && segment.stats.statistics.CanHaveNull()) { *has_nulls_out = true; }

    if (segment.block) {
      seg_info.block_offset = segment.GetBlockOffset();
      seg_info.segment_size = segment.SegmentSize();
      seg_info.block_id     = segment.GetBlockId();
      seg_info.persistent   = true;
      seg_info.compression  = segment.GetCompressionFunction().type;
      result.total_pinned_bytes += segment.SegmentSize();

      if (mmap_state) {
        std::call_once(mmap_state->layout_once,
                       discover_block_layout,
                       std::ref(segment),
                       std::ref(buffer_manager),
                       std::ref(*mmap_state));
      }

      // Note on the path-dependent meaning of `seg_info.block_offset`:
      // The downstream bulk H2D step copies `alloc_size` bytes starting
      // at `(seg_info.data_ptr - seg_info.block_offset)` to land the
      // entire block (header + data) on the GPU. For this arithmetic to
      // work, `block_offset` must be the distance from `data_ptr` back
      // to the start of the block.
      //
      //   - Pin path: `data_ptr = handle.Ptr() + GetBlockOffset()`.
      //     `handle.Ptr()` already points to the start of the block
      //     (BufferManager strips the header), so `block_offset` stays
      //     `GetBlockOffset()` (the on-block offset to the data).
      //
      //   - mmap path: `data_ptr = mmap_base + block_file_offset +
      //     hdr_size + GetBlockOffset()`. The file contains the header
      //     bytes, so to step back to the block start we need
      //     `(hdr_size + GetBlockOffset())`. mmap_block_data_ptr
      //     returns this adjusted value via `out_block_offset` and we
      //     overwrite `block_offset` with it.
      //
      // The asymmetry is intentional. Renaming the field to something
      // like `data_ptr_to_block_start_offset` would be clearer but
      // touches every consumer; for now the comment is the contract.
      bool used_mmap = false;
      if (mmap_state && mmap_state->layout.valid && seg_info.block_id >= 0) {
        size_t adjusted_offset = 0;
        auto* mmap_ptr         = mmap_block_data_ptr(*mmap_state,
                                             seg_info.block_id,
                                             seg_info.block_offset,
                                             seg_info.segment_size,
                                             adjusted_offset);
        if (mmap_ptr) {
          seg_info.data_ptr     = mmap_ptr;
          seg_info.block_offset = adjusted_offset;
          used_mmap             = true;
        }
      }
      if (!used_mmap) {
        seg_info.handle   = buffer_manager.Pin(segment.block);
        seg_info.data_ptr = seg_info.handle.Ptr() + seg_info.block_offset;
      }

      if (duckdb::StringStats::HasMaxStringLength(segment.stats.statistics)) {
        seg_info.max_string_length = duckdb::StringStats::MaxStringLength(segment.stats.statistics);
      }
    } else {
      auto compression = segment.GetCompressionFunction().type;
      if (compression == duckdb::CompressionType::COMPRESSION_CONSTANT &&
          extract_constant_from_stats(segment, seg_info.constant_data)) {
        seg_info.compression = compression;
        seg_info.persistent  = true;
        seg_info.data_ptr    = nullptr;
      } else {
        // Either this is a transient (block-less, non-CONSTANT) segment, or
        // the CONSTANT segment's physical type has no extractor here (e.g.
        // VARCHAR — value can exceed 16 bytes; INTERVAL — packed struct).
        // Either way, mark as unsupported (COMPRESSION_AUTO) so the
        // caller's viability check refuses this segment and routes the
        // query to the CPU scan path.
        seg_info.data_ptr    = nullptr;
        seg_info.persistent  = false;
        seg_info.compression = duckdb::CompressionType::COMPRESSION_AUTO;
      }
    }

    result.segments.push_back(std::move(seg_info));
    seg_node = seg_tree.GetNextSegment(*seg_node);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Public API
//
// `column_scan_result.validity` is intentionally left unpopulated and
// `has_nulls` is the data-side has-null indicator only — it does not yet
// reflect a peek at the validity tree. PR E will populate validity once the
// `StandardColumnData::GetValidityData` accessor is available, and will
// gate gpu_execution at viability so any nullable column falls back to
// duckdb_scan_task until then.
//===----------------------------------------------------------------------===//

column_scan_result direct_block_scan_column_range(duckdb::DataTable& storage,
                                                  duckdb::StorageIndex col_idx,
                                                  duckdb::ClientContext& context,
                                                  const std::vector<duckdb::RowGroup*>& row_groups)
{
  auto& buffer_manager = duckdb::BufferManager::GetBufferManager(context);
  auto* mmap_state     = try_get_mmap_for_table(storage);

  column_scan_result result;
  result.data.total_rows     = 0;
  result.validity.total_rows = 0;

  auto start = std::chrono::steady_clock::now();

  for (auto* rg : row_groups) {
    auto& col_data = rg->GetColumnDirect(col_idx);

    auto data_scan = scan_segment_tree(col_data, buffer_manager, &result.has_nulls, mmap_state);
    result.data.total_pinned_bytes += data_scan.total_pinned_bytes;
    for (auto& s : data_scan.segments) {
      result.data.total_rows += s.row_count;
      result.data.segments.push_back(std::move(s));
    }
  }

  result.validity.total_rows = result.data.total_rows;

  auto end = std::chrono::steady_clock::now();
  result.data.pin_time_us =
    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  result.validity.pin_time_us = result.data.pin_time_us;

  return result;
}

}  // namespace sirius::op::scan
