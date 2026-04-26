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

static void extract_constant_from_stats(const duckdb::ColumnSegment& segment, uint8_t* dest)
{
  auto& stats = segment.stats.statistics;
  switch (segment.type.InternalType()) {
    case duckdb::PhysicalType::BOOL:
    case duckdb::PhysicalType::INT8: extract_typed_constant<int8_t>(stats, dest); break;
    case duckdb::PhysicalType::INT16: extract_typed_constant<int16_t>(stats, dest); break;
    case duckdb::PhysicalType::INT32: extract_typed_constant<int32_t>(stats, dest); break;
    case duckdb::PhysicalType::INT64: extract_typed_constant<int64_t>(stats, dest); break;
    case duckdb::PhysicalType::UINT8: extract_typed_constant<uint8_t>(stats, dest); break;
    case duckdb::PhysicalType::UINT16: extract_typed_constant<uint16_t>(stats, dest); break;
    case duckdb::PhysicalType::UINT32: extract_typed_constant<uint32_t>(stats, dest); break;
    case duckdb::PhysicalType::UINT64: extract_typed_constant<uint64_t>(stats, dest); break;
    case duckdb::PhysicalType::FLOAT: extract_typed_constant<float>(stats, dest); break;
    case duckdb::PhysicalType::DOUBLE: extract_typed_constant<double>(stats, dest); break;
    case duckdb::PhysicalType::INT128:
      extract_typed_constant<duckdb::hugeint_t>(stats, dest);
      break;
    default: std::memset(dest, 0, 16); break;
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
//     Hot-path single-slot cache keyed by BlockManager*. Single-DB
//     workloads (TPC-H, ClickBench) hit this with a lock-free acquire
//     load — zero mutex contention.
//
//   g_mmap_cache : map<BlockManager*, unique_ptr<mmap_file_state>>
//     Full registry guarded by g_mmap_mutex. Only touched on cold miss
//     (first query) or when switching between attached databases.
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

  // Fast path: lock-free acquire load + identity check.
  auto* last = g_mmap_last.load(std::memory_order_acquire);
  if (last && last->owner == bm) { return last; }

  // Slow path: register/init under mutex.
  std::lock_guard<std::mutex> lock(g_mmap_mutex);
  auto it = g_mmap_cache.find(bm);
  if (it != g_mmap_cache.end()) {
    g_mmap_last.store(it->second.get(), std::memory_order_release);
    return it->second.get();
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

static inline uint8_t* mmap_block_data_ptr(const mmap_file_state& mmap_state,
                                           duckdb::block_id_t block_id,
                                           size_t block_offset,
                                           size_t& out_block_offset)
{
  const auto& layout       = mmap_state.layout;
  size_t block_file_offset = layout.block_start + static_cast<size_t>(block_id) * layout.alloc_size;
  size_t data_file_offset  = block_file_offset + layout.hdr_size + block_offset;
  if (data_file_offset >= mmap_state.size) { return nullptr; }
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

      bool used_mmap = false;
      if (mmap_state && mmap_state->layout.valid && seg_info.block_id >= 0) {
        size_t adjusted_offset = 0;
        auto* mmap_ptr         = mmap_block_data_ptr(
          *mmap_state, seg_info.block_id, seg_info.block_offset, adjusted_offset);
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
      if (compression == duckdb::CompressionType::COMPRESSION_CONSTANT) {
        seg_info.compression = compression;
        seg_info.persistent  = true;
        seg_info.data_ptr    = nullptr;
        extract_constant_from_stats(segment, seg_info.constant_data);
      } else {
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
