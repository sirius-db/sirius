/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#include "op/scan/direct_block_scan.hpp"

#include <log/logging.hpp>

#include <duckdb/storage/storage_manager.hpp>

// duckdb
#include <duckdb/storage/buffer/buffer_handle.hpp>
#include <duckdb/storage/buffer_manager.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>
#include <duckdb/storage/statistics/numeric_stats.hpp>
#include <duckdb/storage/statistics/string_stats.hpp>
#include <duckdb/storage/table/column_data.hpp>
#include <duckdb/storage/table/column_segment.hpp>
#include <duckdb/storage/table/column_segment_tree.hpp>
#include <duckdb/storage/table/row_group.hpp>
#include <duckdb/storage/table/row_group_collection.hpp>
#include <duckdb/storage/table/segment_tree.hpp>
#include <duckdb/storage/table/standard_column_data.hpp>

#include <cuda_runtime.h>
#include <sys/mman.h>
#include <cstdlib>
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22  // Linux 5.14+, may not be in older glibc headers
#endif
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace sirius::op::scan {

/// Extract a numeric constant value from segment stats into a byte buffer.
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
// Direct mmap — bypass BufferManager::Pin() for read-only databases
//
// Thread-safe: mmap cache uses a mutex, block layout is discovered once
// via std::call_once and published atomically.
//===----------------------------------------------------------------------===//

/// Per-file mmap state.
struct mmap_file_state {
  uint8_t* base = nullptr;
  size_t size   = 0;
};

static std::mutex g_mmap_mutex;
static std::unordered_map<std::string, mmap_file_state> g_mmap_cache;

/// Block layout discovered at runtime from the first segment's BlockHandle.
/// Immutable after initialization — safe to read from any thread.
struct block_layout_info {
  size_t alloc_size    = 0;   // e.g. 262144 (256KB)
  size_t hdr_size      = 0;   // e.g. 8
  size_t block_start   = 0;   // file offset where block 0 begins (discovered, not hardcoded)
  bool   valid         = false;
};

static std::once_flag g_layout_once;
static block_layout_info g_layout;

static mmap_file_state* get_or_create_mmap(const std::string& path)
{
  std::lock_guard<std::mutex> lock(g_mmap_mutex);
  auto it = g_mmap_cache.find(path);
  if (it != g_mmap_cache.end()) return &it->second;

  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return nullptr;

  struct stat st;
  if (::fstat(fd, &st) < 0) { ::close(fd); return nullptr; }

  auto file_size = static_cast<size_t>(st.st_size);
  void* mapped = ::mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
  ::close(fd);
  if (mapped == MAP_FAILED) return nullptr;

  auto& state = g_mmap_cache[path];
  state.base = static_cast<uint8_t*>(mapped);
  state.size = file_size;

  // Whether to prefault pages. Default: auto-detect.
  //   - PCIe systems (RTX 6000 etc.): MUST prefault — CUDA's internal staging
  //     thread segfaults on lazy mmap page faults during cudaMemcpyAsync.
  //   - Unified-memory systems (GH200): prefault is unnecessary — ATS handles
  //     page faults via the GPU's copy engine transparently. Skipping prefault
  //     saves ~16-19s per cold query (the time to read the whole DB file).
  //
  // Auto-detection uses cudaDeviceProp::pageableMemoryAccessUsesHostPageTables
  // (TRUE on GH200, FALSE on PCIe). Override via SIRIUS_MMAP_PREFAULT=0/1.
  auto should_prefault = []() {
    if (const char* env = std::getenv("SIRIUS_MMAP_PREFAULT")) {
      return env[0] != '0';
    }
    int dev = 0;
    cudaGetDevice(&dev);
    int uses_host_pt = 0;
    if (cudaDeviceGetAttribute(&uses_host_pt,
                               cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev) != cudaSuccess) {
      return true;  // Safe default: prefault
    }
    return uses_host_pt == 0;  // PCIe (0) → prefault; unified memory (1) → skip
  };

  if (should_prefault()) {
    // Force all pages into RAM so cudaMemcpyAsync never hits a page fault.
    // MADV_POPULATE_READ (Linux 5.14+) faults in all pages synchronously.
    if (::madvise(mapped, file_size, MADV_POPULATE_READ) != 0) {
      // Fallback: touch every page manually
      volatile uint8_t sink = 0;
      auto* p = static_cast<uint8_t*>(mapped);
      for (size_t off = 0; off < file_size; off += 4096) {
        sink = p[off];
      }
      (void)sink;
      SIRIUS_LOG_INFO("[direct_block_scan] mmap'd {} ({:.1f} MB) — pages populated (fallback)",
                      path, file_size / 1e6);
    } else {
      SIRIUS_LOG_INFO("[direct_block_scan] mmap'd {} ({:.1f} MB) — pages populated",
                      path, file_size / 1e6);
    }
  } else {
    // Lazy page faults — suitable for unified-memory systems (GH200).
    // Hint sequential access so the kernel can prefetch on demand.
    ::madvise(mapped, file_size, MADV_SEQUENTIAL);
    SIRIUS_LOG_INFO("[direct_block_scan] mmap'd {} ({:.1f} MB) — lazy faults (unified memory)",
                    path, file_size / 1e6);
  }
  return &state;
}

static mmap_file_state* try_get_mmap_for_table(duckdb::DataTable& storage)
{
  // SIRIUS_DISABLE_MMAP=1 forces the Pin() path (DuckDB buffer manager).
  // Used to measure the cost of mmap vs DuckDB-managed block access.
  if (const char* env = std::getenv("SIRIUS_DISABLE_MMAP")) {
    if (env[0] != '0') return nullptr;
  }
  try {
    auto& db   = storage.GetAttached();
    auto& smgr = db.GetStorageManager();
    auto path  = smgr.GetDBPath();
    if (!path.empty()) return get_or_create_mmap(path);
  } catch (...) {}
  return nullptr;
}

/// Discover block layout by comparing one Pin() pointer against the mmap base.
/// Called exactly once via std::call_once. If anything fails, g_layout.valid stays false
/// and we fall back to Pin() for everything.
static void discover_block_layout(duckdb::ColumnSegment& segment,
                                  duckdb::BufferManager& buffer_manager,
                                  const mmap_file_state& mmap_state)
{
  if (!segment.block) return;

  g_layout.alloc_size = segment.block->GetBlockAllocSize();
  g_layout.hdr_size   = segment.block->GetBlockHeaderSize();

  if (g_layout.alloc_size == 0) return;

  // Pin the block to get the canonical host pointer
  auto pin_handle  = buffer_manager.Pin(segment.block);
  auto block_id    = segment.GetBlockId();
  auto block_off   = segment.GetBlockOffset();
  auto* pin_ptr    = pin_handle.Ptr() + block_off;

  // Reverse-engineer BLOCK_START:
  //   pin_ptr == mmap_base + BLOCK_START + block_id * alloc_size + hdr_size + block_off
  // So:
  //   BLOCK_START = (pin_ptr - mmap_base) - block_id * alloc_size - hdr_size - block_off
  //
  // But pin_ptr is a DuckDB buffer pool pointer, NOT an mmap pointer.
  // We can't subtract them directly. Instead, compute the expected mmap pointer
  // and validate that the data matches.

  // DuckDB file layout: 3 header pages × FILE_HEADER_SIZE, then blocks.
  // FILE_HEADER_SIZE is 4096 in all known DuckDB versions.
  // Try the known value and validate.
  constexpr size_t CANDIDATE_BLOCK_START = 4096 * 3;  // 12288

  size_t byte_offset = CANDIDATE_BLOCK_START
                     + static_cast<size_t>(block_id) * g_layout.alloc_size
                     + g_layout.hdr_size + block_off;

  if (byte_offset + 64 > mmap_state.size) return;  // OOB

  auto* mmap_ptr = mmap_state.base + byte_offset;

  // Validate: first 64 bytes must match Pin() data
  if (std::memcmp(mmap_ptr, pin_ptr, std::min<size_t>(64, segment.SegmentSize())) != 0) {
    SIRIUS_LOG_ERROR(
      "[direct_block_scan] mmap validation FAILED — block_id={} offset={} "
      "mmap_data={:02x}{:02x}{:02x}{:02x} pin_data={:02x}{:02x}{:02x}{:02x}. "
      "Disabling mmap bypass.",
      block_id, block_off,
      mmap_ptr[0], mmap_ptr[1], mmap_ptr[2], mmap_ptr[3],
      pin_ptr[0], pin_ptr[1], pin_ptr[2], pin_ptr[3]);
    return;  // g_layout.valid stays false
  }

  g_layout.block_start = CANDIDATE_BLOCK_START;
  g_layout.valid       = true;
  SIRIUS_LOG_INFO(
    "[direct_block_scan] block layout discovered: alloc_size={}, hdr_size={}, "
    "block_start={} — mmap data VALIDATED",
    g_layout.alloc_size, g_layout.hdr_size, g_layout.block_start);
}

/// Compute mmap pointer for a segment's data within a block.
/// Returns the data pointer (after header + block_offset), or nullptr if OOB.
///
/// IMPORTANT: The bulk H2D in compute_task does `data_ptr - block_offset` to get
/// the block base, then copies DUCKDB_BLOCK_SIZE (= alloc_size) bytes.
/// With Pin(), this base is Ptr() which is `heap_buf + hdr_size`, and the heap
/// buffer has alloc_size bytes from Ptr() due to allocator padding.
/// With mmap, there's NO padding — the file is packed. To avoid overreading
/// by hdr_size bytes, we set data_ptr and out_block_offset so that:
///   data_ptr - out_block_offset = block_start (BEFORE header)
///   copy of alloc_size bytes from block_start stays within the block
///
/// We achieve this by pointing data_ptr to `block_start + (hdr_size + block_offset)`
/// and setting out_block_offset to `hdr_size + block_offset`.
static inline uint8_t* mmap_block_data_ptr(const mmap_file_state& mmap_state,
                                           duckdb::block_id_t block_id,
                                           size_t block_offset,
                                           size_t& out_block_offset)
{
  size_t block_file_offset = g_layout.block_start
                           + static_cast<size_t>(block_id) * g_layout.alloc_size;
  size_t data_file_offset  = block_file_offset + g_layout.hdr_size + block_offset;
  if (data_file_offset >= mmap_state.size) return nullptr;
  // Adjust block_offset to include header so that (data_ptr - block_offset) = block_start
  out_block_offset = g_layout.hdr_size + block_offset;
  return mmap_state.base + data_file_offset;
}

//===----------------------------------------------------------------------===//
// Segment tree scan
//===----------------------------------------------------------------------===//

direct_block_scan_result scan_segment_tree(duckdb::ColumnData& col_data,
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
      seg_info.compression  = segment.GetCompressionType();
      result.total_pinned_bytes += segment.SegmentSize();

      // Discover block layout once (thread-safe)
      if (mmap_state) {
        std::call_once(g_layout_once, discover_block_layout,
                       std::ref(segment), std::ref(buffer_manager), std::cref(*mmap_state));
      }

      bool used_mmap = false;
      if (mmap_state && g_layout.valid && seg_info.block_id >= 0) {
        size_t adjusted_offset = 0;
        auto* mmap_ptr = mmap_block_data_ptr(*mmap_state, seg_info.block_id,
                                             seg_info.block_offset, adjusted_offset);
        if (mmap_ptr) {
          seg_info.data_ptr     = mmap_ptr;
          seg_info.block_offset = adjusted_offset;  // includes hdr_size for safe bulk H2D
          used_mmap = true;
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
      auto compression = segment.GetCompressionType();
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
//===----------------------------------------------------------------------===//

column_scan_result direct_block_scan_column_range(duckdb::DataTable& storage,
                                                  duckdb::StorageIndex col_idx,
                                                  duckdb::ClientContext& context,
                                                  const std::vector<duckdb::RowGroup*>& row_groups)
{
  auto& buffer_manager = duckdb::BufferManager::GetBufferManager(context);
  auto* mmap_state = try_get_mmap_for_table(storage);

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

    try {
      auto& std_col  = col_data.Cast<duckdb::StandardColumnData>();
      auto& val_data = std_col.GetValidityData();

      if (result.has_nulls) {
        auto val_scan = scan_segment_tree(val_data, buffer_manager, nullptr, mmap_state);
        result.validity.total_pinned_bytes += val_scan.total_pinned_bytes;
        for (auto& s : val_scan.segments) {
          result.validity.segments.push_back(std::move(s));
        }
      }
    } catch (...) {}
  }

  result.validity.total_rows = result.data.total_rows;

  auto end = std::chrono::steady_clock::now();
  result.data.pin_time_us =
    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  result.validity.pin_time_us = result.data.pin_time_us;

  return result;
}

}  // namespace sirius::op::scan
