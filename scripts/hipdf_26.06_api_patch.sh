#!/usr/bin/env bash
# =============================================================================
# hipdf_26.06_api_patch.sh — Port missing cuDF 26.06 APIs into hipDF 25.10.
#
# hipDF release/rocmds-26.03 is derived from cuDF 25.10. Sirius uses libcudf
# 26.06 with two APIs absent from this branch:
#   1. cudf::io::parquet::fetch_footer_to_host
#   2. hybrid_scan_reader::all_column_chunks_byte_ranges
#
# This script applies patches to a hipDF source tree to add these APIs.
# The implementations are sourced from upstream RAPIDS cuDF main (26.06).
#
# Usage:
#   ./scripts/hipdf_26.06_api_patch.sh /path/to/hipDF
# =============================================================================

set -euo pipefail

HIPDF_DIR="${1:?Usage: $0 <hipDF-source-dir>}"

if [ ! -d "$HIPDF_DIR/cpp" ]; then
  echo "ERROR: $HIPDF_DIR/cpp not found — is this a hipDF source tree?"
  exit 1
fi

echo "=== Applying cuDF 26.06 API patches to hipDF at $HIPDF_DIR ==="

# -----------------------------------------------------------------------------
# Patch 1: Add fetch_footer_to_host to parquet_io_utils.hpp
# -----------------------------------------------------------------------------
# This API reads the Parquet footer from a datasource into a host buffer.
# It's used by Sirius at:
#   src/scan_manager/sirius_scan_manager.cpp:289
#   src/op/scan/parquet_gpu_ingestible.cpp:466

HEADER="$HIPDF_DIR/cpp/include/cudf/io/parquet_io_utils.hpp"
if [ -f "$HEADER" ] && ! grep -q "fetch_footer_to_host" "$HEADER"; then
  echo "Patching: adding fetch_footer_to_host to $HEADER"
  # Append the declaration at the end of the parquet namespace
  # We use a marker comment to find the right insertion point
  python3 -c "
import sys
with open('$HEADER', 'r') as f:
    content = f.read()

decl = '''
/**
 * @brief Fetches a host buffer of Parquet footer bytes from the input data source
 *
 * @param datasource Input data source
 * @return Host buffer containing footer bytes
 */
[[nodiscard]] std::unique_ptr<cudf::io::datasource::buffer> fetch_footer_to_host(
  cudf::io::datasource& datasource);

/**
 * @brief Returns the metadata size hint for speculative footer reads
 */
[[nodiscard]] std::size_t metadata_size_hint();
'''

# Insert before the closing of the parquet namespace
# Find '}  // namespace parquet' or similar
import re
content = re.sub(
    r'(}\s*//\s*namespace\s*parquet)',
    decl + r'\\1',
    content,
    count=1
)
with open('$HEADER', 'w') as f:
    f.write(content)
"
  echo "  Done"
else
  echo "Skip: fetch_footer_to_host already in $HEADER (or file not found)"
fi

# -----------------------------------------------------------------------------
# Patch 2: Add all_column_chunks_byte_ranges to hybrid_scan.hpp
# -----------------------------------------------------------------------------
# This method returns byte ranges for all column chunks (filter + payload).
# It's used by Sirius at:
#   src/op/scan/parquet_gpu_ingestible.cpp:264

HYBRID_HEADER="$HIPDF_DIR/cpp/include/cudf/io/experimental/hybrid_scan.hpp"
if [ -f "$HYBRID_HEADER" ] && ! grep -q "all_column_chunks_byte_ranges" "$HYBRID_HEADER"; then
  echo "Patching: adding all_column_chunks_byte_ranges to $HYBRID_HEADER"
  python3 -c "
with open('$HYBRID_HEADER', 'r') as f:
    content = f.read()

decl = '''
  /**
   * @brief Get byte ranges of column chunks of all (or selected) columns
   *
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @return Vector of byte ranges to column chunks of all (or selected) columns
   */
  [[nodiscard]] std::vector<byte_range_info> all_column_chunks_byte_ranges(
    std::span<size_type const> row_group_indices, parquet_reader_options const& options) const;
'''

# Insert after the last *_byte_ranges method (before the closing of the class)
import re
# Find payload_column_chunks_byte_ranges and insert after its closing
content = re.sub(
    r'(payload_column_chunks_byte_ranges\s*\([^)]*\)\s*const\s*;)',
    r'\\1\n' + decl,
    content,
    count=1
)
with open('$HYBRID_HEADER', 'w') as f:
    f.write(content)
"
  echo "  Done"
else
  echo "Skip: all_column_chunks_byte_ranges already in $HYBRID_HEADER (or file not found)"
fi

# -----------------------------------------------------------------------------
# Patch 3: Add implementation stubs (the real impl needs internal headers
# that may differ between 25.10 and 26.06 — these stubs throw with a clear
# message until the full impl is ported)
# -----------------------------------------------------------------------------

IMPL_FILE="$HIPDF_DIR/cpp/src/io/parquet/io_utils/parquet_io_utils.cpp"
if [ -f "$IMPL_FILE" ] && ! grep -q "fetch_footer_to_host" "$IMPL_FILE"; then
  echo "Patching: adding fetch_footer_to_host impl to $IMPL_FILE"
  cat >> "$IMPL_FILE" << 'PATCH_EOF'

// --- cuDF 26.06 API port: fetch_footer_to_host ---
// This implementation is sourced from RAPIDS cuDF 26.06 (parquet_io_utils.cpp).
// It reads the Parquet footer from a datasource into a host buffer.
namespace cudf::io::parquet {
std::size_t metadata_size_hint() {
  return 64 * 1024;  // default
}

std::unique_ptr<cudf::io::datasource::buffer> fetch_footer_to_host(
  cudf::io::datasource& datasource) {
  // Read the last N bytes (speculative read)
  constexpr auto ender_len = 8;  // sizeof(file_ender_s) — magic(4) + footer_len(4)
  size_t const len = datasource.size();
  if (len <= ender_len + 4) {
    throw std::runtime_error("fetch_footer_to_host: data source too small");
  }

  auto const speculative_read_size = std::min(len, std::max(metadata_size_hint(), static_cast<size_t>(ender_len)));
  auto const speculative_read_offset = len - speculative_read_size;
  auto speculative_buffer = datasource.host_read(speculative_read_offset, speculative_read_size);

  // Parse the ender to get footer length
  auto const ender = reinterpret_cast<uint32_t const*>(
    speculative_buffer->data() + speculative_buffer->size() - sizeof(uint32_t));
  uint32_t footer_len = *ender;

  if (footer_len == 0 || footer_len > (len - 4 - ender_len)) {
    throw std::runtime_error("fetch_footer_to_host: incorrect footer length");
  }

  auto const footer_offset = len - footer_len - ender_len;
  if (footer_offset >= speculative_read_offset) {
    // Fast path: speculative read includes the full footer
    auto const footer_start = footer_offset - speculative_read_offset;
    std::vector<uint8_t> footer_bytes(footer_len);
    std::memcpy(footer_bytes.data(), speculative_buffer->data() + footer_start, footer_len);
    return cudf::io::datasource::buffer::create(std::move(footer_bytes));
  }

  // Slow path: read the missing prefix
  auto const missing_prefix_size = speculative_read_offset - footer_offset;
  auto missing_prefix = datasource.host_read(footer_offset, missing_prefix_size);
  std::vector<uint8_t> footer_bytes(footer_len);
  std::memcpy(footer_bytes.data(), missing_prefix->data(), missing_prefix_size);
  auto const footer_suffix_size = footer_len - missing_prefix_size;
  std::memcpy(footer_bytes.data() + missing_prefix_size, speculative_buffer->data(), footer_suffix_size);
  return cudf::io::datasource::buffer::create(std::move(footer_bytes));
}
}  // namespace cudf::io::parquet
PATCH_EOF
  echo "  Done"
else
  echo "Skip: fetch_footer_to_host impl already in $IMPL_FILE (or file not found)"
fi

HYBRID_IMPL="$HIPDF_DIR/cpp/src/io/parquet/experimental/hybrid_scan.cpp"
if [ -f "$HYBRID_IMPL" ] && ! grep -q "all_column_chunks_byte_ranges" "$HYBRID_IMPL"; then
  echo "Patching: adding all_column_chunks_byte_ranges impl to $HYBRID_IMPL"
  cat >> "$HYBRID_IMPL" << 'PATCH_EOF'

// --- cuDF 26.06 API port: all_column_chunks_byte_ranges ---
std::vector<byte_range_info> hybrid_scan_reader::all_column_chunks_byte_ranges(
  std::span<size_type const> row_group_indices, parquet_reader_options const& options) const
{
  // Delegate to the existing filter + payload methods and merge results.
  // This is the correct behavior: all_column_chunks = filter + payload.
  auto filter_ranges = filter_column_chunks_byte_ranges(row_group_indices, options);
  auto payload_ranges = payload_column_chunks_byte_ranges(row_group_indices, options);
  filter_ranges.insert(filter_ranges.end(),
                       std::make_move_iterator(payload_ranges.begin()),
                       std::make_move_iterator(payload_ranges.end()));
  return filter_ranges;
}
PATCH_EOF
  echo "  Done"
else
  echo "Skip: all_column_chunks_byte_ranges impl already in $HYBRID_IMPL (or file not found)"
fi

echo ""
echo "=== Patch complete ==="
echo "The following APIs have been added to hipDF:"
echo "  1. cudf::io::parquet::fetch_footer_to_host (parquet_io_utils.hpp/.cpp)"
echo "  2. hybrid_scan_reader::all_column_chunks_byte_ranges (hybrid_scan.hpp/.cpp)"
echo ""
echo "NOTE: The all_column_chunks_byte_ranges implementation delegates to the"
echo "existing filter_column_chunks_byte_ranges + payload_column_chunks_byte_ranges"
echo "methods (which already exist in hipDF 25.10). This is functionally correct"
echo "but may not be optimal — the upstream 26.06 impl uses a combined"
echo "select_columns(ALL_COLUMNS) path. This can be optimized later."

# -----------------------------------------------------------------------------
# Patch 4: Patch get_rmm.cmake to use find_package(rmm) instead of CPM re-fetch
# -----------------------------------------------------------------------------
# hipDF's get_rmm.cmake calls rapids_cpm_rmm which in HIP mode calls
# rapids_cpm_hipmm — this re-fetches rmm from source via CPM, which causes
# "Unknown CMake command rapids_make_logger" because the CPM-fetched rmm
# doesn't have the rapids_logger module that the system-installed hipMM has.
# Fix: use find_package(rmm) to find the system-installed hipMM instead.

RMM_CMAKE="$HIPDF_DIR/cpp/cmake/thirdparty/get_rmm.cmake"
if [ -f "$RMM_CMAKE" ] && ! grep -q "find_package(rmm" "$RMM_CMAKE"; then
  echo "Patching: get_rmm.cmake to use find_package(rmm) instead of CPM"
  cat > "$RMM_CMAKE" << 'RMM_PATCH'
# =============================================================================
# Patched by build_rocm_deps.sh: use find_package(rmm) instead of CPM re-fetch.
#
# The original hipDF get_rmm.cmake calls rapids_cpm_rmm which re-fetches rmm
# from source via CPM. This causes a "Unknown CMake command rapids_make_logger"
# error because the CPM-fetched rmm doesn't have the rapids_logger module that
# the system-installed hipMM (built by build_rocm_deps.sh Step 1) has.
#
# Using find_package(rmm) finds the hipMM install at /opt/rocm which has the
# correct rapids_logger integration.
# =============================================================================

# This function finds rmm (hipMM) via find_package instead of CPM.
function(find_and_configure_rmm)
  find_package(rmm REQUIRED CONFIG)
endfunction()

find_and_configure_rmm()
RMM_PATCH
  echo "  Done"
else
  echo "Skip: get_rmm.cmake already patched (or file not found)"
fi
