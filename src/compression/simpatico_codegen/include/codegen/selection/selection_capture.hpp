// SPDX-License-Identifier: Apache-2.0
//
// Wave-seam selection capture for late materialization (SIRIUS_EXP_LATE_MAT).
//
// Today the fused scan-filter converter (try_decompress_scan_filter,
// compression_converters.cpp) lets scan_filter_result's selection buffers die
// function-local after assembling the compacted table — the wave-1 survivors
// (mask words + per-chunk offsets, and the TierB row-index list when built)
// are exactly a late-materialization row selection over the batch's rows and
// need never be recomputed. This header is the wave-side MOVE of that state:
// the converter calls capture_scan_filter_selection right after
// decompress_scan_filter returns, and hands the capture to the scan split's
// annotation carrier (kind = mask row_selection; global survivor id =
// range.start + chunk_id*1024 + offset for each set bit).
//
// NEW header, header-only; nothing on the shipped fused path includes it, so
// the fused pipeline is byte-identical until the annotation carrier wires it
// in. The shipped selection types (selection.hpp) are untouched.

#pragma once

#include "codegen/selection/selection.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cstdint>
#include <memory>
#include <utility>

namespace sirius::codegen {

/// A wave-1 selection moved out of a scan_filter_result. Buffers are
/// stream-ordered RMM allocations shared via shared_ptr — they ride the
/// (immutable) batch annotation and free when the last reference drops, on
/// the stream they were rebound to; no manual frees, no new teardown paths.
struct captured_scan_selection {
  std::shared_ptr<rmm::device_buffer> mask_words;     ///< uint32 x WordsFor(num_rows)
                                                      ///< (full 32-word strips, tail zero)
  std::shared_ptr<rmm::device_buffer> chunk_offsets;  ///< uint32 x (ChunksFor(num_rows)+1),
                                                      ///< exclusive survivor prefix + total
  std::shared_ptr<rmm::device_buffer> row_indices;    ///< int32 x survivor_count, ascending,
                                                      ///< batch-local; null when the TierB
                                                      ///< path did not build it
  std::int64_t num_rows       = 0;  ///< the batch's FULL row count (== origin range rows)
  std::int64_t survivor_count = -1;

  [[nodiscard]] explicit operator bool() const noexcept { return mask_words != nullptr; }
};

/// MOVE the wave-1 selection out of @p result IFF `result.status == applied`.
/// A RULE-2 bail, a refusal, or a mid-flight failure produced classic
/// full-width output — the mask does NOT describe the output rows and must
/// never be captured: on those statuses this returns an empty capture and
/// leaves @p result untouched (the caller's classic-path handling proceeds
/// unchanged).
///
/// Every moved buffer is set_stream(@p stream)-rebound — the same
/// teardown-ordering discipline as rebind_column_stream: the capture outlives
/// the converter call, so its eventual stream-ordered free must be on the
/// stream the consumer hands it to, not the converter-internal one.
/// row_indices is moved only when the TierB path actually built it (non-empty
/// buffer); mask-only captures leave it null and the consumer derives an
/// id list per its own density policy if it wants one.
inline captured_scan_selection capture_scan_filter_selection(scan_filter_result&& result,
                                                             rmm::cuda_stream_view stream)
{
  captured_scan_selection cap;
  if (result.status != scan_filter_status::applied) { return cap; }
  if (result.mask_words.size() == 0 || result.chunk_offsets.size() == 0 ||
      result.survivor_count < 0) {
    return cap;  // malformed applied result: fail closed, never a wrong selection
  }
  cap.num_rows       = result.num_rows;
  cap.survivor_count = result.survivor_count;
  cap.mask_words     = std::make_shared<rmm::device_buffer>(std::move(result.mask_words));
  cap.mask_words->set_stream(stream);
  cap.chunk_offsets = std::make_shared<rmm::device_buffer>(std::move(result.chunk_offsets));
  cap.chunk_offsets->set_stream(stream);
  if (result.row_indices.size() != 0) {
    cap.row_indices = std::make_shared<rmm::device_buffer>(std::move(result.row_indices));
    cap.row_indices->set_stream(stream);
  }
  return cap;
}

}  // namespace sirius::codegen
