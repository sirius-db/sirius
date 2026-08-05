// SPDX-License-Identifier: Apache-2.0
//
// Late-materialization row decode (SIRIUS_EXP_LATE_MAT) — plan-level entry
// point over an arbitrary chunk-bucketed row selection (row_set.hpp), the
// post-scan sibling of decompress_column_compacted (plan_interpreter.hpp).
//
// NEW header; no shipped decode path includes it.

#pragma once

#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/selection/row_set.hpp"

#include <memory>
#include <string>

namespace simpatico {

/// Decode ONLY the rows listed in @p rows (batch-local, chunk-bucketed CSR;
/// rows.num_rows must equal the column's row count), compacted in ascending
/// row order — never a full-width column. Dispatch mirrors
/// decompress_column_compacted, on the sparse launchers:
///   * tier_a / tier_a_delta   — K8 sparse decode on the root fused region
///     (bitpack: true random access; delta: touched-chunk reconstruct,
///     survivor-only stores).
///   * tier_dict_k5            — sparse fast path only (constant-width,
///     null-free, IDENTITY-STORED keys — the shipped K5 fast-path shape).
///     Compressed/variable-width keys return nullptr: the caller falls back
///     to the mask route (decompress_column_compacted), whose general dict
///     route serves them.
///   * tier_str_k6             — K6s sparse survivor metadata + the shipped
///     launch_masked_char_copy. NOTE: this route carries one data-dependent
///     host sync (total survivor chars sizes the chars buffer), same as the
///     shipped K6 — the numeric routes are stream-ordered.
///   * tier_b                  — refused (nullptr + error): no random access
///     exists; the caller must full-decode + gather.
/// Returns nullptr + @p error_out on any refusal or failure; no device state
/// is corrupted, so falling back is always safe.
std::unique_ptr<cudf::column> decompress_column_rows(
  PlanTree const& tree,
  sirius::codegen::chunk_row_set const& rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  std::string* error_out);

/// Re-tag a decompressed column from its CODEC STORAGE type back to the
/// column's stored logical dtype (defined in simpatico_codegen.cpp; declared
/// here for the late-mat column-level consumers). The codecs run on the
/// underlying integer storage of fixed-point/temporal columns — the bytes are
/// already correct, only the type tag differs (e.g. INT64 storage of a
/// DECIMAL64 column, which carries the SCALE: skipping this re-tag silently
/// multiplies decimal values by 10^-scale — the q9 arm-C wrong-results root
/// cause). Same-width fixed-width types only; a no-op when the types already
/// match or either side is not fixed-width (strings pass through untouched).
/// EVERY column-level materialization (decompress_column /
/// decompress_column_compacted / decompress_column_rows) must be routed
/// through this with the owning compressed_column's `dtype` — the table-level
/// decompress overloads apply it internally, the column-level ones do NOT.
std::unique_ptr<cudf::column> apply_stored_dtype(std::unique_ptr<cudf::column> col,
                                                 cudf::data_type stored);

}  // namespace simpatico
