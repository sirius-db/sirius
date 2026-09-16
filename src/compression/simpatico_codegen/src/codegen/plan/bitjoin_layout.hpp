// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "codegen/plan/plan_dsl.hpp"   // bit_range
#include "codegen/plan/plan_tree.hpp"  // ValueId, ValueIdHash

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace simpatico {

// Resolved bitjoin layout: per-field widths, source-LSB and destination-LSB
// positions, plus the output column type. Decoded from a `bitjoin_*` compressor
// name and the input list's optional per-token bit ranges.
struct bitjoin_layout {
  std::vector<uint32_t> widths;
  std::vector<uint32_t> src_los;
  std::vector<uint32_t> dst_los;
  cudf::data_type output_type{cudf::type_id::EMPTY};
};

bool resolve_bitjoin_layout(std::string const& compressor_name,
                            std::size_t n_fields,
                            std::vector<std::optional<bit_range>> const& input_ranges,
                            bitjoin_layout* layout,
                            std::string* error_out);

// For each unique input value referenced by a bitjoin step, OR the field masks
// that target it and check `(input & ~selected_mask) != 0` element-wise on the
// GPU. Logs a stderr warning per input that had truncated bits. Synchronises
// `stream`. `columns` / `input_sources` are keyed structurally by ValueId.
void bitjoin_warn_on_truncation(
  std::unordered_map<ValueId, cudf::column_view, ValueIdHash> const& columns,
  bitjoin_layout const& layout,
  std::vector<ValueId> const& input_sources,
  std::string const& compressor_name,
  cudaStream_t stream);

/// Deep-copy a column_view into an owning column (for identity leaves).
/// Fixed-width types are copied via make_fixed_width_column + memcpy; STRING/LIST
/// are copied via an identity gather. For a LIST view with >=2 children the list's
/// element child (child 1) is copied.
std::unique_ptr<cudf::column> copy_column_view(cudf::column_view const& view,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr);

/// Copy a column view as raw UINT8 (for dictionary keys_chars when view
/// type/layout may vary). Copies the full byte size (size() * element_size) so
/// INT32 or other fixed-width types work.
std::unique_ptr<cudf::column> copy_column_view_as_uint8(cudf::column_view const& view,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr);

}  // namespace simpatico
