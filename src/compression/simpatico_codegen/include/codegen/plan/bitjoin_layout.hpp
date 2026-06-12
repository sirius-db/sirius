// SPDX-License-Identifier: Apache-2.0
#ifndef SIMPATICO_PLAN_BITJOIN_LAYOUT_HPP
#define SIMPATICO_PLAN_BITJOIN_LAYOUT_HPP

#include "codegen/plan/plan_dsl.hpp"  // bit_range

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <cuda_runtime.h>

#include <cstdint>
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
                            std::vector<std::string> const& input_paths,
                            std::vector<std::optional<bit_range>> const& input_ranges,
                            bitjoin_layout* layout,
                            std::string* error_out);

// For each unique input column referenced by a bitjoin step, OR the field masks
// that target it and check `(input & ~selected_mask) != 0` element-wise on the
// GPU. Logs a stderr warning per input that had truncated bits. Synchronises
// `stream`.
void bitjoin_warn_on_truncation(std::unordered_map<std::string, cudf::column_view> const& columns,
                                bitjoin_layout const& layout,
                                std::vector<std::string> const& input_paths,
                                std::string const& compressor_name,
                                cudaStream_t stream);

}  // namespace simpatico

#endif  // SIMPATICO_PLAN_BITJOIN_LAYOUT_HPP
