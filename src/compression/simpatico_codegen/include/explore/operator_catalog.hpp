// SPDX-License-Identifier: Apache-2.0
//
// Shared operator catalog + single-op trial primitive.
//
// Both the beam-search explorer (`explore_column_compression`) and the
// exhaustive operator sweep test build compression cascades the same way:
// repeatedly apply ONE operator to a column and recurse into the typed output
// channels it produces.  Whether an operator is applicable to a given column is
// decided dynamically — `try_operator` just runs `compress_single_op` and
// reports failure if the op cannot handle that dtype.
//
// Centralising the operator list, the terminal/preprocessing classification,
// the DSL-step formatting, and the "apply one op, get typed channels" primitive
// here keeps those two consumers from drifting apart as operators are added or
// their type behaviour changes.

#pragma once

#include "codegen/plan/representation.hpp"  // compressible_output, compressed_representation

#include <cudf/column/column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <string>
#include <vector>

namespace simpatico {

// ---------------------------------------------------------------------------
// Operator catalog + classification
// ---------------------------------------------------------------------------

/// Every operator the explorer / sweep will attempt.  `compress_single_op`
/// resolves both fused (delta/rle/bitpack/for/zigzag) and non-fused ops.
std::vector<std::string> const& all_compressor_names();

/// Terminal ops emit an opaque byte payload that ends a cascade branch
/// (snappy / deflate / lz4 / ans / bitcomp / nvcomp_cascaded).
bool is_terminal_compressor(std::string const& name);

/// Preprocessing (transform) ops reshape data without necessarily shrinking it
/// (delta / rle / for / alp / alp_rd / zigzag / bitextract_*).  The explorer
/// keeps them even when they do not reduce size.
bool is_preprocessing_compressor(std::string const& name);

// ---------------------------------------------------------------------------
// Single-op trial
// ---------------------------------------------------------------------------

/// Result of applying a single operator to a column.
struct operator_trial {
  bool success = false;
  std::string error_message;
  /// Typed output channels (the `named_channels()` of the resulting rep).
  std::vector<compressible_output> outputs;
  /// Sum of logical channel byte sizes (via `column_size_bytes_ex`).
  std::size_t output_bytes = 0;
  /// Owns the representation so the `column_view`s in `outputs` stay valid for
  /// as long as this trial is held.
  std::shared_ptr<compressed_representation> repr;
};

/// Apply operator `name` to `col` via `compress_single_op`, synchronise the
/// stream, and collect the resulting typed output channels.  On any failure
/// (operator not applicable to the column's dtype, GPU error, …) returns
/// `{success = false}` with `error_message` populated — never throws.
operator_trial try_operator(std::string const& name,
                            cudf::column_view col,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr);

// ---------------------------------------------------------------------------
// DSL-step formatting (shared so generated plans are byte-identical)
// ---------------------------------------------------------------------------

/// Format the `a, b, c` channel-name list for a DSL step.
std::string format_output_names(std::vector<compressible_output> const& outs);

/// A single formatted DSL step plus the output paths its non-terminal channels
/// take (for continued chaining).
struct dsl_step {
  /// e.g. "input -> delta -> differences" or "delta.differences -> ans".
  std::string line;
  /// Paths of each output channel, parallel to `outputs`.  For terminal ops the
  /// channels become stored leaves; the paths are still reported here so the
  /// caller may track leaf sizes, but they are not valid chaining inputs.
  std::vector<std::string> output_paths;
};

/// Build the DSL step for applying `op_name` to `input_path`, producing
/// `outputs`.  Uses the same path convention as the explorer: a channel of the
/// first op applied to "input" is referenced as `<op>.<channel>`; deeper
/// channels as `<input_path>.<channel>`.
dsl_step make_dsl_step(std::string const& input_path,
                       std::string const& op_name,
                       std::vector<compressible_output> const& outputs);

}  // namespace simpatico
