// SPDX-License-Identifier: Apache-2.0
//
// Shared operator catalog + single-op trial primitive.  See
// include/explore/operator_catalog.hpp for the rationale.

#include "explore/operator_catalog.hpp"

#include "codegen/plan/operator_registry.hpp"  // operator_registry, op_id_from_name, op_info
#include "codegen/plan/plan_interpreter.hpp"   // compress_single_op
#include "explore/compression_explorer.hpp"    // column_size_bytes_ex

#include <cudf/utilities/traits.hpp>  // cudf::is_floating_point, is_integral

#include <cuda_runtime.h>

#include <sstream>

namespace simpatico {

std::vector<std::string> const& all_compressor_names()
{
  static std::vector<std::string> const names = [] {
    std::vector<std::string> out;
    for (auto const& op : operator_registry()) {
      out.insert(out.end(), op.catalog_names.begin(), op.catalog_names.end());
    }
    return out;
  }();
  return names;
}

bool is_terminal_compressor(std::string const& name)
{
  auto id = op_id_from_name(name);
  return id && op_info(*id).terminal;
}

bool is_preprocessing_compressor(std::string const& name)
{
  auto id = op_id_from_name(name);
  return id && op_info(*id).preprocessing;
}

std::string format_output_names(std::vector<compressible_output> const& outs)
{
  std::ostringstream oss;
  for (std::size_t i = 0; i < outs.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << outs[i].name;
  }
  return oss.str();
}

operator_trial try_operator(std::string const& name,
                            cudf::column_view col,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  operator_trial r;

  // Static type gate: reject operators that are incompatible with the input
  // dtype before touching the GPU so that a type mismatch can never corrupt
  // the CUDA context or produce a misleading "success" result.
  auto const tid      = col.type().id();
  bool const is_float = cudf::is_floating_point(col.type());
  // Chrono columns compress as their integer storage (see encode_fused_subtree),
  // so they take the integer op set.
  bool const is_int = cudf::is_integral(col.type()) || cudf::is_chrono(col.type());
  bool const is_str = (tid == cudf::type_id::STRING);

  // bitextract is meaningful only on its native floating-point type; applying it
  // to integer data produces larger output and its intermediate columns can crash
  // downstream operators.
  if (name == "bitextract_f64" && tid != cudf::type_id::FLOAT64) {
    r.error_message = name + ": requires float64 input";
    return r;
  }
  if (name == "bitextract_f32" && tid != cudf::type_id::FLOAT32) {
    r.error_message = name + ": requires float32 input";
    return r;
  }
  if ((name == "alp" || name == "alp_rd") && !is_float) {
    r.error_message = name + ": requires floating-point input";
    return r;
  }
  if ((name == "dictionary" || name == "dictionary_fast" || name == "str_split") && !is_str) {
    r.error_message = name + ": requires string input";
    return r;
  }
  // Integer-only preprocessing operators
  if ((name == "delta" || name == "for" || name == "zigzag" || name == "bitpack" ||
       name == "rle") &&
      !is_int && !is_float) {
    r.error_message = name + ": requires numeric input";
    return r;
  }

  // Clear any residual CUDA error before attempting so a previous failure on
  // the same stream does not poison this attempt.
  cudaGetLastError();
  try {
    std::string err;
    auto rep             = compress_single_op(name, col, stream, mr, &err);
    cudaError_t sync_err = cudaStreamSynchronize(stream.value());
    if (sync_err != cudaSuccess) {
      // A kernel on this stream faulted.  Clear the error so subsequent
      // operators can proceed and bail out of this attempt cleanly.
      cudaGetLastError();
      r.error_message = "stream sync after " + name + ": " + cudaGetErrorString(sync_err);
      return r;
    }
    if (!rep) {
      r.error_message = "compress_single_op (" + name + "): " + err;
      return r;
    }
    r.outputs = rep->named_channels(stream);
    r.repr    = std::shared_ptr<compressed_representation>(std::move(rep));
    for (auto const& o : r.outputs) {
      r.output_bytes += column_size_bytes_ex(o.view, stream);
    }
    r.success = true;
  } catch (std::exception const& e) {
    cudaGetLastError();  // clear any CUDA error state left by the exception
    r.error_message = e.what();
  } catch (...) {
    cudaGetLastError();
    r.error_message = "unknown exception in " + name;
  }
  return r;
}

dsl_step make_dsl_step(std::string const& input_path,
                       std::string const& op_name,
                       std::vector<compressible_output> const& outputs)
{
  dsl_step step;
  bool const terminal     = is_terminal_compressor(op_name);
  std::string const names = format_output_names(outputs);

  std::ostringstream line;
  line << input_path << " -> " << op_name;
  if (!terminal && !names.empty()) line << " -> " << names;
  step.line = line.str();

  step.output_paths.reserve(outputs.size());
  for (auto const& o : outputs) {
    step.output_paths.push_back(input_path == "input" ? op_name + "." + o.name
                                                      : input_path + "." + o.name);
  }
  return step;
}

}  // namespace simpatico
