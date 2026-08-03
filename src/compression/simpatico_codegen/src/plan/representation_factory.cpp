// SPDX-License-Identifier: Apache-2.0
//
// representation_factory.cpp — per-kind reconstruction of a
// compressed_representation from a compressor name + its named output columns.
//
// Each rep subclass owns its reconstruction-from-stored-buffers via a static
// ``from_outputs`` factory (declared next to the class in representation.hpp).
// This file defines those factories and a thin dispatcher,
// ``simpatico::reconstruct_representation``, that maps a compressor name (or the
// ``bitextract_<spec>`` prefix) to the matching subclass factory. Every place
// that has raw stored channels and needs a typed rep funnels through here.

#include "codegen/plan/operator_registry.hpp"
#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/plan/representation.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace simpatico {
namespace {

// Synchronous device->host read issued on `stream`. Use this instead of a plain
// `cudaMemcpy(..., cudaMemcpyDeviceToHost)` when the source was just written by
// a kernel/copy on a non-blocking pool stream: the default stream does not order
// against pool streams, so a plain sync copy can race and read uninitialised RMM
// memory. Issuing the D2H on the same `stream` orders it after the producer.
inline void d2h_sync(void* dst, const void* src, size_t bytes, rmm::cuda_stream_view stream)
{
  cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream.value());
  cudaStreamSynchronize(stream.value());
}

// Shared validation for the common case: an exact, ordered set of channel
// names with a matching output-column count. Returns false and sets *err on
// any mismatch. Kinds with variable arity (dictionary, bitpack, bitextract,
// identity) validate inline instead.
bool check_outputs(const char* op,
                   std::vector<std::string> const& names,
                   std::vector<std::unique_ptr<cudf::column>> const& outputs,
                   std::initializer_list<const char*> expected,
                   std::string* err)
{
  if (outputs.size() != expected.size() || names.size() != expected.size()) {
    if (err) {
      std::string e =
        std::string(op) + " expects " + std::to_string(expected.size()) + " output channels (";
      bool first = true;
      for (auto* n : expected) {
        if (!first) e += ", ";
        e += n;
        first = false;
      }
      *err = e + ")";
    }
    return false;
  }
  size_t i = 0;
  for (auto* n : expected) {
    if (names[i] != n) {
      if (err) {
        *err = std::string(op) + " channel " + std::to_string(i) + " must be named '" + n +
               "' (got '" + names[i] + "')";
      }
      return false;
    }
    ++i;
  }
  return true;
}

// Validate a single "output" UINT8 channel and adopt its device_buffer. The
// reconstruction entry points own `outputs`, so releasing the column avoids a
// redundant device-to-device copy for the byte-stream codecs
// (ans/bitcomp/snappy/lz4/deflate). Returns nullptr and sets *error_out on
// validation failure; on success sets *out_size to the payload byte count.
std::unique_ptr<rmm::device_buffer> copy_output_payload(
  const char* codec,
  std::vector<std::string> const& output_names,
  std::vector<std::unique_ptr<cudf::column>>& outputs,
  rmm::cuda_stream_view,
  rmm::device_async_resource_ref,
  std::string* error_out,
  std::size_t* out_size)
{
  if (!check_outputs(codec, output_names, outputs, {"output"}, error_out)) return nullptr;
  auto const& payload_col = outputs[0];
  if (payload_col->type().id() != cudf::type_id::UINT8) {
    if (error_out) *error_out = std::string(codec) + " 'output' must be UINT8";
    return nullptr;
  }
  auto const comp = static_cast<std::size_t>(payload_col->size());
  auto contents   = payload_col->release();
  auto payload    = std::move(contents.data);
  if (out_size) *out_size = comp;
  return payload;
}

}  // namespace

std::unique_ptr<compressed_representation> identity_compressed_representation::from_outputs(
  std::vector<std::string> const& /*output_names*/,
  std::vector<std::unique_ptr<cudf::column>> outputs,
  rmm::cuda_stream_view,
  rmm::device_async_resource_ref,
  std::string* error_out)
{
  if (outputs.size() != 1) {
    if (error_out) *error_out = "identity expects exactly one output column";
    return nullptr;
  }
  return std::make_unique<identity_compressed_representation>(std::move(outputs[0]));
}

std::unique_ptr<compressed_representation> dictionary_compressed_representation::from_outputs(
  std::vector<std::string> const& output_names,
  std::vector<std::unique_ptr<cudf::column>> outputs,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  std::string* error_out)
{
  if (output_names.size() != outputs.size()) {
    if (error_out) *error_out = "dictionary: output_names / outputs size mismatch";
    return nullptr;
  }
  if (outputs.size() != 3u || output_names[0] != "keys_offsets" ||
      output_names[1] != "keys_chars" || output_names[2] != "indices") {
    if (error_out) {
      *error_out = "dictionary outputs must be named 'keys_offsets, keys_chars, indices'";
    }
    return nullptr;
  }
  auto keys_offsets = std::move(outputs[0]);
  auto keys_chars   = std::move(outputs[1]);
  auto indices      = std::move(outputs[2]);

  cudf::size_type num_offsets = keys_offsets->size();
  cudf::size_type num_keys    = num_offsets > 0 ? static_cast<cudf::size_type>(num_offsets - 1) : 0;
  if (keys_chars->type().id() != cudf::type_id::UINT8) {
    if (error_out) *error_out = "dictionary keys_chars must be UINT8";
    return nullptr;
  }

  // Hand the chars bytes straight to make_strings_column by moving the
  // column's underlying device_buffer out — no device-to-device copy needed.
  auto chars_contents = keys_chars->release();
  rmm::device_buffer chars_buffer =
    chars_contents.data ? std::move(*chars_contents.data) : rmm::device_buffer(0, stream, mr);

  auto keys_strings = cudf::make_strings_column(num_keys,
                                                std::move(keys_offsets),
                                                std::move(chars_buffer),
                                                0,
                                                rmm::device_buffer(0, stream, mr));

  auto dict_col =
    cudf::make_dictionary_column(std::move(keys_strings), std::move(indices), stream, mr);
  // Indices, keys, and chars are then obtained as views from this column via
  // get_dictionary_child_view(dict_col->view(), ...) / dictionary_column_view.
  return std::make_unique<dictionary_compressed_representation>(std::move(dict_col));
}

// The four buffers are independent leaves that may each be further compressed
// by `for`/`bitpack`/`fastlanes`/`identity`.
std::unique_ptr<compressed_representation> alp_compressed_representation::from_outputs(
  std::vector<std::string> const& output_names,
  std::vector<std::unique_ptr<cudf::column>> outputs,
  rmm::cuda_stream_view,
  rmm::device_async_resource_ref,
  std::string* error_out)
{
  if (!check_outputs("alp",
                     output_names,
                     outputs,
                     {"integers", "exceptions", "exception_positions", "metadata"},
                     error_out))
    return nullptr;
  auto integers            = std::move(outputs[0]);
  auto exceptions          = std::move(outputs[1]);
  auto exception_positions = std::move(outputs[2]);
  auto metadata            = std::move(outputs[3]);

  // Type validation. The (integers, exceptions) pair must match a supported
  // (int_t, value_t) combination from alp_traits: INT32+FLOAT32 for f32
  // input, INT64+FLOAT64 for f64. The other two outputs are precision-
  // independent.
  bool const is_f32 = integers->type().id() == cudf::type_id::INT32 &&
                      exceptions->type().id() == cudf::type_id::FLOAT32;
  bool const is_f64 = integers->type().id() == cudf::type_id::INT64 &&
                      exceptions->type().id() == cudf::type_id::FLOAT64;
  if (!is_f32 && !is_f64) {
    if (error_out)
      *error_out =
        "alp (integers, exceptions) must be (INT32,FLOAT32) or "
        "(INT64,FLOAT64)";
    return nullptr;
  }
  if (exception_positions->type().id() != cudf::type_id::INT32) {
    if (error_out) *error_out = "alp exception_positions must be INT32";
    return nullptr;
  }
  if (metadata->type().id() != cudf::type_id::UINT16) {
    if (error_out) *error_out = "alp metadata must be UINT16";
    return nullptr;
  }
  if (exceptions->size() != exception_positions->size()) {
    if (error_out) *error_out = "alp exceptions and exception_positions must have the same length";
    return nullptr;
  }

  cudf::size_type num_rows    = integers->size();
  cudf::size_type num_vectors = metadata->size();
  // The exceptions column's type IS the original float type by construction
  // (alp_traits stores exceptions as the source value_t).
  cudf::data_type original_type = exceptions->type();

  return std::make_unique<alp_compressed_representation>(original_type,
                                                         num_rows,
                                                         num_vectors,
                                                         std::move(integers),
                                                         std::move(exceptions),
                                                         std::move(exception_positions),
                                                         std::move(metadata));
}

// Six outputs: right_parts, dict_indices, dict, metadata, exceptions,
// exception_positions. Column-wide dict + right_bw.
std::unique_ptr<compressed_representation> alp_rd_compressed_representation::from_outputs(
  std::vector<std::string> const& output_names,
  std::vector<std::unique_ptr<cudf::column>> outputs,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref,
  std::string* error_out)
{
  if (!check_outputs(
        "alp_rd",
        output_names,
        outputs,
        {"right_parts", "dict_indices", "dict", "metadata", "exceptions", "exception_positions"},
        error_out))
    return nullptr;
  auto right_parts         = std::move(outputs[0]);
  auto dict_indices        = std::move(outputs[1]);
  auto dict                = std::move(outputs[2]);
  auto metadata            = std::move(outputs[3]);
  auto exceptions          = std::move(outputs[4]);
  auto exception_positions = std::move(outputs[5]);

  // Type validation. right_parts width selects the precision: UINT32 → f32
  // input, UINT64 → f64. dict / exceptions / dict_indices / metadata /
  // exception_positions are all precision-independent (per alp_rd_traits;
  // dict stays UINT16 because right_bw is clamped so left part fits in 16
  // bits, and exceptions / dict_indices likewise stay at their f32 widths).
  cudf::type_id const rp_id = right_parts->type().id();
  if (rp_id != cudf::type_id::UINT32 && rp_id != cudf::type_id::UINT64) {
    if (error_out) *error_out = "alp_rd right_parts must be UINT32 (f32) or UINT64 (f64)";
    return nullptr;
  }
  if (dict_indices->type().id() != cudf::type_id::UINT8) {
    if (error_out) *error_out = "alp_rd dict_indices must be UINT8";
    return nullptr;
  }
  if (dict->type().id() != cudf::type_id::UINT16) {
    if (error_out) *error_out = "alp_rd dict must be UINT16";
    return nullptr;
  }
  if (metadata->type().id() != cudf::type_id::UINT8 || metadata->size() < 1) {
    if (error_out) *error_out = "alp_rd metadata must be UINT8 with ≥1 element";
    return nullptr;
  }
  if (exceptions->type().id() != cudf::type_id::UINT16) {
    if (error_out) *error_out = "alp_rd exceptions must be UINT16";
    return nullptr;
  }
  if (exception_positions->type().id() != cudf::type_id::INT32) {
    if (error_out) *error_out = "alp_rd exception_positions must be INT32";
    return nullptr;
  }
  if (exceptions->size() != exception_positions->size()) {
    if (error_out)
      *error_out = "alp_rd exceptions and exception_positions must have the same length";
    return nullptr;
  }
  if (right_parts->size() != dict_indices->size()) {
    if (error_out) *error_out = "alp_rd right_parts and dict_indices must have the same length";
    return nullptr;
  }

  cudf::size_type num_rows = right_parts->size();
  // Pull right_bw from the (single-element) metadata column via d2h_sync so
  // this reads through `stream` (which the caller already arranged to wait on
  // the metadata column's producing stream). A plain default-stream
  // `cudaMemcpy` here returned zero from fresh RMM memory in the multistream
  // decompress path, collapsing decode to `(left << 0) | right_parts`.
  uint8_t right_bw_val = 0;
  d2h_sync(&right_bw_val, metadata->view().data<uint8_t>(), sizeof(uint8_t), stream);

  // Derive the original float width from right_parts' element size.
  cudf::data_type original_type{rp_id == cudf::type_id::UINT64 ? cudf::type_id::FLOAT64
                                                               : cudf::type_id::FLOAT32};

  return std::make_unique<alp_rd_compressed_representation>(original_type,
                                                            num_rows,
                                                            right_bw_val,
                                                            std::move(right_parts),
                                                            std::move(dict_indices),
                                                            std::move(dict),
                                                            std::move(metadata),
                                                            std::move(exceptions),
                                                            std::move(exception_positions));
}

std::unique_ptr<compressed_representation> str_split_compressed_representation::from_outputs(
  std::vector<std::string> const& output_names,
  std::vector<std::unique_ptr<cudf::column>> outputs,
  rmm::cuda_stream_view,
  rmm::device_async_resource_ref,
  std::string* error_out)
{
  if (outputs.size() != 2) {
    if (error_out) *error_out = "str_split expects 2 outputs (offsets, chars)";
    return nullptr;
  }
  if (output_names[0] != "offsets" || output_names[1] != "chars") {
    if (error_out) *error_out = "str_split outputs must be 'offsets, chars'";
    return nullptr;
  }
  auto offsets       = std::move(outputs[0]);
  auto chars         = std::move(outputs[1]);
  auto const off_tid = offsets->type().id();
  if (off_tid != cudf::type_id::INT32 && off_tid != cudf::type_id::INT64) {
    if (error_out) *error_out = "str_split offsets must be INT32 or INT64";
    return nullptr;
  }
  // chars is UINT8 normally, or widened (UINT32/UINT64) to hold >2GB under the
  // 2^31-element column cap (see str_split_compressor).
  auto const chars_tid = chars->type().id();
  bool const chars_ok  = chars_tid == cudf::type_id::UINT8 || chars_tid == cudf::type_id::UINT32 ||
                        chars_tid == cudf::type_id::UINT64;
  if (!chars_ok) {
    if (error_out) *error_out = "str_split chars must be UINT8/UINT32/UINT64";
    return nullptr;
  }
  cudf::size_type const n = offsets->size() > 0 ? offsets->size() - 1 : 0;
  return std::make_unique<str_split_compressed_representation>(
    n, std::move(offsets), std::move(chars));
}

std::unique_ptr<compressed_representation> bitextract_compressed_representation::from_outputs(
  bitextract_spec_result spec,
  std::vector<std::string> const& output_names,
  std::vector<std::unique_ptr<cudf::column>> outputs,
  std::string* error_out)
{
  if (spec.fields.empty()) {
    if (error_out) *error_out = "bitextract: empty spec";
    return nullptr;
  }
  if (spec.fields.size() != outputs.size()) {
    if (error_out) *error_out = "bitextract: field count mismatch";
    return nullptr;
  }
  for (size_t i = 0; i < spec.fields.size(); ++i) {
    if (i >= output_names.size() || output_names[i] != spec.fields[i].name) {
      if (error_out) *error_out = "bitextract: output name mismatch at index " + std::to_string(i);
      return nullptr;
    }
  }
  return std::make_unique<bitextract_compressed_representation>(std::move(spec),
                                                                std::move(outputs));
}

// Single ``output`` channel: trimmed UINT8 payload (no header).
// uncompressed_size, original_type_id and algorithm come from leaf_meta::bitcomp.
std::unique_ptr<compressed_representation> bitcomp_compressed_representation::from_outputs(
  std::vector<std::string> const& output_names,
  std::vector<std::unique_ptr<cudf::column>> outputs,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  std::string* error_out,
  leaf_meta_v const& meta)
{
  std::size_t comp = 0;
  auto payload =
    copy_output_payload("bitcomp", output_names, outputs, stream, mr, error_out, &comp);
  if (!payload) return nullptr;
  auto const* bc_meta = std::get_if<leaf_meta::bitcomp>(&meta);
  if (!bc_meta) {
    if (error_out)
      *error_out =
        "bitcomp: missing leaf_meta::bitcomp (uncompressed_size, original_type_id, algorithm)";
    return nullptr;
  }
  cudf::data_type const orig_type{static_cast<cudf::type_id>(bc_meta->original_type_id)};
  size_t const uncomp = bc_meta->uncompressed_size;
  cudf::size_type const n_rows =
    (uncomp > 0 && cudf::is_fixed_width(orig_type))
      ? static_cast<cudf::size_type>(uncomp / static_cast<size_t>(cudf::size_of(orig_type)))
      : 0;
  return std::make_unique<bitcomp_compressed_representation>(
    orig_type, n_rows, std::move(payload), comp, uncomp, bc_meta->algorithm);
}

// Single ``output`` channel: trimmed UINT8 payload (no header).
// uncompressed_size, original_type_id, num_deltas, num_RLEs, use_bp come from
// leaf_meta::nvcomp_cascaded.
std::unique_ptr<compressed_representation> cascaded_compressed_representation::from_outputs(
  std::vector<std::string> const& output_names,
  std::vector<std::unique_ptr<cudf::column>> outputs,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  std::string* error_out,
  leaf_meta_v const& meta)
{
  std::size_t comp = 0;
  auto payload =
    copy_output_payload("nvcomp_cascaded", output_names, outputs, stream, mr, error_out, &comp);
  if (!payload) return nullptr;
  auto const* casc_meta = std::get_if<leaf_meta::nvcomp_cascaded>(&meta);
  if (!casc_meta) {
    if (error_out)
      *error_out =
        "nvcomp_cascaded: missing leaf_meta::nvcomp_cascaded "
        "(uncompressed_size, original_type_id, num_deltas, num_RLEs, use_bp)";
    return nullptr;
  }
  cudf::data_type const orig_type{static_cast<cudf::type_id>(casc_meta->original_type_id)};
  size_t const uncomp = casc_meta->uncompressed_size;
  cudf::size_type const n_rows =
    (uncomp > 0 && cudf::is_fixed_width(orig_type))
      ? static_cast<cudf::size_type>(uncomp / static_cast<size_t>(cudf::size_of(orig_type)))
      : 0;
  return std::make_unique<cascaded_compressed_representation>(orig_type,
                                                              n_rows,
                                                              std::move(payload),
                                                              comp,
                                                              uncomp,
                                                              casc_meta->num_deltas,
                                                              casc_meta->num_RLEs,
                                                              casc_meta->use_bp);
}

// ── Simple nvcomp codecs (Snappy / LZ4 / GDeflate) ───────────────────────────
// All three share the same from_outputs body: single UINT8 'output' channel,
// metadata from the matching leaf_meta type.

namespace {
template <typename RepT, typename MetaT>
std::unique_ptr<compressed_representation> nvcomp_simple_from_outputs(
  std::string_view codec_name,
  std::vector<std::string> const& output_names,
  std::vector<std::unique_ptr<cudf::column>> outputs,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  std::string* error_out,
  leaf_meta_v const& meta)
{
  std::string codec(codec_name);
  std::size_t comp = 0;
  auto payload =
    copy_output_payload(codec.c_str(), output_names, outputs, stream, mr, error_out, &comp);
  if (!payload) return nullptr;
  auto const* m = std::get_if<MetaT>(&meta);
  if (!m) {
    if (error_out)
      *error_out = codec + ": missing leaf metadata (uncompressed_size, original_type_id)";
    return nullptr;
  }
  cudf::data_type const orig_type{static_cast<cudf::type_id>(m->original_type_id)};
  size_t const uncomp = m->uncompressed_size;
  cudf::size_type const n_rows =
    (uncomp > 0 && cudf::is_fixed_width(orig_type))
      ? static_cast<cudf::size_type>(uncomp / static_cast<size_t>(cudf::size_of(orig_type)))
      : 0;
  return std::make_unique<RepT>(orig_type, n_rows, std::move(payload), comp, uncomp);
}
}  // anonymous namespace

// ── thin dispatcher ───────────────────────────────────────────────────────────
// Resolves a compressor name to its OpId (via the operator registry, so all
// parameterised suffix forms are handled uniformly) and dispatches to the
// matching rep subclass's ``from_outputs`` factory. External linkage (called by
// the decode driver and the file-read/deserialize path). The fused ops
// (delta/rle/for/zigzag) have no standalone reconstruction — they are inverted
// by the codegen decode path — and report ``unsupported`` here.
std::unique_ptr<compressed_representation> reconstruct_representation(
  std::string const& compressor_name,
  std::vector<std::string> const& output_names,
  std::vector<std::unique_ptr<cudf::column>> outputs,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  std::string* error_out,
  leaf_meta_v const& meta)
{
  auto const unsupported = [&]() -> std::unique_ptr<compressed_representation> {
    if (error_out) {
      *error_out = "unsupported compressor '" + compressor_name + "' for reconstruction";
    }
    return nullptr;
  };

  auto id = op_id_from_name(compressor_name);
  if (!id) return unsupported();

  switch (*id) {
    case OpId::Identity:
      return identity_compressed_representation::from_outputs(
        output_names, std::move(outputs), stream, mr, error_out);
    case OpId::Dictionary:
      return dictionary_compressed_representation::from_outputs(
        output_names, std::move(outputs), stream, mr, error_out);
    case OpId::Alp:
      return alp_compressed_representation::from_outputs(
        output_names, std::move(outputs), stream, mr, error_out);
    case OpId::AlpRd:
      return alp_rd_compressed_representation::from_outputs(
        output_names, std::move(outputs), stream, mr, error_out);
    case OpId::Bitextract: {
      auto suffix = strip_bitextract_prefix(compressor_name);
      auto spec   = parse_bitextract_spec(suffix ? *suffix : std::string_view{});
      if (spec.fields.empty()) {
        if (error_out) *error_out = "bitextract: bad spec in '" + compressor_name + "'";
        return nullptr;
      }
      return bitextract_compressed_representation::from_outputs(
        std::move(spec), output_names, std::move(outputs), error_out);
    }
    // The simple byte-stream codecs share one body (single UINT8 'output'
    // channel + (uncompressed_size, type_id) meta) — reconstructed generically.
    case OpId::Ans:
      return nvcomp_simple_from_outputs<ans_compressed_representation, leaf_meta::ans>(
        "ans", output_names, std::move(outputs), stream, mr, error_out, meta);
    case OpId::Snappy:
      return nvcomp_simple_from_outputs<snappy_compressed_representation, leaf_meta::snappy>(
        "snappy", output_names, std::move(outputs), stream, mr, error_out, meta);
    case OpId::Lz4:
      return nvcomp_simple_from_outputs<lz4_compressed_representation, leaf_meta::lz4>(
        "lz4", output_names, std::move(outputs), stream, mr, error_out, meta);
    case OpId::Deflate:
      return nvcomp_simple_from_outputs<deflate_compressed_representation, leaf_meta::deflate>(
        "deflate", output_names, std::move(outputs), stream, mr, error_out, meta);
    case OpId::Bitcomp:
      return bitcomp_compressed_representation::from_outputs(
        output_names, std::move(outputs), stream, mr, error_out, meta);
    case OpId::NvcompCascaded:
      return cascaded_compressed_representation::from_outputs(
        output_names, std::move(outputs), stream, mr, error_out, meta);
    case OpId::StrSplit:
      return str_split_compressed_representation::from_outputs(
        output_names, std::move(outputs), stream, mr, error_out);
    case OpId::Bitpack:
    case OpId::Delta:
    case OpId::Rle:
    case OpId::For:
    case OpId::Zigzag: return unsupported();
  }
  return unsupported();
}

std::unique_ptr<cudf::column> decompress_standalone_representation(
  compressed_representation const* rep,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  std::string* error_out)
{
  if (!rep) {
    if (error_out) *error_out = "decompress_standalone_representation: null representation";
    return nullptr;
  }
  auto const* standalone = dynamic_cast<standalone_compressed_representation const*>(rep);
  if (!standalone) {
    if (error_out)
      *error_out = "decompress_standalone_representation: representation of kind " +
                   std::to_string(static_cast<int>(rep->kind())) +
                   " is storage-only and requires the PlanTree decode bridge (e.g. DecodeWalk)";
    return nullptr;
  }
  return standalone->decompress(stream, mr);
}

}  // namespace simpatico
