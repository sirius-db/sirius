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

#include "../decode/decode_session.hpp"
#include "codegen/plan/operator_registry.hpp"
#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/plan/representation.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/null_mask.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace simpatico {

std::size_t compressed_representation::owned_device_bytes_estimate() const
{
  std::size_t bytes = 0;
  for (auto const& column : channels_)
    if (column) bytes += column->alloc_size();
  return bytes;
}

std::size_t dictionary_compressed_representation::owned_device_bytes_estimate() const
{
  auto bytes = compressed_representation::owned_device_bytes_estimate();
  for (auto const* column : {dict_column.get(),
                             keys_chars_copy.get(),
                             keys_offsets_synth.get(),
                             indices_synth.get(),
                             null_mask_copy.get()})
    if (column) bytes += column->alloc_size();
  return bytes;
}

std::size_t bitextract_compressed_representation::owned_device_bytes_estimate() const
{
  auto bytes = compressed_representation::owned_device_bytes_estimate();
  for (auto const& column : fields)
    if (column) bytes += column->alloc_size();
  return bytes;
}

std::size_t codegen_fused_representation::owned_device_bytes_estimate() const
{
  auto bytes = compressed_representation::owned_device_bytes_estimate();
  for (auto const& [name, column] : buffers)
    if (column) bytes += column->alloc_size();
  return bytes;
}

namespace {

// Synchronous device->host read issued on `stream`. Use this instead of a plain
// `cudaMemcpy(..., cudaMemcpyDeviceToHost)` when the source was just written by
// a kernel/copy on a non-blocking pool stream: the default stream does not order
// against pool streams, so a plain sync copy can race and read uninitialised RMM
// memory. Issuing the D2H on the same `stream` orders it after the producer.
inline void d2h_sync(void* dst, const void* src, size_t bytes, rmm::cuda_stream_view stream)
{
  cudaError_t e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream.value());
  if (e == cudaSuccess) e = cudaStreamSynchronize(stream.value());
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("d2h_sync failed: ") + cudaGetErrorString(e));
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

namespace {

// Reads a "null_mask" channel column (UINT8 bitmask bytes, as emitted by
// dictionary/str_split named_channels) into a device_buffer + null count over
// `num_rows` rows. Returns false and sets *error_out on a malformed channel.
bool take_null_mask_channel(std::unique_ptr<cudf::column> mask_col,
                            cudf::size_type num_rows,
                            rmm::device_buffer* mask_out,
                            cudf::size_type* null_count_out,
                            rmm::cuda_stream_view stream,
                            std::string* error_out)
{
  if (mask_col->type().id() != cudf::type_id::UINT8) {
    if (error_out) *error_out = "dictionary null_mask must be UINT8";
    return false;
  }
  if (static_cast<std::size_t>(mask_col->size()) < cudf::bitmask_allocation_size_bytes(num_rows)) {
    if (error_out) *error_out = "dictionary null_mask is shorter than the row count requires";
    return false;
  }
  auto const* bits =
    reinterpret_cast<cudf::bitmask_type const*>(mask_col->view().data<std::uint8_t>());
  *null_count_out = num_rows > 0 ? cudf::null_count(bits, 0, num_rows, stream) : 0;
  auto contents   = mask_col->release();
  *mask_out       = std::move(*contents.data);
  return true;
}

}  // namespace

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
  // Accepts a trailing optional "null_mask" channel (UINT8 bitmask bytes of the
  // decoded column) — reattached below so validity survives channel-based
  // round-trips (.hpln IO and decomposed plans).
  bool const has_mask = !output_names.empty() && output_names.back() == "null_mask";

  // keys_offsets, keys_chars, indices (+ null_mask).
  if (outputs.size() != (has_mask ? 4u : 3u) || output_names[0] != "keys_offsets" ||
      output_names[1] != "keys_chars" || output_names[2] != "indices") {
    if (error_out) {
      *error_out =
        "dictionary outputs must be named 'keys_offsets, keys_chars, indices[, null_mask]'";
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

  rmm::device_buffer mask(0, stream, mr);
  cudf::size_type null_count = 0;
  if (has_mask &&
      !take_null_mask_channel(
        std::move(outputs[3]), indices->size(), &mask, &null_count, stream, error_out)) {
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
    null_count > 0
      ? cudf::make_dictionary_column(
          std::move(keys_strings), std::move(indices), std::move(mask), null_count)
      : cudf::make_dictionary_column(std::move(keys_strings), std::move(indices), stream, mr);
  // Indices, keys, and chars are then obtained as views from this column via
  // get_dictionary_child_view(dict_col->view(), ...) / dictionary_column_view.
  return from_encoded_column(std::move(dict_col), stream, mr);
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
  // Variable arity: {offsets, chars} or {offsets, chars, null_mask}.
  bool const has_mask = outputs.size() == 3;
  if (outputs.size() != 2 && outputs.size() != 3) {
    if (error_out) *error_out = "str_split expects 2 (offsets, chars) or 3 (+ null_mask) outputs";
    return nullptr;
  }
  if (output_names[0] != "offsets" || output_names[1] != "chars" ||
      (has_mask && output_names[2] != "null_mask")) {
    if (error_out) *error_out = "str_split outputs must be 'offsets, chars[, null_mask]'";
    return nullptr;
  }
  auto offsets       = std::move(outputs[0]);
  auto chars         = std::move(outputs[1]);
  auto null_mask     = has_mask ? std::move(outputs[2]) : nullptr;
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
  if (!chars_ok || (null_mask && null_mask->type().id() != cudf::type_id::UINT8)) {
    if (error_out) *error_out = "str_split chars must be UINT8/UINT32/UINT64, null_mask UINT8";
    return nullptr;
  }
  cudf::size_type const n = offsets->size() > 0 ? offsets->size() - 1 : 0;
  return std::make_unique<str_split_compressed_representation>(
    n, std::move(offsets), std::move(chars), std::move(null_mask));
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

namespace {

void require_decode_channels(std::vector<std::string> const& names,
                             std::span<decode_column_slot const> outputs,
                             std::initializer_list<char const*> expected)
{
  if (names.size() != expected.size() || outputs.size() != expected.size() ||
      !std::equal(names.begin(), names.end(), expected.begin())) {
    throw std::invalid_argument("decode reconstruction: invalid output channels");
  }
}

void adopt_decode_channels(compressed_representation& rep,
                           std::span<decode_column_slot const> outputs,
                           decode_frame& frame)
{
  rep.channels_.resize(outputs.size());
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    rep.channels_[i] = frame.release(outputs[i]);
  }
}

template <typename Rep, typename Meta>
compressed_representation& make_decode_payload(std::span<decode_column_slot const> outputs,
                                               leaf_meta_v const& metadata,
                                               decode_frame& frame)
{
  auto const* meta = std::get_if<Meta>(&metadata);
  if (!meta) { throw std::invalid_argument("decode reconstruction: missing nvCOMP metadata"); }
  auto const type  = cudf::data_type{static_cast<cudf::type_id>(meta->original_type_id)};
  auto const bytes = meta->uncompressed_size;
  auto const rows  = bytes > 0 && cudf::is_fixed_width(type)
                       ? static_cast<cudf::size_type>(bytes / cudf::size_of(type))
                       : 0;
  std::unique_ptr<compressed_representation> owner;
  if constexpr (std::is_same_v<Meta, leaf_meta::bitcomp>) {
    owner = std::make_unique<Rep>(type, rows, nullptr, 0, bytes, meta->algorithm);
  } else if constexpr (std::is_same_v<Meta, leaf_meta::nvcomp_cascaded>) {
    owner = std::make_unique<Rep>(
      type, rows, nullptr, 0, bytes, meta->num_deltas, meta->num_RLEs, meta->use_bp);
  } else {
    owner = std::make_unique<Rep>(type, rows, nullptr, 0, bytes);
  }
  auto& rep = frame.keep_representation(std::move(owner));
  adopt_decode_channels(rep, outputs, frame);
  return rep;
}

/** Drains on assembly failure before the preceding local owner bundle unwinds. */
class failed_assembly_drain {
 public:
  explicit failed_assembly_drain(rmm::cuda_stream_view stream)
    : stream_(stream), exceptions_(std::uncaught_exceptions())
  {
  }
  ~failed_assembly_drain() noexcept
  {
    if (std::uncaught_exceptions() > exceptions_) { stream_.synchronize_no_throw(); }
  }

 private:
  rmm::cuda_stream_view stream_;
  int exceptions_;
};

void retain_remaining_contents(cudf::column::contents& contents, decode_frame& frame)
{
  if (contents.data) { frame.keep_buffer(std::move(*contents.data)); }
  if (contents.null_mask) { frame.keep_buffer(std::move(*contents.null_mask)); }
  for (auto& child : contents.children) {
    if (child) { frame.keep_column(std::move(child)); }
  }
}

compressed_representation& make_decode_dictionary(std::span<decode_column_slot const> outputs,
                                                  bool has_mask,
                                                  decode_frame& frame)
{
  auto const offsets_type = outputs[0]->type().id();
  if ((offsets_type != cudf::type_id::INT32 && offsets_type != cudf::type_id::INT64) ||
      outputs[1]->type().id() != cudf::type_id::UINT8) {
    throw std::invalid_argument("dictionary: invalid key offsets/chars types");
  }
  auto const offsets = outputs[0]->size();
  if (offsets < 1 || outputs[0]->null_count() != 0) {
    throw std::invalid_argument("dictionary: key offsets must be nonempty and null-free");
  }
  auto const rows         = outputs[2]->size();
  auto const keys         = offsets - 1;
  auto const indices_type = outputs[2]->type().id();
  cudf::type_id signed_type;
  switch (indices_type) {
    case cudf::type_id::INT8:
    case cudf::type_id::UINT8: signed_type = cudf::type_id::INT8; break;
    case cudf::type_id::INT16:
    case cudf::type_id::UINT16: signed_type = cudf::type_id::INT16; break;
    case cudf::type_id::INT32:
    case cudf::type_id::UINT32: signed_type = cudf::type_id::INT32; break;
    case cudf::type_id::INT64:
    case cudf::type_id::UINT64: signed_type = cudf::type_id::INT64; break;
    default: throw std::invalid_argument("dictionary: indices must have integer type");
  }
  cudf::size_type explicit_null_count = 0;
  if (has_mask) {
    if (outputs[3]->type().id() != cudf::type_id::UINT8 ||
        static_cast<std::size_t>(outputs[3]->size()) < cudf::bitmask_allocation_size_bytes(rows)) {
      throw std::invalid_argument("dictionary: invalid null mask channel");
    }
    // The host null count determines which mask belongs on the dictionary parent.
    auto const* bits = reinterpret_cast<cudf::bitmask_type const*>(outputs[3].view().head<void>());
    explicit_null_count = rows > 0 ? cudf::null_count(bits, 0, rows, frame.stream()) : 0;
  }
  if (explicit_null_count > 0 && (outputs[2]->null_count() > 0 || signed_type != indices_type)) {
    throw std::invalid_argument("dictionary: masked indices must be null-free signed integers");
  }
  auto const null_count = explicit_null_count > 0 ? explicit_null_count : outputs[2]->null_count();
  auto& rep             = static_cast<dictionary_compressed_representation&>(
    frame.keep_representation(std::make_unique<dictionary_compressed_representation>(nullptr)));
  rep.num_rows = rows;

  struct assembly_owners {
    cudf::column::contents chars;
    cudf::column::contents indices;
    cudf::column::contents mask;
    std::vector<std::unique_ptr<cudf::column>> keys_children{1};
    std::vector<std::unique_ptr<cudf::column>> dict_children{2};
  } owned;
  failed_assembly_drain const drain{frame.stream()};
  // Shapes are validated before transfer. The direct column constructor only moves buffers and
  // children; unlike the by-value factories, allocation failure leaves these owners in the guarded
  // bundle.
  owned.keys_children[0] = frame.release(outputs[0]);
  owned.chars            = frame.release(outputs[1])->release();
  owned.dict_children[cudf::dictionary_column_view::keys_column_index] =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                   keys,
                                   std::move(*owned.chars.data),
                                   rmm::device_buffer{},
                                   0,
                                   std::move(owned.keys_children));
  owned.indices = frame.release(outputs[2])->release();
  owned.dict_children[cudf::dictionary_column_view::indices_column_index] =
    std::make_unique<cudf::column>(cudf::data_type{signed_type},
                                   rows,
                                   std::move(*owned.indices.data),
                                   rmm::device_buffer{},
                                   0,
                                   std::move(owned.indices.children));
  if (has_mask) { owned.mask = frame.release(outputs[3])->release(); }
  auto& parent_mask = explicit_null_count > 0 ? *owned.mask.data : *owned.indices.null_mask;
  rep.dict_column   = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::DICTIONARY32},
                                                   rows,
                                                   rmm::device_buffer{},
                                                   std::move(parent_mask),
                                                   null_count,
                                                   std::move(owned.dict_children));
  retain_remaining_contents(owned.chars, frame);
  retain_remaining_contents(owned.indices, frame);
  retain_remaining_contents(owned.mask, frame);
  return rep;
}

}  // namespace

compressed_representation& reconstruct_decode_representation(
  std::string const& compressor_name,
  std::vector<std::string> const& output_names,
  std::span<decode_column_slot const> outputs,
  leaf_meta_v const& meta,
  decode_frame& frame)
{
  if (output_names.size() != outputs.size() ||
      std::any_of(outputs.begin(), outputs.end(), [](auto slot) { return !slot; })) {
    throw std::invalid_argument("decode reconstruction: missing output channel");
  }
  auto const id = op_id_from_name(compressor_name);
  if (!id) { throw std::invalid_argument("decode reconstruction: unknown compressor"); }
  if (*id == OpId::Dictionary) {
    bool const has_mask = !output_names.empty() && output_names.back() == "null_mask";
    if (has_mask) {
      require_decode_channels(
        output_names, outputs, {"keys_offsets", "keys_chars", "indices", "null_mask"});
    } else {
      require_decode_channels(output_names, outputs, {"keys_offsets", "keys_chars", "indices"});
    }
    return make_decode_dictionary(outputs, has_mask, frame);
  }
  if (*id == OpId::Ans || *id == OpId::Snappy || *id == OpId::Lz4 || *id == OpId::Deflate ||
      *id == OpId::Bitcomp || *id == OpId::NvcompCascaded) {
    require_decode_channels(output_names, outputs, {"output"});
    if (outputs[0]->type().id() != cudf::type_id::UINT8) {
      throw std::invalid_argument("nvCOMP: payload must be UINT8");
    }
    switch (*id) {
      case OpId::Ans:
        return make_decode_payload<ans_compressed_representation, leaf_meta::ans>(
          outputs, meta, frame);
      case OpId::Snappy:
        return make_decode_payload<snappy_compressed_representation, leaf_meta::snappy>(
          outputs, meta, frame);
      case OpId::Lz4:
        return make_decode_payload<lz4_compressed_representation, leaf_meta::lz4>(
          outputs, meta, frame);
      case OpId::Deflate:
        return make_decode_payload<deflate_compressed_representation, leaf_meta::deflate>(
          outputs, meta, frame);
      case OpId::Bitcomp:
        return make_decode_payload<bitcomp_compressed_representation, leaf_meta::bitcomp>(
          outputs, meta, frame);
      case OpId::NvcompCascaded:
        return make_decode_payload<cascaded_compressed_representation, leaf_meta::nvcomp_cascaded>(
          outputs, meta, frame);
      default: break;
    }
  }
  std::unique_ptr<compressed_representation> owner;
  switch (*id) {
    case OpId::Identity:
      if (outputs.size() != 1) { throw std::invalid_argument("identity: expected one channel"); }
      owner                = std::make_unique<identity_compressed_representation>(nullptr);
      owner->original_type = outputs[0]->type();
      owner->num_rows      = outputs[0]->size();
      break;
    case OpId::Alp: {
      require_decode_channels(
        output_names, outputs, {"integers", "exceptions", "exception_positions", "metadata"});
      auto const integers   = outputs[0]->type().id();
      auto const exceptions = outputs[1]->type().id();
      if (!((integers == cudf::type_id::INT32 && exceptions == cudf::type_id::FLOAT32) ||
            (integers == cudf::type_id::INT64 && exceptions == cudf::type_id::FLOAT64)) ||
          outputs[2]->type().id() != cudf::type_id::INT32 ||
          outputs[3]->type().id() != cudf::type_id::UINT16 ||
          outputs[1]->size() != outputs[2]->size()) {
        throw std::invalid_argument("alp: invalid channel types or exception sizes");
      }
      owner = std::make_unique<alp_compressed_representation>(outputs[1]->type(),
                                                              outputs[0]->size(),
                                                              outputs[3]->size(),
                                                              nullptr,
                                                              nullptr,
                                                              nullptr,
                                                              nullptr);
      break;
    }
    case OpId::AlpRd: {
      require_decode_channels(
        output_names,
        outputs,
        {"right_parts", "dict_indices", "dict", "metadata", "exceptions", "exception_positions"});
      auto const right = outputs[0]->type().id();
      if ((right != cudf::type_id::UINT32 && right != cudf::type_id::UINT64) ||
          outputs[1]->type().id() != cudf::type_id::UINT8 ||
          outputs[2]->type().id() != cudf::type_id::UINT16 ||
          outputs[3]->type().id() != cudf::type_id::UINT8 || outputs[3]->size() < 1 ||
          outputs[4]->type().id() != cudf::type_id::UINT16 ||
          outputs[5]->type().id() != cudf::type_id::INT32 ||
          outputs[4]->size() != outputs[5]->size() || outputs[0]->size() != outputs[1]->size()) {
        throw std::invalid_argument("alp_rd: invalid channel types or sizes");
      }
      auto const width = frame.read_scalar(outputs[3].view().data<std::uint8_t>());
      auto const type  = cudf::data_type{right == cudf::type_id::UINT64 ? cudf::type_id::FLOAT64
                                                                        : cudf::type_id::FLOAT32};
      owner            = std::make_unique<alp_rd_compressed_representation>(
        type, outputs[0]->size(), width, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
      break;
    }
    case OpId::StrSplit: {
      bool const has_mask = outputs.size() == 3;
      if (has_mask) {
        require_decode_channels(output_names, outputs, {"offsets", "chars", "null_mask"});
      } else {
        require_decode_channels(output_names, outputs, {"offsets", "chars"});
      }
      auto const off   = outputs[0]->type().id();
      auto const chars = outputs[1]->type().id();
      if ((off != cudf::type_id::INT32 && off != cudf::type_id::INT64) ||
          (chars != cudf::type_id::UINT8 && chars != cudf::type_id::UINT32 &&
           chars != cudf::type_id::UINT64) ||
          (has_mask && outputs[2]->type().id() != cudf::type_id::UINT8)) {
        throw std::invalid_argument("str_split: invalid channel types");
      }
      owner = std::make_unique<str_split_compressed_representation>(
        outputs[0]->size() > 0 ? outputs[0]->size() - 1 : 0, nullptr, nullptr, nullptr);
      break;
    }
    case OpId::Bitextract: {
      auto const suffix = strip_bitextract_prefix(compressor_name);
      auto spec         = parse_bitextract_spec(suffix ? *suffix : std::string_view{});
      if (spec.fields.empty() || spec.fields.size() != outputs.size()) {
        throw std::invalid_argument("bitextract: invalid field count");
      }
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (spec.fields[i].name != output_names[i]) {
          throw std::invalid_argument("bitextract: invalid output name");
        }
      }
      auto& rep = static_cast<bitextract_compressed_representation&>(
        frame.keep_representation(std::make_unique<bitextract_compressed_representation>(
          std::move(spec), std::vector<std::unique_ptr<cudf::column>>{})));
      rep.fields.resize(outputs.size());
      rep.original_type = rep.spec.output_type;
      rep.num_rows      = outputs[0]->size();
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        rep.fields[i] = frame.release(outputs[i]);
      }
      return rep;
    }
    default: throw std::invalid_argument("decode reconstruction: unsupported compressor");
  }
  auto& rep = frame.keep_representation(std::move(owner));
  adopt_decode_channels(rep, outputs, frame);
  return rep;
}

std::unique_ptr<cudf::column> standalone_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  std::array const streams{stream};
  decode_session session{streams, mr};
  session.append(column_decode_request{std::cref(*this)});
  auto results = session.finish();
  return std::move(results.front());
}

void identity_compressed_representation::decompress(decode_frame& frame,
                                                    decode_column_slot output) const
{
  if (channels_.size() != 1 || !channels_[0]) {
    throw std::invalid_argument("identity decode: missing stored column");
  }
  output.adopt(std::make_unique<cudf::column>(*channels_[0], frame.stream(), frame.mr()));
}

void decode_standalone(compressed_representation const& rep,
                       decode_frame& frame,
                       decode_column_slot output)
{
  auto const* standalone = dynamic_cast<standalone_compressed_representation const*>(&rep);
  if (!standalone) {
    throw std::invalid_argument("decode standalone: representation requires the plan bridge");
  }
  standalone->decompress(frame, output);
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
