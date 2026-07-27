#pragma once

#include "codegen/plan/bit_spec.hpp"
#include "codegen/plan/leaf_desc.hpp"
#include "codegen/util/dictionary_view_helper.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <cctype>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

/// Returns a human-readable name for the given data type (for error messages).
inline std::string type_id_to_name(cudf::data_type const& type)
{
  switch (type.id()) {
    case cudf::type_id::EMPTY: return "EMPTY";
    case cudf::type_id::INT8: return "INT8";
    case cudf::type_id::INT16: return "INT16";
    case cudf::type_id::INT32: return "INT32";
    case cudf::type_id::INT64: return "INT64";
    case cudf::type_id::UINT8: return "UINT8";
    case cudf::type_id::UINT16: return "UINT16";
    case cudf::type_id::UINT32: return "UINT32";
    case cudf::type_id::UINT64: return "UINT64";
    case cudf::type_id::FLOAT32: return "FLOAT32";
    case cudf::type_id::FLOAT64: return "FLOAT64";
    case cudf::type_id::BOOL8: return "BOOL8";
    case cudf::type_id::TIMESTAMP_DAYS: return "TIMESTAMP_DAYS";
    case cudf::type_id::TIMESTAMP_SECONDS: return "TIMESTAMP_SECONDS";
    case cudf::type_id::TIMESTAMP_MILLISECONDS: return "TIMESTAMP_MILLISECONDS";
    case cudf::type_id::TIMESTAMP_MICROSECONDS: return "TIMESTAMP_MICROSECONDS";
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return "TIMESTAMP_NANOSECONDS";
    case cudf::type_id::DURATION_DAYS: return "DURATION_DAYS";
    case cudf::type_id::DURATION_SECONDS: return "DURATION_SECONDS";
    case cudf::type_id::DURATION_MILLISECONDS: return "DURATION_MILLISECONDS";
    case cudf::type_id::DURATION_MICROSECONDS: return "DURATION_MICROSECONDS";
    case cudf::type_id::DURATION_NANOSECONDS: return "DURATION_NANOSECONDS";
    case cudf::type_id::DICTIONARY32: return "DICTIONARY32";
    case cudf::type_id::STRING: return "STRING";
    case cudf::type_id::LIST: return "LIST";
    case cudf::type_id::STRUCT: return "STRUCT";
    case cudf::type_id::DECIMAL32: return "DECIMAL32";
    case cudf::type_id::DECIMAL64: return "DECIMAL64";
    case cudf::type_id::DECIMAL128: return "DECIMAL128";
    default: {
      std::ostringstream oss;
      oss << "type_id=" << static_cast<int>(type.id());
      return oss.str();
    }
  }
}

namespace simpatico {

struct compressible_output {
  std::string name;
  cudf::column_view view;
};

/// Storage/metadata base for all compressed representations.
///
/// Every rep stored in a PlanNode carries type, row count, serializable channel
/// buffers, and leaf descriptor hooks. Only representations that can be decoded
/// without PlanTree context (generic codecs) derive from
/// standalone_compressed_representation below. codegen_fused_representation
/// stores data that requires the JIT decode bridge and therefore derives only
/// from this base.
struct compressed_representation {
  // Reconstructed column's type and row count, carried uniformly by every rep.
  cudf::data_type original_type{cudf::type_id::EMPTY};
  cudf::size_type num_rows{0};
  // Channel columns in registry order (populated by subclass constructors).
  std::vector<std::unique_ptr<cudf::column>> channels_;

  compressed_representation() = default;
  compressed_representation(cudf::data_type t, cudf::size_type n) : original_type(t), num_rows(n) {}

  virtual ~compressed_representation() = default;

  /// Canonical channel enumeration: this rep's named output channels, in manifest/wire order.
  /// Generic implementation: driven by channels_ + op_info(kind()).channels from the registry.
  /// Subclasses with variable-arity or lazy synthesis (dictionary, bitextract) override this.
  virtual std::vector<compressible_output> named_channels(rmm::cuda_stream_view) const
  {
    auto const& names = op_info(kind()).channels;
    std::vector<compressible_output> out;
    out.reserve(channels_.size());
    for (size_t i = 0; i < channels_.size() && i < names.size(); ++i)
      if (channels_[i]) out.push_back({names[i], channels_[i]->view()});
    return out;
  }

  /// Channels that MUST be routed by the plan, else the driver errors -- preventing silent data
  /// loss. Default: none. str_split returns {"null_mask"} for a nullable input so a plan can never
  /// silently drop validity.
  virtual std::vector<std::string> required_channels() const { return {}; }

  /// SAFETY invariant: a representation owns everything it exposes — all
  /// compressors create new data during compression, and the original user
  /// input (column_view) is never stored in a representation.

  /// Wire size in bytes. Default sums each stored named channel. Fused Bitpack
  /// reps are already Compact when published; encode-only OverAllocate scratch
  /// and decode-only allocation slack are not exposed through named_channels().
  virtual size_t compressed_size_bytes(rmm::cuda_stream_view stream) const
  {
    size_t total = 0;
    for (auto const& o : named_channels(stream)) {
      total +=
        static_cast<size_t>(o.view.size()) * static_cast<size_t>(cudf::size_of(o.view.type()));
    }
    return total;
  }

  // ---- Leaf descriptor hooks (serialization / describe()) ----
  virtual OpId kind() const { return OpId::Unknown; }
  virtual cudf::data_type decoded_type() const { return original_type; }
  virtual leaf_meta_v describe_meta() const { return leaf_meta::none{}; }
};

/// Independently decodable representation. All generic codec representations
/// (identity, dictionary, nvCOMP, ALP, bitextract, str_split) derive from this
/// subtype, which guarantees that decompress(stream, mr) reconstructs the
/// original column without any PlanTree or JIT-bridge context.
///
/// codegen_fused_representation does NOT derive from this type; use the
/// checked decompress_standalone_representation() helper or DecodeWalk to
/// decide which path to take at runtime.
struct standalone_compressed_representation : compressed_representation {
  using compressed_representation::compressed_representation;

  virtual std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr) const = 0;
};

/// Identity / passthrough: stores a column as-is (e.g. keys_chars "stored as-is" in plan).
/// Used for outputs that are not further compressed; decompress() returns a copy.
struct identity_compressed_representation : standalone_compressed_representation {
  static std::unique_ptr<compressed_representation> from_outputs(
    std::vector<std::string> const& output_names,
    std::vector<std::unique_ptr<cudf::column>> outputs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::string* error_out);

  explicit identity_compressed_representation(std::unique_ptr<cudf::column> c)
    : standalone_compressed_representation(c ? c->type() : cudf::data_type{cudf::type_id::EMPTY},
                                           c ? c->size() : 0)
  {
    channels_.push_back(std::move(c));
  }

  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override
  {
    if (channels_.empty() || !channels_[0]) return nullptr;
    return std::make_unique<cudf::column>(*channels_[0], stream, mr);
  }
  OpId kind() const override { return OpId::Identity; }
};

/// Base compressor: compress(column, stream, mr) -> compressed_representation.
/// Decompression is representation-driven (compressed_representation::decompress),
/// so the compressor factory itself has no decompress entry point.
struct compressor {
  virtual ~compressor() = default;
  virtual std::unique_ptr<compressed_representation> compress(
    cudf::column_view column_to_compress,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) = 0;
};

/// Identity compressor: no-op passthrough, used for leaf nodes that are stored as-is.
/// A STRING column has no single contiguous payload, so compress() delegates to
/// str_split; the body is defined inline below, after str_split_compressor is declared.
struct identity_compressor : compressor {
  std::unique_ptr<compressed_representation> compress(cudf::column_view column_to_compress,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) override;
};

/// Dictionary format: stores the encoded dictionary column and a copy of keys chars.
/// In modern cuDF, chars are not accessible as a column_view, so we copy them into a UINT8 column.
struct dictionary_compressed_representation : standalone_compressed_representation {
  // Accepts the (keys_offsets, keys_chars, indices[, null_mask]) form.
  static std::unique_ptr<compressed_representation> from_outputs(
    std::vector<std::string> const& output_names,
    std::vector<std::unique_ptr<cudf::column>> outputs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::string* error_out);

  std::unique_ptr<cudf::column> dict_column;
  mutable std::unique_ptr<cudf::column> keys_chars_copy;  // Lazily created on first access
  // Lazily synthesized channels for shapes whose children cannot be viewed
  // directly: a childless empty dictionary (zero-row compress stores
  // cudf::make_empty_column(DICTIONARY32)) and childless empty keys (encode of
  // an all-null column yields zero keys). See named_channels().
  mutable std::unique_ptr<cudf::column> keys_offsets_synth;
  mutable std::unique_ptr<cudf::column> indices_synth;
  // Lazily copied validity bitmask bytes (UINT8), exposed as the optional
  // "null_mask" channel (and marked required) so a nullable column's validity
  // survives every channel-based path: .hpln IO and decomposed plans rebuild
  // via from_outputs, which cannot see the mask carried on the stored columns.
  mutable std::unique_ptr<cudf::column> null_mask_copy;

  // Constant key byte-width, measured lazily at first decompress (0 = variable, -1 = unmeasured).
  mutable std::int64_t constant_key_width = -1;

  explicit dictionary_compressed_representation(std::unique_ptr<cudf::column> dict_col)
    : dict_column(std::move(dict_col)), keys_chars_copy(nullptr)
  {
    // Per the base-class contract these describe the RECONSTRUCTED column
    // (dictionary decode yields STRING), not the stored dictionary form.
    original_type = cudf::data_type{cudf::type_id::STRING};
    num_rows      = dict_column ? dict_column->size() : 0;
  }

  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override;

  std::vector<compressible_output> named_channels(rmm::cuda_stream_view stream) const override
  {
    std::vector<compressible_output> outputs;

    // Expose keys_offsets, keys_chars, indices and — for a nullable column —
    // null_mask.
    auto dict_view            = dict_column->view();
    bool const childless_dict = dict_column->num_children() == 0;  // zero-row compress
    bool const childless_keys =  // encode of an all-null column: zero keys, no offsets child
      childless_dict || cudf::dictionary_column_view(dict_view).keys().num_children() == 0;

    if (childless_keys) {
      if (!keys_offsets_synth) {
        // Canonical empty keys: a single zero offset.
        keys_offsets_synth =
          cudf::make_fixed_width_column(cudf::data_type(cudf::type_id::INT32),
                                        1,
                                        cudf::mask_state::UNALLOCATED,
                                        stream,
                                        rmm::mr::get_current_device_resource_ref());
        cudaMemsetAsync(
          keys_offsets_synth->mutable_view().head<void>(), 0, sizeof(std::int32_t), stream.value());
        cudaStreamSynchronize(stream.value());
      }
      outputs.push_back({"keys_offsets", keys_offsets_synth->view()});
    } else {
      outputs.push_back(
        {"keys_offsets", get_dictionary_child_view(dict_view, dictionary_view_kind::KeysOffsets)});
    }

    // For keys_chars, we need to create a UINT8 column since chars are not a
    // column child in modern cuDF. Always emitted — zero-row when the keys
    // have no bytes — so the channel arity is stable and an all-empty-keys
    // table reads back.
    if (!keys_chars_copy) {
      // The resulting column's device_buffer remembers ``stream`` for its own
      // eventual deallocation, so the caller-supplied stream must stay valid
      // for the column's lifetime (a private stream destroyed at the end of
      // this scope would leave a dangling handle).
      auto mr                      = rmm::mr::get_current_device_resource_ref();
      auto [chars_ptr, chars_size] = childless_dict
                                       ? std::pair<char const*, int64_t>{nullptr, 0}
                                       : get_dictionary_keys_chars_info(dict_view, stream);
      keys_chars_copy = cudf::make_fixed_width_column(cudf::data_type(cudf::type_id::UINT8),
                                                      static_cast<cudf::size_type>(chars_size),
                                                      cudf::mask_state::UNALLOCATED,
                                                      stream,
                                                      mr);
      if (chars_ptr != nullptr && chars_size > 0) {
        cudaMemcpyAsync(keys_chars_copy->mutable_view().head<void>(),
                        chars_ptr,
                        chars_size,
                        cudaMemcpyDeviceToDevice,
                        stream.value());
      }
      cudaStreamSynchronize(stream.value());
    }
    outputs.push_back({"keys_chars", keys_chars_copy->view()});

    if (childless_dict) {
      if (!indices_synth) {
        indices_synth = cudf::make_empty_column(cudf::data_type(cudf::type_id::INT32));
      }
      outputs.push_back({"indices", indices_synth->view()});
    } else {
      outputs.push_back(
        {"indices", get_dictionary_child_view(dict_view, dictionary_view_kind::Indices)});
    }

    if (dict_column->null_count() > 0) {
      ensure_null_mask_copy(dict_view, stream);
      outputs.push_back({"null_mask", null_mask_copy->view()});
    }
    return outputs;
  }

  std::vector<std::string> required_channels() const override
  {
    if (dict_column && dict_column->null_count() > 0) return {"null_mask"};
    return {};
  }

  OpId kind() const override { return OpId::Dictionary; }

 private:
  // Copy `source`'s validity bitmask into the owned UINT8 null_mask_copy
  // column (no-op if already built).
  void ensure_null_mask_copy(cudf::column_view const& source, rmm::cuda_stream_view stream) const
  {
    if (null_mask_copy) return;
    auto mr                 = rmm::mr::get_current_device_resource_ref();
    rmm::device_buffer bits = cudf::copy_bitmask(source, stream, mr);
    auto const mask_bytes =
      static_cast<cudf::size_type>(cudf::bitmask_allocation_size_bytes(source.size()));
    null_mask_copy = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::UINT8}, mask_bytes, std::move(bits), rmm::device_buffer{}, 0);
    cudaStreamSynchronize(stream.value());
  }
};

/// Dictionary compressor: STRING column only. Encodes via cudf dictionary encode; stores the
/// keys buffer + offsets + indices (keys_offsets/keys_chars/indices form).
struct dictionary_compressor : compressor {
  std::unique_ptr<compressed_representation> compress(cudf::column_view column_to_compress,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) override;
};

// -----------------------------------------------------------------------------
// str_split: decompose a STRING column into {offsets, chars, null_mask} channels.
// Structural (non-codegen) operator -- byte compression is delegated to the
// channel codecs (offsets -> delta -> bitpack; chars -> lz4). Modeled on the
// ALP rep (dedicated struct + from_outputs); decode reassembles via
// cudf::make_strings_column. Conditional arity: a non-null column exposes
// {offsets, chars}; a nullable column also exposes null_mask, which is marked
// required() so the driver errors if the plan fails to route it.
// -----------------------------------------------------------------------------
// channels_[0] = offsets (INT32 or INT64), channels_[1] = chars (UINT8/UINT32/UINT64),
// channels_[2] = null_mask (UINT8 bitmask bytes, only present when nullable).
// decompress() copies channels into make_strings_column; channels_ is left intact.
struct str_split_compressed_representation : standalone_compressed_representation {
  static std::unique_ptr<compressed_representation> from_outputs(
    std::vector<std::string> const& output_names,
    std::vector<std::unique_ptr<cudf::column>> outputs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::string* error_out);

  str_split_compressed_representation(cudf::size_type n_rows,
                                      std::unique_ptr<cudf::column> offsets,
                                      std::unique_ptr<cudf::column> chars,
                                      std::unique_ptr<cudf::column> null_mask)
    : standalone_compressed_representation(cudf::data_type{cudf::type_id::STRING}, n_rows)
  {
    channels_.push_back(std::move(offsets));
    channels_.push_back(std::move(chars));
    if (null_mask) channels_.push_back(std::move(null_mask));
  }

  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override;

  std::vector<std::string> required_channels() const override
  {
    if (channels_.size() > 2 && channels_[2]) return {"null_mask"};
    return {};
  }

  OpId kind() const override { return OpId::StrSplit; }
};

struct str_split_compressor : compressor {
  std::unique_ptr<compressed_representation> compress(cudf::column_view column_to_compress,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) override;
};

// A STRING column has no single contiguous payload, so it decomposes via
// str_split; any other type is copied verbatim into an identity leaf.
inline std::unique_ptr<compressed_representation> identity_compressor::compress(
  cudf::column_view column_to_compress,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (column_to_compress.type().id() == cudf::type_id::STRING) {
    return str_split_compressor{}.compress(column_to_compress, stream, mr);
  }
  auto col_copy = std::make_unique<cudf::column>(column_to_compress, stream, mr);
  return std::make_unique<identity_compressed_representation>(std::move(col_copy));
}

// -----------------------------------------------------------------------------
// nvcomp-backed representations
// -----------------------------------------------------------------------------
//
// Every nvcomp codec (ans/bitcomp/cascaded and the simpler snappy/lz4/deflate)
// stores the same thing: an opaque compressed byte payload held as a single
// UINT8 channel (channels_[0], size = actual compressed bytes). Concrete reps
// add codec-specific metadata fields and supply kind()/describe_meta()/decompress().
struct nvcomp_payload_rep : standalone_compressed_representation {
  size_t uncompressed_size = 0;

  // Takes ownership of the worst-case-sized device_buffer and wraps it as a
  // UINT8 column of exactly comp_sz elements (logical size may be < buffer size;
  // cudf allows that).
  nvcomp_payload_rep(cudf::data_type t,
                     cudf::size_type n,
                     std::unique_ptr<rmm::device_buffer> data,
                     size_t comp_sz,
                     size_t uncomp_sz)
    : standalone_compressed_representation(t, n), uncompressed_size(uncomp_sz)
  {
    channels_.push_back(
      std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT8},
                                     static_cast<cudf::size_type>(comp_sz),
                                     data ? std::move(*data) : rmm::device_buffer{},
                                     rmm::device_buffer{},
                                     0));
  }

  // Raw access for decompress() impls (avoids a device_buffer round-trip).
  const void* payload_data() const
  {
    return (channels_.empty() || !channels_[0]) ? nullptr : channels_[0]->view().head<void>();
  }
  size_t payload_size() const
  {
    return (channels_.empty() || !channels_[0]) ? 0 : static_cast<size_t>(channels_[0]->size());
  }
};

// Template base for nvcomp codecs whose leaf metadata is exactly
// (uncompressed_size, type_id): snappy, lz4, deflate, ans. Fixes
// kind()/describe_meta() from its parameters; each concrete struct supplies its
// own decompress() (.cu file) and from_outputs() (representation_factory.cpp).
// bitcomp/cascaded carry extra compress opts and derive from nvcomp_payload_rep.
template <OpId K, typename MetaT>
struct nvcomp_simple_rep_base : nvcomp_payload_rep {
  using nvcomp_payload_rep::nvcomp_payload_rep;

  OpId kind() const override { return K; }
  leaf_meta_v describe_meta() const override
  {
    return MetaT{uncompressed_size, static_cast<std::int32_t>(original_type.id())};
  }

  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override = 0;
};

// -----------------------------------------------------------------------------
// nvcomp ANS / Bitcomp compressors
// -----------------------------------------------------------------------------

// Reconstructed generically via nvcomp_simple_from_outputs (representation_factory.cpp).
struct ans_compressed_representation : nvcomp_simple_rep_base<OpId::Ans, leaf_meta::ans> {
  using nvcomp_simple_rep_base::nvcomp_simple_rep_base;
  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override;
};

struct ans_compressor : compressor {
  std::unique_ptr<compressed_representation> compress(cudf::column_view column_to_compress,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) override;
};

// Parses a bitcomp suffix into the algorithm knob. `suffix` is the
// part AFTER the underscore in `bitcomp_<suffix>`. Recognised values:
// "default" → algorithm=0 (best ratio, == bare `bitcomp`); "sparse"
// → algorithm=1 (faster on zero-rich data). Bare `bitcomp` is
// matched separately by the compressor registry. Returns false on
// any other suffix.
bool parse_bitcomp_suffix(std::string_view suffix, int* algorithm);

struct bitcomp_compressed_representation : nvcomp_payload_rep {
  static std::unique_ptr<compressed_representation> from_outputs(
    std::vector<std::string> const& output_names,
    std::vector<std::unique_ptr<cudf::column>> outputs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::string* error_out,
    leaf_meta_v const& meta = leaf_meta::none{});

  // Algorithm used at compress time so decompress hits the same Manager cache slot.
  int compress_algorithm;

  bitcomp_compressed_representation(cudf::data_type t,
                                    cudf::size_type n,
                                    std::unique_ptr<rmm::device_buffer> data,
                                    size_t comp_size,
                                    size_t uncomp_size,
                                    int algorithm = 0)
    : nvcomp_payload_rep(t, n, std::move(data), comp_size, uncomp_size),
      compress_algorithm(algorithm)
  {
  }

  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override;

  OpId kind() const override { return OpId::Bitcomp; }
  leaf_meta_v describe_meta() const override
  {
    return leaf_meta::bitcomp{
      uncompressed_size, static_cast<std::int32_t>(original_type.id()), compress_algorithm};
  }
};

struct bitcomp_compressor : compressor {
  // algorithm: 0 = default (best ratio), 1 = sparse (faster on data
  // with lots of zeroes). Matches nvcomp's `algorithm` field in
  // nvcompBatchedBitcompCompressOpts_t.
  bitcomp_compressor(int algorithm = 0) : algorithm_(algorithm) {}

  std::unique_ptr<compressed_representation> compress(cudf::column_view column_to_compress,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) override;

 private:
  int algorithm_;
};

// -----------------------------------------------------------------------------
// nvcomp Cascaded compressor
// -----------------------------------------------------------------------------
//
// DSL surface:
//   `input -> nvcomp_cascaded`           uses nvcomp defaults
//                                        (num_deltas=1, num_RLEs=2, use_bp=1).
//   `input -> nvcomp_cascaded_<N>D<M>R<K>B`
//                                        explicit opts: N deltas, M RLEs,
//                                        K bitpack (0 or 1).
//
// Parse a suffix of the form "<N>D<M>R<K>B" into (deltas, rles, bp).
// Returns true on success. Empty suffix → use nvcomp defaults (caller's
// responsibility to fall back).
bool parse_nvcomp_cascaded_suffix(std::string_view suffix, int* deltas, int* rles, int* bp);

struct cascaded_compressed_representation : nvcomp_payload_rep {
  static std::unique_ptr<compressed_representation> from_outputs(
    std::vector<std::string> const& output_names,
    std::vector<std::unique_ptr<cudf::column>> outputs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::string* error_out,
    leaf_meta_v const& meta = leaf_meta::none{});

  // Opts used at compress time — stashed so decompress hits the same Manager
  // cache key.
  int compress_num_deltas;
  int compress_num_RLEs;
  int compress_use_bp;

  cascaded_compressed_representation(cudf::data_type t,
                                     cudf::size_type n,
                                     std::unique_ptr<rmm::device_buffer> data,
                                     size_t comp_size,
                                     size_t uncomp_size,
                                     int num_deltas,
                                     int num_RLEs,
                                     int use_bp)
    : nvcomp_payload_rep(t, n, std::move(data), comp_size, uncomp_size),
      compress_num_deltas(num_deltas),
      compress_num_RLEs(num_RLEs),
      compress_use_bp(use_bp)
  {
  }

  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override;

  OpId kind() const override { return OpId::NvcompCascaded; }
  leaf_meta_v describe_meta() const override
  {
    return leaf_meta::nvcomp_cascaded{uncompressed_size,
                                      static_cast<std::int32_t>(original_type.id()),
                                      compress_num_deltas,
                                      compress_num_RLEs,
                                      compress_use_bp};
  }
};

struct cascaded_compressor : compressor {
  // nvcomp default opts: num_deltas=1, num_RLEs=2, use_bp=1.
  cascaded_compressor(int num_deltas = 1, int num_RLEs = 2, int use_bp = 1)
    : num_deltas_(num_deltas), num_RLEs_(num_RLEs), use_bp_(use_bp)
  {
  }

  std::unique_ptr<compressed_representation> compress(cudf::column_view column_to_compress,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) override;

 private:
  int num_deltas_;
  int num_RLEs_;
  int use_bp_;
};

// -----------------------------------------------------------------------------
// Simple nvcomp codecs: Snappy, LZ4, GDeflate ("deflate" in the DSL) — see
// nvcomp_simple_rep_base above.
// -----------------------------------------------------------------------------

// snappy/lz4/deflate: reconstructed generically via nvcomp_simple_from_outputs.
struct snappy_compressed_representation : nvcomp_simple_rep_base<OpId::Snappy, leaf_meta::snappy> {
  using nvcomp_simple_rep_base::nvcomp_simple_rep_base;
  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override;
};

struct lz4_compressed_representation : nvcomp_simple_rep_base<OpId::Lz4, leaf_meta::lz4> {
  using nvcomp_simple_rep_base::nvcomp_simple_rep_base;
  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override;
};

struct deflate_compressed_representation
  : nvcomp_simple_rep_base<OpId::Deflate, leaf_meta::deflate> {
  using nvcomp_simple_rep_base::nvcomp_simple_rep_base;
  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override;
};

struct snappy_compressor : compressor {
  std::unique_ptr<compressed_representation> compress(cudf::column_view column_to_compress,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) override;
};

struct lz4_compressor : compressor {
  std::unique_ptr<compressed_representation> compress(cudf::column_view column_to_compress,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) override;
};

struct deflate_compressor : compressor {
  std::unique_ptr<compressed_representation> compress(cudf::column_view column_to_compress,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) override;
};

// -----------------------------------------------------------------------------
// ALP (Adaptive Lossless floating-Point), FLOAT32, multi-output.
// See SIGMOD '24 (Afroozeh) + G-ALP DaMoN '25.
// channels_[0]=integers (INT32/INT64), [1]=exceptions (FLOAT32/FLOAT64),
// [2]=exception_positions (INT32), [3]=metadata (UINT16, one per 1024-vector).
struct alp_compressed_representation : standalone_compressed_representation {
  static std::unique_ptr<compressed_representation> from_outputs(
    std::vector<std::string> const& output_names,
    std::vector<std::unique_ptr<cudf::column>> outputs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::string* error_out);

  cudf::size_type num_vectors;  // ceil(num_rows / 1024)

  alp_compressed_representation(cudf::data_type type,
                                cudf::size_type n_rows,
                                cudf::size_type n_vectors,
                                std::unique_ptr<cudf::column> integers_in,
                                std::unique_ptr<cudf::column> exceptions_in,
                                std::unique_ptr<cudf::column> exception_positions_in,
                                std::unique_ptr<cudf::column> metadata_in);

  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override;

  // Named accessors for decompress impl (channels_ in registry order).
  const cudf::column* integers() const
  {
    return channels_.size() > 0 ? channels_[0].get() : nullptr;
  }
  const cudf::column* exceptions() const
  {
    return channels_.size() > 1 ? channels_[1].get() : nullptr;
  }
  const cudf::column* exception_positions() const
  {
    return channels_.size() > 2 ? channels_[2].get() : nullptr;
  }
  const cudf::column* metadata() const
  {
    return channels_.size() > 3 ? channels_[3].get() : nullptr;
  }

  OpId kind() const override { return OpId::Alp; }
};

struct alp_compressor : compressor {
  std::unique_ptr<compressed_representation> compress(cudf::column_view column_to_compress,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) override;
};

// ALP-RD (Right-Dictionary), FLOAT32, multi-output. Column-wide K=8 dict +
// right_bw; deviates from cwida CPU reference (per-rowgroup dict) to keep
// right_parts fixed-width for downstream bitpack.
// channels_[0]=right_parts (UINT32/UINT64), [1]=dict_indices (UINT8),
// [2]=dict (UINT16, 8 entries), [3]=metadata (UINT8, 1 entry: right_bw),
// [4]=exceptions (UINT16), [5]=exception_positions (INT32).
struct alp_rd_compressed_representation : standalone_compressed_representation {
  static std::unique_ptr<compressed_representation> from_outputs(
    std::vector<std::string> const& output_names,
    std::vector<std::unique_ptr<cudf::column>> outputs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::string* error_out);

  uint8_t right_bw;  // bits in right part (1..31 f32, 1..63 f64)

  alp_rd_compressed_representation(cudf::data_type type,
                                   cudf::size_type n_rows,
                                   uint8_t right_bw_in,
                                   std::unique_ptr<cudf::column> right_parts_in,
                                   std::unique_ptr<cudf::column> dict_indices_in,
                                   std::unique_ptr<cudf::column> dict_in,
                                   std::unique_ptr<cudf::column> metadata_in,
                                   std::unique_ptr<cudf::column> exceptions_in,
                                   std::unique_ptr<cudf::column> exception_positions_in);

  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override;

  // Named accessors for decompress impl (channels_ in registry order).
  const cudf::column* right_parts() const
  {
    return channels_.size() > 0 ? channels_[0].get() : nullptr;
  }
  const cudf::column* dict_indices() const
  {
    return channels_.size() > 1 ? channels_[1].get() : nullptr;
  }
  const cudf::column* dict() const { return channels_.size() > 2 ? channels_[2].get() : nullptr; }
  const cudf::column* metadata() const
  {
    return channels_.size() > 3 ? channels_[3].get() : nullptr;
  }
  const cudf::column* exceptions() const
  {
    return channels_.size() > 4 ? channels_[4].get() : nullptr;
  }
  const cudf::column* exception_positions() const
  {
    return channels_.size() > 5 ? channels_[5].get() : nullptr;
  }

  OpId kind() const override { return OpId::AlpRd; }
  leaf_meta_v describe_meta() const override { return leaf_meta::alp_rd{right_bw}; }
};

struct alp_rd_compressor : compressor {
  std::unique_ptr<compressed_representation> compress(cudf::column_view column_to_compress,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) override;
};

// ── bitextract / bitjoin ──────────────────────────────────────────────────────
// Bit-spec parsing (bitfield_spec, bitextract_spec_result, parse_* etc.) lives
// in bit_spec.hpp (included above).

// Forward declarations — implemented in bitjoin_bitextract.cu
void launch_bitextract_field(cudf::column_view const& input_col,
                             cudf::mutable_column_view const& output_col,
                             int lo_bit,
                             uint32_t n_bits,
                             cudaStream_t stream);

void launch_bitjoin_field(cudf::mutable_column_view const& output_col,
                          cudf::column_view const& input_col,
                          int src_lo_bit,
                          int dst_lo_bit,
                          uint32_t n_bits,
                          cudaStream_t stream);

void launch_check_truncation(cudf::column_view const& input_col,
                             uint64_t selected_mask,
                             uint32_t* d_flag,
                             uint32_t flag_bit,
                             cudaStream_t stream);

struct bitextract_compressed_representation : standalone_compressed_representation {
  static std::unique_ptr<compressed_representation> from_outputs(
    bitextract_spec_result spec,
    std::vector<std::string> const& output_names,
    std::vector<std::unique_ptr<cudf::column>> outputs,
    std::string* error_out);

  bitextract_spec_result spec;
  std::vector<std::unique_ptr<cudf::column>> fields;

  bitextract_compressed_representation(bitextract_spec_result spec_in,
                                       std::vector<std::unique_ptr<cudf::column>> fields_in)
    : spec(std::move(spec_in)), fields(std::move(fields_in))
  {
  }

  std::vector<compressible_output> named_channels(rmm::cuda_stream_view) const override
  {
    std::vector<compressible_output> out;
    out.reserve(spec.fields.size());
    for (size_t i = 0; i < spec.fields.size(); ++i) {
      if (fields[i]) out.push_back({spec.fields[i].name, fields[i]->view()});
    }
    return out;
  }

  // Implemented in bitjoin_bitextract.cu
  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override;
};

struct bitextract_compressor : compressor {
  bitextract_spec_result spec;
  std::string suffix_str;

  explicit bitextract_compressor(std::string_view suffix)
    : spec(parse_bitextract_spec(suffix)), suffix_str(suffix)
  {
    if (spec.fields.empty())
      throw std::invalid_argument("bitextract: invalid spec '" + std::string(suffix) + "'");
  }

  // Implemented in bitjoin_bitextract.cu
  std::unique_ptr<compressed_representation> compress(cudf::column_view column,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) override;
};

// -----------------------------------------------------------------------------
// Codegen-fused representations — produced only by the JIT encoder.
// Storage-only: reconstruction goes through the PlanTree decode bridge.
// -----------------------------------------------------------------------------

// Holds a codegen-fused subtree node (Delta, Rle, Bitpack, ...). Buffers are
// tagged with their manifest field name ("delta_first", "rle_runs_offsets", ...),
// which differs from the registry's logical channel names — so this rep keeps its
// own named-buffer storage and named_channels() rather than the registry-driven
// generic path. The tree structure and tail-routing are recovered from PlanTree;
// DecodeWalk reconstructs it through the generated inverse kernels.
struct codegen_fused_representation : compressed_representation {
  OpId op_id_;
  // Buffers in manifest order, each tagged with its manifest field name.
  std::vector<std::pair<std::string, std::unique_ptr<cudf::column>>> buffers;

  codegen_fused_representation(OpId id, cudf::data_type type, cudf::size_type n_rows)
    : compressed_representation(type, n_rows), op_id_(id)
  {
  }

  std::vector<compressible_output> named_channels(rmm::cuda_stream_view) const override
  {
    std::vector<compressible_output> out;
    out.reserve(buffers.size());
    for (auto const& [name, col] : buffers) {
      if (col) out.push_back({name, col->view()});
    }
    return out;
  }

  OpId kind() const override { return op_id_; }
};

/// Safely decompress a representation that must be standalone-decodable.
///
/// Downcasts to standalone_compressed_representation; calls decompress() on
/// success. Returns nullptr and writes a deterministic message to error_out
/// when rep is a storage-only type such as codegen_fused_representation.
/// Callers should use this instead of calling rep->decompress() directly.
std::unique_ptr<cudf::column> decompress_standalone_representation(
  compressed_representation const* rep,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  std::string* error_out);

}  // namespace simpatico
