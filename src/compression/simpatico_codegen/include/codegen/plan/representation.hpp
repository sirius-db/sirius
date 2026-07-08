#ifndef SIMPATICO_REPRESENTATION_HPP
#define SIMPATICO_REPRESENTATION_HPP

#include "codegen/plan/leaf_desc.hpp"
#include "dictionary_view_helper.hpp"  // src/util on include path

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

/// Opaque compressed data; decompress(stream, mr) reconstructs the original column.
struct compressed_representation {
  // Reconstructed column's type and row count, carried uniformly by every rep.
  cudf::data_type original_type{cudf::type_id::EMPTY};
  cudf::size_type num_rows{0};

  compressed_representation() = default;
  compressed_representation(cudf::data_type t, cudf::size_type n) : original_type(t), num_rows(n) {}

  virtual ~compressed_representation() = default;
  virtual std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr) const = 0;

  /// Canonical channel enumeration: this rep's named output channels, in
  /// manifest/wire order. This is the ONE accessor for a rep's channels,
  /// serving both the compress/writer side (further-compress a channel, sum
  /// wire size) and the decode-side JIT gather.
  ///
  virtual std::vector<compressible_output> named_channels(rmm::cuda_stream_view stream) const
  {
    return {};
  }

  /// Channels that MUST be routed by the plan, else the driver errors -- preventing silent data
  /// loss. Default: none. str_split returns {"null_mask"} for a nullable input so a plan can never
  /// silently drop validity.
  virtual std::vector<std::string> required_channels() const { return {}; }

  /// Release ownership of a named output. Returns nullptr if not supported or name not found.
  /// This avoids copying when the output is a leaf and the representation already owns the data.
  ///
  /// SAFETY: This method should only be called for data that the representation OWNS, not for
  /// views of user-provided data. All compressors create new data during compression:
  /// - the fused codegen backend creates new buffers (e.g. bitpack chunk_min,
  ///   chunk_count, chunk_bits, packed) for delta / rle / bitpack / for regions
  /// - identity_compressor makes a COPY of the input (so safe to release the copy)
  ///
  /// The original user input (column_view) is never stored in representations.
  virtual std::unique_ptr<cudf::column> release_output(std::string const& name) { return nullptr; }

  /// Wire size in bytes (tight Compact layout). Default sums each dense
  /// named_channels channel; override when actual size is tracked out-of-band
  /// (e.g. sparse BITPACK OverAllocate buffers).
  virtual size_t compressed_size_bytes(rmm::cuda_stream_view stream) const
  {
    size_t total = 0;
    for (auto const& o : named_channels(stream)) {
      // cudf::size_of() only supports fixed-width types (e.g. dictionary's
      // fast-mode "keys" channel is a raw STRING column) — account for
      // offsets + chars directly instead of calling it on a STRING view.
      if (o.view.type().id() == cudf::type_id::STRING) {
        cudf::strings_column_view scv(o.view);
        total += static_cast<size_t>(o.view.size() + 1) * sizeof(int32_t) +
                 static_cast<size_t>(scv.chars_size(stream));
      } else {
        total +=
          static_cast<size_t>(o.view.size()) * static_cast<size_t>(cudf::size_of(o.view.type()));
      }
    }
    return total;
  }

  // ---- Leaf descriptor hooks (serialization / describe()) ----
  virtual PlanLeafKind kind() const { return PlanLeafKind::Unknown; }
  virtual cudf::data_type decoded_type() const { return original_type; }
  virtual leaf_meta_v describe_meta() const { return leaf_meta::none{}; }
};

/// Identity / passthrough: stores a column as-is (e.g. keys_chars "stored as-is" in plan).
/// Used for outputs that are not further compressed; decompress() returns a copy.
struct identity_compressed_representation : compressed_representation {
  static std::unique_ptr<compressed_representation> from_outputs(
    std::vector<std::string> const& output_names,
    std::vector<std::unique_ptr<cudf::column>> outputs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::string* error_out);

  std::unique_ptr<cudf::column> col;

  explicit identity_compressed_representation(std::unique_ptr<cudf::column> c)
    : compressed_representation(c ? c->type() : cudf::data_type{cudf::type_id::EMPTY},
                                c ? c->size() : 0),
      col(std::move(c))
  {
  }

  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override
  {
    if (col == nullptr) return nullptr;
    return std::make_unique<cudf::column>(*col, stream, mr);
  }
  std::vector<compressible_output> named_channels(rmm::cuda_stream_view) const override
  {
    if (col == nullptr) return {};
    return {{"data", col->view()}};
  }
  PlanLeafKind kind() const override { return PlanLeafKind::Identity; }
};

/// Base compressor: compress(column, stream, mr) -> compressed_representation, decompress(stream,
/// mr) -> column.
struct compressor {
  virtual ~compressor() = default;
  virtual std::unique_ptr<compressed_representation> compress(
    cudf::column_view column_to_compress,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) = 0;
  virtual std::unique_ptr<cudf::column> decompress(
    compressed_representation const& data_to_decompress,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) = 0;
};

/// Identity compressor: no-op passthrough, used for leaf nodes that are stored as-is.
struct identity_compressor : compressor {
  std::unique_ptr<compressed_representation> compress(cudf::column_view column_to_compress,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) override
  {
    auto col_copy = std::make_unique<cudf::column>(column_to_compress, stream, mr);
    return std::make_unique<identity_compressed_representation>(std::move(col_copy));
  }
  std::unique_ptr<cudf::column> decompress(compressed_representation const& data_to_decompress,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) override
  {
    return data_to_decompress.decompress(stream, mr);
  }
};

/// Dictionary format: stores the encoded dictionary column and a copy of keys chars.
/// In modern cuDF, chars are not accessible as a column_view, so we copy them into a UINT8 column.
///
/// Two modes of operation:
/// 1. Full dict_column mode: stores the complete dictionary column from encode()
/// 2. Keys+indices mode: stores separate keys (strings) and indices columns for fast reconstruction
///    This mode avoids the expensive make_strings_column reconstruction.
struct dictionary_compressed_representation : compressed_representation {
  // Accepts the (keys, indices[, null_mask]) form or the (keys_offsets,
  // keys_chars, indices[, null_mask]) form.
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

  // For fast reconstruction mode: store keys and indices separately
  std::unique_ptr<cudf::column> keys_column;   // Keys as strings column (for fast reconstruction)
  std::unique_ptr<cudf::column> indices_only;  // Indices column (for fast reconstruction)
  bool fast_mode = false;                      // True if using keys+indices mode

  explicit dictionary_compressed_representation(std::unique_ptr<cudf::column> dict_col)
    : dict_column(std::move(dict_col)), keys_chars_copy(nullptr), fast_mode(false)
  {
    // Per the base-class contract these describe the RECONSTRUCTED column
    // (dictionary decode yields STRING), not the stored dictionary form.
    original_type = cudf::data_type{cudf::type_id::STRING};
    num_rows      = dict_column ? dict_column->size() : 0;
  }

  // Fast reconstruction constructor: keys (strings column) + indices
  dictionary_compressed_representation(std::unique_ptr<cudf::column> keys,
                                       std::unique_ptr<cudf::column> indices)
    : dict_column(nullptr),
      keys_chars_copy(nullptr),
      keys_column(std::move(keys)),
      indices_only(std::move(indices)),
      fast_mode(true)
  {
    original_type = cudf::data_type{cudf::type_id::STRING};
    num_rows      = indices_only ? indices_only->size() : 0;
  }

  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override;

  std::vector<compressible_output> named_channels(rmm::cuda_stream_view stream) const override
  {
    std::vector<compressible_output> outputs;

    if (fast_mode) {
      // Fast mode: expose keys (strings) and indices directly. The indices
      // column carries the parent validity (get_indices_annotated), but a
      // codec routed onto the indices channel keeps only its data bytes, so
      // for a nullable column the mask additionally travels as its own
      // channel.
      if (keys_column) { outputs.push_back({"keys", keys_column->view()}); }
      if (indices_only) { outputs.push_back({"indices", indices_only->view()}); }
      if (indices_only && indices_only->null_count() > 0) {
        ensure_null_mask_copy(indices_only->view(), stream);
        outputs.push_back({"null_mask", null_mask_copy->view()});
      }
      return outputs;
    }

    // Full dict_column mode: expose keys_offsets, keys_chars, indices and —
    // for a nullable column — null_mask.
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
        cudaMemsetAsync(keys_offsets_synth->mutable_view().head<void>(),
                        0,
                        sizeof(std::int32_t),
                        stream.value());
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
    bool const nullable = fast_mode ? (indices_only && indices_only->null_count() > 0)
                                    : (dict_column && dict_column->null_count() > 0);
    if (nullable) return {"null_mask"};
    return {};
  }

  std::unique_ptr<cudf::column> release_output(std::string const& name) override
  {
    if (fast_mode) {
      if (name == "keys" && keys_column) return std::move(keys_column);
      if (name == "indices" && indices_only) return std::move(indices_only);
    }
    return nullptr;
  }
  PlanLeafKind kind() const override { return PlanLeafKind::Dictionary; }

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
    null_mask_copy = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT8},
                                                    mask_bytes,
                                                    std::move(bits),
                                                    rmm::device_buffer{},
                                                    0);
    cudaStreamSynchronize(stream.value());
  }
};

/// Dictionary compressor: STRING column only. Encodes via cudf dictionary encode; stores keys
/// buffer + offsets + indices. `fast` selects the 2-buffer keys+indices in-memory representation
/// (DSL name "dictionary_fast") instead of the 3-buffer keys_offsets/keys_chars/indices form.
struct dictionary_compressor : compressor {
  explicit dictionary_compressor(bool fast = false) : fast_(fast) {}

  std::unique_ptr<compressed_representation> compress(cudf::column_view column_to_compress,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) override;
  std::unique_ptr<cudf::column> decompress(compressed_representation const& data_to_decompress,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) override;

  bool fast_;
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
struct str_split_compressed_representation : compressed_representation {
  static std::unique_ptr<compressed_representation> from_outputs(
    std::vector<std::string> const& output_names,
    std::vector<std::unique_ptr<cudf::column>> outputs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::string* error_out);

  // The decomposed channels. mutable: decompress() moves them into make_strings_column.
  mutable std::unique_ptr<cudf::column> offsets_;    // INT32 (or INT64 for >2GB chars), size num_rows+1
  mutable std::unique_ptr<cudf::column> chars_;      // UINT8, or widened (UINT32/UINT64) past 2GB
  mutable std::unique_ptr<cudf::column> null_mask_;  // UINT8 bitmask bytes, or null (no nulls)

  str_split_compressed_representation(cudf::size_type n_rows,
                                      std::unique_ptr<cudf::column> offsets,
                                      std::unique_ptr<cudf::column> chars,
                                      std::unique_ptr<cudf::column> null_mask)
    : compressed_representation(cudf::data_type{cudf::type_id::STRING}, n_rows),
      offsets_(std::move(offsets)),
      chars_(std::move(chars)),
      null_mask_(std::move(null_mask))
  {
  }

  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override;

  std::vector<compressible_output> named_channels(rmm::cuda_stream_view /*stream*/) const override
  {
    std::vector<compressible_output> out;
    // The single-shot decompress() MOVES the channels into make_strings_column;
    // a consumed rep has no channels left to enumerate (describe/serialize
    // must run before decompress).
    if (!offsets_ || !chars_) return out;
    out.push_back({"offsets", offsets_->view()});
    out.push_back({"chars", chars_->view()});
    if (null_mask_) out.push_back({"null_mask", null_mask_->view()});  // 2- or 3-channel
    return out;
  }

  std::vector<std::string> required_channels() const override
  {
    if (null_mask_) return {"null_mask"};
    return {};
  }

  std::unique_ptr<cudf::column> release_output(std::string const& name) override
  {
    if (name == "offsets" && offsets_) return std::move(offsets_);
    if (name == "chars" && chars_) return std::move(chars_);
    if (name == "null_mask" && null_mask_) return std::move(null_mask_);
    return nullptr;
  }
  // kind() defaults to Unknown -- only the deferred .hpln path reads it.
};

struct str_split_compressor : compressor {
  std::unique_ptr<compressed_representation> compress(cudf::column_view column_to_compress,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) override;
  std::unique_ptr<cudf::column> decompress(compressed_representation const& data_to_decompress,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) override;
};

// -----------------------------------------------------------------------------
// Bitpack compressor: variance-based chunking, min per chunk, bit-packed (value - min)
// -----------------------------------------------------------------------------

/// Bitpack format: chunks determined by variance heuristic (when range would need ~2x bits, start
/// new chunk). Each chunk stores: min (original type), then bit-packed (value - min) with bits =
/// ceil(log2(range+1)). Uses cudf::column for type-erased storage so chunk_min preserves
/// INT8/INT16/INT32/INT64/timestamp/etc.
struct bitpack_compressed_representation : compressed_representation {
  // ``packed`` may be absent (entropy-tail-routed); matched by channel name.
  static std::unique_ptr<compressed_representation> from_outputs(
    std::vector<std::string> const& output_names,
    std::vector<std::unique_ptr<cudf::column>> outputs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::string* error_out);

  std::unique_ptr<cudf::column> chunk_min;    // min per chunk (original_type; one row per chunk)
  std::unique_ptr<cudf::column> chunk_count;  // INT32, one row per chunk
  std::unique_ptr<cudf::column> chunk_bits;   // UINT8, bits per element per chunk
  std::unique_ptr<cudf::column> packed;       // UINT8, concatenated bit-packed bytes
  // ``packed`` is always the tight, dense live-bytes buffer for any rep that
  // escapes the encoder: file-read reps are built dense, and fused-encode reps
  // are densified in place (compact_in_place()) before they are stored.

  // ---- OverAllocate→Compact transient scratch ----
  // The fused encode kernel emits ``packed`` in slot-strided OverAllocate
  // layout; the OverAllocate ctor records the per-chunk slot stride
  // (``stride_words_``, uint32 words = CHUNK*elem_size/4) and the precomputed
  // total live byte count (``live_packed_bytes_sparse_``) so compact_in_place()
  // can gather ``packed`` down to the tight Compact layout. They exist only to
  // carry the OverAllocate layout from the encode kernel into that single
  // compaction, and are cleared once it runs; a dense (file-read) rep leaves
  // stride_words_=0.
  std::int32_t stride_words_{0};
  std::int64_t live_packed_bytes_sparse_{0};

  PlanLeafKind kind() const override { return PlanLeafKind::Bitpack; }

  /// Live byte count of the ``packed`` channel (UINT32 words: size x elem width).
  std::int64_t live_packed_bytes() const
  {
    return packed ? static_cast<std::int64_t>(packed->size()) *
                      static_cast<std::int64_t>(cudf::size_of(packed->type()))
                  : 0;
  }

  /// Dense ctor. ``packed_data`` must already be tight/compact (no slot
  /// padding) — the layout every stored rep has.
  bitpack_compressed_representation(cudf::data_type type,
                                    cudf::size_type n_rows,
                                    std::unique_ptr<cudf::column> mins,
                                    std::unique_ptr<cudf::column> counts,
                                    std::unique_ptr<cudf::column> bits_per_chunk,
                                    std::unique_ptr<cudf::column> packed_data)
    : compressed_representation(type, n_rows),
      chunk_min(std::move(mins)),
      chunk_count(std::move(counts)),
      chunk_bits(std::move(bits_per_chunk)),
      packed(std::move(packed_data))
  {
  }

  /// OverAllocate ctor. Takes ownership of the slot-strided ``packed_data``
  /// straight from the fused encode kernel. ``stride_words`` is the per-chunk
  /// slot stride in uint32 words; ``live_packed_bytes`` the precomputed total
  /// live byte count. The caller MUST compact_in_place() this rep before it is
  /// stored or enumerated — that gathers ``packed`` down to the tight Compact
  /// layout (computing per-chunk live_words from chunk_bits × chunk_count).
  bitpack_compressed_representation(cudf::data_type type,
                                    cudf::size_type n_rows,
                                    std::unique_ptr<cudf::column> mins,
                                    std::unique_ptr<cudf::column> counts,
                                    std::unique_ptr<cudf::column> bits_per_chunk,
                                    std::unique_ptr<cudf::column> packed_data_overalloc,
                                    std::int32_t stride_words,
                                    std::int64_t live_packed_bytes)
    : compressed_representation(type, n_rows),
      chunk_min(std::move(mins)),
      chunk_count(std::move(counts)),
      chunk_bits(std::move(bits_per_chunk)),
      packed(std::move(packed_data_overalloc)),
      stride_words_(stride_words),
      live_packed_bytes_sparse_(live_packed_bytes)
  {
  }

  /// Densify ``*this`` in place: replace the OverAllocate ``packed`` with its
  /// tight Compact bytes (scan+gather) and clear the OverAllocate scratch. The
  /// existing meta columns (chunk_min/count/bits) are REUSED — no clone — so
  /// this is the cheap path for the ephemeral OverAllocate rep the fused encode
  /// owns and keeps (eager-compaction right after encode). No-op for a rep that
  /// is already dense (stride_words_==0). Work is enqueued on ``stream``; the
  /// caller must sync before reading the dense bytes from another stream.
  void compact_in_place(
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

  // Bitpack is a codegen-only operator: encode/decode go through the fused
  // codegen backend (Bitpack node), so there is no C++ kernel decode path.
  // The rep still carries the packed buffers for the codegen gather, compact,
  // and file-write paths.
  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view,
                                           rmm::device_async_resource_ref) const override
  {
    throw std::runtime_error(
      "bitpack_compressed_representation: reconstruct via the codegen decode "
      "path, not C++ decompress()");
  }
  std::vector<compressible_output> named_channels(rmm::cuda_stream_view) const override
  {
    std::vector<compressible_output> outs;
    outs.reserve(4);
    if (chunk_min) outs.push_back({"chunk_min", chunk_min->view()});
    if (chunk_count) outs.push_back({"chunk_count", chunk_count->view()});
    if (chunk_bits) outs.push_back({"chunk_bits", chunk_bits->view()});
    // ``packed`` is always the tight Compact buffer (the fused encode densifies
    // via compact_in_place() before storing).
    if (packed) { outs.push_back({"packed", packed->view()}); }
    return outs;
  }
  size_t compressed_size_bytes(rmm::cuda_stream_view) const override
  {
    size_t total = 0;
    if (chunk_min) {
      total += static_cast<size_t>(chunk_min->size()) *
               static_cast<size_t>(cudf::size_of(chunk_min->type()));
    }
    if (chunk_count) {
      total += static_cast<size_t>(chunk_count->size()) *
               static_cast<size_t>(cudf::size_of(chunk_count->type()));
    }
    if (chunk_bits) {
      total += static_cast<size_t>(chunk_bits->size()) *
               static_cast<size_t>(cudf::size_of(chunk_bits->type()));
    }
    total += static_cast<size_t>(live_packed_bytes());
    return total;
  }
  std::unique_ptr<cudf::column> release_output(std::string const& name) override
  {
    if (name == "chunk_min" && chunk_min) return std::move(chunk_min);
    if (name == "chunk_count" && chunk_count) return std::move(chunk_count);
    if (name == "chunk_bits" && chunk_bits) return std::move(chunk_bits);
    if (name == "packed" && packed) {
      // Releasing ``packed`` happens when an entropy tail (e.g. ``…packed ->
      // ans``) took over the bytes. The rep is already dense here; clear the
      // OverAllocate scratch defensively so the degraded rep reports zero live
      // packed bytes.
      stride_words_             = 0;
      live_packed_bytes_sparse_ = 0;
      return std::move(packed);
    }
    return nullptr;
  }
};

// -----------------------------------------------------------------------------
// nvcomp ANS / Bitcomp compressors
// -----------------------------------------------------------------------------

struct ans_compressed_representation : compressed_representation {
  static std::unique_ptr<compressed_representation> from_outputs(
    std::vector<std::string> const& output_names,
    std::vector<std::unique_ptr<cudf::column>> outputs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::string* error_out,
    leaf_meta_v const& meta = leaf_meta::none{});

  size_t uncompressed_size;
  std::unique_ptr<rmm::device_buffer> compressed_data;  // worst-case sized
  size_t compressed_size;                               // actual bytes used
  // STRING path (original_type == STRING): compressed_data holds the offsets
  // codec stream (first `offsets_compressed_size` bytes) followed by the chars
  // stream; `uncompressed_size` is the chars byte count. Zero for fixed-width.
  size_t offsets_compressed_size   = 0;
  size_t offsets_uncompressed_size = 0;
  cudf::data_type offsets_type{cudf::type_id::INT32};
  // Lazily built UINT8 column of exactly compressed_size bytes (trimmed
  // payload, no header), exposed via named_channels() for downstream
  // chaining (e.g. `ans.output -> bitcomp`).
  mutable std::unique_ptr<cudf::column> serialized_output;

  ans_compressed_representation(cudf::data_type t,
                                cudf::size_type n,
                                std::unique_ptr<rmm::device_buffer> data,
                                size_t comp_size,
                                size_t uncomp_size)
    : compressed_representation(t, n),
      uncompressed_size(uncomp_size),
      compressed_data(std::move(data)),
      compressed_size(comp_size)
  {
  }

  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override;

  std::vector<compressible_output> named_channels(rmm::cuda_stream_view stream) const override;

  // compressed_size is known at construction — no need to build the lazy column.
  size_t compressed_size_bytes(rmm::cuda_stream_view) const override { return compressed_size; }

  std::unique_ptr<cudf::column> release_output(std::string const& name) override
  {
    if (name == "output" && serialized_output) return std::move(serialized_output);
    return nullptr;
  }
  PlanLeafKind kind() const override { return PlanLeafKind::Ans; }
  leaf_meta_v describe_meta() const override
  {
    leaf_meta::ans a{uncompressed_size, static_cast<std::int32_t>(original_type.id())};
    if (original_type.id() == cudf::type_id::STRING) {
      a.strings = {offsets_compressed_size,
                   offsets_uncompressed_size,
                   static_cast<std::int32_t>(offsets_type.id()),
                   num_rows};
    }
    return a;
  }
};

struct ans_compressor : compressor {
  std::unique_ptr<compressed_representation> compress(cudf::column_view column_to_compress,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) override;
  std::unique_ptr<cudf::column> decompress(compressed_representation const& data_to_decompress,
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

struct bitcomp_compressed_representation : compressed_representation {
  static std::unique_ptr<compressed_representation> from_outputs(
    std::vector<std::string> const& output_names,
    std::vector<std::unique_ptr<cudf::column>> outputs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::string* error_out,
    leaf_meta_v const& meta = leaf_meta::none{});

  size_t uncompressed_size;
  std::unique_ptr<rmm::device_buffer> compressed_data;  // worst-case sized
  size_t compressed_size;                               // actual bytes used
  // Algorithm used at compress time so decompress hits the same
  // Manager cache slot.
  int compress_algorithm;
  // STRING path (original_type == STRING): compressed_data holds the offsets
  // codec stream (first `offsets_compressed_size` bytes) then the chars stream;
  // `uncompressed_size` is the chars byte count. Zero for fixed-width.
  size_t offsets_compressed_size   = 0;
  size_t offsets_uncompressed_size = 0;
  cudf::data_type offsets_type{cudf::type_id::INT32};
  // Lazily built UINT8 column of exactly compressed_size bytes (trimmed
  // payload, no header), exposed via named_channels() for downstream
  // chaining (e.g. `bitcomp.output -> ans`).
  mutable std::unique_ptr<cudf::column> serialized_output;

  bitcomp_compressed_representation(cudf::data_type t,
                                    cudf::size_type n,
                                    std::unique_ptr<rmm::device_buffer> data,
                                    size_t comp_size,
                                    size_t uncomp_size,
                                    int algorithm = 0)
    : compressed_representation(t, n),
      uncompressed_size(uncomp_size),
      compressed_data(std::move(data)),
      compressed_size(comp_size),
      compress_algorithm(algorithm)
  {
  }

  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override;

  std::vector<compressible_output> named_channels(rmm::cuda_stream_view stream) const override;

  // compressed_size is known at construction — no need to build the lazy column.
  size_t compressed_size_bytes(rmm::cuda_stream_view) const override { return compressed_size; }

  std::unique_ptr<cudf::column> release_output(std::string const& name) override
  {
    if (name == "output" && serialized_output) return std::move(serialized_output);
    return nullptr;
  }
  PlanLeafKind kind() const override { return PlanLeafKind::Bitcomp; }
  leaf_meta_v describe_meta() const override
  {
    leaf_meta::bitcomp b{
      uncompressed_size, static_cast<std::int32_t>(original_type.id()), compress_algorithm};
    if (original_type.id() == cudf::type_id::STRING) {
      b.strings = {offsets_compressed_size,
                   offsets_uncompressed_size,
                   static_cast<std::int32_t>(offsets_type.id()),
                   num_rows};
    }
    return b;
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
  std::unique_ptr<cudf::column> decompress(compressed_representation const& data_to_decompress,
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

struct cascaded_compressed_representation : compressed_representation {
  static std::unique_ptr<compressed_representation> from_outputs(
    std::vector<std::string> const& output_names,
    std::vector<std::unique_ptr<cudf::column>> outputs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::string* error_out,
    leaf_meta_v const& meta = leaf_meta::none{});

  size_t uncompressed_size;
  std::unique_ptr<rmm::device_buffer> compressed_data;
  size_t compressed_size;
  // Opts used at compress time — stashed so decompress hits the same Manager
  // cache key.
  int compress_num_deltas;
  int compress_num_RLEs;
  int compress_use_bp;
  // Lazily built UINT8 column of exactly compressed_size bytes.
  mutable std::unique_ptr<cudf::column> serialized_output;

  cascaded_compressed_representation(cudf::data_type t,
                                     cudf::size_type n,
                                     std::unique_ptr<rmm::device_buffer> data,
                                     size_t comp_size,
                                     size_t uncomp_size,
                                     int num_deltas,
                                     int num_RLEs,
                                     int use_bp)
    : compressed_representation(t, n),
      uncompressed_size(uncomp_size),
      compressed_data(std::move(data)),
      compressed_size(comp_size),
      compress_num_deltas(num_deltas),
      compress_num_RLEs(num_RLEs),
      compress_use_bp(use_bp)
  {
  }

  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override;

  std::vector<compressible_output> named_channels(rmm::cuda_stream_view stream) const override;

  size_t compressed_size_bytes(rmm::cuda_stream_view) const override { return compressed_size; }

  std::unique_ptr<cudf::column> release_output(std::string const& name) override
  {
    if (name == "output" && serialized_output) return std::move(serialized_output);
    return nullptr;
  }
  PlanLeafKind kind() const override { return PlanLeafKind::NvcompCascaded; }
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
  std::unique_ptr<cudf::column> decompress(compressed_representation const& data_to_decompress,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) override;

 private:
  int num_deltas_;
  int num_RLEs_;
  int use_bp_;
};

// -----------------------------------------------------------------------------
// Simple nvcomp codecs: Snappy, LZ4, GDeflate ("deflate" in the DSL)
//
// All three share the same rep layout (compressed byte payload + sizes) and
// the same metadata structure.  The template base holds all common fields and
// methods; each concrete struct supplies:
//  - named_channels() / decompress()  — defined in the matching .cu file
//  - from_outputs()                   — defined in representation_factory.cpp
// -----------------------------------------------------------------------------

template <PlanLeafKind K, typename MetaT>
struct nvcomp_simple_rep_base : compressed_representation {
  size_t uncompressed_size = 0;
  std::unique_ptr<rmm::device_buffer> compressed_data;
  size_t compressed_size = 0;
  mutable std::unique_ptr<cudf::column> serialized_output;

  nvcomp_simple_rep_base(cudf::data_type t,
                         cudf::size_type n,
                         std::unique_ptr<rmm::device_buffer> data,
                         size_t comp_sz,
                         size_t uncomp_sz)
    : compressed_representation(t, n),
      uncompressed_size(uncomp_sz),
      compressed_data(std::move(data)),
      compressed_size(comp_sz)
  {
  }

  size_t compressed_size_bytes(rmm::cuda_stream_view) const override { return compressed_size; }

  std::unique_ptr<cudf::column> release_output(std::string const& name) override
  {
    if (name == "output" && serialized_output) return std::move(serialized_output);
    return nullptr;
  }
  PlanLeafKind kind() const override { return K; }
  leaf_meta_v describe_meta() const override
  {
    return MetaT{uncompressed_size, static_cast<std::int32_t>(original_type.id())};
  }

  // named_channels() is identical for all simple nvcomp reps: lazy D2D copy
  // of the compressed payload into a UINT8 column.
  std::vector<compressible_output> named_channels(rmm::cuda_stream_view stream) const override
  {
    if (!serialized_output) {
      // The resulting column's device_buffer remembers ``stream`` for its own
      // eventual deallocation, so the caller-supplied stream must stay valid
      // for the column's lifetime (a private stream destroyed at the end of
      // this scope would leave a dangling handle).
      auto mr  = rmm::mr::get_current_device_resource_ref();
      auto out = cudf::make_fixed_width_column(cudf::data_type(cudf::type_id::UINT8),
                                               static_cast<cudf::size_type>(compressed_size),
                                               cudf::mask_state::UNALLOCATED,
                                               stream,
                                               mr);
      if (compressed_size > 0 && compressed_data) {
        cudaMemcpyAsync(out->mutable_view().head<uint8_t>(),
                        compressed_data->data(),
                        compressed_size,
                        cudaMemcpyDeviceToDevice,
                        stream.value());
      }
      cudaStreamSynchronize(stream.value());
      serialized_output = std::move(out);
    }
    return {{"output", serialized_output->view()}};
  }

  // decompress() is manager-specific — defined in the matching .cu file.
  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override = 0;
};

struct snappy_compressed_representation
  : nvcomp_simple_rep_base<PlanLeafKind::Snappy, leaf_meta::snappy> {
  using nvcomp_simple_rep_base::nvcomp_simple_rep_base;
  static std::unique_ptr<compressed_representation> from_outputs(
    std::vector<std::string> const& output_names,
    std::vector<std::unique_ptr<cudf::column>> outputs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::string* error_out,
    leaf_meta_v const& meta = leaf_meta::none{});
  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override;
};

struct lz4_compressed_representation : nvcomp_simple_rep_base<PlanLeafKind::Lz4, leaf_meta::lz4> {
  using nvcomp_simple_rep_base::nvcomp_simple_rep_base;
  static std::unique_ptr<compressed_representation> from_outputs(
    std::vector<std::string> const& output_names,
    std::vector<std::unique_ptr<cudf::column>> outputs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::string* error_out,
    leaf_meta_v const& meta = leaf_meta::none{});
  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override;
};

struct deflate_compressed_representation
  : nvcomp_simple_rep_base<PlanLeafKind::Deflate, leaf_meta::deflate> {
  using nvcomp_simple_rep_base::nvcomp_simple_rep_base;
  static std::unique_ptr<compressed_representation> from_outputs(
    std::vector<std::string> const& output_names,
    std::vector<std::unique_ptr<cudf::column>> outputs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::string* error_out,
    leaf_meta_v const& meta = leaf_meta::none{});
  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override;
};

struct snappy_compressor : compressor {
  std::unique_ptr<compressed_representation> compress(cudf::column_view column_to_compress,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) override;
  std::unique_ptr<cudf::column> decompress(compressed_representation const& data_to_decompress,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) override;
};

struct lz4_compressor : compressor {
  std::unique_ptr<compressed_representation> compress(cudf::column_view column_to_compress,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) override;
  std::unique_ptr<cudf::column> decompress(compressed_representation const& data_to_decompress,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) override;
};

struct deflate_compressor : compressor {
  std::unique_ptr<compressed_representation> compress(cudf::column_view column_to_compress,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) override;
  std::unique_ptr<cudf::column> decompress(compressed_representation const& data_to_decompress,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) override;
};

// -----------------------------------------------------------------------------
// ALP (Adaptive Lossless floating-Point), FLOAT32, multi-output.
// See SIGMOD '24 (Afroozeh) + G-ALP DaMoN '25. Outputs:
//   integers             INT32  — n_rows (0 at exception slots)
//   exceptions           FLOAT32 — values that failed lossless encode
//   exception_positions  INT32  — sorted row indices of exceptions
//   metadata             UINT16 — one per 1024-vector: (e<<8) | f
struct alp_compressed_representation : compressed_representation {
  static std::unique_ptr<compressed_representation> from_outputs(
    std::vector<std::string> const& output_names,
    std::vector<std::unique_ptr<cudf::column>> outputs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::string* error_out);

  cudf::size_type num_vectors;                        // ceil(num_rows / 1024)
  std::unique_ptr<cudf::column> integers;             // INT32 (f32) / INT64 (f64)
  std::unique_ptr<cudf::column> exceptions;           // FLOAT32 / FLOAT64
  std::unique_ptr<cudf::column> exception_positions;  // INT32
  std::unique_ptr<cudf::column> metadata;             // UINT16

  alp_compressed_representation(cudf::data_type type,
                                cudf::size_type n_rows,
                                cudf::size_type n_vectors,
                                std::unique_ptr<cudf::column> integers_in,
                                std::unique_ptr<cudf::column> exceptions_in,
                                std::unique_ptr<cudf::column> exception_positions_in,
                                std::unique_ptr<cudf::column> metadata_in);

  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override;

  std::vector<compressible_output> named_channels(rmm::cuda_stream_view) const override
  {
    return {
      {"integers", integers->view()},
      {"exceptions", exceptions->view()},
      {"exception_positions", exception_positions->view()},
      {"metadata", metadata->view()},
    };
  }

  std::unique_ptr<cudf::column> release_output(std::string const& name) override
  {
    if (name == "integers" && integers) return std::move(integers);
    if (name == "exceptions" && exceptions) return std::move(exceptions);
    if (name == "exception_positions" && exception_positions) return std::move(exception_positions);
    if (name == "metadata" && metadata) return std::move(metadata);
    return nullptr;
  }
  PlanLeafKind kind() const override { return PlanLeafKind::Alp; }
};

struct alp_compressor : compressor {
  std::unique_ptr<compressed_representation> compress(cudf::column_view column_to_compress,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) override;
  std::unique_ptr<cudf::column> decompress(compressed_representation const& data_to_decompress,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) override;
};

// ALP-RD (Right-Dictionary), FLOAT32, multi-output. Column-wide K=8 dict +
// right_bw; deviates from cwida CPU reference (per-rowgroup dict) to keep
// right_parts fixed-width for downstream bitpack. Outputs:
//   right_parts          UINT32 — low right_bw bits of each value
//   dict_indices         UINT8  — 0..7 dict slot, 8 = exception marker
//   dict                 UINT16 — 8 entries (column-wide)
//   metadata             UINT8  — 1 entry: right_bw
//   exceptions           UINT16 — rejected left parts
//   exception_positions  INT32  — sorted row indices of exceptions
struct alp_rd_compressed_representation : compressed_representation {
  static std::unique_ptr<compressed_representation> from_outputs(
    std::vector<std::string> const& output_names,
    std::vector<std::unique_ptr<cudf::column>> outputs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::string* error_out);

  uint8_t right_bw;                                   // bits in right part (1..31 f32, 1..63 f64)
  std::unique_ptr<cudf::column> right_parts;          // UINT32 (f32) / UINT64 (f64)
  std::unique_ptr<cudf::column> dict_indices;         // UINT8
  std::unique_ptr<cudf::column> dict;                 // UINT16, 8 entries (both precisions)
  std::unique_ptr<cudf::column> metadata;             // UINT8, 1 entry
  std::unique_ptr<cudf::column> exceptions;           // UINT16
  std::unique_ptr<cudf::column> exception_positions;  // INT32

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

  std::vector<compressible_output> named_channels(rmm::cuda_stream_view) const override
  {
    return {
      {"right_parts", right_parts->view()},
      {"dict_indices", dict_indices->view()},
      {"dict", dict->view()},
      {"metadata", metadata->view()},
      {"exceptions", exceptions->view()},
      {"exception_positions", exception_positions->view()},
    };
  }

  std::unique_ptr<cudf::column> release_output(std::string const& name) override
  {
    if (name == "right_parts" && right_parts) return std::move(right_parts);
    if (name == "dict_indices" && dict_indices) return std::move(dict_indices);
    if (name == "dict" && dict) return std::move(dict);
    if (name == "metadata" && metadata) return std::move(metadata);
    if (name == "exceptions" && exceptions) return std::move(exceptions);
    if (name == "exception_positions" && exception_positions) return std::move(exception_positions);
    return nullptr;
  }
  PlanLeafKind kind() const override { return PlanLeafKind::AlpRd; }
  leaf_meta_v describe_meta() const override { return leaf_meta::alp_rd{right_bw}; }
};

struct alp_rd_compressor : compressor {
  std::unique_ptr<compressed_representation> compress(cudf::column_view column_to_compress,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) override;
  std::unique_ptr<cudf::column> decompress(compressed_representation const& data_to_decompress,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) override;
};

// ── bitextract / bitjoin ──────────────────────────────────────────────────────
//
// Bit-field layout convention (shared by bitextract and bitjoin):
//
//   Fields are packed MSB-first in the order they appear, both in the DSL
//   line and in a spec like `bitextract_1sign_8exponent_23mantissa`. The
//   first field occupies the top bits of the packed value; the next field
//   the bits immediately below, and so on. If the listed fields cover fewer
//   bits than the packed type's width, the unused high-end bits above the
//   first field are zero (bitjoin) or ignored (bitextract).
//
//   Concretely, for a packed type of width `W` and fields with widths
//   `b_0, b_1, ..., b_{n-1}` listed in DSL order, field `i` occupies bits
//   `[W - sum(b_0..b_i) , W - sum(b_0..b_i) + b_i - 1]` of the packed value.
//
//   Example: `input_3:0, input_7:4 -> bitjoin_u8 -> swapped` packs input
//   bits [3:0] into output bits [7:4] and input bits [7:4] into [3:0].
//
// Packed-type token (where the type lives in each operator's name):
//
//   - bitjoin's packed type is the OUTPUT and is required at the END of
//     the spec:           `bitjoin_<fields>_<type>` or `bitjoin_<type>`
//   - bitextract's packed type is the INPUT and is optional at the FRONT:
//                         `bitextract_[<type>_]<fields>` or alias `bitextract_f{16,32,64}`
//
//   The recognised type tokens are `u8`/`u16`/`u32`/`u64`, `i8`/`i16`/`i32`/`i64`,
//   `f32`/`f64` (with `uint8`/`int8`/... long forms accepted by bitjoin).
//   For bitextract the type token is unambiguous because field tokens always
//   start with a digit (`<bits><name>`).
//
//   When the bitextract type prefix is absent, the spec reconstructs as the
//   smallest unsigned int that fits the sum of field widths. This is enough
//   for symmetric round-trips on unsigned inputs, but loses signedness/width
//   information for signed inputs. Compression therefore auto-injects the
//   actual column type into the stored DSL (see `compress_with_plan`), so
//   `.hpln` files always carry an explicit type prefix and round-trips are
//   exact without any side-channel.

struct bitfield_spec {
  uint32_t bits;
  std::string name;
};

/// Result of parsing a bitextract or bitjoin spec suffix.
struct bitextract_spec_result {
  std::vector<bitfield_spec> fields;
  cudf::data_type output_type{cudf::type_id::EMPTY};
};

/// Compressor-name prefix for the bitextract family ("bitextract_<spec>").
inline constexpr std::string_view kBitextractPrefix = "bitextract_";

/// If `name` starts with `kBitextractPrefix`, return the spec suffix; otherwise nullopt.
inline std::optional<std::string_view> strip_bitextract_prefix(std::string_view name)
{
  if (name.size() > kBitextractPrefix.size() &&
      name.compare(0, kBitextractPrefix.size(), kBitextractPrefix) == 0) {
    return name.substr(kBitextractPrefix.size());
  }
  return std::nullopt;
}

/// IEEE-754 float aliases shared by parse_bitextract_spec and parse_bitjoin_spec.
inline std::optional<bitextract_spec_result> parse_float_alias(std::string_view suffix)
{
  if (suffix == "f16")
    return bitextract_spec_result{{{1, "sign"}, {5, "exponent"}, {10, "mantissa"}},
                                  cudf::data_type(cudf::type_id::UINT16)};
  if (suffix == "f32")
    return bitextract_spec_result{{{1, "sign"}, {8, "exponent"}, {23, "mantissa"}},
                                  cudf::data_type(cudf::type_id::FLOAT32)};
  if (suffix == "f64")
    return bitextract_spec_result{{{1, "sign"}, {11, "exponent"}, {52, "mantissa"}},
                                  cudf::data_type(cudf::type_id::FLOAT64)};
  return std::nullopt;
}

/// Map a packed-type token (`u8`/`i16`/`f32`/...) to a cudf::data_type.
/// Returns EMPTY if the token is unrecognised. Long forms (`uint8`) are also accepted.
inline cudf::data_type parse_packed_type_token(std::string_view tok)
{
  if (tok == "u8" || tok == "uint8") return cudf::data_type(cudf::type_id::UINT8);
  if (tok == "u16" || tok == "uint16") return cudf::data_type(cudf::type_id::UINT16);
  if (tok == "u32" || tok == "uint32") return cudf::data_type(cudf::type_id::UINT32);
  if (tok == "u64" || tok == "uint64") return cudf::data_type(cudf::type_id::UINT64);
  if (tok == "i8" || tok == "int8") return cudf::data_type(cudf::type_id::INT8);
  if (tok == "i16" || tok == "int16") return cudf::data_type(cudf::type_id::INT16);
  if (tok == "i32" || tok == "int32") return cudf::data_type(cudf::type_id::INT32);
  if (tok == "i64" || tok == "int64") return cudf::data_type(cudf::type_id::INT64);
  if (tok == "f32") return cudf::data_type(cudf::type_id::FLOAT32);
  if (tok == "f64") return cudf::data_type(cudf::type_id::FLOAT64);
  return cudf::data_type(cudf::type_id::EMPTY);
}

/// Canonical "namespace" form of a compressor name, with any optional
/// packed-type prefix stripped from the bitextract spec. Float aliases are
/// preserved because they are part of the operator's identity.
///
/// This is what `parse_plan_dsl` uses to derive output paths, so a bitextract
/// step's downstream paths are stable across DSL forms — i.e. both
/// `bitextract_3hi_5lo` and `bitextract_i16_3hi_5lo` share the namespace
/// `bitextract_3hi_5lo.<field>`. That decouples the auto-injected type prefix
/// (stored in `step.compressor` for type reconstruction) from the path layout
/// (used by the leaves map).
inline std::string bitextract_canonical_name(std::string const& compressor)
{
  auto suffix = strip_bitextract_prefix(compressor);
  if (!suffix) return compressor;
  if (*suffix == "f32" || *suffix == "f64") return compressor;
  if (suffix->empty() || std::isdigit(static_cast<unsigned char>((*suffix)[0]))) {
    return compressor;  // generic form, no type prefix to strip
  }
  size_t us = suffix->find('_');
  if (us == std::string_view::npos) return compressor;
  if (parse_packed_type_token(suffix->substr(0, us)).id() == cudf::type_id::EMPTY) {
    return compressor;  // first segment isn't a type token, leave alone
  }
  return std::string("bitextract_") + std::string(suffix->substr(us + 1));
}

/// Canonicalize a path token by applying `bitextract_canonical_name` to the
/// leftmost dot-separated segment. Lets users reference bitextract outputs
/// downstream in either typed or untyped form interchangeably.
inline std::string canonicalize_path(std::string const& path)
{
  size_t dot            = path.find('.');
  std::string head      = (dot == std::string::npos) ? path : path.substr(0, dot);
  std::string canonical = bitextract_canonical_name(head);
  if (canonical == head) return path;
  return (dot == std::string::npos) ? canonical : canonical + path.substr(dot);
}

/// Parse the field-list portion of a bitextract spec ("3hi_5lo", "1sign_8exponent_23mantissa").
/// Each token must be `<digits><name>`. Returns empty fields on malformed input.
inline std::vector<bitfield_spec> parse_bitfield_list(std::string_view fields_suffix)
{
  std::vector<bitfield_spec> fields;
  size_t pos = 0;
  while (pos <= fields_suffix.size()) {
    size_t next          = fields_suffix.find('_', pos);
    std::string_view tok = (next == std::string_view::npos) ? fields_suffix.substr(pos)
                                                            : fields_suffix.substr(pos, next - pos);
    size_t d             = 0;
    while (d < tok.size() && std::isdigit(static_cast<unsigned char>(tok[d])))
      ++d;
    if (d == 0 || d == tok.size()) return {};
    uint32_t bits = 0;
    for (size_t k = 0; k < d; ++k)
      bits = bits * 10 + static_cast<uint32_t>(tok[k] - '0');
    if (bits == 0) return {};
    fields.push_back({bits, std::string(tok.substr(d))});
    if (next == std::string_view::npos) break;
    pos = next + 1;
  }
  return fields;
}

/// Parse a bitextract spec. Accepted forms:
///   - alias              "f32" / "f64"
///   - generic            "3hi_5lo"                 (output type defaults to smallest uintN that
///   fits)
///   - typed              "i16_3hi_5lo" / "u32_4a_4b_24c"
/// On failure returns a result with empty fields.
///
/// "f16" is deliberately not an accepted alias here (unlike parse_bitjoin_spec):
/// cudf has no native FLOAT16 storage, so there is no genuine 16-bit-wide
/// column to split — bitextract needs its input to actually be the width the
/// spec assumes, which only holds for f32/f64.
inline bitextract_spec_result parse_bitextract_spec(std::string_view suffix)
{
  if (suffix != "f16") {
    if (auto alias = parse_float_alias(suffix)) return *alias;
  }

  // Optional leading packed-type token: unambiguous because field tokens
  // always start with a digit (`<bits><name>`).
  cudf::data_type explicit_type(cudf::type_id::EMPTY);
  std::string_view fields_suffix = suffix;
  if (!suffix.empty() && !std::isdigit(static_cast<unsigned char>(suffix[0]))) {
    size_t us = suffix.find('_');
    if (us == std::string_view::npos) return {};
    explicit_type = parse_packed_type_token(suffix.substr(0, us));
    if (explicit_type.id() == cudf::type_id::EMPTY) return {};
    fields_suffix = suffix.substr(us + 1);
  }

  auto fields = parse_bitfield_list(fields_suffix);
  if (fields.empty()) return {};

  cudf::data_type out_type = explicit_type;
  if (out_type.id() == cudf::type_id::EMPTY) {
    uint32_t total_bits = 0;
    for (auto const& f : fields)
      total_bits += f.bits;
    out_type = (total_bits <= 8)    ? cudf::data_type(cudf::type_id::UINT8)
               : (total_bits <= 16) ? cudf::data_type(cudf::type_id::UINT16)
               : (total_bits <= 32) ? cudf::data_type(cudf::type_id::UINT32)
                                    : cudf::data_type(cudf::type_id::UINT64);
  }
  return {std::move(fields), out_type};
}

/// Parse a bitjoin spec. Accepted forms:
///   - alias              "f16" / "f32" / "f64"
///   - bare type          "u32"                       (fields provided via DSL bit-range tokens)
///   - typed fields       "1sign_8exponent_23mantissa_f32" / "3hi_5lo_u8"
/// The packed type is the LAST underscore-separated token. On failure returns empty fields.
inline bitextract_spec_result parse_bitjoin_spec(std::string_view suffix)
{
  if (auto alias = parse_float_alias(suffix)) return *alias;

  size_t last               = suffix.rfind('_');
  std::string_view type_tok = (last == std::string_view::npos) ? suffix : suffix.substr(last + 1);
  std::string_view fields_tok =
    (last == std::string_view::npos) ? std::string_view{} : suffix.substr(0, last);

  cudf::data_type out_type = parse_packed_type_token(type_tok);
  if (out_type.id() == cudf::type_id::EMPTY) return {};

  // Bare output type (e.g. "bitjoin_u32"): caller provides bit ranges in the DSL input list.
  if (fields_tok.empty()) return {{}, out_type};

  auto fields = parse_bitfield_list(fields_tok);
  if (fields.empty()) return {};
  return {std::move(fields), out_type};
}

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

struct bitextract_compressed_representation : compressed_representation {
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

  std::unique_ptr<cudf::column> release_output(std::string const& name) override
  {
    for (size_t i = 0; i < spec.fields.size(); ++i) {
      if (spec.fields[i].name == name && fields[i]) return std::move(fields[i]);
    }
    return nullptr;
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

  std::unique_ptr<cudf::column> decompress(compressed_representation const& repr,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) override
  {
    return repr.decompress(stream, mr);
  }
};

// -----------------------------------------------------------------------------
// Codegen-fused representations — produced only by the JIT encoder.
// decompress() is unsupported; reconstruction goes through the decode kernels.
// -----------------------------------------------------------------------------

// Holds a codegen-fused subtree node (Delta, Rle, ...). Carries the node's
// device buffers tagged by field name; the tree structure, op kind, and
// tail-routing are recovered from the plan DSL at decode time. ``kind_tag``
// ("DeltaFused"/"RleFused"/...) maps to the op kind. decompress() throws.
struct codegen_fused_representation : compressed_representation {
  std::string kind_tag;  // DSL op name for fusable ops ("delta", "rle"), or "RawFused"
  // Buffers in manifest order, each tagged with its manifest field name.
  std::vector<std::pair<std::string, std::unique_ptr<cudf::column>>> buffers;

  codegen_fused_representation(std::string k, cudf::data_type type, cudf::size_type n_rows)
    : compressed_representation(type, n_rows), kind_tag(std::move(k))
  {
  }

  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view,
                                           rmm::device_async_resource_ref) const override
  {
    throw std::runtime_error(
      "codegen_fused_representation: reconstruct via the codegen "
      "decode path, not C++ decompress()");
  }

  // File writer / codegen gather: expose all manifest buffers (all dense).
  std::vector<compressible_output> named_channels(rmm::cuda_stream_view) const override
  {
    std::vector<compressible_output> out;
    out.reserve(buffers.size());
    for (auto const& [name, col] : buffers) {
      if (col) out.push_back({name, col->view()});
    }
    return out;
  }

  PlanLeafKind kind() const override
  {
    if (kind_tag == "delta") return PlanLeafKind::Delta;
    if (kind_tag == "rle") return PlanLeafKind::Rle;
    if (kind_tag == "bitpack") return PlanLeafKind::Bitpack;
    if (kind_tag == "for") return PlanLeafKind::For;
    if (kind_tag == "zigzag") return PlanLeafKind::Zigzag;
    if (kind_tag == "RawFused") return PlanLeafKind::Identity;
    return PlanLeafKind::Unknown;
  }
};

/// Resolve a DSL compressor name to a compressor instance, or nullptr if
/// the name is unknown. The fused ops (delta / rle / bitpack) are not here —
/// they go through the JIT codegen encoder, not a compressor factory.
///
/// Recognised names (incl. parameterised suffix forms):
///   identity, dictionary, for, alp, alp_rd, ans,
///   bitcomp[_default|_sparse], bitextract_<spec>.
std::unique_ptr<compressor> make_compressor(std::string const& name);

}  // namespace simpatico

#endif
