// SPDX-License-Identifier: Apache-2.0
//
// representation_factory.cpp — reconstruction of a compressed_representation from a compressor
// name and its named output columns. One throwing factory validates the channels and adopts them
// into the matching representation; the loader (reconstruct_representation) and the decoder
// (reconstruct_decode_representation) differ only in how a dictionary publishes its key width and
// in how they report errors.

#include "codegen/plan/operator_registry.hpp"
#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/plan/plan_tree.hpp"
#include "codegen/plan/representation.hpp"
#include "decode/decode_session.hpp"
#include "util/host_observation.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/null_mask.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace simpatico {

namespace {

using channel_columns = std::vector<std::unique_ptr<cudf::column>>;

[[noreturn]] void reject(std::string const& message) { throw std::invalid_argument(message); }

// Require exactly the `expected` channel names, in order.
void require_channels(std::string const& op,
                      std::vector<std::string> const& names,
                      std::initializer_list<char const*> expected)
{
  if (names.size() != expected.size() ||
      !std::equal(names.begin(), names.end(), expected.begin())) {
    std::string listed;
    for (auto const* name : expected)
      listed += (listed.empty() ? "" : ", ") + std::string{name};
    reject(op + " outputs must be named '" + listed + "'");
  }
}

// The single UINT8 `output` payload of an nvCOMP codec, with the uncompressed type, row count, and
// byte count from its leaf metadata, passed to `make` together with the adopted payload buffer.
template <typename Meta, typename Make>
std::unique_ptr<compressed_representation> make_payload(std::string const& codec,
                                                        std::vector<std::string> const& names,
                                                        channel_columns& channels,
                                                        leaf_meta_v const& metadata,
                                                        Make make)
{
  require_channels(codec, names, {"output"});
  if (channels[0]->type().id() != cudf::type_id::UINT8) reject(codec + " 'output' must be UINT8");
  auto const* meta = std::get_if<Meta>(&metadata);
  if (!meta) reject(codec + ": missing leaf metadata (uncompressed_size, original_type_id)");
  auto const type       = cudf::data_type{static_cast<cudf::type_id>(meta->original_type_id)};
  auto const bytes      = meta->uncompressed_size;
  auto const rows       = bytes > 0 && cudf::is_fixed_width(type)
                            ? static_cast<cudf::size_type>(bytes / cudf::size_of(type))
                            : 0;
  auto const compressed = static_cast<std::size_t>(channels[0]->size());
  auto payload = std::make_unique<rmm::device_buffer>(std::move(*channels[0]->release().data));
  return make(*meta, type, rows, std::move(payload), compressed, bytes);
}

template <typename Rep, typename Meta>
std::unique_ptr<compressed_representation> make_simple_payload(
  std::string const& codec,
  std::vector<std::string> const& names,
  channel_columns& channels,
  leaf_meta_v const& metadata)
{
  return make_payload<Meta>(codec, names, channels, metadata, [](Meta const&, auto&&... args) {
    return std::make_unique<Rep>(std::forward<decltype(args)>(args)...);
  });
}

// Assemble the dictionary by adopting its channels. An optional trailing `null_mask`
// channel carries the validity of a nullable column. cuDF's factories retag unsigned indices (for
// example a narrow field a bitjoin decodes) as the signed type of the same width.
std::unique_ptr<compressed_representation> make_dictionary(std::vector<std::string> const& names,
                                                           channel_columns& channels,
                                                           std::optional<std::int64_t> key_width,
                                                           ::cuda::stream_ref stream,
                                                           rmm::device_async_resource_ref mr)
{
  bool const has_mask = !names.empty() && names.back() == "null_mask";
  if (has_mask) {
    require_channels("dictionary", names, {"keys_offsets", "keys_chars", "indices", "null_mask"});
  } else {
    require_channels("dictionary", names, {"keys_offsets", "keys_chars", "indices"});
  }
  auto const offsets_type = channels[0]->type().id();
  if ((offsets_type != cudf::type_id::INT32 && offsets_type != cudf::type_id::INT64) ||
      channels[1]->type().id() != cudf::type_id::UINT8) {
    reject("dictionary: invalid key offsets/chars types");
  }
  if (channels[0]->size() < 1 || channels[0]->null_count() != 0) {
    reject("dictionary: key offsets must be nonempty and null-free");
  }
  auto const keys = channels[0]->size() - 1;
  auto const rows = channels[2]->size();
  // A positive width fixes the chars extent exactly; anything else is corrupt metadata, not a shape
  // to decline, because the gather addresses the chars with it.
  if (key_width &&
      (*key_width < -1 || (*key_width > 0 && static_cast<std::int64_t>(channels[1]->size()) !=
                                               std::int64_t{keys} * *key_width))) {
    reject("dictionary: key width hint does not describe the key channels");
  }
  auto mask                  = cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr);
  cudf::size_type mask_nulls = 0;
  if (has_mask) {
    if (channels[3]->type().id() != cudf::type_id::UINT8 ||
        static_cast<std::size_t>(channels[3]->size()) < cudf::bitmask_allocation_size_bytes(rows)) {
      reject("dictionary: invalid null mask channel");
    }
    // The host null count decides whether the mask belongs on the dictionary parent.
    auto const* bits = static_cast<cudf::bitmask_type const*>(channels[3]->view().head<void>());
    mask_nulls       = rows > 0 ? cudf::null_count(bits, 0, rows, stream) : 0;
    if (mask_nulls > 0) {
      auto contents = channels[3]->release();
#if CUDF_VERSION_MAJOR > 26 || (CUDF_VERSION_MAJOR == 26 && CUDF_VERSION_MINOR >= 12)
      // Keep the RMM source alive until the copy into cuDF's CUDA mask buffer completes.
      contents.data->set_stream(stream);
      mask = cudf::copy_bitmask(bits, 0, rows, stream, mr);
#else
      mask = std::move(*contents.data);
#endif
    }
  }
  auto key_strings =
    cudf::make_strings_column(keys,
                              std::move(channels[0]),
                              std::move(*channels[1]->release().data),
                              0,
                              cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr));
  auto dictionary =
    mask_nulls > 0
      ? cudf::make_dictionary_column(
          std::move(key_strings), std::move(channels[2]), std::move(mask), mask_nulls)
      : cudf::make_dictionary_column(std::move(key_strings), std::move(channels[2]), stream, mr);
  if (!key_width) {
    return dictionary_compressed_representation::from_encoded_column(
      std::move(dictionary), stream, mr);
  }
  auto rep = std::make_unique<dictionary_compressed_representation>(std::move(dictionary));
  rep->constant_key_width = *key_width;
  return rep;
}

// Validate `channels` against the codec named `compressor_name` and adopt them into its
// representation. `key_width` is the dictionary width to publish; when absent it is measured on
// `stream`, which waits for the stream.
std::unique_ptr<compressed_representation> reconstruct(std::string const& compressor_name,
                                                       std::vector<std::string> const& names,
                                                       channel_columns channels,
                                                       leaf_meta_v const& meta,
                                                       std::optional<std::int64_t> key_width,
                                                       ::cuda::stream_ref stream,
                                                       rmm::device_async_resource_ref mr)
{
  if (names.size() != channels.size() ||
      std::any_of(channels.begin(), channels.end(), [](auto const& column) { return !column; })) {
    reject("reconstruction: missing output channel for '" + compressor_name + "'");
  }
  auto const id = op_id_from_name(compressor_name);
  if (!id) reject("unsupported compressor '" + compressor_name + "' for reconstruction");
  switch (*id) {
    case OpId::Identity:
      if (channels.size() != 1) reject("identity expects exactly one output column");
      return std::make_unique<identity_compressed_representation>(std::move(channels[0]));
    case OpId::Dictionary: return make_dictionary(names, channels, key_width, stream, mr);
    case OpId::Alp: {
      require_channels("alp", names, {"integers", "exceptions", "exception_positions", "metadata"});
      auto const integers   = channels[0]->type().id();
      auto const exceptions = channels[1]->type().id();
      if (!((integers == cudf::type_id::INT32 && exceptions == cudf::type_id::FLOAT32) ||
            (integers == cudf::type_id::INT64 && exceptions == cudf::type_id::FLOAT64)) ||
          channels[2]->type().id() != cudf::type_id::INT32 ||
          channels[3]->type().id() != cudf::type_id::UINT16 ||
          channels[1]->size() != channels[2]->size()) {
        reject("alp: invalid channel types or exception sizes");
      }
      // The exceptions keep the original float type by construction.
      return std::make_unique<alp_compressed_representation>(channels[1]->type(),
                                                             channels[0]->size(),
                                                             channels[3]->size(),
                                                             std::move(channels[0]),
                                                             std::move(channels[1]),
                                                             std::move(channels[2]),
                                                             std::move(channels[3]));
    }
    case OpId::AlpRd: {
      require_channels(
        "alp_rd",
        names,
        {"right_parts", "dict_indices", "dict", "metadata", "exceptions", "exception_positions"});
      auto const right = channels[0]->type().id();
      if ((right != cudf::type_id::UINT32 && right != cudf::type_id::UINT64) ||
          channels[1]->type().id() != cudf::type_id::UINT8 ||
          channels[2]->type().id() != cudf::type_id::UINT16 ||
          channels[3]->type().id() != cudf::type_id::UINT8 || channels[3]->size() < 1 ||
          channels[4]->type().id() != cudf::type_id::UINT16 ||
          channels[5]->type().id() != cudf::type_id::INT32 ||
          channels[4]->size() != channels[5]->size() ||
          channels[0]->size() != channels[1]->size()) {
        reject("alp_rd: invalid channel types or sizes");
      }
      // The right-part bit width is the single metadata byte, written on `stream`.
      std::uint8_t right_bw = 0;
      read_device_bytes_completed(
        &right_bw, channels[3]->view().head<void>(), sizeof right_bw, stream);
      auto const type = cudf::data_type{right == cudf::type_id::UINT64 ? cudf::type_id::FLOAT64
                                                                       : cudf::type_id::FLOAT32};
      auto const rows = channels[0]->size();
      return std::make_unique<alp_rd_compressed_representation>(type,
                                                                rows,
                                                                right_bw,
                                                                std::move(channels[0]),
                                                                std::move(channels[1]),
                                                                std::move(channels[2]),
                                                                std::move(channels[3]),
                                                                std::move(channels[4]),
                                                                std::move(channels[5]));
    }
    case OpId::StrSplit: {
      // chars is UINT8, or widened (UINT32/UINT64) to hold more than 2 GB under the 2^31-element
      // column cap (see str_split_compressor).
      bool const has_mask = channels.size() == 3;
      if (has_mask) {
        require_channels("str_split", names, {"offsets", "chars", "null_mask"});
      } else {
        require_channels("str_split", names, {"offsets", "chars"});
      }
      auto const offsets = channels[0]->type().id();
      auto const chars   = channels[1]->type().id();
      if ((offsets != cudf::type_id::INT32 && offsets != cudf::type_id::INT64) ||
          (chars != cudf::type_id::UINT8 && chars != cudf::type_id::UINT32 &&
           chars != cudf::type_id::UINT64) ||
          (has_mask && channels[2]->type().id() != cudf::type_id::UINT8)) {
        reject("str_split: invalid channel types");
      }
      auto const rows = channels[0]->size() > 0 ? channels[0]->size() - 1 : 0;
      return std::make_unique<str_split_compressed_representation>(
        rows,
        std::move(channels[0]),
        std::move(channels[1]),
        has_mask ? std::move(channels[2]) : nullptr);
    }
    case OpId::Bitextract: {
      auto const suffix = strip_bitextract_prefix(compressor_name);
      auto spec         = parse_bitextract_spec(suffix ? *suffix : std::string_view{});
      if (spec.fields.empty() || spec.fields.size() != channels.size()) {
        reject("bitextract: bad spec or field count in '" + compressor_name + "'");
      }
      for (std::size_t i = 0; i < channels.size(); ++i) {
        if (spec.fields[i].name != names[i]) {
          reject("bitextract: output name mismatch at index " + std::to_string(i));
        }
      }
      return std::make_unique<bitextract_compressed_representation>(std::move(spec),
                                                                    std::move(channels));
    }
    case OpId::Ans:
      return make_simple_payload<ans_compressed_representation, leaf_meta::ans>(
        "ans", names, channels, meta);
    case OpId::Snappy:
      return make_simple_payload<snappy_compressed_representation, leaf_meta::snappy>(
        "snappy", names, channels, meta);
    case OpId::Lz4:
      return make_simple_payload<lz4_compressed_representation, leaf_meta::lz4>(
        "lz4", names, channels, meta);
    case OpId::Deflate:
      return make_simple_payload<deflate_compressed_representation, leaf_meta::deflate>(
        "deflate", names, channels, meta);
    case OpId::Bitcomp:
      return make_payload<leaf_meta::bitcomp>(
        "bitcomp", names, channels, meta, [](auto const& bitcomp, auto&&... args) {
          return std::make_unique<bitcomp_compressed_representation>(
            std::forward<decltype(args)>(args)..., bitcomp.algorithm);
        });
    case OpId::NvcompCascaded:
      return make_payload<leaf_meta::nvcomp_cascaded>(
        "nvcomp_cascaded", names, channels, meta, [](auto const& cascaded, auto&&... args) {
          return std::make_unique<cascaded_compressed_representation>(
            std::forward<decltype(args)>(args)...,
            cascaded.num_deltas,
            cascaded.num_RLEs,
            cascaded.use_bp);
        });
    // The fused ops have no standalone reconstruction; the codegen decode path inverts them.
    case OpId::Bitpack:
    case OpId::Delta:
    case OpId::Rle:
    case OpId::For:
    case OpId::Zigzag: break;
  }
  reject("unsupported compressor '" + compressor_name + "' for reconstruction");
}

}  // namespace

std::unique_ptr<compressed_representation> reconstruct_representation(
  std::string const& compressor_name,
  std::vector<std::string> const& output_names,
  std::vector<std::unique_ptr<cudf::column>> outputs,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  std::string* error_out,
  leaf_meta_v const& meta)
{
  try {
    return reconstruct(
      compressor_name, output_names, std::move(outputs), meta, std::nullopt, stream, mr);
  } catch (std::invalid_argument const& error) {
    if (error_out) *error_out = error.what();
    return nullptr;
  }
}

std::unique_ptr<compressed_representation> reconstruct_decode_representation(
  PlanNode const& node,
  std::vector<std::string> const& output_names,
  std::vector<std::unique_ptr<cudf::column>> channels,
  decode_frame& frame)
{
  return reconstruct(node.op,
                     output_names,
                     std::move(channels),
                     node.meta,
                     node.dictionary_key_width_hint,
                     frame.stream(),
                     frame.mr());
}

std::unique_ptr<cudf::column> identity_compressed_representation::decompress(
  decode_frame& frame) const
{
  if (channels_.size() != 1 || !channels_[0]) {
    throw std::invalid_argument("identity decode: missing stored column");
  }
  return std::make_unique<cudf::column>(*channels_[0], frame.stream(), frame.mr());
}

std::unique_ptr<cudf::column> decode_standalone(compressed_representation const& rep,
                                                decode_frame& frame)
{
  auto const* standalone = dynamic_cast<standalone_compressed_representation const*>(&rep);
  if (!standalone) {
    throw std::invalid_argument("decode standalone: representation requires the plan bridge");
  }
  return standalone->decompress(frame);
}

std::unique_ptr<cudf::column> decompress_standalone_representation(
  compressed_representation const* rep,
  ::cuda::stream_ref stream,
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
  return decode_one(column_decode_request{std::cref(*standalone)}, stream, mr);
}

}  // namespace simpatico
