// SPDX-License-Identifier: Apache-2.0
#ifndef CODEGEN_LEAF_DESC_HPP
#define CODEGEN_LEAF_DESC_HPP

#include <cudf/types.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace simpatico {

// Identifies which leaf representation a compressed node carries.
enum class PlanLeafKind : std::uint8_t {
  Delta          = 1,
  Rle            = 2,
  Dictionary     = 3,
  Bitpack        = 4,
  Identity       = 5,
  For            = 6,
  Snappy         = 7,
  Deflate        = 8,
  Lz4            = 9,
  Ans            = 10,
  Bitcomp        = 11,
  NvcompCascaded = 12,
  Alp            = 13,
  AlpRd          = 14,
  Zigzag         = 15,
  Unknown        = 255,
};

// Per-leaf decode metadata reported by compressed_representation::describe_meta.
namespace leaf_meta {

struct none {};

// ALP-RD: right_bw (and the K=8 dict) are COLUMN-WIDE — chosen once and applied
// to every row, keeping right_parts fixed-width with a uniform shift/mask
// decode. The cwida/ALP reference instead fits these per rowgroup (1024 values).
// Per-chunk right_bw would cut exceptions on heterogeneous columns but needs a
// [num_chunks] right_bw array, an 8×num_chunks dict, and a chunk-indexed decode.
struct alp_rd {
  std::uint8_t right_bw = 0;
};

// String-column extras carried by ANS/bitcomp when the compressed column is a
// STRING (original_type_id == cudf::type_id::STRING). The payload then holds two
// concatenated codec streams: the offsets stream (first `offsets_compressed_size`
// bytes) followed by the chars stream (the remainder). `uncompressed_size` is the
// chars byte count; these fields describe the offsets stream and the row count so
// the strings column can be rebuilt without touching the payload bytes.
struct string_extras {
  std::uint64_t offsets_compressed_size   = 0;  // 0 => not a string (fixed-width column)
  std::uint64_t offsets_uncompressed_size = 0;
  std::int32_t offsets_type_id            = 0;  // cast of cudf::type_id (INT32/INT64)
  std::int64_t num_rows                   = 0;  // number of strings
};

// ANS: opaque byte blob — uncompressed_size and original type_id cannot be
// recovered from the compressed payload alone.
struct ans {
  std::uint64_t uncompressed_size = 0;
  std::int32_t original_type_id   = 0;  // cast of cudf::type_id
  string_extras strings{};              // populated iff original_type_id == STRING
};

// Bitcomp: same as ANS, plus the algorithm knob used at compress time.
struct bitcomp {
  std::uint64_t uncompressed_size = 0;
  std::int32_t original_type_id   = 0;
  std::int32_t algorithm          = 0;
  string_extras strings{};  // populated iff original_type_id == STRING
};

// NvcompCascaded: uncompressed_size + type + the three opts knobs (num_deltas,
// num_RLEs, use_bp) used at compress time so decompress can reconstruct the
// same Manager cache key.
struct nvcomp_cascaded {
  std::uint64_t uncompressed_size = 0;
  std::int32_t original_type_id   = 0;
  std::int32_t num_deltas         = 1;
  std::int32_t num_RLEs           = 2;
  std::int32_t use_bp             = 1;
};

// Simple nvcomp codecs (Snappy, LZ4, GDeflate): opaque byte blob, no extra
// knobs needed at decompress time beyond knowing the original type.
struct snappy {
  std::uint64_t uncompressed_size = 0;
  std::int32_t original_type_id   = 0;
};
struct lz4 {
  std::uint64_t uncompressed_size = 0;
  std::int32_t original_type_id   = 0;
};
struct deflate {
  std::uint64_t uncompressed_size = 0;
  std::int32_t original_type_id   = 0;
};

}  // namespace leaf_meta

using leaf_meta_v = std::variant<leaf_meta::none,
                                 leaf_meta::alp_rd,
                                 leaf_meta::ans,
                                 leaf_meta::bitcomp,
                                 leaf_meta::nvcomp_cascaded,
                                 leaf_meta::snappy,
                                 leaf_meta::lz4,
                                 leaf_meta::deflate>;

// ---------------------------------------------------------------------------
// Per-buffer descriptor used by compressed_table::describe() and the IO layer.
// ---------------------------------------------------------------------------

struct leaf_buffer_desc {
  std::string name;
  std::uint8_t type_tag    = 0;
  std::uint64_t num_rows   = 0;
  std::uint64_t size_bytes = 0;
  const void* device_ptr   = nullptr;
};

// `slot` value marking a leaf as its node's own rep (rather than a terminal
// output-channel rep, which uses the output-port index).
inline constexpr std::int32_t kSelfRepSlot = -1;

// Descriptor for one compressed leaf (one rep slot in the PlanTree), located
// structurally: `node_index` is its owning node, `slot` is kSelfRepSlot for the
// node's own rep or an output-port index for a terminal channel rep. Produced
// by compressed_table::describe(); consumed by the file writer and read path.
struct leaf_desc {
  std::uint32_t node_index = 0;
  std::int32_t slot        = kSelfRepSlot;
  PlanLeafKind kind        = PlanLeafKind::Unknown;
  std::uint8_t type_tag    = 0;  // decoded column dtype tag
  leaf_meta_v meta{leaf_meta::none{}};
  std::vector<leaf_buffer_desc> buffers;  // channel buffers in named_channels() order
};

// ---------------------------------------------------------------------------
// Type-tag encoding. One table drives both directions; 255 = unsupported type.
// ---------------------------------------------------------------------------

inline constexpr std::pair<cudf::type_id, std::uint8_t> kTypeTags[] = {
  {cudf::type_id::INT8, 0},
  {cudf::type_id::INT16, 1},
  {cudf::type_id::INT32, 2},
  {cudf::type_id::INT64, 3},
  {cudf::type_id::UINT8, 4},
  {cudf::type_id::UINT16, 5},
  {cudf::type_id::UINT32, 6},
  {cudf::type_id::UINT64, 7},
  {cudf::type_id::FLOAT32, 8},
  {cudf::type_id::FLOAT64, 9},
  {cudf::type_id::STRING, 10},
};

inline std::uint8_t dtype_to_tag(cudf::data_type dt) noexcept
{
  for (auto const& [id, tag] : kTypeTags)
    if (id == dt.id()) return tag;
  return 255;
}

inline cudf::data_type tag_to_dtype(std::uint8_t tag) noexcept
{
  for (auto const& [id, t] : kTypeTags)
    if (t == tag) return cudf::data_type{id};
  return cudf::data_type{cudf::type_id::EMPTY};
}

}  // namespace simpatico

#endif  // CODEGEN_LEAF_DESC_HPP
