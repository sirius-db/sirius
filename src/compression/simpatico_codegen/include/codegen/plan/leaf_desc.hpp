// SPDX-License-Identifier: Apache-2.0
#ifndef CODEGEN_LEAF_DESC_HPP
#define CODEGEN_LEAF_DESC_HPP

#include <cudf/types.hpp>

#include <cstdint>
#include <string>
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

// ANS: opaque byte blob — uncompressed_size and original type_id cannot be
// recovered from the compressed payload alone.
struct ans {
  std::uint64_t uncompressed_size = 0;
  std::int32_t original_type_id   = 0;  // cast of cudf::type_id
};

// Bitcomp: same as ANS, plus the algorithm knob used at compress time.
struct bitcomp {
  std::uint64_t uncompressed_size = 0;
  std::int32_t original_type_id   = 0;
  std::int32_t algorithm          = 0;
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

// Flat descriptor for one compressed leaf (one rep slot in the PlanTree).
// Produced by compressed_table::describe(); consumed by the file writer and
// by the IO read path to reconstruct reps.
struct leaf_desc {
  std::string path;  // DSL dotted path keying this leaf
  PlanLeafKind kind     = PlanLeafKind::Unknown;
  std::uint8_t type_tag = 0;  // decoded column dtype tag
  leaf_meta_v meta{leaf_meta::none{}};
  std::vector<leaf_buffer_desc> buffers;  // channel buffers in named_channels() order
};

// ---------------------------------------------------------------------------
// Type-tag encoding — mirrors cudf_shim.cpp type_id_to_tag / type_id_from_tag.
// ---------------------------------------------------------------------------

inline std::uint8_t dtype_to_tag(cudf::data_type dt) noexcept
{
  switch (dt.id()) {
    case cudf::type_id::INT8: return 0;
    case cudf::type_id::INT16: return 1;
    case cudf::type_id::INT32: return 2;
    case cudf::type_id::INT64: return 3;
    case cudf::type_id::UINT8: return 4;
    case cudf::type_id::UINT16: return 5;
    case cudf::type_id::UINT32: return 6;
    case cudf::type_id::UINT64: return 7;
    case cudf::type_id::FLOAT32: return 8;
    case cudf::type_id::FLOAT64: return 9;
    default: return 255;
  }
}

inline cudf::data_type tag_to_dtype(std::uint8_t tag) noexcept
{
  switch (tag) {
    case 0: return cudf::data_type{cudf::type_id::INT8};
    case 1: return cudf::data_type{cudf::type_id::INT16};
    case 2: return cudf::data_type{cudf::type_id::INT32};
    case 3: return cudf::data_type{cudf::type_id::INT64};
    case 4: return cudf::data_type{cudf::type_id::UINT8};
    case 5: return cudf::data_type{cudf::type_id::UINT16};
    case 6: return cudf::data_type{cudf::type_id::UINT32};
    case 7: return cudf::data_type{cudf::type_id::UINT64};
    case 8: return cudf::data_type{cudf::type_id::FLOAT32};
    case 9: return cudf::data_type{cudf::type_id::FLOAT64};
    default: return cudf::data_type{cudf::type_id::EMPTY};
  }
}

}  // namespace simpatico

#endif  // CODEGEN_LEAF_DESC_HPP
