// SPDX-License-Identifier: Apache-2.0
#pragma once

// C++-native .hpln v6 writer/reader for compressed_table.
//
// File layout (v6):
//   [DSL section]  — human-readable DSL per column, separated by "---\n"
//   \n---END-HEADERS-V6---\n
//   [Binary header]
//     "HPLN" (4 bytes)
//     version = 6 (uint8)
//     num_cols (uint16 LE)
//     per column:
//       name_len (uint16 LE) + name bytes   [0 = no name]
//       dtype_tag (uint8)                   [decoded column type]
//       num_rows (int64 LE)
//       dsl_len (uint32 LE) + dsl bytes     [column's own plan DSL]
//       num_leaves (uint16 LE)
//       per leaf:
//         path_len (uint16 LE) + path bytes
//         kind (uint8)                      [PlanLeafKind]
//         type_tag (uint8)                  [decoded element type]
//         meta_kind (uint8)  0=none 1=alp_rd 2=ans 3=bitcomp 4=cascaded 5=snappy 6=lz4 7=deflate
//         meta bytes (variable per meta_kind):
//           none:      (empty)
//           alp_rd:    right_bw (uint8)
//           ans:       uncompressed_size (uint64 LE) + original_type_id (int32 LE)
//           bitcomp:   uncompressed_size (uint64 LE) + original_type_id (int32 LE) + algorithm
//           (int32 LE) cascaded:  uncompressed_size (uint64) + original_type_id (int32) +
//           num_deltas (int32) + num_RLEs (int32) + use_bp (int32) snappy:    uncompressed_size
//           (uint64 LE) + original_type_id (int32 LE) lz4:       uncompressed_size (uint64 LE) +
//           original_type_id (int32 LE) deflate:   uncompressed_size (uint64 LE) + original_type_id
//           (int32 LE)
//         num_bufs (uint8)
//         per buffer:
//           name_len (uint16 LE) + name bytes
//           buf_type_tag (uint8)
//           size_bytes (uint64 LE)
//           payload_offset (uint64 LE)
//   [Payload]  — all buffer bytes concatenated in write order, copied D→H

#include "api/simpatico_codegen.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <string>

namespace simpatico {

/// Write a compressed_table to *path*.
/// Returns an empty string on success; a human-readable error message otherwise.
std::string write_compressed_table(compressed_table const& table, std::string const& path);

/// Read a v6 compressed_table from *path*.
/// On failure writes an error to *error_out (if non-null) and returns an empty
/// compressed_table.
compressed_table read_compressed_table(
  std::string const& path,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref(),
  std::string* error_out            = nullptr);

}  // namespace simpatico
