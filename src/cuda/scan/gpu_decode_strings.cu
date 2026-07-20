/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//===----------------------------------------------------------------------===//
// String column decode — orchestrator.
//
// gpu_decode_strings_column decodes one VARCHAR column. The four string codecs
// each live in strings/<codec>.cu and expose host wrappers:
//   prepare_*         build per-run descriptors (some also run on-device prep,
//                     e.g. FSST symbol tables, DICT_FSST dictionary predecode)
//   launch_*_lengths  phase 1: write each row's decoded byte length
//   launch_*_gather   phase 2: copy each row's decoded bytes to the output
//
// All codecs share one two-phase flow:
//   1. each codec writes its rows' lengths into a single d_lengths array
//   2. one exclusive scan turns d_lengths into the column's d_offsets
//   3. each codec gathers its bytes into d_chars at those offsets
// then the cudf strings column is assembled (offsets + chars + null mask).
//
// Each codec's on-disk segment layout is documented at the top of its file.
//===----------------------------------------------------------------------===//

#include "cuda/scan/gpu_decode_strings.cuh"
#include "cuda/scan/strings/common.cuh"
#include "cuda/scan/strings/dict_fsst.cuh"
#include "cuda/scan/strings/dictionary.cuh"
#include "cuda/scan/strings/fsst.cuh"
#include "cuda/scan/strings/uncompressed.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>

#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius::cuda::scan {

namespace {

/// Overlays an UNCOMPRESSED validity run onto the null mask (sibling to
/// `dispatch_validity_run` in gpu_native_decode.cu).
void overlay_validity_run(gpu_codec_run const& run, uint8_t* d_mask, rmm::cuda_stream_view stream)
{
  if (run.codec != duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
    throw std::runtime_error(
      "gpu_decode_strings_column: viability invariant violated — "
      "validity codec " +
      std::to_string(static_cast<int>(run.codec)) + " not implemented");
  }
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    if (seg.row_offset % 8 != 0) {
      throw std::runtime_error("gpu_decode_strings_column: validity row_offset (" +
                               std::to_string(seg.row_offset) + ") not byte-aligned");
    }
    auto const bytes  = ::cuda::ceil_div(seg.row_count, 8);
    auto const offset = seg.row_offset / 8;
    if (seg.bytes_size < bytes) {
      throw std::runtime_error("gpu_decode_strings_column: validity segment bytes_size (" +
                               std::to_string(seg.bytes_size) + ") < required " +
                               std::to_string(bytes));
    }
    RMM_CUDA_TRY(cudaMemcpyAsync(
      d_mask + offset, seg.d_bytes, bytes, cudaMemcpyDeviceToDevice, stream.value()));
  }
}

}  // namespace

/// Decodes one VARCHAR column via the shared two-phase flow (see file banner):
/// aggregate per-codec prepared state, write per-row lengths, scan to offsets,
/// gather bytes, then build the cudf strings column.
std::unique_ptr<cudf::column> gpu_decode_strings_column(gpu_string_column_decode_input const& col,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  uint32_t const total_rows = col.total_rows;
  if (total_rows == 0) { return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}); }
  // Reject total_rows >= cudf::size_type max, not just >. cudf strings columns
  // carry `total_rows + 1` offsets (one per row plus a trailing offset), so
  // total_rows == INT32_MAX makes `total_rows + 1` overflow: the exclusive-sum
  // scan_n at the +1 site wraps to a negative int32 (UB), and the offsets
  // column size `total_rows + 1u` wraps to 0. The tight bound
  // (total_rows < cudf::size_type max) leaves room for the trailing offset.
  if (total_rows >= static_cast<uint32_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw std::runtime_error("gpu_decode_strings_column: total_rows (" +
                             std::to_string(total_rows) + ") >= cudf::size_type max "
                             "(no room for the trailing offsets row)");
  }

  prepared_uncomp prep_uncomp;
  prepared_dict prep_dict;
  prepared_fsst prep_fsst;
  prepared_dict_fsst prep_dict_fsst;
  prep_dict_fsst.any_inline_nulls      = false;
  prep_dict_fsst.total_predecode_bytes = 0;
  size_t cum_chars_upper               = 0;
  bool needs_exact_total               = false;
  for (auto const& run : col.data) {
    switch (run.codec) {
      case duckdb::CompressionType::COMPRESSION_DICTIONARY: {
        auto p = prepare_dict(run);
        prep_dict.descs_short.insert(
          prep_dict.descs_short.end(), p.descs_short.begin(), p.descs_short.end());
        prep_dict.descs_long.insert(
          prep_dict.descs_long.end(), p.descs_long.begin(), p.descs_long.end());
        break;
      }
      case duckdb::CompressionType::COMPRESSION_FSST: {
        auto p = prepare_fsst(run);
        // Rebase row_starts + decoder indices into the merged FSST set.
        auto const row_count_base     = prep_fsst.total_fsst_row_count;
        auto const decoder_count_base = static_cast<uint32_t>(prep_fsst.decoders.size());
        for (auto& s : p.row_starts) {
          s += row_count_base;
        }
        for (auto& c : p.gather_chunks) {
          c.fsst_row_start += row_count_base;
          c.seg_decoder_idx += decoder_count_base;
        }
        prep_fsst.length_descs.insert(
          prep_fsst.length_descs.end(), p.length_descs.begin(), p.length_descs.end());
        prep_fsst.row_starts.insert(
          prep_fsst.row_starts.end(), p.row_starts.begin(), p.row_starts.end());
        prep_fsst.decoders.insert(prep_fsst.decoders.end(), p.decoders.begin(), p.decoders.end());
        prep_fsst.gather_chunks.insert(
          prep_fsst.gather_chunks.end(), p.gather_chunks.begin(), p.gather_chunks.end());
        prep_fsst.total_fsst_row_count += p.total_fsst_row_count;
        break;
      }
      case duckdb::CompressionType::COMPRESSION_DICT_FSST: {
        auto p                    = prepare_dict_fsst(run, stream, mr);
        // The device struct (dict_fsst_desc) uses uint32_t offset bases, so
        // the cross-run cumulative sizes must fit in 32 bits — including the
        // new run's entries that will be appended below. The old check tested
        // only the current cumulative size (before appending), so it missed
        // the case where cumulative + new_run > UINT32_MAX. Check the sum.
        if (prep_dict_fsst.byte_offsets.size() + p.byte_offsets.size() > std::numeric_limits<uint32_t>::max() ||
            prep_dict_fsst.decoders.size() + p.decoders.size() > std::numeric_limits<uint32_t>::max()) {
          throw std::runtime_error(
            "gpu_decode_strings: dict_fsst cross-run offset base exceeds UINT32_MAX "
            "(too many dictionary entries across runs)");
        }
        auto const bo_base        = static_cast<uint32_t>(prep_dict_fsst.byte_offsets.size());
        auto const dec_base       = static_cast<uint32_t>(prep_dict_fsst.decoders.size());
        auto const predecode_base = prep_dict_fsst.total_predecode_bytes;
        for (auto& d : p.descs) {
          d.seg_dict_offset_base += bo_base;
          d.seg_decoder_idx += dec_base;
          if (d.mode == DICT_FSST_MODE_DICT_FSST) { d.predecode_seg_offset += predecode_base; }
        }
        prep_dict_fsst.byte_offsets.insert(
          prep_dict_fsst.byte_offsets.end(), p.byte_offsets.begin(), p.byte_offsets.end());
        prep_dict_fsst.decoded_offsets.insert(
          prep_dict_fsst.decoded_offsets.end(), p.decoded_offsets.begin(), p.decoded_offsets.end());
        prep_dict_fsst.decoders.insert(
          prep_dict_fsst.decoders.end(), p.decoders.begin(), p.decoders.end());
        prep_dict_fsst.descs.insert(prep_dict_fsst.descs.end(), p.descs.begin(), p.descs.end());
        prep_dict_fsst.any_inline_nulls = prep_dict_fsst.any_inline_nulls || p.any_inline_nulls;
        prep_dict_fsst.total_predecode_bytes += p.total_predecode_bytes;
        break;
      }
      case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED: {
        auto p = prepare_uncomp(run);
        prep_uncomp.descs.insert(prep_uncomp.descs.end(), p.descs.begin(), p.descs.end());
        break;
      }
      default:
        throw std::runtime_error(
          "gpu_decode_strings_column: viability invariant violated — "
          "data codec " +
          std::to_string(static_cast<int>(run.codec)) + " not implemented");
    }
    // Upper-bound from walker stats; 0 means unknown → take the sync path.
    for (auto const& seg : run.segments) {
      if (seg.max_string_length == 0u) {
        needs_exact_total = true;
        continue;
      }
      cum_chars_upper += size_t{seg.row_count} * seg.max_string_length;
    }
  }

  // Allocate output and intermediate buffers.
  rmm::device_uvector<uint32_t> d_lengths(size_t{total_rows} + 1, stream, mr);
  rmm::device_uvector<int32_t> d_offsets(size_t{total_rows} + 1, stream, mr);
  rmm::device_buffer d_comp_offsets(prep_fsst.total_fsst_row_count * sizeof(uint32_t), stream, mr);

  // Per-row kernels take chunked descriptors; predecode + mark_nulls stay
  // per-segment via prep_dict_fsst.descs.
  auto const target_ctas       = get_target_ctas();
  auto const uncomp_chunks     = expand_chunks(prep_uncomp.descs, target_ctas);
  auto const dict_chunks_short = expand_chunks(prep_dict.descs_short, target_ctas);
  auto const dict_chunks_long  = expand_chunks(prep_dict.descs_long, target_ctas);
  auto const dict_fsst_chunks  = expand_chunks(prep_dict_fsst.descs, target_ctas);

  auto upload = [&](void const* src, size_t bytes) {
    rmm::device_buffer buf(bytes, stream, mr);
    if (bytes > 0) {
      RMM_CUDA_TRY(cudaMemcpyAsync(buf.data(), src, bytes, cudaMemcpyHostToDevice, stream.value()));
    }
    return buf;
  };
  rmm::device_buffer d_uncomp_chunks_buf =
    upload(uncomp_chunks.data(), uncomp_chunks.size() * sizeof(string_chunk_desc));
  rmm::device_buffer d_dict_short_buf =
    upload(dict_chunks_short.data(), dict_chunks_short.size() * sizeof(string_chunk_desc));
  rmm::device_buffer d_dict_long_buf =
    upload(dict_chunks_long.data(), dict_chunks_long.size() * sizeof(string_chunk_desc));
  rmm::device_buffer d_dict_fsst_chunks_buf =
    upload(dict_fsst_chunks.data(), dict_fsst_chunks.size() * sizeof(dict_fsst_desc));
  rmm::device_buffer d_fsst_lengths_buf = upload(
    prep_fsst.length_descs.data(), prep_fsst.length_descs.size() * sizeof(string_chunk_desc));
  rmm::device_buffer d_fsst_chunks_buf = upload(
    prep_fsst.gather_chunks.data(), prep_fsst.gather_chunks.size() * sizeof(fsst_chunk_desc));
  rmm::device_buffer d_fsst_starts_buf =
    upload(prep_fsst.row_starts.data(), prep_fsst.row_starts.size() * sizeof(uint32_t));
  rmm::device_buffer d_fsst_decoders_buf =
    upload(prep_fsst.decoders.data(), prep_fsst.decoders.size() * sizeof(fsst_decoder_compact));
  rmm::device_buffer d_dict_fsst_descs_buf =
    upload(prep_dict_fsst.descs.data(), prep_dict_fsst.descs.size() * sizeof(dict_fsst_desc));
  rmm::device_buffer d_dict_fsst_decoders_buf = upload(
    prep_dict_fsst.decoders.data(), prep_dict_fsst.decoders.size() * sizeof(fsst_decoder_compact));
  rmm::device_buffer d_byte_offsets_buf = upload(
    prep_dict_fsst.byte_offsets.data(), prep_dict_fsst.byte_offsets.size() * sizeof(uint32_t));
  rmm::device_buffer d_decoded_offsets_buf =
    upload(prep_dict_fsst.decoded_offsets.data(),
           prep_dict_fsst.decoded_offsets.size() * sizeof(uint32_t));

  // Pageable host sources — sync before kernels consume to avoid free-mid-copy.
  RMM_CUDA_TRY(cudaStreamSynchronize(stream.value()));

  auto* d_comp_offsets_p     = static_cast<uint32_t*>(d_comp_offsets.data());
  auto* d_uncomp_chunks_p    = static_cast<string_chunk_desc*>(d_uncomp_chunks_buf.data());
  auto* d_dict_short_p       = static_cast<string_chunk_desc*>(d_dict_short_buf.data());
  auto* d_dict_long_p        = static_cast<string_chunk_desc*>(d_dict_long_buf.data());
  auto* d_fsst_lengths_p     = static_cast<string_chunk_desc*>(d_fsst_lengths_buf.data());
  auto* d_fsst_chunks_p      = static_cast<fsst_chunk_desc*>(d_fsst_chunks_buf.data());
  auto* d_fsst_starts_p      = static_cast<uint32_t*>(d_fsst_starts_buf.data());
  auto* d_fsst_decs_p        = static_cast<fsst_decoder_compact*>(d_fsst_decoders_buf.data());
  auto* d_dict_fsst_p        = static_cast<dict_fsst_desc*>(d_dict_fsst_descs_buf.data());
  auto* d_dict_fsst_chunks_p = static_cast<dict_fsst_desc*>(d_dict_fsst_chunks_buf.data());
  auto* d_dict_fsst_decs_p   = static_cast<fsst_decoder_compact*>(d_dict_fsst_decoders_buf.data());
  auto* d_byte_off_p         = static_cast<uint32_t*>(d_byte_offsets_buf.data());
  auto* d_decoded_off_p      = static_cast<uint32_t*>(d_decoded_offsets_buf.data());

  // Pass 1: lengths. Same kernel for short/long DICTIONARY — only gather forks.
  launch_uncomp_lengths(
    d_uncomp_chunks_p, d_lengths.data(), static_cast<uint32_t>(uncomp_chunks.size()), stream);
  launch_dict_lengths(
    d_dict_short_p, d_lengths.data(), static_cast<uint32_t>(dict_chunks_short.size()), stream);
  launch_dict_lengths(
    d_dict_long_p, d_lengths.data(), static_cast<uint32_t>(dict_chunks_long.size()), stream);
  launch_fsst_lengths(d_fsst_decs_p,
                      d_comp_offsets_p,
                      d_lengths.data(),
                      d_fsst_lengths_p,
                      d_fsst_starts_p,
                      d_fsst_chunks_p,
                      static_cast<uint32_t>(prep_fsst.length_descs.size()),
                      static_cast<uint32_t>(prep_fsst.gather_chunks.size()),
                      stream);
  // Predecode buffer holds decoded dict bytes for mode-1 segments.
  rmm::device_buffer d_predecode_buf(
    prep_dict_fsst.total_predecode_bytes > 0 ? prep_dict_fsst.total_predecode_bytes : 1u,
    stream,
    mr);
  auto* d_predecode_p = static_cast<uint8_t*>(d_predecode_buf.data());

  // Lengths chunk for SM-fill; predecode stays per-segment (one decode/dict).
  launch_dict_fsst_lengths(d_dict_fsst_chunks_p,
                           d_lengths.data(),
                           d_decoded_off_p,
                           static_cast<uint32_t>(dict_fsst_chunks.size()),
                           stream);
  launch_dict_fsst_predecode(d_dict_fsst_p,
                             d_byte_off_p,
                             d_decoded_off_p,
                             d_dict_fsst_decs_p,
                             d_predecode_p,
                             static_cast<uint32_t>(prep_dict_fsst.descs.size()),
                             prep_dict_fsst.total_predecode_bytes,
                             stream);

  // Prefix-sum lengths → byte offsets per row.
  size_t cub_bytes  = 0;
  auto const scan_n = static_cast<int>(total_rows) + 1;
  cub::DeviceScan::ExclusiveSum(nullptr,
                                cub_bytes,
                                d_lengths.data(),
                                reinterpret_cast<uint32_t*>(d_offsets.data()),
                                scan_n,
                                stream.value());
  rmm::device_buffer cub_temp_buf(cub_bytes, stream, mr);
  cub::DeviceScan::ExclusiveSum(cub_temp_buf.data(),
                                cub_bytes,
                                d_lengths.data(),
                                reinterpret_cast<uint32_t*>(d_offsets.data()),
                                scan_n,
                                stream.value());

  // cudf strings offsets are int32; reject up front if the upper bound exceeds it.
  constexpr auto INT32_MAX_SIZE = static_cast<size_t>(std::numeric_limits<int32_t>::max());
  if (!needs_exact_total && cum_chars_upper > INT32_MAX_SIZE) {
    throw std::runtime_error("gpu_decode_strings_column: estimated total_chars (" +
                             std::to_string(cum_chars_upper) + ") exceeds int32 max");
  }
  size_t alloc_chars = 0;
  if (!needs_exact_total && cum_chars_upper <= HOST_UPPER_BOUND_LIMIT) {
    alloc_chars = cum_chars_upper;
  } else {
    RMM_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    uint32_t total_chars_u = 0;
    RMM_CUDA_TRY(cudaMemcpy(
      &total_chars_u, d_offsets.data() + total_rows, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (total_chars_u > static_cast<uint32_t>(INT32_MAX_SIZE)) {
      throw std::runtime_error("gpu_decode_strings_column: total_chars (" +
                               std::to_string(total_chars_u) + ") exceeds int32 max");
    }
    alloc_chars = total_chars_u;
  }

  rmm::device_buffer d_chars(alloc_chars > 0 ? alloc_chars : 1u, stream, mr);
  auto* d_chars_p = static_cast<uint8_t*>(d_chars.data());

  // Pass 2: gather. See DICT_WARP_COOP_MIN_LEN for the partition rationale.
  launch_uncomp_gather(d_uncomp_chunks_p,
                       d_offsets.data(),
                       d_chars_p,
                       static_cast<uint32_t>(uncomp_chunks.size()),
                       stream);
  launch_dict_gather_short(d_dict_short_p,
                           d_offsets.data(),
                           d_chars_p,
                           static_cast<uint32_t>(dict_chunks_short.size()),
                           stream);
  launch_dict_gather_long(d_dict_long_p,
                          d_offsets.data(),
                          d_chars_p,
                          static_cast<uint32_t>(dict_chunks_long.size()),
                          stream);
  launch_fsst_gather(d_fsst_chunks_p,
                     d_offsets.data(),
                     d_chars_p,
                     d_comp_offsets_p,
                     d_fsst_decs_p,
                     static_cast<uint32_t>(prep_fsst.gather_chunks.size()),
                     stream);
  launch_dict_fsst_gather(d_dict_fsst_chunks_p,
                          d_offsets.data(),
                          d_chars_p,
                          d_byte_off_p,
                          d_decoded_off_p,
                          d_predecode_p,
                          d_dict_fsst_decs_p,
                          static_cast<uint32_t>(dict_fsst_chunks.size()),
                          stream);

  // All-valid → overlay UNCOMPRESSED validity → fold in DICT_FSST inline NULLs.
  rmm::device_buffer null_mask{};
  cudf::size_type null_count = 0;
  bool need_mask             = col.has_nulls || prep_dict_fsst.any_inline_nulls;
  if (need_mask) {
    null_mask = cudf::create_null_mask(
      static_cast<cudf::size_type>(total_rows), cudf::mask_state::ALL_VALID, stream, mr);
    for (auto const& run : col.validity) {
      overlay_validity_run(run, static_cast<uint8_t*>(null_mask.data()), stream);
    }
    if (prep_dict_fsst.any_inline_nulls) {
      launch_dict_fsst_mark_nulls(d_dict_fsst_p,
                                  static_cast<uint8_t*>(null_mask.data()),
                                  static_cast<uint32_t>(prep_dict_fsst.descs.size()),
                                  stream);
    }
    null_count = cudf::null_count(static_cast<cudf::bitmask_type const*>(null_mask.data()),
                                  0,
                                  static_cast<cudf::size_type>(total_rows),
                                  stream);
  }

  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    static_cast<cudf::size_type>(total_rows + 1u),
                                                    d_offsets.release(),
                                                    rmm::device_buffer{0, stream, mr},
                                                    0);

  RMM_CUDA_TRY(cudaPeekAtLastError());
  return cudf::make_strings_column(static_cast<cudf::size_type>(total_rows),
                                   std::move(offsets_col),
                                   std::move(d_chars),
                                   null_count,
                                   std::move(null_mask));
}

}  // namespace sirius::cuda::scan
