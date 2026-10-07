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

#pragma once

#include "vss/kmeans_functions.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace duckdb {
class SiriusContext;
}  // namespace duckdb

namespace sirius::scan_manager {
struct pinned_entry;
}  // namespace sirius::scan_manager

namespace sirius::vss {

/**
 * @brief A pinned corpus column rewritten in cluster order: the inverted lists of an IVF index.
 *
 * The clustered join needs every cluster's rows contiguous so that a probe routed to a cluster
 * searches one slice. Getting that order through SQL means handing every row's label back to
 * DuckDB and sorting the corpus there, which at 100M rows is minutes of work around seconds of
 * GPU math. This is the same order built in two streaming passes over the pin instead: label
 * every row, count, then scatter each row straight to its place.
 *
 * Layout row r holds pin row @c row_ids[r], so a neighbour found in the lists is reported in
 * the pin's own row space and every downstream reader of the corpus is unaffected.
 */
/// How the lists hold a vector. The join always searches FP32; an encoding is only ever used
/// where it is lossless, so a staged chunk widens back to exactly the values that were pinned.
enum class list_encoding : std::uint8_t {
  float32,
  /// One byte per component: every value was an integer in [0, 255] (SIFT, BigANN, and other
  /// byte-quantized descriptors stored as FLOAT), held as int8 x - 128. A quarter of the memory
  /// and of the transfer, and exact input for an int8 GEMM.
  uint8,
  /// Two bytes per component, IEEE half. LOSSY (11-bit mantissa), so it is only ever used when
  /// asked for: half the memory and the transfer of FP32 for float embeddings.
  float16,
  /// One byte per component, scalar-quantized FP32: code = round((x - offset) / scale), with a
  /// per-component offset and one scale, so an int8 dot product of two codes is the FP32 one up
  /// to that scale. LOSSY, only used when asked for, and only searched by the bounded search,
  /// which filters with the codes under a proven error bound and re-scores what passes in FP32.
  int8,
};

/// Bytes one component takes in @p e.
[[nodiscard]] inline std::size_t list_encoding_bytes(list_encoding e)
{
  switch (e) {
    case list_encoding::uint8:
    case list_encoding::int8: return 1;
    case list_encoding::float16: return 2;
    default: return 4;
  }
}

struct cluster_lists {
  const scan_manager::pinned_entry* pin{nullptr};  ///< The pin the lists were built from.
  std::string table;
  std::string column;
  std::int64_t n_rows{0};
  std::int64_t dim{0};
  std::int64_t n_clusters{0};
  /// Rows per staged chunk; only the last chunk may be shorter.
  std::int64_t chunk_rows{0};
  /// List c is layout rows [offsets[c], offsets[c + 1]).
  std::vector<std::int64_t> offsets;
  ::cucascade::memory::Tier tier{::cucascade::memory::Tier::GPU};
  list_encoding encoding{list_encoding::float32};

  /// GPU tier: the whole [n_rows x dim] matrix.
  std::unique_ptr<rmm::device_buffer> device_vectors;
  /// HOST tier: one engine data batch per chunk, the chunk's rows packed densely as a single
  /// flat column (INT8, INT16 for half, FLOAT32) in pinned host blocks, so the engine owns their
  /// residency: a query registers them in a repository where the downgrade executor may spill
  /// them to disk, and the join read-locks a chunk for as long as it copies it in.
  std::vector<std::shared_ptr<::cucascade::data_batch>> chunk_batches;
  /// The host space the chunks were allocated in, and where a spilled one is brought back.
  ::cucascade::memory::memory_space* host_space{nullptr};

  /// INT64 [n_rows], device-resident on either tier: the pin row of each layout row.
  std::unique_ptr<rmm::device_buffer> row_ids;
  /// UINT8 lists only: INT32 [n_rows] |x - 128|^2 per layout row, device-resident, which is what
  /// the int8 search adds to its dot products in place of a per-query norm pass.
  std::unique_ptr<rmm::device_buffer> row_sq;
  /// FLOAT16 lists only, all three or none: the FP32 rows in layout order, in pinned host blocks
  /// of exact_rows_per_block rows that the device reads through their mapping (exact_blocks holds
  /// the device pointers); FP32 [n_rows] |x|^2 of those rows on the device; and the largest row
  /// norm. A bounded FP16 search filters with the half rows and re-scores what passes against
  /// these, so its answer is the FP32 one.
  ::cucascade::memory::fixed_multiple_blocks_allocation exact_vectors;
  std::int64_t exact_rows_per_block{0};
  std::unique_ptr<rmm::device_buffer> exact_blocks;
  std::unique_ptr<rmm::device_buffer> row_sq_f32;
  float max_row_norm{0};
  /// FLOAT16 lists: the largest |x - half(x)| over all rows (row_sq_f32 then holds |half(x)|^2
  /// and max_row_norm the largest |half(x)|).
  float half_error{0};
  /// Built with metric => 'cosine': every row (and its kept FP32 copy) divided by its norm, so a
  /// cosine join searches them as L2 over unit vectors, where |q - x|^2 = 2 (1 - cos).
  bool unit_rows{false};
  /// INT8 lists only: FP32 [dim] offset and the one scale the codes were made with, on the
  /// device, and the largest |x - (offset + scale * code)| over all rows.
  std::unique_ptr<rmm::device_buffer> code_offset;
  float code_scale{0};
  float code_error{0};
  /// INT32 [chunk_rows + 1] offsets 0, dim, 2 dim, ... A LIST view of any chunk borrows a
  /// prefix of these, since every list in the column has the same width.
  std::unique_ptr<rmm::device_buffer> list_offsets;

  /// Stored bytes per row, in the list encoding.
  [[nodiscard]] std::size_t row_bytes() const
  {
    return static_cast<std::size_t>(dim) * list_encoding_bytes(encoding);
  }
  [[nodiscard]] std::int64_t num_chunks() const
  {
    return chunk_rows == 0 ? 0 : (n_rows + chunk_rows - 1) / chunk_rows;
  }
  [[nodiscard]] std::int64_t rows_in_chunk(std::int64_t i) const
  {
    return std::min(chunk_rows, n_rows - i * chunk_rows);
  }
};

/// Read-lock chunk @p chunk of HOST-tier lists for staging, bringing it back from the disk
/// tier first if the downgrade executor moved it there.
[[nodiscard]] ::cucascade::read_only_data_batch lock_host_list_chunk(const cluster_lists& lists,
                                                                     std::size_t chunk,
                                                                     rmm::cuda_stream_view stream);

/// Copy rows [row0, row0 + rows) of a locked HOST-tier chunk into device memory at @p dst,
/// walking its pinned blocks densely. Issued on @p stream and not waited for: keep @p ro alive
/// until the stream has run the copy.
void copy_host_list_rows(const cluster_lists& lists,
                         const ::cucascade::read_only_data_batch& ro,
                         std::int64_t row0,
                         std::int64_t rows,
                         std::byte* dst,
                         rmm::cuda_stream_view stream);

/// What a lists build did, for the table function to report.
struct cluster_lists_result {
  std::int64_t n_rows{0};
  std::int64_t n_clusters{0};
  std::int64_t min_list{0};
  std::int64_t max_list{0};
  std::int64_t empty_lists{0};
  std::string tier;
  std::string encoding;
};

/// `storage =>` of the build: FLOAT32 always, UINT8 or fail, or the tightest lossless one.
/// `exact` is what a query builds for itself: UINT8 when every value is a byte, else FLOAT16 with
/// its FP32 copy, so the lists answer exactly either way.
enum class list_storage : std::uint8_t { automatic, float32, uint8, float16, int8, exact };

/**
 * @brief `sirius_kmeans_build_lists(table, column, clustering)`: build @ref cluster_lists for a
 *        pinned column under a fitted clustering, replacing any lists that clustering had.
 *
 * GPU tier when the pin is GPU-resident and the copy fits the device, HOST tier otherwise, or
 * always HOST with @p host_tier.
 */
cluster_lists_result run_kmeans_build_lists(duckdb::SiriusContext& ctx,
                                            const kmeans_assign_request& req,
                                            list_storage storage = list_storage::automatic,
                                            bool host_tier       = false,
                                            bool unit_rows       = false);

/// The lists built for @p clustering, or nullptr when there are none.
[[nodiscard]] const cluster_lists* find_cluster_lists(duckdb::SiriusContext& ctx,
                                                      const std::string& clustering);

/// A clustering whose lists answer a join exactly once every cluster is probed.
struct exact_lists_choice {
  std::string clustering;
  std::int64_t n_clusters{0};
  list_encoding encoding{list_encoding::float32};
  bool on_device{false};
  /// FLOAT16 rows in clusters small enough that the first sweep is seeded (no fixed sweep 0).
  bool seeded{false};
};

/// A clustering of (@p catalog, @p schema, @p table, @p column) whose lists hold all @p n_rows
/// rows and answer exactly when every cluster is probed: FLOAT32 or UINT8 rows, or FLOAT16 rows
/// searched with the bounded GEMM and re-scored from the kept FP32 rows. Lists of unit rows
/// answer cosine joins and only those; INT8 codes are left out (their search refuses too many
/// shapes to pick them unasked). Nullopt when there is none.
[[nodiscard]] std::optional<exact_lists_choice> find_exact_lists(duckdb::SiriusContext& ctx,
                                                                 const std::string& catalog,
                                                                 const std::string& schema,
                                                                 const std::string& table,
                                                                 const std::string& column,
                                                                 bool cosine,
                                                                 std::int64_t n_rows);

/// The lists of the clustering @p clustering when it is one of (@p catalog, @p schema, @p table,
/// @p column)'s and they hold all @p n_rows rows, unit rows exactly when @p cosine: what a join
/// searches approximately under `vector_join_clustering`, in any encoding. Nullopt otherwise.
[[nodiscard]] std::optional<exact_lists_choice> find_named_lists(duckdb::SiriusContext& ctx,
                                                                 const std::string& clustering,
                                                                 const std::string& catalog,
                                                                 const std::string& schema,
                                                                 const std::string& table,
                                                                 const std::string& column,
                                                                 bool cosine,
                                                                 std::int64_t n_rows);

/// Drop the lists built for @p clustering, if any; a re-fit makes them stale.
void erase_cluster_lists(duckdb::SiriusContext& ctx, const std::string& clustering);

/// row_ids[dest[i]] = row_base + i for i in [0, n): the row map of one scattered chunk.
void scatter_row_ids(std::int64_t const* dest,
                     std::int64_t n,
                     std::int64_t row_base,
                     std::int64_t* row_ids,
                     rmm::cuda_stream_view stream);

/// Adds to @p out the number of the @p n values that are not an integer in [0, 255].
void count_non_uint8(float const* values,
                     std::int64_t n,
                     unsigned long long* out,
                     rmm::cuda_stream_view stream);

/// out[i] = in[i] - 128 as int8, for values already known to be integers in [0, 255]. Byte-valued
/// lists are stored this way: L2 does not change under the shift, and int8 is what the
/// tensor-core GEMM multiplies exactly.
void narrow_to_shifted_int8(float const* in,
                            std::int64_t n,
                            std::int8_t* out,
                            rmm::cuda_stream_view stream);

/// out[i] = in[i] + 128 as FP32: the stored bytes back to the pinned values.
void widen_shifted_int8(std::int8_t const* in,
                        std::int64_t n,
                        float* out,
                        rmm::cuda_stream_view stream);

/// out[i] = in[i] rounded to IEEE half (stored as its 16 bits).
void narrow_to_float16(float const* in,
                       std::int64_t n,
                       std::uint16_t* out,
                       rmm::cuda_stream_view stream);

/// out[i] = in[i] (IEEE half bits) as FP32.
void widen_float16(std::uint16_t const* in,
                   std::int64_t n,
                   float* out,
                   rmm::cuda_stream_view stream);

/// Raises max_bits[j] / lowers min_bits[j] to the largest / smallest component j over @p rows rows
/// of @p d floats; the bits are order-preserving encodings (see decode_ordered_float).
void column_min_max(float const* x,
                    std::int64_t rows,
                    std::int64_t d,
                    unsigned int* min_bits,
                    unsigned int* max_bits,
                    rmm::cuda_stream_view stream);

/// The float an order-preserving encoding from column_min_max stands for.
float decode_ordered_float(unsigned int bits);

/// out = clamp(round((x - offset) / scale), -127, 127) per component. Each row's error
/// |x - (offset + scale * out)| goes to row_error[i] when it is given, and raises *max_error_bits
/// (a non-negative float's bits) when that is given.
void quantize_rows_int8(float const* x,
                        std::int64_t rows,
                        std::int64_t d,
                        float const* offset,
                        float scale,
                        std::int8_t* out,
                        float* row_error,
                        unsigned int* max_error_bits,
                        rmm::cuda_stream_view stream);

/// The code-space bound an int8 search filters with: a pair within squared distance bound[r] in
/// FP32 has its code distance within ((sqrt(bound[r]) + probe_error[r] + code_error) / scale)^2,
/// since each side's decoded vector is within its error of the original. With @p lower, the
/// opposite: the largest code distance that proves a pair is within bound[r] (-inf when none).
void int8_code_limit(float const* bound,
                     float const* probe_error,
                     float code_error,
                     float scale,
                     std::int64_t n,
                     float* limit,
                     rmm::cuda_stream_view stream,
                     bool lower = false);

/// An INT8 seed (seed_bound_int8's k-th code distance b) as an FP32 bound, in place: k rows lie
/// within code distance b of probe row r, so each lies within (scale·√b + probe_error[r] +
/// code_error)² of it in FP32, the decoded pair being within both sides' coding error of the true
/// one. That bounds row r's k-th nearest distance.
void int8_seed_upper_bound(float* bound,
                           float const* probe_error,
                           float code_error,
                           float scale,
                           std::int64_t n,
                           rmm::cuda_stream_view stream);

/// |half(x)|^2 per row of FP32 rows @p x and their half-rounded copy @p h, with each row's
/// rounding error |x - half(x)| to row_error[i] and/or into *max_error_bits, and the largest
/// |half(x)|^2 into *max_norm_bits, for whichever are given (non-negative floats' bits).
void half_rows_norms(float const* x,
                     std::uint16_t const* h,
                     std::int64_t rows,
                     std::int64_t d,
                     float* sq,
                     float* row_error,
                     unsigned int* max_error_bits,
                     unsigned int* max_norm_bits,
                     rmm::cuda_stream_view stream);

/// The bound an FP16 search filters with: a pair within squared distance bound[r] in FP32 has
/// its FP16-computed distance within (sqrt(bound[r]) + probe_error[r] + row_error)^2 plus the
/// FP32 sums' error for norms up to sqrt(probe_sq[r]) and @p row_norm. With @p lower, the largest
/// FP16-computed distance that proves a pair is within bound[r] (-inf when none).
void float16_bound_limit(float const* bound,
                         float const* probe_sq,
                         float const* probe_error,
                         float row_error,
                         float row_norm,
                         std::int64_t d,
                         std::int64_t n,
                         float* limit,
                         rmm::cuda_stream_view stream,
                         bool lower = false);

/// A FLOAT16 seed (seed_bound_f16's k-th FP16-computed distance) as an FP32 bound, in place: the
/// rounded pair's distance is within the FP32 sums' error of the computed one (as in
/// float16_bound_limit), and the true pair within probe_error[r] + row_error of the rounded one.
void float16_seed_upper_bound(float* bound,
                              float const* probe_sq,
                              float const* probe_error,
                              float row_error,
                              float row_norm,
                              std::int64_t d,
                              std::int64_t n,
                              rmm::cuda_stream_view stream);

/// |x|^2 per row of shifted int8 rows, exact in int32.
void int8_row_sq_norms(std::int8_t const* x,
                       std::int64_t rows,
                       std::int64_t d,
                       std::int32_t* out,
                       rmm::cuda_stream_view stream);

/// out[i] = |x_i|^2 per FP32 row; *max_bits is raised to the largest one's bit pattern.
void float_row_sq_norms(float const* x,
                        std::int64_t rows,
                        std::int64_t d,
                        float* out,
                        unsigned int* max_bits,
                        rmm::cuda_stream_view stream);

/// out[i, :] = src[rows[i], :] for rows of @p row_bytes bytes.
void gather_bytes(void const* src,
                  std::int64_t row_bytes,
                  std::int64_t const* rows,
                  std::int64_t m,
                  void* out,
                  rmm::cuda_stream_view stream);

/// out[i] = src[rows[i]].
void gather_int32(std::int32_t const* src,
                  std::int64_t const* rows,
                  std::int64_t m,
                  std::int32_t* out,
                  rmm::cuda_stream_view stream);

/// INT32 offsets 0, dim, ..., n * dim into @p out (n + 1 entries).
void fill_list_offsets(std::int32_t* out,
                       std::int64_t n,
                       std::int64_t dim,
                       rmm::cuda_stream_view stream);

}  // namespace sirius::vss
