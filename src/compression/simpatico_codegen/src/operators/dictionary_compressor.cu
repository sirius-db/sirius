/**
 * Dictionary compressor: STRING column -> dictionary column (encode); decompress via decode.
 * Stores the encoded dictionary column to avoid copying keys chars from cuDF (invalid pointers).
 */

#include "codegen/plan/representation.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/reduction/approx_distinct_count.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>

namespace simpatico {

namespace {

constexpr size_t MAX_INDICES = 1 << 28;  // 256M rows (sanity bound)

// cudf::dictionary::encode faults with a context-corrupting illegal access on
// very large, very-high-cardinality strings — inputs a dictionary can't help
// anyway. Skip such columns before the full encode, gating on an estimate of
// the *full-column* distinct fraction: a prefix probe mis-reads long columns
// with moderate absolute cardinality (e.g. 1M distinct over 100M rows is ~0.23
// unique in a 256K-row prefix but 0.01 over the whole column — an ideal
// dictionary target). A fixed-memory HyperLogLog sketch gives the true
// fraction in one pass without materializing the keys column that faults.
constexpr size_t kDictCardCheckMinRows = 1 << 20;
constexpr double kDictMaxCardFraction  = 0.5;

std::unique_ptr<dictionary_compressed_representation> dictionary_compress_impl(
  cudf::column_view const& col,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  bool fast)
{
  if (col.type().id() != cudf::type_id::STRING) {
    throw std::runtime_error("dictionary_compressor: column must be STRING, got '" +
                             type_id_to_name(col.type()) + "'");
  }
  auto const n = col.size();
  if (n < 0) { throw std::runtime_error("dictionary_compressor: column size is negative"); }
  if (static_cast<size_t>(n) > MAX_INDICES) {
    throw std::runtime_error("dictionary_compressor: column size exceeds maximum");
  }
  if (n == 0) {
    if (fast) {
      // Empty fast rep exposes the {keys, indices} channels the fast plan expects.
      return std::make_unique<dictionary_compressed_representation>(
        cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING)),
        cudf::make_empty_column(cudf::data_type(cudf::type_id::INT32)));
    }
    auto empty_dict = cudf::make_empty_column(cudf::data_type(cudf::type_id::DICTIONARY32));
    return std::make_unique<dictionary_compressed_representation>(std::move(empty_dict));
  }

  if (static_cast<size_t>(n) > kDictCardCheckMinRows) {
    cudf::approx_distinct_count sketch(cudf::table_view{{col}},
                                       12,  // precision -> ~1.6% standard error
                                       cudf::null_policy::INCLUDE,
                                       cudf::nan_policy::NAN_IS_NULL,
                                       stream);
    auto const keys = sketch.estimate(stream);
    if (static_cast<double>(keys) > kDictMaxCardFraction * static_cast<double>(n)) {
      throw std::runtime_error("dictionary_compressor: cardinality too high (skipping)");
    }
  }

  auto dict_col = cudf::dictionary::encode(col, cudf::data_type(cudf::type_id::INT32), stream, mr);
  cudaStreamSynchronize(stream.value());
  if (fast) {
    // Fast mode: split the encoded dictionary into an owned keys (STRING) column and
    // an owned indices column carrying the parent null mask (get_indices_annotated), so
    // validity survives make_dictionary_column + decode. Leaner in-memory footprint than
    // the 3-buffer keys_offsets/keys_chars/indices form.
    cudf::dictionary_column_view dv(dict_col->view());
    auto keys    = std::make_unique<cudf::column>(dv.keys(), stream, mr);
    auto indices = std::make_unique<cudf::column>(dv.get_indices_annotated(), stream, mr);
    cudaStreamSynchronize(stream.value());
    return std::make_unique<dictionary_compressed_representation>(std::move(keys),
                                                                  std::move(indices));
  }
  return std::make_unique<dictionary_compressed_representation>(std::move(dict_col));
}

}  // namespace

std::unique_ptr<cudf::column> dictionary_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  nvtx3::scoped_range r{"dictionary_decompress"};
  // Fast reconstruction mode: create dictionary from keys + indices directly
  if (fast_mode) {
    if (!keys_column || !indices_only) { return nullptr; }
    if (indices_only->size() == 0) {
      return cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
    }
    if (keys_column->size() == 0) {
      // Zero keys with rows present means every row is null (encode drops
      // null rows from the key set); decode would gather from the empty —
      // possibly childless — keys column, so build the all-null strings
      // column directly.
      return cudf::make_column_from_scalar(
        cudf::string_scalar("", false, stream, mr), indices_only->size(), stream, mr);
    }
    // Create dictionary column from keys and indices, then decode
    // This avoids the expensive make_strings_column reconstruction from offsets+chars
    auto dict_col =
      cudf::make_dictionary_column(keys_column->view(), indices_only->view(), stream, mr);
    return cudf::dictionary::decode(dict_col->view(), stream, mr);
  }

  // Full dict_column mode: decode from the stored dictionary column.
  if (dict_column == nullptr) { return nullptr; }
  if (dict_column->size() == 0) {
    return cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  }
  if (cudf::dictionary_column_view(dict_column->view()).keys().size() == 0) {
    // Zero keys with rows present: all rows are null (see fast-mode note).
    return cudf::make_column_from_scalar(
      cudf::string_scalar("", false, stream, mr), dict_column->size(), stream, mr);
  }
  return cudf::dictionary::decode(dict_column->view(), stream, mr);
}

std::unique_ptr<compressed_representation> dictionary_compressor::compress(
  cudf::column_view column_to_compress,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return dictionary_compress_impl(column_to_compress, stream, mr, fast_);
}

std::unique_ptr<cudf::column> dictionary_compressor::decompress(
  compressed_representation const& data_to_decompress,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const* dict_repr =
    dynamic_cast<dictionary_compressed_representation const*>(&data_to_decompress);
  if (dict_repr == nullptr) { return nullptr; }
  return dict_repr->decompress(stream, mr);
}

}  // namespace simpatico
