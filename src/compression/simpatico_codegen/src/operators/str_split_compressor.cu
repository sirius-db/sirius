// SPDX-License-Identifier: Apache-2.0
//
// str_split: decompose STRING into {offsets, chars, null_mask}; reassemble via
// make_strings_column on decode. Structural operator.

#include "codegen/plan/bitjoin_layout.hpp"  // copy_column_view
#include "codegen/plan/representation.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

namespace simpatico {

std::unique_ptr<cudf::column> str_split_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  auto& offsets   = channels_[0];
  auto& chars     = channels_[1];
  auto* null_mask = channels_.size() > 2 ? channels_[2].get() : nullptr;
  if (num_rows == 0 || !offsets) {
    return cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  }

  auto const chars_bytes = static_cast<std::size_t>(chars->size()) *
                           static_cast<std::size_t>(cudf::size_of(chars->type()));
  rmm::device_buffer chars_buf(chars->view().head<void>(), chars_bytes, stream, mr);

  cudf::size_type nc = 0;
  rmm::device_buffer mask_buf(0, stream, mr);
  if (null_mask) {
    auto const* bits =
      reinterpret_cast<cudf::bitmask_type const*>(null_mask->view().data<std::uint8_t>());
    nc = cudf::null_count(bits, 0, num_rows, stream);
    if (nc > 0) {
      mask_buf = rmm::device_buffer(
        null_mask->view().head<void>(), static_cast<std::size_t>(null_mask->size()), stream, mr);
    }
  }
  auto offsets_copy = std::make_unique<cudf::column>(*offsets, stream, mr);
  return cudf::make_strings_column(
    num_rows, std::move(offsets_copy), std::move(chars_buf), nc, std::move(mask_buf));
}

std::unique_ptr<cudf::column> str_split_compressed_representation::decompress_consume(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  if (num_rows == 0 || !channels_[0]) {
    return cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  }

  auto chars_contents = channels_[1]->release();
  rmm::device_buffer chars_buf(std::move(*chars_contents.data));

  cudf::size_type nc = 0;
  rmm::device_buffer mask_buf(0, stream, mr);
  if (channels_.size() > 2 && channels_[2]) {
    auto const* bits =
      reinterpret_cast<cudf::bitmask_type const*>(channels_[2]->view().data<std::uint8_t>());
    nc = cudf::null_count(bits, 0, num_rows, stream);
    if (nc > 0) { mask_buf = std::move(*channels_[2]->release().data); }
  }
  return cudf::make_strings_column(
    num_rows, std::move(channels_[0]), std::move(chars_buf), nc, std::move(mask_buf));
}

std::unique_ptr<compressed_representation> str_split_compressor::compress(
  cudf::column_view column_to_compress,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (column_to_compress.type().id() != cudf::type_id::STRING) {
    throw std::runtime_error("str_split_compressor: column must be STRING, got '" +
                             type_id_to_name(column_to_compress.type()) + "'");
  }

  if (column_to_compress.size() == 0) {
    // Canonical empty rep: one zero offset and no chars, built directly rather
    // than copied from the input — the canonical empty STRING column,
    // cudf::make_empty_column(STRING), has no offsets child to copy from.
    auto offsets = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT32}, 1, cudf::mask_state::UNALLOCATED, stream, mr);
    cudaMemsetAsync(offsets->mutable_view().head<void>(), 0, sizeof(std::int32_t), stream.value());
    auto chars = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::UINT8}, 0, cudf::mask_state::UNALLOCATED, stream, mr);
    cudaStreamSynchronize(stream.value());
    return std::make_unique<str_split_compressed_representation>(
      0, std::move(offsets), std::move(chars), nullptr);
  }

  std::unique_ptr<cudf::column>
    owned;  ///< Owned copy of the input column, if needed (post-gather).
  cudf::column_view src = column_to_compress;
  // Normalize any sliced view to an owned compact copy: a non-zero offset
  // needs rebasing, and a head-slice (offset 0, size < parent) still views the
  // parent's full offsets child, so the emitted channels would be mutually
  // inconsistent (offsets/chars parent-sized, null_mask slice-sized).
  if (column_to_compress.offset() != 0 ||
      cudf::strings_column_view(column_to_compress).offsets().size() !=
        column_to_compress.size() + 1) {
    owned = copy_column_view(column_to_compress, stream, mr);
    src   = owned->view();
  }

  cudf::strings_column_view scv(src);
  auto const n = src.size();

  // offsets: owned copy, INT32 for normal strings or INT64 for cudf "large strings"
  // (chars > 2GB); copy_column_view preserves the child's type.
  auto offsets = copy_column_view(scv.offsets(), stream, mr);

  // chars: a fixed-width column caps at 2^31 ELEMENTS, so >2GB chars can't be UINT8.
  // Widen the element type (bytes = elements x sizeof) to fit under the cap; byte
  // codecs are type-agnostic and decompress reads the raw buffer back via offsets.
  std::int64_t const chars_bytes = scv.chars_size(stream);
  std::int64_t const kElemCap    = std::numeric_limits<cudf::size_type>::max();
  cudf::type_id chars_tid        = cudf::type_id::UINT8;
  std::int64_t bpe               = 1;
  if (chars_bytes > kElemCap) { chars_tid = cudf::type_id::UINT32, bpe = 4; }
  if (chars_bytes > kElemCap * 4) { chars_tid = cudf::type_id::UINT64, bpe = 8; }
  if (chars_bytes > kElemCap * 8) {
    throw std::runtime_error("str_split_compressor: chars > 16GB out of scope");
  }
  auto const nelem = static_cast<cudf::size_type>((chars_bytes + bpe - 1) / bpe);
  auto chars       = cudf::make_fixed_width_column(
    cudf::data_type(chars_tid), nelem, cudf::mask_state::UNALLOCATED, stream, mr);
  if (chars_bytes > 0) {
    auto* dst = chars->mutable_view().head<std::uint8_t>();
    cudaMemcpyAsync(dst,
                    scv.chars_begin(stream),
                    static_cast<size_t>(chars_bytes),
                    cudaMemcpyDeviceToDevice,
                    stream.value());
    std::int64_t const padded = static_cast<std::int64_t>(nelem) * bpe;
    if (padded > chars_bytes) {
      cudaMemsetAsync(
        dst + chars_bytes, 0, static_cast<size_t>(padded - chars_bytes), stream.value());
    }
  }

  // null_mask: owned UINT8 bitmask bytes, copied only if the column has nulls.
  // A non-null column gets NO mask channel (2-channel str_split).
  std::unique_ptr<cudf::column> null_mask;
  if (src.null_count() > 0) {
    rmm::device_buffer mbuf = cudf::copy_bitmask(src, stream, mr);
    auto const mbytes       = static_cast<cudf::size_type>(cudf::bitmask_allocation_size_bytes(n));
    null_mask               = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::UINT8}, mbytes, std::move(mbuf), rmm::device_buffer{}, 0);
  }
  cudaStreamSynchronize(stream.value());

  return std::make_unique<str_split_compressed_representation>(
    n, std::move(offsets), std::move(chars), std::move(null_mask));
}

}  // namespace simpatico
