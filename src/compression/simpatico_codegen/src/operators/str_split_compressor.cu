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
  if (num_rows == 0 || !offsets_) {
    return cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  }

  auto chars_contents = chars_->release();
  rmm::device_buffer chars_buf =
    chars_contents.data ? std::move(*chars_contents.data) : rmm::device_buffer(0, stream, mr);

  cudf::size_type nc = 0;
  rmm::device_buffer mask_buf(0, stream, mr);
  if (null_mask_) {
    auto const* bits =
      reinterpret_cast<cudf::bitmask_type const*>(null_mask_->view().data<std::uint8_t>());
    nc = cudf::null_count(bits, 0, num_rows, stream);
    if (nc > 0) {
      auto mask_contents = null_mask_->release();
      mask_buf           = std::move(*mask_contents.data);
    }
  }
  return cudf::make_strings_column(
    num_rows, std::move(offsets_), std::move(chars_buf), nc, std::move(mask_buf));
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
    cudaMemsetAsync(
      offsets->mutable_view().head<void>(), 0, sizeof(std::int32_t), stream.value());
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
  if (scv.chars_size(stream) >
      static_cast<std::int64_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw std::runtime_error("str_split_compressor: chars > 2GB out of scope");
  }

  // offsets: owned INT32 copy (size n+1).
  auto offsets = copy_column_view(scv.offsets(), stream, mr);

  // chars: owned UINT8 copy.
  auto const nc = static_cast<cudf::size_type>(scv.chars_size(stream));
  auto chars    = cudf::make_fixed_width_column(
    cudf::data_type(cudf::type_id::UINT8), nc, cudf::mask_state::UNALLOCATED, stream, mr);
  if (nc > 0) {
    cudaMemcpyAsync(chars->mutable_view().head<void>(),
                    scv.chars_begin(stream),
                    static_cast<size_t>(nc),
                    cudaMemcpyDeviceToDevice,
                    stream.value());
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

std::unique_ptr<cudf::column> str_split_compressor::decompress(
  compressed_representation const& data_to_decompress,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const* repr = dynamic_cast<str_split_compressed_representation const*>(&data_to_decompress);
  if (repr == nullptr) return nullptr;
  return repr->decompress(stream, mr);
}

}  // namespace simpatico
