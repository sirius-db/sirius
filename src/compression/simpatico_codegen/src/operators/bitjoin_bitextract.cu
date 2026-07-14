/**
 * bitextract / bitjoin operators: split a packed fixed-width column into
 * named bit-field sub-columns, or re-join them back.
 *
 * bitextract: 1 input buffer → N output buffers (MSB-first bit fields)
 * bitjoin   : N input buffers → 1 output buffer (multi-input plan step only)
 *
 * Supported input widths: 8, 16, 32, 64 bits.
 * Each field output type is the smallest UINT that fits the field's bit count.
 * FLOAT32/FLOAT64 inputs are reinterpreted as UINT32/UINT64 bit patterns.
 */

#include "codegen/plan/representation.hpp"
#include "codegen/util/cuda_check.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace simpatico {

// ── CUDA kernels ─────────────────────────────────────────────────────────────

/// Extract a single bit-field from each element of an input array.
/// lo_bit: position of the least-significant bit of the field (0 = LSB of element).
/// mask: bitmask of width n_bits (applied after shift).
template <typename TIn, typename TOut>
__global__ void bitextract_field_kernel(
  const TIn* __restrict__ input, TOut* __restrict__ output, int lo_bit, TIn mask, int64_t n)
{
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    TOut val    = static_cast<TOut>((input[idx] >> lo_bit) & mask);
    output[idx] = val;
  }
}

/// OR a single bit-field from input into output.
/// Reads bits [src_lo, src_lo + n_bits) of input, OR-writes them at [dst_lo, dst_lo + n_bits) of
/// output. output must be zero-initialised before the first call.
template <typename TOut, typename TIn>
__global__ void bitjoin_field_kernel(TOut* __restrict__ output,
                                     const TIn* __restrict__ input,
                                     int src_lo,
                                     int dst_lo,
                                     TOut field_mask,
                                     int64_t n)
{
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    TIn shifted_in = input[idx] >> src_lo;
    TOut field     = static_cast<TOut>(shifted_in) & field_mask;
    output[idx] |= field << dst_lo;
  }
}

/// Sets bit (1u << flag_bit) in *flag if any element has non-zero bits outside `selected_mask`.
/// Used by bitjoin to warn when compression would be lossy.
template <typename TIn>
__global__ void check_truncation_kernel(const TIn* __restrict__ input,
                                        TIn selected_mask,
                                        uint32_t* __restrict__ flag,
                                        uint32_t flag_bit,
                                        int64_t n)
{
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    if ((input[idx] & ~selected_mask) != static_cast<TIn>(0)) { atomicOr(flag, flag_bit); }
  }
}

// ── dispatch helpers ──────────────────────────────────────────────────────────

static constexpr int kBlockSize = 256;

inline int64_t grid_size(int64_t n) { return (n + kBlockSize - 1) / kBlockSize; }

void launch_bitextract_field(cudf::column_view const& input_col,
                             cudf::mutable_column_view const& output_col,
                             int lo_bit,
                             uint32_t n_bits,
                             cudaStream_t stream)
{
  int64_t n = static_cast<int64_t>(input_col.size());
  if (n == 0) return;

  // Dispatch on output type (field type)
  auto dispatch_out = [&](auto* out_ptr) {
    using TOut = std::remove_pointer_t<decltype(out_ptr)>;
    TOut mask  = (n_bits >= sizeof(TOut) * 8) ? static_cast<TOut>(~TOut{0})
                                              : static_cast<TOut>((TOut{1} << n_bits) - 1);

    // Dispatch on input type (reinterpret as unsigned)
    auto id = input_col.type().id();
    if (id == cudf::type_id::UINT8 || id == cudf::type_id::INT8) {
      const uint8_t* in = input_col.head<uint8_t>();
      bitextract_field_kernel<uint8_t, TOut><<<grid_size(n), kBlockSize, 0, stream>>>(
        in, out_ptr, lo_bit, static_cast<uint8_t>(mask), n);
      throw_if_cuda_error(cudaGetLastError(), "bitextract_field UINT8 launch");
    } else if (id == cudf::type_id::UINT16 || id == cudf::type_id::INT16) {
      const uint16_t* in = input_col.head<uint16_t>();
      bitextract_field_kernel<uint16_t, TOut><<<grid_size(n), kBlockSize, 0, stream>>>(
        in, out_ptr, lo_bit, static_cast<uint16_t>(mask), n);
      throw_if_cuda_error(cudaGetLastError(), "bitextract_field UINT16 launch");
    } else if (id == cudf::type_id::UINT32 || id == cudf::type_id::INT32 ||
               id == cudf::type_id::FLOAT32) {
      const uint32_t* in = input_col.head<uint32_t>();
      bitextract_field_kernel<uint32_t, TOut><<<grid_size(n), kBlockSize, 0, stream>>>(
        in, out_ptr, lo_bit, static_cast<uint32_t>(mask), n);
      throw_if_cuda_error(cudaGetLastError(), "bitextract_field UINT32 launch");
    } else if (id == cudf::type_id::UINT64 || id == cudf::type_id::INT64 ||
               id == cudf::type_id::FLOAT64) {
      const uint64_t* in = input_col.head<uint64_t>();
      bitextract_field_kernel<uint64_t, TOut><<<grid_size(n), kBlockSize, 0, stream>>>(
        in, out_ptr, lo_bit, static_cast<uint64_t>(mask), n);
      throw_if_cuda_error(cudaGetLastError(), "bitextract_field UINT64 launch");
    } else {
      throw std::invalid_argument("bitextract_field: unsupported input type");
    }
  };

  auto out_id = output_col.type().id();
  if (out_id == cudf::type_id::UINT8) {
    dispatch_out(output_col.head<uint8_t>());
  } else if (out_id == cudf::type_id::UINT16) {
    dispatch_out(output_col.head<uint16_t>());
  } else if (out_id == cudf::type_id::UINT32) {
    dispatch_out(output_col.head<uint32_t>());
  } else if (out_id == cudf::type_id::UINT64) {
    dispatch_out(output_col.head<uint64_t>());
  } else {
    throw std::invalid_argument("bitextract_field: unsupported output type");
  }
}

void launch_bitjoin_field(cudf::mutable_column_view const& output_col,
                          cudf::column_view const& input_col,
                          int src_lo_bit,
                          int dst_lo_bit,
                          uint32_t n_bits,
                          cudaStream_t stream)
{
  int64_t n = static_cast<int64_t>(input_col.size());
  if (output_col.size() != input_col.size()) {
    throw std::invalid_argument("bitjoin_field: input and output row counts differ");
  }
  auto const src_width_bits = static_cast<uint32_t>(cudf::size_of(input_col.type())) * 8;
  auto const dst_width_bits = static_cast<uint32_t>(cudf::size_of(output_col.type())) * 8;
  if (src_lo_bit < 0 || dst_lo_bit < 0 || n_bits == 0 || n_bits > src_width_bits ||
      n_bits > dst_width_bits || static_cast<uint32_t>(src_lo_bit) > src_width_bits - n_bits ||
      static_cast<uint32_t>(dst_lo_bit) > dst_width_bits - n_bits) {
    throw std::invalid_argument("bitjoin_field: bit range exceeds input or output type width");
  }
  if (n == 0) return;

  // Dispatch on output (packed) type
  auto dispatch_in = [&](auto* out_ptr) {
    using TOut      = std::remove_pointer_t<decltype(out_ptr)>;
    TOut field_mask = (n_bits >= sizeof(TOut) * 8) ? static_cast<TOut>(~TOut{0})
                                                   : static_cast<TOut>((TOut{1} << n_bits) - 1);

    auto id = input_col.type().id();
    if (id == cudf::type_id::UINT8 || id == cudf::type_id::INT8) {
      const uint8_t* in = input_col.head<uint8_t>();
      bitjoin_field_kernel<TOut, uint8_t><<<grid_size(n), kBlockSize, 0, stream>>>(
        out_ptr, in, src_lo_bit, dst_lo_bit, field_mask, n);
      throw_if_cuda_error(cudaGetLastError(), "bitjoin_field UINT8 launch");
    } else if (id == cudf::type_id::UINT16 || id == cudf::type_id::INT16) {
      const uint16_t* in = input_col.head<uint16_t>();
      bitjoin_field_kernel<TOut, uint16_t><<<grid_size(n), kBlockSize, 0, stream>>>(
        out_ptr, in, src_lo_bit, dst_lo_bit, field_mask, n);
      throw_if_cuda_error(cudaGetLastError(), "bitjoin_field UINT16 launch");
    } else if (id == cudf::type_id::UINT32 || id == cudf::type_id::INT32 ||
               id == cudf::type_id::FLOAT32) {
      const uint32_t* in = input_col.head<uint32_t>();
      bitjoin_field_kernel<TOut, uint32_t><<<grid_size(n), kBlockSize, 0, stream>>>(
        out_ptr, in, src_lo_bit, dst_lo_bit, field_mask, n);
      throw_if_cuda_error(cudaGetLastError(), "bitjoin_field UINT32 launch");
    } else if (id == cudf::type_id::UINT64 || id == cudf::type_id::INT64 ||
               id == cudf::type_id::FLOAT64) {
      const uint64_t* in = input_col.head<uint64_t>();
      bitjoin_field_kernel<TOut, uint64_t><<<grid_size(n), kBlockSize, 0, stream>>>(
        out_ptr, in, src_lo_bit, dst_lo_bit, field_mask, n);
      throw_if_cuda_error(cudaGetLastError(), "bitjoin_field UINT64 launch");
    } else {
      throw std::invalid_argument("bitjoin_field: unsupported input type");
    }
  };

  auto out_id = output_col.type().id();
  if (out_id == cudf::type_id::UINT8 || out_id == cudf::type_id::INT8) {
    dispatch_in(output_col.head<uint8_t>());
  } else if (out_id == cudf::type_id::UINT16 || out_id == cudf::type_id::INT16) {
    dispatch_in(output_col.head<uint16_t>());
  } else if (out_id == cudf::type_id::UINT32 || out_id == cudf::type_id::INT32 ||
             out_id == cudf::type_id::FLOAT32) {
    dispatch_in(output_col.head<uint32_t>());
  } else if (out_id == cudf::type_id::UINT64 || out_id == cudf::type_id::INT64 ||
             out_id == cudf::type_id::FLOAT64) {
    dispatch_in(output_col.head<uint64_t>());
  } else {
    throw std::invalid_argument("bitjoin_field: unsupported output type");
  }
}

void launch_check_truncation(cudf::column_view const& input_col,
                             uint64_t selected_mask,
                             uint32_t* d_flag,
                             uint32_t flag_bit,
                             cudaStream_t stream)
{
  int64_t n = static_cast<int64_t>(input_col.size());
  if (n == 0) return;
  auto id = input_col.type().id();
  if (id == cudf::type_id::UINT8 || id == cudf::type_id::INT8) {
    check_truncation_kernel<uint8_t><<<grid_size(n), kBlockSize, 0, stream>>>(
      input_col.head<uint8_t>(), static_cast<uint8_t>(selected_mask), d_flag, flag_bit, n);
  } else if (id == cudf::type_id::UINT16 || id == cudf::type_id::INT16) {
    check_truncation_kernel<uint16_t><<<grid_size(n), kBlockSize, 0, stream>>>(
      input_col.head<uint16_t>(), static_cast<uint16_t>(selected_mask), d_flag, flag_bit, n);
  } else if (id == cudf::type_id::UINT32 || id == cudf::type_id::INT32 ||
             id == cudf::type_id::FLOAT32) {
    check_truncation_kernel<uint32_t><<<grid_size(n), kBlockSize, 0, stream>>>(
      input_col.head<uint32_t>(), static_cast<uint32_t>(selected_mask), d_flag, flag_bit, n);
  } else if (id == cudf::type_id::UINT64 || id == cudf::type_id::INT64 ||
             id == cudf::type_id::FLOAT64) {
    check_truncation_kernel<uint64_t><<<grid_size(n), kBlockSize, 0, stream>>>(
      input_col.head<uint64_t>(), static_cast<uint64_t>(selected_mask), d_flag, flag_bit, n);
  } else {
    throw std::invalid_argument("check_truncation: unsupported input type");
  }
}

// ── bitextract_compressed_representation::decompress ─────────────────────────

std::unique_ptr<cudf::column> bitextract_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  if (fields.empty()) {
    throw std::invalid_argument("bitextract decompress: no field columns stored");
  }

  int64_t n                = static_cast<int64_t>(fields[0]->size());
  cudf::data_type out_type = spec.output_type;

  // Allocate output column
  auto out_col = cudf::make_fixed_width_column(
    out_type, static_cast<cudf::size_type>(n), cudf::mask_state::UNALLOCATED, stream, mr);

  // Zero-initialise
  cudaMemsetAsync(out_col->mutable_view().head<void>(),
                  0,
                  static_cast<size_t>(n) * static_cast<size_t>(cudf::size_of(out_type)),
                  stream.value());

  // Compute total output width in bits
  uint32_t out_width_bits = static_cast<uint32_t>(cudf::size_of(out_type)) * 8;

  // OR each field back in (MSB-first)
  uint32_t offset_from_msb = 0;
  for (size_t fi = 0; fi < spec.fields.size(); ++fi) {
    if (!fields[fi]) continue;
    auto const& field_spec = spec.fields[fi];
    int lo_bit             = static_cast<int>(out_width_bits - offset_from_msb - field_spec.bits);
    launch_bitjoin_field(out_col->mutable_view(),
                         fields[fi]->view(),
                         /*src_lo=*/0,
                         /*dst_lo=*/lo_bit,
                         field_spec.bits,
                         stream.value());
    offset_from_msb += field_spec.bits;
  }
  cudaStreamSynchronize(stream.value());
  throw_if_cuda_error(cudaGetLastError(), "bitextract decompress sync");
  return out_col;
}

// ── bitextract_compressor::compress ──────────────────────────────────────────

std::unique_ptr<compressed_representation> bitextract_compressor::compress(
  cudf::column_view column, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  if (spec.fields.empty()) { throw std::invalid_argument("bitextract compress: empty spec"); }

  // Validate total bits == input element width
  uint32_t input_bits = static_cast<uint32_t>(cudf::size_of(column.type())) * 8;
  uint32_t total_bits = 0;
  for (auto const& f : spec.fields)
    total_bits += f.bits;
  if (total_bits != input_bits) {
    throw std::invalid_argument("bitextract: field bit sum (" + std::to_string(total_bits) +
                                ") != input type width (" + std::to_string(input_bits) + ")");
  }

  int64_t n = static_cast<int64_t>(column.size());

  // For each field: allocate output column and extract
  std::vector<std::unique_ptr<cudf::column>> field_cols;
  field_cols.reserve(spec.fields.size());

  uint32_t offset_from_msb = 0;
  for (auto const& field : spec.fields) {
    int lo_bit = static_cast<int>(input_bits - offset_from_msb - field.bits);

    // Determine output type: smallest UINT that fits field.bits
    cudf::type_id field_type_id = (field.bits <= 8)    ? cudf::type_id::UINT8
                                  : (field.bits <= 16) ? cudf::type_id::UINT16
                                  : (field.bits <= 32) ? cudf::type_id::UINT32
                                                       : cudf::type_id::UINT64;

    auto field_col = cudf::make_fixed_width_column(cudf::data_type(field_type_id),
                                                   static_cast<cudf::size_type>(n),
                                                   cudf::mask_state::UNALLOCATED,
                                                   stream,
                                                   mr);

    launch_bitextract_field(column, field_col->mutable_view(), lo_bit, field.bits, stream.value());

    field_cols.push_back(std::move(field_col));
    offset_from_msb += field.bits;
  }

  cudaStreamSynchronize(stream.value());
  throw_if_cuda_error(cudaGetLastError(), "bitextract compress sync");

  // Build the spec result to store in the representation.
  // We use spec directly (already parsed at constructor time).
  // The output_type in spec is the inferred/aliased type (for float aliases it's FLOAT32/FLOAT64).
  // For decompress(), we need the original input type so the reconstructed column matches.
  // Store original type in output_type so decompress() allocates the right type.
  bitextract_spec_result repr_spec = spec;
  repr_spec.output_type            = column.type();

  return std::make_unique<bitextract_compressed_representation>(std::move(repr_spec),
                                                                std::move(field_cols));
}

}  // namespace simpatico
