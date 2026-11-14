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

#include "op/order/encoded_sort_order.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace sirius::op::order {

namespace {

constexpr std::uint64_t kSignBit = std::uint64_t{1} << 63;
constexpr int kMaxFieldsPerWord  = 8;

/// What a bit field of the encoded key holds.
enum class field_kind : std::uint8_t {
  null_flag,  ///< one bit: whether the row of a nullable column is null
  i8,
  i16,
  i32,
  i64,
  u8,
  u16,
  u32,
  u64,
  b8,
  f32,
  f64,
  d128_hi,    ///< upper (signed) half of a 128-bit integer
  d128_lo,    ///< lower (unsigned) half of a 128-bit integer
  str_chunk,  ///< 8 bytes of a string from `chunk`, big-endian, zero-padded past its end
  str_len     ///< byte length of a string
};

/// One bit field of the encoded key: an unsigned value of `bits` bits whose order over the rows
/// is the order of (a part of) one key column. Trivially copyable; used on the device.
struct field {
  const void* data{nullptr};   ///< element 0 of the column's values (strings: of its offsets)
  const char* chars{nullptr};  ///< strings: the character buffer
  const cudf::bitmask_type* mask{nullptr};  ///< nullptr when the column has no nulls
  std::uint64_t base{0};           ///< ascending: the column minimum; descending: its maximum
  std::int64_t chunk{0};           ///< str_chunk: byte offset of the chunk inside the string
  cudf::size_type mask_offset{0};  ///< bit of row 0 in `mask`
  field_kind kind{field_kind::i64};
  std::uint8_t bits{0};   ///< width of the field; inside a word: of the part the word holds
  std::uint8_t part{0};   ///< inside a word: the part is the field's bits [part, part + bits)
  std::uint8_t shift{0};  ///< inside a word: position of the part's lowest bit
  bool descending{false};
  bool null_bit{false};   ///< null_flag: the field's value for a null row
  bool offsets64{false};  ///< strings: the offsets are INT64
};

template <typename T>
__device__ inline T load(const void* data, cudf::size_type row)
{
  return static_cast<const T*>(data)[row];
}

template <typename T>
__device__ inline std::uint64_t signed_key(const void* data, cudf::size_type row)
{
  return static_cast<std::uint64_t>(static_cast<std::int64_t>(load<T>(data, row))) ^ kSignBit;
}

__device__ inline std::int64_t string_begin(field const& f, cudf::size_type row)
{
  return f.offsets64 ? load<std::int64_t>(f.data, row)
                     : static_cast<std::int64_t>(load<std::int32_t>(f.data, row));
}

/// Unsigned value of a valid row whose order is the column's ascending order under the cuDF row
/// comparator: NaN is the greatest value and equals NaN, -0.0 equals 0.0, strings compare by
/// unsigned bytes (a string's chunks then its length order it against every other string).
__device__ inline std::uint64_t ascending_key(field const& f, cudf::size_type row)
{
  switch (f.kind) {
    case field_kind::i8: return signed_key<std::int8_t>(f.data, row);
    case field_kind::i16: return signed_key<std::int16_t>(f.data, row);
    case field_kind::i32: return signed_key<std::int32_t>(f.data, row);
    case field_kind::i64: return signed_key<std::int64_t>(f.data, row);
    case field_kind::u8: return load<std::uint8_t>(f.data, row);
    case field_kind::u16: return load<std::uint16_t>(f.data, row);
    case field_kind::u32: return load<std::uint32_t>(f.data, row);
    case field_kind::u64: return load<std::uint64_t>(f.data, row);
    case field_kind::b8: return load<std::uint8_t>(f.data, row) != 0 ? 1 : 0;
    case field_kind::f32: {
      const float v = load<float>(f.data, row);
      if (v != v) { return 0xFFFFFFFFull; }
      std::uint32_t u;
      std::memcpy(&u, &v, sizeof(u));
      if ((u << 1) == 0) { u = 0; }  // -0.0 equals 0.0
      return (u & 0x80000000u) != 0 ? static_cast<std::uint32_t>(~u) : (u | 0x80000000u);
    }
    case field_kind::f64: {
      const double v = load<double>(f.data, row);
      if (v != v) { return ~std::uint64_t{0}; }
      std::uint64_t u;
      std::memcpy(&u, &v, sizeof(u));
      if ((u << 1) == 0) { u = 0; }  // -0.0 equals 0.0
      return (u & kSignBit) != 0 ? ~u : (u | kSignBit);
    }
    case field_kind::d128_hi:
      return static_cast<std::uint64_t>(
               static_cast<std::int64_t>(load<__int128_t>(f.data, row) >> 64)) ^
             kSignBit;
    case field_kind::d128_lo: return static_cast<std::uint64_t>(load<__int128_t>(f.data, row));
    case field_kind::str_chunk: {
      const std::int64_t begin = string_begin(f, row);
      const std::int64_t rest  = string_begin(f, row + 1) - begin - f.chunk;
      if (rest <= 0) { return 0; }
      const auto* p   = reinterpret_cast<const unsigned char*>(f.chars) + begin + f.chunk;
      const int count = rest < 8 ? static_cast<int>(rest) : 8;
      std::uint64_t w = 0;
      for (int i = 0; i < count; ++i) {
        w |= static_cast<std::uint64_t>(p[i]) << (56 - 8 * i);
      }
      return w;
    }
    case field_kind::str_len:
      return static_cast<std::uint64_t>(string_begin(f, row + 1) - string_begin(f, row));
    case field_kind::null_flag: return 0;
  }
  return 0;
}

/// The field's value for @p row: in [0, 2^bits), ordered as the key column orders the rows. A
/// null row has value 0 in every field of its column but the null flag, so nulls tie.
__device__ inline std::uint64_t field_value(field const& f, cudf::size_type row)
{
  const bool valid = f.mask == nullptr || cudf::bit_is_set(f.mask, row + f.mask_offset);
  if (f.kind == field_kind::null_flag) { return valid != f.null_bit ? 1 : 0; }
  if (!valid) { return 0; }
  const std::uint64_t v = ascending_key(f, row);
  return f.descending ? f.base - v : v - f.base;
}

/// Closed range of the ascending keys of a field over the valid rows; empty when lo > hi.
struct key_range {
  std::uint64_t lo{~std::uint64_t{0}};
  std::uint64_t hi{0};
};

struct key_range_fn {
  field f;
  __device__ key_range operator()(cudf::size_type row) const
  {
    if (f.mask != nullptr && !cudf::bit_is_set(f.mask, row + f.mask_offset)) { return {}; }
    const std::uint64_t v = ascending_key(f, row);
    return {v, v};
  }
};

struct key_range_merge_fn {
  __device__ key_range operator()(key_range const& a, key_range const& b) const
  {
    return {a.lo < b.lo ? a.lo : b.lo, a.hi > b.hi ? a.hi : b.hi};
  }
};

/// One 64-bit word of the encoded key: up to kMaxFieldsPerWord fields or parts of fields, packed.
/// `rows` maps the index to the table row (nullptr: the index is the row).
struct word_fn {
  field fields[kMaxFieldsPerWord];
  int count{0};
  const cudf::size_type* rows{nullptr};

  __device__ std::uint64_t operator()(cudf::size_type k) const
  {
    const cudf::size_type row = rows != nullptr ? rows[k] : k;
    std::uint64_t w           = 0;
    for (int i = 0; i < count; ++i) {
      std::uint64_t v = field_value(fields[i], row) >> fields[i].part;
      if (fields[i].bits < 64) { v &= (std::uint64_t{1} << fields[i].bits) - 1; }
      w |= v << fields[i].shift;
    }
    return w;
  }
};

/// Bits in which a row's word differs from the word of the first row of its tie group.
struct group_diff_fn {
  const std::uint64_t* words;
  const cudf::size_type* head;  ///< index of the first row of the row's group; nullptr: one group
  __device__ std::uint64_t operator()(cudf::size_type k) const
  {
    return words[k] ^ words[head != nullptr ? head[k] : 0];
  }
};

/// Sort key of a tied row: (group, the word's bits [lo, lo + wbits)).
template <typename K>
struct group_word_key_fn {
  const std::uint64_t* words;
  const std::uint32_t* group;  ///< 1-based group number; nullptr: one group
  int lo;
  int wbits;
  __device__ K operator()(cudf::size_type k) const
  {
    std::uint64_t x = words[k] >> lo;
    if (wbits < 64) { x &= (std::uint64_t{1} << wbits) - 1; }
    if (group != nullptr) { x |= static_cast<std::uint64_t>(group[k] - 1) << wbits; }
    return static_cast<K>(x);
  }
};

/// Whether sorted position j starts a group: its key differs from the previous one.
template <typename K>
struct head_of_keys_fn {
  const K* keys;
  __device__ std::uint8_t operator()(cudf::size_type j) const
  {
    return (j == 0 || keys[j] != keys[j - 1]) ? 1 : 0;
  }
};

/// A row stays tied when its group has more than one row.
struct stays_tied_fn {
  const std::uint8_t* head;
  cudf::size_type count;
  __device__ bool operator()(cudf::size_type j) const
  {
    return !(head[j] != 0 && (j + 1 == count || head[j + 1] != 0));
  }
};

/// Writes the rows that became unique to their final position.
struct place_unique_fn {
  stays_tied_fn tied;
  const cudf::size_type* position;  ///< nullptr: sorted position j is the final position
  const cudf::size_type* row;
  cudf::size_type* out;
  __device__ void operator()(cudf::size_type j) const
  {
    if (!tied(j)) { out[position != nullptr ? position[j] : j] = row[j]; }
  }
};

struct head_index_fn {
  const std::uint8_t* head;
  __device__ cudf::size_type operator()(cudf::size_type k) const { return head[k] != 0 ? k : 0; }
};

struct head_count_fn {
  __device__ std::uint32_t operator()(std::uint8_t head) const { return head; }
};

void check_cuda(cudaError_t status, const char* what)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("encoded_sorted_order: ") + what + ": " +
                             cudaGetErrorString(status));
  }
}

/// Stable LSD radix sort of (keys, values) by key bits [0, end_bit).
template <typename K>
void radix_sort_pairs(rmm::device_uvector<K>& keys,
                      rmm::device_uvector<cudf::size_type>& values,
                      int end_bit,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  const auto count = static_cast<std::int64_t>(keys.size());
  rmm::device_uvector<K> keys_alt(keys.size(), stream, mr);
  rmm::device_uvector<cudf::size_type> values_alt(values.size(), stream, mr);
  cub::DoubleBuffer<K> key_buffers(keys.data(), keys_alt.data());
  cub::DoubleBuffer<cudf::size_type> value_buffers(values.data(), values_alt.data());
  std::size_t temp_bytes = 0;
  check_cuda(cub::DeviceRadixSort::SortPairs(
               nullptr, temp_bytes, key_buffers, value_buffers, count, 0, end_bit, stream.value()),
             "radix sort sizing");
  rmm::device_buffer temp(temp_bytes, stream, mr);
  check_cuda(
    cub::DeviceRadixSort::SortPairs(
      temp.data(), temp_bytes, key_buffers, value_buffers, count, 0, end_bit, stream.value()),
    "radix sort");
  if (key_buffers.selector != 0) { std::swap(keys, keys_alt); }
  if (value_buffers.selector != 0) { std::swap(values, values_alt); }
}

const void* values_begin(cudf::column_view const& col)
{
  return col.head<std::uint8_t>() +
         static_cast<std::size_t>(col.offset()) * cudf::size_of(col.type());
}

std::optional<field_kind> fixed_width_kind(cudf::type_id id)
{
  switch (id) {
    case cudf::type_id::INT8: return field_kind::i8;
    case cudf::type_id::INT16: return field_kind::i16;
    case cudf::type_id::INT32:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::DURATION_DAYS:
    case cudf::type_id::DECIMAL32: return field_kind::i32;
    case cudf::type_id::INT64:
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
    case cudf::type_id::DURATION_SECONDS:
    case cudf::type_id::DURATION_MILLISECONDS:
    case cudf::type_id::DURATION_MICROSECONDS:
    case cudf::type_id::DURATION_NANOSECONDS:
    case cudf::type_id::DECIMAL64: return field_kind::i64;
    case cudf::type_id::UINT8: return field_kind::u8;
    case cudf::type_id::UINT16: return field_kind::u16;
    case cudf::type_id::UINT32: return field_kind::u32;
    case cudf::type_id::UINT64: return field_kind::u64;
    case cudf::type_id::BOOL8: return field_kind::b8;
    case cudf::type_id::FLOAT32: return field_kind::f32;
    case cudf::type_id::FLOAT64: return field_kind::f64;
    default: return std::nullopt;
  }
}

/// The fields of all key columns in comparison order, or nullopt when a column is not supported.
/// Fields that are constant over the valid rows are left out. Shifts are not set yet.
std::optional<std::vector<field>> build_fields(cudf::table_view const& keys,
                                               std::vector<cudf::order> const& column_order,
                                               std::vector<cudf::null_order> const& null_precedence,
                                               encoded_sort_options const& options,
                                               rmm::cuda_stream_view stream)
{
  const auto rows     = keys.num_rows();
  const auto policy   = rmm::exec_policy_nosync(stream);
  const auto range_of = [&](field const& f) {
    return thrust::transform_reduce(policy,
                                    thrust::make_counting_iterator<cudf::size_type>(0),
                                    thrust::make_counting_iterator<cudf::size_type>(rows),
                                    key_range_fn{f},
                                    key_range{},
                                    key_range_merge_fn{});
  };

  std::vector<field> fields;
  // Appends a value field spanning its range over the valid rows; nothing when it is constant.
  const auto add_ranged = [&](field f) {
    const auto range = range_of(f);
    if (range.lo >= range.hi) { return; }
    f.bits = static_cast<std::uint8_t>(std::bit_width(range.hi - range.lo));
    f.base = f.descending ? range.hi : range.lo;
    fields.push_back(f);
  };

  for (cudf::size_type c = 0; c < keys.num_columns(); ++c) {
    const auto& col       = keys.column(c);
    const bool descending = column_order[c] == cudf::order::DESCENDING;
    field proto;
    proto.descending  = descending;
    proto.mask        = col.has_nulls() ? col.null_mask() : nullptr;
    proto.mask_offset = col.offset();

    if (col.has_nulls() && col.null_count() < rows) {
      field flag = proto;
      flag.kind  = field_kind::null_flag;
      flag.bits  = 1;
      // Ascending frame: a null is 0 for BEFORE and 1 for AFTER; DESCENDING reverses it.
      flag.null_bit = (null_precedence[c] == cudf::null_order::AFTER) != descending;
      fields.push_back(flag);
    }
    if (col.null_count() == rows) { continue; }

    const auto id = col.type().id();
    if (const auto kind = fixed_width_kind(id)) {
      proto.data = values_begin(col);
      proto.kind = *kind;
      add_ranged(proto);
    } else if (id == cudf::type_id::DECIMAL128) {
      proto.data = values_begin(col);
      proto.kind = field_kind::d128_hi;
      add_ranged(proto);
      proto.kind = field_kind::d128_lo;
      add_ranged(proto);
    } else if (id == cudf::type_id::STRING) {
      if (col.num_children() == 0) { return std::nullopt; }
      const cudf::strings_column_view strings(col);
      const auto offsets = strings.offsets();
      proto.offsets64    = offsets.type().id() == cudf::type_id::INT64;
      proto.data =
        offsets.head<std::uint8_t>() +
        (static_cast<std::size_t>(offsets.offset()) + col.offset()) * cudf::size_of(offsets.type());
      proto.chars        = strings.chars_begin(stream);
      proto.kind         = field_kind::str_len;
      const auto lengths = range_of(proto);
      if (lengths.lo > lengths.hi) { continue; }
      if (lengths.hi > static_cast<std::uint64_t>(options.max_string_bytes)) {
        return std::nullopt;
      }
      for (std::uint64_t at = 0; at < lengths.hi; at += 8) {
        field chunk = proto;
        chunk.kind  = field_kind::str_chunk;
        chunk.chunk = static_cast<std::int64_t>(at);
        chunk.bits  = 64;
        chunk.base  = descending ? ~std::uint64_t{0} : 0;
        fields.push_back(chunk);
      }
      if (lengths.hi > lengths.lo) {
        proto.bits = static_cast<std::uint8_t>(std::bit_width(lengths.hi - lengths.lo));
        proto.base = descending ? lengths.hi : lengths.lo;
        fields.push_back(proto);
      }
    } else {
      return std::nullopt;
    }
  }
  return fields;
}

/// Packs the fields into 64-bit words as one bit string, most significant field first: a field
/// that does not fit the rest of a word continues in the next one, so every word but the last is
/// full and the first sort separates as many rows as 64 bits can.
std::vector<word_fn> pack_words(std::vector<field> const& fields)
{
  std::vector<word_fn> words;
  int used = 64;
  for (auto const& f : fields) {
    int remaining = f.bits;
    while (remaining > 0) {
      if (used == 64 || words.back().count == kMaxFieldsPerWord) {
        words.emplace_back();
        used = 0;
      }
      const int take = remaining < 64 - used ? remaining : 64 - used;
      field part     = f;
      part.bits      = static_cast<std::uint8_t>(take);
      part.part      = static_cast<std::uint8_t>(remaining - take);
      part.shift     = static_cast<std::uint8_t>(64 - used - take);
      used += take;
      remaining -= take;
      auto& w             = words.back();
      w.fields[w.count++] = part;
    }
  }
  return words;
}

/// Whether cuDF sorts this key shape with its own radix path.
bool is_cudf_radix_shape(cudf::table_view const& keys)
{
  return keys.num_columns() == 1 && fixed_width_kind(keys.column(0).type().id()).has_value();
}

/// The rows still tied on every word applied so far, in their current order. Before the first
/// sort (`initial`) they are all rows in input order, in one group, and the vectors are empty.
struct tied_rows {
  bool initial{true};
  cudf::size_type count{0};
  std::uint32_t groups{1};
  rmm::device_uvector<cudf::size_type> position;  ///< final position of the row
  rmm::device_uvector<cudf::size_type> row;       ///< table row
  rmm::device_uvector<cudf::size_type> head;      ///< index (here) of the first row of its group
  rmm::device_uvector<std::uint32_t> group;       ///< 1-based group number, ascending

  tied_rows(cudf::size_type rows, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
    : count(rows),
      position(0, stream, mr),
      row(0, stream, mr),
      head(0, stream, mr),
      group(0, stream, mr)
  {
  }
};

/// After a sort: `sorted_rows` are the tied rows in their new order and `head` flags the rows
/// that start a group. Places the rows that became unique in @p out and keeps the others.
void regroup(tied_rows& tied,
             rmm::device_uvector<cudf::size_type>&& sorted_rows,
             rmm::device_uvector<std::uint8_t> const& head,
             cudf::size_type* out,
             rmm::cuda_stream_view stream,
             rmm::device_async_resource_ref mr)
{
  const auto policy = rmm::exec_policy_nosync(stream, mr);
  const auto first  = thrust::make_counting_iterator<cudf::size_type>(0);
  const auto last   = first + tied.count;
  const stays_tied_fn stays{head.data(), tied.count};
  const cudf::size_type* position = tied.initial ? nullptr : tied.position.data();

  thrust::for_each(policy, first, last, place_unique_fn{stays, position, sorted_rows.data(), out});
  const auto kept = static_cast<cudf::size_type>(thrust::count_if(policy, first, last, stays));

  rmm::device_uvector<cudf::size_type> new_position(kept, stream, mr);
  rmm::device_uvector<cudf::size_type> new_row(kept, stream, mr);
  rmm::device_uvector<std::uint8_t> new_head(kept, stream, mr);
  if (kept > 0) {
    const auto output =
      thrust::make_zip_iterator(new_position.begin(), new_row.begin(), new_head.begin());
    if (tied.initial) {
      thrust::copy_if(policy,
                      thrust::make_zip_iterator(first, sorted_rows.begin(), head.begin()),
                      thrust::make_zip_iterator(last, sorted_rows.end(), head.end()),
                      first,
                      output,
                      stays);
    } else {
      thrust::copy_if(
        policy,
        thrust::make_zip_iterator(tied.position.begin(), sorted_rows.begin(), head.begin()),
        thrust::make_zip_iterator(tied.position.end(), sorted_rows.end(), head.end()),
        first,
        output,
        stays);
    }
  }
  sorted_rows = rmm::device_uvector<cudf::size_type>(0, stream, mr);

  tied.initial  = false;
  tied.count    = kept;
  tied.position = std::move(new_position);
  tied.row      = std::move(new_row);
  tied.head     = rmm::device_uvector<cudf::size_type>(kept, stream, mr);
  tied.group    = rmm::device_uvector<std::uint32_t>(kept, stream, mr);
  tied.groups   = 0;
  if (kept == 0) { return; }
  const auto kept_last = first + kept;
  thrust::inclusive_scan(policy,
                         thrust::make_transform_iterator(new_head.begin(), head_count_fn{}),
                         thrust::make_transform_iterator(new_head.end(), head_count_fn{}),
                         tied.group.begin());
  thrust::inclusive_scan(policy,
                         thrust::make_transform_iterator(first, head_index_fn{new_head.data()}),
                         thrust::make_transform_iterator(kept_last, head_index_fn{new_head.data()}),
                         tied.head.begin(),
                         thrust::maximum<cudf::size_type>{});
  tied.groups = tied.group.back_element(stream);
}

/// The sort keys of the tied rows, made from their words; frees @p words. 64-bit keys reuse the
/// words' buffer.
template <typename K>
rmm::device_uvector<K> make_keys(rmm::device_uvector<std::uint64_t>&& words,
                                 group_word_key_fn<K> const& key_of,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  const auto policy = rmm::exec_policy_nosync(stream, mr);
  const auto first  = thrust::make_counting_iterator<cudf::size_type>(0);
  const auto last   = first + static_cast<cudf::size_type>(words.size());
  if constexpr (std::is_same_v<K, std::uint64_t>) {
    // In place: key k reads only word k.
    thrust::transform(policy, first, last, words.begin(), key_of);
    return std::move(words);
  } else {
    rmm::device_uvector<K> keys(words.size(), stream, mr);
    thrust::transform(policy, first, last, keys.begin(), key_of);
    words = rmm::device_uvector<std::uint64_t>(0, stream, mr);
    return keys;
  }
}

/// One refinement step: a stable radix sort of the tied rows by (group, word bits [lo, lo + wbits))
/// packed into one key of type K, then regrouping. Consumes @p words and the group numbers and
/// row list of @p tied, so that the sort's two buffers per array are all that is held.
template <typename K>
void sort_packed(tied_rows& tied,
                 rmm::device_uvector<std::uint64_t>&& words,
                 int lo,
                 int wbits,
                 int gbits,
                 cudf::size_type* out,
                 rmm::cuda_stream_view stream,
                 rmm::device_async_resource_ref mr)
{
  const auto policy = rmm::exec_policy_nosync(stream, mr);
  const auto first  = thrust::make_counting_iterator<cudf::size_type>(0);
  const auto last   = first + tied.count;
  auto keys         = make_keys<K>(
    std::move(words),
    group_word_key_fn<K>{words.data(), gbits > 0 ? tied.group.data() : nullptr, lo, wbits},
    stream,
    mr);
  tied.group = rmm::device_uvector<std::uint32_t>(0, stream, mr);
  rmm::device_uvector<cudf::size_type> rows(0, stream, mr);
  if (tied.initial) {
    rows = rmm::device_uvector<cudf::size_type>(tied.count, stream, mr);
    thrust::sequence(policy, rows.begin(), rows.end(), 0);
  } else {
    rows = std::move(tied.row);
  }
  radix_sort_pairs(keys, rows, gbits + wbits, stream, mr);
  rmm::device_uvector<std::uint8_t> head(tied.count, stream, mr);
  thrust::transform(policy, first, last, head.begin(), head_of_keys_fn<K>{keys.data()});
  keys = rmm::device_uvector<K>(0, stream, mr);
  regroup(tied, std::move(rows), head, out, stream, mr);
}

}  // namespace

std::unique_ptr<cudf::column> encoded_sorted_order(
  cudf::table_view const& keys,
  std::vector<cudf::order> const& column_order,
  std::vector<cudf::null_order> const& null_precedence,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  encoded_sort_options const& options,
  encoded_sort_stats* stats)
{
  const auto rows = keys.num_rows();
  if (keys.num_columns() == 0 || rows < 2 || rows < options.min_rows) { return nullptr; }
  if (options.max_working_bytes != 0 &&
      encoded_sort_working_bytes(rows) > options.max_working_bytes) {
    return nullptr;
  }
  if (column_order.size() != static_cast<std::size_t>(keys.num_columns()) ||
      null_precedence.size() != static_cast<std::size_t>(keys.num_columns())) {
    return nullptr;
  }
  if (is_cudf_radix_shape(keys)) { return nullptr; }
  for (auto const& col : keys) {
    const auto id = col.type().id();
    if (!fixed_width_kind(id) && id != cudf::type_id::DECIMAL128 && id != cudf::type_id::STRING) {
      return nullptr;
    }
  }

  const auto fields = build_fields(keys, column_order, null_precedence, options, stream);
  if (!fields) { return nullptr; }
  auto words = pack_words(*fields);

  // The result and all temporaries come from mr: the caller's memory space sees, and its
  // reservation covers, everything this function holds.
  const auto temp_mr = mr;
  const auto policy  = rmm::exec_policy_nosync(stream, temp_mr);
  const auto first   = thrust::make_counting_iterator<cudf::size_type>(0);
  rmm::device_uvector<cudf::size_type> result(rows, stream, mr);
  tied_rows tied(rows, stream, temp_mr);
  encoded_sort_stats local;
  local.words = static_cast<int>(words.size());

  for (auto& word : words) {
    // Usually one round per word. When the tie group number and the word's differing bits do not
    // fit one 64-bit key, a round sorts by the highest bits that fit and the next round applies
    // the rest of the word to the rows still tied.
    for (bool first_round = true; tied.count > 0; first_round = false) {
      word.rows = tied.initial ? nullptr : tied.row.data();
      rmm::device_uvector<std::uint64_t> values(tied.count, stream, temp_mr);
      thrust::transform(policy, first, first + tied.count, values.begin(), word);
      const std::uint64_t diff = thrust::transform_reduce(
        policy,
        first,
        first + tied.count,
        group_diff_fn{values.data(), tied.initial ? nullptr : tied.head.data()},
        std::uint64_t{0},
        thrust::bit_or<std::uint64_t>{});
      if (diff == 0) {
        if (first_round) { ++local.words_skipped; }
        break;
      }
      // The group heads were only needed for diff; regrouping rebuilds them.
      tied.head       = rmm::device_uvector<cudf::size_type>(0, stream, temp_mr);
      const int hi    = 64 - std::countl_zero(diff);
      const int wbits = hi - std::countr_zero(diff);
      const int gbits = tied.groups > 1 ? std::bit_width(tied.groups - 1) : 0;
      const int take  = wbits < 64 - gbits ? wbits : 64 - gbits;
      ++local.radix_sorts;
      local.rows_sorted += tied.count;
      if (gbits + take <= 32) {
        sort_packed<std::uint32_t>(
          tied, std::move(values), hi - take, take, gbits, result.data(), stream, temp_mr);
      } else {
        sort_packed<std::uint64_t>(
          tied, std::move(values), hi - take, take, gbits, result.data(), stream, temp_mr);
      }
      if (take == wbits) { break; }
    }
    if (tied.count == 0) { break; }
  }

  if (tied.initial) {
    // No word separated any rows: all keys are equal and the input order stands.
    thrust::sequence(policy, result.begin(), result.end(), 0);
  } else if (tied.count > 0) {
    thrust::scatter(
      policy, tied.row.begin(), tied.row.end(), tied.position.begin(), result.begin());
  }
  if (stats != nullptr) { *stats = local; }
  return std::make_unique<cudf::column>(std::move(result), rmm::device_buffer{}, 0);
}

}  // namespace sirius::op::order
