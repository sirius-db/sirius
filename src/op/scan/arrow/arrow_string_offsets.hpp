/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cudf/types.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace sirius::op::scan::detail {
// Host offsets; the caller establishes readable extent, readiness and lifetime.
class arrow_string_offsets {
 public:
  arrow_string_offsets(std::string_view format, void const* data)
    : data_(data), large_(format == "U")
  {
    if (format != "u" && format != "U")
      throw std::invalid_argument("arrow_layout_unsupported: invalid string storage format");
  }

  std::size_t width() const noexcept { return large_ ? sizeof(int64_t) : sizeof(int32_t); }

  int64_t operator[](int64_t index) const
  {
    check_index(index);
    return large_ ? read<int64_t>(index) : read<int32_t>(index);
  }

  // Inclusive window; check addresses and dispatch width once, then visit every value.
  template <typename F>
  void for_each(int64_t begin, int64_t end, F&& visit) const
  {
    if (end < begin)
      throw std::invalid_argument("arrow_layout_unsupported: reversed string offset window");
    check_index(begin);
    check_index(end);
    if (large_) {
      for (auto index = begin; index <= end; ++index)
        visit(index, read<int64_t>(index));
    } else {
      for (auto index = begin; index <= end; ++index)
        visit(index, read<int32_t>(index));
    }
  }

 private:
  void check_index(int64_t index) const
  {
    if (!data_ || index < 0 ||
        static_cast<uint64_t>(index) >
          static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / width())
      throw std::invalid_argument("arrow_layout_unsupported: invalid string offset address");
  }
  template <typename T>
  int64_t read(int64_t index) const
  {
    auto const* at = static_cast<std::byte const*>(data_) + index * sizeof(T);
    T value;
    std::memcpy(&value, at, sizeof(value));
    return value;
  }
  void const* data_;
  bool large_;
};

// INT32 destination offsets bound the selected character count, not the parent origin.
inline cudf::size_type arrow_string_window_size(int64_t first, int64_t last)
{
  if (first < 0 || last < first || last - first > std::numeric_limits<cudf::size_type>::max())
    throw std::invalid_argument("arrow_layout_unsupported: string window is not representable");
  return static_cast<cudf::size_type>(last - first);
}
}  // namespace sirius::op::scan::detail
