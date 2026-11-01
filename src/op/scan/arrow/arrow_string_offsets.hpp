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
// Host offsets only. The caller must retain readable backing for the declared
// Arrow window; this reader does not establish allocation extent or readiness.
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
    if (!data_ || index < 0 ||
        static_cast<uint64_t>(index) >
          static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / width())
      throw std::invalid_argument("arrow_layout_unsupported: invalid string offset address");
    auto const* at = static_cast<std::byte const*>(data_) + index * width();
    if (large_) {
      int64_t value;
      std::memcpy(&value, at, sizeof(value));
      return value;
    }
    int32_t value;
    std::memcpy(&value, at, sizeof(value));
    return value;
  }

 private:
  void const* data_;
  bool large_;
};

// The importer currently constructs INT32 destination offsets. A large parent
// origin is valid: only the rebased selected character count must fit. Kept
// separate from payload access so boundary cases need no multi-GiB allocation.
inline cudf::size_type arrow_string_window_size(int64_t first, int64_t last)
{
  if (first < 0 || last < first || last - first > std::numeric_limits<cudf::size_type>::max())
    throw std::invalid_argument("arrow_layout_unsupported: string window is not representable");
  return static_cast<cudf::size_type>(last - first);
}
}  // namespace sirius::op::scan::detail
