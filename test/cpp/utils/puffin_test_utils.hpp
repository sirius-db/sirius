/*
 * Copyright 2026, Sirius Contributors.
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
#include "op/scan/puffin_reader.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace sirius::test {
// Footer text is deliberately unchecked so tests can construct malformed containers.
inline std::vector<std::byte> puffin_container(std::span<std::byte const> payload,
                                               std::string_view footer,
                                               uint32_t flags = 0)
{
  std::vector<std::byte> bytes;
  auto append = [&](std::string_view text) {
    for (auto c : text)
      bytes.push_back(std::byte(c));
  };
  append("PFA1");
  bytes.insert(bytes.end(), payload.begin(), payload.end());
  append("PFA1");
  append(footer);
  for (uint32_t value : {static_cast<uint32_t>(footer.size()), flags})
    for (int i = 0; i < 4; ++i)
      bytes.push_back(std::byte((value >> (i * 8)) & 255));
  append("PFA1");
  return bytes;
}

class memory_puffin_source final : public op::scan::puffin_byte_source {
 public:
  explicit memory_puffin_source(std::vector<std::byte> data = {}) : bytes(std::move(data)) {}
  bool seek(std::streamoff position) override
  {
    if (position < 0 || static_cast<uint64_t>(position) > bytes.size()) return false;
    offset = position;
    return true;
  }
  bool read(std::span<std::byte> out) override
  {
    if (fail_read || offset > bytes.size() || out.size() > bytes.size() - offset) return false;
    std::copy_n(bytes.data() + offset, out.size(), out.data());
    offset += out.size();
    return true;
  }
  std::streamoff size() const override { return bytes.size(); }
  std::vector<std::byte> bytes;
  size_t offset  = 0;
  bool fail_read = false;
};
}  // namespace sirius::test
