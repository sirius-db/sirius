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

#include <cucascade/data/common.hpp>

namespace sirius {

/** @brief RTTI base for Simpatico-compressed data representations. */
class simpatico_compressed_representation : public cucascade::idata_representation {
 public:
  ~simpatico_compressed_representation() override = default;

 protected:
  explicit simpatico_compressed_representation(cucascade::memory::memory_space& memory_space)
    : cucascade::idata_representation(memory_space)
  {
  }
};

[[nodiscard]] inline bool is_simpatico_compressed_representation(
  const cucascade::idata_representation* representation) noexcept
{
  return dynamic_cast<const simpatico_compressed_representation*>(representation) != nullptr;
}

}  // namespace sirius
