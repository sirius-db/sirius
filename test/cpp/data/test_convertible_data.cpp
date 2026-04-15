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

#include "catch.hpp"
#include "data/convertible_data.hpp"

#include <cstddef>
#include <memory>
#include <vector>

// Forward declarations needed for the stub implementations
namespace cucascade {
namespace memory {
class memory_space;
}  // namespace memory
}  // namespace cucascade

namespace sirius {
namespace memory {
class sirius_memory_reservation_manager;
}  // namespace memory
}  // namespace sirius

namespace {

// Minimal concrete subclass to verify convertible_data compiles and can be subclassed
class stub_convertible_data : public sirius::convertible_data {
 public:
  bool convert(const std::vector<cucascade::memory::memory_space*>& /*target_spaces*/,
               rmm::cuda_stream_view /*stream*/,
               sirius::memory::sirius_memory_reservation_manager& /*res_mgr*/) override
  {
    return false;
  }

  std::size_t bytes_in_space(cucascade::memory::memory_space* /*space*/) const override
  {
    return 0;
  }
};

// Minimal concrete subclass to verify convertible_data_provider compiles and can be subclassed
class stub_convertible_data_provider : public sirius::convertible_data_provider {
 public:
  std::unique_ptr<sirius::convertible_data> get_next_convertible(
    cucascade::memory::memory_space* /*space*/, bool /*front_to_back*/) override
  {
    return nullptr;
  }

  std::vector<std::unique_ptr<sirius::convertible_data>> get_all_convertible(
    cucascade::memory::memory_space* /*space*/, bool /*front_to_back*/) override
  {
    return {};
  }

  std::size_t get_bytes_in_space(cucascade::memory::memory_space* /*space*/) const override
  {
    return 0;
  }
};

}  // anonymous namespace

TEST_CASE("convertible_data interface can be subclassed and instantiated",
          "[convertible_data]")
{
  auto data = std::make_unique<stub_convertible_data>();
  REQUIRE(data != nullptr);

  // Verify the interface pointer works (polymorphism)
  std::unique_ptr<sirius::convertible_data> base = std::move(data);
  REQUIRE(base != nullptr);
  REQUIRE(base->bytes_in_space(nullptr) == 0);
}

TEST_CASE("convertible_data_provider interface can be subclassed and instantiated",
          "[convertible_data]")
{
  auto provider = std::make_unique<stub_convertible_data_provider>();
  REQUIRE(provider != nullptr);

  // Verify the interface pointer works (polymorphism)
  std::unique_ptr<sirius::convertible_data_provider> base = std::move(provider);
  REQUIRE(base != nullptr);
  REQUIRE(base->get_next_convertible(nullptr, true) == nullptr);
  REQUIRE(base->get_all_convertible(nullptr, false).empty());
  REQUIRE(base->get_bytes_in_space(nullptr) == 0);
}
