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

#include "simpatico_bridge.hpp"

#include <atomic>
#include <filesystem>
#include <random>
#include <sstream>

namespace sirius::compression {

void initialize_simpatico_jit()
{
  // Simpatico's JIT uses the CUDA primary context established by Sirius at
  // startup — no explicit warm-up call is needed in the current build.
  // This function is a no-op placeholder retained for call-site symmetry and
  // to allow a warm-up step to be added here if future Simpatico versions
  // require one (e.g. seeding the on-disk cubin cache).
}

std::string make_compressed_temp_path(const std::string& dir)
{
  // Generate a random 64-bit hex suffix to make the name unique.
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dist;
  std::ostringstream ss;
  ss << std::hex << dist(gen);

  auto path = std::filesystem::path(dir) / (ss.str() + ".hpln");
  return path.string();
}

}  // namespace sirius::compression
