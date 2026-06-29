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

#include <codegen/jit/nvrtc_compiler.hpp>

#include <random>
#include <sstream>
#include <string>

namespace sirius::compression {

void initialize_simpatico_jit() { codegen::jit::ensure_cuda_context(); }

std::string make_compressed_temp_path(const std::string& temp_dir)
{
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dist;
  std::ostringstream oss;
  oss << temp_dir << "/sirius_comp_" << std::hex << dist(gen) << ".hpln";
  return oss.str();
}

}  // namespace sirius::compression
