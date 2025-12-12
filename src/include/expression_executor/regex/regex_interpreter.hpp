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

#pragma once

#include <mutex>
#include <string>
#include <unordered_map>

namespace sirius {
namespace expression {

struct RegexUdf {
  std::string function_name;
  std::string source;
};

class RegexInterpreter {
public:
  RegexUdf Generate(std::string pattern, std::string replacement) const;
};

class RegexUdfCache {
public:
  static RegexUdfCache& Instance();

  const RegexUdf& GetOrCreate(const std::string& pattern, const std::string& replacement);

private:
  RegexUdfCache() = default;

private:
  std::mutex mutex_;
  std::unordered_map<std::string, RegexUdf> cache_;
  RegexInterpreter interpreter_;
};

} // namespace expression
} // namespace sirius
