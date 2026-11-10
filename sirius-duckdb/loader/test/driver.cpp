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

#include <cstdlib>

#ifndef MISSING_VERSION_SYMBOL
extern "C" int cuDriverGetVersion(int* version)
{
  auto const* value = std::getenv("SIRIUS_TEST_DRIVER_VERSION");
  *version          = value ? std::atoi(value) : 0;
  auto const* error = std::getenv("SIRIUS_TEST_DRIVER_ERROR");
  return error ? std::atoi(error) : 0;
}
#endif
