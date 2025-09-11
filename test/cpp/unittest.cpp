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

#define CATCH_CONFIG_RUNNER

#include "catch.hpp"
#include "log/logging.hpp"

using namespace duckdb;

int main(int argc, char* argv[]) {
  // Initialize the logger
  std::string log_dir = SIRIUS_UNITTEST_LOG_DIR;
  InitGlobalLogger(log_dir + "/sirius_unittest.log");

  // Run tests
  return Catch::Session().run(argc, argv);
}
