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

#include <sirius_config.hpp>
#include <telemetry/telemetry_context.hpp>

#include <memory>

namespace sirius::test {

inline std::shared_ptr<const telemetry::telemetry_context> make_test_telemetry_context()
{
  telemetry_config config;
  config.enable_quent = false;
  config.engine_name  = "test-engine";
  return telemetry::telemetry_context::create(config);
}

}  // namespace sirius::test
