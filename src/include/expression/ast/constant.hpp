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

namespace sirius::ast {

namespace detail {
/// Placeholder carrier for the constant payload.
/// Phase 2 replaces this with sirius::value.
struct value_placeholder {};
}  // namespace detail

/**
 * @brief Sirius-native mirror of duckdb::BoundConstantExpression.
 *
 * Carries a literal value. The payload is a placeholder until Phase 2.
 */
struct constant {
  detail::value_placeholder payload{};
};

}  // namespace sirius::ast
