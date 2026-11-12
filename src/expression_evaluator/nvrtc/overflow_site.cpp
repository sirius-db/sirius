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

#include <expression/function_id.hpp>
#include <expression_evaluator/nvrtc/overflow_site.hpp>

#include <algorithm>
#include <cstddef>
#include <format>
#include <vector>

namespace sirius::nvrtc {

namespace {

constexpr std::size_t max_listed_candidates = 4;

// Arithmetic operators such as `+` read best infix; every other function reads as a call.
[[nodiscard]] bool is_infix_operator(sirius::function_id function) noexcept
{
  switch (function) {
    case sirius::function_id::add:
    case sirius::function_id::sub:
    case sirius::function_id::mul:
    case sirius::function_id::div:
    case sirius::function_id::int_div:
    case sirius::function_id::mod: return true;
    default: return false;
  }
}

[[nodiscard]] std::string describe(sirius::ast::function_call const& call)
{
  auto const name   = sirius::to_duckdb_function_name(call.function());
  auto const& args  = call.arguments();
  auto const result = call.return_type().to_string();
  if (is_infix_operator(call.function()) && args.size() == 2) {
    return std::format("{} {} {} -> {}",
                       args[0]->return_type().to_string(),
                       name,
                       args[1]->return_type().to_string(),
                       result);
  }
  std::string operands;
  for (auto const& arg : args) {
    if (!operands.empty()) { operands += ", "; }
    operands += arg->return_type().to_string();
  }
  return std::format("{}({}) -> {}", name, operands, result);
}

[[nodiscard]] std::string describe(sirius::ast::cast const& cast)
{
  return std::format(
    "CAST({} AS {})", cast.child->return_type().to_string(), cast.target_type.to_string());
}

}  // namespace

std::string overflow_message(std::span<overflow_site const> sites)
{
  std::vector<std::string> candidates;
  for (auto const& site : sites) {
    auto text = std::visit([](auto const* op) { return describe(*op); }, site);
    if (std::ranges::find(candidates, text) == candidates.end()) {
      candidates.push_back(std::move(text));
    }
  }

  std::string message = "[expression_evaluator]: Overflow in NVRTC evaluation";
  if (candidates.empty()) { return message; }
  if (candidates.size() == 1) { return message + " of " + candidates.front(); }
  message += " of one of: ";
  auto const listed = std::min(candidates.size(), max_listed_candidates);
  for (std::size_t i = 0; i < listed; ++i) {
    if (i > 0) { message += "; "; }
    message += candidates[i];
  }
  if (candidates.size() > listed) {
    message += std::format("; and {} more", candidates.size() - listed);
  }
  return message;
}

}  // namespace sirius::nvrtc
