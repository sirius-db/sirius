/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

#include <utils/tpch_havoc_queries.hpp>
#include <yaml-cpp/yaml.h>

#include <cstdio>
#include <cstdlib>
#include <regex>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace sirius::test {

namespace {

constexpr int kBaseQueries = 22;
constexpr std::string_view kQ11Fraction{"{q11_fraction}"};

std::string trim(std::string const& text)
{
  auto const first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) { return {}; }
  auto const last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

}  // namespace

std::optional<std::filesystem::path> tpch_havoc_dir()
{
  if (char const* dir = std::getenv("SIRIUS_TPCH_HAVOC_DIR"); dir != nullptr && *dir != '\0') {
    return std::filesystem::path{dir};
  }
#ifdef SIRIUS_TPCH_HAVOC_DEFAULT_DIR
  return std::filesystem::path{SIRIUS_TPCH_HAVOC_DEFAULT_DIR};
#else
  return std::nullopt;
#endif
}

std::vector<tpch_havoc_query> load_tpch_havoc_queries(std::filesystem::path const& dir,
                                                      double scale_factor)
{
  // BenchBox renders the threshold as f"{0.0001 / SF:.10f}".
  char fraction[32];
  std::snprintf(fraction, sizeof(fraction), "%.10f", 0.0001 / scale_factor);
  static std::regex const parameter{R"(\{[A-Za-z_][A-Za-z0-9_]*\})"};

  std::vector<tpch_havoc_query> queries;
  std::unordered_map<std::string, std::size_t> first_with_sql;
  for (int base_query = 1; base_query <= kBaseQueries; ++base_query) {
    char file_name[16];
    std::snprintf(file_name, sizeof(file_name), "q%02d.yaml", base_query);
    auto const path = dir / file_name;
    if (!std::filesystem::exists(path)) {
      throw std::runtime_error("TPC-Havoc query file not found: " + path.string());
    }
    auto const document = YAML::LoadFile(path.string());
    for (auto const& entry : document["variants"]) {
      tpch_havoc_query query;
      query.base_query  = base_query;
      query.variant     = entry["id"].as<int>();
      query.description = entry["description"].as<std::string>();
      query.sql         = trim(entry["sql"].as<std::string>());
      char label[16];
      std::snprintf(label, sizeof(label), "Q%02d_V%02d", base_query, query.variant);
      query.label = label;
      for (auto at = query.sql.find(kQ11Fraction); at != std::string::npos;
           at      = query.sql.find(kQ11Fraction, at)) {
        query.sql.replace(at, kQ11Fraction.size(), fraction);
      }
      if (std::smatch match; std::regex_search(query.sql, match, parameter)) {
        throw std::runtime_error(query.label + " has an unknown template parameter " + match.str());
      }
      auto const [first, inserted] = first_with_sql.try_emplace(query.sql, queries.size());
      if (!inserted) {
        queries[first->second].duplicates.push_back(std::move(query.label));
        continue;
      }
      queries.push_back(std::move(query));
    }
  }
  return queries;
}

std::string to_string(tpch_havoc_outcome outcome)
{
  switch (outcome) {
    case tpch_havoc_outcome::gpu: return "gpu";
    case tpch_havoc_outcome::mismatch: return "mismatch";
    case tpch_havoc_outcome::plan_rejected: return "plan_rejected";
    case tpch_havoc_outcome::runtime_failed: return "runtime_failed";
    case tpch_havoc_outcome::skip: return "skip";
  }
  return "unknown";
}

bool tpch_havoc_expectation::applies_to(std::string const& scan_name,
                                        std::string const& evaluator_name) const
{
  return (scan.empty() || scan == scan_name) && (evaluator.empty() || evaluator == evaluator_name);
}

std::filesystem::path tpch_havoc_expectations_path()
{
  return std::filesystem::path{SIRIUS_PROJECT_ROOT} / "test/tpch_havoc/expectations.yaml";
}

std::vector<tpch_havoc_expectation> load_tpch_havoc_expectations(std::filesystem::path const& path)
{
  static std::unordered_map<std::string, tpch_havoc_outcome> const outcomes{
    {"mismatch", tpch_havoc_outcome::mismatch},
    {"plan_rejected", tpch_havoc_outcome::plan_rejected},
    {"runtime_failed", tpch_havoc_outcome::runtime_failed},
    {"skip", tpch_havoc_outcome::skip}};

  std::vector<tpch_havoc_expectation> expectations;
  for (auto const& entry : YAML::LoadFile(path.string())["expectations"]) {
    tpch_havoc_expectation expectation;
    auto const outcome    = entry["outcome"].as<std::string>("");
    auto const known      = outcomes.find(outcome);
    expectation.reason    = entry["reason"].as<std::string>("");
    expectation.issue     = entry["issue"].as<int>(0);
    expectation.scan      = entry["scan"].as<std::string>("");
    expectation.evaluator = entry["evaluator"].as<std::string>("");
    expectation.queries = entry["queries"].as<std::vector<std::string>>(std::vector<std::string>{});
    auto const where    = path.string() + " line " + std::to_string(entry.Mark().line + 1);
    if (known == outcomes.end()) {
      throw std::runtime_error(where + ": unknown outcome '" + outcome + "'");
    }
    expectation.outcome = known->second;
    if (expectation.issue <= 0) { throw std::runtime_error(where + ": needs an issue"); }
    if (expectation.reason.empty()) { throw std::runtime_error(where + ": needs a reason"); }
    if (expectation.queries.empty()) { throw std::runtime_error(where + ": lists no queries"); }
    if (!expectation.scan.empty() && expectation.scan != "native" &&
        expectation.scan != "parquet") {
      throw std::runtime_error(where + ": unknown scan '" + expectation.scan + "'");
    }
    if (!expectation.evaluator.empty() && expectation.evaluator != "ast_interpret" &&
        expectation.evaluator != "ast_jit") {
      throw std::runtime_error(where + ": unknown evaluator '" + expectation.evaluator + "'");
    }
    try {
      std::regex const reason{expectation.reason};
    } catch (std::regex_error const& error) {
      throw std::runtime_error(where + ": reason is not a valid regex: " + error.what());
    }
    expectations.push_back(std::move(expectation));
  }
  return expectations;
}

tpch_havoc_expectation const* find_tpch_havoc_expectation(
  std::vector<tpch_havoc_expectation> const& expectations,
  std::string const& label,
  std::string const& scan,
  std::string const& evaluator)
{
  tpch_havoc_expectation const* found = nullptr;
  for (auto const& expectation : expectations) {
    if (!expectation.applies_to(scan, evaluator)) { continue; }
    for (auto const& query : expectation.queries) {
      if (query != label) { continue; }
      if (found != nullptr) {
        throw std::runtime_error(label + " is listed by more than one expectation for " + scan +
                                 " / " + evaluator);
      }
      found = &expectation;
    }
  }
  return found;
}

}  // namespace sirius::test
