/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "op/scan/scan_filter_analysis.hpp"

#include <stdexcept>

namespace sirius::op::scan {
// Owns the complete decomposition and its lowered residual. The same object
// travels through preparation, decode and shared materialization; counts alone
// cannot bind a producer's progress to expressions or materialized slots.
class static_predicate_catalogue {
 public:
  static_predicate_catalogue(uint64_t contract,
                             uint64_t generation,
                             std::vector<table_filter_conjunct> decomposition,
                             std::unordered_set<std::size_t> const& answerable = {})
    : contract_(contract), generation_(generation), residual_(std::move(decomposition), answerable)
  {
    if (!contract_ || !generation_)
      throw std::invalid_argument("predicate catalogue requires a bound generation");
  }
  [[nodiscard]] uint64_t contract() const noexcept { return contract_; }
  [[nodiscard]] uint64_t generation() const noexcept { return generation_; }
  [[nodiscard]] residual_filter const& residual() const noexcept { return residual_; }
  [[nodiscard]] std::size_t size() const noexcept { return residual_.size(); }
  [[nodiscard]] duckdb::Expression const& expression(std::size_t id) const
  {
    return residual_.expression(id);
  }
  [[nodiscard]] std::span<std::size_t const> columns(std::size_t id) const
  {
    return residual_.columns(id);
  }
  [[nodiscard]] bool answerable(std::size_t id) const { return residual_.answerable(id); }

 private:
  uint64_t contract_, generation_;
  residual_filter residual_;
};
}  // namespace sirius::op::scan
