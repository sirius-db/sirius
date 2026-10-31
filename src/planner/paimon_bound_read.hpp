// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "planner/paimon_read_bridge.h"

#include <duckdb/function/table_function.hpp>

#include <memory>

namespace duckdb {
class DatabaseInstance;
}

namespace sirius::planner {
// Descriptor-only consumer. This does not admit a scan, plan native work, or authorize replay.
class paimon_bound_read {
 public:
  paimon_bound_read(paimon_bound_read&&) noexcept;
  paimon_bound_read& operator=(paimon_bound_read&&) noexcept;
  ~paimon_bound_read();

  // Borrowed until this owner is moved or destroyed. Options can contain credentials:
  // never log/serialize this view. Empty-at-bind is distinct from resolving latest.
  // Synchronize access against moving/destroying this owner.
  PaimonBoundReadViewV1 const& view() const;

 private:
  struct state;
  explicit paimon_bound_read(std::unique_ptr<state>);
  std::unique_ptr<state> state_;
  friend class paimon_read_provider;
};

class paimon_read_provider {
 public:
  // Uses only an already-loaded, ABI-checked dynamic Paimon extension. Trust comes
  // from independent registration in a private CPU catalog, not the caller's catalog.
  // The provider must support repeated initialization into fresh database instances
  // without invalidating existing registrations, bindings or retained descriptors.
  // Acquire once per provider; later LOADs require a new acquisition.
  static paimon_read_provider from_loaded(duckdb::DatabaseInstance&);

  // Independent retains may overlap: keep the provider, function and binding alive
  // and unchanged until each call returns. Each descriptor has its own owner.
  paimon_bound_read retain(duckdb::BoundTableFunction const&, duckdb::FunctionData const*) const;

 private:
  struct state;
  explicit paimon_read_provider(std::shared_ptr<state>);
  std::shared_ptr<state> state_;
  friend class paimon_bound_read;
};
}  // namespace sirius::planner
