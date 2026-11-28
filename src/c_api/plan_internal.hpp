// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "c_api/handle_internal.hpp"

#include <sirius-c/plan/plan.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::c_api::plan {
template <class T>
struct release {
  void operator()(T* value) const noexcept { c_api::release(value); }
};
template <class T>
using owner = std::unique_ptr<T, release<T>>;
template <class T>
owner<T> acquire(const T* value) noexcept
{
  auto* handle = const_cast<T*>(value);
  c_api::retain(handle);
  return owner<T>{handle};
}
struct field {
  std::string name;
  sirius_data_type type;
  bool nullable;
};
struct column {
  size_t index;
};
struct comparison {
  owner<sirius_expression> left;
  owner<sirius_expression> right;
  sirius_comparison operation;
};
struct local_file {
  std::string path;
};
struct s3_object {
  std::string uri;
  std::size_t bucket_end;
};
class parquet_scan {
 public:
  using source_type = std::variant<local_file, s3_object>;
  static parquet_scan parse_scan_source(char const* location, std::size_t location_size);
  source_type const& source() const noexcept { return source_; }

 private:
  explicit parquet_scan(source_type source) : source_(std::move(source)) {}
  source_type source_;
};
struct filter {
  owner<sirius_plan_node> input;
  owner<sirius_expression> predicate;
};
inline constexpr size_t max_depth = 256;
}  // namespace sirius::c_api::plan

struct sirius_schema {
  std::atomic<size_t> references{1};
  const std::vector<sirius::c_api::plan::field> fields;
  explicit sirius_schema(std::vector<sirius::c_api::plan::field> fields) : fields(std::move(fields))
  {
  }
};
struct sirius_expression {
  using value_type =
    std::variant<int64_t, bool, sirius::c_api::plan::column, sirius::c_api::plan::comparison>;
  std::atomic<size_t> references{1};
  const sirius_data_type type;
  const bool nullable;
  const sirius::c_api::plan::owner<sirius_plan_node> scope;
  const value_type value;
  const size_t depth;
  sirius_expression(sirius_data_type type,
                    bool nullable,
                    sirius::c_api::plan::owner<sirius_plan_node> scope,
                    value_type value,
                    size_t depth)
    : type(type), nullable(nullable), scope(std::move(scope)), value(std::move(value)), depth(depth)
  {
  }
};
struct sirius_plan_node {
  using value_type = std::variant<sirius::c_api::plan::parquet_scan, sirius::c_api::plan::filter>;
  std::atomic<size_t> references{1};
  const sirius::c_api::plan::owner<sirius_schema> schema;
  const value_type value;
  const size_t depth;
  sirius_plan_node(sirius::c_api::plan::owner<sirius_schema> schema, value_type value, size_t depth)
    : schema(std::move(schema)), value(std::move(value)), depth(depth)
  {
  }
};
struct sirius_plan {
  std::atomic<size_t> references{1};
  const sirius::c_api::plan::owner<sirius_plan_node> root;
  const std::string description;
  sirius_plan(sirius::c_api::plan::owner<sirius_plan_node> root, std::string description)
    : root(std::move(root)), description(std::move(description))
  {
  }
};
