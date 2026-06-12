// SPDX-License-Identifier: Apache-2.0
#include "api/simpatico_codegen.hpp"

#include "api/compress_internals.hpp"
#include "codegen/plan/representation.hpp"

#include <cudf/types.hpp>

namespace simpatico {

namespace {

leaf_desc make_leaf_desc(std::string path, PlanLeafKind kind, compressed_representation const* rep)
{
  leaf_desc d;
  d.path     = std::move(path);
  d.kind     = kind;
  d.type_tag = dtype_to_tag(rep->decoded_type());
  d.meta     = rep->describe_meta();
  for (auto const& ch : rep->named_channels()) {
    leaf_buffer_desc bd;
    bd.name       = ch.name;
    bd.type_tag   = dtype_to_tag(ch.view.type());
    bd.num_rows   = static_cast<std::uint64_t>(ch.view.size());
    bd.size_bytes = static_cast<std::uint64_t>(ch.view.size()) *
                    static_cast<std::uint64_t>(cudf::size_of(ch.view.type()));
    bd.device_ptr = ch.view.head<void>();
    d.buffers.push_back(std::move(bd));
  }
  return d;
}

}  // namespace

// ---------------------------------------------------------------------------
// compressed_table::describe()
// ---------------------------------------------------------------------------
// Walk each column's plan_compound::tree; for every stored rep emit one
// leaf_desc. Two storage slots per PlanNode:
//   * node.rep      (path = node.rep_path)
//   * node.channels (path = the map key)
// rep->kind() is used for all rep types including codegen_fused_representation,
// which returns Delta/Rle/Identity for fused delta/rle/raw ops respectively.

std::vector<std::vector<leaf_desc>> compressed_table::describe() const
{
  std::vector<std::vector<leaf_desc>> result;
  result.reserve(columns.size());
  for (auto const& col : columns) {
    std::vector<leaf_desc> descs;
    if (!col.compound) {
      result.push_back({});
      continue;
    }
    for (auto const& node : col.compound->tree.nodes) {
      if (node.rep) {
        descs.push_back(make_leaf_desc(node.rep_path, node.rep->kind(), node.rep.get()));
      }
      for (auto const& out_path : node.output_paths) {
        auto it = node.channels.find(out_path);
        if (it != node.channels.end() && it->second) {
          descs.push_back(make_leaf_desc(out_path, it->second->kind(), it->second.get()));
        }
      }
    }
    result.push_back(std::move(descs));
  }
  return result;
}

using detail::compress_columns_parallel;
using detail::decompress_columns_parallel;
using detail::make_internal_pool;
using detail::plan_error;
using detail::split_plan_dsl_impl;
using detail::validate_column_names;
using detail::validate_plan_count;

// ── compressed_table ─────────────────────────────────────────────────────────

std::int64_t compressed_table::num_rows() const
{
  return columns.empty() ? 0 : columns.front().num_rows;
}

std::unique_ptr<cudf::table> compressed_table::decompress(rmm::cuda_stream_view stream,
                                                          rmm::device_async_resource_ref mr) const
{
  return simpatico::decompress(*this, stream, mr);
}

// ── split_plan_dsl ────────────────────────────────────────────────────────────

std::vector<std::string> split_plan_dsl(std::string_view plan_dsl)
{
  return split_plan_dsl_impl(plan_dsl);
}

// ── compress_with_plan ────────────────────────────────────────────────────────

compressed_table compress_with_plan(cudf::table_view table,
                                    std::string_view plan_dsl,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr,
                                    std::vector<std::string> column_names)
{
  auto plans = split_plan_dsl_impl(plan_dsl);
  validate_plan_count(plans.size(), table.num_columns());
  validate_column_names(column_names, plans.size());

  compressed_table out;
  out.columns.reserve(plans.size());
  for (size_t i = 0; i < plans.size(); ++i) {
    std::string err;
    auto compound =
      compress_column(table.column(static_cast<cudf::size_type>(i)), plans[i], stream, mr, &err);
    if (!compound) throw plan_error(err.empty() ? "compress failed" : err);
    compressed_column col;
    col.dtype    = table.column(static_cast<cudf::size_type>(i)).type();
    col.num_rows = table.num_rows();
    col.compound = std::move(compound);
    if (!column_names.empty()) col.name = column_names[i];
    out.columns.push_back(std::move(col));
  }
  return out;
}

compressed_table compress_with_plan(cudf::table_view table,
                                    std::string_view plan_dsl,
                                    int column_threads,
                                    rmm::device_async_resource_ref mr,
                                    std::vector<std::string> column_names)
{
  auto plans = split_plan_dsl_impl(plan_dsl);
  validate_plan_count(plans.size(), table.num_columns());
  validate_column_names(column_names, plans.size());
  auto pool = make_internal_pool(column_threads);
  return compress_columns_parallel(table, plans, pool, mr, column_names);
}

compressed_table compress_with_plan(cudf::table_view table,
                                    std::string_view plan_dsl,
                                    simpatico::stream_pool& pool,
                                    rmm::device_async_resource_ref mr,
                                    std::vector<std::string> column_names)
{
  auto plans = split_plan_dsl_impl(plan_dsl);
  validate_plan_count(plans.size(), table.num_columns());
  validate_column_names(column_names, plans.size());
  return compress_columns_parallel(table, plans, pool, mr, column_names);
}

// ── decompress ────────────────────────────────────────────────────────────────

std::unique_ptr<cudf::column> decompress(const simpatico::plan_compound& compound,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  std::string err;
  auto col = decompress_column(compound, stream, mr, &err);
  if (!col) throw plan_error(err.empty() ? "decompress failed" : err);
  return col;
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(table.num_columns());
  for (auto const& col : table.columns) {
    if (!col.compound) throw plan_error("compressed_table column missing compound");
    cols.push_back(simpatico::decompress(*col.compound, stream, mr));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        int column_threads,
                                        rmm::device_async_resource_ref mr)
{
  auto pool = make_internal_pool(column_threads);
  return decompress_columns_parallel(table, pool, mr);
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        simpatico::stream_pool& pool,
                                        rmm::device_async_resource_ref mr)
{
  return decompress_columns_parallel(table, pool, mr);
}

}  // namespace simpatico
