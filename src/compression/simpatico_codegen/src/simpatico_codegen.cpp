// SPDX-License-Identifier: Apache-2.0
#include "api/simpatico_codegen.hpp"

#include "codegen/plan/representation.hpp"
#include "compress_internals.hpp"

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

namespace simpatico {

namespace {

leaf_desc make_leaf_desc(std::uint32_t node_index,
                         std::int32_t slot,
                         PlanLeafKind kind,
                         compressed_representation const* rep,
                         rmm::cuda_stream_view stream)
{
  leaf_desc d;
  d.node_index = node_index;
  d.slot       = slot;
  d.kind       = kind;
  d.type_tag   = dtype_to_tag(rep->decoded_type());
  // The node's own output length. Decode sizes the codegen kernel grid from this,
  // so a nested fused subtree (whose length is far below the column row count)
  // must round-trip its true length rather than inherit the column's.
  d.num_rows = rep->num_rows > 0 ? static_cast<std::uint64_t>(rep->num_rows) : 0;
  d.meta     = rep->describe_meta();
  for (auto const& ch : rep->named_channels(stream)) {
    leaf_buffer_desc bd;
    bd.name     = ch.name;
    bd.type_tag = dtype_to_tag(ch.view.type());
    bd.num_rows = static_cast<std::uint64_t>(ch.view.size());
    // cudf::size_of() only supports fixed-width types (e.g. dictionary's
    // fast-mode "keys" channel is a raw STRING column) — account for
    // offsets + chars directly instead of calling it on a STRING view.
    if (ch.view.type().id() == cudf::type_id::STRING) {
      if (ch.view.num_children() == 0) {
        if (ch.view.size() != 0) {
          throw std::logic_error("non-empty STRING channel has no offsets child.");
        }
        bd.size_bytes = 0;
      } else {
        cudf::strings_column_view scv(ch.view);
        auto const offsets_width = static_cast<size_t>(cudf::size_of(scv.offsets().type()));
        bd.size_bytes            = static_cast<std::uint64_t>(ch.view.size() + 1) * offsets_width +
                        static_cast<std::uint64_t>(scv.chars_size(stream));
      }
    } else {
      bd.size_bytes = static_cast<std::uint64_t>(ch.view.size()) *
                      static_cast<std::uint64_t>(cudf::size_of(ch.view.type()));
    }
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

std::vector<std::vector<leaf_desc>> compressed_table::describe(rmm::cuda_stream_view stream) const
{
  std::vector<std::vector<leaf_desc>> result;
  result.reserve(columns.size());
  for (auto const& col : columns) {
    std::vector<leaf_desc> descs;
    if (!col.compound) {
      result.push_back({});
      continue;
    }
    auto const& nodes = col.compound->tree.nodes;
    for (std::uint32_t ni = 0; ni < nodes.size(); ++ni) {
      auto const& node = nodes[ni];
      if (node.rep) {
        descs.push_back(make_leaf_desc(ni, kSelfRepSlot, node.rep->kind(), node.rep.get(), stream));
      }
      for (std::size_t k = 0; k < node.output_paths.size(); ++k) {
        auto it = node.channels.find(node.output_paths[k]);
        if (it != node.channels.end() && it->second) {
          descs.push_back(make_leaf_desc(
            ni, static_cast<std::int32_t>(k), it->second->kind(), it->second.get(), stream));
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
    cols.push_back(
      detail::apply_stored_dtype(simpatico::decompress(*col.compound, stream, mr), col.dtype));
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
