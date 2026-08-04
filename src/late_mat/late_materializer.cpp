// SPDX-License-Identifier: Apache-2.0
//
// Late materializer implementation (SIRIUS_EXP_LATE_MAT). See
// late_materializer.hpp for the contract. Device-side selection
// preprocessing lives in simpatico
// (src/compression/simpatico_codegen/src/selection/latemat_rowset.cu); this
// file is orchestration: route policy per batch, cudf assembly, order
// restoration.

#include "late_materializer.hpp"

#include "api/simpatico_codegen.hpp"
#include "codegen/plan/row_decode.hpp"
#include "codegen/selection/selection.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table_view.hpp>

#include <cuda_runtime.h>

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace sirius::late_mat {

namespace {

using sirius::codegen::chunk_row_set;
using sirius::codegen::selection_mask;

double env_knob(char const* name, double fallback)
{
  char const* v = std::getenv(name);
  if (v == nullptr || *v == '\0') { return fallback; }
  char* end          = nullptr;
  double const parsed = std::strtod(v, &end);
  return (end != v && parsed >= 0.0 && parsed <= 1.0) ? parsed : fallback;
}

// Route thresholds — MEASURED (late-mat crossover microbench, GB300
// sm_103a, n=2^29 FOR-bitpack, 13- and 24-bit payloads, 8 densities x
// {uniform, clustered64, one-per-chunk}, all arms bit-identical-verified;
// bench: latemat_bench.cu, results 2026-08-03):
//
//  * MASK_SEL = 1.0: the sparse chunk-CSR decode is tried at EVERY density.
//    The mask route never wins a density-based choice — it is 1.2x-11x
//    slower than the best index route at every measured point (K3 walks
//    all n rows' mask strips; the index kernels do per-survivor work), and
//    the CSR-vs-dense-index difference is bounded at -10% (uniform
//    1%-35%) against wins of 1.5x-47x when chunks are untouched (sparse or
//    clustered selections) and 1.22x at 50% (u16 vs int32 index traffic).
//    The mask route below survives as the CAPABILITY fallback only
//    (general dict shapes, render rejections).
//  * DENSE_SEL = 0.35: fallback ordering when the sparse route declines —
//    mask beats full+gather below the crossover (uniform tie band
//    0.05-0.15, clustered 0.35-0.5); 0.35 forfeits <=3.5% on uniform
//    mid-band but preserves up to 33% on clustered selections
//    (asymmetric-loss pick; converges with the fused campaign's measured
//    MAX_SEL=0.35). Dict/str tiers are exempt from the cut (their "full"
//    alternative materializes full-width chars — the dict-fusion-wins-at-
//    any-selectivity lesson).
//
//  Width sensitivity: 13-bit vs 24-bit crossovers land in the same
//  density buckets (ratio deltas <= 6%, no bucket moves) — one
//  width-independent default, no width conditioning.
double mask_sel_threshold() { return env_knob("SIRIUS_LATE_MAT_MASK_SEL", 1.0); }
double dense_sel_threshold() { return env_knob("SIRIUS_LATE_MAT_DENSE_SEL", 0.35); }

[[noreturn]] void fail(std::string const& what)
{
  throw std::runtime_error("late_mat: " + what);
}

cudf::column_view int32_map_view(rmm::device_buffer const& buf, std::int64_t count)
{
  return cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                           static_cast<cudf::size_type>(count),
                           buf.data(),
                           nullptr,
                           0};
}

std::unique_ptr<cudf::column> gather_one(cudf::column_view const& source,
                                         cudf::column_view const& map,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  auto gathered = cudf::gather(cudf::table_view{{source}},
                               map,
                               cudf::out_of_bounds_policy::DONT_CHECK,
                               stream,
                               mr);
  auto cols = gathered->release();
  return std::move(cols.front());
}

// One batch of a COMPRESSED origin: route per density — sparse chunk-CSR
// decode below MASK_SEL, the shipped mask route below DENSE_SEL, full
// decode + gather above it and for plans with no random access.
std::unique_ptr<cudf::column> materialize_compressed_batch(
  simpatico::compressed_table const& table,
  std::size_t column_index,
  prepared_selection::batch_selection const& bsel,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (column_index >= table.columns.size()) { fail("column_index out of range"); }
  auto const& col = table.columns[column_index];
  if (!col.plan_tree) { fail("compressed column has no plan tree"); }
  auto const& tree = *col.plan_tree;
  chunk_row_set const rows = bsel.rows.view();
  std::string err;

  // EVERY compressed return must re-tag from the codec's storage type to the
  // column's stored dtype (row_decode.hpp contract): the column-level decode
  // entry points return raw storage — a DECIMAL64 left as INT64 silently
  // drops its scale (the q9 arm-C wrong-results root cause). No-op for
  // matching types and for strings.
  auto retag = [&](std::unique_ptr<cudf::column> c) {
    return simpatico::apply_stored_dtype(std::move(c), col.dtype);
  };

  // Dense batch (whole batch live): plain full decode, no selection kernels.
  if (bsel.dense) {
    auto full = simpatico::decompress_column(tree, stream, mr, &err);
    if (!full) { fail("full decode failed: " + err); }
    if (full->null_count() != 0) { fail("nullable columns are not supported (v1)"); }
    return retag(std::move(full));
  }

  // Fused-capture mask form: the wave-1 buffers are already the shipped mask
  // route's exact inputs — go straight through decompress_column_compacted
  // (zero conversion; the mask cost was paid at scan time). tier_b / refusal
  // falls through to full decode + gather via the prepare-built id expansion.
  if (bsel.mask_words) {
    selection_mask mask;
    mask.words          = static_cast<std::uint32_t*>(bsel.mask_words->data());
    mask.num_rows       = rows.num_rows;
    mask.survivor_count = rows.num_survivors;
    mask.chunk_offsets  = static_cast<std::uint32_t*>(bsel.mask_chunk_offsets->data());
    auto compacted = simpatico::decompress_column_compacted(tree, mask, stream, mr, &err);
    if (compacted) { return retag(std::move(compacted)); }
    err.clear();
    auto full = simpatico::decompress_column(tree, stream, mr, &err);
    if (!full) { fail("full decode failed: " + err); }
    if (full->null_count() != 0) { fail("nullable columns are not supported (v1)"); }
    return retag(gather_one(full->view(),
                            int32_map_view(bsel.local_indices, rows.num_survivors), stream,
                            mr));
  }

  // (b) Sparse route: chunk-CSR decode, touched-chunk grid.
  if (bsel.density < mask_sel_threshold()) {
    auto sparse = simpatico::decompress_column_rows(tree, rows, stream, mr, &err);
    if (sparse) { return retag(std::move(sparse)); }
    err.clear();  // fall through — refusal is always safe (tier_b, dict
                  // general shape, render rejection)
  }

  // (b') Mask route: 100% shipped kernels. Reached only when the sparse
  // route declined (capability fallback — general dict shapes, render
  // rejections). Skipped for tier_b plans (no compacted route exists) and
  // above DENSE_SEL — except dict/str tiers, whose "full" alternative
  // materializes full-width chars and loses at any selectivity.
  auto const tier = simpatico::plan_selection_tier(tree);
  bool const chars_tier = tier == sirius::codegen::output_tier::tier_dict_k5 ||
                          tier == sirius::codegen::output_tier::tier_str_k6;
  if ((bsel.density < dense_sel_threshold() || chars_tier) &&
      tier != sirius::codegen::output_tier::tier_b) {
    std::int64_t const n  = rows.num_rows;
    std::int64_t const nc = selection_mask::ChunksFor(n);
    rmm::device_buffer mask_words(
      static_cast<std::size_t>(selection_mask::WordsFor(n)) * sizeof(std::uint32_t), stream,
      mr);
    rmm::device_buffer chunk_offsets(static_cast<std::size_t>(nc + 1) * sizeof(std::uint32_t),
                                     stream, mr);
    sirius::codegen::row_set_to_mask(rows,
                                     static_cast<std::uint32_t*>(mask_words.data()),
                                     static_cast<std::uint32_t*>(chunk_offsets.data()),
                                     stream,
                                     mr);
    selection_mask mask;
    mask.words          = static_cast<std::uint32_t*>(mask_words.data());
    mask.num_rows       = n;
    mask.survivor_count = rows.num_survivors;
    mask.chunk_offsets  = static_cast<std::uint32_t*>(chunk_offsets.data());
    auto compacted = simpatico::decompress_column_compacted(tree, mask, stream, mr, &err);
    if (compacted) { return retag(std::move(compacted)); }
    err.clear();  // fall through to the dense fallback
  }

  // (c) Dense fallback / tier_b: full decode + one gather.
  auto full = simpatico::decompress_column(tree, stream, mr, &err);
  if (!full) { fail("full decode failed: " + err); }
  if (full->null_count() != 0) { fail("nullable columns are not supported (v1)"); }
  return retag(gather_one(full->view(),
                          int32_map_view(bsel.local_indices, rows.num_survivors), stream, mr));
}

}  // namespace

pinned_table_layout pinned_table_layout::from_batch_rows(std::vector<std::int64_t> rows,
                                                         std::uint64_t generation)
{
  pinned_table_layout layout;
  layout.pin_generation = generation;
  layout.batch_row_start.reserve(rows.size() + 1);
  std::int64_t base = 0;
  for (auto const r : rows) {
    if (r < 0) { fail("negative batch row count"); }
    layout.batch_row_start.push_back(base);
    base += r;
  }
  layout.batch_row_start.push_back(base);
  layout.batch_rows = std::move(rows);
  return layout;
}

namespace {

// The canonical (sort/bucket) prepare — the compressed-origin machinery.
// `allow_raw` lets the public entry short-circuit to the raw-gather fast
// path; prepared_selection::canonical() calls back in with allow_raw=false.
std::shared_ptr<prepared_selection> prepare_selection_impl(pinned_table_layout const& layout,
                                                           row_id_list const& ids,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::device_async_resource_ref mr,
                                                           bool allow_raw);

}  // namespace

prepared_selection const& prepared_selection::canonical(rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr) const
{
  if (raw_ids == nullptr) { return *this; }  // already canonical
  std::call_once(_canon_once, [&]() {
    row_id_list raw;
    raw.ids           = raw_ids;
    raw.count         = raw_count;
    raw.sorted_unique = raw_sorted_unique;
    _canon = prepare_selection_impl(layout, raw, stream, mr, /*allow_raw=*/false);
  });
  return *_canon;
}

std::shared_ptr<prepared_selection> prepare_selection(pinned_table_layout const& layout,
                                                      row_id_list const& ids,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  return prepare_selection_impl(layout, ids, stream, mr, /*allow_raw=*/true);
}

namespace {

std::shared_ptr<prepared_selection> prepare_selection_impl(pinned_table_layout const& layout,
                                                           row_id_list const& ids,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::device_async_resource_ref mr,
                                                           bool allow_raw)
{
  if (layout.batch_row_start.size() != layout.batch_rows.size() + 1) {
    fail("layout batch_row_start must have batch_rows.size()+1 entries");
  }
  if (ids.count < 0 || (ids.count > 0 && ids.ids == nullptr)) { fail("invalid id list"); }
  if (ids.count > std::numeric_limits<cudf::size_type>::max()) {
    fail("selection exceeds 2^31-1 rows — batch the materialization");
  }

  auto sel            = std::make_shared<prepared_selection>();
  sel->layout         = layout;
  sel->original_count = ids.count;

  // Raw-gather fast path (the q9-nation +61 ms fix): single-batch layout ⇒
  // batch 0 starts at global row 0, so the raw id list IS the gather map for
  // uncompressed columns — NO sort, NO unique, NO restore, NO device work,
  // NO host sync at prepare. Compressed consumers canonicalize lazily
  // (prepared_selection::canonical). The list is BORROWED per the
  // row_id_list lifetime contract.
  if (allow_raw && layout.batch_rows.size() == 1) {
    sel->raw_ids           = ids.ids;
    sel->raw_count         = ids.count;
    sel->raw_sorted_unique = ids.sorted_unique;
    sel->total_survivors   = ids.count;  // gather semantics: output rows = input ids
    sel->out_base          = {0, ids.count};
    sel->batches.resize(1);
    sel->batches[0].rows.num_rows       = layout.batch_rows[0];
    sel->batches[0].rows.num_survivors  = ids.count;
    sel->batches[0].density =
      layout.batch_rows[0] > 0 ? static_cast<double>(ids.count) / layout.batch_rows[0] : 0.0;
    return sel;
  }

  // Sorted-unique canonical id stream (+ restoration ranks when needed).
  // Sync surgery: sort_unique_global_ids is now fully asynchronous (worst-
  // case unique buffer, device-resident count); the ONE boundary host sync of
  // this path lives in split_sorted_ids_by_batch, which folds the unique
  // count into the same readback as the batch starts.
  sirius::codegen::sorted_unique_ids canon;
  std::uint64_t const* sorted_ids     = ids.ids;
  std::int64_t max_count              = ids.count;
  std::int32_t const* count_dev       = nullptr;
  if (!ids.sorted_unique && ids.count > 0) {
    canon             = sirius::codegen::sort_unique_global_ids(ids.ids, ids.count, stream, mr);
    sorted_ids        = static_cast<std::uint64_t const*>(canon.ids.data());
    count_dev         = static_cast<std::int32_t const*>(canon.count_dev.data());
    sel->restore_rank = std::move(canon.restore_rank);
  }

  // Per-batch slices + the folded-in unique count: ONE host sync.
  std::int64_t sorted_count = 0;
  auto const starts         = sirius::codegen::split_sorted_ids_by_batch(
    sorted_ids, max_count, count_dev, layout.batch_row_start, &sorted_count, stream, mr);
  sel->total_survivors = sorted_count;
  if (!starts.empty() && starts.back() != sorted_count) {
    fail("row id beyond the pinned table's row count");
  }
  auto const num_batches = layout.batch_rows.size();
  sel->batches.resize(num_batches);
  sel->out_base.assign(num_batches + 1, 0);
  for (std::size_t k = 0; k < num_batches; ++k) {
    std::int64_t const begin = starts[k];
    std::int64_t const end   = (k + 1 < starts.size()) ? starts[k + 1] : sorted_count;
    std::int64_t const s_k   = end - begin;
    sel->out_base[k]         = begin;
    auto& bsel               = sel->batches[k];
    if (s_k == 0) {
      bsel.rows.num_rows = layout.batch_rows[k];
      continue;
    }
    rmm::device_buffer local(static_cast<std::size_t>(s_k) * sizeof(std::uint32_t), stream, mr);
    sirius::codegen::global_slice_to_local(sorted_ids + begin, s_k,
                                           layout.batch_row_start[k],
                                           static_cast<std::uint32_t*>(local.data()), stream);
    bsel.rows = sirius::codegen::bucket_sorted_local_ids(
      static_cast<std::uint32_t const*>(local.data()), s_k, layout.batch_rows[k], stream, mr);
    bsel.local_indices =
      rmm::device_buffer(static_cast<std::size_t>(s_k) * sizeof(std::int32_t), stream, mr);
    sirius::codegen::row_set_to_local_indices(
      bsel.rows.view(), static_cast<std::int32_t*>(bsel.local_indices.data()), stream);
    bsel.density =
      layout.batch_rows[k] > 0 ? static_cast<double>(s_k) / layout.batch_rows[k] : 0.0;
  }
  sel->out_base[num_batches] = sorted_count;
  return sel;
}

}  // namespace

std::shared_ptr<prepared_selection> prepare_selection_from_batch(
  pinned_table_layout const& layout,
  std::span<sirius::late_mat::row_selection const> batch_selections,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (layout.batch_row_start.size() != layout.batch_rows.size() + 1) {
    fail("layout batch_row_start must have batch_rows.size()+1 entries");
  }
  if (batch_selections.size() != layout.batch_rows.size()) {
    fail("batch_selections must be parallel to layout.batch_rows");
  }

  auto sel    = std::make_shared<prepared_selection>();
  sel->layout = layout;
  auto const num_batches = layout.batch_rows.size();
  sel->batches.resize(num_batches);
  sel->out_base.assign(num_batches + 1, 0);

  std::int64_t total = 0;
  for (std::size_t k = 0; k < num_batches; ++k) {
    auto const& in         = batch_selections[k];
    auto& bsel             = sel->batches[k];
    std::int64_t const n_k = layout.batch_rows[k];
    sel->out_base[k]       = total;
    // A filled range must agree with the layout; an all-zero range is the
    // annotation carrier's "unset" and is accepted.
    if (in.range.rows != 0 && in.range.rows != n_k) {
      fail("batch selection range.rows disagrees with the layout");
    }
    // Dense selections may carry an unset (all-zero) range from the
    // annotation carrier — the layout is authoritative for their row count.
    std::int64_t const live =
      in.kind == sirius::late_mat::row_selection_kind::dense ? n_k : in.live_rows();
    if (live < 0 || live > n_k) { fail("batch selection live_rows out of range"); }
    bsel.rows.num_rows = n_k;
    bsel.density       = n_k > 0 ? static_cast<double>(live) / n_k : 0.0;
    if (live == 0) { continue; }

    switch (in.kind) {
      case sirius::late_mat::row_selection_kind::dense: {
        bsel.dense              = true;
        bsel.rows.num_survivors = n_k;
        break;
      }
      case sirius::late_mat::row_selection_kind::mask: {
        if (!in.mask_words || !in.chunk_offsets || in.survivor_count < 0) {
          fail("mask selection missing buffers or survivor_count");
        }
        bsel.mask_words         = in.mask_words;
        bsel.mask_chunk_offsets = in.chunk_offsets;
        bsel.rows.num_survivors = in.survivor_count;
        // int32 id expansion (gather map for uncompressed / tier_b routes),
        // via the shipped mask->indices wave kernel.
        bsel.local_indices =
          rmm::device_buffer(static_cast<std::size_t>(live) * sizeof(std::int32_t), stream, mr);
        selection_mask mask;
        mask.words          = static_cast<std::uint32_t*>(in.mask_words->data());
        mask.num_rows       = n_k;
        mask.survivor_count = in.survivor_count;
        mask.chunk_offsets  = static_cast<std::uint32_t*>(in.chunk_offsets->data());
        sirius::codegen::mask_to_row_indices(
          mask, static_cast<std::int32_t*>(bsel.local_indices.data()), stream);
        break;
      }
      case sirius::late_mat::row_selection_kind::id_list: {
        if (!in.row_ids || in.num_ids <= 0) { fail("id_list selection missing ids"); }
        // Batch-local ascending non-negative int32 == bit-identical uint32:
        // bucket directly (no sort, no split — already batch-scoped).
        bsel.rows = sirius::codegen::bucket_sorted_local_ids(
          static_cast<std::uint32_t const*>(in.row_ids->data()), in.num_ids, n_k, stream, mr);
        bsel.local_indices =
          rmm::device_buffer(static_cast<std::size_t>(live) * sizeof(std::int32_t), stream, mr);
        if (cudaMemcpyAsync(bsel.local_indices.data(), in.row_ids->data(),
                            static_cast<std::size_t>(live) * sizeof(std::int32_t),
                            cudaMemcpyDeviceToDevice, stream.value()) != cudaSuccess) {
          fail("id_list copy failed");
        }
        break;
      }
      default: fail("unknown row_selection kind");
    }
    total += live;
  }
  sel->out_base[num_batches] = total;
  sel->total_survivors       = total;
  sel->original_count        = total;  // per-batch forms are ascending: no restore
  if (total > std::numeric_limits<cudf::size_type>::max()) {
    fail("selection exceeds 2^31-1 rows — batch the materialization");
  }
  return sel;
}

std::unique_ptr<cudf::column> materialize(pinned_column_view const& origin,
                                          prepared_selection const& sel,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  if (origin.batches.size() != sel.layout.batch_rows.size()) {
    fail("origin/selection batch count mismatch");
  }
  if (origin.pin_generation != sel.layout.pin_generation) {
    fail("pin generation mismatch — origin re-pinned since prepare_selection");
  }

  // Raw-gather fast path (single-batch layouts): an uncompressed column
  // materializes as ONE direct cudf::gather with the borrowed u64 id list as
  // the map — gather needs neither sorted nor unique ids, output order is the
  // caller's id order, no restore. Stream-ordered end to end: no host sync
  // (the shipped-launcher syncs of the compressed routes never run here).
  // A compressed column canonicalizes once (call_once) and re-enters the
  // canonical path below.
  if (sel.raw_ids != nullptr) {
    if (sel.raw_count == 0) { return cudf::make_empty_column(origin.dtype); }
    auto const& src = origin.batches.front();
    if (src.compressed == nullptr) {
      if (src.uncompressed.null_count() != 0) {
        fail("nullable columns are not supported (v1)");
      }
      // Zero-copy UINT64 gather-map view over the borrowed ids (cudf's
      // index normalizer accepts any integral map type). DONT_CHECK: ids
      // are pin-order positions inside this batch by the annotation
      // contract.
      cudf::column_view const map{cudf::data_type{cudf::type_id::UINT64},
                                  static_cast<cudf::size_type>(sel.raw_count),
                                  static_cast<void const*>(sel.raw_ids),
                                  nullptr,
                                  0};
      return gather_one(src.uncompressed, map, stream, mr);
    }
    return materialize(origin, sel.canonical(stream, mr), stream, mr);
  }

  // Per-batch survivor columns, ascending global row order.
  std::vector<std::unique_ptr<cudf::column>> parts;
  parts.reserve(origin.batches.size());
  for (std::size_t k = 0; k < origin.batches.size(); ++k) {
    auto const& bsel = sel.batches[k];
    if (bsel.rows.num_survivors == 0) { continue; }
    auto const& src = origin.batches[k];
    if (src.num_rows != sel.layout.batch_rows[k]) { fail("origin batch row count drift"); }
    if (src.compressed != nullptr) {
      parts.push_back(materialize_compressed_batch(*src.compressed, src.column_index, bsel,
                                                   stream, mr));
    } else {
      // (a) Uncompressed pinned batch.
      if (src.uncompressed.null_count() != 0) {
        fail("nullable columns are not supported (v1)");
      }
      if (bsel.dense || bsel.rows.num_survivors == src.num_rows) {
        // Whole batch survives: identity map — deep copy, skip the gather.
        parts.push_back(std::make_unique<cudf::column>(src.uncompressed, stream, mr));
      } else {
        parts.push_back(gather_one(
          src.uncompressed, int32_map_view(bsel.local_indices, bsel.rows.num_survivors),
          stream, mr));
      }
    }
  }

  std::unique_ptr<cudf::column> merged;
  if (parts.empty()) {
    merged = cudf::make_empty_column(origin.dtype);
  } else if (parts.size() == 1) {
    merged = std::move(parts.front());
  } else {
    std::vector<cudf::column_view> views;
    views.reserve(parts.size());
    for (auto const& p : parts) {
      views.push_back(p->view());
    }
    merged = cudf::concatenate(views, stream, mr);
  }

  // Restore caller order/duplicates (gather semantics) when the input id
  // list was not sorted-unique.
  if (sel.needs_restore() && sel.original_count > 0) {
    return gather_one(merged->view(), int32_map_view(sel.restore_rank, sel.original_count),
                      stream, mr);
  }
  return merged;
}

}  // namespace sirius::late_mat
