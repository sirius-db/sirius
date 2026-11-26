// Differential test of sirius::agg_table (src/cuda/agg_table) against cudf::groupby.
//
// Every case builds batches on the host, folds them into a table from 1 or 4 threads (each with
// its own CUDA stream), sends the rows the table hands back down the cuDF path (cudf::groupby on
// the gathered rows, as the engine does), merges all partial rows on the host the way the
// engine's merge stage does (counts and sums added, MIN / MAX merged) and compares the result
// with cudf::groupby over the whole input. Where cuDF does not have SQL's semantics - integer
// sums that leave int64, NaN in float MIN / MAX - the reference is a host aggregation with
// 128-bit sums and DuckDB's float order instead.
//
// Float values are integers or half-integers, so float sums are exact in any order and are
// compared exactly; `random-doubles` is the one case compared with a tolerance.
//
//   agg_table_test [--filter substring] [--scale N] [--verbose]
//   agg_table_test --bench [--filter shape/aggs] [--bench-log2 24] [--bench-reps 4] [--bench-threads 1]
//     (a timing harness, not a test: see run_bench)
#include "agg_table.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/callback_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <new>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace at = sirius::agg_table;
using i128   = __int128;
using tid    = cudf::type_id;

namespace {

bool g_verbose = false;

// ---- host columns ---------------------------------------------------------------------------------
bool is_float(tid t) { return t == tid::FLOAT32 || t == tid::FLOAT64; }
std::size_t width(tid t) { return cudf::size_of(cudf::data_type{t}); }
bool is_unsigned(tid t) { return t == tid::UINT8 || t == tid::UINT16 || t == tid::UINT32 || t == tid::UINT64; }

struct hcol {
  cudf::data_type type{tid::INT64};
  std::vector<i128> i;         // integers, decimals (unscaled), timestamps
  std::vector<double> d;       // floats
  std::vector<uint8_t> valid;  // empty: no NULLs
  std::size_t size() const { return is_float(type.id()) ? d.size() : i.size(); }
  bool ok(std::size_t r) const { return valid.empty() || valid[r] != 0; }
};

std::unique_ptr<cudf::column> to_device(hcol const& h, ::cuda::stream_ref stream)
{
  std::size_t n = h.size(), w = width(h.type.id());
  std::vector<uint8_t> bytes(n * w + 16);
  for (std::size_t r = 0; r < n; ++r) {
    uint8_t* p = bytes.data() + r * w;
    switch (h.type.id()) {
      case tid::FLOAT32: {
        float f = static_cast<float>(h.d[r]);
        std::memcpy(p, &f, 4);
        break;
      }
      case tid::FLOAT64: std::memcpy(p, &h.d[r], 8); break;
      default: {
        i128 v = h.i[r];
        std::memcpy(p, &v, w);  // (little endian: the low bytes)
      }
    }
  }
  rmm::device_buffer data(n * w, stream);
  if (n != 0) cudaMemcpy(data.data(), bytes.data(), n * w, cudaMemcpyHostToDevice);
  rmm::device_buffer mask{0, stream};
  cudf::size_type nulls = 0;
  if (!h.valid.empty()) {
    std::vector<uint32_t> words((n + 31) / 32 + 16, 0);
    for (std::size_t r = 0; r < n; ++r) {
      if (h.valid[r]) {
        words[r / 32] |= 1u << (r % 32);
      } else {
        ++nulls;
      }
    }
    mask = cudf::create_null_mask(static_cast<cudf::size_type>(n), cudf::mask_state::UNINITIALIZED, stream);
    cudaMemcpy(mask.data(), words.data(), ((n + 31) / 32) * 4, cudaMemcpyHostToDevice);
    if (nulls == 0) mask = rmm::device_buffer{0, stream};
  }
  return std::make_unique<cudf::column>(h.type, static_cast<cudf::size_type>(n), std::move(data), std::move(mask),
                                        nulls);
}

hcol to_host(cudf::column_view const& c)
{
  hcol h;
  h.type        = c.type();
  std::size_t n = static_cast<std::size_t>(c.size()), w = width(c.type().id());
  std::vector<uint8_t> bytes(n * w + 16);
  if (n != 0) {
    cudaMemcpy(bytes.data(), static_cast<uint8_t const*>(c.head()) + static_cast<std::size_t>(c.offset()) * w, n * w,
               cudaMemcpyDeviceToHost);
  }
  bool f = is_float(c.type().id());
  if (f) {
    h.d.resize(n);
  } else {
    h.i.resize(n);
  }
  for (std::size_t r = 0; r < n; ++r) {
    uint8_t const* p = bytes.data() + r * w;
    switch (c.type().id()) {
      case tid::FLOAT32: {
        float v;
        std::memcpy(&v, p, 4);
        h.d[r] = v;
        break;
      }
      case tid::FLOAT64: std::memcpy(&h.d[r], p, 8); break;
      default: {
        if (w == 16) {
          std::memcpy(&h.i[r], p, 16);
        } else if (is_unsigned(c.type().id())) {
          uint64_t v = 0;
          std::memcpy(&v, p, w);
          h.i[r] = v;
        } else {
          int64_t v = 0;
          std::memcpy(&v, p, w);
          int sh = 64 - 8 * static_cast<int>(w);
          h.i[r] = (v << sh) >> sh;
        }
      }
    }
  }
  if (c.has_nulls()) {
    std::size_t bits = static_cast<std::size_t>(c.offset()) + n;
    std::vector<uint32_t> words((bits + 31) / 32);
    cudaMemcpy(words.data(), c.null_mask(), words.size() * 4, cudaMemcpyDeviceToHost);
    h.valid.resize(n);
    for (std::size_t r = 0; r < n; ++r) {
      std::size_t b = static_cast<std::size_t>(c.offset()) + r;
      h.valid[r]    = (words[b / 32] >> (b % 32)) & 1u;
    }
  }
  return h;
}

// ---- results on the host ------------------------------------------------------------------------
struct cell {
  bool valid = false;
  i128 i     = 0;
  double d   = 0;
};
// The total order of float64 the table uses (DuckDB): NaN greatest, -0.0 below +0.0.
int64_t f64_ord(double d)
{
  uint64_t b;
  std::memcpy(&b, &d, 8);
  if ((b & 0x7fffffffffffffffull) > 0x7ff0000000000000ull) return 0x7ff8000000000000ll;
  return static_cast<int64_t>(b ^ (static_cast<uint64_t>(static_cast<int64_t>(b) >> 63) >> 1));
}

struct agg_def {
  at::agg_kind kind;
  int col;
  bool f;  // float input
};

void merge_cell(cell& acc, cell const& v, agg_def const& a, bool raw_count)
{
  using k = at::agg_kind;
  if (a.kind == k::count_star || a.kind == k::count) {
    acc.valid = true;
    acc.i += raw_count ? (v.valid ? 1 : 0) : v.i;
    return;
  }
  if (!v.valid) return;
  if (!acc.valid) {
    acc = v;
    return;
  }
  if (a.kind == k::sum) {
    acc.i += v.i;
    acc.d += v.d;
  } else if (a.f) {
    bool take = a.kind == k::min ? f64_ord(v.d) < f64_ord(acc.d) : f64_ord(v.d) > f64_ord(acc.d);
    if (take) acc.d = v.d;
  } else {
    bool take = a.kind == k::min ? v.i < acc.i : v.i > acc.i;
    if (take) acc.i = v.i;
  }
}

struct key_hash {
  std::size_t operator()(i128 k) const
  {
    auto x = static_cast<uint64_t>(k) * 0x9E3779B97F4A7C15ull;
    return x ^ (x >> 29);
  }
};
struct result {
  std::unordered_map<i128, std::vector<cell>, key_hash> groups;
  bool has_null = false;
  std::vector<cell> null_group;
  std::vector<cell>& row(bool key_ok, i128 key, std::size_t naggs)
  {
    if (!key_ok) {
      if (!has_null) null_group.assign(naggs, cell{});
      has_null = true;
      return null_group;
    }
    auto it = groups.find(key);
    if (it == groups.end()) it = groups.emplace(key, std::vector<cell>(naggs)).first;
    return it->second;
  }
};

// Partial rows `[key, one column per aggregate]` (of the table, or of cudf::groupby) into `res`.
void absorb(result& res, cudf::table_view const& t, std::vector<agg_def> const& aggs)
{
  hcol key = to_host(t.column(0));
  std::vector<hcol> cols;
  for (std::size_t a = 0; a < aggs.size(); ++a) cols.push_back(to_host(t.column(static_cast<int>(1 + a))));
  for (std::size_t r = 0; r < key.size(); ++r) {
    auto& row = res.row(key.ok(r), key.i[r], aggs.size());
    for (std::size_t a = 0; a < aggs.size(); ++a) {
      cell v;
      v.valid = cols[a].ok(r);
      if (is_float(cols[a].type.id())) {
        v.d = cols[a].d[r];
      } else {
        v.i = cols[a].i[r];
      }
      merge_cell(row[a], v, aggs[a], false);
    }
  }
}

// ---- a test case ----------------------------------------------------------------------------------
struct batch {
  hcol key;
  std::vector<hcol> vals;
  std::vector<at::column_bound> bounds;
};
struct test_case {
  std::string name;
  at::config cfg;
  std::vector<batch> batches;
  int threads        = 1;
  int flush_every    = 0;      // rounds between flushes (0: only at the end)
  std::size_t max_rows = 0;    // of a flush chunk
  bool host_ref      = false;  // the reference is the host aggregation, not cudf::groupby
  double sum_tol     = 0;      // relative tolerance of float sums (0: exact)
  bool expect_unapplied = false;
  bool expect_wide   = false;
  bool replay_done   = false;  // every fold is called again with its finished ticket
  int fail_after     = 0;      // table::test_fail_after_mutation before the folds
  bool flush_by_next = false;  // flush with flush_next()
  std::string expect;          // substring the table's description must contain
  std::string expect2;         // ... and a second one
};

std::vector<agg_def> defs_of(at::config const& cfg)
{
  std::vector<agg_def> d;
  for (auto const& a : cfg.aggs) d.push_back({a.kind, a.value_column, is_float(a.input.id())});
  return d;
}

// cudf::groupby over (key, values): the aggregates of `cfg`, as partial rows `[key, columns]`.
std::unique_ptr<cudf::table> cudf_aggregate(at::config const& cfg, cudf::column_view const& key,
                                            cudf::table_view const& vals, ::cuda::stream_ref stream)
{
  cudf::groupby::groupby gb(cudf::table_view{{key}}, cudf::null_policy::INCLUDE);
  std::vector<cudf::groupby::aggregation_request> reqs;
  std::vector<std::unique_ptr<cudf::column>> keep;
  for (auto const& a : cfg.aggs) {
    cudf::groupby::aggregation_request r;
    using k = at::agg_kind;
    if (a.kind == k::count_star) {
      r.values = key;
      r.aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE));
    } else {
      r.values = vals.column(a.value_column);
      if (a.kind == k::sum && a.input.id() == tid::DECIMAL32) {  // (cuDF sums DECIMAL32 in 32 bits)
        keep.push_back(cudf::cast(r.values, cudf::data_type{tid::DECIMAL64, a.input.scale()}, stream));
        r.values = keep.back()->view();
      }
      if (a.kind == k::count) r.aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
      if (a.kind == k::sum) r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      if (a.kind == k::min) r.aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
      if (a.kind == k::max) r.aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
    }
    reqs.push_back(std::move(r));
  }
  auto out  = gb.aggregate(reqs, stream);
  auto cols = out.first->release();
  for (auto& r : out.second) cols.push_back(std::move(r.results[0]));
  return std::make_unique<cudf::table>(std::move(cols));
}

// The host reference: SQL semantics, 128-bit integer sums. `only`: the rows to take (null: all).
void host_aggregate(result& res, at::config const& cfg, batch const& b, std::vector<i128> const* only = nullptr)
{
  auto defs = defs_of(cfg);
  std::size_t const count = only != nullptr ? only->size() : b.key.size();
  for (std::size_t x = 0; x < count; ++x) {
    std::size_t r = only != nullptr ? static_cast<std::size_t>((*only)[x]) : x;
    auto& row = res.row(b.key.ok(r), b.key.i[r], defs.size());
    for (std::size_t a = 0; a < defs.size(); ++a) {
      cell v;
      if (defs[a].kind == at::agg_kind::count_star) {
        v.valid = true;
      } else {
        auto const& c = b.vals[static_cast<std::size_t>(defs[a].col)];
        v.valid       = c.ok(r);
        if (defs[a].f) {
          // (FLOAT32 inputs: the value the column holds.)
          v.d = c.type.id() == tid::FLOAT32 ? static_cast<double>(static_cast<float>(c.d[r])) : c.d[r];
        } else {
          v.i = c.i[r];
        }
      }
      merge_cell(row[a], v, defs[a], true);
    }
  }
}

std::string show(cell const& c, bool f)
{
  if (!c.valid) return "NULL";
  char buf[96];
  if (f) {
    std::snprintf(buf, sizeof buf, "%.17g", c.d);
  } else {
    std::snprintf(buf, sizeof buf, "%lld*2^64+%llu", static_cast<long long>(c.i >> 64),
                  static_cast<unsigned long long>(c.i));
  }
  return buf;
}

bool same_cell(cell const& a, cell const& b, agg_def const& d, double tol)
{
  if (a.valid != b.valid) return false;
  if (!a.valid) return true;
  if (!d.f || d.kind == at::agg_kind::count || d.kind == at::agg_kind::count_star) return a.i == b.i;
  if (std::isnan(a.d) || std::isnan(b.d)) return std::isnan(a.d) && std::isnan(b.d);
  if (d.kind == at::agg_kind::sum) {
    if (tol == 0) return a.d == b.d;
    double m = std::max({1.0, std::fabs(a.d), std::fabs(b.d)});
    return std::fabs(a.d - b.d) <= tol * m;
  }
  if (a.d == 0 && b.d == 0) return true;  // (-0.0 and +0.0 are equal in SQL; cuDF returns either)
  return std::memcmp(&a.d, &b.d, 8) == 0;
}

std::string compare(result const& got, result const& ref, std::vector<agg_def> const& defs, double tol)
{
  if (got.groups.size() != ref.groups.size() || got.has_null != ref.has_null) {
    return "groups: got " + std::to_string(got.groups.size()) + (got.has_null ? "+NULL" : "") + ", expected " +
           std::to_string(ref.groups.size()) + (ref.has_null ? "+NULL" : "");
  }
  auto row_diff = [&](std::vector<cell> const& g, std::vector<cell> const& r, std::string const& key) {
    for (std::size_t a = 0; a < defs.size(); ++a) {
      if (!same_cell(g[a], r[a], defs[a], tol)) {
        return "key " + key + " aggregate " + std::to_string(a) + ": got " + show(g[a], defs[a].f) + ", expected " +
               show(r[a], defs[a].f);
      }
    }
    return std::string();
  };
  for (auto const& [k, r] : ref.groups) {
    auto it = got.groups.find(k);
    if (it == got.groups.end()) return "missing key " + std::to_string(static_cast<long long>(k));
    auto d = row_diff(it->second, r, std::to_string(static_cast<long long>(k)));
    if (!d.empty()) return d;
  }
  if (ref.has_null) return row_diff(got.null_group, ref.null_group, "NULL");
  return {};
}

struct dev_batch {
  std::unique_ptr<cudf::column> key;
  std::vector<std::unique_ptr<cudf::column>> vals;
  cudf::table_view values() const
  {
    std::vector<cudf::column_view> v;
    for (auto const& c : vals) v.push_back(c->view());
    return cudf::table_view{v};
  }
};
dev_batch upload(batch const& b)
{
  dev_batch d;
  d.key = to_device(b.key, cudf::get_default_stream());
  for (auto const& v : b.vals) d.vals.push_back(to_device(v, cudf::get_default_stream()));
  return d;
}

struct run_stats {
  uint64_t unapplied = 0, unapplied_partial = 0, replays = 0, flush_replays = 0, chunks = 0, wide_chunks = 0;
};

// Checks a flush chunk against the partial schema; returns an error or "".
std::string check_chunk(test_case const& tc, cudf::table_view const& t, run_stats& st)
{
  auto schema = at::partial_schema(tc.cfg);
  if (static_cast<std::size_t>(t.num_columns()) != schema.size()) return "flush: wrong column count";
  if (tc.max_rows != 0 && static_cast<std::size_t>(t.num_rows()) > tc.max_rows) return "flush: chunk above max_rows";
  for (std::size_t c = 0; c < schema.size(); ++c) {
    auto ty = t.column(static_cast<int>(c)).type();
    if (ty == schema[c]) continue;
    bool int_sum = c > 0 && tc.cfg.aggs[c - 1].kind == at::agg_kind::sum && schema[c].id() == tid::INT64;
    if (int_sum && ty == cudf::data_type{tid::DECIMAL128, 0}) {
      st.wide_chunks += 1;
      continue;
    }
    return "flush: column " + std::to_string(c) + " has the wrong type";
  }
  st.chunks += 1;
  return {};
}

// Folds one batch with replay on allocation failure; the rows the table hands back go through
// cudf::groupby. Returns an error or "".
std::string fold_batch(at::table& table, test_case const& tc, batch const& hb, dev_batch const& db, result& res,
                       std::mutex& res_m, run_stats& st, ::cuda::stream_ref stream)
{
  at::fold_ticket ticket;
  at::fold_result fr;
  for (int attempt = 0;; ++attempt) {
    try {
      fr = table.fold(db.key->view(), db.values(), hb.bounds, ticket, stream);
      break;
    } catch (std::bad_alloc const&) {
      st.replays += 1;
      if (attempt > 10000) return "fold: replay does not converge";
    }
  }
  if (ticket.st != at::fold_ticket::state::done) return "fold: ticket not done";
  uint64_t left = fr.unapplied ? static_cast<uint64_t>(fr.unapplied->size()) : 0;
  if (tc.replay_done) {
    // A finished ticket passed again: the same result, nothing applied twice.
    auto again = table.fold(db.key->view(), db.values(), hb.bounds, ticket, stream);
    uint64_t l2 = again.unapplied ? static_cast<uint64_t>(again.unapplied->size()) : 0;
    uint64_t p1 = fr.unapplied_partial ? static_cast<uint64_t>(fr.unapplied_partial->num_rows()) : 0;
    uint64_t p2 = again.unapplied_partial ? static_cast<uint64_t>(again.unapplied_partial->num_rows()) : 0;
    if (l2 != left || p1 != p2 || again.rows_applied != fr.rows_applied) return "fold: the replay of a done ticket differs";
    fr = std::move(again);
  }
  if (fr.rows_applied + left != hb.key.size()) return "fold: rows_applied + unapplied != rows";
  auto defs = defs_of(tc.cfg);
  if (left != 0 && tc.host_ref) {
    // (cuDF's int64 SUM wraps; the engine's cuDF path widens such sums itself. Here the rows the
    // table handed back are aggregated on the host instead.)
    hcol map = to_host(fr.unapplied->view());
    std::lock_guard<std::mutex> lk(res_m);
    host_aggregate(res, tc.cfg, hb, &map.i);
    st.unapplied += left;
  } else if (left != 0) {
    auto gk  = cudf::gather(cudf::table_view{{db.key->view()}}, fr.unapplied->view(),
                            cudf::out_of_bounds_policy::DONT_CHECK, stream);
    auto gv  = cudf::gather(db.values(), fr.unapplied->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream);
    auto agg = cudf_aggregate(tc.cfg, gk->view().column(0), gv->view(), stream);
    std::lock_guard<std::mutex> lk(res_m);
    absorb(res, agg->view(), defs);
    st.unapplied += left;
  }
  if (fr.unapplied_partial) {
    std::lock_guard<std::mutex> lk(res_m);
    absorb(res, fr.unapplied_partial->view(), defs);
    st.unapplied_partial += static_cast<uint64_t>(fr.unapplied_partial->num_rows());
  }
  return {};
}

std::string flush_table(at::table& table, test_case const& tc, result& res, run_stats& st,
                        std::vector<std::unique_ptr<cudf::table>>* keep = nullptr)
{
  std::vector<std::unique_ptr<cudf::table>> chunks;
  for (int attempt = 0;; ++attempt) {
    try {
      if (tc.flush_by_next) {
        while (auto c = table.flush_next(tc.max_rows, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref())) {
          chunks.push_back(std::move(c));
        }
      } else {
        chunks = table.flush(tc.max_rows, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref());
      }
      break;
    } catch (std::bad_alloc const&) {
      st.flush_replays += 1;
      if (attempt > 10000) return "flush: replay does not converge";
    }
  }
  if (table.groups() != 0 || table.device_bytes() > 4096) return "flush: the table is not empty afterwards";
  auto defs = defs_of(tc.cfg);
  // Every key once per flush.
  result own;
  uint64_t rows = 0;
  for (auto& c : chunks) {
    auto e = check_chunk(tc, c->view(), st);
    if (!e.empty()) return e;
    absorb(own, c->view(), defs);
    absorb(res, c->view(), defs);
    rows += static_cast<uint64_t>(c->num_rows());
    if (keep != nullptr) keep->push_back(std::move(c));
  }
  if (own.groups.size() + (own.has_null ? 1 : 0) != rows) return "flush: a key appears in more than one row";
  return {};
}

result reference(test_case const& tc, std::vector<dev_batch> const& dev)
{
  result ref;
  auto defs = defs_of(tc.cfg);
  if (tc.host_ref) {
    for (auto const& b : tc.batches) host_aggregate(ref, tc.cfg, b);
  } else {
    // cudf::groupby per batch, merged on the host (one groupby over the concatenation would need
    // the whole input on the device at once).
    for (auto const& d : dev) {
      auto agg = cudf_aggregate(tc.cfg, d.key->view(), d.values(), cudf::get_default_stream());
      absorb(ref, agg->view(), defs);
    }
  }
  return ref;
}

// Runs one case against a table using `mr`. Returns an error or "".
std::string run_case(test_case const& tc, rmm::device_async_resource_ref mr, run_stats& st, std::string* desc = nullptr,
                     result const* known_ref = nullptr, result* ref_out = nullptr)
{
  std::string why;
  if (!at::eligible(tc.cfg, &why)) return "not eligible: " + why;
  std::vector<dev_batch> dev;
  for (auto const& b : tc.batches) dev.push_back(upload(b));
  auto table = at::table::create(tc.cfg, mr);
  if (tc.fail_after != 0) table->test_fail_after_mutation(tc.fail_after);
  result got;
  std::mutex res_m;
  std::vector<rmm::cuda_stream> streams(static_cast<std::size_t>(tc.threads));
  std::size_t next = 0;
  int round        = 0;
  while (next < dev.size()) {
    std::size_t end = std::min(dev.size(), next + static_cast<std::size_t>(tc.threads));
    std::vector<std::string> errs(end - next);
    std::vector<std::thread> th;
    for (std::size_t b = next; b < end; ++b) {
      th.emplace_back([&, b] {
        try {
          errs[b - next] = fold_batch(*table, tc, tc.batches[b], dev[b], got, res_m, st,
                                      ::cuda::stream_ref{streams[b - next].value()});
        } catch (std::exception const& e) {
          errs[b - next] = std::string("exception: ") + e.what();
        }
      });
    }
    for (auto& t : th) t.join();
    for (auto const& e : errs) {
      if (!e.empty()) return e;
    }
    next = end;
    ++round;
    if (tc.flush_every != 0 && round % tc.flush_every == 0 && next < dev.size()) {
      auto e = flush_table(*table, tc, got, st);
      if (!e.empty()) return e;
    }
  }
  if (desc != nullptr) *desc = table->describe();
  auto e = flush_table(*table, tc, got, st);
  if (!e.empty()) return e;
  if (known_ref != nullptr) return compare(got, *known_ref, defs_of(tc.cfg), tc.sum_tol);
  result ref = reference(tc, dev);
  auto err   = compare(got, ref, defs_of(tc.cfg), tc.sum_tol);
  if (ref_out != nullptr) *ref_out = std::move(ref);
  return err;
}

// ---- generators -----------------------------------------------------------------------------------
struct rng {
  std::mt19937_64 g;
  explicit rng(uint64_t seed) : g(seed) {}
  uint64_t u(uint64_t n) { return n == 0 ? 0 : g() % n; }
  double unit() { return static_cast<double>(g() >> 11) * (1.0 / 9007199254740992.0); }
};

i128 key_value(tid t, uint64_t id)
{
  // Group id -> a key of the type, scattered over its range (signed types get negative keys too).
  uint64_t x = id * 0x9E3779B97F4A7C15ull + 0x1234567;
  switch (t) {
    case tid::INT8: return static_cast<int8_t>(id);
    case tid::UINT8: return static_cast<uint8_t>(id);
    case tid::INT16: return static_cast<int16_t>(id * 3);
    case tid::UINT16: return static_cast<uint16_t>(id * 3);
    case tid::INT32:
    case tid::TIMESTAMP_DAYS:
    case tid::DURATION_DAYS:
    case tid::DECIMAL32: return static_cast<int32_t>(static_cast<uint32_t>(id) * 2654435761u);
    case tid::UINT32: return static_cast<uint32_t>(id) * 2654435761u;
    case tid::UINT64: return x;
    default: return static_cast<int64_t>(x);
  }
}
uint64_t key_domain(tid t)
{
  switch (t) {
    case tid::INT8:
    case tid::UINT8: return 200;
    case tid::INT16:
    case tid::UINT16: return 20000;
    default: return ~0ull;
  }
}

enum class kdist { uniform, hot, clustered, zipf };
struct key_opt {
  kdist dist       = kdist::uniform;
  uint64_t groups  = 1000;
  double null_frac = 0;
  double reserved_frac = 0;  // rows with key -1 / -2 (64-bit keys)
  bool reserved_runs = false;
  uint64_t run     = 64;     // clustered: rows per run
};

hcol make_keys(cudf::data_type type, std::size_t n, uint64_t first_row, key_opt const& o, rng& R)
{
  hcol h;
  h.type          = type;
  uint64_t groups = std::min(o.groups, key_domain(type.id()));
  h.i.resize(n);
  if (o.null_frac > 0) h.valid.assign(n, 1);
  for (std::size_t r = 0; r < n; ++r) {
    uint64_t id = 0;
    switch (o.dist) {
      case kdist::uniform: id = R.u(groups); break;
      case kdist::hot: id = R.unit() < 0.9 ? R.u(16) * 977 % groups : R.u(groups); break;
      case kdist::clustered: id = ((first_row + r) / o.run) % groups; break;
      case kdist::zipf: {
        double x = R.unit();
        id       = static_cast<uint64_t>(static_cast<double>(groups) * x * x * x * x) % groups;
        break;
      }
    }
    h.i[r] = key_value(type.id(), id);
    if (o.reserved_frac > 0) {
      bool hit = o.reserved_runs ? ((first_row + r) / 4096) % 4 == 1 : R.unit() < o.reserved_frac;
      if (hit) h.i[r] = type.id() == tid::UINT64 ? static_cast<i128>(~0ull - R.u(2)) : -1 - static_cast<i128>(R.u(2));
    }
    if (o.null_frac > 0 && R.unit() < o.null_frac) {
      h.valid[r] = 0;
      h.i[r]     = static_cast<int64_t>(R.g());  // (the data under a NULL is arbitrary)
      if (width(type.id()) < 8) h.i[r] &= (static_cast<i128>(1) << (8 * width(type.id()) - 1)) - 1;
    }
  }
  return h;
}

enum class vdist { small, nan_mix, big, big_late, cancel, random_double };
struct val_opt {
  vdist dist       = vdist::small;
  double null_frac = 0;
  bool null_groups = false;  // every value of the keys with (key % 5 == 0) is NULL
};

hcol make_values(cudf::data_type type, hcol const& key, uint64_t first_row, uint64_t total_rows, val_opt const& o,
                 rng& R)
{
  hcol h;
  h.type        = type;
  std::size_t n = key.size();
  bool f        = is_float(type.id());
  if (f) {
    h.d.resize(n);
  } else {
    h.i.resize(n);
  }
  if (o.null_frac > 0 || o.null_groups) h.valid.assign(n, 1);
  int64_t lim = type.id() == tid::INT8 ? 100 : 1000;
  for (std::size_t r = 0; r < n; ++r) {
    uint64_t kh = static_cast<uint64_t>(key.i[r]) * 0x9E3779B97F4A7C15ull >> 40;
    if (f) {
      double v = static_cast<double>(static_cast<int64_t>(R.u(33)) - 16) * 0.5;
      if (o.dist == vdist::nan_mix) {
        // By key: groups of NaNs only, of some NaNs, +-0.0 only, +-inf, negative values.
        switch (kh % 7) {
          case 0: v = std::nan(""); break;
          case 1: v = R.u(3) == 0 ? std::nan("") : v; break;
          case 2: v = R.u(2) == 0 ? 0.0 : -0.0; break;
          case 3: v = R.u(4) == 0 ? INFINITY : v; break;
          case 4: v = R.u(4) == 0 ? -INFINITY : -std::fabs(v); break;
          case 5: v = R.u(2) == 0 ? -std::nan("") : INFINITY; break;
          default: break;
        }
      } else if (o.dist == vdist::random_double) {
        v = (R.unit() - 0.5) * 1e6;
      }
      h.d[r] = v;
    } else {
      i128 v = static_cast<int64_t>(R.u(2 * static_cast<uint64_t>(lim) + 1)) - lim;
      if (width(type.id()) == 8) {
        bool late = first_row + r >= total_rows - total_rows / 8;
        switch (o.dist) {
          case vdist::big:  // by key: near +2^62, near -2^62, alternating near INT64_MAX / INT64_MIN
            switch (kh % 4) {
              case 0: v = (static_cast<i128>(1) << 62) - static_cast<i128>(R.u(1000)); break;
              case 1: v = -(static_cast<i128>(1) << 62) + static_cast<i128>(R.u(1000)); break;
              case 2: v = R.u(2) == 0 ? static_cast<i128>(INT64_MAX) - static_cast<i128>(R.u(5)) : static_cast<i128>(INT64_MIN) + static_cast<i128>(R.u(5)); break;
              default: break;
            }
            break;
          case vdist::big_late:
            if (late) v = (static_cast<i128>(1) << 61) + static_cast<i128>(R.u(1ull << 40));
            break;
          case vdist::cancel: v = (r & 1) != 0 ? (static_cast<i128>(1) << 62) : -(static_cast<i128>(1) << 62); break;
          default: break;
        }
      }
      h.i[r] = v;
    }
    bool null = (o.null_frac > 0 && R.unit() < o.null_frac) || (o.null_groups && kh % 5 == 0);
    if (null) {
      h.valid[r] = 0;
      if (f) {
        h.d[r] = 12345.5;
      } else {
        h.i[r] = type.id() == tid::INT8 ? 77 : 12345;
      }
    }
  }
  return h;
}

at::column_bound exact_bound(hcol const& c)
{
  at::column_bound b;
  if (is_float(c.type.id()) || c.size() == 0) return b;
  b.known = true;
  b.min   = INT64_MAX;
  b.max   = INT64_MIN;
  for (std::size_t r = 0; r < c.size(); ++r) {
    if (!c.ok(r)) continue;
    b.min = std::min<int64_t>(b.min, static_cast<int64_t>(c.i[r]));
    b.max = std::max<int64_t>(b.max, static_cast<int64_t>(c.i[r]));
  }
  if (b.min > b.max) b.known = false;
  return b;
}

struct case_opt {
  std::size_t batch_rows = 1 << 16;
  int batches            = 8;
  key_opt keys;
  std::vector<val_opt> vals;  // per value column (default: small, no NULLs)
  int stats              = 0;  // 0 none, 1 exact bounds on every integer column, 2 on every second batch
  uint64_t seed          = 1;
};

test_case make_case(std::string name, cudf::data_type key_type, std::vector<cudf::data_type> const& val_types,
                    std::vector<at::agg_spec> aggs, case_opt const& o)
{
  test_case tc;
  tc.name         = std::move(name);
  tc.cfg.key_type = key_type;
  tc.cfg.aggs     = std::move(aggs);
  rng R(o.seed * 7919 + 13);
  uint64_t total = static_cast<uint64_t>(o.batch_rows) * static_cast<uint64_t>(o.batches);
  for (int b = 0; b < o.batches; ++b) {
    batch B;
    uint64_t first = static_cast<uint64_t>(b) * o.batch_rows;
    B.key          = make_keys(key_type, o.batch_rows, first, o.keys, R);
    for (std::size_t c = 0; c < val_types.size(); ++c) {
      val_opt vo = c < o.vals.size() ? o.vals[c] : val_opt{};
      B.vals.push_back(make_values(val_types[c], B.key, first, total, vo, R));
      at::column_bound cb;
      if (o.stats == 1 || (o.stats == 2 && b % 2 == 0)) cb = exact_bound(B.vals.back());
      B.bounds.push_back(cb);
    }
    tc.batches.push_back(std::move(B));
  }
  return tc;
}

at::agg_spec A(at::agg_kind k, int col, cudf::data_type t) { return at::agg_spec{k, col, t}; }
at::agg_spec CS() { return at::agg_spec{at::agg_kind::count_star, -1, cudf::data_type{tid::INT64}}; }

// ---- allocation failures ------------------------------------------------------------------------
// A memory resource that fails its `fail_at`-th allocation (once).
struct fault_state {
  std::atomic<int64_t> count{0};
  int64_t fail_at = -1;
  std::atomic<int64_t> injected{0};
};
rmm::mr::callback_memory_resource make_fault_mr(fault_state* fs)
{
  auto up = rmm::mr::get_current_device_resource_ref();
  return rmm::mr::callback_memory_resource(
    [fs, up](std::size_t bytes, cuda::stream_ref stream, void*) mutable -> void* {
      int64_t c = fs->count.fetch_add(1);
      if (c == fs->fail_at) {
        fs->injected += 1;
        throw std::bad_alloc();
      }
      return up.allocate(stream, bytes);
    },
    [up](void* p, std::size_t bytes, cuda::stream_ref stream, void*) mutable { up.deallocate(stream, p, bytes); });
}

int g_fail = 0, g_run = 0;
void report(std::string const& name, std::string const& err, run_stats const& st, std::string const& desc)
{
  ++g_run;
  if (!err.empty()) ++g_fail;
  if (!err.empty() || g_verbose) {
    std::printf("%s %-44s unapplied=%llu unapplied_partial=%llu chunks=%llu wide_chunks=%llu replays=%llu %s%s%s\n",
                err.empty() ? "ok  " : "FAIL", name.c_str(), static_cast<unsigned long long>(st.unapplied),
                static_cast<unsigned long long>(st.unapplied_partial), static_cast<unsigned long long>(st.chunks),
                static_cast<unsigned long long>(st.wide_chunks),
                static_cast<unsigned long long>(st.replays + st.flush_replays), desc.c_str(), err.empty() ? "" : " :: ",
                err.c_str());
    std::fflush(stdout);
  }
}

void run_plain(test_case const& tc)
{
  run_stats st;
  std::string desc, err;
  try {
    err = run_case(tc, rmm::mr::get_current_device_resource_ref(), st, &desc);
  } catch (std::exception const& e) {
    err = std::string("exception: ") + e.what();
  }
  if (err.empty() && tc.expect_unapplied && st.unapplied + st.unapplied_partial == 0) err = "expected unapplied rows";
  if (err.empty() && tc.expect_wide && st.wide_chunks == 0) err = "expected a 128-bit sum column";
  if (err.empty() && !tc.expect.empty() && desc.find(tc.expect) == std::string::npos) err = "expected '" + tc.expect + "'";
  if (err.empty() && !tc.expect2.empty() && desc.find(tc.expect2) == std::string::npos) err = "expected '" + tc.expect2 + "'";
  report(tc.name, err, st, desc);
}

// The case with an allocation failure injected at every allocation point (sampled when there are
// more than `max_points`), each replayed with the same ticket.
void run_faults(test_case const& tc, int64_t max_points)
{
  fault_state fs;
  auto mr = make_fault_mr(&fs);
  run_stats base;
  result ref;
  std::string err = run_case(tc, mr, base, nullptr, nullptr, &ref);
  int64_t total   = fs.count.load();
  int64_t step    = total > max_points ? (total + max_points - 1) / max_points : 1;
  uint64_t injected = 0, replays = 0, points = 0;
  for (int64_t k = 0; err.empty() && k < total; k += step) {
    fault_state f2;
    f2.fail_at = k;
    auto mr2   = make_fault_mr(&f2);
    run_stats st;
    try {
      err = run_case(tc, mr2, st, nullptr, &ref);
    } catch (std::exception const& e) {
      err = std::string("exception: ") + e.what();
    }
    if (!err.empty()) err = "allocation " + std::to_string(k) + " of " + std::to_string(total) + ": " + err;
    injected += static_cast<uint64_t>(f2.injected.load());
    replays += st.replays + st.flush_replays;
    ++points;
  }
  run_stats st;
  st.replays = replays;
  report(tc.name + " [faults]", err, st,
         "points=" + std::to_string(points) + " of " + std::to_string(total) + " injected=" + std::to_string(injected));
}

// Two tables alive at once, folded alternately from two threads.
void run_two(test_case const& a, test_case const& b)
{
  std::string ea, eb;
  run_stats sa, sb;
  std::thread ta([&] {
    try {
      ea = run_case(a, rmm::mr::get_current_device_resource_ref(), sa);
    } catch (std::exception const& e) {
      ea = std::string("exception: ") + e.what();
    }
  });
  std::thread tb([&] {
    try {
      eb = run_case(b, rmm::mr::get_current_device_resource_ref(), sb);
    } catch (std::exception const& e) {
      eb = std::string("exception: ") + e.what();
    }
  });
  ta.join();
  tb.join();
  report("two-tables/" + a.name, ea, sa, "");
  report("two-tables/" + b.name, eb, sb, "");
}

// Two tables fold the two halves of the input, flush; a third folds their partial rows (and the
// partial rows cuDF makes of what the first two handed back): it must equal the single table.
void run_partial_roundtrip(test_case const& tc)
{
  std::string err;
  run_stats st;
  try {
    auto mr   = rmm::mr::get_current_device_resource_ref();
    auto defs = defs_of(tc.cfg);
    std::vector<dev_batch> dev;
    for (auto const& b : tc.batches) dev.push_back(upload(b));
    auto t1 = at::table::create(tc.cfg, mr), t2 = at::table::create(tc.cfg, mr);
    at::config c3 = tc.cfg;
    c3.budget_bytes = 0;
    auto t3 = at::table::create(c3, mr);
    result got, side;
    std::mutex m;
    std::vector<std::unique_ptr<cudf::table>> parts;
    for (std::size_t b = 0; b < dev.size() && err.empty(); ++b) {
      at::table& t = b % 2 == 0 ? *t1 : *t2;
      at::fold_ticket ticket;
      auto fr = t.fold(dev[b].key->view(), dev[b].values(), tc.batches[b].bounds, ticket, cudf::get_default_stream());
      if (fr.unapplied && tc.host_ref) {
        hcol map = to_host(fr.unapplied->view());
        host_aggregate(got, tc.cfg, tc.batches[b], &map.i);
        st.unapplied += static_cast<uint64_t>(fr.unapplied->size());
      } else if (fr.unapplied) {
        auto gk = cudf::gather(cudf::table_view{{dev[b].key->view()}}, fr.unapplied->view());
        auto gv = cudf::gather(dev[b].values(), fr.unapplied->view());
        absorb(got, cudf_aggregate(tc.cfg, gk->view().column(0), gv->view(), cudf::get_default_stream())->view(), defs);
        st.unapplied += static_cast<uint64_t>(fr.unapplied->size());
      }
      if (fr.unapplied_partial) parts.push_back(std::move(fr.unapplied_partial));
    }
    if (err.empty()) err = flush_table(*t1, tc, side, st, &parts);
    if (err.empty()) err = flush_table(*t2, tc, side, st, &parts);
    for (auto& p : parts) {
      if (!err.empty()) break;
      at::fold_ticket ticket;
      auto fr = t3->fold_partial(p->view(), ticket, cudf::get_default_stream());
      if (fr.unapplied) {  // partial rows the third table could not take: merged as they are
        auto g = cudf::gather(p->view(), fr.unapplied->view());
        absorb(got, g->view(), defs);
        st.unapplied_partial += static_cast<uint64_t>(fr.unapplied->size());
      }
      // A replay of a finished ticket changes nothing.
      auto again = t3->fold_partial(p->view(), ticket, cudf::get_default_stream());
      if ((again.unapplied != nullptr) != (fr.unapplied != nullptr) || again.rows_applied != fr.rows_applied) {
        err = "fold_partial: the replay of a done ticket differs";
      }
    }
    if (err.empty()) err = flush_table(*t3, tc, got, st);
    if (err.empty()) err = compare(got, reference(tc, dev), defs, tc.sum_tol);
  } catch (std::exception const& e) {
    err = std::string("exception: ") + e.what();
  }
  report(tc.name + " [partial round trip]", err, st, "");
}

void check_eligibility()
{
  auto rej = [](std::string const& name, at::config cfg) {
    std::string why;
    bool ok = at::eligible(cfg, &why);
    report("eligible rejects " + name, ok ? "accepted" : (why.empty() ? "no reason given" : ""), run_stats{}, why);
  };
  auto dt = [](tid t) { return cudf::data_type{t}; };
  at::config base;
  base.key_type = dt(tid::INT64);
  base.aggs     = {CS()};
  for (tid t : {tid::FLOAT64, tid::FLOAT32, tid::STRING, tid::DECIMAL128, tid::BOOL8, tid::LIST, tid::STRUCT}) {
    at::config c = base;
    c.key_type   = dt(t);
    rej("key type " + std::to_string(static_cast<int>(t)), c);
  }
  for (tid t : {tid::STRING, tid::DECIMAL128, tid::UINT64, tid::UINT32, tid::BOOL8, tid::TIMESTAMP_DAYS}) {
    at::config c = base;
    c.aggs       = {A(at::agg_kind::sum, 0, dt(t))};
    rej("sum of type " + std::to_string(static_cast<int>(t)), c);
  }
  for (tid t : {tid::STRING, tid::DECIMAL128, tid::UINT64, tid::BOOL8}) {
    at::config c = base;
    c.aggs       = {A(at::agg_kind::min, 0, dt(t))};
    rej("min of type " + std::to_string(static_cast<int>(t)), c);
  }
  {
    at::config c = base;
    c.aggs       = {A(at::agg_kind::sum, -1, dt(tid::INT64))};
    rej("sum without a column", c);
    c.aggs.clear();
    for (int i = 0; i < 8; ++i) c.aggs.push_back(A(at::agg_kind::sum, i, dt(tid::INT64)));
    rej("nine state words", c);
    c              = base;
    c.budget_bytes = 1 << 20;
    rej("budget below the smallest table", c);
    c      = base;
    c.aggs = {A(at::agg_kind::sum, 0, dt(tid::INT64)), A(at::agg_kind::min, 0, dt(tid::INT32))};
    rej("two types for one column", c);
  }
}

// ---- timing harness (--bench [filter]) -------------------------------------------------------------
// Not a test: folds one batch of 2^`--bench-log2` rows (default 24) per key shape and aggregate set
// several times into one table and prints rows per second next to cudf::groupby on the same batch.
// For comparing kernels on one machine; see experimental/agg_bench/engine_bench for the engine.
uint64_t mix64(uint64_t x)
{
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
template <typename T>
std::unique_ptr<cudf::column> raw_column(cudf::data_type type, std::vector<T> const& v, double null_frac, uint64_t salt)
{
  auto stream = cudf::get_default_stream();
  rmm::device_buffer data(v.size() * sizeof(T), stream);
  cudaMemcpy(data.data(), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  rmm::device_buffer mask{0, stream};
  cudf::size_type nulls = 0;
  if (null_frac > 0) {
    std::vector<uint32_t> words((v.size() + 31) / 32 + 16, 0);
    for (std::size_t r = 0; r < v.size(); ++r) {
      if (static_cast<double>(mix64(r * 31 + salt) >> 11) * (1.0 / 9007199254740992.0) >= null_frac) {
        words[r / 32] |= 1u << (r % 32);
      } else {
        ++nulls;
      }
    }
    mask = cudf::create_null_mask(static_cast<cudf::size_type>(v.size()), cudf::mask_state::UNINITIALIZED, stream);
    cudaMemcpy(mask.data(), words.data(), ((v.size() + 31) / 32) * 4, cudaMemcpyHostToDevice);
  }
  return std::make_unique<cudf::column>(type, static_cast<cudf::size_type>(v.size()), std::move(data), std::move(mask), nulls);
}

int run_bench(std::string const& filter, int log2_rows, int reps, int threads)
{
  using k = at::agg_kind;
  auto dt = [](tid t, int32_t s = 0) { return cudf::data_type{t, s}; };
  const cudf::data_type I64 = dt(tid::INT64), F64 = dt(tid::FLOAT64), DEC = dt(tid::DECIMAL64, -2);
  std::size_t const n = static_cast<std::size_t>(1) << log2_rows;
  // (A pool, as in the engine: without one every scratch column is a cudaMalloc.)
  rmm::mr::set_current_device_resource(
    rmm::mr::pool_memory_resource(rmm::mr::cuda_memory_resource{}, std::size_t{8} << 30));
  struct shape {
    std::string name;
    std::function<uint64_t(uint64_t)> id;  // row -> group id
  };
  auto uni = [n](uint64_t g) { return [g](uint64_t r) { return mix64(r) % g; }; };
  auto run = [](uint64_t len) { return [len](uint64_t r) { return 1000 + r / len; }; };
  std::vector<shape> shapes = {
    {"runs350k", run(350000)},
    {"runs2000", run(2000)},
    {"runs64", run(64)},
    {"runs8", run(8)},
    {"g1", uni(1)},
    {"g4", uni(4)},
    {"g16", uni(16)},
    {"g40", uni(40)},
    {"g175", uni(175)},
    {"g3000", uni(3000)},
    {"g187500", uni(187500)},
    {"n32", uni(n / 32)},
    {"n4", uni(n / 4)},
    {"hot16", [n](uint64_t r) { uint64_t h = mix64(r); return h % 100 < 90 ? (h >> 8) % 16 : 16 + (h >> 8) % (n / 4); }},
    {"hot2", [n](uint64_t r) { uint64_t h = mix64(r); return h % 100 < 50 ? (h >> 8) % 2 : 16 + (h >> 8) % (n / 4); }},
    {"hot64-30", [n](uint64_t r) { uint64_t h = mix64(r); return h % 100 < 30 ? (h >> 8) % 64 : 64 + (h >> 8) % (n / 4); }},
    {"zipf", [n](uint64_t r) {
       double u = static_cast<double>(mix64(r) >> 11) * (1.0 / 9007199254740992.0);
       return static_cast<uint64_t>(std::pow(static_cast<double>(n / 4 + 1), u));
     }},
  };
  struct aggset {
    std::string name;
    std::vector<at::agg_spec> aggs;
    double null_d;  // NULL share of column d (2)
  };
  // Value columns: 0 v1 BIGINT, 1 v2 BIGINT, 2 d DOUBLE, 3 dec DECIMAL(15,2) (as the engine benchmark).
  std::vector<aggset> sets = {
    {"a8", {CS(), A(k::sum, 0, I64), A(k::min, 0, I64), A(k::max, 1, I64), A(k::sum, 2, F64), A(k::min, 2, F64),
            A(k::max, 2, F64), A(k::sum, 3, DEC)}, 0},
    {"canon8", {CS(), A(k::sum, 0, I64), A(k::min, 1, I64), A(k::max, 2, F64), A(k::sum, 2, F64), A(k::max, 0, I64),
                A(k::min, 2, F64), A(k::sum, 1, I64)}, 0},
    {"c4", {CS(), A(k::sum, 2, F64), A(k::min, 2, F64), A(k::max, 0, I64)}, 0},
    {"c4null", {CS(), A(k::sum, 2, F64), A(k::min, 2, F64), A(k::max, 0, I64)}, 0.1},
    {"sumd", {A(k::sum, 2, F64)}, 0},
    {"sumdnull", {A(k::sum, 2, F64)}, 0.1},
    {"minmax", {A(k::min, 0, I64), A(k::max, 1, I64), A(k::min, 2, F64)}, 0},
    {"a2", {CS(), A(k::sum, 0, I64)}, 0},
  };
  std::printf("%-10s %-9s %9s %9s %7s  %s\n", "shape", "aggs", "table M/s", "cudf M/s", "ratio", "table");
  for (auto const& sh : shapes) {
    std::vector<int64_t> key(n), v1(n), v2(n), dec(n);
    std::vector<double> d(n);
    for (std::size_t r = 0; r < n; ++r) {
      key[r] = static_cast<int64_t>((sh.id(r) * 0x9E3779B97F4A7C15ull) & 0x7fffffffffffffffull);
      uint64_t h = mix64(r ^ 0x55aa55aa);
      v1[r]  = static_cast<int64_t>(h % 1000000);
      v2[r]  = static_cast<int64_t>((h >> 20) % 2000000001ull) - 1000000000;
      d[r]   = static_cast<double>((h >> 7) % 2000) * 0.5;
      dec[r] = static_cast<int64_t>((h >> 13) % 10000000);
    }
    auto kc = raw_column(I64, key, 0, 0);
    for (auto const& s : sets) {
      std::string name = sh.name + "/" + s.name;
      if (!filter.empty() && name.find(filter) == std::string::npos) continue;
      std::vector<std::unique_ptr<cudf::column>> cols;
      cols.push_back(raw_column(I64, v1, 0, 1));
      cols.push_back(raw_column(I64, v2, 0, 2));
      cols.push_back(raw_column(F64, d, s.null_d, 3));
      cols.push_back(raw_column(DEC, dec, 0, 4));
      std::vector<cudf::column_view> cv;
      for (auto const& c : cols) cv.push_back(c->view());
      cudf::table_view vals{cv};
      at::config cfg;
      cfg.key_type = I64;
      cfg.aggs     = s.aggs;
      auto table   = at::table::create(cfg, rmm::mr::get_current_device_resource_ref());
      std::vector<rmm::cuda_stream> streams(static_cast<std::size_t>(threads));
      auto fold_all = [&](int times) {
        std::vector<std::thread> th;
        for (int t = 0; t < threads; ++t) {
          th.emplace_back([&, t] {
            for (int i = 0; i < times; ++i) {
              at::fold_ticket ticket;
              auto fr = table->fold(kc->view(), vals, {}, ticket, ::cuda::stream_ref{streams[static_cast<std::size_t>(t)].value()});
              if (fr.unapplied && s.null_d == 0) std::printf("  (unapplied rows: %d)\n", fr.unapplied->size());
            }
          });
        }
        for (auto& x : th) x.join();
      };
      fold_all(2);  // warm-up: the table grows to its size, kernels are loaded
      auto t0 = std::chrono::steady_clock::now();
      fold_all(reps);
      double tt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::string desc = table->describe();
      table.reset();
      (void)cudf_aggregate(cfg, kc->view(), vals, cudf::get_default_stream());
      t0 = std::chrono::steady_clock::now();
      int const creps = 2;
      for (int i = 0; i < creps; ++i) (void)cudf_aggregate(cfg, kc->view(), vals, cudf::get_default_stream());
      double tc = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      double mt = static_cast<double>(n) * reps * threads / tt / 1e6, mc = static_cast<double>(n) * creps / tc / 1e6;
      std::printf("%-10s %-9s %9.0f %9.0f %7.2f  %s\n", sh.name.c_str(), s.name.c_str(), mt, mc, mt / mc, desc.c_str());
      std::fflush(stdout);
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv)
{
  std::string filter;
  int scale = 1;
  bool bench = false;
  int bench_log2 = 24, bench_reps = 4, bench_threads = 1;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--filter" && i + 1 < argc) filter = argv[++i];
    if (a == "--scale" && i + 1 < argc) scale = std::atoi(argv[++i]);
    if (a == "--verbose") g_verbose = true;
    if (a == "--bench") bench = true;
    if (a == "--bench-log2" && i + 1 < argc) bench_log2 = std::atoi(argv[++i]);
    if (a == "--bench-reps" && i + 1 < argc) bench_reps = std::atoi(argv[++i]);
    if (a == "--bench-threads" && i + 1 < argc) bench_threads = std::atoi(argv[++i]);
  }
  if (bench) return run_bench(filter, bench_log2, bench_reps, bench_threads);
  auto want = [&](std::string const& name) { return filter.empty() || name.find(filter) != std::string::npos; };
  auto dt   = [](tid t, int32_t s = 0) { return cudf::data_type{t, s}; };
  using k   = at::agg_kind;
  const cudf::data_type I64 = dt(tid::INT64), F64 = dt(tid::FLOAT64);

  std::vector<test_case> cases;
  auto add = [&](test_case tc) { cases.push_back(std::move(tc)); };

  // The aggregate sets: the prototype's canonical sets (specialised kernels) and others.
  struct aggset {
    std::string name;
    std::vector<cudf::data_type> vals;
    std::vector<at::agg_spec> aggs;
  };
  std::vector<aggset> sets = {
    {"count", {}, {CS()}},
    {"count+sum", {I64}, {CS(), A(k::sum, 0, I64)}},
    {"avg(dec64)", {dt(tid::DECIMAL64, -2)}, {A(k::sum, 0, dt(tid::DECIMAL64, -2)), A(k::count, 0, dt(tid::DECIMAL64, -2))}},
    {"canon4", {I64, I64, F64}, {CS(), A(k::sum, 0, I64), A(k::min, 1, I64), A(k::max, 2, F64)}},
    {"canon8", {I64, I64, F64, F64},
     {CS(), A(k::sum, 0, I64), A(k::min, 1, I64), A(k::max, 2, F64), A(k::sum, 3, F64), A(k::max, 0, I64),
      A(k::min, 2, F64), A(k::sum, 1, I64)}},
    {"generic3", {F64, dt(tid::INT32)}, {A(k::sum, 0, F64), A(k::max, 1, dt(tid::INT32)), A(k::min, 1, dt(tid::INT32))}},
    {"two-sums", {I64, I64}, {A(k::sum, 0, I64), A(k::sum, 1, I64), A(k::count, 1, I64)}},
  };
  std::size_t const BR = static_cast<std::size_t>(1 << 16) * static_cast<std::size_t>(scale);

  // 1. Key distributions x aggregate sets x 1 and 4 threads.
  struct kd {
    std::string name;
    key_opt o;
    std::size_t rows;
    int batches;
  };
  std::vector<kd> kds = {
    {"few", {kdist::uniform, 100}, BR, 8},
    {"mid", {kdist::uniform, 20000}, BR, 8},
    {"many", {kdist::uniform, 600000}, 4 * BR, 8},
    {"hot", {kdist::hot, 300000}, 4 * BR, 8},
    {"zipf", {kdist::zipf, 400000}, 4 * BR, 6},
    {"clustered", {kdist::clustered, 1u << 30, 0, 0, false, 700}, 4 * BR, 6},
    {"runs", {kdist::clustered, 3000, 0, 0, false, 40000}, 4 * BR, 6},
  };
  for (auto const& s : sets) {
    for (auto const& d : kds) {
      for (int threads : {1, 4}) {
        case_opt o;
        o.batch_rows = d.rows;
        o.batches    = d.batches;
        o.keys       = d.o;
        o.seed       = cases.size() + 1;
        auto tc      = make_case("dist/" + s.name + "/" + d.name + "/t" + std::to_string(threads), I64, s.vals, s.aggs, o);
        tc.threads   = threads;
        add(std::move(tc));
      }
    }
  }
  // 2. SQL keys: reserved values rare / hot / in runs, NULL keys, both.
  for (auto const& s : sets) {
    int v = 0;
    for (key_opt ko : {key_opt{kdist::uniform, 50000, 0, 1.0 / 1024}, key_opt{kdist::uniform, 50000, 0, 0.4},
                       key_opt{kdist::uniform, 50000, 0, 0.25, true}, key_opt{kdist::uniform, 50000, 0.05, 0},
                       key_opt{kdist::hot, 300000, 0.02, 0.01}}) {
      static const char* names[] = {"rare", "hot", "runs", "nulls", "nulls+reserved"};
      case_opt o;
      o.batch_rows = 2 * BR;
      o.batches    = 6;
      o.keys       = ko;
      o.seed       = cases.size() + 1;
      auto tc      = make_case(std::string("sql-keys/") + names[v] + "/" + s.name, I64, s.vals, s.aggs, o);
      tc.threads   = v % 2 == 0 ? 1 : 4;
      tc.replay_done = v >= 2;
      tc.fail_after  = v == 4 ? 5 : 0;
      add(std::move(tc));
      ++v;
    }
  }
  {
    case_opt o;
    o.keys = key_opt{kdist::uniform, 50000, 0.03, 0.05};
    add(make_case("sql-keys/uint64-reserved", dt(tid::UINT64), {I64}, {CS(), A(k::sum, 0, I64)}, o));
  }
  // 3. NULL values: scattered, whole groups, both layouts (the first batch decides).
  for (auto const& s : sets) {
    if (s.vals.empty()) continue;
    for (int mode = 0; mode < 3; ++mode) {
      case_opt o;
      o.batch_rows = 2 * BR;
      o.batches    = 6;
      o.keys       = key_opt{kdist::uniform, mode == 2 ? 300000u : 5000u, 0.01};
      for (std::size_t c = 0; c < s.vals.size(); ++c) o.vals.push_back(val_opt{vdist::small, mode == 1 ? 0.0 : 0.1, mode >= 1});
      o.seed  = cases.size() + 1;
      auto tc = make_case("sql-nulls/" + s.name + "/" + (mode == 0 ? "scattered" : mode == 1 ? "groups" : "both-many"), I64,
                          s.vals, s.aggs, o);
      tc.threads = mode == 2 ? 4 : 1;
      add(std::move(tc));
    }
    // NULLs only after the first batch: the compact layout hands those rows back.
    case_opt o;
    o.batch_rows = BR;
    o.batches    = 6;
    o.keys       = key_opt{kdist::uniform, 5000};
    for (std::size_t c = 0; c < s.vals.size(); ++c) o.vals.push_back(val_opt{vdist::small, 0.1, true});
    o.seed  = cases.size() + 1;
    auto tc = make_case("sql-nulls/" + s.name + "/late", I64, s.vals, s.aggs, o);
    for (auto& v : tc.batches[0].vals) v.valid.clear();
    tc.expect_unapplied = true;
    add(std::move(tc));
  }
  // 4. NaN, infinities and signed zeros in float MIN / MAX (host reference: DuckDB's order).
  for (int threads : {1, 4}) {
    for (int set : {3, 4, 5}) {
      case_opt o;
      o.batch_rows = 2 * BR;
      o.batches    = 6;
      o.keys       = key_opt{kdist::uniform, set == 4 ? 300000u : 3000u};
      for (std::size_t c = 0; c < sets[set].vals.size(); ++c) {
        o.vals.push_back(val_opt{is_float(sets[set].vals[c].id()) && c == 2 ? vdist::nan_mix : vdist::small});
      }
      if (set == 5) o.vals[0] = val_opt{vdist::small};
      o.seed       = cases.size() + 1;
      auto tc      = make_case("sql-nan/" + sets[set].name + "/t" + std::to_string(threads), I64, sets[set].vals,
                               sets[set].aggs, o);
      tc.threads   = threads;
      tc.host_ref  = true;
      add(std::move(tc));
    }
  }
  {
    case_opt o;
    o.keys  = key_opt{kdist::uniform, 2000};
    o.vals  = {val_opt{vdist::nan_mix}, val_opt{vdist::nan_mix, 0.1}};
    auto tc = make_case("sql-nan/float32+float64-minmax", I64, {dt(tid::FLOAT32), F64},
                        {A(k::min, 0, dt(tid::FLOAT32)), A(k::max, 0, dt(tid::FLOAT32)), A(k::min, 1, F64),
                         A(k::max, 1, F64)},
                        o);
    tc.host_ref = true;
    add(std::move(tc));
  }
  // 5. Integer sums beyond int64 (host reference: 128-bit sums), with and without statistics.
  for (vdist vd : {vdist::big, vdist::big_late, vdist::cancel}) {
    for (int stats : {0, 1, 2}) {
      for (int threads : {1, 4}) {
        case_opt o;
        o.batch_rows = 2 * BR;
        o.batches    = 8;
        o.keys       = key_opt{kdist::uniform, threads == 1 ? 2000u : 200000u, 0.01, 0.01};
        o.vals       = {val_opt{vd}, val_opt{vd == vdist::big ? vdist::big : vdist::small}};
        o.stats      = stats;
        o.seed       = cases.size() + 1;
        std::string n = std::string("sql-sum/") + (vd == vdist::big ? "big" : vd == vdist::big_late ? "late" : "cancel") +
                        "/stats" + std::to_string(stats) + "/t" + std::to_string(threads);
        auto tc     = make_case(n + "/canon8", I64, sets[4].vals, sets[4].aggs, o);
        tc.threads  = threads;
        tc.host_ref = true;
        tc.expect_wide = true;
        add(std::move(tc));
        if (stats == 0) {
          auto t2     = make_case(n + "/two-sums", I64, sets[6].vals, sets[6].aggs, o);
          t2.threads  = threads;
          t2.host_ref = true;
          t2.expect_wide = true;
          add(std::move(t2));
        }
      }
    }
  }
  {
    // DECIMAL64 sums are always 128-bit; with huge values the high halves matter.
    case_opt o;
    o.keys  = key_opt{kdist::uniform, 3000};
    o.vals  = {val_opt{vdist::big, 0.05}};
    auto tc = make_case("sql-sum/decimal64-big", I64, {dt(tid::DECIMAL64, -4)},
                        {A(k::sum, 0, dt(tid::DECIMAL64, -4)), A(k::count, 0, dt(tid::DECIMAL64, -4))}, o);
    tc.host_ref = true;
    add(std::move(tc));
    // Statistics that prove the sums narrow: no 128-bit column.
    case_opt p;
    p.keys  = key_opt{kdist::uniform, 3000};
    p.stats = 1;
    add(make_case("sql-sum/proved-narrow", I64, {I64}, {CS(), A(k::sum, 0, I64)}, p));
  }
  // 6. Every supported key and value type.
  for (tid kt : {tid::INT8, tid::UINT8, tid::INT16, tid::UINT16, tid::INT32, tid::UINT32, tid::INT64, tid::UINT64,
                 tid::TIMESTAMP_DAYS, tid::TIMESTAMP_SECONDS, tid::TIMESTAMP_MILLISECONDS, tid::TIMESTAMP_MICROSECONDS,
                 tid::TIMESTAMP_NANOSECONDS, tid::DURATION_DAYS, tid::DURATION_SECONDS, tid::DECIMAL32,
                 tid::DECIMAL64}) {
    for (int nulls : {0, 1}) {
      case_opt o;
      o.keys = key_opt{kdist::uniform, 30000, nulls != 0 ? 0.03 : 0.0};
      o.seed = cases.size() + 1;
      add(make_case("types/key-" + std::to_string(static_cast<int>(kt)) + (nulls != 0 ? "-nulls" : ""),
                    dt(kt, kt == tid::DECIMAL32 || kt == tid::DECIMAL64 ? -2 : 0), {I64},
                    {CS(), A(k::sum, 0, I64), A(k::min, 0, I64)}, o));
    }
  }
  for (tid vt : {tid::INT8, tid::INT16, tid::INT32, tid::INT64, tid::DECIMAL32, tid::DECIMAL64, tid::FLOAT32,
                 tid::FLOAT64, tid::TIMESTAMP_DAYS, tid::TIMESTAMP_MICROSECONDS, tid::DURATION_SECONDS}) {
    bool summable = !(vt == tid::TIMESTAMP_DAYS || vt == tid::TIMESTAMP_MICROSECONDS || vt == tid::DURATION_SECONDS);
    cudf::data_type t = dt(vt, vt == tid::DECIMAL32 || vt == tid::DECIMAL64 ? -3 : 0);
    for (int nulls : {0, 1}) {
      for (uint64_t groups : {500ull, 200000ull}) {
        case_opt o;
        o.batch_rows = groups > 1000 ? 4 * BR : BR;
        o.keys       = key_opt{kdist::uniform, groups, nulls != 0 ? 0.02 : 0.0};
        o.vals       = {val_opt{vdist::small, nulls != 0 ? 0.15 : 0.0, nulls != 0}};
        o.seed       = cases.size() + 1;
        std::vector<at::agg_spec> aggs = {A(k::count, 0, t), A(k::min, 0, t), A(k::max, 0, t)};
        if (summable) aggs.push_back(A(k::sum, 0, t));
        add(make_case("types/value-" + std::to_string(static_cast<int>(vt)) + (nulls != 0 ? "-nulls" : "") + "/g" +
                        std::to_string(groups),
                      dt(tid::INT32), {t}, aggs, o));
        if (summable) {
          add(make_case("types/sum-" + std::to_string(static_cast<int>(vt)) + (nulls != 0 ? "-nulls" : "") + "/g" +
                          std::to_string(groups),
                        I64, {t}, {CS(), A(k::sum, 0, t)}, o));
        }
      }
    }
  }
  {
    case_opt o;
    o.keys  = key_opt{kdist::uniform, 20000};
    o.vals  = {val_opt{vdist::random_double}};
    auto tc = make_case("types/random-doubles", I64, {F64}, {A(k::sum, 0, F64), A(k::min, 0, F64), A(k::max, 0, F64)}, o);
    tc.sum_tol = 1e-9;
    add(std::move(tc));
  }
  // 7. Tiny budgets (rows handed back), mid-stream flushes, chunked flushes, hints.
  for (auto const& s : sets) {
    for (int threads : {1, 4}) {
      case_opt o;
      o.batch_rows = 4 * BR;
      o.batches    = 8;
      o.keys       = key_opt{kdist::uniform, 1500000, 0.01, 0.001};
      o.seed       = cases.size() + 1;
      auto tc      = make_case("budget/" + s.name + "/t" + std::to_string(threads), I64, s.vals, s.aggs, o);
      tc.threads   = threads;
      // (The smallest budget eligible() accepts for this set, with a little to spare.)
      tc.cfg.budget_bytes = (static_cast<std::size_t>(1) << 18) * 65 / 64 * 8 * 9 + (1 << 20);
      tc.expect_unapplied = true;
      tc.replay_done      = true;
      tc.fail_after       = 3;
      add(tc);
      tc.name += "/flush";
      tc.flush_by_next = threads == 4;
      tc.flush_every      = 1;
      tc.max_rows         = 50000;
      tc.expect_unapplied = false;
      add(std::move(tc));
    }
  }
  for (auto const& s : sets) {
    case_opt o;
    o.batch_rows = 4 * BR;
    o.batches    = 8;
    o.keys       = key_opt{kdist::zipf, 400000, 0.01, 0.01};
    o.seed       = cases.size() + 1;
    auto tc      = make_case("flush/" + s.name + "/every2/chunks", I64, s.vals, s.aggs, o);
    tc.threads     = 2;
    tc.flush_every = 2;
    tc.max_rows    = 10000;
    tc.flush_by_next = true;
    tc.fail_after    = 100;  // every fold fails once after it changed the table
    add(tc);
    tc.flush_by_next = false;
    tc.fail_after    = 0;
    tc.name            = "hint/" + s.name + "/exact";
    tc.flush_every     = 0;
    tc.cfg.groups_hint = 400000;
    tc.cfg.rows_hint   = static_cast<uint64_t>(o.batch_rows) * 8;
    add(tc);
    tc.name            = "hint/" + s.name + "/too-low";
    tc.cfg.groups_hint = 5000;
    add(tc);
    tc.name            = "hint/" + s.name + "/rows-only";
    tc.cfg.groups_hint.reset();
    add(std::move(tc));
  }
  // 8. Paths that must have been taken (the table's description).
  {
    case_opt o;
    o.batch_rows = 8 * BR;
    o.batches    = 6;
    o.keys       = key_opt{kdist::hot, 300000};
    auto tc      = make_case("path/hot-keys", I64, sets[3].vals, sets[3].aggs, o);
    tc.expect    = "hot_chunks=";
    add(tc);
    o.keys    = key_opt{kdist::uniform, 1000000};
    tc        = make_case("path/growth", I64, sets[1].vals, sets[1].aggs, o);
    tc.expect = "kernels=canonical";
    add(tc);
    o.keys    = key_opt{kdist::clustered, 3000, 0, 0, false, 100000};
    tc        = make_case("path/long-runs", I64, sets[4].vals, sets[4].aggs, o);
    tc.expect = "run_chunks=";
    add(tc);
    tc        = make_case("path/generic", I64, sets[6].vals, sets[6].aggs, o);
    tc.expect = "kernels=generic";
    add(tc);
  }
  // 12. Keys that repeat through the GENERIC kernel (aggregate sets without a specialised kernel,
  // the nullable layout): long runs and hot keys take seg_fold_gen_kernel (run_chunks= / hot_chunks=
  // in the description), with reserved and NULL keys inside the runs, NULL values, sums that leave
  // int64, a table at its budget, 1 and 4 fold threads.
  {
    const cudf::data_type DEC = dt(tid::DECIMAL64, -2);
    std::vector<aggset> gsets = {
      // Two float sums are not a canonical set: 5 state words, 8 in the nullable layout.
      {"generic5", {F64, F64, dt(tid::INT32)},
       {CS(), A(k::sum, 0, F64), A(k::sum, 1, F64), A(k::min, 2, dt(tid::INT32)), A(k::max, 2, dt(tid::INT32))}},
      sets[6],  // two-sums: 3 state words (4 in the nullable layout)
      {"generic8", {I64, I64, F64, DEC},
       {CS(), A(k::sum, 0, I64), A(k::min, 0, I64), A(k::max, 1, I64), A(k::sum, 2, F64), A(k::min, 2, F64),
        A(k::max, 2, F64), A(k::sum, 3, DEC)}},
    };
    struct rd {
      std::string name;
      key_opt o;
      std::string path;
    };
    std::vector<rd> rds = {
      {"runs", {kdist::clustered, 3000, 0, 0, false, 100000}, "run_chunks="},
      {"one-key", {kdist::uniform, 1}, "run_chunks="},
      {"hot", {kdist::hot, 300000}, "hot_chunks="},
      {"three-keys", {kdist::uniform, 3}, "hot_chunks="},
    };
    for (auto const& s : gsets) {
      for (auto const& d : rds) {
        for (int threads : {1, 4}) {
          std::string const tail = s.name + "/" + d.name + "/t" + std::to_string(threads);
          case_opt o;
          o.batch_rows = 4 * BR;
          o.batches    = 6;
          o.keys       = d.o;
          o.seed       = cases.size() + 1;
          auto tc      = make_case("repeat/plain/" + tail, I64, s.vals, s.aggs, o);
          tc.threads   = threads;
          tc.expect    = "kernels=generic";
          tc.expect2   = d.path;
          add(tc);
          if (d.name == "one-key" || d.name == "three-keys") continue;
          // Reserved and NULL keys: a few inside the long runs (more would end the runs),
          // scattered among the hot keys.
          o.keys.null_frac     = d.name == "runs" ? 0.0003 : 0.02;
          o.keys.reserved_frac = d.name == "runs" ? 0.0003 : 0.05;
          o.seed               = cases.size() + 1;
          tc                   = make_case("repeat/sql-keys/" + tail, I64, s.vals, s.aggs, o);
          tc.threads           = threads;
          tc.replay_done       = true;
          tc.fail_after        = threads == 4 ? 3 : 0;
          tc.expect            = "kernels=generic";
          tc.expect2           = d.path;
          add(tc);
          if (d.name == "runs") {  // runs of reserved keys between the runs of keys
            o.keys.reserved_frac = 0.25;
            o.keys.reserved_runs = true;
            o.seed               = cases.size() + 1;
            tc                   = make_case("repeat/sql-key-runs/" + tail, I64, s.vals, s.aggs, o);
            tc.threads           = threads;
            tc.expect            = "kernels=generic";
            add(tc);
          }
          // NULL values from the first batch: the nullable layout where its words fit. (generic8:
          // the compact layout hands the rows with a NULL back; they end the runs of the rows
          // that are folded, whose few keys are then hot keys.)
          o.keys = d.o;
          o.vals.clear();
          for (std::size_t c = 0; c < s.vals.size(); ++c) o.vals.push_back(val_opt{vdist::small, 0.1, true});
          o.seed     = cases.size() + 1;
          tc         = make_case("repeat/nulls/" + tail, I64, s.vals, s.aggs, o);
          tc.threads = threads;
          tc.expect  = s.name == "generic8" ? "layout=compact" : "layout=nullable";
          tc.expect2 = s.name == "generic8" ? "hot_chunks=" : d.path;
          add(tc);
          // Sums beyond int64 inside the runs / on the hot keys (host reference: 128-bit sums).
          if (s.name != "generic5") {
            o.keys = d.o;
            o.keys.null_frac     = d.name == "runs" ? 0.0003 : 0.01;
            o.keys.reserved_frac = d.name == "runs" ? 0.0003 : 0.01;
            o.vals = {val_opt{vdist::big}, val_opt{vdist::big}};
            o.seed = cases.size() + 1;
            tc     = make_case("repeat/big-sums/" + tail, I64, s.vals, s.aggs, o);
            tc.threads     = threads;
            tc.host_ref    = true;
            tc.expect_wide = true;
            tc.expect      = d.path;
            add(tc);
          }
          // A table at its budget: hot keys in front of more keys than it can hold (a tenth of
          // the rows have other keys).
          if (d.name == "hot") {
            o      = case_opt{};
            o.batch_rows = 4 * BR;
            o.batches    = 32;
            o.keys       = key_opt{kdist::hot, 4000000, 0.01, 0.001};
            o.seed       = cases.size() + 1;
            tc           = make_case("repeat/budget/" + tail, I64, s.vals, s.aggs, o);
            tc.threads   = threads;
            tc.cfg.budget_bytes = (static_cast<std::size_t>(1) << 18) * 65 / 64 * 8 * 9 + (1 << 20);
            tc.expect_unapplied = true;
            tc.replay_done      = true;
            tc.fail_after       = 3;
            tc.expect           = d.path;
            add(tc);
            tc.name += "/flush";
            tc.flush_every      = 1;
            tc.max_rows         = 50000;
            tc.expect_unapplied = false;
            add(std::move(tc));
          }
        }
      }
    }
  }

  int n = 0;
  for (auto const& tc : cases) {
    if (!want(tc.name)) continue;
    run_plain(tc);
    ++n;
  }
  // 9. fold_partial round trips.
  for (auto const& s : sets) {
    for (int mode = 0; mode < 3; ++mode) {
      case_opt o;
      o.batch_rows = 2 * BR;
      o.batches    = 6;
      o.keys       = key_opt{kdist::uniform, 100000, 0.01, 0.01};
      if (mode >= 1) {
        for (std::size_t c = 0; c < s.vals.size(); ++c) o.vals.push_back(val_opt{mode == 2 ? vdist::big : vdist::small, 0.1, true});
      }
      o.seed  = 1000 + static_cast<uint64_t>(mode);
      auto tc = make_case("partial/" + s.name + (mode == 0 ? "" : mode == 1 ? "/nulls" : "/nulls+big"), I64, s.vals, s.aggs, o);
      tc.host_ref = mode == 2;
      if (mode == 1 && s.vals.size() > 0) {
        // (Also the compact layout fed NULL partial states: the first batches have no NULLs.)
        for (int b = 0; b < 2; ++b) {
          for (auto& v : tc.batches[static_cast<std::size_t>(b)].vals) v.valid.clear();
        }
      }
      if (want(tc.name)) run_partial_roundtrip(tc);
    }
    case_opt o;
    o.batch_rows = 4 * BR;
    o.batches    = 6;
    o.keys       = key_opt{kdist::uniform, 1200000};
    auto tc      = make_case("partial/" + s.name + "/budget", I64, s.vals, s.aggs, o);
    tc.cfg.budget_bytes = (static_cast<std::size_t>(1) << 18) * 65 / 64 * 8 * 9 + (1 << 20);
    if (want(tc.name)) run_partial_roundtrip(tc);
  }
  // 10. Allocation failure at every allocation point, replayed with the same ticket.
  for (int set : {1, 4, 5, 6}) {
    for (int variant = 0; variant < 3; ++variant) {
      case_opt o;
      o.batch_rows = variant == 2 ? 4 * BR : BR;
      o.batches    = 4;
      o.keys       = key_opt{kdist::uniform, variant == 0 ? 3000u : 400000u, 0.02, 0.02};
      for (std::size_t c = 0; c < sets[set].vals.size(); ++c) {
        o.vals.push_back(val_opt{c == 0 && !is_float(sets[set].vals[c].id()) ? vdist::big : vdist::small,
                                 variant == 0 ? 0.0 : 0.1, false});
      }
      auto tc = make_case("fault/" + sets[set].name + (variant == 0 ? "/few" : variant == 1 ? "/many+nulls" : "/budget"),
                          I64, sets[set].vals, sets[set].aggs, o);
      // (With a budget the segments are separate allocations: a growth step can fail half way.)
      if (variant == 2) tc.cfg.budget_bytes = 96u << 20;
      tc.host_ref    = true;
      tc.flush_every = 2;
      tc.max_rows    = 20000;
      if (want(tc.name)) run_faults(tc, variant == 2 ? 120 : 400);
    }
  }
  // (... and with keys that repeat through the generic kernel - the nullable layout of generic3:
  // long runs, hot keys.)
  for (int variant = 0; variant < 2; ++variant) {
    case_opt o;
    o.batch_rows = 2 * BR;
    o.batches    = 4;
    o.keys       = variant == 0 ? key_opt{kdist::clustered, 3000, 0.0003, 0.0003, false, 100000} : key_opt{kdist::hot, 300000, 0.02, 0.02};
    o.vals       = {val_opt{vdist::small, 0.1, false}, val_opt{vdist::small}};
    auto tc      = make_case(std::string("fault/generic3/") + (variant == 0 ? "runs+nulls" : "hot+nulls"), I64, sets[5].vals,
                             sets[5].aggs, o);
    tc.host_ref    = true;
    tc.flush_every = 2;
    tc.max_rows    = 20000;
    if (want(tc.name)) run_faults(tc, 200);
  }
  // 11. Two tables alive at once.
  {
    case_opt o;
    o.batch_rows = 4 * BR;
    o.keys       = key_opt{kdist::uniform, 500000, 0.01, 0.01};
    auto a       = make_case("canon8", I64, sets[4].vals, sets[4].aggs, o);
    o.keys       = key_opt{kdist::hot, 300000, 0.02, 0};
    o.seed       = 77;
    auto b       = make_case("avg", dt(tid::INT32), sets[2].vals, sets[2].aggs, o);
    a.threads    = 2;
    b.threads    = 2;
    if (want("two-tables")) run_two(a, b);
  }
  if (want("eligible")) check_eligibility();
  std::printf("cases=%d fail=%d\n", g_run, g_fail);
  return g_fail == 0 ? 0 : 1;
}
