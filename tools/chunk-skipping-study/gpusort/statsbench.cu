// Phase 2 feasibility: what does computing per-GROUP min/max cost at pin time?
//
// W2 today runs one cudf::minmax per (pin chunk, column). A group index needs
// n_groups = chunk_rows / (G*1024) min/max pairs per column instead. This times
// cudf::segmented_reduce over a fixed-stride offsets column and compares against
// the whole-chunk cudf::minmax that W2 already pays for.
//
// Build: see gpusort/sortbench.cu header, swap the source name.
// Run:   ./statsbench <rows> [G]      # G = simpatico chunks per group (default 8)
#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/reduction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using clk = std::chrono::high_resolution_clock;

template <typename T>
std::unique_ptr<cudf::column> rand_col(int64_t n, cudf::data_type dt, uint32_t lo, uint32_t hi,
                                       rmm::cuda_stream_view s)
{
  std::vector<T> h(n);
  std::mt19937 g(42);
  std::uniform_int_distribution<uint32_t> d(lo, hi);
  for (int64_t i = 0; i < n; ++i) h[i] = static_cast<T>(d(g));
  auto col = cudf::make_fixed_width_column(dt, n, cudf::mask_state::UNALLOCATED, s);
  cudaMemcpyAsync(col->mutable_view().template data<T>(), h.data(), n * sizeof(T),
                  cudaMemcpyHostToDevice, s.value());
  s.synchronize();
  return col;
}

int main(int argc, char** argv)
{
  int64_t n      = (argc > 1) ? std::atoll(argv[1]) : 189'000'000;
  int const G    = (argc > 2) ? std::atoi(argv[2]) : 8;
  int const kChunk = 1024;
  int const stride = G * kChunk;

  rmm::mr::pool_memory_resource pool{rmm::mr::cuda_memory_resource{}, 120ull << 30};
  rmm::mr::set_current_device_resource_ref(pool);
  auto s = cudf::get_default_stream();

  // All 16 lineitem columns (strings dict-coded to int32), ~88 B/row.
  std::vector<std::unique_ptr<cudf::column>> cols;
  auto I32 = cudf::data_type{cudf::type_id::INT32};
  auto I64 = cudf::data_type{cudf::type_id::INT64};
  cols.push_back(rand_col<int64_t>(n, I64, 0, 6'000'000, s));   // orderkey
  cols.push_back(rand_col<int32_t>(n, I32, 0, 20'000'000, s));  // partkey
  cols.push_back(rand_col<int32_t>(n, I32, 0, 1'000'000, s));   // suppkey
  cols.push_back(rand_col<int32_t>(n, I32, 1, 7, s));           // linenumber
  cols.push_back(rand_col<int64_t>(n, I64, 100, 5000, s));      // quantity
  cols.push_back(rand_col<int64_t>(n, I64, 90000, 1e7, s));     // extendedprice
  cols.push_back(rand_col<int64_t>(n, I64, 0, 10, s));          // discount
  cols.push_back(rand_col<int64_t>(n, I64, 0, 8, s));           // tax
  cols.push_back(rand_col<int32_t>(n, I32, 0, 3, s));           // returnflag
  cols.push_back(rand_col<int32_t>(n, I32, 0, 2, s));           // linestatus
  cols.push_back(rand_col<int32_t>(n, I32, 8000, 10500, s));    // shipdate
  cols.push_back(rand_col<int32_t>(n, I32, 8000, 10500, s));    // commitdate
  cols.push_back(rand_col<int32_t>(n, I32, 8000, 10500, s));    // receiptdate
  cols.push_back(rand_col<int32_t>(n, I32, 0, 4, s));           // shipinstruct
  cols.push_back(rand_col<int32_t>(n, I32, 0, 7, s));           // shipmode
  cols.push_back(rand_col<int64_t>(n, I64, 0, 1u << 30, s));    // comment proxy
  auto tbl = std::make_unique<cudf::table>(std::move(cols));
  auto tv  = tbl->view();

  int64_t const n_groups = (n + stride - 1) / stride;
  std::vector<cudf::size_type> h_off(n_groups + 1);
  for (int64_t i = 0; i <= n_groups; ++i)
    h_off[i] = static_cast<cudf::size_type>(std::min<int64_t>(i * stride, n));
  rmm::device_uvector<cudf::size_type> d_off(h_off.size(), s);
  cudaMemcpyAsync(d_off.data(), h_off.data(), h_off.size() * sizeof(cudf::size_type),
                  cudaMemcpyHostToDevice, s.value());
  s.synchronize();

  auto min_agg = cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>();
  auto max_agg = cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>();
  auto span    = cudf::device_span<cudf::size_type const>{d_off.data(), d_off.size()};

  double bpr = 0;
  for (auto const& c : tv) bpr += cudf::size_of(c.type());
  double const gb = double(n) * bpr / 1e9;

  for (int it = 0; it < 3; ++it) {
    // (a) what W2 pays today: one whole-chunk minmax per column
    s.synchronize();
    auto t0 = clk::now();
    for (auto const& c : tv) { auto r = cudf::minmax(c, s); }
    s.synchronize();
    auto t1 = clk::now();
    // (b) what a G-group index costs: segmented min + max per column
    for (auto const& c : tv) {
      auto lo = cudf::segmented_reduce(c, span, *min_agg, c.type(), cudf::null_policy::EXCLUDE, s);
      auto hi = cudf::segmented_reduce(c, span, *max_agg, c.type(), cudf::null_policy::EXCLUDE, s);
    }
    s.synchronize();
    auto t2 = clk::now();
    double whole = std::chrono::duration<double>(t1 - t0).count();
    double seg   = std::chrono::duration<double>(t2 - t1).count();
    std::printf("n=%lld G=%d stride=%d groups=%lld cols=%d (%.2f GB)  whole-chunk minmax %.4fs"
                "  segmented %.4fs  ratio %.1fx  -> %.1f GB/s\n",
                (long long)n, G, stride, (long long)n_groups, tv.num_columns(), gb, whole, seg,
                seg / whole, gb / seg);
  }
  return 0;
}
