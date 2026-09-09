// Times a GPU sort of a lineitem-shaped table: the pin-time clustering step.
//
// Build (E = the repo's pixi env, e.g. <repo>/.pixi/envs/default):
//   $E/bin/nvcc -ccbin $E/bin/aarch64-conda-linux-gnu-g++ -std=c++20 -O2 -arch=sm_100 \
//     --expt-extended-lambda --expt-relaxed-constexpr -diag-suppress 20012 \
//     -I$E/include -I$E/include/rapids sortbench.cu -o sortbench \
//     -L$E/lib -lcudf -lrmm -lcudart -Xlinker -rpath -Xlinker $E/lib
//
// Run:  ./sortbench <rows> [1]      # 1 = widen to all 16 lineitem columns (88 B/row)
//       Always under /home/nvidia/joost/bench-lock.sh on the shared box.
#include <cudf/column/column_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/copying.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <cstdio>
#include <chrono>
#include <vector>
#include <random>

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
  int64_t n = (argc > 1) ? std::atoll(argv[1]) : 100'000'000;
  rmm::mr::pool_memory_resource pool{rmm::mr::cuda_memory_resource{}, 160ull << 30};
  rmm::mr::set_current_device_resource_ref(pool);
  auto s = cudf::get_default_stream();

  // lineitem's 7 hottest columns: shipdate(4) key + orderkey(8) partkey(4) suppkey(4)
  // quantity(8) extendedprice(8) discount(8) = 44 B/row
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(rand_col<int32_t>(n, cudf::data_type{cudf::type_id::INT32}, 8000, 10500, s));  // key
  cols.push_back(rand_col<int64_t>(n, cudf::data_type{cudf::type_id::INT64}, 0, 6'000'000, s));
  cols.push_back(rand_col<int32_t>(n, cudf::data_type{cudf::type_id::INT32}, 0, 20'000'000, s));
  cols.push_back(rand_col<int32_t>(n, cudf::data_type{cudf::type_id::INT32}, 0, 1'000'000, s));
  cols.push_back(rand_col<int64_t>(n, cudf::data_type{cudf::type_id::INT64}, 100, 5000, s));
  cols.push_back(rand_col<int64_t>(n, cudf::data_type{cudf::type_id::INT64}, 90000, 10000000, s));
  cols.push_back(rand_col<int64_t>(n, cudf::data_type{cudf::type_id::INT64}, 0, 10, s));
  // optional: widen to all 16 lineitem columns (~101 B/row with dict-coded strings)
  int wide = (argc > 2) ? std::atoi(argv[2]) : 0;
  if (wide) {
    cols.push_back(rand_col<int64_t>(n, cudf::data_type{cudf::type_id::INT64}, 0, 10, s));   // tax
    cols.push_back(rand_col<int32_t>(n, cudf::data_type{cudf::type_id::INT32}, 1, 7, s));    // linenumber
    cols.push_back(rand_col<int32_t>(n, cudf::data_type{cudf::type_id::INT32}, 8000, 10500, s)); // commitdate
    cols.push_back(rand_col<int32_t>(n, cudf::data_type{cudf::type_id::INT32}, 8000, 10500, s)); // receiptdate
    cols.push_back(rand_col<int32_t>(n, cudf::data_type{cudf::type_id::INT32}, 0, 3, s));    // returnflag
    cols.push_back(rand_col<int32_t>(n, cudf::data_type{cudf::type_id::INT32}, 0, 2, s));    // linestatus
    cols.push_back(rand_col<int32_t>(n, cudf::data_type{cudf::type_id::INT32}, 0, 4, s));    // shipinstruct
    cols.push_back(rand_col<int32_t>(n, cudf::data_type{cudf::type_id::INT32}, 0, 7, s));    // shipmode
    cols.push_back(rand_col<int64_t>(n, cudf::data_type{cudf::type_id::INT64}, 0, 1u<<30, s)); // comment proxy
  }
  auto tbl = std::make_unique<cudf::table>(std::move(cols));
  auto tv  = tbl->view();
  double bpr = 0; for (auto const& c : tbl->view()) bpr += cudf::size_of(c.type());
  double gb = double(n) * bpr / 1e9;

  cudf::table_view keys{{tv.column(0)}};
  for (int it = 0; it < 3; ++it) {
    s.synchronize();
    auto t0  = clk::now();
    auto ord = cudf::sorted_order(keys, {}, {}, s);
    s.synchronize();
    auto t1  = clk::now();
    auto out = cudf::gather(tv, ord->view(), cudf::out_of_bounds_policy::DONT_CHECK, s);
    s.synchronize();
    auto t2 = clk::now();
    double so = std::chrono::duration<double>(t1 - t0).count();
    double ga = std::chrono::duration<double>(t2 - t1).count();
    std::printf("n=%lld (%.0f B/row, %.2f GB payload)  sorted_order %.3fs  gather %.3fs  total %.3fs"
                "  -> %.1f Mrows/s, %.1f GB/s\n",
                (long long)n, bpr, gb, so, ga, so + ga, n / (so + ga) / 1e6, gb / (so + ga));
  }
  return 0;
}
