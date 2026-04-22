#include "io/uring/uring_ioctx.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>

#include <spdlog/spdlog.h>

#include <cuda_runtime.h>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static const std::string PATH =
    "/home/aaramoon/Documents/tpch/sf100/parquet/lineitem_indexed.parquet";

static const std::vector<std::string> COLUMNS = {
    "l_orderkey",
    "l_extendedprice",
    "l_discount",
    "l_shipdate",
};

static cudf::io::table_with_metadata read_cudf(size_t num_rows) {
    auto src  = cudf::io::source_info{{PATH}};
    auto b    = cudf::io::parquet_reader_options::builder(src).columns(COLUMNS);
    if (num_rows > 0) b.num_rows(num_rows);
    return cudf::io::read_parquet(b.build());
}

static cudf::io::table_with_metadata read_device(size_t num_rows) {
    auto io_ctx = std::make_shared<sirius::io::uring_ioctx>();
    auto ds     = io_ctx->make_datasource(
        std::make_unique<sirius::io::uring_io_object>(PATH));
    std::vector<cudf::io::datasource *> ptrs{ds.get()};
    auto src = cudf::io::source_info{ptrs};
    auto b   = cudf::io::parquet_reader_options::builder(src).columns(COLUMNS);
    if (num_rows > 0) b.num_rows(num_rows);
    return cudf::io::read_parquet(b.build());
}

// Copy a device buffer to a host vector.
static std::vector<uint8_t> to_host(void const *dev_ptr, size_t bytes) {
    std::vector<uint8_t> h(bytes);
    if (bytes > 0)
        cudaMemcpy(h.data(), dev_ptr, bytes, cudaMemcpyDeviceToHost);
    return h;
}

// Compare one pair of columns. Returns true if identical.
static bool compare_column(cudf::column_view const &a, cudf::column_view const &b,
                            std::string const &name) {
    if (a.size() != b.size()) {
        std::cerr << "  FAIL " << name << ": row count differs ("
                  << a.size() << " vs " << b.size() << ")\n";
        return false;
    }
    if (a.null_count() != b.null_count()) {
        std::cerr << "  FAIL " << name << ": null_count differs ("
                  << a.null_count() << " vs " << b.null_count() << ")\n";
        return false;
    }

    // Element count and byte width of the data buffer.
    size_t n_elems  = static_cast<size_t>(a.size());
    size_t type_sz  = cudf::size_of(a.type());
    size_t data_bytes = n_elems * type_sz;

    auto ha = to_host(a.head<uint8_t>(), data_bytes);
    auto hb = to_host(b.head<uint8_t>(), data_bytes);

    if (ha != hb) {
        // Find first differing byte for diagnostics.
        for (size_t i = 0; i < data_bytes; ++i) {
            if (ha[i] != hb[i]) {
                std::cerr << "  FAIL " << name << ": first diff at byte " << i
                          << " (element " << i / type_sz << "): "
                          << (int)ha[i] << " vs " << (int)hb[i] << "\n";
                return false;
            }
        }
    }

    // Compare validity masks if present.
    if (a.nullable()) {
        size_t mask_bytes = (n_elems + 7) / 8;
        auto ma = to_host(a.null_mask(), mask_bytes);
        auto mb = to_host(b.null_mask(), mask_bytes);
        if (ma != mb) {
            std::cerr << "  FAIL " << name << ": null masks differ\n";
            return false;
        }
    }

    return true;
}

int main(int argc, char **argv) {
    size_t num_rows = (argc > 1) ? (size_t)std::stoll(argv[1]) : 10'000'000;

    spdlog::set_level(spdlog::level::warn); // suppress trace noise

    cudaFree(nullptr);
    rmm::mr::cuda_async_memory_resource async_mr;
    rmm::mr::set_current_device_resource(&async_mr);

    std::cout << "Reading " << (num_rows == 0 ? "all" : std::to_string(num_rows))
              << " rows via cudf default...\n";
    auto ref = read_cudf(num_rows);

    std::cout << "Reading " << (num_rows == 0 ? "all" : std::to_string(num_rows))
              << " rows via device path...\n";
    auto dev = read_device(num_rows);

    size_t n_ref = static_cast<size_t>(ref.tbl->num_rows());
    size_t n_dev = static_cast<size_t>(dev.tbl->num_rows());
    std::cout << "cudf rows=" << n_ref << "  device rows=" << n_dev << "\n\n";

    if (n_ref != n_dev) {
        std::cerr << "FAIL: row counts differ\n";
        return 1;
    }

    bool all_ok = true;
    for (int c = 0; c < ref.tbl->num_columns(); ++c) {
        std::string col_name = (c < (int)COLUMNS.size()) ? COLUMNS[c]
                                                          : "col" + std::to_string(c);
        bool ok = compare_column(ref.tbl->view().column(c),
                                 dev.tbl->view().column(c), col_name);
        std::cout << "  " << (ok ? "PASS" : "FAIL") << "  " << col_name << "\n";
        all_ok &= ok;
    }

    std::cout << "\n" << (all_ok ? "ALL PASS" : "SOME COLUMNS FAILED") << "\n";
    return all_ok ? 0 : 1;
}
