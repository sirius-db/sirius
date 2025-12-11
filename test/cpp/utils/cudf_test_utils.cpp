/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "catch.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/cuda_stream.hpp>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>

namespace sirius {
namespace test {

// Forward declarations for helpers defined later in this file
static void install_rmm_logging_resource_once();
static inline bool host_mem_equal(const uint8_t* a, const uint8_t* b, size_t n);

static void dump_hex_context(const uint8_t* data,
                             size_t size,
                             size_t center,
                             size_t context_len = 64)
{
  size_t start = (center > context_len / 2) ? (center - context_len / 2) : 0;
  if (start + context_len > size) { context_len = (size > start) ? (size - start) : 0; }
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (size_t i = 0; i < context_len; ++i) {
    size_t idx = start + i;
    if (i && (i % 16 == 0)) oss << " | ";
    if (idx < size) { oss << std::setw(2) << static_cast<unsigned int>(data[idx]) << ' '; }
  }
  std::cout << "  hex@" << std::dec << start << " (" << context_len << "B): " << oss.str()
            << std::endl
            << std::flush;
}

// Stream-aware comparison that compares actual column data (not packed representation padding)
bool cudf_tables_have_equal_contents_on_stream(const cudf::table& left,
                                               const cudf::table& right,
                                               rmm::cuda_stream_view stream_view)
{
  if (left.num_rows() != right.num_rows()) {
    std::cout << "[cudf-equal] row count mismatch: left=" << left.num_rows()
              << " right=" << right.num_rows() << std::endl
              << std::flush;
    return false;
  }
  if (left.num_columns() != right.num_columns()) {
    std::cout << "[cudf-equal] column count mismatch: left=" << left.num_columns()
              << " right=" << right.num_columns() << std::endl
              << std::flush;
    return false;
  }

  // Sync stream before reading column data
  stream_view.synchronize();

  // Compare each column's actual data (not packed representation which includes padding)
  for (int col_idx = 0; col_idx < left.num_columns(); ++col_idx) {
    auto left_col  = left.view().column(col_idx);
    auto right_col = right.view().column(col_idx);

    // Check type matches
    if (left_col.type().id() != right_col.type().id()) {
      std::cout << "[cudf-equal] col[" << col_idx
                << "] type mismatch: left=" << static_cast<int>(left_col.type().id())
                << " right=" << static_cast<int>(right_col.type().id()) << std::endl
                << std::flush;
      return false;
    }

    // Check size matches
    if (left_col.size() != right_col.size()) {
      std::cout << "[cudf-equal] col[" << col_idx << "] size mismatch: left=" << left_col.size()
                << " right=" << right_col.size() << std::endl
                << std::flush;
      return false;
    }

    // Compare actual column data bytes
    size_t data_bytes = left_col.size() * cudf::size_of(left_col.type());
    if (data_bytes == 0) continue;

    std::vector<uint8_t> left_data(data_bytes);
    std::vector<uint8_t> right_data(data_bytes);

    cudaMemcpy(left_data.data(), left_col.head(), data_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(right_data.data(), right_col.head(), data_bytes, cudaMemcpyDeviceToHost);

    if (!host_mem_equal(left_data.data(), right_data.data(), data_bytes)) {
      // Find first differing byte
      size_t diff_idx = 0;
      for (; diff_idx < data_bytes; ++diff_idx) {
        if (left_data[diff_idx] != right_data[diff_idx]) break;
      }
      std::cout << "[cudf-equal] col[" << col_idx << "] data differs at byte " << diff_idx
                << " left=" << static_cast<unsigned int>(left_data[diff_idx])
                << " right=" << static_cast<unsigned int>(right_data[diff_idx]) << std::endl
                << std::flush;

      // Show context around the difference
      std::cout << "[cudf-equal] col[" << col_idx << "] left data context:" << std::endl
                << std::flush;
      dump_hex_context(left_data.data(), data_bytes, diff_idx);
      std::cout << "[cudf-equal] col[" << col_idx << "] right data context:" << std::endl
                << std::flush;
      dump_hex_context(right_data.data(), data_bytes, diff_idx);
      return false;
    }
  }

  return true;
}

void expect_cudf_tables_equal_on_stream(const cudf::table& left,
                                        const cudf::table& right,
                                        rmm::cuda_stream_view stream_view)
{
  REQUIRE(cudf_tables_have_equal_contents_on_stream(left, right, stream_view));
}

// Simple logging adaptor to print all RMM device allocations/frees with pointers/sizes/stream/tid
class logging_device_resource : public rmm::mr::device_memory_resource {
 public:
  explicit logging_device_resource(rmm::mr::device_memory_resource* upstream) : upstream_(upstream)
  {
  }

  ~logging_device_resource() override = default;

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override
  {
    void* ptr = upstream_->allocate(bytes, stream);
    std::ostringstream oss;
    oss << "[rmm-alloc] ptr=" << ptr << " size=" << bytes << " stream=" << stream.value()
        << " tid=" << std::this_thread::get_id();
    std::cout << oss.str() << std::endl << std::flush;
    return ptr;
  }

  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) noexcept override
  {
    std::ostringstream oss;
    oss << "[rmm-free ] ptr=" << ptr << " size=" << bytes << " stream=" << stream.value()
        << " tid=" << std::this_thread::get_id();
    std::cout << oss.str() << std::endl << std::flush;
    upstream_->deallocate(ptr, bytes, stream);
  }

  bool do_is_equal(const rmm::mr::device_memory_resource& other) const noexcept override
  {
    return this == &other;
  }

  rmm::mr::device_memory_resource* upstream_;
};

// Install the logging resource once per process (wraps whatever the current device resource is)
static void install_rmm_logging_resource_once()
{
  static bool installed = false;
  static std::unique_ptr<logging_device_resource> logging_resource;
  if (!installed) {
    auto* prev       = rmm::mr::get_current_device_resource();
    logging_resource = std::make_unique<logging_device_resource>(prev);
    rmm::mr::set_current_device_resource(logging_resource.get());
    installed = true;
    std::cout << "[rmm-log ] installed logging device resource adaptor" << std::endl << std::flush;
  }
}

static inline bool host_mem_equal(const uint8_t* a, const uint8_t* b, size_t n)
{
  if (a == b) return true;
  if ((a == nullptr) || (b == nullptr)) return false;
  return std::memcmp(a, b, n) == 0;
}

// Removed non-stream variants to enforce explicit stream usage in tests

}  // namespace test
}  // namespace sirius
