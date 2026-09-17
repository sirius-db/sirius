// SPDX-License-Identifier: Apache-2.0
#include "api/simpatico_codegen.hpp"
#include "codegen/plan/plan_interpreter.hpp"
#include "test_utils.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <array>
#include <cstdlib>
#include <numeric>

namespace {

namespace sc                        = sirius::codegen;
constexpr cudf::size_type row_count = 1037;  // partial mask word and partial chunk

void check(cudaError_t status)
{
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

class resource_guard {
 public:
  explicit resource_guard(rmm::mr::cuda_async_memory_resource const& resource)
    : previous_(rmm::mr::set_current_device_resource(
        cuda::mr::any_resource<cuda::mr::device_accessible>{resource}))
  {
  }
  ~resource_guard()
  {
    cudaDeviceSynchronize();
    rmm::mr::set_current_device_resource(std::move(previous_));
  }

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> previous_;
};

std::unique_ptr<cudf::column> sequence(int base,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  std::vector<std::int32_t> values(row_count);
  std::iota(values.begin(), values.end(), base);
  auto column = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, row_count, cudf::mask_state::UNALLOCATED, stream, mr);
  check(cudaMemcpyAsync(column->mutable_view().head<std::int32_t>(),
                        values.data(),
                        values.size() * sizeof(values[0]),
                        cudaMemcpyHostToDevice,
                        stream.value()));
  stream.synchronize();
  return column;
}

template <typename T>
std::vector<T> read(cudf::column_view column, rmm::cuda_stream_view stream)
{
  std::vector<T> values(column.size());
  // No producer wait/event here: completed public results must be readable on
  // this unrelated stream. Finish the verification copy before output teardown.
  check(cudaMemcpyAsync(values.data(),
                        column.head<T>(),
                        values.size() * sizeof(T),
                        cudaMemcpyDeviceToHost,
                        stream.value()));
  stream.synchronize();
  return values;
}

void verify_values(cudf::table_view output,
                   std::vector<std::int32_t> const& rows,
                   rmm::cuda_stream_view stream)
{
  expect(output.num_columns() == 2 && output.num_rows() == static_cast<int>(rows.size()),
         "numeric filtered output shape");
  for (int column = 0; column < 2; ++column) {
    auto actual = read<std::int32_t>(output.column(column), stream);
    for (std::size_t i = 0; i < rows.size(); ++i)
      expect(actual[i] == rows[i] + column * 10000, "numeric filtered output value/order");
  }
}

struct probe_failure : std::runtime_error {
  probe_failure() : std::runtime_error("injected membership failure") {}
};

void numeric_sources(simpatico::stream_pool& pool,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(sequence(0, stream, mr));
  columns.push_back(sequence(10000, stream, mr));
  cudf::table input{std::move(columns)};
  auto compressed = simpatico::compress_with_plan(
    input.view(), "input -> bitpack\n---\ninput -> identity\n", stream, mr);
  std::array<std::size_t, 2> selected{0, 1};
  std::vector<std::int32_t> first_twenty(20);
  std::iota(first_twenty.begin(), first_twenty.end(), 0);
  sc::scan_filter_request request;
  request.routes            = {sc::decode_route::bitpack_mask, sc::decode_route::full};
  request.source_generation = 42;
  request.filters.push_back({0, {0, 19}});
  sc::scan_filter_result result;
  auto output =
    simpatico::decompress_scan_filter(compressed, selected, request, result, pool, stream, mr);
  expect(result.applied && result.survivor_count == 20, "range ballot + full grouped gather");
  verify_values(output->view(), first_twenty, stream);
  {
    auto mask = result.view();
    simpatico::decode_selection selection;
    selection.mask             = &mask;
    selection.survivor_count   = mask.survivor_count;
    selection.survivor_indices = cudf::column_view{
      cudf::data_type{cudf::type_id::INT32}, 20, result.row_indices.data(), nullptr, 0};
    selection.route = sc::decode_route::full;
    std::string error;
    auto full = simpatico::decompress_column(
      *compressed.columns[0].plan_tree, stream, mr, &error, nullptr, &selection);
    expect(full != nullptr, error.c_str());
    expect(read<std::int32_t>(full->view(), stream) == first_twenty,
           "direct FULL remains legal on a compact-capable bitpack plan");
  }
  output.reset();

  request.filters.front().pred = {0, row_count - 1};
  output =
    simpatico::decompress_scan_filter(compressed, selected, request, result, pool, stream, mr);
  expect(!result.applied && result.status == sc::scan_filter_status::declined_unselective,
         "unselective range uses explicit completed policy decline");
  std::vector<std::int32_t> all(row_count);
  std::iota(all.begin(), all.end(), 0);
  verify_values(output->view(), all, stream);
  output.reset();

  auto limit = std::make_shared<cudf::numeric_scalar<std::int32_t>>(20, true, stream, mr);
  stream.synchronize();
  std::weak_ptr<cudf::numeric_scalar<std::int32_t>> snapshot = limit;
  request.filters.clear();
  request.routes     = {sc::decode_route::full, sc::decode_route::full};
  std::size_t probes = 0;
  request.membership_filters.push_back(
    {0,
     [limit, &probes, &pool](cudf::column_view keys,
                             rmm::cuda_stream_view lane,
                             rmm::device_async_resource_ref resource) {
       expect(
         std::find(pool.streams.begin(), pool.streams.end(), lane.value()) != pool.streams.end(),
         "probe uses a supplied lane");
       ++probes;
       return cudf::binary_operation(keys,
                                     *limit,
                                     cudf::binary_operator::LESS,
                                     cudf::data_type{cudf::type_id::BOOL8},
                                     lane,
                                     resource);
     }});
  limit.reset();
  expect(!snapshot.expired(), "membership closure pins immutable snapshot");
  request.membership_filters.push_back(
    {1, [](cudf::column_view, rmm::cuda_stream_view, rmm::device_async_resource_ref) {
       return std::unique_ptr<cudf::column>{};
     }});
  std::vector<std::uint32_t> keep((row_count + 31) / 32, 0);
  for (int i = 0; i < row_count; i += 2)
    keep[i / 32] |= std::uint32_t{1} << (i % 32);
  request.keep_mask_words = keep.data();
  request.keep_mask_rows  = row_count;
  output =
    simpatico::decompress_scan_filter(compressed, selected, request, result, pool, stream, mr);
  expect(result.applied && result.survivor_count == 10 && result.keep_mask_applied,
         "accepted + declined membership sources and keep-mask tail");
  expect(probes == 1 && result.source_generation == 42, "membership call/generation preserved");
  std::vector<std::int32_t> even;
  for (int i = 0; i < 20; i += 2)
    even.push_back(i);
  verify_values(output->view(), even, stream);
  output.reset();

  request.membership_filters.erase(request.membership_filters.begin());
  expect(snapshot.expired(), "completed session does not retain probe capture");
  output =
    simpatico::decompress_scan_filter(compressed, selected, request, result, pool, stream, mr);
  expect(!result.applied && result.status == sc::scan_filter_status::refused,
         "all-declined source is explicit policy decline before padded mask count");
  expect(result.source_generation == 42, "decline preserves source generation");
  verify_values(output->view(), all, stream);
  output.reset();
  request.keep_mask_words = nullptr;
  request.keep_mask_rows  = 0;

  request.membership_filters.front().probe =
    [](cudf::column_view,
       rmm::cuda_stream_view,
       rmm::device_async_resource_ref) -> std::unique_ptr<cudf::column> { throw probe_failure{}; };
  bool propagated = false;
  try {
    (void)simpatico::decompress_scan_filter(
      compressed, selected, request, result, pool, stream, mr);
  } catch (probe_failure const&) {
    propagated = true;
  }
  expect(propagated && result.status == sc::scan_filter_status::failed,
         "probe execution failure is not converted to plain decode");

  request.membership_filters.front().probe = [](cudf::column_view keys,
                                                rmm::cuda_stream_view lane,
                                                rmm::device_async_resource_ref resource) {
    return cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                     keys.size(),
                                     cudf::mask_state::UNALLOCATED,
                                     lane,
                                     resource);
  };
  bool malformed = false;
  try {
    (void)simpatico::decompress_scan_filter(
      compressed, selected, request, result, pool, stream, mr);
  } catch (std::invalid_argument const&) {
    malformed = true;
  }
  expect(malformed, "non-BOOL8 membership result is an error, not semantic decline");
  output = simpatico::decompress(compressed, selected, pool, mr);
  verify_values(output->view(), all, stream);
}

void bool8_delivery(simpatico::stream_pool& pool,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr)
{
  std::vector<std::string> strings(row_count, "other");
  std::vector<std::int32_t> matched;
  for (int i = 0; i < row_count; i += 97) {
    strings[i] = "match";
    matched.push_back(i);
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(make_strings_column(strings, {}, stream));
  columns.push_back(sequence(0, stream, mr));
  cudf::table input{std::move(columns)};
  auto compressed =
    simpatico::compress_with_plan(input.view(),
                                  "input -> dictionary -> keys_offsets, keys_chars, indices\n"
                                  "dictionary.indices -> bitpack\n---\ninput -> identity\n",
                                  stream,
                                  mr);

  for (std::vector<std::size_t> selected : {std::vector<std::size_t>{0}, {0, 1}}) {
    sc::scan_filter_request request;
    request.routes = {sc::decode_route::dict_codes};
    if (selected.size() == 2) request.routes.push_back(sc::decode_route::full);
    request.bool8_filters.push_back({0, {"match"}});
    sc::scan_filter_result result;
    auto output =
      simpatico::decompress_scan_filter(compressed, selected, request, result, pool, stream, mr);
    expect(result.applied && output->num_rows() == static_cast<int>(matched.size()),
           "BOOL8 dual delivery survivor count");
    expect(output->view().column(0).type().id() == cudf::type_id::BOOL8,
           "predicate result keeps BOOL8, not stored STRING type");
    auto flags = read<std::uint8_t>(output->view().column(0), stream);
    expect(std::all_of(flags.begin(), flags.end(), [](auto value) { return value != 0; }),
           "BOOL8 dual delivery values");
    {
      auto mask = result.view();
      simpatico::decode_selection selection;
      selection.mask             = &mask;
      selection.survivor_count   = mask.survivor_count;
      selection.survivor_indices = cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                                     static_cast<cudf::size_type>(matched.size()),
                                                     result.row_indices.data(),
                                                     nullptr,
                                                     0};
      selection.route            = sc::decode_route::full;
      simpatico::decode_predicate predicate;
      predicate.equals_any = {"match"};
      std::string error;
      auto full = simpatico::decompress_column(
        *compressed.columns[0].plan_tree, stream, mr, &error, &predicate, &selection);
      expect(full != nullptr, error.c_str());
      expect(full->type().id() == cudf::type_id::BOOL8 && full->size() == output->num_rows(),
             "direct dictionary predicate + FULL keeps BOOL8 and survivor shape");
      expect(read<std::uint8_t>(full->view(), stream) == flags,
             "direct dictionary predicate + FULL gathers once");
    }
    if (selected.size() == 2)
      expect(read<std::int32_t>(output->view().column(1), stream) == matched,
             "BOOL8 and full values share one correctly ordered gather");
  }

  std::array<std::size_t, 2> selected{0, 1};
  sc::scan_filter_request request;
  request.routes = {sc::decode_route::dict_codes, sc::decode_route::full};
  request.bool8_filters.push_back({0, {"other"}});
  sc::scan_filter_result result;
  auto output =
    simpatico::decompress_scan_filter(compressed, selected, request, result, pool, stream, mr);
  expect(!result.applied && result.status == sc::scan_filter_status::declined_unselective,
         "unselective BOOL8/full request declines selection");
  expect(
    output->num_rows() == row_count && output->view().column(0).type().id() == cudf::type_id::BOOL8,
    "policy decline preserves full-width BOOL8 substitution");
  auto flags = read<std::uint8_t>(output->view().column(0), stream);
  for (int i = 0; i < row_count; ++i)
    expect((flags[i] != 0) == (i % 97 != 0), "unselected BOOL8 predicate value");
}

}  // namespace

int main()
{
  try {
    // Policy values cache on first read; use a dedicated process, not mutations
    // of environment knobs between cases in a shared engine test binary.
    setenv("SIRIUS_EXP_FUSED_SCAN_FILTER", "1", 1);
    setenv("SIRIUS_EXP_FUSED_SCAN_MAX_MEMBER", "4", 1);
    setenv("SIRIUS_EXP_FUSED_SCAN_MAX_SEL", "0.35", 1);
    setenv("SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL", "0.10", 1);
    setenv("SIRIUS_EXP_FUSED_SCAN_K4_MAX_SEL", "0.15", 1);
    // cuDF intentionally retains its default pinned pool until process exit.
    // Use individually freed pinned allocations so this standalone fixture can
    // check complete teardown without suppressing process-global pool leaks.
    setenv("LIBCUDF_PINNED_POOL_SIZE", "0", 1);
    setenv("LIBCUDF_PINNED_POOL_MAX_SIZE", "0", 1);
    expect(cudf::config_default_pinned_memory_resource({.pool_size = 0}),
           "pinned resource was initialized before test configuration");
    rmm::mr::cuda_async_memory_resource resource{64U << 20};
    resource_guard current{resource};
    rmm::cuda_stream stream;
    simpatico::stream_pool pool;
    expect(pool.init(4), "stream pool initialization");
    numeric_sources(pool, stream.view(), resource);
    bool8_delivery(pool, stream.view(), resource);
    check(pool.sync_all());
    stream.synchronize();
    std::puts("test_scan_filter_session: OK");
    return 0;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "test_scan_filter_session: FAIL: %s\n", error.what());
    return 1;
  }
}
