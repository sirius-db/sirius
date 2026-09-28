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

#include <algorithm>
#include <array>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

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
                                       ::cuda::stream_ref stream,
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
                        stream.get()));
  stream.sync();
  return column;
}

template <typename T>
std::vector<T> read(cudf::column_view column, ::cuda::stream_ref stream)
{
  std::vector<T> values(column.size());
  // No producer wait/event here: completed public results must be readable on
  // this unrelated stream. Finish the verification copy before output teardown.
  check(cudaMemcpyAsync(values.data(),
                        column.head<T>(),
                        values.size() * sizeof(T),
                        cudaMemcpyDeviceToHost,
                        stream.get()));
  stream.sync();
  return values;
}

void verify_values(cudf::table_view output,
                   std::vector<std::int32_t> const& rows,
                   ::cuda::stream_ref stream)
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
                     ::cuda::stream_ref stream,
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
  stream.sync();
  std::weak_ptr<cudf::numeric_scalar<std::int32_t>> snapshot = limit;
  request.filters.clear();
  request.routes     = {sc::decode_route::full, sc::decode_route::full};
  std::size_t probes = 0;
  request.membership_filters.push_back(
    {0,
     [limit, &probes, &pool](
       cudf::column_view keys, ::cuda::stream_ref lane, rmm::device_async_resource_ref resource) {
       expect(std::find(pool.streams.begin(), pool.streams.end(), lane.get()) != pool.streams.end(),
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
    {1, [](cudf::column_view, ::cuda::stream_ref, rmm::device_async_resource_ref) {
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
       ::cuda::stream_ref,
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

  request.membership_filters.front().probe =
    [](cudf::column_view keys, ::cuda::stream_ref lane, rmm::device_async_resource_ref resource) {
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
                    ::cuda::stream_ref stream,
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

// Borrowed lanes may repeat and may include the output stream, as lanes taken from a shared pool
// do. The phase joins, the index waits, and the cleanup waits must stay correct; results must match
// the stream_pool overloads; and the call must leave every lane usable.
void aliased_borrowed_lanes(simpatico::stream_pool& pool,
                            ::cuda::stream_ref stream,
                            rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(sequence(0, stream, mr));
  columns.push_back(sequence(10000, stream, mr));
  cudf::table input{std::move(columns)};
  auto compressed = simpatico::compress_with_plan(
    input.view(), "input -> bitpack\n---\ninput -> identity\n", stream, mr);
  // Non-blocking, as production lanes are: no implicit ordering with the legacy default stream.
  rmm::cuda_stream lane{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream out{rmm::cuda_stream::flags::non_blocking};
  std::array<::cuda::stream_ref, 4> const lanes{lane, out, lane, out};
  std::array<std::size_t, 2> selected{0, 1};

  // Two sources on different lanes, a full-width column, and a keep mask exercise every join.
  sc::scan_filter_request request;
  request.routes = {sc::decode_route::bitpack_mask, sc::decode_route::full};
  request.filters.push_back({0, {0, 19}});
  request.filters.push_back({0, {5, row_count - 1}});
  std::vector<std::uint32_t> keep((row_count + 31) / 32, ~std::uint32_t{0});
  request.keep_mask_words = keep.data();
  request.keep_mask_rows  = row_count;
  std::vector<std::int32_t> survivors(15);
  std::iota(survivors.begin(), survivors.end(), 5);
  for (bool pooled : {true, false}) {
    sc::scan_filter_result result;
    auto output =
      pooled
        ? simpatico::decompress_scan_filter(compressed, selected, request, result, pool, out, mr)
        : simpatico::decompress_scan_filter(compressed, selected, request, result, lanes, out, mr);
    expect(result.applied && result.survivor_count == 15,
           "borrowed lanes: filtered decode applied");
    verify_values(output->view(), survivors, stream);
  }

  // A membership probe runs on its assigned lane, whichever handle that repeats.
  request.filters.pop_back();
  request.keep_mask_words = nullptr;
  request.keep_mask_rows  = 0;
  cudf::numeric_scalar<std::int32_t> const lower(10005, true, stream, mr);
  stream.sync();
  request.membership_filters.push_back(
    {1,
     [&lanes, &lower](cudf::column_view keys,
                      ::cuda::stream_ref assigned,
                      rmm::device_async_resource_ref resource) {
       expect(std::find(lanes.begin(), lanes.end(), assigned) != lanes.end(),
              "probe uses a borrowed lane");
       return cudf::binary_operation(keys,
                                     lower,
                                     cudf::binary_operator::GREATER_EQUAL,
                                     cudf::data_type{cudf::type_id::BOOL8},
                                     assigned,
                                     resource);
     }});
  {
    sc::scan_filter_result result;
    auto output =
      simpatico::decompress_scan_filter(compressed, selected, request, result, lanes, out, mr);
    expect(result.applied && result.survivor_count == 15, "borrowed lanes: membership source");
    verify_values(output->view(), survivors, stream);
  }

  std::vector<std::int32_t> all(row_count);
  std::iota(all.begin(), all.end(), 0);
  std::array<simpatico::decode_predicate, 2> const no_predicates{};
  verify_values(simpatico::decompress(compressed, selected, pool, mr)->view(), all, stream);
  verify_values(simpatico::decompress(compressed, selected, lanes, mr)->view(), all, stream);
  verify_values(
    simpatico::decompress(compressed, selected, no_predicates, lanes, mr)->view(), all, stream);

  // A BOOL8-only request leaves wave 2 empty, so the phase itself waits on the repeated lanes.
  {
    std::vector<std::string> strings(row_count, "other");
    cudf::size_type matches = 0;
    for (int i = 0; i < row_count; i += 97, ++matches)
      strings[i] = "match";
    std::vector<std::unique_ptr<cudf::column>> string_columns;
    string_columns.push_back(make_strings_column(strings, {}, stream));
    cudf::table string_input{std::move(string_columns)};
    auto dictionary =
      simpatico::compress_with_plan(string_input.view(),
                                    "input -> dictionary -> keys_offsets, keys_chars, indices\n"
                                    "dictionary.indices -> bitpack\n",
                                    stream,
                                    mr);
    std::array<std::size_t, 1> const only{0};
    sc::scan_filter_request bool8;
    bool8.routes = {sc::decode_route::dict_codes};
    bool8.bool8_filters.push_back({0, {"match"}});
    sc::scan_filter_result result;
    auto output =
      simpatico::decompress_scan_filter(dictionary, only, bool8, result, lanes, out, mr);
    expect(result.applied && output->num_rows() == matches, "borrowed lanes: BOOL8-only request");
    auto flags = read<std::uint8_t>(output->view().column(0), stream);
    expect(std::all_of(flags.begin(), flags.end(), [](auto flag) { return flag != 0; }),
           "borrowed lanes: BOOL8 values");
  }

  // The borrowed lanes still accept work after the calls.
  rmm::device_buffer probe(sizeof(std::uint32_t), stream, mr);
  stream.sync();
  for (auto const borrowed : lanes) {
    check(cudaMemsetAsync(probe.data(), 0, probe.size(), borrowed.get()));
    check(cudaStreamSynchronize(borrowed.get()));
  }
}

// The dict_codes route with a published key width gathers straight from the stored key chars;
// with the hint cleared the same request takes the general route (compacted codes, then a
// dictionary rebuilt with an unknown width it measures). Both must yield the filtered strings.
void dict_codes_gather(simpatico::stream_pool& pool,
                       ::cuda::stream_ref stream,
                       rmm::device_async_resource_ref mr)
{
  std::vector<std::string> const keys{"AB", "CD", "EF"};
  std::vector<std::string> strings(row_count);
  for (int i = 0; i < row_count; ++i)
    strings[i] = keys[(i * 5 + i / 7) % keys.size()];
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(make_strings_column(strings, {}, stream));
  columns.push_back(sequence(0, stream, mr));
  cudf::table input{std::move(columns)};
  auto compressed =
    simpatico::compress_with_plan(input.view(),
                                  "input -> dictionary -> keys_offsets, keys_chars, indices\n"
                                  "dictionary.indices -> bitpack\n---\ninput -> bitpack\n",
                                  stream,
                                  mr);
  auto& tree      = *compressed.columns[0].plan_tree;
  auto dictionary = std::find_if(
    tree.nodes.begin(), tree.nodes.end(), [](auto const& node) { return node.op == "dictionary"; });
  expect(dictionary != tree.nodes.end() && dictionary->dictionary_key_width_hint == 2,
         "dict_codes fixture did not publish its key width");

  std::vector<std::string> const expected_strings(strings.begin(), strings.begin() + 20);
  auto expected = make_strings_column(expected_strings, {}, stream);
  std::array<std::size_t, 2> selected{0, 1};
  sc::scan_filter_request request;
  request.routes = {sc::decode_route::dict_codes, sc::decode_route::bitpack_mask};
  request.filters.push_back({1, {0, 19}});
  sc::scan_filter_result result;
  auto output =
    simpatico::decompress_scan_filter(compressed, selected, request, result, pool, stream, mr);
  expect(result.applied && result.survivor_count == 20, "dict_codes range filter applied");
  expect(output->view().column(0).type().id() == cudf::type_id::STRING &&
           strings_equal(expected->view(), output->view().column(0), stream),
         "dict_codes gather strings");

  auto mask = result.view();
  simpatico::decode_selection selection;
  selection.mask           = &mask;
  selection.survivor_count = mask.survivor_count;
  selection.route          = sc::decode_route::dict_codes;
  std::string error;
  auto hinted = simpatico::decompress_column(tree, stream, mr, &error, nullptr, &selection);
  expect(hinted != nullptr, error.c_str());
  expect(strings_equal(expected->view(), hinted->view(), stream),
         "direct dict_codes gather with the published width");
  // Clearing the hint before the decode is a test-only way to force the general route.
  dictionary->dictionary_key_width_hint = -1;
  auto measured = simpatico::decompress_column(tree, stream, mr, &error, nullptr, &selection);
  expect(measured != nullptr, error.c_str());
  expect(strings_equal(hinted->view(), measured->view(), stream),
         "dict_codes general route differs from the hinted gather");
  // A positive hint that does not describe the stored key chars is corrupt metadata: the gather
  // specialization declines instead of addressing the chars with it, and the general route then
  // rejects the hint.
  dictionary->dictionary_key_width_hint = 3;
  bool rejected                         = false;
  try {
    (void)simpatico::decompress_column(tree, stream, mr, &error, nullptr, &selection);
  } catch (std::invalid_argument const&) {
    rejected = true;
  }
  expect(rejected, "dict_codes gather accepted a key width hint that contradicts the key chars");
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
    dict_codes_gather(pool, stream.view(), resource);
    aliased_borrowed_lanes(pool, stream.view(), resource);
    check(pool.sync_all());
    stream.synchronize();
    std::puts("test_scan_filter_session: OK");
    return 0;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "test_scan_filter_session: FAIL: %s\n", error.what());
    return 1;
  }
}
