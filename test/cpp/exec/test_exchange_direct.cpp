/*
 * Copyright 2026, Sirius Contributors.
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
#include "data/data_batch_utils.hpp"
#include "data/data_repository_manager_registry.hpp"
#include "exec/exchange_direct.hpp"
#include "exec/exchange_staging.hpp"
#include "memory/slab_memory_resource.hpp"
#include "sirius/exception.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/memory/memory_space.hpp>

#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::MessageMatches;
using cucascade::memory::memory_space;
using namespace sirius::exec;

namespace {

constexpr cudf::size_type rows = 100;

cudf::data_type type(cudf::type_id id) { return cudf::data_type{id}; }

direct_column strings(cudf::type_id offsets, std::uint64_t chars)
{
  return {type(cudf::type_id::STRING), 0, false, offsets, chars};
}

template <typename T>
void poke(std::vector<std::uint8_t>& bytes, std::size_t at, T value)
{
  std::memcpy(bytes.data() + at, &value, sizeof(T));
}

std::uint64_t address(void const* p) { return reinterpret_cast<std::uint64_t>(p); }

cucascade::memory::gpu_memory_space_config slab_config()
{
  cucascade::memory::gpu_memory_space_config config;
  config.device_id              = 0;
  config.memory_capacity        = std::size_t{64} << 20;
  config.per_stream_reservation = false;
  config.mr_factory_fn          = sirius::memory::make_slab_pool_factory();
  return config;
}

std::unique_ptr<cudf::column> random_column(cudf::data_type dtype,
                                            rmm::device_async_resource_ref mr)
{
  auto column = cudf::make_fixed_width_column(
    dtype, rows, cudf::mask_state::UNALLOCATED, cudf::get_default_stream(), mr);
  std::vector<std::uint8_t> bytes(rows * cudf::size_of(dtype));
  std::mt19937 random{static_cast<std::uint32_t>(dtype.id())};
  for (auto& b : bytes) {
    b = static_cast<std::uint8_t>(dtype.id() == cudf::type_id::BOOL8 ? random() & 1 : random());
  }
  REQUIRE(
    cudaMemcpy(column->mutable_view().head(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice) ==
    cudaSuccess);
  return column;
}

std::unique_ptr<cudf::column> strings_column(std::string_view value,
                                             rmm::device_async_resource_ref mr)
{
  auto const stream = cudf::get_default_stream();
  return cudf::make_column_from_scalar(
    cudf::string_scalar{value, true, stream, mr}, rows, stream, mr);
}

std::unique_ptr<cudf::column> with_nulls(std::unique_ptr<cudf::column> column,
                                         rmm::device_async_resource_ref mr)
{
  auto mask =
    cudf::create_null_mask(rows, cudf::mask_state::ALL_VALID, cudf::get_default_stream(), mr);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), 0, 3, false);
  column->set_null_mask(std::move(mask), 3);
  return column;
}

std::shared_ptr<cudf::table> sample_table(rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(with_nulls(random_column(type(cudf::type_id::INT32), mr), mr));
  for (auto const t : {type(cudf::type_id::INT64),
                       type(cudf::type_id::FLOAT64),
                       type(cudf::type_id::BOOL8),
                       type(cudf::type_id::TIMESTAMP_DAYS),
                       cudf::data_type{cudf::type_id::DECIMAL64, -2},
                       cudf::data_type{cudf::type_id::DECIMAL128, 3}}) {
    columns.push_back(random_column(t, mr));
  }
  columns.push_back(with_nulls(strings_column("abc", mr), mr));
  columns.push_back(strings_column("", mr));
  return std::make_shared<cudf::table>(std::move(columns));
}

std::shared_ptr<cucascade::data_batch> batch_of(memory_space& gpu,
                                                cudf::table_view view,
                                                std::shared_ptr<cudf::table> owner)
{
  auto const bytes = owner->alloc_size();
  return sirius::make_data_batch_from_view(
    view, std::move(owner), bytes, gpu, cudf::get_default_stream(), {});
}

/// Moves @p sent into a received token, a cudaMemcpy standing in for the transport's write.
std::unique_ptr<cudf::table> deliver(direct_exchange& exchange,
                                     direct_exchange::exported const& sent)
{
  auto [token, dst] = exchange.allocate(sent.layout);
  REQUIRE(dst.size() == sent.src.size());
  for (std::size_t i = 0; i < dst.size(); i += 2) {
    REQUIRE(dst[i + 1] == sent.src[i + 1]);
    REQUIRE(cudaMemcpy(reinterpret_cast<void*>(dst[i]),
                       reinterpret_cast<void const*>(sent.src[i]),
                       dst[i + 1],
                       cudaMemcpyDeviceToDevice) == cudaSuccess);
  }
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  exchange.release(sent.token);
  return exchange.take(token);
}

std::vector<std::uint32_t> row_hashes(cudf::table_view table)
{
  auto const hashes = cudf::hashing::murmurhash3_x86_32(table);
  std::vector<std::uint32_t> host(table.num_rows());
  REQUIRE(cudaMemcpy(host.data(),
                     hashes->view().head(),
                     host.size() * sizeof(std::uint32_t),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  return host;
}

void check_same_rows(cudf::table_view actual, cudf::table_view expected)
{
  REQUIRE(actual.num_columns() == expected.num_columns());
  for (cudf::size_type i = 0; i < actual.num_columns(); ++i) {
    CHECK(actual.column(i).type() == expected.column(i).type());
    CHECK(actual.column(i).null_count() == expected.column(i).null_count());
  }
  CHECK(row_hashes(actual) == row_hashes(expected));
}

}  // namespace

TEST_CASE("decode_layout inverts encode_layout", "[exchange_direct]")
{
  direct_layout const layout{5,
                             {{type(cudf::type_id::INT64), 0, false},
                              {cudf::data_type{cudf::type_id::DECIMAL64, -2}, 2, true},
                              {type(cudf::type_id::BOOL8), 0, true},
                              {type(cudf::type_id::TIMESTAMP_DAYS), 5, true},
                              {type(cudf::type_id::STRING), 1, true, cudf::type_id::INT32, 17},
                              strings(cudf::type_id::INT64, 0)}};
  CHECK(decode_layout(encode_layout(layout)) == layout);
}

TEST_CASE("decode_layout rejects each malformed field", "[exchange_direct]")
{
  // Encoded as: header at 0 (magic, rows at 4, ncols at 8); the INT32 column at 12 (type, scale
  // at 16, null count at 20, mask flag at 24); the STRING column at 25 (offsets type at 38, chars
  // at 42).
  direct_layout const base{
    3, {{type(cudf::type_id::INT32), 1, true}, strings(cudf::type_id::INT32, 5)}};
  auto const valid = encode_layout(base);
  REQUIRE(decode_layout(valid) == base);

  using mutation = std::function<void(std::vector<std::uint8_t>&)>;
  std::vector<std::pair<std::string_view, mutation>> const cases{
    {"bad magic", [](auto& b) { b[3] = '2'; }},
    {"truncated", [](auto& b) { b.pop_back(); }},
    {"trailing bytes", [](auto& b) { b.push_back(0); }},
    {"row count", [](auto& b) { poke<std::int32_t>(b, 4, 0); }},
    {"row count", [](auto& b) { poke(b, 4, std::numeric_limits<std::int32_t>::max()); }},
    {"column count", [](auto& b) { poke<std::uint32_t>(b, 8, 0); }},
    {"column count", [](auto& b) { poke<std::uint32_t>(b, 8, 3); }},
    {"type id", [](auto& b) { poke<std::int32_t>(b, 12, -1); }},
    {"type id", [](auto& b) { poke(b, 12, cudf::type_id::NUM_TYPE_IDS); }},
    {"neither fixed-width nor STRING", [](auto& b) { poke(b, 12, cudf::type_id::EMPTY); }},
    {"neither fixed-width nor STRING", [](auto& b) { poke(b, 12, cudf::type_id::LIST); }},
    {"scale", [](auto& b) { poke<std::int32_t>(b, 16, 2); }},
    {"null count", [](auto& b) { poke<std::int32_t>(b, 20, 4); }},
    {"nulls without a mask", [](auto& b) { poke<std::uint8_t>(b, 24, 0); }},
    {"mask flag", [](auto& b) { poke<std::uint8_t>(b, 24, 2); }},
    {"not INT32 or INT64", [](auto& b) { poke(b, 38, cudf::type_id::INT16); }},
    {"chars overflow INT32", [](auto& b) { poke<std::uint64_t>(b, 42, std::uint64_t{1} << 31); }},
  };
  for (auto const& [error, mutate] : cases) {
    auto bytes = valid;
    mutate(bytes);
    CHECK_THROWS_MATCHES(decode_layout(bytes),
                         sirius::invalid_input_exception,
                         MessageMatches(ContainsSubstring(std::string{error})));
  }
}

TEST_CASE("plan_buffers sizes buffers in walk order within the limit", "[exchange_direct]")
{
  direct_layout const layout{100,
                             {{cudf::data_type{cudf::type_id::DECIMAL128, 3}, 1, true},
                              strings(cudf::type_id::INT64, 1000)}};
  // The mask sends 4 words into a 64-byte allocation; aligned to 256 bytes the four take 4096.
  CHECK(plan_buffers(layout, 4096) ==
        std::vector<direct_buffer>{{0, 16, 64}, {0, 1600, 1600}, {1, 1000, 1000}, {1, 808, 808}});
  CHECK_THROWS_AS(plan_buffers(layout, 4095), sirius::invalid_input_exception);

  // Sizes whose aligned sum wraps around size_t must not pass as small ones.
  auto const max = std::numeric_limits<std::size_t>::max();
  CHECK_THROWS_AS(plan_buffers({1, {strings(cudf::type_id::INT64, max)}}, max),
                  sirius::invalid_input_exception);
  CHECK_THROWS_AS(
    plan_buffers(
      {1, {strings(cudf::type_id::INT64, max / 2 + 1), strings(cudf::type_id::INT64, max / 2 + 1)}},
      max),
    sirius::invalid_input_exception);
}

TEST_CASE("describe_table lists a table's buffers in plan order", "[exchange_direct]")
{
  auto const stream = cudf::get_default_stream();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
    cudf::make_numeric_column(type(cudf::type_id::INT32), 3, cudf::mask_state::ALL_NULL));
  columns.push_back(
    cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::DECIMAL64, -2}, 3));
  columns.push_back(cudf::make_column_from_scalar(cudf::string_scalar{std::string_view{"abc"}}, 3));
  cudf::table const table(std::move(columns));
  auto const view = table.view();

  auto const [layout, buffers] = describe_table(view, stream);
  CHECK(layout == direct_layout{3,
                                {{type(cudf::type_id::INT32), 3, true},
                                 {cudf::data_type{cudf::type_id::DECIMAL64, -2}, 0, false},
                                 strings(cudf::type_id::INT32, 9)}});
  CHECK(buffers == std::vector<void const*>{view.column(0).null_mask(),
                                            view.column(0).head(),
                                            view.column(1).head(),
                                            view.column(2).head(),
                                            view.column(2).child(0).head()});
  CHECK(plan_buffers(layout, std::numeric_limits<std::size_t>::max()).size() == buffers.size());

  cudf::column_view const empty{type(cudf::type_id::EMPTY), 3, nullptr, nullptr, 3};
  CHECK_THROWS_AS(describe_table(cudf::table_view{std::vector<cudf::column_view>{empty}}, stream),
                  sirius::invalid_input_exception);
}

TEST_CASE("direct_exchange sends each type, copying only slices and buffers outside the slab",
          "[exchange_direct]")
{
  memory_space gpu(slab_config());
  direct_exchange exchange(gpu, *sirius::memory::find_slab(gpu));
  bool const in_slab         = GENERATE(true, false);
  cudf::size_type const from = GENERATE(0, 5);
  auto const table =
    sample_table(in_slab ? gpu.get_default_allocator() : cudf::get_current_device_resource_ref());
  auto const view = cudf::slice(table->view(), {from, rows}).front();

  auto const sent = exchange.export_batch(batch_of(gpu, view, table));
  REQUIRE(sent);
  CHECK(sent->rows == static_cast<std::uint64_t>(rows - from));
  // The first buffer is column 0's mask.
  CHECK((sent->src[0] == address(view.column(0).null_mask())) == (in_slab && from == 0));
  check_same_rows(deliver(exchange, *sent)->view(), view);
  CHECK(exchange.outstanding() == 0);

  CHECK_FALSE(
    exchange.export_batch(batch_of(gpu, cudf::slice(table->view(), {0, 0}).front(), table)));
}

TEST_CASE("direct_exchange release is idempotent and close frees every token", "[exchange_direct]")
{
  memory_space gpu(slab_config());
  direct_exchange exchange(gpu, *sirius::memory::find_slab(gpu));
  auto const available    = gpu.get_available_memory();
  auto const layout       = encode_layout({rows, {{type(cudf::type_id::INT64), 0, false}}});
  auto const [token, dst] = exchange.allocate(layout);
  CHECK(dst == std::vector<std::uint64_t>{dst[0], rows * sizeof(std::int64_t)});
  CHECK(exchange.allocate(layout).first == token + 1);

  exchange.release(token);
  exchange.release(token);
  exchange.release(0);
  CHECK(exchange.outstanding() == 1);
  exchange.close();
  CHECK(gpu.get_available_memory() == available);
  CHECK_THROWS_AS(exchange.allocate(layout), sirius::invalid_input_exception);
}

TEST_CASE("direct_exchange seals a received batch until take_batch or release", "[exchange_direct]")
{
  memory_space gpu(slab_config());
  direct_exchange exchange(gpu, *sirius::memory::find_slab(gpu));
  sirius::data::data_repository_manager_registry registry;
  exchange_staging staging(registry);
  std::size_t made_room = 0;
  exchange.enable_spill(staging, [&](std::size_t) { ++made_room; });

  auto const table = sample_table(gpu.get_default_allocator());
  auto const sent  = exchange.export_batch(batch_of(gpu, table->view(), table));
  REQUIRE(sent);
  auto [token, dst] = exchange.allocate(sent->layout);
  for (std::size_t i = 0; i < dst.size(); i += 2) {
    REQUIRE(cudaMemcpy(reinterpret_cast<void*>(dst[i]),
                       reinterpret_cast<void const*>(sent->src[i]),
                       dst[i + 1],
                       cudaMemcpyDeviceToDevice) == cudaSuccess);
  }
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  exchange.release(sent->token);

  exchange.seal(token);
  exchange.seal(token);        // idempotent
  exchange.seal(token + 100);  // unknown: ignored, like release()
  CHECK(exchange.outstanding() == 1);
  // The sealed batch is what the downgrade executor's sweep sees under the reserved query id.
  auto const manager = registry.get(exchange_staging::query_id);
  REQUIRE(manager);
  REQUIRE(manager->get_repositories().size() == 1);
  CHECK(manager->get_repositories().front()->get_batch_ids().size() == 1);

  auto batch = exchange.take_batch(token);
  REQUIRE(batch);
  check_same_rows(sirius::get_cudf_table_view(batch->to_read_only()), table->view());
  CHECK(exchange.outstanding() == 0);
  CHECK(manager->get_repositories().front()->get_batch_ids().empty());
  CHECK_THROWS_AS(exchange.take_batch(token), sirius::invalid_input_exception);
  CHECK(made_room == 0);

  // A sealed token that is released, or still held at close(), is freed.
  auto const second = exchange.allocate(sent->layout).first;
  exchange.seal(second);
  exchange.release(second);
  CHECK(exchange.outstanding() == 0);
  auto const third = exchange.allocate(sent->layout).first;
  exchange.seal(third);
  exchange.close();
  CHECK(manager->get_repositories().front()->get_batch_ids().empty());
}

TEST_CASE("direct_exchange makes room once before refusing an allocation", "[exchange_direct]")
{
  memory_space gpu(slab_config());
  direct_exchange exchange(gpu, *sirius::memory::find_slab(gpu));
  sirius::data::data_repository_manager_registry registry;
  exchange_staging staging(registry);
  std::vector<std::size_t> asked;
  exchange.enable_spill(staging, [&](std::size_t bytes) { asked.push_back(bytes); });

  // Half the slab is taken, then three quarters are asked for: within the slab's size, so the
  // request is well formed, but more than is left. The reservation fails, room is asked for
  // once, and the retry fails too because nothing here can spill.
  auto const slab_rows = [&](double share) {
    return static_cast<cudf::size_type>(static_cast<double>(exchange.region().len) * share /
                                        sizeof(std::int64_t));
  };
  auto const int64_layout = [](cudf::size_type n) {
    return encode_layout({n, {{type(cudf::type_id::INT64), 0, false}}});
  };
  auto const held = exchange.allocate(int64_layout(slab_rows(0.5))).first;
  CHECK_THROWS_AS(exchange.allocate(int64_layout(slab_rows(0.75))), rmm::out_of_memory);
  REQUIRE(asked.size() == 1);
  CHECK(asked.front() >= static_cast<std::size_t>(slab_rows(0.75)) * sizeof(std::int64_t));
  exchange.release(held);
}

TEST_CASE("direct_exchange asks again when making room was cancelled", "[exchange_direct]")
{
  memory_space gpu(slab_config());
  direct_exchange exchange(gpu, *sirius::memory::find_slab(gpu));
  sirius::data::data_repository_manager_registry registry;
  exchange_staging staging(registry);

  auto const int64_layout = [&](double share) {
    auto const rows = static_cast<cudf::size_type>(static_cast<double>(exchange.region().len) *
                                                   share / sizeof(std::int64_t));
    return encode_layout({rows, {{type(cudf::type_id::INT64), 0, false}}});
  };
  // The first request is cancelled, as a draining downgrade executor does; the second frees
  // what blocks the allocation.
  auto held = exchange.allocate(int64_layout(0.5)).first;
  int asked = 0;
  exchange.enable_spill(staging, [&](std::size_t) {
    if (++asked == 1) { throw std::runtime_error("downgrade executor shutting down"); }
    exchange.release(std::exchange(held, 0));
  });
  auto const [token, buffers] = exchange.allocate(int64_layout(0.75));
  CHECK(asked == 2);
  CHECK_FALSE(buffers.empty());
  exchange.release(token);

  // One that keeps failing gives up after a few attempts, with the pool's own error.
  held  = exchange.allocate(int64_layout(0.5)).first;
  asked = 0;
  exchange.enable_spill(staging, [&](std::size_t) {
    ++asked;
    throw std::runtime_error("downgrade executor shutting down");
  });
  CHECK_THROWS_AS(exchange.allocate(int64_layout(0.75)), rmm::out_of_memory);
  CHECK(asked == 3);
  exchange.release(held);
}

TEST_CASE("exchange_staging lists tracked repositories until they are gone", "[exchange_direct]")
{
  memory_space gpu(slab_config());
  sirius::data::data_repository_manager_registry registry;
  exchange_staging staging(registry);
  auto const manager = registry.get(exchange_staging::query_id);
  REQUIRE(manager);
  auto* const view = manager->get_repositories().front();

  auto repository  = std::make_shared<cucascade::shared_data_repository>();
  auto const table = sample_table(gpu.get_default_allocator());
  auto batch       = batch_of(gpu, table->view(), table);
  auto const id    = batch->get_batch_id();
  repository->add_data_batch(batch);
  staging.track(repository);
  CHECK(staging.tracked() == 1);
  CHECK(view->get_batch_ids() == std::vector<std::uint64_t>{id});
  CHECK(view->get_data_batch_by_id(id) == batch);
  CHECK(view->pop_next_data_batch() == nullptr);  // the view owns nothing

  repository.reset();
  CHECK(staging.tracked() == 0);
  CHECK(view->get_batch_ids().empty());
}
