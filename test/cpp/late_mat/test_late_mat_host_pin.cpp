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

// [late_mat][host_pin] — materializing out of a HOST-tier pin. GPU required.
//
// A host chunk stores every pinned column in one representation, over pinned
// blocks that are not contiguous with one another, which is the opposite
// orientation from a GPU pin and has no cudf::column_view to gather from. The
// resolver bridges the two by handing the materializer the blocks plus a byte
// offset, and the gather reads the rows where they lie.
//
// The pin holds row i at value i, so a materialized value IS its own global row
// id: a resolver that mis-orders the chunks, or a gather that translates an
// offset wrongly, produces a column that names the rows it actually read.
//
// The chunks here are deliberately larger than one 1 MiB block, because the
// whole point of the blocked addressing is the boundary a column's buffer
// crosses; a chunk that fits in one block would exercise none of it.

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/sirius_converter_registry.hpp>
#include <late_mat/host_gather_policy.hpp>
#include <late_mat/materialize.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>
#include <scan_manager/late_mat_resolver.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <sirius_context.hpp>
#include <utils/utils.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using cucascade::memory::Tier;
using sirius::late_mat::column_origin;
using sirius::late_mat::materialize;
using sirius::late_mat::pin_entry_handle;
using sirius::late_mat::prepared_selection;
using sirius::late_mat::row_id_list;
using sirius::scan_manager::host_pinned_column_is_addressable;
using sirius::scan_manager::pinned_entry;
using sirius::scan_manager::resolve_pinned_column;
using sirius::scan_manager::resolve_pinned_layout;

namespace {

/// Enough rows that a single INT64 column's buffer spans several 1 MiB blocks.
constexpr std::int64_t kRowsPerChunk = 400'000;

std::filesystem::path test_config_path()
{
  return std::filesystem::path(__FILE__).parent_path().parent_path() / "operator" / "result.yaml";
}

/// Every third row null, so the mask has both states in every word.
bool row_is_valid(std::int64_t global_row) { return global_row % 3 != 0; }

/// A STRING column of @p values, built by hand so the fixture depends on no
/// test-utility template.
std::unique_ptr<cudf::column> make_labels(std::vector<std::string> const& values,
                                          rmm::cuda_stream_view stream)
{
  auto const mr = rmm::mr::get_current_device_resource_ref();
  std::vector<cudf::size_type> offsets;
  offsets.reserve(values.size() + 1);
  std::string chars;
  cudf::size_type cursor = 0;
  for (auto const& v : values) {
    offsets.push_back(cursor);
    chars += v;
    cursor += static_cast<cudf::size_type>(v.size());
  }
  offsets.push_back(cursor);

  auto offsets_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                               static_cast<cudf::size_type>(offsets.size()),
                                               cudf::mask_state::UNALLOCATED,
                                               stream,
                                               mr);
  cudaMemcpyAsync(offsets_col->mutable_view().data<cudf::size_type>(),
                  offsets.data(),
                  offsets.size() * sizeof(cudf::size_type),
                  cudaMemcpyHostToDevice,
                  stream.value());
  rmm::device_buffer chars_buf(chars.size(), stream, mr);
  if (!chars.empty()) {
    cudaMemcpyAsync(
      chars_buf.data(), chars.data(), chars.size(), cudaMemcpyHostToDevice, stream.value());
  }
  cudaStreamSynchronize(stream.value());
  return cudf::make_strings_column(static_cast<cudf::size_type>(values.size()),
                                   std::move(offsets_col),
                                   std::move(chars_buf),
                                   0,
                                   rmm::device_buffer{0, stream, mr});
}

/// One chunk of the pin: an INT64 column holding its own global row id, an
/// optional nullable twin of it, and a STRING column that the host gather has to
/// refuse.
std::unique_ptr<cudf::table> make_chunk(std::int64_t first_row,
                                        std::int64_t rows,
                                        rmm::cuda_stream_view stream)
{
  std::vector<std::int64_t> ids(static_cast<std::size_t>(rows));
  std::iota(ids.begin(), ids.end(), first_row);

  auto upload = [&](cudf::mask_state mask) {
    auto col = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64}, static_cast<cudf::size_type>(rows), mask, stream);
    cudaMemcpyAsync(col->mutable_view().data<std::int64_t>(),
                    ids.data(),
                    ids.size() * sizeof(std::int64_t),
                    cudaMemcpyHostToDevice,
                    stream.value());
    return col;
  };

  auto plain    = upload(cudf::mask_state::UNALLOCATED);
  auto nullable = upload(cudf::mask_state::UNINITIALIZED);

  std::vector<cudf::bitmask_type> words(
    static_cast<std::size_t>(cudf::num_bitmask_words(static_cast<cudf::size_type>(rows))), 0);
  std::size_t nulls = 0;
  for (std::int64_t i = 0; i < rows; ++i) {
    if (row_is_valid(first_row + i)) {
      words[static_cast<std::size_t>(i) / 32] |= (1U << (static_cast<std::size_t>(i) % 32));
    } else {
      ++nulls;
    }
  }
  cudaMemcpyAsync(nullable->mutable_view().null_mask(),
                  words.data(),
                  words.size() * sizeof(cudf::bitmask_type),
                  cudaMemcpyHostToDevice,
                  stream.value());
  cudaStreamSynchronize(stream.value());
  nullable->set_null_count(static_cast<cudf::size_type>(nulls));

  std::vector<std::string> labels;
  labels.reserve(static_cast<std::size_t>(rows));
  for (std::int64_t i = 0; i < rows; ++i) {
    labels.push_back("row-" + std::to_string(first_row + i));
  }
  auto text = make_labels(labels, stream);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(plain));
  columns.push_back(std::move(nullable));
  columns.push_back(std::move(text));
  return std::make_unique<cudf::table>(std::move(columns));
}

/// A HOST-tier pin of the chunks above, built the way the pin driver builds one:
/// each chunk materialized on the GPU and converted to a pinned host
/// representation, with the GPU table released before the next.
struct host_pin {
  pinned_entry entry;
  std::shared_ptr<pin_entry_handle> handle;

  host_pin(std::vector<std::int64_t> const& chunk_rows,
           cucascade::memory::memory_space& gpu_space,
           cucascade::memory::memory_space& host_space,
           rmm::cuda_stream_view stream)
  {
    entry.tier         = Tier::HOST;
    entry.memory_space = &host_space;
    entry.cache_info.names.assign({"id", "id_nullable", "label"});

    auto& registry     = sirius::converter_registry::get();
    std::int64_t first = 0;
    for (auto const rows : chunk_rows) {
      auto table = make_chunk(first, rows, stream);
      first += rows;
      entry.num_rows += static_cast<std::size_t>(rows);

      cucascade::gpu_table_representation gpu(std::move(table), gpu_space, stream);
      auto host = registry.convert<cucascade::host_data_representation>(gpu, &host_space, stream);
      REQUIRE(host);
      entry.host_chunks.push_back(
        std::shared_ptr<cucascade::idata_representation>(std::move(host)));
    }

    handle = std::make_shared<pin_entry_handle>("host_pin", 7);
    handle->set_entry(&entry);
  }

  [[nodiscard]] column_origin origin(std::uint32_t pos) const
  {
    column_origin o;
    o.handle     = handle;
    o.column_pos = pos;
    o.generation = handle->generation();
    return o;
  }
};

rmm::device_buffer upload_ids(std::vector<std::uint64_t> const& host, rmm::cuda_stream_view stream)
{
  rmm::device_buffer buf(
    host.size() * sizeof(std::uint64_t), stream, rmm::mr::get_current_device_resource_ref());
  cudaMemcpyAsync(buf.data(),
                  host.data(),
                  host.size() * sizeof(std::uint64_t),
                  cudaMemcpyHostToDevice,
                  stream.value());
  cudaStreamSynchronize(stream.value());
  return buf;
}

std::vector<std::int64_t> read_values(cudf::column_view const& col)
{
  std::vector<std::int64_t> host(static_cast<std::size_t>(col.size()));
  if (!host.empty()) {
    cudaMemcpy(host.data(),
               col.data<std::int64_t>(),
               host.size() * sizeof(std::int64_t),
               cudaMemcpyDeviceToHost);
  }
  return host;
}

std::vector<bool> read_validity(cudf::column_view const& col)
{
  std::vector<bool> valid(static_cast<std::size_t>(col.size()), true);
  if (!col.nullable() || col.size() == 0) { return valid; }
  std::vector<cudf::bitmask_type> words(
    static_cast<std::size_t>(cudf::num_bitmask_words(col.size())));
  cudaMemcpy(words.data(),
             col.null_mask(),
             words.size() * sizeof(cudf::bitmask_type),
             cudaMemcpyDeviceToHost);
  for (cudf::size_type i = 0; i < col.size(); ++i) {
    valid[static_cast<std::size_t>(i)] =
      ((words[static_cast<std::size_t>(i) / 32] >> (static_cast<std::size_t>(i) % 32)) & 1U) != 0U;
  }
  return valid;
}

/// Pins a route for the duration of a scope and hands it back afterwards, so a
/// failed assertion cannot leak the setting into the rest of the binary.
struct forced_route {
  explicit forced_route(bool inplace) { sirius::late_mat::force_host_gather_route(inplace); }
  forced_route(forced_route const&)            = delete;
  forced_route& operator=(forced_route const&) = delete;
  ~forced_route() { sirius::late_mat::force_host_gather_route(std::nullopt); }
};

}  // namespace

TEST_CASE("late-mat materializes out of a host-tier pin", "[late_mat][host_pin][shared_context]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  auto sirius_ctx      = sirius::get_sirius_context(con, test_config_path());
  auto& manager        = sirius_ctx->get_memory_manager();
  auto* gpu_space =
    const_cast<cucascade::memory::memory_space*>(manager.get_memory_space(Tier::GPU, 0));
  auto* host_space =
    const_cast<cucascade::memory::memory_space*>(manager.get_memory_space(Tier::HOST, 0));
  if (host_space == nullptr) {
    auto const spaces = manager.get_memory_spaces_for_tier(Tier::HOST);
    REQUIRE_FALSE(spaces.empty());
    host_space = const_cast<cucascade::memory::memory_space*>(spaces.front());
  }
  REQUIRE(gpu_space != nullptr);
  REQUIRE(host_space != nullptr);

  // A real stream, not the legacy default one: the GPU-to-host converter fires
  // cudaMemcpyBatchAsync, which rejects the default stream. It outlives the pin
  // below, whose host chunks were written on it.
  rmm::cuda_stream owned_stream;
  auto const stream = owned_stream.view();
  auto mr           = rmm::mr::get_current_device_resource_ref();

  std::vector<std::int64_t> const chunk_rows{kRowsPerChunk, kRowsPerChunk / 2, kRowsPerChunk};
  host_pin pin(chunk_rows, *gpu_space, *host_space, stream);

  SECTION("the layout describes the chunks the entry actually holds")
  {
    auto const layout = resolve_pinned_layout(pin.origin(0));
    REQUIRE(layout.has_value());
    REQUIRE(layout->batch_rows == chunk_rows);
    REQUIRE(layout->total_rows() == static_cast<std::int64_t>(pin.entry.num_rows));
  }

  SECTION("a fixed-width column gathers its own row ids back, across block boundaries")
  {
    auto const layout = resolve_pinned_layout(pin.origin(0));
    REQUIRE(layout.has_value());
    auto const column = resolve_pinned_column(pin.origin(0));
    REQUIRE(column.has_value());
    REQUIRE(column->dtype == cudf::data_type{cudf::type_id::INT64});
    REQUIRE(column->batches.size() == chunk_rows.size());
    REQUIRE(column->batches.front().is_host());

    // Ids picked to land in every chunk, unordered and with a repeat, since
    // those are ordinary gather semantics rather than a special case.
    std::vector<std::uint64_t> ids{0,
                                   131'071,   // last INT64 of the first 1 MiB block
                                   131'072,   // first INT64 of the second
                                   399'999,   // last row of chunk 0
                                   400'000,   // first row of chunk 1
                                   599'999,   // last row of chunk 1
                                   600'000,   // first row of chunk 2
                                   999'999,   // last row of the pin
                                   131'072};  // a repeat
    auto const buf = upload_ids(ids, stream);
    row_id_list list{
      static_cast<std::uint64_t const*>(buf.data()), static_cast<std::int64_t>(ids.size()), false};
    prepared_selection selection(*layout, list);

    auto const produced = materialize(*column, selection, stream, mr);
    cudaStreamSynchronize(stream.value());
    REQUIRE(produced->size() == static_cast<cudf::size_type>(ids.size()));
    REQUIRE(produced->null_count() == 0);
    auto const values = read_values(produced->view());
    for (std::size_t i = 0; i < ids.size(); ++i) {
      REQUIRE(values[i] == static_cast<std::int64_t>(ids[i]));
    }
  }

  SECTION("validity rides along with the values")
  {
    auto const layout = resolve_pinned_layout(pin.origin(1));
    REQUIRE(layout.has_value());
    auto const column = resolve_pinned_column(pin.origin(1));
    REQUIRE(column.has_value());
    REQUIRE(column->batches.front().host->has_null_mask);

    std::vector<std::uint64_t> ids;
    for (std::uint64_t id = 399'990; id < 400'010; ++id) {
      ids.push_back(id);
    }
    auto const buf = upload_ids(ids, stream);
    row_id_list list{
      static_cast<std::uint64_t const*>(buf.data()), static_cast<std::int64_t>(ids.size()), true};
    prepared_selection selection(*layout, list);

    auto const produced = materialize(*column, selection, stream, mr);
    cudaStreamSynchronize(stream.value());
    auto const values = read_values(produced->view());
    auto const valid  = read_validity(produced->view());
    for (std::size_t i = 0; i < ids.size(); ++i) {
      auto const id = static_cast<std::int64_t>(ids[i]);
      REQUIRE(valid[i] == row_is_valid(id));
      if (valid[i]) { REQUIRE(values[i] == id); }
    }
    REQUIRE(produced->null_count() ==
            static_cast<cudf::size_type>(std::count(valid.begin(), valid.end(), false)));
  }

  SECTION("a variable-width column is refused rather than read as fixed width")
  {
    REQUIRE_FALSE(host_pinned_column_is_addressable(pin.entry, 2));
    REQUIRE_FALSE(resolve_pinned_column(pin.origin(2)).has_value());
    // Its fixed-width neighbours in the same chunks stay addressable: the
    // refusal is per column, not per pin.
    REQUIRE(host_pinned_column_is_addressable(pin.entry, 0));
    REQUIRE(host_pinned_column_is_addressable(pin.entry, 1));
  }

  SECTION("a column position the entry does not have is refused")
  {
    REQUIRE_FALSE(host_pinned_column_is_addressable(pin.entry, 7));
    REQUIRE_FALSE(resolve_pinned_column(pin.origin(7)).has_value());
  }

  SECTION("a stale origin resolves to nothing, whatever the tier")
  {
    auto stale = pin.origin(0);
    pin.handle->bump_generation(pin.handle->generation() + 1);
    REQUIRE_FALSE(resolve_pinned_layout(stale).has_value());
    REQUIRE_FALSE(resolve_pinned_column(stale).has_value());
  }

  SECTION("the route is chosen, not incidental, and both routes agree")
  {
    using sirius::late_mat::host_gather_routes_taken;

    auto const layout = resolve_pinned_layout(pin.origin(1));
    REQUIRE(layout.has_value());
    auto const column = resolve_pinned_column(pin.origin(1));
    REQUIRE(column.has_value());

    // Spread across all three chunks, and nullable, so each route has to carry
    // validity as well as values.
    std::vector<std::uint64_t> ids;
    for (std::uint64_t id = 0; id < 1'000'000; id += 9973) {
      ids.push_back(id);
    }
    auto const buf = upload_ids(ids, stream);
    row_id_list list{
      static_cast<std::uint64_t const*>(buf.data()), static_cast<std::int64_t>(ids.size()), true};

    auto run = [&](bool inplace) {
      forced_route pinned{inplace};
      auto const before = host_gather_routes_taken();
      prepared_selection selection(*layout, list);
      auto produced = materialize(*column, selection, stream, mr);
      cudaStreamSynchronize(stream.value());
      auto const after = host_gather_routes_taken();
      // The branch that ran, asserted directly: a values-only check passes
      // whichever route was taken and would not notice the choice being ignored.
      if (inplace) {
        REQUIRE(after.inplace == before.inplace + 1);
        REQUIRE(after.staged == before.staged);
      } else {
        REQUIRE(after.staged == before.staged + 1);
        REQUIRE(after.inplace == before.inplace);
      }
      return produced;
    };

    auto const in_place = run(true);
    auto const staged   = run(false);

    REQUIRE(in_place->size() == static_cast<cudf::size_type>(ids.size()));
    REQUIRE(staged->size() == in_place->size());
    REQUIRE(staged->null_count() == in_place->null_count());
    REQUIRE(read_values(staged->view()) == read_values(in_place->view()));
    REQUIRE(read_validity(staged->view()) == read_validity(in_place->view()));

    // And the values are the rows that were asked for, not merely two routes
    // agreeing on the same wrong answer.
    auto const values = read_values(in_place->view());
    auto const valid  = read_validity(in_place->view());
    for (std::size_t i = 0; i < ids.size(); ++i) {
      auto const id = static_cast<std::int64_t>(ids[i]);
      REQUIRE(valid[i] == row_is_valid(id));
      if (valid[i]) { REQUIRE(values[i] == id); }
    }
  }

  SECTION("a host entry with no chunks is refused rather than resolved empty")
  {
    pinned_entry empty;
    empty.tier = Tier::HOST;
    empty.cache_info.names.assign({"id"});
    auto handle = std::make_shared<pin_entry_handle>("empty", 1);
    handle->set_entry(&empty);
    column_origin origin;
    origin.handle     = handle;
    origin.column_pos = 0;
    origin.generation = handle->generation();

    REQUIRE_FALSE(host_pinned_column_is_addressable(empty, 0));
    REQUIRE_FALSE(resolve_pinned_layout(origin).has_value());
    REQUIRE_FALSE(resolve_pinned_column(origin).has_value());
  }
}

TEST_CASE("the host gather probe measures this machine's link", "[late_mat][host_pin][probe]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  auto sirius_ctx      = sirius::get_sirius_context(con, test_config_path());
  auto const& policy   = sirius::late_mat::measured_host_gather_policy();

  // Machine-independent invariants only. The crossover is a property of the
  // link, so pinning a number here would assert this GB300's answer on every
  // machine that runs the suite; what must hold everywhere is that the probe
  // either measured something usable or fell back to the conservative policy.
  REQUIRE(policy.max_inplace_density >= 0.0);
  REQUIRE(policy.max_inplace_density <= 1.0);
  REQUIRE(policy.cost_multiplier >= 1);
  REQUIRE(policy.cost_multiplier <= 64);
  if (!policy.measured) {
    REQUIRE(policy.max_inplace_density == 0.0);  // fail closed: stage
  }

  WARN("host gather probe: measured=" << policy.measured
                                      << " bulk=" << policy.bulk_bytes_per_second / 1e9
                                      << " GB/s in-place=" << policy.inplace_bytes_per_second / 1e9
                                      << " GB/s crossover=" << policy.max_inplace_density
                                      << " multiplier=" << policy.cost_multiplier);
}
