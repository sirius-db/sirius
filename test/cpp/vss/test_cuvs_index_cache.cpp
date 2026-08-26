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

// test
#include "operator/operator_test_utils.hpp"

#include <catch.hpp>

// sirius
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <cuvs/distance/distance.hpp>
#include <vss/cuvs_index_cache.hpp>
#include <vss/ivf_flat_index.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace {

using sirius::vss::cuvs_index_cache;
using sirius::vss::index_kind;
using sirius::vss::index_metadata;
using Metric = cuvs::distance::DistanceType;

// Build a Sirius-style FLOAT[dim] column (a cudf LIST with a contiguous FLOAT32
// values child), the per-batch shape build_ivf_flat_index_from_batches expects.
std::unique_ptr<cudf::column> make_float_list(std::vector<float> const& values,
                                              cudf::size_type n_rows,
                                              cudf::size_type dim)
{
  auto child = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT32}, n_rows * dim, cudf::mask_state::UNALLOCATED);
  cudaMemcpy(child->mutable_view().data<float>(),
             values.data(),
             sizeof(float) * values.size(),
             cudaMemcpyHostToDevice);

  std::vector<int32_t> offsets(static_cast<std::size_t>(n_rows) + 1);
  for (cudf::size_type i = 0; i <= n_rows; ++i) {
    offsets[static_cast<std::size_t>(i)] = i * dim;
  }
  auto offsets_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, n_rows + 1, cudf::mask_state::UNALLOCATED);
  cudaMemcpy(offsets_col->mutable_view().data<int32_t>(),
             offsets.data(),
             sizeof(int32_t) * offsets.size(),
             cudaMemcpyHostToDevice);

  return cudf::make_lists_column(
    n_rows, std::move(offsets_col), std::move(child), 0, rmm::device_buffer{});
}

index_metadata make_meta(std::string table,
                         std::string column,
                         Metric metric,
                         std::string catalog = "mem",
                         std::string schema  = "main")
{
  index_metadata meta;
  meta.kind           = index_kind::ivf_flat;
  meta.catalog_name   = std::move(catalog);
  meta.schema_name    = std::move(schema);
  meta.table_name     = std::move(table);
  meta.column_name    = std::move(column);
  meta.dim            = 3;
  meta.num_rows       = 100;
  meta.n_lists        = 4;
  meta.metric         = metric;
  meta.resident_bytes = 1024;
  return meta;
}

// Pin a dummy (type-erased) index under `name`. The cache is index-type agnostic,
// so an int holder stands in for a real cuVS index; only the metadata and the
// entry lifetime are under test here.
void insert_dummy(cuvs_index_cache& cache, std::string name, index_metadata meta)
{
  cache.insert(
    std::move(name), std::move(meta), sirius::vss::make_cuvs_index(int{0}), rmm::cuda_stream{});
}

}  // namespace

TEST_CASE("cuvs_index_cache inserts and looks up by management name", "[vss]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  cuvs_index_cache cache(*manager);

  REQUIRE(cache.size() == 0);
  REQUIRE(cache.find("idx") == nullptr);
  REQUIRE_FALSE(cache.contains("idx"));

  insert_dummy(cache, "idx", make_meta("docs", "vec", Metric::L2SqrtExpanded));

  REQUIRE(cache.size() == 1);
  REQUIRE(cache.contains("idx"));
  auto entry = cache.find("idx");
  REQUIRE(entry != nullptr);
  REQUIRE(entry->meta.table_name == "docs");
  REQUIRE(entry->meta.column_name == "vec");
  REQUIRE(entry->meta.metric == Metric::L2SqrtExpanded);
  REQUIRE(entry->index != nullptr);
  REQUIRE(entry->meta.resident_bytes == 1024);
}

TEST_CASE("cuvs_index_cache find_by_column matches the auto-routing identity", "[vss]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  cuvs_index_cache cache(*manager);

  insert_dummy(cache, "idx", make_meta("docs", "vec", Metric::L2SqrtExpanded));

  SECTION("exact (catalog, schema, table, column, metric) match hits")
  {
    auto e = cache.find_by_column("mem", "main", "docs", "vec", Metric::L2SqrtExpanded);
    REQUIRE(e != nullptr);
    REQUIRE(e->meta.table_name == "docs");
  }
  SECTION("wrong column misses")
  {
    REQUIRE(cache.find_by_column("mem", "main", "docs", "other", Metric::L2SqrtExpanded) ==
            nullptr);
  }
  SECTION("wrong table misses")
  {
    REQUIRE(cache.find_by_column("mem", "main", "other", "vec", Metric::L2SqrtExpanded) == nullptr);
  }
  SECTION("wrong schema misses (a same-named table in another schema does not route here)")
  {
    REQUIRE(cache.find_by_column("mem", "other", "docs", "vec", Metric::L2SqrtExpanded) == nullptr);
  }
  SECTION("wrong catalog misses")
  {
    REQUIRE(cache.find_by_column("other", "main", "docs", "vec", Metric::L2SqrtExpanded) ==
            nullptr);
  }
  SECTION("right column but wrong metric misses (an l2 index can't serve a cosine query)")
  {
    REQUIRE(cache.find_by_column("mem", "main", "docs", "vec", Metric::CosineExpanded) == nullptr);
  }
}

TEST_CASE("cuvs_index_cache find_by_column folds L2 expanded/unexpanded", "[vss]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  cuvs_index_cache cache(*manager);

  SECTION("index built L2SqrtExpanded matches an L2SqrtUnexpanded query")
  {
    insert_dummy(cache, "idx", make_meta("docs", "vec", Metric::L2SqrtExpanded));
    REQUIRE(cache.find_by_column("mem", "main", "docs", "vec", Metric::L2SqrtUnexpanded) !=
            nullptr);
  }
  SECTION("index built L2SqrtUnexpanded matches an L2SqrtExpanded query")
  {
    insert_dummy(cache, "idx", make_meta("docs", "vec", Metric::L2SqrtUnexpanded));
    REQUIRE(cache.find_by_column("mem", "main", "docs", "vec", Metric::L2SqrtExpanded) != nullptr);
  }
  SECTION("the fold stays within the metric family (an L2 query misses a cosine index)")
  {
    insert_dummy(cache, "idx", make_meta("docs", "vec", Metric::CosineExpanded));
    REQUIRE(cache.find_by_column("mem", "main", "docs", "vec", Metric::L2SqrtUnexpanded) ==
            nullptr);
  }
}

TEST_CASE("cuvs_index_cache insert replaces an existing name in place", "[vss]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  cuvs_index_cache cache(*manager);

  insert_dummy(cache, "idx", make_meta("docs", "vec", Metric::L2SqrtExpanded));
  insert_dummy(cache, "idx", make_meta("docs", "embedding", Metric::CosineExpanded));

  REQUIRE(cache.size() == 1);  // replaced, not appended
  auto entry = cache.find("idx");
  REQUIRE(entry != nullptr);
  REQUIRE(entry->meta.column_name == "embedding");
  REQUIRE(entry->meta.metric == Metric::CosineExpanded);
}

TEST_CASE("cuvs_index_cache erase and clear release entries", "[vss]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  cuvs_index_cache cache(*manager);

  insert_dummy(cache, "a", make_meta("t", "v", Metric::L2SqrtExpanded));
  insert_dummy(cache, "b", make_meta("t", "w", Metric::L2SqrtExpanded));
  REQUIRE(cache.size() == 2);

  SECTION("erase removes exactly the named entry")
  {
    REQUIRE(cache.erase("a"));
    REQUIRE(cache.size() == 1);
    REQUIRE(cache.find("a") == nullptr);
    REQUIRE(cache.find("b") != nullptr);
    REQUIRE_FALSE(cache.erase("a"));  // already gone
  }
  SECTION("clear drops everything")
  {
    cache.clear();
    REQUIRE(cache.size() == 0);
    REQUIRE(cache.find("a") == nullptr);
    REQUIRE(cache.find("b") == nullptr);
  }
}

TEST_CASE("cuvs_index_cache find_by_column returns a match among several entries", "[vss]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  cuvs_index_cache cache(*manager);

  insert_dummy(cache, "idx_a", make_meta("images", "clip", Metric::CosineExpanded));
  insert_dummy(cache, "idx_b", make_meta("docs", "vec", Metric::L2SqrtExpanded));
  insert_dummy(cache, "idx_c", make_meta("docs", "title_vec", Metric::L2SqrtExpanded));

  auto e = cache.find_by_column("mem", "main", "docs", "title_vec", Metric::L2SqrtExpanded);
  REQUIRE(e != nullptr);
  REQUIRE(e->meta.table_name == "docs");
  REQUIRE(e->meta.column_name == "title_vec");
}

TEST_CASE("cuvs_index_cache keeps same-named tables in different schemas distinct", "[vss]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  cuvs_index_cache cache(*manager);

  // s1.docs.vec and s2.docs.vec share table/column/metric and differ only by schema.
  insert_dummy(cache, "s1_idx", make_meta("docs", "vec", Metric::L2SqrtExpanded, "mem", "s1"));
  insert_dummy(cache, "s2_idx", make_meta("docs", "vec", Metric::L2SqrtExpanded, "mem", "s2"));

  REQUIRE(cache.size() == 2);  // distinct identities, not a replace

  auto e1 = cache.find_by_column("mem", "s1", "docs", "vec", Metric::L2SqrtExpanded);
  auto e2 = cache.find_by_column("mem", "s2", "docs", "vec", Metric::L2SqrtExpanded);
  REQUIRE(e1 != nullptr);
  REQUIRE(e2 != nullptr);
  REQUIRE(e1->meta.schema_name == "s1");
  REQUIRE(e2->meta.schema_name == "s2");
  REQUIRE(e1 != e2);  // each schema routes to its own index
}

TEST_CASE("cuvs_index_cache reserve_index_memory is non-blocking", "[vss]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  cuvs_index_cache cache(*manager);

  SECTION("a request that fits returns a reservation")
  {
    auto r = cache.reserve_index_memory(/*bytes=*/1024, /*preferred_gpu=*/0);
    REQUIRE(r != nullptr);
  }
  SECTION("a request larger than the GPU can hold returns null instead of blocking")
  {
    // The test GPU space is 512 MiB; ask for far more. A blocking reserve would
    // hang here, so returning null is what lets CREATE INDEX fail cleanly (the
    // "not enough free GPU memory" error in SiriusCreateAnnIndexFunction).
    auto r = cache.reserve_index_memory(/*bytes=*/8ull << 30, /*preferred_gpu=*/0);
    REQUIRE(r == nullptr);
  }
  SECTION("a negative device id returns null")
  {
    auto r = cache.reserve_index_memory(/*bytes=*/1024, /*preferred_gpu=*/-1);
    REQUIRE(r == nullptr);
  }
}

// Regression for the reservation-accounting bug: a build must be charged to its
// reservation, the reservation released afterward (no reserved-memory leak), and
// the index left as ordinary capacity-accounted memory that returns to baseline
// once the entry is dropped. See the reserve -> attach -> build -> reset -> insert
// sequence in SiriusCreateAnnIndexFunction.
TEST_CASE("cuvs_index_cache create binds the build to its reservation and leaks nothing", "[vss]")
{
  namespace mem = cucascade::memory;
  auto manager  = sirius::test::operator_utils::initialize_memory_manager();
  cuvs_index_cache cache(*manager);

  auto* gpu_space = const_cast<mem::memory_space*>(manager->get_memory_space(mem::Tier::GPU, 0));
  REQUIRE(gpu_space != nullptr);
  auto* adaptor = gpu_space->get_memory_resource_of<mem::Tier::GPU>();
  REQUIRE(adaptor != nullptr);

  // Two well-separated clusters, six rows. The baseline counters are read after the
  // source column is built and the column is kept alive through every assertion, so
  // whatever it costs is already in the baseline and cancels out of the deltas.
  constexpr cudf::size_type dim = 2;
  auto col                      = make_float_list(
    {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 10.0f, 10.0f, 11.0f, 10.0f, 10.0f, 11.0f}, 6, dim);

  auto const reserved_before = adaptor->get_total_reserved_bytes();
  auto const alloc_before    = adaptor->get_total_allocated_bytes();

  // Reserve, bind the reservation to the build stream, build on it, measure the
  // resident footprint, then release: the same sequence the create path runs.
  std::size_t const footprint = 2ull << 20;  // 2 MiB, ample for this tiny index
  auto reservation            = cache.reserve_index_memory(footprint, /*preferred_gpu=*/0);
  REQUIRE(reservation != nullptr);

  rmm::cuda_stream build_stream;
  auto* alloc = reservation->get_memory_resource_of<mem::Tier::GPU>();
  REQUIRE(alloc == adaptor);
  REQUIRE(alloc->attach_reservation_to_tracker(build_stream.view(), std::move(reservation)));

  auto handle = sirius::vss::build_ivf_flat_index_from_batches({col->view()},
                                                               dim,
                                                               /*n_lists=*/2,
                                                               Metric::L2SqrtExpanded,
                                                               gpu_space->get_default_allocator(),
                                                               build_stream.view());

  auto const index_bytes = adaptor->get_allocated_bytes(build_stream.view());
  adaptor->reset_stream_reservation(build_stream.view());

  index_metadata meta = make_meta("docs", "vec", Metric::L2SqrtExpanded);
  meta.resident_bytes = index_bytes;
  cache.insert("idx", std::move(meta), std::move(handle), std::move(build_stream));

  // After create: the reservation is fully released (no reserved leak), and the
  // index's resident bytes are accounted as ordinary capacity.
  REQUIRE(index_bytes > 0);
  REQUIRE(adaptor->get_total_reserved_bytes() == reserved_before);
  REQUIRE(adaptor->get_total_allocated_bytes() - alloc_before == index_bytes);

  // A caller mid-search holds a lookup handle. Erasing unlinks the map entry but
  // the index (and its accounting) stays until that handle is released.
  auto held = cache.find("idx");
  REQUIRE(held != nullptr);
  REQUIRE(cache.erase("idx"));
  REQUIRE(adaptor->get_total_allocated_bytes() - alloc_before == index_bytes);

  // Releasing the last handle frees the index; both counters return to baseline.
  held.reset();
  REQUIRE(adaptor->get_total_reserved_bytes() == reserved_before);
  REQUIRE(adaptor->get_total_allocated_bytes() == alloc_before);
}

// Exercises the list-growth path the current tests miss: one list, two batches of
// 32 then 1 row. IVF-Flat sizes each list in 32-row units, so the 33rd row forces
// the single list to grow past its initial 32-row capacity. resize_list allocates
// the larger buffer and copies while the old one is still owned, which is the
// transient the footprint estimate has to cover. This prints footprint vs the
// measured resident/peak so the estimate can be checked against reality.
TEST_CASE("cuvs_index_cache build peak when a list outgrows its 32-row capacity", "[vss]")
{
  namespace mem = cucascade::memory;
  auto manager  = sirius::test::operator_utils::initialize_memory_manager();
  cuvs_index_cache cache(*manager);

  auto* gpu_space = const_cast<mem::memory_space*>(manager->get_memory_space(mem::Tier::GPU, 0));
  REQUIRE(gpu_space != nullptr);
  auto* adaptor = gpu_space->get_memory_resource_of<mem::Tier::GPU>();
  REQUIRE(adaptor != nullptr);

  constexpr cudf::size_type dim   = 3;
  constexpr std::uint32_t n_lists = 1;

  std::vector<float> a(32 * dim);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(i);
  }
  std::vector<float> const b(1 * dim, 1.0f);
  auto batch_a = make_float_list(a, 32, dim);  // fills the list to its 32-row capacity
  auto batch_b = make_float_list(b, 1, dim);   // the 33rd row forces resize_list to grow it

  std::int64_t const n_rows   = 33;
  std::size_t const footprint = sirius::vss::ivf_flat_reservation_bytes(n_rows, dim, n_lists);

  auto reservation = cache.reserve_index_memory(footprint, /*preferred_gpu=*/0);
  REQUIRE(reservation != nullptr);
  rmm::cuda_stream build_stream;
  auto* alloc = reservation->get_memory_resource_of<mem::Tier::GPU>();
  REQUIRE(alloc->attach_reservation_to_tracker(build_stream.view(), std::move(reservation)));

  auto handle = sirius::vss::build_ivf_flat_index_from_batches({batch_a->view(), batch_b->view()},
                                                               dim,
                                                               n_lists,
                                                               Metric::L2SqrtExpanded,
                                                               gpu_space->get_default_allocator(),
                                                               build_stream.view());

  std::size_t const resident = adaptor->get_allocated_bytes(build_stream.view());
  std::size_t const peak     = adaptor->get_peak_allocated_bytes(build_stream.view());
  adaptor->reset_stream_reservation(build_stream.view());

  REQUIRE(handle != nullptr);
  WARN("footprint=" << footprint << " resident=" << resident << " peak=" << peak);
  CHECK(peak >= resident);
  // The growth path is exercised (a list reallocated past its 32-row capacity) and
  // the reservation covers the build peak.
  CHECK(footprint >= peak);
}

// The low-dim / many-rows-per-list case that a naive 2x-vectors estimate misses:
// one list, 1,000,000 rows at dim 3, fed as ten equal 100k batches. The final extend
// keeps the 900k-row old list buffer while allocating the 1M-row replacement, so the
// list buffers alone peak near (900k + 1M) * (4*dim + 8) = 38,000,000 bytes. The
// row-id and clone-and-replace terms in ivf_flat_reservation_bytes are what keep the
// reservation above that peak here; a formula that only doubled the vector floats
// would fall below it.
TEST_CASE("cuvs_index_cache reservation covers the build peak for a large low-dim list", "[vss]")
{
  namespace mem = cucascade::memory;
  auto manager  = sirius::test::operator_utils::initialize_memory_manager();
  cuvs_index_cache cache(*manager);

  auto* gpu_space = const_cast<mem::memory_space*>(manager->get_memory_space(mem::Tier::GPU, 0));
  REQUIRE(gpu_space != nullptr);
  auto* adaptor = gpu_space->get_memory_resource_of<mem::Tier::GPU>();
  REQUIRE(adaptor != nullptr);

  constexpr cudf::size_type dim   = 3;
  constexpr std::uint32_t n_lists = 1;
  constexpr cudf::size_type per   = 100000;
  constexpr int n_batches         = 10;

  // Build the ten source batches; keep the columns alive for the whole build.
  std::vector<std::unique_ptr<cudf::column>> cols;
  std::vector<cudf::column_view> batches;
  cols.reserve(n_batches);
  batches.reserve(n_batches);
  for (int c = 0; c < n_batches; ++c) {
    std::vector<float> vals(static_cast<std::size_t>(per) * dim);
    for (std::size_t i = 0; i < vals.size(); ++i) {
      vals[i] = static_cast<float>(static_cast<std::size_t>(c) * per + i / dim);
    }
    cols.push_back(make_float_list(vals, per, dim));
    batches.push_back(cols.back()->view());
  }

  std::int64_t const n_rows   = static_cast<std::int64_t>(per) * n_batches;  // 1,000,000
  std::size_t const footprint = sirius::vss::ivf_flat_reservation_bytes(n_rows, dim, n_lists);

  auto reservation = cache.reserve_index_memory(footprint, /*preferred_gpu=*/0);
  REQUIRE(reservation != nullptr);
  rmm::cuda_stream build_stream;
  auto* alloc = reservation->get_memory_resource_of<mem::Tier::GPU>();
  REQUIRE(alloc->attach_reservation_to_tracker(build_stream.view(), std::move(reservation)));

  auto handle = sirius::vss::build_ivf_flat_index_from_batches(batches,
                                                               dim,
                                                               n_lists,
                                                               Metric::L2SqrtExpanded,
                                                               gpu_space->get_default_allocator(),
                                                               build_stream.view());

  std::size_t const resident = adaptor->get_allocated_bytes(build_stream.view());
  std::size_t const peak     = adaptor->get_peak_allocated_bytes(build_stream.view());
  adaptor->reset_stream_reservation(build_stream.view());

  REQUIRE(handle != nullptr);
  WARN("footprint=" << footprint << " resident=" << resident << " peak=" << peak);
  CHECK(footprint >= peak);
}

TEST_CASE("cuvs_index_cache erase_by_column drops the matching index before a rebuild", "[vss]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  cuvs_index_cache cache(*manager);

  SECTION("removes the identity match regardless of its management name, leaving others")
  {
    insert_dummy(cache, "custom_name", make_meta("docs", "vec", Metric::L2SqrtExpanded));
    insert_dummy(cache, "other", make_meta("docs", "title_vec", Metric::L2SqrtExpanded));

    REQUIRE(cache.erase_by_column("mem", "main", "docs", "vec", Metric::L2SqrtExpanded) == 1);
    REQUIRE(cache.find("custom_name") == nullptr);
    REQUIRE(cache.find("other") != nullptr);  // different column untouched
  }
  SECTION("matches up to metric canonicalization (unexpanded query drops an expanded index)")
  {
    insert_dummy(cache, "idx", make_meta("docs", "vec", Metric::L2SqrtExpanded));
    REQUIRE(cache.erase_by_column("mem", "main", "docs", "vec", Metric::L2SqrtUnexpanded) == 1);
    REQUIRE(cache.size() == 0);
  }
  SECTION("scoped to the schema (rebuilding s1's index leaves s2's alone)")
  {
    insert_dummy(cache, "s1_idx", make_meta("docs", "vec", Metric::L2SqrtExpanded, "mem", "s1"));
    insert_dummy(cache, "s2_idx", make_meta("docs", "vec", Metric::L2SqrtExpanded, "mem", "s2"));

    REQUIRE(cache.erase_by_column("mem", "s1", "docs", "vec", Metric::L2SqrtExpanded) == 1);
    REQUIRE(cache.find("s1_idx") == nullptr);
    REQUIRE(cache.find("s2_idx") != nullptr);
  }
  SECTION("no match removes nothing")
  {
    insert_dummy(cache, "idx", make_meta("docs", "vec", Metric::L2SqrtExpanded));
    REQUIRE(cache.erase_by_column("mem", "main", "docs", "vec", Metric::CosineExpanded) == 0);
    REQUIRE(cache.erase_by_column("mem", "main", "docs", "other", Metric::L2SqrtExpanded) == 0);
    REQUIRE(cache.size() == 1);
  }
}

TEST_CASE("cuvs_index_cache lookup handle outlives an erase", "[vss]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  cuvs_index_cache cache(*manager);

  insert_dummy(cache, "idx", make_meta("docs", "vec", Metric::L2SqrtExpanded));

  // A caller mid-search holds the handle. Erasing (a concurrent DROP) unlinks the
  // map entry but must not destroy the entry or its index under the caller.
  auto handle = cache.find("idx");
  REQUIRE(handle != nullptr);

  REQUIRE(cache.erase("idx"));
  REQUIRE(cache.size() == 0);
  REQUIRE(cache.find("idx") == nullptr);

  // The held handle is still valid: entry and index are all alive.
  REQUIRE(handle->meta.column_name == "vec");
  REQUIRE(handle->index != nullptr);
}

TEST_CASE("cuvs_index_cache lookup handle outlives a replace", "[vss]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  cuvs_index_cache cache(*manager);

  insert_dummy(cache, "idx", make_meta("docs", "vec", Metric::L2SqrtExpanded));
  auto old_handle = cache.find("idx");
  REQUIRE(old_handle != nullptr);

  // Replacing the name installs a new entry; the old handle keeps pointing at the
  // old one, which stays alive and unchanged for as long as the handle is held.
  insert_dummy(cache, "idx", make_meta("docs", "embedding", Metric::CosineExpanded));

  REQUIRE(old_handle->meta.column_name == "vec");
  REQUIRE(cache.find("idx")->meta.column_name == "embedding");
}
