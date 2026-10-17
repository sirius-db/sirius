#include "utils/cache_bypass_test_utils.hpp"

namespace {
using namespace sirius::test::cache_bypass;
using sirius::io::datasource_cache_mode;
using sirius::io::open_hint;
}  // namespace

TEST_CASE("C3 datasource factory overloads preserve cache mode", "[io][datasource][cache_mode]")
{
  rest_fixture fixture;
  auto plain       = fixture.context->open_datasource(object_uri);
  auto hinted      = fixture.context->open_datasource(object_uri, open_hint::parquet_footer_probe);
  auto const heads = fixture.server.head_count();
  auto const gets  = fixture.server.requests().size();
  auto sized       = fixture.context->open_datasource(object_uri, std::uint64_t{object_bytes});
  REQUIRE(fixture.server.head_count() == heads);
  REQUIRE(fixture.server.requests().size() == gets);
  for (auto* source : {plain.get(), hinted.get(), sized.get()}) {
    REQUIRE(source != nullptr);
    CHECK(source->cache_mode() == datasource_cache_mode::use_cache);
    CHECK(source->uses_prefetching_cache());
    CHECK(source->size() == object_bytes);
  }
  auto bypass = fixture.context->open_datasource(object_uri, datasource_cache_mode::bypass_cache);
  auto bypass_hint = fixture.context->open_datasource(
    object_uri, open_hint::parquet_footer_probe, datasource_cache_mode::bypass_cache);
  auto const bypass_heads = fixture.server.head_count();
  auto const bypass_gets  = fixture.server.requests().size();
  auto bypass_sized       = fixture.context->open_datasource(
    object_uri, std::uint64_t{object_bytes}, datasource_cache_mode::bypass_cache);
  REQUIRE(fixture.server.head_count() == bypass_heads);
  REQUIRE(fixture.server.requests().size() == bypass_gets);
  for (auto* source : {bypass.get(), bypass_hint.get(), bypass_sized.get()}) {
    REQUIRE(source != nullptr);
    CHECK(source->cache_mode() == datasource_cache_mode::bypass_cache);
    CHECK_FALSE(source->uses_prefetching_cache());
    CHECK(source->size() == object_bytes);
  }
  CHECK(bypass->get_io_object().validation_tag() == object_tag);
  CHECK(bypass_hint->get_io_object().validation_tag() == object_tag);
  CHECK(bypass_sized->get_io_object().validation_tag().empty());
  fixture.server.check();
}

TEST_CASE("C3 datasource duplicates preserve cache mode", "[io][datasource][cache_mode]")
{
  rest_fixture fixture;
  for (auto mode : {datasource_cache_mode::use_cache, datasource_cache_mode::bypass_cache}) {
    auto source      = fixture.context->open_datasource(object_uri, mode);
    auto const heads = fixture.server.head_count();
    auto duplicate   = source->duplicate();
    auto second      = duplicate->duplicate();
    REQUIRE(source.get() != duplicate.get());
    REQUIRE(duplicate.get() != second.get());
    for (auto* current : {source.get(), duplicate.get(), second.get()}) {
      CHECK(current->cache_mode() == mode);
      CHECK(current->uses_prefetching_cache() == (mode == datasource_cache_mode::use_cache));
      CHECK(current->io_ctx() == fixture.context);
      CHECK(&current->get_io_object() == &source->get_io_object());
      CHECK(current->size() == object_bytes);
    }
    CHECK(fixture.server.head_count() == heads);
  }
  fixture.server.check();
}

TEST_CASE("C3 bypass advisory calls leave no cache request", "[io][datasource][cache_mode]")
{
  rest_fixture fixture;
  auto source = fixture.context->open_datasource(object_uri, datasource_cache_mode::bypass_cache);
  auto duplicate     = source->duplicate();
  auto const before  = snapshot(*fixture.context);
  auto const claimed = fixture.context->cache()->claimed_bytes();
  std::array<cudf::io::text::byte_range_info, 1> ranges{
    cudf::io::text::byte_range_info(0, static_cast<std::int64_t>(chunk_bytes))};
  for (auto* current : {source.get(), duplicate.get()}) {
    current->fadvise(ranges, 0);
    current->fadvise(ranges, std::nullopt);
    CHECK(current->prepare_prefetch(false) == sirius::io::prepare_result::nothing_to_prepare);
    unsigned calls     = 0;
    bool success       = true;
    auto const refusal = current->prefetch_async([&](bool ok) noexcept {
      ++calls;
      success = ok;
    });
    CHECK(refusal == sirius::io::prefetch_refusal::no_cache);
    CHECK(calls == 1);
    CHECK_FALSE(success);
    CHECK(current->prefetch_failure() == nullptr);
    CHECK_FALSE(current->uses_prefetching_cache());
  }
  CHECK(snapshot(*fixture.context) == before);
  CHECK(fixture.context->cache()->claimed_bytes() == claimed);
  CHECK(fixture.server.requests().empty());
  fixture.server.check();
}

TEST_CASE("C3 default datasource still prepares cache requests", "[io][datasource][cache_mode]")
{
  rest_fixture fixture;
  auto source = fixture.context->open_datasource(object_uri);
  REQUIRE(source->cache_mode() == datasource_cache_mode::use_cache);
  REQUIRE(source->uses_prefetching_cache());
  std::array<cudf::io::text::byte_range_info, 1> ranges{
    cudf::io::text::byte_range_info(0, static_cast<std::int64_t>(chunk_bytes))};
  auto const before = fixture.context->cache()->claimed_bytes();
  source->fadvise(ranges, 0);
  REQUIRE(source->prepare_prefetch(false) == sirius::io::prepare_result::prepared);
  CHECK(fixture.context->cache()->claimed_bytes() > before);
  REQUIRE(read(*source) == fixture.server.expected());
  auto const filled   = snapshot(*fixture.context);
  auto const requests = fixture.server.requests().size();
  REQUIRE(read(*source) == fixture.server.expected());
  CHECK(snapshot(*fixture.context).hits == filled.hits + 1);
  CHECK(fixture.server.requests().size() == requests);
  fixture.server.check();
}

TEST_CASE("C3 bypass datasource keeps its mode across cache reset", "[io][datasource][cache_mode]")
{
  rest_fixture fixture;
  auto source = fixture.context->open_datasource(object_uri, datasource_cache_mode::bypass_cache);
  REQUIRE(source->cache_mode() == datasource_cache_mode::bypass_cache);
  REQUIRE_FALSE(source->uses_prefetching_cache());
  REQUIRE(read(*source) == fixture.server.expected());
  auto const before = snapshot(*fixture.context);
  CHECK(before.hits == 0);
  CHECK(before.loads == 0);
  CHECK(before.misses == 0);
  fixture.manager.reset_caches();
  CHECK(source->cache_mode() == datasource_cache_mode::bypass_cache);
  CHECK_FALSE(source->uses_prefetching_cache());
  auto duplicate = source->duplicate();
  CHECK(duplicate->cache_mode() == datasource_cache_mode::bypass_cache);
  REQUIRE(read(*duplicate) == fixture.server.expected());
  CHECK(snapshot(*fixture.context) == before);
  CHECK(fixture.server.requests().size() == 2);
  fixture.server.check();
}
