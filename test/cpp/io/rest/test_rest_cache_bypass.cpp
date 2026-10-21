#include "utils/cache_bypass_test_utils.hpp"
#include "utils/isolated_checkpoint_test.hpp"

namespace {
using namespace sirius::test::cache_bypass;
using sirius::io::datasource_cache_mode;
using sirius::io::open_hint;

std::shared_ptr<sirius::io::sirius_datasource> open_bypass(rest_fixture& fixture,
                                                           open_hint hint = open_hint::generic)
{
  auto source = fixture.manager.open_datasource_on(
    fixture.context, object_uri, hint, datasource_cache_mode::bypass_cache);
  REQUIRE(source != nullptr);
  REQUIRE(source->cache_mode() == datasource_cache_mode::bypass_cache);
  REQUIRE_FALSE(source->uses_prefetching_cache());
  return source;
}

void require_zero_cache_reads(rest_fixture& fixture)
{
  auto const state = snapshot(*fixture.context);
  CHECK(state.hits == 0);
  CHECK(state.loads == 0);
  CHECK(state.misses == 0);
}

void seed_reset_counter(rest_fixture& fixture)
{
  auto source = populate(fixture);
  REQUIRE(snapshot(*fixture.context).hits > 0);
  source.reset();
}
}  // namespace

TEST_CASE("C3 bypass reads skip resident cache ranges", "[rest][c3_bypass]")
{
  rest_fixture fixture;
  auto resident = populate(fixture);
  auto source   = open_bypass(fixture);
  REQUIRE(source->get_io_object().raw_file_cache_id() ==
          resident->get_io_object().raw_file_cache_id());
  auto const before   = snapshot(*fixture.context);
  auto const requests = fixture.server.requests().size();
  REQUIRE(read(*source) == fixture.server.expected());
  auto const after = snapshot(*fixture.context);
  CHECK(after == before);
  auto records = fixture.server.requests();
  REQUIRE(records.size() == requests + 1);
  CHECK(records.back().path == "/c3-fixture/object.bin");
  CHECK(records.back().ranges == std::vector<std::string>{first_chunk_range});
  CHECK(records.back().if_match == std::vector<std::string>{object_tag});
  CHECK_FALSE(records.back().suffix);
  REQUIRE(read(*resident) == fixture.server.expected());
  auto const control = snapshot(*fixture.context);
  CHECK(control.hits == before.hits + 1);
  CHECK(control.loads == before.loads);
  CHECK(control.misses == before.misses);
  CHECK(fixture.server.requests().size() == requests + 1);
  fixture.server.check();
}

TEST_CASE("C3 bypass reads survive reset between reads", "[rest][c3_bypass][reset]")
{
  if (sirius::test::run_isolated()) { return; }
  rest_fixture fixture;
  seed_reset_counter(fixture);
  auto source       = open_bypass(fixture);
  auto const seeded = snapshot(*fixture.context);
  REQUIRE(seeded.hits > 0);
  REQUIRE(read(*source) == fixture.server.expected());
  REQUIRE(snapshot(*fixture.context) == seeded);
  fixture.manager.reset_caches();
  REQUIRE(fixture.manager.ioctx_for_path(object_uri).get() == fixture.context.get());
  require_zero_cache_reads(fixture);
  REQUIRE(read(*source) == fixture.server.expected());
  require_zero_cache_reads(fixture);
  fixture.server.check();
}

TEST_CASE("C3 bypass read completes after concurrent cache reset", "[rest][c3_bypass][reset]")
{
  if (sirius::test::run_isolated()) { return; }
  rest_fixture fixture;
  seed_reset_counter(fixture);
  auto source = open_bypass(fixture);
  REQUIRE(snapshot(*fixture.context).hits > 0);
  auto bytes = reset_during_read(fixture, [&] { return read(*source); });
  REQUIRE(bytes == fixture.server.expected());
  require_zero_cache_reads(fixture);
  REQUIRE(read(*source) == fixture.server.expected());
  require_zero_cache_reads(fixture);
  fixture.server.check();
}

TEST_CASE("C3 bypass GET preserves conditional identity failures", "[rest][c3_bypass][identity]")
{
  for (bool precondition_failed : {true, false}) {
    DYNAMIC_SECTION("precondition_failed=" << precondition_failed)
    {
      rest_fixture fixture;
      auto source = open_bypass(fixture);
      REQUIRE(source->get_io_object().validation_tag() == object_tag);
      fixture.server.data_response(precondition_failed ? 412 : 206, "\"c3-replacement\"");
      REQUIRE_THROWS_AS(read(*source), sirius::io::object_changed_error);
      auto const records = fixture.server.requests();
      REQUIRE(records.size() == 1);
      CHECK(records.front().ranges == std::vector<std::string>{first_chunk_range});
      CHECK(records.front().if_match == std::vector<std::string>{object_tag});
      require_zero_cache_reads(fixture);
      fixture.server.check();
    }
  }
}

TEST_CASE("C3 bypass unqualified opens use unconditional GET", "[rest][c3_bypass][identity]")
{
  for (std::string const tag : {"", "W/\"c3-weak\"", "c3-unquoted"}) {
    DYNAMIC_SECTION("tag=" << tag)
    {
      rest_fixture fixture(tag);
      auto source = open_bypass(fixture);
      CHECK_FALSE(
        sirius::io::rest::rest_io_object::is_strong_tag(source->get_io_object().validation_tag()));
      REQUIRE(read(*source) == fixture.server.expected());
      auto const records = fixture.server.requests();
      REQUIRE(records.size() == 1);
      CHECK(records.front().if_match.empty());
      CHECK(records.front().ranges == std::vector<std::string>{first_chunk_range});
      require_zero_cache_reads(fixture);
      fixture.server.check();
    }
  }
}

TEST_CASE("C3 bypass footer stash survives cache reset", "[rest][c3_bypass][identity]")
{
  rest_fixture fixture;
  auto source = open_bypass(fixture, open_hint::parquet_footer_probe);
  auto const& object =
    dynamic_cast<sirius::io::rest::rest_io_object const&>(source->get_io_object());
  REQUIRE(object.stash() != nullptr);
  REQUIRE(object.validation_tag() == object_tag);
  REQUIRE(object.stash_window_lo() > chunk_bytes);
  auto const requests = fixture.server.requests();
  REQUIRE(requests.size() == 1);
  REQUIRE(requests.front().suffix);
  auto const offset = object_bytes - 4096;
  fixture.server.data_response(412, "\"c3-replacement\"");
  REQUIRE(read(*source, offset, 4096) == fixture.server.expected(offset, 4096));
  CHECK(fixture.server.requests().size() == requests.size());
  fixture.manager.reset_caches();
  REQUIRE(read(*source, offset, 4096) == fixture.server.expected(offset, 4096));
  CHECK(fixture.server.requests().size() == requests.size());
  require_zero_cache_reads(fixture);
  fixture.server.check();
}
