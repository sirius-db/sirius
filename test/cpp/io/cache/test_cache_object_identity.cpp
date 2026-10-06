#include "catch.hpp"
#include "io/cache/prefetching_cache.hpp"
#include "io/rest/rest_ioctx.hpp"
#include "io/rest/s3/sigv4_authorizer.hpp"
#include "io/sirius_datasource.hpp"
#include "io/templated_ioctx.hpp"
#include "op/scan/parquet_metadata.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "utils/s3_container.hpp"
#include "utils/s3_test_env.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <duckdb.hpp>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;
using sirius::io::sirius_datasource;
using sirius::io::rest::rest_io_object;
using sirius::test::s3::env_or;
using sirius::test::s3::require_env;
using sirius::test::s3::sql_quote;
constexpr std::size_t chunk_bytes = 1U << 20;

std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> identity_memory()
{
  sirius::converter_registry::reset_for_testing();
  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(2ULL << 30)
    .set_reservation_fraction_per_gpu(0.75)
    .set_per_numa_region_capacity(256ULL << 20)
    .use_gpu_id_as_host_id()
    .set_reservation_fraction_per_numa_region(1.0);
  auto memory =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
  sirius::converter_registry::initialize();
  return memory;
}

sirius::io::cache::config identity_cache_config()
{
  sirius::io::cache::config cfg;
  cfg.mode                            = sirius::io::cache::cache_mode::sirius;
  cfg.eviction                        = sirius::io::cache::eviction_policy::lru;
  cfg.min_prefetching_budget_fraction = 0.5;
  cfg.eviction_threshold_fraction     = 1.0;
  cfg.apply_mode();
  return cfg;
}

void advise(sirius_datasource& ds, std::size_t size)
{
  std::vector<cudf::io::text::byte_range_info> ranges;
  ranges.emplace_back(0, static_cast<std::int64_t>(size));
  ds.fadvise(ranges, 0);
  REQUIRE(ds.prepare_prefetch(false) == sirius::io::prepare_result::prepared);
}

std::vector<std::uint8_t> read_all(sirius_datasource& ds)
{
  std::vector<std::uint8_t> bytes(ds.size());
  REQUIRE(ds.host_read(0, bytes.size(), bytes.data()) == bytes.size());
  return bytes;
}

struct parquet_objects {
  parquet_objects()
  {
    REQUIRE(sirius::test::ensure_s3_container_env());
    std::string pattern = (std::filesystem::temp_directory_path() / "sirius-c2-XXXXXX").string();
    auto* dir           = ::mkdtemp(pattern.data());
    REQUIRE(dir != nullptr);
    directory = dir;
    key       = "cache-identity/" + directory.filename().string() + "/object.parquet";
    uri       = "s3://" + require_env("SIRIUS_TEST_S3_BUCKET") + "/" + key;
    first     = generate(1, "column_a");
    second    = generate(2, "column_b");
    REQUIRE(first.size() == second.size());
    REQUIRE(first != second);
    REQUIRE(first.size() < chunk_bytes);
    REQUIRE(first.size() * 16 <= (128U << 20));
  }

  ~parquet_objects()
  {
    std::error_code error;
    std::filesystem::remove_all(directory, error);
  }

  std::vector<std::uint8_t> generate(int value, std::string const& name)
  {
    auto const path = directory / ("generation-" + std::to_string(value) + ".parquet");
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    auto result = con.Query("COPY (SELECT " + std::to_string(value) + "::INTEGER AS " + name +
                            " FROM range(4096)) TO " + sql_quote(path.string()) +
                            " (FORMAT PARQUET, COMPRESSION UNCOMPRESSED)");
    REQUIRE(result != nullptr);
    INFO((result->HasError() ? result->GetError() : ""));
    REQUIRE_FALSE(result->HasError());
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.good());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  }

  void publish(std::span<std::uint8_t const> bytes)
  {
    REQUIRE(sirius::test::put_s3_container_object(key, bytes));
  }

  std::filesystem::path directory;
  std::string key;
  std::string uri;
  std::vector<std::uint8_t> first;
  std::vector<std::uint8_t> second;
};

class counting_authorizer final : public sirius::io::rest::request_authorizer {
 public:
  counting_authorizer()
    : delegate(sirius::io::rest::s3::static_credentials{require_env("SIRIUS_TEST_S3_ACCESS_KEY"),
                                                        require_env("SIRIUS_TEST_S3_SECRET_KEY"),
                                                        env_or("SIRIUS_TEST_S3_SESSION_TOKEN"),
                                                        std::nullopt},
               env_or("SIRIUS_TEST_S3_REGION", "us-east-1"),
               require_env("SIRIUS_TEST_S3_ENDPOINT"))
  {
  }

  sirius::io::rest::authorized_request authorize(sirius::io::rest::object_ref const& object,
                                                 sirius::io::rest::request_method method,
                                                 std::chrono::seconds timeout) override
  {
    if (method == sirius::io::rest::request_method::GET) { gets.fetch_add(1); }
    return delegate.authorize(object, method, timeout);
  }

  std::atomic<std::size_t> gets{0};

 private:
  sirius::io::rest::s3::sigv4_presigned_authorizer delegate;
};

struct identity_fixture {
  identity_fixture()
    : memory(identity_memory()), authorizer(std::make_shared<counting_authorizer>())
  {
    auto* host = sirius::scan_test_utils::get_space(*memory, cucascade::memory::Tier::HOST);
    REQUIRE(host != nullptr);
    sirius::io::rest::config cfg;
    cfg.max_connections         = 1;
    cfg.max_retry_attempts      = 1;
    cfg.max_auth_retry_attempts = 1;
    cfg.request_timeout_s       = 5;
    auto reactor_context        = std::make_shared<sirius::io::rest::rest_reactor::reactor_context>(
      cfg, authorizer, host->get_memory_resource_of<cucascade::memory::Tier::HOST>());
    context = std::make_shared<sirius::io::rest::rest_ioctx>(1, std::move(reactor_context));
    context->start();
    context->initialize_cache(
      *memory, identity_cache_config(), sirius::test::s3::single_gpu_index(0));
    REQUIRE(context->cache() != nullptr);
    REQUIRE(context->cache()->chunk_size() == chunk_bytes);
  }

  ~identity_fixture()
  {
    if (context) {
      context->shutdown_cache();
      context->shutdown();
    }
  }

  std::unique_ptr<sirius_datasource> open_resident(std::string const& uri,
                                                   std::vector<std::uint8_t> const& expected)
  {
    auto ds = context->open_datasource(uri);
    REQUIRE(ds->size() == expected.size());
    REQUIRE_FALSE(ds->get_io_object().validation_tag().empty());
    advise(*ds, expected.size());
    auto const before = authorizer->gets.load();
    CHECK(read_all(*ds) == expected);
    CHECK(authorizer->gets.load() == before + 1);
    CHECK(read_all(*ds) == expected);
    CHECK(authorizer->gets.load() == before + 1);
    return ds;
  }

  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory;
  std::shared_ptr<counting_authorizer> authorizer;
  std::shared_ptr<sirius::io::rest::rest_ioctx> context;
};

void require_generations(sirius::io::cache::prefetching_cache const& cache,
                         std::string const& path,
                         std::size_t alive,
                         std::size_t retired)
{
  CHECK(cache.generation_count(path) == alive);
  CHECK(cache.retired_generation_count() == retired);
}

void metadata_overwrite(sirius::io::cache::cache_mode mode)
{
  parquet_objects objects;
  objects.publish(objects.first);
  auto memory = identity_memory();
  sirius::scan_manager::scan_manager_config cfg;
  cfg.backend                    = sirius::scan_manager::io_backend::sirius;
  cfg.object_store.endpoint      = require_env("SIRIUS_TEST_S3_ENDPOINT");
  cfg.object_store.region        = env_or("SIRIUS_TEST_S3_REGION", "us-east-1");
  cfg.object_store.access_key    = require_env("SIRIUS_TEST_S3_ACCESS_KEY");
  cfg.object_store.secret_key    = require_env("SIRIUS_TEST_S3_SECRET_KEY");
  cfg.object_store.session_token = env_or("SIRIUS_TEST_S3_SESSION_TOKEN");
  cfg.cache                      = identity_cache_config();
  cfg.cache.mode                 = mode;
  cfg.uring_n_reactors           = 1;
  cfg.rest_n_reactors            = 1;
  cfg.apply_cache_mode();
  sirius::scan_manager::sirius_scan_manager manager{
    cfg, *memory, sirius::test::s3::single_gpu_index(0)};
  auto first_shape = manager.describe_parquet(objects.uri);
  REQUIRE(first_shape.names.size() == 1);
  CHECK(first_shape.names.front() == "column_a");
  auto first        = manager.create_datasource(objects.uri);
  auto old_metadata = first->metadata();
  REQUIRE(old_metadata != nullptr);
  auto const old_key = first->get_io_object().raw_file_cache_id();
  auto const old_tag = std::string(first->get_io_object().validation_tag());
  REQUIRE_FALSE(old_tag.empty());
  CHECK(old_key == rest_io_object::generation_key(objects.uri, old_tag));
  auto& store = first->io_ctx()->metadata_store();
  CHECK(store.has_path(objects.uri));
  objects.publish(objects.second);
  auto second_shape = manager.describe_parquet(objects.uri);
  REQUIRE(second_shape.names.size() == 1);
  CHECK(second_shape.names.front() == "column_b");
  auto second = manager.create_datasource(objects.uri);
  REQUIRE(second->get_io_object().validation_tag() != old_tag);
  CHECK(second->get_io_object().raw_file_cache_id() != old_key);
  CHECK(store.has_path(objects.uri));
  CHECK(store.get_metadata(old_key) == nullptr);
  CHECK(second->metadata() != nullptr);
  auto parsed = std::dynamic_pointer_cast<sirius::op::scan::parquet_metadata>(old_metadata);
  REQUIRE(parsed != nullptr);
  REQUIRE(parsed->file_metadata() != nullptr);
  REQUIRE(parsed->file_metadata()->schema.size() == 2);
  CHECK(parsed->file_metadata()->schema[1].name == "column_a");
}

struct pinned_copy_gate {
  pinned_copy_gate(rmm::cuda_stream& producer, rmm::cuda_stream& consumer)
    : producer(producer), consumer(consumer)
  {
  }
  ~pinned_copy_gate()
  {
    release.store(true);
    std::ignore = cudaStreamSynchronize(producer.value());
    std::ignore = cudaStreamSynchronize(consumer.value());
    if (event != nullptr) { std::ignore = cudaEventDestroy(event); }
  }
  void arm()
  {
    REQUIRE(cudaEventCreateWithFlags(&event, cudaEventDisableTiming) == cudaSuccess);
    REQUIRE(cudaLaunchHostFunc(
              producer.value(),
              [](void* opaque) {
                auto& gate = *static_cast<pinned_copy_gate*>(opaque);
                gate.entered.store(true);
                while (!gate.release.load()) {
                  std::this_thread::yield();
                }
              },
              this) == cudaSuccess);
    REQUIRE(cudaEventRecord(event, producer.value()) == cudaSuccess);
    REQUIRE(cudaStreamWaitEvent(consumer.value(), event, 0) == cudaSuccess);
    auto const deadline = std::chrono::steady_clock::now() + 5s;
    while (!entered.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    REQUIRE(entered.load());
  }

  rmm::cuda_stream& producer;
  rmm::cuda_stream& consumer;
  cudaEvent_t event{nullptr};
  std::atomic<bool> entered{false};
  std::atomic<bool> release{false};
};

struct held_backend_config {
  std::size_t min_alignment_requirement() const noexcept { return 1; }
  std::size_t merge_gap_size() const noexcept { return 0; }
  std::size_t n_max_concurrent_scans{1};
};

class held_backend {
 public:
  using io_object_type                  = rest_io_object;
  using reactor_config_type             = held_backend_config;
  static constexpr bool prefers_bulk_io = false;

  held_backend_config const& get_config() const noexcept { return config; }
  std::size_t staging_block_size() const noexcept { return chunk_bytes; }
  std::size_t queued_bytes() const noexcept { return 0; }
  void enqueue(std::unique_ptr<sirius::io::grouped_io_request> request) noexcept
  {
    requests.push_back(std::move(request));
  }
  std::unique_ptr<sirius::io::grouped_io_request> take()
  {
    if (requests.empty()) { return {}; }
    auto result = std::move(requests.front());
    requests.pop_front();
    return result;
  }
  std::size_t host_read(rest_io_object const&, std::size_t, std::size_t, std::uint8_t*)
  {
    throw std::logic_error("held backend reads must use the request queue");
  }
  void start() {}
  void interrupt() {}
  void shutdown()
  {
    while (auto request = take()) {
      request->cancel_remaining(std::make_exception_ptr(std::runtime_error("test shutdown")));
    }
  }
  static std::unique_ptr<rest_io_object> create_io_object(std::string path)
  {
    return std::make_unique<rest_io_object>(
      std::move(path), "bucket", "key", chunk_bytes, "\"one\"");
  }
  static bool supports(std::string_view) { return true; }
  static std::vector<cudf::io::text::byte_range_info> align_and_coalesce(
    std::span<cudf::io::text::byte_range_info const> ranges, std::optional<std::size_t>)
  {
    return {ranges.begin(), ranges.end()};
  }

 private:
  held_backend_config config;
  std::deque<std::unique_ptr<sirius::io::grouped_io_request>> requests;
};

class held_context final : public sirius::io::templated_ioctx<held_backend> {
 public:
  held_context() : templated_ioctx(1, [] { return std::make_unique<held_backend>(); }) {}
  sirius::io::io_context_type type() const noexcept override
  {
    return sirius::io::io_context_type::restful;
  }
  held_backend& backend() { return *_reactors.front(); }
};

struct held_request {
  ~held_request()
  {
    if (request) {
      request->cancel_remaining(std::make_exception_ptr(std::runtime_error("test cleanup")));
    }
  }

  void complete(std::uint8_t value)
  {
    REQUIRE(request != nullptr);
    while (!request->empty()) {
      auto slice = request->take_front();
      if (slice.h_buffer.is_contiguous()) {
        std::fill_n(std::get<std::uint8_t*>(slice.h_buffer.buffer), slice.size(), value);
      } else {
        for (auto* chunk : slice.h_buffer.fragments()) {
          std::fill_n(chunk->data + slice.offset() - chunk->offset, slice.size(), value);
        }
      }
      if (slice.on_complete) { (*slice.on_complete)(slice.h_buffer.fragments(), true); }
      request->coordinator->on_complete();
    }
    request.reset();
  }

  std::unique_ptr<sirius::io::grouped_io_request> request;
};

}  // namespace

TEST_CASE("cache identity isolates equal-size overwrites", "[s3][integration][cache_identity]")
{
  parquet_objects objects;
  objects.publish(objects.first);
  identity_fixture fixture;
  auto first     = fixture.open_resident(objects.uri, objects.first);
  auto const tag = std::string(first->get_io_object().validation_tag());
  auto const key = first->get_io_object().raw_file_cache_id();
  CHECK(key == rest_io_object::generation_key(objects.uri, tag));
  objects.publish(objects.second);
  auto second = fixture.open_resident(objects.uri, objects.second);
  REQUIRE(second->get_io_object().validation_tag() != tag);
  CHECK(second->get_io_object().raw_file_cache_id() != key);
  CHECK(fixture.authorizer->gets.load() == 2);
}

TEST_CASE("cache identity replaces metadata with the byte cache enabled",
          "[s3][integration][cache_identity]")
{
  metadata_overwrite(sirius::io::cache::cache_mode::sirius);
}

TEST_CASE("cache identity replaces metadata with the byte cache disabled",
          "[s3][integration][cache_identity]")
{
  metadata_overwrite(sirius::io::cache::cache_mode::none);
}

TEST_CASE("cache identity reclaims a retired generation on final handle release",
          "[s3][integration][cache_identity]")
{
  parquet_objects objects;
  objects.publish(objects.first);
  identity_fixture fixture;
  auto first           = fixture.open_resident(objects.uri, objects.first);
  auto const first_tag = std::string(first->get_io_object().validation_tag());
  objects.publish(objects.second);
  auto second = fixture.open_resident(objects.uri, objects.second);
  REQUIRE(second->get_io_object().validation_tag() != first_tag);
  auto& cache = *fixture.context->cache();
  require_generations(cache, objects.uri, 2, 1);
  CHECK(read_all(*first) == objects.first);
  CHECK(fixture.authorizer->gets.load() == 2);
  first.reset();
  require_generations(cache, objects.uri, 1, 0);
  CHECK(cache.claimed_bytes() == chunk_bytes);
}

TEST_CASE("cache identity retains a disposed datasource generation until destruction",
          "[s3][integration][cache_identity]")
{
  parquet_objects objects;
  objects.publish(objects.first);
  identity_fixture fixture;
  auto first           = fixture.open_resident(objects.uri, objects.first);
  auto const first_tag = std::string(first->get_io_object().validation_tag());
  first->update(sirius::io::cache::scan_stage::disposed);
  objects.publish(objects.second);
  auto second = fixture.open_resident(objects.uri, objects.second);
  REQUIRE(second->get_io_object().validation_tag() != first_tag);
  auto& cache = *fixture.context->cache();
  require_generations(cache, objects.uri, 2, 1);
  first.reset();
  require_generations(cache, objects.uri, 1, 0);
  CHECK(cache.claimed_bytes() == chunk_bytes);
}

TEST_CASE("cache identity bounds retired generations across five overwrites",
          "[s3][integration][cache_identity]")
{
  parquet_objects objects;
  objects.publish(objects.first);
  identity_fixture fixture;
  auto current = fixture.open_resident(objects.uri, objects.first);
  auto& cache  = *fixture.context->cache();
  for (int generation = 2; generation <= 6; ++generation) {
    auto payload = objects.generate(generation, "column_a");
    REQUIRE(payload.size() == objects.first.size());
    auto const previous_tag = std::string(current->get_io_object().validation_tag());
    objects.publish(payload);
    auto next = fixture.open_resident(objects.uri, payload);
    REQUIRE(next->get_io_object().validation_tag() != previous_tag);
    require_generations(cache, objects.uri, 2, 1);
    current.reset();
    require_generations(cache, objects.uri, 1, 0);
    CHECK(cache.claimed_bytes() == chunk_bytes);
    current = std::move(next);
  }
  CHECK(fixture.authorizer->gets.load() == 6);
}

TEST_CASE("cache identity pins retired bytes until a gated device copy completes",
          "[s3][integration][cache_identity]")
{
  parquet_objects objects;
  objects.publish(objects.first);
  identity_fixture fixture;
  auto first           = fixture.open_resident(objects.uri, objects.first);
  auto const first_tag = std::string(first->get_io_object().validation_tag());
  rmm::cuda_stream producer;
  rmm::cuda_stream consumer;
  rmm::device_buffer destination(objects.first.size(), consumer);
  pinned_copy_gate gate(producer, consumer);
  gate.arm();
  auto copying = first->device_read_async(
    0, objects.first.size(), static_cast<std::uint8_t*>(destination.data()), consumer);
  CHECK(copying.wait_for(0ms) == std::future_status::timeout);
  CHECK(fixture.authorizer->gets.load() == 1);
  objects.publish(objects.second);
  auto second = fixture.open_resident(objects.uri, objects.second);
  REQUIRE(second->get_io_object().validation_tag() != first_tag);
  auto& cache = *fixture.context->cache();
  require_generations(cache, objects.uri, 2, 1);
  first.reset();
  require_generations(cache, objects.uri, 2, 1);
  CHECK(cache.claimed_bytes() == 2 * chunk_bytes);
  gate.release.store(true);
  REQUIRE(copying.wait_for(5s) == std::future_status::ready);
  CHECK(copying.get() == objects.first.size());
  require_generations(cache, objects.uri, 1, 0);
  CHECK(cache.claimed_bytes() == chunk_bytes);
  std::vector<std::uint8_t> bytes(objects.first.size());
  REQUIRE(
    cudaMemcpyAsync(
      bytes.data(), destination.data(), bytes.size(), cudaMemcpyDeviceToHost, consumer.value()) ==
    cudaSuccess);
  consumer.synchronize();
  CHECK(bytes == objects.first);
  CHECK(fixture.authorizer->gets.load() == 2);
}

TEST_CASE("cache identity retains a retired generation until a held fill releases",
          "[cache][cache_identity]")
{
  auto memory  = identity_memory();
  auto context = std::make_shared<held_context>();
  context->initialize_cache(
    *memory, identity_cache_config(), sirius::test::s3::single_gpu_index(0));
  REQUIRE(context->cache() != nullptr);
  std::string const path = "s3://controlled/generation.bin";
  auto first             = std::make_unique<sirius_datasource>(
    context,
    std::make_shared<rest_io_object>(path, "controlled", "generation.bin", chunk_bytes, "\"one\""));
  auto second = std::make_unique<sirius_datasource>(
    context,
    std::make_shared<rest_io_object>(path, "controlled", "generation.bin", chunk_bytes, "\"two\""));
  std::atomic<int> first_completions{0};
  std::atomic<int> second_completions{0};
  std::atomic<bool> second_ok{false};
  held_request old_fill;
  held_request new_fill;
  advise(*first, chunk_bytes);
  REQUIRE(first->prefetch_async([&](bool) noexcept { first_completions.fetch_add(1); }) ==
          sirius::io::prefetch_refusal::issued);
  old_fill.request = context->backend().take();
  REQUIRE(old_fill.request != nullptr);
  advise(*second, chunk_bytes);
  REQUIRE(second->prefetch_async([&](bool ok) noexcept {
    second_ok.store(ok);
    second_completions.fetch_add(1);
  }) == sirius::io::prefetch_refusal::issued);
  new_fill.request = context->backend().take();
  REQUIRE(new_fill.request != nullptr);
  new_fill.complete(0x22);
  CHECK(second_ok.load());
  CHECK(second_completions.load() == 1);
  auto& cache = *context->cache();
  require_generations(cache, path, 2, 1);
  first.reset();
  require_generations(cache, path, 2, 1);
  old_fill.complete(0x11);
  CHECK(first_completions.load() == 1);
  require_generations(cache, path, 1, 0);
  CHECK(cache.claimed_bytes() == chunk_bytes);
  auto bytes = read_all(*second);
  CHECK(std::ranges::all_of(bytes, [](auto value) { return value == 0x22; }));
  second.reset();
  context->shutdown_cache();
}

TEST_CASE("cache identity releases read pins when a read of a retired generation fails",
          "[s3][integration][cache_identity]")
{
  parquet_objects objects;
  objects.publish(objects.first);
  identity_fixture fixture;
  auto first           = fixture.open_resident(objects.uri, objects.first);
  auto const first_tag = std::string(first->get_io_object().validation_tag());
  objects.publish(objects.second);
  auto second = fixture.open_resident(objects.uri, objects.second);
  REQUIRE(second->get_io_object().validation_tag() != first_tag);
  auto& cache = *fixture.context->cache();
  require_generations(cache, objects.uri, 2, 1);
  REQUIRE(cache.claimed_bytes() == 2 * chunk_bytes);

  std::vector<std::uint8_t> head(64);
  std::array<sirius::io::slice, 2> slices{};
  slices[0]     = sirius::io::slice{0, head.size(), head.data()};
  slices[1].rng = sirius::io::range{head.size(), head.size()};
  auto failed   = first->host_read_ranges_async(slices);
  CHECK_THROWS_AS(failed.get(), std::invalid_argument);
  CHECK(fixture.authorizer->gets.load() == 2);

  CHECK(read_all(*first) == objects.first);
  CHECK(fixture.authorizer->gets.load() == 2);
  first.reset();
  require_generations(cache, objects.uri, 1, 0);
  CHECK(cache.claimed_bytes() == chunk_bytes);
}

TEST_CASE("cache identity releases a retired footer stash with its last datasource",
          "[s3][integration][cache_identity]")
{
  parquet_objects objects;
  objects.publish(objects.first);
  identity_fixture fixture;
  auto& cache = *fixture.context->cache();

  auto first =
    fixture.context->open_datasource(objects.uri, sirius::io::open_hint::parquet_footer_probe);
  REQUIRE(first != nullptr);
  REQUIRE(first->size() == objects.first.size());
  auto const& first_object = dynamic_cast<rest_io_object const&>(first->get_io_object());
  REQUIRE(rest_io_object::is_strong_tag(first_object.validation_tag()));
  REQUIRE(first_object.stash() != nullptr);
  std::weak_ptr<sirius::io::io_object const> object_alive        = first_object.shared_from_this();
  std::weak_ptr<std::span<std::uint8_t const> const> stash_alive = first_object.stash();
  auto const first_tag = std::string(first_object.validation_tag());

  advise(*first, objects.first.size());
  CHECK(read_all(*first) == objects.first);
  auto const deadline = std::chrono::steady_clock::now() + 5s;
  while (cache.eviction_batch_size_for_testing() == 0 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  REQUIRE(cache.eviction_batch_size_for_testing() > 0);

  objects.publish(objects.second);
  auto second = fixture.open_resident(objects.uri, objects.second);
  REQUIRE(second->get_io_object().validation_tag() != first_tag);
  require_generations(cache, objects.uri, 2, 1);
  REQUIRE_FALSE(object_alive.expired());
  REQUIRE_FALSE(stash_alive.expired());

  first.reset();
  CHECK(object_alive.expired());
  CHECK(stash_alive.expired());
  require_generations(cache, objects.uri, 1, 0);
  CHECK(cache.claimed_bytes() == chunk_bytes);
}
