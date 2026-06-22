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

// Tests for duckdb_native_gpu_ingestible: ctor validation, range claiming, and
// consumer-side coalescing of ranges into cap-sized batches (including the
// single-split tail case).

#include <cudf/utilities/span.hpp>

#include <catch.hpp>
#include <curl/curl.h>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/single_file_block_manager.hpp>
#include <duckdb/storage/storage_manager.hpp>
#include <helper/logical_type.hpp>
#include <io/io_context.hpp>
#include <io/s3/s3_blocking_ioctx.hpp>
#include <io/s3/s3_ioctx.hpp>
#include <io/s3/sigv4.hpp>
#include <io/s3/sirius_sigv4_authorizer.hpp>
#include <io/types.hpp>
#include <op/scan/duckdb_block_layout.hpp>
#include <op/scan/duckdb_native_gpu_ingestible.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <scan_manager/split_connector.hpp>
#include <scan_manager/split_provider.hpp>
#include <utils/utils.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace sirius;
using namespace sirius::op::scan;
using namespace sirius::scan_manager;

namespace {

namespace fs = std::filesystem;
using namespace std::chrono_literals;

void exec_ok(duckdb::Connection& con, std::string const& q)
{
  auto result = con.Query(q);
  REQUIRE(result);
  if (result->HasError()) {
    INFO("query failed: " << q << "\n  error: " << result->GetError());
    REQUIRE_FALSE(result->HasError());
  }
}

duckdb::DataTable& get_storage(duckdb::Connection& con, std::string const& table_name)
{
  exec_ok(con, "BEGIN TRANSACTION");
  auto& ctx     = *con.context;
  auto& catalog = duckdb::Catalog::GetCatalog(ctx, "");
  duckdb::CatalogTransaction txn(catalog, ctx);
  auto& schema = catalog.GetSchema(txn, "main");
  auto entry   = schema.GetEntry(txn, duckdb::CatalogType::TABLE_ENTRY, table_name);
  REQUIRE(entry);
  return entry->Cast<duckdb::DuckTableEntry>().GetStorage();
}

projected_column real_col(duckdb::idx_t col_id)
{
  projected_column pc;
  pc.storage_idx = duckdb::StorageIndex(col_id);
  pc.is_rowid    = false;
  return pc;
}

// Minimal duckdb-native bind data. Caller fills storage / context / projected
// vectors / batch size / db_path. column_ids / projection_ids / output_types /
// table_filters stay empty: no filter and no projection-down.
std::unique_ptr<io::ingestible_table_info> make_table_info(
  duckdb::DataTable* storage,
  duckdb::ClientContext* ctx,
  std::vector<projected_column> cols,
  std::vector<sirius::logical_type> types,
  std::size_t approximate_batch_size = sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE,
  std::string db_path                = "")
{
  auto info                    = std::make_unique<duckdb_native_ingestible_table_info>();
  info->storage                = storage;
  info->context                = ctx;
  info->db_path                = std::move(db_path);
  info->projected_cols         = std::move(cols);
  info->projected_types        = std::move(types);
  info->approximate_batch_size = approximate_batch_size;
  return info;
}

// A scan_manager with the default config (it builds a real local-file uring
// backend). With an empty db_path the ctor resolves no datasource, so the
// emitted payloads carry null io handles. The decoder, the only io_ctx
// consumer, is not driven here.
std::unique_ptr<sirius_scan_manager> make_scan_manager()
{
  return std::make_unique<sirius_scan_manager>(scan_manager_config{});
}

// Scheduler that runs each enqueued thunk synchronously, so split_provider::run()
// produces every range and closes the connector before returning.
struct inline_scheduler {
  template <typename F>
  void enqueue(F&& f)
  {
    f();
  }
};

// Drive the producer directly and flatten the raw row-group ranges it emits.
// Reports the number of ranges handed out.
struct drained {
  std::vector<duckdb_row_group_metadata> row_groups;
  std::size_t range_count = 0;
};

drained drain_ranges(duckdb_native_gpu_ingestible& ingestible)
{
  drained out;
  while (ingestible.has_more_splits()) {
    auto factory = ingestible.next_split_provider();
    if (!factory) break;
    auto carriers = factory();
    if (carriers.empty()) break;
    ++out.range_count;
    for (auto& c : carriers) {
      auto* range = dynamic_cast<duckdb_native_range_input*>(c.get());
      REQUIRE(range != nullptr);
      for (auto& rg : range->row_groups) {
        out.row_groups.push_back(std::move(rg));
      }
    }
  }
  return out;
}

// Producer->connector->consumer round-trip: run the producer through a
// split_connector via the inline scheduler, then drain the consumer's
// coalescer. Returns each coalesced batch.
std::vector<std::unique_ptr<op::operator_data>> run_and_consume(
  duckdb_native_gpu_ingestible& ingestible)
{
  split_connector connector;
  split_provider provider{ingestible};
  inline_scheduler sched;
  provider.run(sched, connector);  // pushes all carriers; closes connector on return

  std::vector<std::unique_ptr<op::operator_data>> batches;
  for (;;) {
    auto batch = ingestible.consume_next_input(connector);
    if (!batch) break;
    batches.push_back(std::move(batch));
  }
  // After the consumer returns nullptr the coalescer must be drained.
  REQUIRE(ingestible.consumer_drained());
  REQUIRE(connector.is_closed());
  return batches;
}

duckdb_native_split_payload const& payload_of(op::operator_data const& batch)
{
  auto const* input = dynamic_cast<scan_operator_input const*>(&batch);
  REQUIRE(input != nullptr);
  REQUIRE(input->metadata != nullptr);
  auto const* split_info = dynamic_cast<duckdb_native_split_info const*>(&input->metadata->scan());
  REQUIRE(split_info != nullptr);
  return split_info->payload;
}

std::string env_or(std::string_view name, std::string fallback = {})
{
  auto const* value = std::getenv(std::string{name}.c_str());
  return value ? std::string{value} : std::move(fallback);
}

bool truthy_env(std::string_view name)
{
  auto value = env_or(name);
  return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "YES";
}

struct s3_test_env {
  std::string endpoint;
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string session_token;
  std::string bucket;
  bool strict{false};
};

std::optional<s3_test_env> read_s3_test_env()
{
  auto endpoint   = env_or("SIRIUS_TEST_S3_ENDPOINT");
  auto access_key = env_or("SIRIUS_TEST_S3_ACCESS_KEY");
  auto secret_key = env_or("SIRIUS_TEST_S3_SECRET_KEY");
  auto bucket     = env_or("SIRIUS_TEST_S3_BUCKET");

  if (endpoint.empty() || access_key.empty() || secret_key.empty() || bucket.empty()) {
    return std::nullopt;
  }

  return s3_test_env{std::move(endpoint),
                     env_or("SIRIUS_TEST_S3_REGION", "us-east-1"),
                     std::move(access_key),
                     std::move(secret_key),
                     env_or("SIRIUS_TEST_S3_SESSION_TOKEN"),
                     std::move(bucket),
                     truthy_env("SIRIUS_TEST_S3_STRICT")};
}

bool skip_if_no_s3_env(std::optional<s3_test_env> const& env)
{
  if (env) { return false; }
  if (truthy_env("SIRIUS_TEST_S3_STRICT")) {
    FAIL("SIRIUS_TEST_S3_* environment is required in strict mode");
  }
  SUCCEED("SIRIUS_TEST_S3_* not set; skipping duckdb-native S3 integration test");
  return true;
}

std::string endpoint_authority(std::string const& endpoint)
{
  auto start = endpoint.find("://");
  start      = start == std::string::npos ? 0 : start + 3;
  auto end   = endpoint.find('/', start);
  return endpoint.substr(start, end == std::string::npos ? std::string::npos : end - start);
}

std::string trim_trailing_slash(std::string endpoint)
{
  while (!endpoint.empty() && endpoint.back() == '/') {
    endpoint.pop_back();
  }
  return endpoint;
}

std::string s3_uri(std::string_view bucket, std::string_view key)
{
  return "s3://" + std::string{bucket} + "/" + std::string{key};
}

std::string s3_uri_path(std::string_view bucket, std::string_view key)
{
  return "/" + sirius::io::s3::uri_encode(bucket, false) + "/" +
         sirius::io::s3::uri_encode(key, false);
}

std::size_t curl_read_file(char* buffer, std::size_t size, std::size_t nitems, void* userdata)
{
  auto* file = static_cast<std::FILE*>(userdata);
  if (file == nullptr) { return 0; }
  return std::fread(buffer, 1, size * nitems, file);
}

void upload_file_to_s3(s3_test_env const& env, fs::path const& path, std::string_view key)
{
  auto* body = std::fopen(path.string().c_str(), "rb");
  REQUIRE(body != nullptr);

  auto const close_body = [&] { std::fclose(body); };
  auto const body_len   = static_cast<std::int64_t>(fs::file_size(path));
  auto const authority  = endpoint_authority(env.endpoint);
  auto const uri        = s3_uri_path(env.bucket, key);

  sirius::io::s3::sigv4_signer_config creds;
  creds.access_key    = env.access_key;
  creds.secret_key    = env.secret_key;
  creds.region        = env.region;
  creds.service       = "s3";
  creds.session_token = env.session_token;

  auto signed_req = sirius::io::s3::sign_request("PUT",
                                                 authority,
                                                 uri,
                                                 /*canonical_query=*/"",
                                                 "UNSIGNED-PAYLOAD",
                                                 /*extra_headers=*/{},
                                                 creds,
                                                 std::time(nullptr));

  REQUIRE(curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK);
  CURL* curl = curl_easy_init();
  REQUIRE(curl != nullptr);

  curl_slist* headers = nullptr;
  for (auto const& [name, value] : signed_req.headers) {
    headers = curl_slist_append(headers, (name + ": " + value).c_str());
  }
  headers = curl_slist_append(headers, "Expect:");

  auto const url = trim_trailing_slash(env.endpoint) + uri;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
  curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE, static_cast<curl_off_t>(body_len));
  curl_easy_setopt(curl, CURLOPT_READFUNCTION, curl_read_file);
  curl_easy_setopt(curl, CURLOPT_READDATA, body);
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

  long code   = -1;
  CURLcode rc = curl_easy_perform(curl);
  if (rc == CURLE_OK) { curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code); }

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  close_body();

  INFO("PUT " << url << " returned curl=" << static_cast<int>(rc) << " http=" << code);
  REQUIRE(rc == CURLE_OK);
  REQUIRE((code == 200 || code == 201 || code == 204));
}

sirius::io::s3::s3_ioctx_config make_s3_config(s3_test_env const& env)
{
  sirius::io::s3::static_credentials creds;
  creds.access_key_id     = env.access_key;
  creds.secret_access_key = env.secret_key;
  creds.session_token     = env.session_token;

  sirius::io::s3::s3_ioctx_config cfg{};
  cfg.creds = std::make_shared<sirius::io::s3::sirius_sigv4_presigned_authorizer>(
    std::move(creds), env.region, env.endpoint, 30min);
  cfg.max_connections    = 8;
  cfg.request_timeout_s  = 20;
  cfg.max_retry_attempts = 3;
  cfg.retry_backoff_base = 10ms;
  cfg.retry_jitter       = 0ms;
  cfg.honor_retry_after  = false;
  return cfg;
}

std::unique_ptr<sirius_scan_manager> make_s3_scan_manager(s3_test_env const& env)
{
  scan_manager_config cfg{};
  cfg.use_sirius_datasource             = true;
  cfg.uring_n_reactors                  = 1;
  cfg.s3_config                         = make_s3_config(env);
  cfg.s3_use_async_backend              = true;
  cfg.s3_thread_pool.num_threads        = 4;
  cfg.s3_thread_pool.thread_name_prefix = "native_s3";
  return std::make_unique<sirius_scan_manager>(std::move(cfg));
}

class uploaded_native_db_fixture {
 public:
  explicit uploaded_native_db_fixture(s3_test_env const& env)
    : _dir(fs::temp_directory_path() /
           ("sirius_native_s3_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)))),
      _local_path(_dir / "blocks.duckdb"),
      _key("native/blocks-" + std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ".duckdb"),
      _uri(s3_uri(env.bucket, _key))
  {
    fs::remove_all(_dir);
    fs::create_directories(_dir);

    {
      duckdb::DuckDB writer(_local_path.string());
      duckdb::Connection con(writer);
      exec_ok(con,
              "CREATE TABLE native_blocks AS "
              "SELECT i::BIGINT AS id, "
              "       hash(i)::UBIGINT AS h0, "
              "       hash(i * 1315423911)::UBIGINT AS h1, "
              "       hash(i + 2654435761)::UBIGINT AS h2 "
              "FROM range(0, 400000) tbl(i)");
      exec_ok(con, "CHECKPOINT");
    }

    REQUIRE(fs::exists(_local_path));
    REQUIRE(fs::file_size(_local_path) > 0);
    upload_file_to_s3(env, _local_path, _key);

    _db_owner = std::make_unique<duckdb::DuckDB>(_local_path.string());
    _con      = std::make_unique<duckdb::Connection>(*_db_owner);
    _storage  = &get_storage(*_con, "native_blocks");

    auto& sm = _storage->GetAttached().GetStorageManager();
    _blocks  = dynamic_cast<duckdb::SingleFileBlockManager*>(&sm.GetBlockManager());
    REQUIRE(_blocks != nullptr);
    REQUIRE(_blocks->TotalBlocks() >= 4);
  }

  ~uploaded_native_db_fixture()
  {
    std::error_code ec;
    fs::remove_all(_dir, ec);
  }

  fs::path const& local_path() const noexcept { return _local_path; }
  std::string const& uri() const noexcept { return _uri; }
  duckdb::DataTable& storage() const noexcept { return *_storage; }
  duckdb::Connection& connection() const noexcept { return *_con; }
  duckdb::SingleFileBlockManager& block_manager() const noexcept { return *_blocks; }

 private:
  fs::path _dir;
  fs::path _local_path;
  std::string _key;
  std::string _uri;
  std::unique_ptr<duckdb::DuckDB> _db_owner;
  std::unique_ptr<duckdb::Connection> _con;
  duckdb::DataTable* _storage{nullptr};
  duckdb::SingleFileBlockManager* _blocks{nullptr};
};

cudf::io::text::byte_range_info range(std::size_t offset, std::size_t size)
{
  return cudf::io::text::byte_range_info{static_cast<int64_t>(offset), static_cast<int64_t>(size)};
}

std::vector<std::uint8_t> read_local_range(fs::path const& path,
                                           std::size_t offset,
                                           std::size_t size)
{
  REQUIRE(offset + size <= fs::file_size(path));
  std::vector<std::uint8_t> out(size);
  std::ifstream in(path, std::ios::binary);
  REQUIRE(in.good());
  in.seekg(static_cast<std::streamoff>(offset));
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
  REQUIRE(static_cast<std::size_t>(in.gcount()) == size);
  return out;
}

std::vector<std::vector<std::uint8_t>> read_s3_ranges(
  sirius::io::sirius_ioctx& ctx,
  sirius::io::sirius_io_object& obj,
  std::vector<cudf::io::text::byte_range_info> const& ranges)
{
  std::vector<std::vector<std::byte>> buffers;
  buffers.reserve(ranges.size());
  std::vector<cudf::host_span<std::byte>> spans;
  spans.reserve(ranges.size());

  std::size_t expected_total = 0;
  for (auto const& r : ranges) {
    REQUIRE(r.size() >= 0);
    auto const size = static_cast<std::size_t>(r.size());
    expected_total += size;
    buffers.emplace_back(size);
    spans.emplace_back(buffers.back().data(), buffers.back().size());
  }

  auto done   = std::make_shared<std::promise<std::pair<std::size_t, std::exception_ptr>>>();
  auto future = done->get_future();
  ctx.host_read_ranges_async_io(obj,
                                ranges,
                                std::span<cudf::host_span<std::byte>>{spans.data(), spans.size()},
                                [done](std::size_t bytes, std::exception_ptr ep) mutable {
                                  try {
                                    done->set_value({bytes, ep});
                                  } catch (std::future_error const&) {
                                    // A backend bug that fires the completion twice should fail
                                    // through the first observed result, not crash the test
                                    // process.
                                  }
                                });

  REQUIRE(future.wait_for(30s) == std::future_status::ready);
  auto [bytes, ep] = future.get();
  if (ep) { std::rethrow_exception(ep); }
  REQUIRE(bytes == expected_total);

  std::vector<std::vector<std::uint8_t>> out;
  out.reserve(buffers.size());
  for (auto const& buf : buffers) {
    std::vector<std::uint8_t> bytes_out(buf.size());
    std::transform(buf.begin(), buf.end(), bytes_out.begin(), [](std::byte b) {
      return std::to_integer<std::uint8_t>(b);
    });
    out.push_back(std::move(bytes_out));
  }
  return out;
}

void require_s3_ranges_match_local(sirius::io::sirius_ioctx& ctx,
                                   sirius::io::sirius_io_object& obj,
                                   fs::path const& local_path,
                                   std::vector<cudf::io::text::byte_range_info> const& ranges)
{
  auto got = read_s3_ranges(ctx, obj, ranges);
  REQUIRE(got.size() == ranges.size());
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    auto const offset = static_cast<std::size_t>(ranges[i].offset());
    auto const size   = static_cast<std::size_t>(ranges[i].size());
    CHECK(got[i] == read_local_range(local_path, offset, size));
  }
}

std::vector<cudf::io::text::byte_range_info> duckdb_header_ranges()
{
  auto const header = static_cast<std::size_t>(duckdb::Storage::FILE_HEADER_SIZE);
  REQUIRE(sirius::op::scan::DUCKDB_BLOCK_START == 3 * header);
  return {range(0, header), range(header, header), range(2 * header, header)};
}

std::vector<cudf::io::text::byte_range_info> duckdb_payload_ranges(
  duckdb::SingleFileBlockManager& bm, std::size_t file_size)
{
  std::vector<cudf::io::text::byte_range_info> ranges;
  auto const block_size   = static_cast<std::size_t>(bm.GetBlockSize());
  auto const total_blocks = bm.TotalBlocks();
  for (duckdb::idx_t block = 0; block < total_blocks; ++block) {
    auto const offset =
      sirius::op::scan::duckdb_block_payload_offset(bm, static_cast<duckdb::block_id_t>(block));
    if (offset + block_size <= file_size) { ranges.push_back(range(offset, block_size)); }
  }
  return ranges;
}

std::vector<cudf::io::text::byte_range_info> small_interleaved_ranges(std::size_t file_size)
{
  REQUIRE(file_size > 4096);
  std::vector<cudf::io::text::byte_range_info> ranges;
  ranges.reserve(36);
  for (std::size_t i = 0; i < 36; ++i) {
    auto const size   = (i == 0) ? std::size_t{1} : std::size_t{1 + ((i * 17) % 97)};
    auto const stride = (i % 2 == 0) ? 9973 : 4099;
    auto const offset = (i * stride) % (file_size - size);
    ranges.push_back(range(offset, size));
  }
  return ranges;
}

}  // namespace

//===----------------------------------------------------------------------===//
// ctor hardening
//===----------------------------------------------------------------------===//
TEST_CASE("duckdb_native_gpu_ingestible throws on null storage",
          "[scan][duckdb_native_gpu_ingestible]")
{
  auto mgr  = make_scan_manager();
  auto info = make_table_info(/*storage=*/nullptr, /*ctx=*/nullptr, {}, {});
  REQUIRE_THROWS_AS((duckdb_native_gpu_ingestible{std::move(info), *mgr}), std::invalid_argument);
}

TEST_CASE("duckdb_native_gpu_ingestible throws on null context",
          "[scan][duckdb_native_gpu_ingestible]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 16)");
  auto& storage = get_storage(con, "t");

  auto mgr  = make_scan_manager();
  auto info = make_table_info(&storage, /*ctx=*/nullptr, {}, {});
  REQUIRE_THROWS_AS((duckdb_native_gpu_ingestible{std::move(info), *mgr}), std::invalid_argument);
}

TEST_CASE("duckdb_native_gpu_ingestible throws on projected vectors size mismatch",
          "[scan][duckdb_native_gpu_ingestible]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 16)");
  auto& storage = get_storage(con, "t");

  auto mgr  = make_scan_manager();
  auto info = make_table_info(&storage, con.context.get(), {real_col(0)}, /*types=*/{});
  REQUIRE_THROWS_AS((duckdb_native_gpu_ingestible{std::move(info), *mgr}), std::invalid_argument);
}

TEST_CASE("duckdb_native_gpu_ingestible leaves io handles null for an empty db_path",
          "[scan][duckdb_native_gpu_ingestible]")
{
  // With no .db path the ctor resolves no datasource: the io handles stay null
  // and construction still succeeds. A backend-less scan is rejected by the
  // decoder, not the ctor; this exercises the null-handle path without driving
  // the decoder.
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 16)");
  auto& storage = get_storage(con, "t");

  auto mgr  = make_scan_manager();
  auto info = make_table_info(&storage,
                              con.context.get(),
                              {real_col(0)},
                              {sirius::logical_type::make(sirius::type_id::INTEGER)});
  duckdb_native_gpu_ingestible ingestible{std::move(info), *mgr};

  auto batches = run_and_consume(ingestible);
  REQUIRE_FALSE(batches.empty());
  auto const& payload = payload_of(*batches[0]);
  REQUIRE(payload.io_ctx == nullptr);
  REQUIRE(payload.db_io_object == nullptr);
}

TEST_CASE("S3 ioctx serves DuckDB-native block range shapes byte-identically",
          "[.][s3][integration][duckdbnative]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  uploaded_native_db_fixture fixture{*env};
  auto mgr = make_s3_scan_manager(*env);
  auto ds  = mgr->create_datasource(fixture.uri());
  REQUIRE(ds != nullptr);
  REQUIRE(ds->io_ctx() != nullptr);
  REQUIRE(ds->io_object() != nullptr);
  CHECK(dynamic_cast<sirius::io::s3::s3_ioctx*>(ds->io_ctx().get()) != nullptr);
  auto const object_size = static_cast<std::size_t>(fs::file_size(fixture.local_path()));
  CHECK(ds->io_object()->size() == object_size);

  auto& ctx = *ds->io_ctx();
  auto& obj = *ds->io_object();

  require_s3_ranges_match_local(ctx, obj, fixture.local_path(), duckdb_header_ranges());

  auto payload_ranges = duckdb_payload_ranges(fixture.block_manager(), object_size);
  REQUIRE(payload_ranges.size() >= 4);
  require_s3_ranges_match_local(ctx, obj, fixture.local_path(), payload_ranges);

  require_s3_ranges_match_local(ctx, obj, fixture.local_path(), {range(0, object_size)});

  auto tiny_ranges = small_interleaved_ranges(object_size);
  REQUIRE(tiny_ranges.size() >= 32);
  require_s3_ranges_match_local(ctx, obj, fixture.local_path(), tiny_ranges);

  sirius_scan_manager no_s3_mgr{scan_manager_config{}};
  CHECK(no_s3_mgr.create_datasource(fixture.uri()) == nullptr);
}

TEST_CASE("duckdb_native_gpu_ingestible carries S3 handles for remote DuckDB-native db_path",
          "[.][s3][integration][duckdbnative]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  uploaded_native_db_fixture fixture{*env};

  auto s3_mgr  = make_s3_scan_manager(*env);
  auto s3_info = make_table_info(&fixture.storage(),
                                 fixture.connection().context.get(),
                                 {real_col(0)},
                                 {sirius::logical_type::make(sirius::type_id::BIGINT)},
                                 sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE,
                                 fixture.uri());
  duckdb_native_gpu_ingestible s3_ingestible{std::move(s3_info), *s3_mgr};

  auto s3_batches = run_and_consume(s3_ingestible);
  REQUIRE_FALSE(s3_batches.empty());
  auto const& s3_payload = payload_of(*s3_batches[0]);
  REQUIRE(s3_payload.io_ctx != nullptr);
  REQUIRE(s3_payload.db_io_object != nullptr);
  CHECK(dynamic_cast<sirius::io::s3::s3_ioctx*>(s3_payload.io_ctx.get()) != nullptr);
  CHECK(s3_payload.db_io_object->size() ==
        static_cast<std::size_t>(fs::file_size(fixture.local_path())));

  auto no_s3_mgr  = make_scan_manager();
  auto no_s3_info = make_table_info(&fixture.storage(),
                                    fixture.connection().context.get(),
                                    {real_col(0)},
                                    {sirius::logical_type::make(sirius::type_id::BIGINT)},
                                    sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE,
                                    fixture.uri());
  duckdb_native_gpu_ingestible no_s3_ingestible{std::move(no_s3_info), *no_s3_mgr};

  auto no_s3_batches = run_and_consume(no_s3_ingestible);
  REQUIRE_FALSE(no_s3_batches.empty());
  auto const& no_s3_payload = payload_of(*no_s3_batches[0]);
  REQUIRE(no_s3_payload.io_ctx == nullptr);
  REQUIRE(no_s3_payload.db_io_object == nullptr);
}

TEST_CASE("duckdb_native_gpu_ingestible throws when walker rejects the table",
          "[scan][duckdb_native_gpu_ingestible]")
{
  // HUGEINT is unsupported: prepare_duckdb_native_walk reports non-viable and
  // the ctor surfaces the rejection as a runtime_error.
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a HUGEINT)");
  exec_ok(con, "INSERT INTO t VALUES (1), (2), (3)");
  auto& storage = get_storage(con, "t");

  auto mgr  = make_scan_manager();
  auto info = make_table_info(&storage,
                              con.context.get(),
                              {real_col(0)},
                              {sirius::logical_type::make(sirius::type_id::HUGEINT)});
  REQUIRE_THROWS_AS((duckdb_native_gpu_ingestible{std::move(info), *mgr}), std::runtime_error);
}

TEST_CASE("duckdb_native_gpu_ingestible rejects fully stats-pruned scans",
          "[scan][duckdb_native_gpu_ingestible]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 300000)");
  exec_ok(con, "CHECKPOINT");
  auto& storage = get_storage(con, "t");

  auto mgr          = make_scan_manager();
  auto info         = make_table_info(&storage,
                              con.context.get(),
                                      {real_col(0)},
                                      {sirius::logical_type::make(sirius::type_id::INTEGER)});
  auto* native_info = dynamic_cast<duckdb_native_ingestible_table_info*>(info.get());
  REQUIRE(native_info != nullptr);
  native_info->table_filters             = duckdb::make_uniq<duckdb::TableFilterSet>();
  native_info->table_filters->filters[0] = duckdb::make_uniq<duckdb::ConstantFilter>(
    duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(1000000));
  native_info->column_ids.push_back(duckdb::ColumnIndex(0));

  REQUIRE_THROWS_WITH((duckdb_native_gpu_ingestible{std::move(info), *mgr}),
                      Catch::Contains("fully pruned"));
}

//===----------------------------------------------------------------------===//
// producer: range claiming
//===----------------------------------------------------------------------===//
TEST_CASE("duckdb_native_gpu_ingestible emits one range for a small INTEGER table",
          "[scan][duckdb_native_gpu_ingestible]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 1024)");
  auto& storage = get_storage(con, "t");

  auto mgr  = make_scan_manager();
  auto info = make_table_info(&storage,
                              con.context.get(),
                              {real_col(0)},
                              {sirius::logical_type::make(sirius::type_id::INTEGER)});
  duckdb_native_gpu_ingestible ingestible{std::move(info), *mgr};
  REQUIRE(ingestible.has_more_splits());

  auto factory = ingestible.next_split_provider();
  REQUIRE(factory);
  auto carriers = factory();
  REQUIRE(carriers.size() == 1);

  auto* range = dynamic_cast<duckdb_native_range_input*>(carriers[0].get());
  REQUIRE(range != nullptr);
  REQUIRE_FALSE(range->row_groups.empty());

  // After the only range is claimed both signals report exhaustion.
  REQUIRE_FALSE(ingestible.has_more_splits());
  auto exhausted = ingestible.next_split_provider();
  REQUIRE_FALSE(exhausted);
}

TEST_CASE("duckdb_native_gpu_ingestible ranges cover all row groups exactly once",
          "[scan][duckdb_native_gpu_ingestible]")
{
  // ~1.2M INTEGER rows span several DuckDB row groups (<=122,880 rows each), so
  // the producer hands out multiple parse ranges. Their union must cover every
  // row group exactly once.
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 1200000)");
  auto& storage = get_storage(con, "t");

  auto mgr  = make_scan_manager();
  auto info = make_table_info(&storage,
                              con.context.get(),
                              {real_col(0)},
                              {sirius::logical_type::make(sirius::type_id::INTEGER)});
  duckdb_native_gpu_ingestible ingestible{std::move(info), *mgr};
  auto result = drain_ranges(ingestible);

  REQUIRE(result.row_groups.size() >= 2);
  REQUIRE(result.range_count >= 1);

  std::sort(result.row_groups.begin(), result.row_groups.end(), [](auto const& a, auto const& b) {
    return a.row_group_index < b.row_group_index;
  });
  for (std::size_t i = 0; i < result.row_groups.size(); ++i) {
    REQUIRE(result.row_groups[i].row_group_index == i);
    REQUIRE(result.row_groups[i].row_count > 0);
  }
  for (std::size_t i = 1; i < result.row_groups.size(); ++i) {
    REQUIRE(result.row_groups[i].row_group_start > result.row_groups[i - 1].row_group_start);
  }
  REQUIRE_FALSE(ingestible.has_more_splits());
}

//===----------------------------------------------------------------------===//
// consumer: coalescing + single-split race
//===----------------------------------------------------------------------===//
TEST_CASE("duckdb_native_gpu_ingestible consumer coalesces ranges, covers all row groups once",
          "[scan][duckdb_native_gpu_ingestible]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 1200000)");
  auto& storage = get_storage(con, "t");

  // A small byte cap forces the coalescer to close multiple batches across the
  // parsed ranges.
  auto mgr  = make_scan_manager();
  auto info = make_table_info(&storage,
                              con.context.get(),
                              {real_col(0)},
                              {sirius::logical_type::make(sirius::type_id::INTEGER)},
                              /*approximate_batch_size=*/4096);
  duckdb_native_gpu_ingestible ingestible{std::move(info), *mgr};

  auto batches = run_and_consume(ingestible);
  REQUIRE(batches.size() >= 2);  // multiple coalesced batches

  // Every batch's payload aliases the same bind data, and the union of their
  // row groups covers every row group exactly once.
  duckdb_native_ingestible_table_info const* first_table_info = nullptr;
  std::vector<duckdb::idx_t> seen_indices;
  for (auto const& b : batches) {
    auto const& payload = payload_of(*b);
    REQUIRE(payload.table_info != nullptr);
    REQUIRE_FALSE(payload.row_groups.empty());
    if (first_table_info == nullptr) {
      first_table_info = payload.table_info;
    } else {
      REQUIRE(payload.table_info == first_table_info);
    }
    for (auto const& rg : payload.row_groups) {
      seen_indices.push_back(rg.row_group_index);
    }
  }
  std::sort(seen_indices.begin(), seen_indices.end());
  for (std::size_t i = 0; i < seen_indices.size(); ++i) {
    REQUIRE(seen_indices[i] == i);  // contiguous from 0, no gaps or duplicates
  }
  REQUIRE(seen_indices.size() >= 2);
}

TEST_CASE("duckdb_native_gpu_ingestible single-split scan serves its batch then drains",
          "[scan][duckdb_native_gpu_ingestible]")
{
  // A single-split scan (one parse range) must serve its one batch before
  // reporting drained, not drop it when the connector closes.
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 1024)");
  auto& storage = get_storage(con, "t");

  auto mgr  = make_scan_manager();
  auto info = make_table_info(&storage,
                              con.context.get(),
                              {real_col(0)},
                              {sirius::logical_type::make(sirius::type_id::INTEGER)});
  duckdb_native_gpu_ingestible ingestible{std::move(info), *mgr};

  auto batches = run_and_consume(ingestible);
  REQUIRE(batches.size() == 1);  // one row group → one tail batch, not dropped

  auto const& payload = payload_of(*batches[0]);
  REQUIRE_FALSE(payload.row_groups.empty());
}
