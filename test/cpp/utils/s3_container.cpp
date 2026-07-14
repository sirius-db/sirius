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

#include "utils/s3_container.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sirius::test {

namespace {

bool env_truthy(char const* name)
{
  auto const* v = std::getenv(name);
  if (v == nullptr) return false;
  std::string_view s{v};
  return s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "YES";
}

bool env_set(char const* name)
{
  auto const* v = std::getenv(name);
  return v != nullptr && v[0] != '\0';
}

}  // namespace

}  // namespace sirius::test

#include "io/s3/sigv4.hpp"

#include <curl/curl.h>

extern "C" {
#include <testcontainers-c/container.h>
}

#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace sirius::test {
namespace {

namespace fs = std::filesystem;

// Pinned MinIO image — kept in sync with the (now-retired) docker-compose.yml so
// the same Sirius commit produces reproducible S3 test behavior over time.
constexpr char const* kMinioImage = "minio/minio:RELEASE.2025-09-07T16-13-09Z-cpuv1";
constexpr int kMinioPort          = 9000;
constexpr char const* kAccessKey  = "minioadmin";
constexpr char const* kSecretKey  = "minioadmin";
constexpr char const* kRegion     = "us-east-1";
constexpr char const* kBucket     = "sirius-test";
constexpr char const* kDefaultKey = "hello.txt";

// ---- small helpers ---------------------------------------------------------

std::string env_or(char const* name, std::string fallback = {})
{
  auto const* v = std::getenv(name);
  return v != nullptr ? std::string{v} : std::move(fallback);
}

// Run argv synchronously (inheriting stdout/stderr) and return its exit code,
// or -1 if it could not be spawned / exited abnormally.
int run_process(std::vector<std::string> const& argv)
{
  std::vector<char*> c_argv;
  c_argv.reserve(argv.size() + 1);
  for (auto const& a : argv)
    c_argv.push_back(const_cast<char*>(a.c_str()));
  c_argv.push_back(nullptr);

  pid_t pid = fork();
  if (pid < 0) return -1;
  if (pid == 0) {
    execvp(c_argv[0], c_argv.data());
    _exit(127);  // execvp only returns on failure
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) return -1;
  return (WIFEXITED(status)) ? WEXITSTATUS(status) : -1;
}

// ---- testcontainers wrappers -----------------------------------------------

std::vector<int> g_running_containers;  // ids to terminate at shutdown

struct minio_instance {
  std::string host;       // e.g. "localhost"
  int port{0};            // dynamically-mapped host port
  std::string endpoint;   // "<scheme>://<host>:<port>"
  std::string authority;  // "<host>:<port>" (Host header / signing)
};

// Configure the common MinIO request (image, creds, command, exposed port).
int make_minio_request()
{
  int req = tc_container_create(kMinioImage);
  if (req < 0) throw std::runtime_error("tc_container_create failed for MinIO image");

  tc_container_with_env(req, "MINIO_ROOT_USER", kAccessKey);
  tc_container_with_env(req, "MINIO_ROOT_PASSWORD", kSecretKey);
  tc_container_with_env(req, "MINIO_REGION", kRegion);

  // The MinIO image takes its mode from the command — no env-var equivalent —
  // which is exactly the capability our testcontainers-native patch adds.
  char const* cmd[] = {"server", "/data", "--console-address", ":9001"};
  tc_container_with_cmd(req, cmd, sizeof(cmd) / sizeof(cmd[0]));

  tc_container_with_exposed_tcp_port(req, kMinioPort);
  return req;
}

minio_instance run_minio(int req, char const* scheme)
{
  char* run_err = nullptr;
  int id        = tc_container_run(req, &run_err);
  if (id < 0) {
    std::string msg = run_err != nullptr ? std::string{run_err} : std::string{"unknown error"};
    std::free(run_err);
    throw std::runtime_error(
      "failed to start MinIO container (is Docker running and reachable?): " + msg);
  }
  g_running_containers.push_back(id);

  char* herr       = nullptr;
  char* hp         = tc_container_get_hostname(id, &herr);
  std::string host = (hp != nullptr) ? std::string{hp} : std::string{"localhost"};
  std::free(hp);
  std::free(herr);

  char* perr = nullptr;
  int port   = tc_container_get_mapped_port(id, kMinioPort, &perr);
  std::free(perr);
  if (port < 0) throw std::runtime_error("could not resolve MinIO mapped port");

  minio_instance inst;
  inst.host      = host;
  inst.port      = port;
  inst.authority = host + ":" + std::to_string(port);
  inst.endpoint  = std::string{scheme} + "://" + inst.authority;
  return inst;
}

// ---- TLS cert + fixtures ----------------------------------------------------

// Generate a self-signed cert/key into dir (public.crt / private.key), matching
// the SANs the old ensure_tls_certs.sh used. Returns the public cert path.
fs::path generate_self_signed_cert(fs::path const& dir)
{
  fs::create_directories(dir);
  fs::path cert = dir / "public.crt";
  fs::path key  = dir / "private.key";
  int rc        = run_process({"openssl",
                               "req",
                               "-x509",
                               "-newkey",
                               "rsa:2048",
                               "-nodes",
                               "-days",
                               "365",
                               "-keyout",
                               key.string(),
                               "-out",
                               cert.string(),
                               "-subj",
                               "/CN=localhost",
                               "-addext",
                               "subjectAltName=IP:127.0.0.1,DNS:localhost,DNS:host.docker.internal"});
  if (rc != 0) throw std::runtime_error("openssl failed to generate self-signed cert");
  return cert;
}

// Run generate_fixtures.py into out_dir; returns the populated directory.
fs::path generate_fixtures(fs::path const& out_dir)
{
  fs::path root   = SIRIUS_PROJECT_ROOT;
  fs::path script = root / "test" / "cpp" / "integration" / "s3" / "generate_fixtures.py";
  fs::path parquet_src =
    env_or("SIRIUS_TEST_S3_PARQUET_SOURCE",
           (root / "test" / "cpp" / "integration" / "data" / "parquet").string());
  fs::create_directories(out_dir);
  int rc = run_process(
    {"python3", script.string(), "--out", out_dir.string(), "--parquet-source", parquet_src});
  if (rc != 0) throw std::runtime_error("generate_fixtures.py failed");
  return out_dir;
}

// ---- host-side SigV4 + libcurl upload --------------------------------------

size_t curl_read_file(char* buffer, size_t size, size_t nitems, void* userdata)
{
  auto* f = static_cast<std::FILE*>(userdata);
  if (f == nullptr) return 0;
  return std::fread(buffer, 1, size * nitems, f);
}

// Signed PUT to <endpoint><canonical_uri>. body may be null (zero-length, used
// for CreateBucket). Returns the HTTP status code, or -1 on transport error.
long s3_put(minio_instance const& inst,
            std::string const& scheme,
            std::string const& canonical_uri,
            std::FILE* body,
            std::int64_t body_len,
            std::optional<fs::path> const& ca_bundle)
{
  sirius::io::s3::sigv4_signer_config creds;
  creds.access_key = kAccessKey;
  creds.secret_key = kSecretKey;
  creds.region     = kRegion;
  creds.service    = "s3";

  // UNSIGNED-PAYLOAD lets us stream arbitrarily large bodies (e.g. the SF10
  // lineitem fixture) without hashing them; MinIO accepts it.
  auto signed_req = sirius::io::s3::sign_request("PUT",
                                                 inst.authority,
                                                 canonical_uri,
                                                 /*query=*/"",
                                                 "UNSIGNED-PAYLOAD",
                                                 /*extra_headers=*/{},
                                                 creds,
                                                 std::time(nullptr));

  CURL* curl = curl_easy_init();
  if (curl == nullptr) return -1;

  std::string url  = inst.endpoint + canonical_uri;
  curl_slist* hdrs = nullptr;
  for (auto const& [k, v] : signed_req.headers) {
    hdrs = curl_slist_append(hdrs, (k + ": " + v).c_str());
  }
  // Suppress libcurl's automatic "Expect: 100-continue" (unsigned, but avoids a
  // round-trip stall against MinIO).
  hdrs = curl_slist_append(hdrs, "Expect:");

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
  curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE, static_cast<curl_off_t>(body_len));
  curl_easy_setopt(curl, CURLOPT_READFUNCTION, curl_read_file);
  curl_easy_setopt(curl, CURLOPT_READDATA, body);
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
  if (scheme == "https" && ca_bundle.has_value()) {
    curl_easy_setopt(curl, CURLOPT_CAINFO, ca_bundle->c_str());
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
  }

  long code   = -1;
  CURLcode rc = curl_easy_perform(curl);
  if (rc == CURLE_OK) curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);

  curl_slist_free_all(hdrs);
  curl_easy_cleanup(curl);
  return code;
}

std::string uri_path_for(std::string const& bucket, std::string const& key)
{
  std::string p = "/" + sirius::io::s3::uri_encode(bucket, false);
  if (!key.empty()) p += "/" + sirius::io::s3::uri_encode(key, false);
  return p;
}

// Poll <endpoint>/minio/health/ready until it returns 200 (or time out).
bool wait_minio_ready(minio_instance const& inst,
                      std::string const& scheme,
                      std::optional<fs::path> const& ca_bundle)
{
  std::string url = inst.endpoint + "/minio/health/ready";
  for (int attempt = 0; attempt < 60; ++attempt) {
    CURL* curl = curl_easy_init();
    if (curl == nullptr) return false;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3L);
    if (scheme == "https" && ca_bundle.has_value()) {
      curl_easy_setopt(curl, CURLOPT_CAINFO, ca_bundle->c_str());
      curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
      curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    }
    long code   = -1;
    CURLcode rc = curl_easy_perform(curl);
    if (rc == CURLE_OK) curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    curl_easy_cleanup(curl);
    if (code == 200) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  return false;
}

void upload_fixtures(minio_instance const& inst,
                     std::string const& scheme,
                     fs::path const& fixture_dir,
                     std::optional<fs::path> const& ca_bundle)
{
  // Create the bucket (ignore 200/204 and 409 BucketAlreadyOwnedByYou).
  long bc = s3_put(inst, scheme, uri_path_for(kBucket, ""), nullptr, 0, ca_bundle);
  if (!(bc == 200 || bc == 204 || bc == 409)) {
    throw std::runtime_error("create-bucket PUT failed (HTTP " + std::to_string(bc) + ") at " +
                             inst.endpoint);
  }

  for (auto const& entry : fs::recursive_directory_iterator(fixture_dir)) {
    if (!entry.is_regular_file()) continue;
    std::string key = fs::relative(entry.path(), fixture_dir).generic_string();

    std::FILE* f = std::fopen(entry.path().c_str(), "rb");
    if (f == nullptr) throw std::runtime_error("cannot open fixture: " + entry.path().string());
    auto size = static_cast<std::int64_t>(entry.file_size());
    long pc   = s3_put(inst, scheme, uri_path_for(kBucket, key), f, size, ca_bundle);
    std::fclose(f);
    if (!(pc == 200 || pc == 204)) {
      throw std::runtime_error("put-object PUT failed for '" + key + "' (HTTP " +
                               std::to_string(pc) + ") at " + inst.endpoint);
    }
  }
}

// Opt-in (SIRIUS_TEST_S3_LARGE=1): generate the SF10 lineitem parquet with the
// DuckDB CLI and upload it for the [s3][sql][large] / benchmark tests. Replaces
// the old `fixtures.sh --perf` path. Uploaded to both HTTP and TLS endpoints.
void maybe_upload_large_fixture(minio_instance const& http,
                                minio_instance const& tls,
                                std::optional<fs::path> const& ca,
                                fs::path const& work)
{
  if (!env_truthy("SIRIUS_TEST_S3_LARGE")) return;

  // The SF10 generation costs minutes, so cache the parquet at a stable path and
  // reuse it across the separate-process invocations the large gate runs (each
  // prewarm/no-prewarm case needs its own process). Only the first generates.
  fs::path parquet = work / "lineitem_sf10.parquet";
  std::error_code ec;
  if (!(fs::exists(parquet, ec) && fs::file_size(parquet, ec) > 0)) {
    fs::path duckdb_bin =
      env_or("SIRIUS_TEST_DUCKDB",
             (fs::path{SIRIUS_PROJECT_ROOT} / "build" / "release" / "duckdb").string());
    fs::path db = work / "tpch_sf10.duckdb";
    // Generate to a temp file and atomically rename, so an interrupted run never
    // leaves a truncated-but-non-empty file that the cache check would reuse.
    fs::path parquet_tmp = work / "lineitem_sf10.parquet.tmp";
    fs::remove(db, ec);
    fs::remove(parquet_tmp, ec);
    std::string sql =
      "INSTALL tpch; LOAD tpch; CALL dbgen(sf=10); "
      "COPY (SELECT * FROM lineitem) TO '" +
      parquet_tmp.string() + "' (FORMAT PARQUET);";
    int rc = run_process({duckdb_bin.string(), db.string(), "-c", sql});
    fs::remove(db, ec);
    if (rc != 0) {
      fs::remove(parquet_tmp, ec);
      throw std::runtime_error("failed to generate SF10 lineitem via DuckDB CLI (" +
                               duckdb_bin.string() +
                               "); run `make release` or set SIRIUS_TEST_DUCKDB");
    }
    fs::rename(parquet_tmp, parquet, ec);
    if (ec) throw std::runtime_error("failed to finalize SF10 parquet cache: " + ec.message());
  } else {
    std::cout << "[s3] reusing cached SF10 lineitem fixture at " << parquet << std::endl;
  }

  std::string key = env_or("SIRIUS_BENCH_S3_KEY", "tpch/lineitem_sf10.parquet");
  std::FILE* f    = std::fopen(parquet.c_str(), "rb");
  if (f == nullptr) throw std::runtime_error("cannot open generated SF10 parquet");
  long pc = s3_put(http,
                   "http",
                   uri_path_for(kBucket, key),
                   f,
                   static_cast<std::int64_t>(fs::file_size(parquet)),
                   std::nullopt);
  std::fclose(f);
  if (!(pc == 200 || pc == 204)) {
    throw std::runtime_error("SF10 upload failed (HTTP " + std::to_string(pc) + ")");
  }
  std::FILE* ft = std::fopen(parquet.c_str(), "rb");
  if (ft == nullptr) throw std::runtime_error("cannot reopen SF10 parquet for TLS upload");
  long tc = s3_put(tls,
                   "https",
                   uri_path_for(kBucket, key),
                   ft,
                   static_cast<std::int64_t>(fs::file_size(parquet)),
                   ca);
  std::fclose(ft);
  if (!(tc == 200 || tc == 204)) {
    throw std::runtime_error("SF10 TLS upload failed (HTTP " + std::to_string(tc) + ")");
  }
  std::cout << "[s3] uploaded SF10 lineitem fixture (" << fs::file_size(parquet) << " bytes) to "
            << http.endpoint << "/" << kBucket << "/" << key << " and " << tls.endpoint << "/"
            << kBucket << "/" << key << std::endl;

  // The large tests read the same SF10 file locally to build a CPU oracle to
  // compare the GPU-over-S3 result against. Point them at the file we just
  // uploaded so the local copy and the S3 object are byte-identical. (Respect an
  // explicit override.)
  if (!env_set("SIRIUS_PR6_LARGE_LOCAL_PARQUET")) {
    ::setenv("SIRIUS_PR6_LARGE_LOCAL_PARQUET", parquet.c_str(), /*overwrite=*/1);
  }
}

void maybe_upload_tpch_sf1_fixture(minio_instance const& http,
                                   minio_instance const& tls,
                                   std::optional<fs::path> const& ca,
                                   fs::path const& work)
{
  if (!env_truthy("SIRIUS_TEST_S3_TPCH") && !env_truthy("SIRIUS_BENCH_S3_TPCH")) return;

  constexpr std::array<std::string_view, 8> tables = {
    "nation", "region", "customer", "orders", "part", "partsupp", "supplier", "lineitem"};
  fs::path fixture_dir = work / "tpch_sf1";

  auto fixture_complete = [&] {
    std::error_code ec;
    for (auto const table : tables) {
      auto const parquet = fixture_dir / (std::string{table} + ".parquet");
      if (!fs::exists(parquet, ec) || fs::file_size(parquet, ec) == 0 || ec) return false;
    }
    return true;
  };

  if (!fixture_complete()) {
    fs::path duckdb_bin =
      env_or("SIRIUS_TEST_DUCKDB",
             (fs::path{SIRIUS_PROJECT_ROOT} / "build" / "release" / "duckdb").string());
    fs::path db          = work / "tpch_sf1.duckdb";
    fs::path fixture_tmp = work / "tpch_sf1.tmp";
    std::error_code ec;
    fs::remove(db, ec);
    fs::remove_all(fixture_tmp, ec);
    fs::create_directories(fixture_tmp);

    std::string sql = "INSTALL tpch; LOAD tpch; CALL dbgen(sf=1); ";
    for (auto const table : tables) {
      auto const parquet = fixture_tmp / (std::string{table} + ".parquet");
      sql += "COPY (SELECT * FROM " + std::string{table} + ") TO '" + parquet.string() +
             "' (FORMAT PARQUET); ";
    }

    int rc = run_process({duckdb_bin.string(), db.string(), "-c", sql});
    fs::remove(db, ec);
    if (rc != 0) {
      fs::remove_all(fixture_tmp, ec);
      throw std::runtime_error("failed to generate SF1 TPC-H parquet fixtures via DuckDB CLI (" +
                               duckdb_bin.string() + ")");
    }

    fs::remove_all(fixture_dir, ec);
    fs::rename(fixture_tmp, fixture_dir, ec);
    if (ec) throw std::runtime_error("failed to finalize SF1 TPC-H fixture cache: " + ec.message());
    if (!fixture_complete()) {
      throw std::runtime_error("SF1 TPC-H fixture generation left an incomplete cache");
    }
  } else {
    std::cout << "[s3] reusing cached SF1 TPC-H fixtures at " << fixture_dir << std::endl;
  }

  std::uintmax_t total_bytes = 0;
  for (auto const table : tables) {
    auto const parquet = fixture_dir / (std::string{table} + ".parquet");
    auto const key     = "tpch/sf1/" + std::string{table} + ".parquet";
    auto const bytes   = static_cast<std::int64_t>(fs::file_size(parquet));
    total_bytes += static_cast<std::uintmax_t>(bytes);

    auto upload = [&](minio_instance const& instance,
                      std::string const& scheme,
                      std::optional<fs::path> const& ca_bundle) {
      std::FILE* file = std::fopen(parquet.c_str(), "rb");
      if (file == nullptr) {
        throw std::runtime_error("cannot open SF1 TPC-H fixture: " + parquet.string());
      }
      auto const code =
        s3_put(instance, scheme, uri_path_for(kBucket, key), file, bytes, ca_bundle);
      std::fclose(file);
      if (!(code == 200 || code == 204)) {
        throw std::runtime_error("SF1 TPC-H upload failed for '" + key + "' (HTTP " +
                                 std::to_string(code) + ") at " + instance.endpoint);
      }
    };

    upload(http, "http", std::nullopt);
    upload(tls, "https", ca);
  }

  setenv("SIRIUS_TEST_S3_TPCH_LOCAL_DIR", fixture_dir.c_str(), /*overwrite=*/1);
  std::cout << "[s3] uploaded 8 SF1 TPC-H parquet fixtures (" << total_bytes << " bytes) to "
            << http.endpoint << " and " << tls.endpoint << std::endl;
}

// ---- orchestration ---------------------------------------------------------

void setenv_kv(char const* k, std::string const& v) { ::setenv(k, v.c_str(), /*overwrite=*/1); }

// Returns true if it brought everything up and published the env successfully.
bool bring_up()
{
  curl_global_init(CURL_GLOBAL_DEFAULT);

  fs::path work        = fs::temp_directory_path() / "sirius-s3-testcontainers";
  fs::path certs_dir   = work / "certs";
  fs::path fixture_dir = work / "fixtures" / "local";

  fs::path ca_bundle = generate_self_signed_cert(certs_dir);
  generate_fixtures(fixture_dir);

  // HTTP instance: testcontainers' HTTP wait makes it ready before run returns.
  int http_req = make_minio_request();
  tc_container_with_wait_for_http(http_req, kMinioPort, "/minio/health/ready");
  minio_instance http = run_minio(http_req, "http");

  // TLS instance: mount the generated cert so MinIO serves HTTPS on 9000. We
  // can't use the plain-HTTP wait strategy against it, so poll readiness below.
  // NOTE: the bridge reads these host paths lazily (C.GoString runs inside the
  // run-time customizer, not at registration), so the strings must outlive the
  // run_minio() call below — keep them in locals, not temporaries.
  int tls_req               = make_minio_request();
  std::string tls_cert_path = (certs_dir / "public.crt").string();
  std::string tls_key_path  = (certs_dir / "private.key").string();
  tc_container_with_file(tls_req, tls_cert_path.c_str(), "/root/.minio/certs/public.crt");
  tc_container_with_file(tls_req, tls_key_path.c_str(), "/root/.minio/certs/private.key");
  minio_instance tls = run_minio(tls_req, "https");

  std::optional<fs::path> ca = ca_bundle;
  if (!wait_minio_ready(http, "http", std::nullopt)) {
    throw std::runtime_error("HTTP MinIO did not become ready at " + http.endpoint);
  }
  if (!wait_minio_ready(tls, "https", ca)) {
    throw std::runtime_error("TLS MinIO did not become ready at " + tls.endpoint);
  }

  upload_fixtures(http, "http", fixture_dir, std::nullopt);
  upload_fixtures(tls, "https", fixture_dir, ca);
  maybe_upload_large_fixture(http, tls, ca, work);
  maybe_upload_tpch_sf1_fixture(http, tls, ca, work);

  // Publish the env contract the [s3] tests consume (mirrors the old env.sh).
  setenv_kv("SIRIUS_TEST_S3_ENDPOINT", http.endpoint);
  setenv_kv("SIRIUS_TEST_S3_HTTPS_ENDPOINT", tls.endpoint);
  setenv_kv("SIRIUS_TEST_S3_REGION", kRegion);
  setenv_kv("SIRIUS_TEST_S3_ACCESS_KEY", kAccessKey);
  setenv_kv("SIRIUS_TEST_S3_SECRET_KEY", kSecretKey);
  setenv_kv("SIRIUS_TEST_S3_BUCKET", kBucket);
  setenv_kv("SIRIUS_TEST_S3_KEY", kDefaultKey);
  setenv_kv("SIRIUS_TEST_S3_LOCAL_DIR", fixture_dir.string());
  setenv_kv("SIRIUS_TEST_S3_CA_BUNDLE", ca_bundle.string());

  std::cout << "[s3] testcontainers MinIO ready: " << http.endpoint << " (http), " << tls.endpoint
            << " (https); fixtures in " << fixture_dir << std::endl;
  return true;
}

// Bring-up state, resolved once: 0 = untried, 1 = ready, 2 = skip,
// 3 = failed under strict mode (every [s3] test should fail loudly).
std::mutex g_state_mutex;
int g_state = 0;

}  // namespace

bool ensure_s3_container_env()
{
  // Externally-provided endpoint (manual run / real AWS): use as-is.
  if (env_set("SIRIUS_TEST_S3_ENDPOINT")) return true;

  std::lock_guard<std::mutex> lk(g_state_mutex);  // Catch2 is single-threaded; defensive.
  switch (g_state) {
    case 1: return true;
    case 2: return false;
    case 3:
      throw std::runtime_error("[s3] testcontainers bring-up previously failed (strict mode)");
    default: break;  // untried
  }

  if (!env_truthy("SIRIUS_TEST_S3_AUTO")) {
    g_state = 2;  // opt-in not requested → skip, exactly as the old flow
    return false;
  }

  try {
    bring_up();
    g_state = 1;
    return true;
  } catch (std::exception const& e) {
    std::cerr << "[s3] testcontainers bring-up failed: " << e.what() << std::endl;
    if (env_truthy("SIRIUS_TEST_S3_STRICT")) {
      g_state = 3;
      throw;
    }
    g_state = 2;  // best-effort: skip
    return false;
  }
}

void shutdown_s3_container_env()
{
  for (int id : g_running_containers) {
    char* err = tc_container_terminate(id);
    std::free(err);
  }
  g_running_containers.clear();
}

}  // namespace sirius::test
