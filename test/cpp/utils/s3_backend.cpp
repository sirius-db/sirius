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

#include "utils/s3_backend.hpp"

#include "io/rest/s3/sigv4.hpp"

#include <arpa/inet.h>
#include <curl/curl.h>
#include <duckdb.hpp>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace sirius::test {
namespace {

namespace fs = std::filesystem;

// Test-only static credentials, plus the fixed bucket/region the [s3] tests
// expect. SeaweedFS enforces the credentials via the -s3.config identities file
// below, and the harness publishes them through the SIRIUS_TEST_S3_* env vars.
constexpr char const* kAccessKey  = "siriustest";
constexpr char const* kSecretKey  = "siriustest-secret";
constexpr char const* kRegion     = "us-east-1";
constexpr char const* kBucket     = "sirius-test";
constexpr char const* kDefaultKey = "hello.txt";

// ---- small helpers ---------------------------------------------------------

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

std::string env_or(char const* name, std::string fallback = {})
{
  auto const* v = std::getenv(name);
  return v != nullptr ? std::string{v} : std::move(fallback);
}

// Run argv synchronously (inheriting stdout/stderr) and return its exit code,
// or -1 if it could not be spawned / exited abnormally. Used for the one-shot
// helper tools (openssl, python3, the DuckDB CLI) — not the long-lived server.
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
  while (waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) return -1;
  }
  return (WIFEXITED(status)) ? WEXITSTATUS(status) : -1;
}

class fixture_cache_lock {
 public:
  explicit fixture_cache_lock(fs::path const& path)
  {
    fd_ = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd_ < 0) throw std::runtime_error("cannot open fixture cache lock");
    while (::flock(fd_, LOCK_EX) != 0) {
      if (errno == EINTR) continue;
      ::close(fd_);
      throw std::runtime_error("cannot lock fixture cache");
    }
  }
  ~fixture_cache_lock() { ::close(fd_); }
  fixture_cache_lock(fixture_cache_lock const&)            = delete;
  fixture_cache_lock& operator=(fixture_cache_lock const&) = delete;

 private:
  int fd_;
};

// ---- local process management ----------------------------------------------

// Reserve `n` mutually-distinct free TCP ports on the loopback interface. We
// hold every socket open until all are chosen so the OS hands out a different
// ephemeral port each time, then close them and hand the numbers to `weed`.
// There is a small TOCTOU window between close and the server's bind — the same
// risk profile as any OS-assigned test port — and a clashing port surfaces as a
// loud strict-mode bring-up failure rather than a silent skip.
std::vector<int> reserve_free_ports(int n)
{
  std::vector<int> fds;
  std::vector<int> ports;
  fds.reserve(n);
  ports.reserve(n);
  for (int i = 0; i < n; ++i) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      for (int f : fds)
        ::close(f);
      throw std::runtime_error("failed to create socket for free-port reservation");
    }
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = 0;  // let the OS pick a free ephemeral port
    socklen_t len        = sizeof(addr);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
        ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
      ::close(fd);
      for (int f : fds)
        ::close(f);
      throw std::runtime_error("failed to reserve a free port");
    }
    fds.push_back(fd);
    ports.push_back(ntohs(addr.sin_port));
  }
  for (int f : fds)
    ::close(f);
  return ports;
}

// Spawn argv as a detached child whose stdout/stderr go to log_path, returning
// its pid. On Linux the child requests SIGKILL when its parent (the test
// binary) dies, so a crashing test run never leaks the server.
pid_t spawn_process(std::vector<std::string> const& argv, fs::path const& log_path)
{
  std::vector<char*> c_argv;
  c_argv.reserve(argv.size() + 1);
  for (auto const& a : argv)
    c_argv.push_back(const_cast<char*>(a.c_str()));
  c_argv.push_back(nullptr);

  pid_t parent = ::getpid();
  pid_t pid    = fork();
  if (pid < 0) throw std::runtime_error("fork failed for weed server");
  if (pid == 0) {
#ifdef __linux__
    if (::prctl(PR_SET_PDEATHSIG, SIGKILL) != 0) _exit(127);
    // Guard the fork→prctl race: if the parent already exited, bail out.
    if (::getppid() != parent) _exit(127);
#endif
    int fd = ::open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) _exit(127);
    if (::dup2(fd, STDOUT_FILENO) < 0 || ::dup2(fd, STDERR_FILENO) < 0) _exit(127);
    if (fd > STDERR_FILENO) ::close(fd);
    execvp(c_argv[0], c_argv.data());
    _exit(127);  // execvp only returns on failure
  }
  return pid;
}

// SIGTERM the child, give it a moment to exit cleanly, then SIGKILL; reap it so
// no zombie is left behind. Safe if the process is already gone.
void terminate_pid(pid_t pid)
{
  if (pid <= 0) return;
  ::kill(pid, SIGTERM);
  int status = 0;
  for (int i = 0; i < 50; ++i) {  // up to ~5s
    auto const result = waitpid(pid, &status, WNOHANG);
    if (result == pid || (result < 0 && errno == ECHILD)) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  ::kill(pid, SIGKILL);
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
}

// ---- S3 endpoint addressing -------------------------------------------------

struct s3_endpoint {
  std::string endpoint;   // "<scheme>://<host>:<port>"
  std::string authority;  // "<host>:<port>" (Host header / signing)
};

s3_endpoint make_endpoint(std::string const& scheme, int port)
{
  s3_endpoint ep;
  ep.authority = "127.0.0.1:" + std::to_string(port);
  ep.endpoint  = scheme + "://" + ep.authority;
  return ep;
}

// ---- TLS cert + fixtures + identities --------------------------------------

// Generate a self-signed cert/key into dir (public.crt / private.key). Returns
// the public cert path (used both to serve HTTPS and as the test CA bundle).
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
                               "subjectAltName=IP:127.0.0.1,DNS:localhost"});
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
  int rc = run_process({"python3",
                        script.string(),
                        "--out",
                        out_dir.string(),
                        "--parquet-source",
                        parquet_src,
                        "--manifest",
                        (out_dir.parent_path() / "MANIFEST.sha256").string()});
  if (rc != 0) throw std::runtime_error("generate_fixtures.py failed");
  return out_dir;
}

void create_special_key_fixtures(fs::path const& fixture_dir)
{
  auto const nation = fixture_dir / "parquet" / "nation.parquet";
  auto const region = fixture_dir / "parquet" / "region.parquet";
  if (!fs::is_regular_file(nation) || !fs::is_regular_file(region)) {
    throw std::runtime_error("special-key glob fixtures require nation.parquet and region.parquet");
  }

  auto const root = fixture_dir / "glob-enc";
  fs::remove_all(root);
  std::array<std::pair<fs::path, fs::path>, 12> const fixtures{{
    {nation, root / "a%2Fb.parquet"},
    {region, root / "a" / "b.parquet"},
    {nation, root / "x#1.parquet"},
    {nation, root / "y?v.parquet"},
    {nation, root / "100%.parquet"},
    {nation, root / "t" / "col=a%20b" / "p0.parquet"},
    {nation, root / "q" / "col=a%3Fb" / "p0.parquet"},
    {nation, root / "q" / "col=a?b" / "p0.parquet"},
    {nation, root / "f%20g.parquet"},
    {region, root / "f g.parquet"},
    {nation, root / "guard-before" / "co?l=value" / "p0.parquet"},
    {nation, root / "guard-filename" / "report=foo?bar.parquet"},
  }};
  for (auto const& [source, destination] : fixtures) {
    fs::create_directories(destination.parent_path());
    fs::copy_file(source, destination, fs::copy_options::overwrite_existing);
  }
}

void create_edge_types_fixture(fs::path const& fixture_dir)
{
  auto const output = fixture_dir / "parquet" / "edge_types.parquet";
  auto temporary    = output;
  temporary += ".tmp";

  fs::create_directories(output.parent_path());
  std::error_code ec;
  fs::remove(temporary, ec);
  if (ec) { throw std::runtime_error("cannot remove stale edge-types fixture: " + ec.message()); }

  auto sql_quote = [](std::string_view value) {
    std::string quoted{"'"};
    for (auto const c : value) {
      if (c == '\'') quoted.push_back('\'');
      quoted.push_back(c);
    }
    quoted.push_back('\'');
    return quoted;
  };

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto result = con.Query(
    "COPY ("
    "SELECT CAST(id AS INTEGER) AS id, CAST(n AS INTEGER) AS n, "
    "CAST(d AS DECIMAL(18,4)) AS d, CAST(ts AS TIMESTAMP) AS ts "
    "FROM (VALUES "
    "(1, CAST(NULL AS INTEGER), '0.0000', '1970-01-01 00:00:00'), "
    "(2, 0, '-0.0001', '1970-01-01 00:00:00.000001'), "
    "(3, -1, '0.0001', '1985-07-13 12:34:56.123456'), "
    "(4, CAST(NULL AS INTEGER), '99999999999999.9999', '1999-12-31 23:59:59.999999'), "
    "(5, 1, '-99999999999999.9999', '2000-02-29 12:00:00'), "
    "(6, 2147483647, '12345678901234.5678', '2001-09-09 01:46:40'), "
    "(7, CAST(NULL AS INTEGER), '-12345678901234.5678', '2010-01-01 00:00:00'), "
    "(8, -2147483648, '42.4242', '2016-02-29 08:15:30'), "
    "(9, 42, '-42.4242', '2020-02-29 23:59:59'), "
    "(10, CAST(NULL AS INTEGER), '1.0000', '2024-07-24 03:01:01'), "
    "(11, -42, '-1.0000', '2038-01-19 03:14:07'), "
    "(12, 7, '9876543210.4321', '2050-06-30 10:20:30.4005'), "
    "(13, CAST(NULL AS INTEGER), '-9876543210.4321', '2099-12-31 23:59:59.999999'), "
    "(14, 8, '3141592653.5897', '2100-03-01 00:00:00'), "
    "(15, 9, '-2718281828.4590', '1969-12-31 23:59:59.999999'), "
    "(16, CAST(NULL AS INTEGER), '10000000000000.0000', '1900-01-01 00:00:00'), "
    "(17, 10, '-10000000000000.0000', '2199-12-31 23:59:59'), "
    "(18, 11, '0.0100', '2200-01-01 00:00:00'), "
    "(19, CAST(NULL AS INTEGER), '-0.0100', '2262-04-11 23:47:16'), "
    "(20, 12, '77777777777777.7777', '2200-12-31 23:59:59.123456')"
    ") AS edge(id, n, d, ts)"
    ") TO " +
    sql_quote(temporary.string()) + " (FORMAT PARQUET)");
  if (!result || result->HasError()) {
    auto const error = result ? result->GetError() : std::string{"no query result"};
    fs::remove(temporary, ec);
    throw std::runtime_error("failed to generate edge-types parquet fixture: " + error);
  }

  fs::rename(temporary, output, ec);
  if (ec) {
    auto const error = ec.message();
    fs::remove(temporary, ec);
    throw std::runtime_error("failed to finalize edge-types parquet fixture: " + error);
  }
}

// Write the SeaweedFS S3 identities file (the -s3.config payload) granting our
// single test identity full access under the static access/secret keys, so the
// SigV4-signed requests the [s3] tests issue are actually authenticated.
void write_s3_identities_config(fs::path const& path)
{
  std::ofstream out(path, std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write S3 identities config: " + path.string());
  out << "{\n"
         "  \"identities\": [\n"
         "    {\n"
         "      \"name\": \"sirius-test\",\n"
         "      \"credentials\": [{\"accessKey\": \""
      << kAccessKey << "\", \"secretKey\": \"" << kSecretKey
      << "\"}],\n"
         "      \"actions\": [\"Admin\", \"Read\", \"Write\", \"List\", \"Tagging\"]\n"
         "    }\n"
         "  ]\n"
         "}\n";
  if (!out) throw std::runtime_error("failed writing S3 identities config: " + path.string());
}

// ---- host-side SigV4 + libcurl request -------------------------------------

size_t curl_read_file(char* buffer, size_t size, size_t nitems, void* userdata)
{
  auto* f = static_cast<std::FILE*>(userdata);
  if (f == nullptr) return 0;
  return std::fread(buffer, 1, size * nitems, f);
}

size_t curl_discard(char*, size_t size, size_t nitems, void*) { return size * nitems; }

// Stream a signed PUT; a null body creates an empty bucket. Returns the HTTP
// status code, or -1 on transport error.
long s3_put(s3_endpoint const& ep,
            std::string const& canonical_uri,
            std::FILE* body,
            std::int64_t body_len,
            std::optional<fs::path> const& ca_bundle,
            long timeout_seconds = 0)
{
  sirius::io::rest::s3::sigv4_signer_config creds;
  creds.access_key = kAccessKey;
  creds.secret_key = kSecretKey;
  creds.region     = kRegion;
  creds.service    = "s3";

  // UNSIGNED-PAYLOAD lets us stream arbitrarily large bodies (e.g. the SF10
  // lineitem fixture) without hashing them; SeaweedFS accepts it.
  auto signed_req = sirius::io::rest::s3::sign_request("PUT",
                                                       ep.authority,
                                                       canonical_uri,
                                                       /*query=*/"",
                                                       "UNSIGNED-PAYLOAD",
                                                       /*extra_headers=*/{},
                                                       creds,
                                                       std::time(nullptr));

  CURL* curl = curl_easy_init();
  if (curl == nullptr) return -1;

  std::string url  = ep.endpoint + canonical_uri;
  curl_slist* hdrs = nullptr;
  for (auto const& [k, v] : signed_req.headers) {
    hdrs = curl_slist_append(hdrs, (k + ": " + v).c_str());
  }
  // Suppress libcurl's automatic "Expect: 100-continue" (unsigned, but avoids a
  // round-trip stall on PUTs).
  hdrs = curl_slist_append(hdrs, "Expect:");

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
  curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
  curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE, static_cast<curl_off_t>(body_len));
  curl_easy_setopt(curl, CURLOPT_READFUNCTION, curl_read_file);
  curl_easy_setopt(curl, CURLOPT_READDATA, body);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_discard);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 2L);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_seconds);
  curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1L);
  curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 60L);
  if (ca_bundle.has_value()) {
    curl_easy_setopt(curl, CURLOPT_CAINFO, ca_bundle->c_str());
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
  }

  long code   = -1;
  CURLcode rc = curl_easy_perform(curl);
  if (rc == CURLE_OK) {
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
  } else if (timeout_seconds == 0) {
    std::cerr << "[s3] PUT " << url << ": " << curl_easy_strerror(rc) << std::endl;
  }

  curl_slist_free_all(hdrs);
  curl_easy_cleanup(curl);
  return code;
}

std::string uri_path_for(std::string const& bucket, std::string const& key)
{
  std::string p = "/" + sirius::io::rest::s3::uri_encode(bucket, false);
  if (!key.empty()) p += "/" + sirius::io::rest::s3::uri_encode(key, false);
  return p;
}

// A non-empty PUT checks that master, filer, and volume storage are writable.
// ListBuckets alone can succeed before the volume server has registered.
bool wait_s3_ready(pid_t pid,
                   s3_endpoint const& ep,
                   fs::path const& fixture,
                   std::optional<fs::path> const& ca_bundle)
{
  std::unique_ptr<std::FILE, decltype(&std::fclose)> body(std::fopen(fixture.c_str(), "rb"),
                                                          &std::fclose);
  if (!body) throw std::runtime_error("cannot open readiness fixture");
  auto const size     = static_cast<std::int64_t>(fs::file_size(fixture));
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (std::chrono::steady_clock::now() < deadline) {
    siginfo_t info{};
    if (::waitid(P_PID, pid, &info, WEXITED | WNOHANG | WNOWAIT) == 0 && info.si_pid == pid) {
      return false;
    }
    auto const bucket = s3_put(ep, uri_path_for(kBucket, ""), nullptr, 0, ca_bundle, 2);
    if (bucket == 200 || bucket == 204 || bucket == 409) {
      std::rewind(body.get());
      auto const code =
        s3_put(ep, uri_path_for(kBucket, kDefaultKey), body.get(), size, ca_bundle, 2);
      if (code == 200 || code == 204) return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  return false;
}

void upload_fixtures(s3_endpoint const& ep, fs::path const& fixture_dir)
{
  for (auto const& entry : fs::recursive_directory_iterator(fixture_dir)) {
    if (!entry.is_regular_file()) continue;
    std::string key = fs::relative(entry.path(), fixture_dir).generic_string();

    // Size first, so a throwing file_size() never leaks the FILE* below.
    auto size    = static_cast<std::int64_t>(entry.file_size());
    std::FILE* f = std::fopen(entry.path().c_str(), "rb");
    if (f == nullptr) throw std::runtime_error("cannot open fixture: " + entry.path().string());
    long pc = s3_put(ep, uri_path_for(kBucket, key), f, size, std::nullopt);
    std::fclose(f);
    if (!(pc == 200 || pc == 204)) {
      throw std::runtime_error("put-object PUT failed for '" + key + "' (HTTP " +
                               std::to_string(pc) + ") at " + ep.endpoint);
    }
  }
}

// Opt-in (SIRIUS_TEST_S3_LARGE=1): generate the SF10 lineitem parquet with the
// DuckDB CLI and upload it for the [s3][sql][large] tests. Both endpoints share
// the uploaded object.
void maybe_upload_large_fixture(s3_endpoint const& http, fs::path const& work)
{
  if (!env_truthy("SIRIUS_TEST_S3_LARGE")) return;

  // The SF10 generation costs minutes, so cache the parquet at a stable path and
  // reuse it across the two processes that make s3-test-large runs. Only the
  // first generates.
  fixture_cache_lock lock(work / "lineitem_sf10.lock");
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
                   uri_path_for(kBucket, key),
                   f,
                   static_cast<std::int64_t>(fs::file_size(parquet)),
                   std::nullopt);
  std::fclose(f);
  if (!(pc == 200 || pc == 204)) {
    throw std::runtime_error("SF10 upload failed (HTTP " + std::to_string(pc) + ")");
  }
  // The large tests read the same SF10 file locally to build a CPU oracle to
  // compare the GPU-over-S3 result against. Point them at the file we just
  // uploaded so the local copy and the S3 object are byte-identical. (Respect an
  // explicit override.)
  if (!env_set("SIRIUS_PR6_LARGE_LOCAL_PARQUET")) {
    ::setenv("SIRIUS_PR6_LARGE_LOCAL_PARQUET", parquet.c_str(), /*overwrite=*/1);
  }
}

void maybe_upload_tpch_sf1_fixture(s3_endpoint const& http, fs::path const& work)
{
  if (!env_truthy("SIRIUS_TEST_S3_TPCH") && !env_truthy("SIRIUS_BENCH_S3_TPCH")) return;

  constexpr std::array<std::string_view, 8> tables = {
    "nation", "region", "customer", "orders", "part", "partsupp", "supplier", "lineitem"};
  fixture_cache_lock lock(work / "tpch_sf1.lock");
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

    std::FILE* file = std::fopen(parquet.c_str(), "rb");
    if (file == nullptr) {
      throw std::runtime_error("cannot open SF1 TPC-H fixture: " + parquet.string());
    }
    auto const code = s3_put(http, uri_path_for(kBucket, key), file, bytes, std::nullopt);
    std::fclose(file);
    if (!(code == 200 || code == 204)) {
      throw std::runtime_error("SF1 TPC-H upload failed for '" + key + "' (HTTP " +
                               std::to_string(code) + ") at " + http.endpoint);
    }
  }

  setenv("SIRIUS_TEST_S3_TPCH_LOCAL_DIR", fixture_dir.c_str(), /*overwrite=*/1);
  std::cout << "[s3] uploaded 8 SF1 TPC-H parquet fixtures (" << total_bytes << " bytes) to "
            << http.endpoint << std::endl;
}

void maybe_upload_glob_scale_fixture(s3_endpoint const& http, fs::path const& fixture_dir)
{
  if (!env_truthy("SIRIUS_TEST_S3_GLOB_SCALE")) return;

  constexpr std::size_t object_count = 1001;
  auto const parquet                 = fixture_dir / "parquet" / "nation.parquet";
  auto const parquet_size            = static_cast<std::int64_t>(fs::file_size(parquet));

  for (std::size_t index = 0; index < object_count; ++index) {
    auto const key = "glob-scale/part_" + std::to_string(index) + ".parquet";
    std::FILE* f   = std::fopen(parquet.c_str(), "rb");
    if (f == nullptr) throw std::runtime_error("cannot open glob-scale parquet fixture");
    auto const code = s3_put(http, uri_path_for(kBucket, key), f, parquet_size, std::nullopt);
    std::fclose(f);
    if (!(code == 200 || code == 204)) {
      throw std::runtime_error("glob-scale upload failed for '" + key + "' (HTTP " +
                               std::to_string(code) + ")");
    }
  }

  std::cout << "[s3] uploaded " << object_count << " parquet objects under " << http.endpoint << "/"
            << kBucket << "/glob-scale/" << std::endl;
}

// ---- weed server lifecycle -------------------------------------------------

pid_t g_weed_pid{0};  // server to terminate at shutdown
fs::path g_run_dir;   // this process's run dir, removed at shutdown

struct weed_server {
  pid_t pid{0};
  s3_endpoint http;
  s3_endpoint https;
};

// Spawn a single `weed server` exposing the S3 API over both HTTP and HTTPS on
// dynamically-chosen free ports. All cluster services (master/volume/filer/s3 +
// their grpc variants) get distinct free ports; the fixed-port Iceberg catalog
// is disabled to avoid collisions across concurrent runs.
weed_server start_weed(fs::path const& data_dir,
                       fs::path const& s3_config,
                       fs::path const& cert,
                       fs::path const& key,
                       fs::path const& log_path)
{
  auto p          = reserve_free_ports(9);
  int master_port = p[0], master_grpc = p[1];
  int volume_port = p[2], volume_grpc = p[3];
  int filer_port = p[4], filer_grpc = p[5];
  int s3_http = p[6], s3_grpc = p[7], s3_https = p[8];

  auto flag = [](char const* name, int value) {
    return std::string{name} + "=" + std::to_string(value);
  };
  std::string weed = env_or("SIRIUS_TEST_WEED", "weed");

  std::vector<std::string> argv = {weed,
                                   "server",
                                   "-ip=127.0.0.1",
                                   "-dir=" + data_dir.string(),
                                   "-filer",
                                   "-s3",
                                   flag("-master.port", master_port),
                                   flag("-master.port.grpc", master_grpc),
                                   flag("-volume.port", volume_port),
                                   flag("-volume.port.grpc", volume_grpc),
                                   flag("-filer.port", filer_port),
                                   flag("-filer.port.grpc", filer_grpc),
                                   flag("-s3.port", s3_http),
                                   flag("-s3.port.grpc", s3_grpc),
                                   flag("-s3.port.https", s3_https),
                                   "-s3.port.iceberg=0",
                                   "-s3.port.lance=0",
                                   "-s3.config=" + s3_config.string(),
                                   "-s3.cert.file=" + cert.string(),
                                   "-s3.key.file=" + key.string()};

  pid_t pid  = spawn_process(argv, log_path);
  g_weed_pid = pid;

  weed_server srv;
  srv.pid   = pid;
  srv.http  = make_endpoint("http", s3_http);
  srv.https = make_endpoint("https", s3_https);
  return srv;
}

// ---- orchestration ---------------------------------------------------------

void setenv_kv(char const* k, std::string const& v) { ::setenv(k, v.c_str(), /*overwrite=*/1); }

// Returns true if it brought everything up and published the env successfully.
bool bring_up()
{
  curl_global_init(CURL_GLOBAL_DEFAULT);

  // `base` holds the shared SF1 and SF10 fixture caches. Everything mutable for
  // this run — certs, fixtures, the `weed` data dir, its config and log — lives
  // under a per-process `run` dir so concurrent S3 test processes never stomp
  // each other's filer state. The run dir is removed at shutdown.
  fs::path base = fs::temp_directory_path() / ("sirius-s3-seaweedfs-" + std::to_string(::getuid()));
  fs::create_directories(base);
  std::string run_template = (base / "run-XXXXXX").string();
  if (::mkdtemp(run_template.data()) == nullptr) {
    throw std::runtime_error("cannot create SeaweedFS run directory");
  }
  fs::path run         = run_template;
  g_run_dir            = run;
  fs::path certs_dir   = run / "certs";
  fs::path fixture_dir = run / "fixtures" / "local";
  fs::path data_dir    = run / "data";
  fs::path s3_config   = run / "s3_identities.json";
  fs::path weed_log    = run / "weed.log";

  fs::create_directories(data_dir);

  fs::path ca_bundle = generate_self_signed_cert(certs_dir);
  generate_fixtures(fixture_dir);
  create_special_key_fixtures(fixture_dir);
  create_edge_types_fixture(fixture_dir);
  write_s3_identities_config(s3_config);

  weed_server srv =
    start_weed(data_dir, s3_config, certs_dir / "public.crt", certs_dir / "private.key", weed_log);

  std::optional<fs::path> ca = ca_bundle;
  if (!wait_s3_ready(srv.pid, srv.http, fixture_dir / kDefaultKey, std::nullopt)) {
    throw std::runtime_error("SeaweedFS S3 (HTTP) did not become ready at " + srv.http.endpoint);
  }
  if (!wait_s3_ready(srv.pid, srv.https, fixture_dir / kDefaultKey, ca)) {
    throw std::runtime_error("SeaweedFS S3 (HTTPS) did not become ready at " + srv.https.endpoint);
  }

  // One backend serves both endpoints, so a single upload covers HTTP and HTTPS.
  upload_fixtures(srv.http, fixture_dir);
  maybe_upload_large_fixture(srv.http, base);
  maybe_upload_tpch_sf1_fixture(srv.http, base);
  maybe_upload_glob_scale_fixture(srv.http, fixture_dir);

  // Publish the env contract the [s3] tests consume.
  setenv_kv("SIRIUS_TEST_S3_ENDPOINT", srv.http.endpoint);
  setenv_kv("SIRIUS_TEST_S3_HTTPS_ENDPOINT", srv.https.endpoint);
  setenv_kv("SIRIUS_TEST_S3_REGION", kRegion);
  setenv_kv("SIRIUS_TEST_S3_ACCESS_KEY", kAccessKey);
  setenv_kv("SIRIUS_TEST_S3_SECRET_KEY", kSecretKey);
  setenv_kv("SIRIUS_TEST_S3_BUCKET", kBucket);
  setenv_kv("SIRIUS_TEST_S3_KEY", kDefaultKey);
  setenv_kv("SIRIUS_TEST_S3_LOCAL_DIR", fixture_dir.string());
  setenv_kv("SIRIUS_TEST_S3_CA_BUNDLE", ca_bundle.string());

  std::cout << "[s3] SeaweedFS ready: " << srv.http.endpoint << " (http), " << srv.https.endpoint
            << " (https); fixtures in " << fixture_dir << std::endl;
  return true;
}

// Bring-up state, resolved once: 0 = untried, 1 = ready, 2 = skip,
// 3 = failed under strict mode (every [s3] test should fail loudly).
std::mutex g_state_mutex;
int g_state = 0;

}  // namespace

bool ensure_s3_test_env()
{
  // Externally-provided endpoint (manual run / real AWS): use as-is.
  if (env_set("SIRIUS_TEST_S3_ENDPOINT")) return true;

  std::lock_guard<std::mutex> lk(g_state_mutex);  // Catch2 is single-threaded; defensive.
  switch (g_state) {
    case 1: return true;
    case 2: return false;
    case 3: throw std::runtime_error("[s3] SeaweedFS bring-up previously failed (strict mode)");
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
    std::cerr << "[s3] SeaweedFS bring-up failed: " << e.what() << std::endl;
    if (!g_run_dir.empty()) {
      std::ifstream log(g_run_dir / "weed.log");
      if (log) std::cerr << log.rdbuf();
    }
    shutdown_s3_test_env();
    if (env_truthy("SIRIUS_TEST_S3_STRICT")) {
      g_state = 3;
      throw;
    }
    g_state = 2;  // best-effort: skip
    return false;
  }
}

bool put_s3_test_object(std::string_view key, std::span<std::uint8_t const> bytes)
{
  if (g_state != 1) { return false; }

  auto endpoint         = env_or("SIRIUS_TEST_S3_ENDPOINT");
  auto const scheme_end = endpoint.find("://");
  if (scheme_end == std::string::npos) {
    throw std::runtime_error("managed SeaweedFS endpoint has no URI scheme");
  }
  auto const scheme = endpoint.substr(0, scheme_end);
  auto authority    = endpoint.substr(scheme_end + 3);
  if (scheme != "http" || authority.empty() || authority.find('/') != std::string::npos) {
    throw std::runtime_error("managed SeaweedFS object PUT requires the HTTP endpoint");
  }

  std::unique_ptr<std::FILE, decltype(&std::fclose)> body(std::tmpfile(), &std::fclose);
  if (!body) { throw std::runtime_error("cannot create temporary S3 upload body"); }
  auto const written = std::fwrite(bytes.data(), 1, bytes.size(), body.get());
  if (written != bytes.size()) {
    throw std::runtime_error("cannot write temporary S3 upload body");
  }
  std::rewind(body.get());

  s3_endpoint instance;
  instance.endpoint  = std::move(endpoint);
  instance.authority = std::move(authority);
  auto const bucket  = env_or("SIRIUS_TEST_S3_BUCKET", kBucket);
  auto const code    = s3_put(instance,
                           uri_path_for(bucket, std::string{key}),
                           body.get(),
                           static_cast<std::int64_t>(bytes.size()),
                           std::nullopt);
  if (!(code == 200 || code == 204)) {
    throw std::runtime_error("managed SeaweedFS object PUT failed for '" + std::string{key} +
                             "' (HTTP " + std::to_string(code) + ")");
  }
  return true;
}

void shutdown_s3_test_env()
{
  terminate_pid(g_weed_pid);
  g_weed_pid = 0;

  // Remove this run's mutable state (the shared fixture caches are left
  // intact for reuse). Done after the servers are gone so nothing is in use.
  if (!g_run_dir.empty()) {
    std::error_code ec;
    fs::remove_all(g_run_dir, ec);
    g_run_dir.clear();
  }
}

}  // namespace sirius::test
