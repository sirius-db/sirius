// SPDX-License-Identifier: Apache-2.0
// Test-only serial REST workload. Never use production credentials with this probe.
#include "io/io_errors.hpp"
#include "io/rest/rest_ioctx.hpp"
#include "io/rest/s3/sigv4_authorizer.hpp"
#include "io/sirius_datasource.hpp"
#include "log/sink.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace {
struct fixture_mismatch : std::runtime_error {
  using std::runtime_error::runtime_error;
};
std::string quote(std::string_view s)
{
  std::string out = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += static_cast<char>(c);
    } else if (c < 32) {
      constexpr char hex[] = "0123456789abcdef";
      out += "\\u00";
      out += hex[c >> 4];
      out += hex[c & 15];
    } else {
      out += static_cast<char>(c);
    }
  }
  return out + "\"";
}
class recorder final : public sirius::log::sink {
 public:
  explicit recorder(std::string const& path) : stream(path)
  {
    stream.exceptions(std::ios::badbit | std::ios::failbit);
  }
  void set_level(sirius::log::level) override {}
  bool should_log(sirius::log::level level) const override
  {
    return level >= sirius::log::level::info;
  }
  void log(sirius::log::level level, std::source_location const&, std::string_view message) override
  {
    if (should_log(level)) {
      emit("\"kind\":\"log\",\"op\":" + std::to_string(op.load()) +
           ",\"message\":" + quote(message));
    }
  }
  bool flush() override
  {
    std::lock_guard lock(mutex);
    stream.flush();
    return true;
  }
  void emit(std::string const& fields)
  {
    std::lock_guard lock(mutex);
    stream << "{\"schema\":2," << fields << "}\n";
    stream.flush();
  }
  std::atomic<int> op{-1};

 private:
  std::mutex mutex;
  std::ofstream stream;
};
}  // namespace
int main(int argc, char** argv)
{
  std::shared_ptr<recorder> records;
  std::shared_ptr<sirius::io::rest::rest_ioctx> ctx;
  std::string stage = "setup";
  std::string object;
  try {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 >= argc || !args.emplace(argv[i], argv[i + 1]).second) {
        throw std::runtime_error("expected unique option/value pairs");
      }
    }
    for (auto const& [key, value] : args) {
      if (key != "--endpoint" && key != "--bucket" && key != "--key" && key != "--fixture" &&
          key != "--out" && key != "--target" && key != "--reuse" && key != "--retention" &&
          key != "--connections" && key != "--workers" && key != "--reads" &&
          key != "--footer-first" && key != "--request-log") {
        throw std::runtime_error("unknown option: " + key);
      }
    }
    auto endpoint = args.at("--endpoint");
    if (!endpoint.starts_with("http://127.0.0.1:") ||
        endpoint.find_first_of("/?#", 7) != std::string::npos) {
      throw std::runtime_error("probe requires loopback endpoint without path");
    }
    auto reuse     = args.contains("--reuse") ? args.at("--reuse") : "off";
    auto retention = args.contains("--retention") ? args.at("--retention") : "expanded";
    if ((reuse != "off" && reuse != "same_open") ||
        (retention != "legacy" && retention != "expanded")) {
      throw std::runtime_error("invalid policy mode");
    }
    auto number = [&](char const* key, unsigned fallback) {
      if (!args.contains(key)) { return fallback; }
      auto const& value = args.at(key);
      unsigned result{};
      auto const parsed = std::from_chars(value.data(), value.data() + value.size(), result);
      if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw std::runtime_error("invalid unsigned option");
      }
      return result;
    };
    auto connections = number("--connections", 1);
    auto workers     = number("--workers", 1);
    auto reads       = number("--reads", 4);
    if (connections < 1 || connections > 64 || workers != 1 || reads < 1 || reads > 256) {
      throw std::runtime_error("serial probe requires worker=1, C=1..64, reads=1..256");
    }
    for (auto name : {"--footer-first", "--request-log"}) {
      if (args.contains(name) && args.at(name) != "true" && args.at(name) != "false") {
        throw std::runtime_error("invalid boolean");
      }
    }
    if (args.contains("--target") &&
        (!args.at("--target").starts_with("127.0.0.1:") ||
         args.at("--target").find_first_of("/?#@") != std::string::npos)) {
      throw std::runtime_error("target must be loopback authority");
    }
    bool footer  = args.contains("--footer-first") && args.at("--footer-first") == "true";
    bool logging = !args.contains("--request-log") || args.at("--request-log") == "true";
    std::ifstream input(args.at("--fixture"), std::ios::binary);
    if (!input) { throw std::runtime_error("fixture missing"); }
    std::vector<std::uint8_t> expected{std::istreambuf_iterator<char>(input), {}};
    if (expected.size() < 64 || expected.size() > (4U << 20)) {
      throw std::runtime_error("fixture must contain 64 bytes..4 MiB");
    }
    records = std::make_shared<recorder>(args.at("--out"));
    sirius::log::set_sink(records);
    sirius::io::rest::config cfg;
    cfg.max_connections          = connections;
    cfg.request_timeout_s        = 5;
    cfg.stall_time_s             = 2;
    cfg.max_retry_attempts       = 2;
    cfg.max_auth_retry_attempts  = 1;
    cfg.retry_backoff_base       = std::chrono::milliseconds{1};
    cfg.retry_jitter             = std::chrono::milliseconds{0};
    cfg.honor_retry_after        = false;
    cfg.request_log              = logging;
    cfg.cluster_redirect.enabled = args.contains("--target");
    cfg.cluster_redirect.origin  = endpoint.substr(7);
    if (cfg.cluster_redirect.enabled) {
      cfg.cluster_redirect.allowed_targets = {args.at("--target")};
    }
    cfg.cluster_redirect.retention   = retention == "legacy"
                                         ? sirius::io::rest::redirect_retention::legacy
                                         : sirius::io::rest::redirect_retention::expanded;
    cfg.cluster_redirect.route_reuse = reuse == "off"
                                         ? sirius::io::rest::redirect_route_reuse::off
                                         : sirius::io::rest::redirect_route_reuse::same_open;
    records->emit(
      "\"kind\":\"config\",\"client\":\"sirius-rest-probe-v2\",\"endpoint\":" + quote(endpoint) +
      ",\"bucket\":" + quote(args.at("--bucket")) + ",\"key\":" + quote(args.at("--key")) +
      ",\"fixture\":" + quote(args.at("--fixture")) + ",\"out\":" + quote(args.at("--out")) +
      ",\"fixture_size\":" + std::to_string(expected.size()) +
      ",\"target\":" + quote(args.contains("--target") ? args.at("--target") : "") +
      ",\"reuse\":" + quote(reuse) + ",\"retention\":" + quote(retention) +
      ",\"workers\":1,\"connections\":" + std::to_string(connections) +
      ",\"reads\":" + std::to_string(reads) + ",\"footer_first\":" + (footer ? "true" : "false") +
      ",\"request_log\":" + (logging ? "true" : "false"));
    sirius::io::rest::s3::static_credentials credentials{
      "cucascade-local-test", "cucascade-local-test-secret", "", std::nullopt};
    auto auth = std::make_shared<sirius::io::rest::s3::sigv4_header_authorizer>(
      credentials, "us-east-1", endpoint);
    auto context =
      std::make_shared<sirius::io::rest::rest_reactor::reactor_context>(cfg, auth, nullptr);
    ctx = std::make_shared<sirius::io::rest::rest_ioctx>(workers, context);
    ctx->start();
    object      = args.at("--bucket") + "/" + args.at("--key");
    stage       = "open";
    records->op = 0;
    records->emit("\"kind\":\"begin\",\"op\":0,\"action\":\"open\"");
    auto path   = "s3://" + args.at("--bucket") + "/" + args.at("--key");
    auto source = ctx->open_datasource(path);
    if (source->size() != expected.size()) { throw std::runtime_error("object size mismatch"); }
    records->emit("\"kind\":\"end\",\"op\":0,\"status\":\"PASS\"");
    for (std::size_t i = 0; i < reads; ++i) {
      records->op = static_cast<int>(i + 1);
      auto length = std::min<std::size_t>(65536, expected.size());
      auto offset = (footer && i == 0) ? expected.size() - length
                                       : ((i * 7919) % (expected.size() - length + 1));
      records->emit("\"kind\":\"begin\",\"op\":" + std::to_string(i + 1) +
                    ",\"action\":\"read\",\"offset\":" + std::to_string(offset) +
                    ",\"length\":" + std::to_string(length));
      std::vector<std::uint8_t> got(length);
      stage      = "read";
      auto count = source->host_read(offset, length, got.data());
      stage      = "verify_bytes";
      if (count != length || !std::equal(got.begin(), got.end(), expected.begin() + offset)) {
        throw fixture_mismatch("fixture bytes mismatch");
      }
      records->emit("\"kind\":\"end\",\"op\":" + std::to_string(i + 1) +
                    ",\"status\":\"PASS\",\"bytes\":" + std::to_string(count));
    }
    stage = "shutdown";
    ctx->shutdown();
    records->emit("\"kind\":\"complete\",\"status\":\"PASS\",\"operations\":" +
                  std::to_string(reads + 1));
    sirius::log::set_sink(nullptr);
    return 0;
  } catch (std::exception const& error) {
    // The engine currently uses runtime_error for HTTP refusals. Match its
    // exact message plus object; unknown errors never become expected faults.
    std::string code = "unexpected_error";
    if (dynamic_cast<sirius::io::object_changed_error const*>(&error)) {
      code = "object_changed";
    } else if (dynamic_cast<fixture_mismatch const*>(&error)) {
      code = "fixture_mismatch";
    } else if (error.what() ==
               "rest_reactor: HTTP 307 (the storage node redirected again) for " + object) {
      code = "second_redirect";
    } else if (error.what() == "rest_reactor: HTTP 403 from the storage node for " + object) {
      code = "delegated_forbidden";
    }
    if (ctx) { ctx->shutdown(); }
    if (records) {
      auto fields = "\"op\":" + std::to_string(records->op.load()) +
                    ",\"status\":\"FAIL\",\"stage\":" + quote(stage) +
                    ",\"error_code\":" + quote(code) + ",\"error\":" + quote(error.what());
      records->emit("\"kind\":\"end\"," + fields);
      records->emit("\"kind\":\"complete\"," + fields);
    }
    std::cerr << "probe: " << error.what() << '\n';
    sirius::log::set_sink(nullptr);
    return 1;
  }
}
