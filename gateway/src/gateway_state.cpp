#include "gateway_state.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/parser/keyword_helper.hpp"
#include "duckdb/parser/parsed_data/create_info.hpp"
#include "duckdb/parser/parsed_expression_iterator.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/attach_statement.hpp"
#include "duckdb/parser/statement/create_statement.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/statement/set_statement.hpp"
#include "duckdb/parser/tableref/basetableref.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"

#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <random>
#include <regex>
#include <set>
#include <string_view>
#include <thread>

namespace sirius_gateway {

using duckdb::ClientContext;
using duckdb::Connection;
using duckdb::InvalidInputException;
using duckdb::IOException;
using duckdb::KeywordHelper;
using std::chrono::steady_clock;
using namespace std::chrono_literals;

namespace {

std::string random_token()
{
  std::random_device device;
  std::string token;
  for (int i = 0; i < 4; ++i) {
    token += duckdb::StringUtil::Format("%08x", static_cast<uint32_t>(device()));
  }
  return token;
}

std::string setting(ClientContext& context, char const* name, std::string fallback)
{
  duckdb::Value value;
  if (context.TryGetCurrentSetting(name, value) && !value.IsNull()) { return value.ToString(); }
  return fallback;
}

/// Words in setting and option names that hold credentials.
constexpr std::array<char const*, 7> credential_words{
  "secret", "token", "password", "key_id", "access_key", "connection_string", "credential"};

/// Statements whose effect a fresh rank must see again: everything but reads.
bool changes_state(std::string const& sql)
{
  duckdb::Parser parser{duckdb::ParserOptions::Builtin()};
  try {
    parser.ParseQuery(sql);
  } catch (std::exception&) {
    return false;
  }
  return std::any_of(parser.statements.begin(), parser.statements.end(), [](auto const& stmt) {
    switch (stmt->type) {
      case duckdb::StatementType::SELECT_STATEMENT:
      case duckdb::StatementType::EXPLAIN_STATEMENT:
      case duckdb::StatementType::TRANSACTION_STATEMENT: return false;
      default: return true;
    }
  });
}

/// Whether @p sql carries credentials (CREATE SECRET, a secret-like setting, an ATTACH token),
/// which the gateway keeps out of the DDL log file.
bool holds_credentials(std::string const& sql)
{
  duckdb::Parser parser{duckdb::ParserOptions::Builtin()};
  try {
    parser.ParseQuery(sql);
  } catch (std::exception&) {
    return false;
  }
  auto const secret_like = [](std::string name) {
    name = duckdb::StringUtil::Lower(name);
    return std::any_of(std::begin(credential_words), std::end(credential_words), [&](auto word) {
      return name.find(word) != std::string::npos;
    });
  };
  return std::any_of(parser.statements.begin(), parser.statements.end(), [&](auto const& stmt) {
    switch (stmt->type) {
      case duckdb::StatementType::CREATE_STATEMENT:
        return stmt->template Cast<duckdb::CreateStatement>().info->type ==
               duckdb::CatalogType::SECRET_ENTRY;
      case duckdb::StatementType::SET_STATEMENT:
        return secret_like(stmt->template Cast<duckdb::SetStatement>().name.GetIdentifierName());
      case duckdb::StatementType::ATTACH_STATEMENT: {
        auto const& info = *stmt->template Cast<duckdb::AttachStatement>().info;
        return std::any_of(info.parsed_options.begin(),
                           info.parsed_options.end(),
                           [&](auto const& option) { return secret_like(option.first); });
      }
      default: return false;
    }
  });
}

/// The DDL log file: `<sirius_gateway_state_dir>/ddl.sql`, or empty when that is not set.
/// Each entry follows a line "-- entry <bytes>", so multi-line statements read back exactly, and
/// ends with ";", so the file also runs as a script.
std::filesystem::path ddl_file(ClientContext& context)
{
  auto const dir = setting(context, "sirius_gateway_state_dir", "");
  return dir.empty() ? std::filesystem::path{} : std::filesystem::path{dir} / "ddl.sql";
}

constexpr std::string_view entry_mark = "-- entry ";

/// Replaces @p file with @p log, less the entries that carry credentials: written to a new
/// file, mode 600, then renamed, so a crash leaves the old log or the new one.
void save_ddl_log(std::filesystem::path const& file, std::vector<std::string> const& log)
{
  std::string text =
    "-- The sirius_gateway DDL log, replayed on every start of the ranks. Statements with "
    "credentials are not saved.\n";
  for (auto const& entry : log) {
    if (holds_credentials(entry)) { continue; }
    text += std::string{entry_mark} + std::to_string(entry.size()) + "\n" + entry + ";\n";
  }
  if (std::filesystem::create_directories(file.parent_path())) {
    std::filesystem::permissions(file.parent_path(), std::filesystem::perms::owner_all);
  }
  auto const temp = file.string() + ".tmp";
  int const fd    = open(temp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (fd < 0) { throw IOException("cannot write %s: %s", temp, std::strerror(errno)); }
  bool written = fchmod(fd, 0600) == 0;
  for (std::size_t at = 0; written && at < text.size();) {
    auto const n = write(fd, text.data() + at, text.size() - at);
    written      = n > 0;
    at += written ? static_cast<std::size_t>(n) : 0;
  }
  written = fsync(fd) == 0 && written;
  close(fd);
  if (!written || rename(temp.c_str(), file.c_str()) != 0) {
    auto const error = errno;
    unlink(temp.c_str());
    throw IOException("cannot write %s: %s", file.string(), std::strerror(error));
  }
}

std::vector<std::string> load_ddl_log(std::filesystem::path const& file)
{
  std::ifstream in(file, std::ios::binary);
  std::string const text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  std::vector<std::string> log;
  for (std::size_t at = 0; at < text.size();) {
    auto const end = std::min(text.find('\n', at), text.size());
    std::string_view const line{text.data() + at, end - at};
    at = end + 1;
    if (line.substr(0, entry_mark.size()) != entry_mark) { continue; }
    std::size_t bytes = 0;
    try {
      bytes = std::stoull(std::string{line.substr(entry_mark.size())});
    } catch (std::exception&) {
      throw IOException("%s: bad entry line \"%s\"", file.string(), std::string{line});
    }
    if (at + bytes + 2 > text.size() || text.compare(at + bytes, 2, ";\n") != 0) {
      throw IOException("%s: entry %d is cut short", file.string(), log.size() + 1);
    }
    log.push_back(text.substr(at, bytes));
    at += bytes + 2;
  }
  return log;
}

/// Saves @p log to @p file, if set. Its statements already ran on every rank, so a failure to
/// save is reported, not thrown: returns the error, also written to the gateway's stderr.
std::string try_save_ddl_log(std::filesystem::path const& file, std::vector<std::string> const& log)
{
  if (file.empty()) { return {}; }
  try {
    save_ddl_log(file, log);
    return {};
  } catch (std::exception& e) {
    auto const error = "cannot save the DDL log: " + duckdb::ErrorData(e).RawMessage();
    std::fprintf(stderr, "sirius_gateway: %s\n", error.c_str());
    return error;
  }
}

/// Whether @p sql only reads the catalog: SELECTs whose every table, at any depth, is a
/// `duckdb_*()`, `pragma_*()` or `sirius_rank_info()` table function or one of the statement's
/// top-level CTEs. Quack sends such reads on every client's ATTACH. @p touches is set when one of
/// those functions is read, which `SELECT 1` does not.
bool reads_only_catalog(std::string const& sql, bool* touches = nullptr)
{
  using namespace duckdb;
  Parser parser{ParserOptions::Builtin()};
  try {
    parser.ParseQuery(sql);
  } catch (std::exception&) {
    return false;
  }
  if (parser.statements.empty()) { return false; }
  for (auto& stmt : parser.statements) {
    if (stmt->type != StatementType::SELECT_STATEMENT) { return false; }
    auto& node = *stmt->Cast<SelectStatement>().node;
    std::set<std::string> ctes;
    for (auto const& kv : node.cte_map.map) {
      ctes.insert(StringUtil::Lower(kv.first.GetIdentifierName()));
    }
    bool only                             = true;
    std::function<void(TableRef&)> on_ref = [&](TableRef& ref) {
      switch (ref.type) {
        case TableReferenceType::BASE_TABLE:
          only =
            only &&
            ctes.count(StringUtil::Lower(ref.Cast<BaseTableRef>().Table().GetIdentifierName())) > 0;
          break;
        case TableReferenceType::TABLE_FUNCTION: {
          auto& f = *ref.Cast<TableFunctionRef>().function;
          auto const name =
            f.GetExpressionClass() == ExpressionClass::FUNCTION
              ? StringUtil::Lower(f.Cast<FunctionExpression>().FunctionName().GetIdentifierName())
              : std::string{};
          bool const catalog = StringUtil::StartsWith(name, "duckdb_") ||
                               StringUtil::StartsWith(name, "pragma_") ||
                               name == "sirius_rank_info";
          only = only && catalog;
          if (catalog && touches) { *touches = true; }
          break;
        }
        case TableReferenceType::JOIN:
        case TableReferenceType::SUBQUERY:
        case TableReferenceType::EMPTY_FROM:
        case TableReferenceType::EXPRESSION_LIST: break;
        default: only = false;
      }
    };
    std::function<void(unique_ptr<ParsedExpression>&)> on_expr =
      [&](unique_ptr<ParsedExpression>& e) {
        if (e->GetExpressionClass() == ExpressionClass::SUBQUERY) {
          ParsedExpressionIterator::EnumerateQueryNodeChildren(
            *e->Cast<SubqueryExpression>().SubqueryMutable()->node, on_expr, on_ref);
        }
        ParsedExpressionIterator::EnumerateChildren(*e, on_expr);
      };
    try {
      ParsedExpressionIterator::EnumerateQueryNodeChildren(node, on_expr, on_ref);
    } catch (std::exception&) {
      return false;
    }
    if (!only) { return false; }
  }
  return true;
}

std::shared_ptr<rank_result> collect(duckdb::QueryResult& query)
{
  auto result   = std::make_shared<rank_result>();
  result->types = query.GetTypes();
  for (auto const& name : query.GetNames()) {
    result->names.push_back(name.GetIdentifierName());
  }
  while (auto chunk = query.Fetch()) {
    if (chunk->size() > 0) { result->chunks.push_back(std::move(chunk)); }
  }
  return result;
}

/// The rank another rank's error names as the one that failed first, if any: peers see a rank's
/// failure as "exchange: query K failed on rank R: ..." or "exchange: rank R failed: ...".
std::optional<uint32_t> blamed_rank(std::string const& message)
{
  static std::regex const query_failed{R"(failed on rank (\d+))"};
  static std::regex const rank_failed{R"(exchange: rank (\d+) failed)"};
  std::smatch match;
  if (std::regex_search(message, match, query_failed) ||
      std::regex_search(message, match, rank_failed)) {
    return static_cast<uint32_t>(std::stoul(match[1].str()));
  }
  return std::nullopt;
}

/// The error to report, prefixed with its rank: following each error's "failed on rank R" to
/// the rank whose own error does not blame another one. @p errors has one entry per rank.
duckdb::ErrorData root_cause(std::vector<std::optional<duckdb::ErrorData>> const& errors)
{
  uint32_t at = 0;
  while (at < errors.size() && !errors[at]) {
    ++at;
  }
  for (std::size_t hops = 0; hops < errors.size(); ++hops) {
    auto const blamed = blamed_rank(errors[at]->RawMessage());
    if (!blamed || *blamed == at || *blamed >= errors.size() || !errors[*blamed]) { break; }
    at = *blamed;
  }
  return duckdb::ErrorData(errors[at]->Type(),
                           "rank " + std::to_string(at) + ": " + errors[at]->RawMessage());
}

/// Whether @p message reports that the exchange itself failed on a rank: its transport is gone
/// for every later query, so the ranks must restart.
bool transport_failed(std::string const& message)
{
  static constexpr std::array<char const*, 4> markers{
    "exchange: the transport to the other ranks failed",
    "exchange: no progress for",
    "exchange: a write to a peer failed",
    "exchange: shut down with the Sirius context"};
  static std::regex const rank_failed{R"(exchange: rank \d+ failed:)"};
  return std::any_of(
           markers.begin(),
           markers.end(),
           [&](char const* marker) { return message.find(marker) != std::string::npos; }) ||
         std::regex_search(message, rank_failed);
}

/// Whether @p message says the gateway could not reach the rank at all: its process or its
/// forwarded port is gone, so the rank is lost.
bool rank_unreachable(std::string const& message)
{
  return message.find("Failed to send message") != std::string::npos ||
         message.find("Invalid connection id") != std::string::npos ||
         message.find("did not answer; abandoned") != std::string::npos;
}

/// One fan-out's calls, shared with the threads that run them so a call the gateway gives up on
/// can finish on its own.
struct fan_out_shared {
  struct call {
    std::shared_ptr<Connection> channel;
    std::vector<std::string> statements;
    std::unique_ptr<duckdb::QueryResult> result;
    std::optional<duckdb::ErrorData> error;
    bool done = false;  // under mutex
  };
  explicit fan_out_shared(std::size_t ranks) : calls(ranks), remaining(ranks) {}

  std::vector<call> calls;
  /// Set once the statement is cancelled: no rank starts another statement of its list.
  std::atomic<bool> cancel{false};
  std::mutex mutex;
  std::condition_variable changed;
  std::size_t remaining;
};

void run_call(fan_out_shared& shared, std::size_t index)
{
  auto& call = shared.calls[index];
  std::unique_ptr<duckdb::QueryResult> result;
  std::optional<duckdb::ErrorData> error;
  try {
    for (auto const& sql : call.statements) {
      if (shared.cancel) {
        error = duckdb::ErrorData(duckdb::ExceptionType::INTERRUPT,
                                  "cancelled before this rank started it");
        break;
      }
      result = call.channel->Query(sql);
      if (result->HasError()) {
        error = result->GetErrorObject();
        break;
      }
    }
  } catch (std::exception& e) {
    error = duckdb::ErrorData(e);
  }
  std::lock_guard lock{shared.mutex};
  call.result = std::move(result);
  call.error  = std::move(error);
  call.done   = true;
  --shared.remaining;
  shared.changed.notify_all();
}

/// Sends quack_cancel every 100 ms to each rank whose call is still running, until told to
/// stop. A cancel to a rank that does not answer blocks, so this runs apart from the fan-out,
/// which never waits on it for long.
struct canceller {
  std::shared_ptr<fan_out_shared> shared;
  std::vector<std::pair<std::string, std::string>> targets;  // catalog, Quack session
  std::atomic<bool> stop{false};
  std::mutex mutex;
  std::condition_variable finished_cv;
  bool finished = false;
};

void run_canceller(duckdb::DatabaseInstance& db, std::shared_ptr<canceller> self)
{
  try {
    Connection control{db};
    while (!self->stop) {
      for (std::size_t i = 0; i < self->targets.size() && !self->stop; ++i) {
        {
          std::lock_guard lock{self->shared->mutex};
          if (self->shared->calls[i].done) { continue; }
        }
        control.Query("FROM quack_cancel(" + KeywordHelper::WriteQuoted(self->targets[i].first) +
                      ", " + KeywordHelper::WriteQuoted(self->targets[i].second) + ")");
      }
      std::this_thread::sleep_for(100ms);
    }
  } catch (std::exception&) {  // the fan-out's grace period still bounds the statement
  }
  std::lock_guard lock{self->mutex};
  self->finished = true;
  self->finished_cv.notify_all();
}

}  // namespace

char const* to_string(cluster_state state)
{
  switch (state) {
    case cluster_state::stopped: return "stopped";
    case cluster_state::starting: return "starting";
    case cluster_state::ready: return "ready";
    case cluster_state::failed: return "failed";
  }
  return "unknown";
}

void read_settings(ClientContext& context, start_options& options)
{
  options.duckdb_binary    = setting(context, "sirius_gateway_duckdb_binary", "");
  options.sirius_extension = setting(context, "sirius_gateway_sirius_extension", "");
  options.rank_init        = setting(context, "sirius_gateway_rank_init", "");
  options.log_dir          = setting(context, "sirius_gateway_log_dir", "");
  options.boot_id_sql      = setting(context, "sirius_gateway_rank_boot_id_sql", "");
  options.joined_sql       = setting(context, "sirius_gateway_rank_joined_sql", "");
  options.exit_sql         = setting(context, "sirius_gateway_rank_exit_sql", "");
  if (options.log_dir.empty()) {
    options.log_dir =
      (std::filesystem::temp_directory_path() / ("sirius_gateway_" + std::to_string(getpid())))
        .string();
  }
}

void gateway_state::fail_locked(std::string reason)
{
  if (state_ == cluster_state::failed) { return; }
  state_  = cluster_state::failed;
  reason_ = std::move(reason);
}

/// A place in the FIFO of statements; the holder runs alone. Leaving the line on interrupt
/// keeps a cancelled client from blocking those behind it.
class gateway_state::turn {
 public:
  turn(gateway_state& gateway, ClientContext& context) : gateway_(gateway)
  {
    std::unique_lock lock{gateway_.queue_mutex_};
    ticket_ = gateway_.next_ticket_++;
    gateway_.queue_.push_back(ticket_);
    while (gateway_.queue_.front() != ticket_) {
      gateway_.queue_cv_.wait_for(lock, 100ms);
      if (context.IsInterrupted()) {
        leave();
        throw duckdb::InterruptException();
      }
    }
  }
  ~turn()
  {
    std::lock_guard lock{gateway_.queue_mutex_};
    leave();
  }

 private:
  void leave()
  {
    auto& queue = gateway_.queue_;
    queue.erase(std::find(queue.begin(), queue.end(), ticket_));
    gateway_.queue_cv_.notify_all();
  }

  gateway_state& gateway_;
  uint64_t ticket_;
};

void gateway_state::start(ClientContext& context, start_options options, bool replay_saved)
{
  turn t{*this, context};
  {
    std::lock_guard lock{state_mutex_};
    if (!replay_saved) { ddl_log_.clear(); }
    auto_start_ = false;
    if (replay_saved) { options_ = options; }
  }
  ranks_cv_.notify_all();
  // A launch script restarts a gateway whose spawned ranks died with it: the saved log is the
  // only record of their catalog, so it is replayed, not discarded.
  if (replay_saved) {
    recover_locked(context);
    return;
  }
  start_locked(context, std::move(options));
  // Only once the new ranks run: a start that fails keeps the saved log.
  if (auto const file = ddl_file(context); !file.empty()) { std::filesystem::remove(file); }
}

void gateway_state::stop(ClientContext& context)
{
  turn t{*this, context};
  stop_locked();
  {
    std::lock_guard lock{state_mutex_};
    state_      = cluster_state::stopped;
    reason_     = "stopped: CALL sirius_gateway_start(<ranks>) or sirius_gateway_restart()";
    auto_start_ = false;
  }
  ranks_cv_.notify_all();
}

void gateway_state::restart(ClientContext& context)
{
  turn t{*this, context};
  recover_locked(context);
}

std::shared_ptr<rank_result> gateway_state::run(ClientContext& context, std::string const& sql)
{
  wait_for_ranks(context, sql);
  turn t{*this, context};
  return run_locked(context, sql, false);
}

void gateway_state::wait_for_ranks(ClientContext& context, std::string const& sql)
{
  auto const min_ranks = std::stoull(setting(context, "sirius_gateway_min_ranks", "1"));
  // Clamped to a year: a larger wait (the setting is UBIGINT) overflows the time point, which
  // would put the deadline in the past and fail the statement at once.
  auto const wait_s = std::min<uint64_t>(
    std::stoull(setting(context, "sirius_gateway_join_wait", "600")), 365ULL * 24 * 3600);
  auto const deadline = steady_clock::now() + std::chrono::seconds(wait_s);
  std::unique_lock lock{state_mutex_};
  auto const missing = [&] {
    return auto_start_ && ranks_.empty() && registered_.size() < min_ranks;
  };
  if (!missing()) { return; }
  lock.unlock();
  // run_locked answers a client's ATTACH from the gateway's own catalog.
  bool touches = false;
  if (reads_only_catalog(sql, &touches) && touches) { return; }
  lock.lock();
  ++join_waiters_;
  join_min_ranks_ = min_ranks;
  struct leave {
    std::size_t& waiters;
    ~leave() { --waiters; }
  } left{join_waiters_};
  while (missing() && steady_clock::now() < deadline) {
    ranks_cv_.wait_for(lock, 100ms);
    if (context.IsInterrupted()) { throw duckdb::InterruptException(); }
  }
}

void gateway_state::start_locked(ClientContext& context, start_options options)
{
  stop_locked();
  {
    std::lock_guard lock{state_mutex_};
    state_ = cluster_state::starting;
    reason_.clear();
  }
  std::vector<rank_spec> registered;
  {
    std::lock_guard lock{state_mutex_};
    registered = registered_;
  }
  // By name, so that which rank is rank 0 does not depend on the order the nodes registered.
  std::sort(registered.begin(), registered.end(), [](auto const& a, auto const& b) {
    return a.name < b.name;
  });
  if (options.ranks == 0 && registered.empty()) {
    throw InvalidInputException(
      "sirius_gateway_start needs at least one rank, or ranks added with sirius_gateway_add_rank");
  }
  if (!options.gpus.empty() && options.gpus.size() != options.ranks) {
    throw InvalidInputException(
      "gpus lists %d GPUs for %d ranks", options.gpus.size(), options.ranks);
  }
  if (options.ranks > 0 && options.duckdb_binary.empty()) {
    throw InvalidInputException(
      "SET sirius_gateway_duckdb_binary to the DuckDB that runs each rank");
  }
  std::filesystem::create_directories(options.log_dir);

  auto const token = random_token();
  std::vector<rank> ranks(options.ranks + registered.size());
  for (uint32_t i = 0; i < options.ranks; ++i) {
    auto& r         = ranks[i];
    r.index         = i;
    r.name          = "rank_" + std::to_string(i);
    r.token         = token;
    r.gpu           = options.gpus.empty() ? std::to_string(i) : options.gpus[i];
    r.host          = "127.0.0.1";
    r.quack_host    = r.host;
    r.quack_port    = options.quack_base_port + i;
    r.exchange_port = options.exchange_base_port + i;
    r.uri           = "quack:127.0.0.1:" + std::to_string(r.quack_port);
  }
  // Registered ranks follow the spawned ones.
  for (uint32_t j = 0; j < registered.size(); ++j) {
    auto& r         = ranks[options.ranks + j];
    auto const& s   = registered[j];
    r.index         = options.ranks + j;
    r.name          = s.name;
    r.registered    = true;
    r.host          = s.exchange_host;
    r.quack_host    = s.quack_host;
    r.exchange_port = s.exchange_port;
    r.quack_port    = s.quack_port;
    r.token         = s.token;
    r.fingerprint   = s.fingerprint;
    r.uri           = "quack:" + s.quack_host + ":" + std::to_string(s.quack_port);
  }
  // A loopback exchange address reaches only its own host's ranks, and NIXL then waits two
  // minutes for the others; refuse the mix at once.
  auto const loopback = [](std::string const& host) {
    return host == "localhost" || host == "::1" || host.rfind("127.", 0) == 0;
  };
  for (auto& r : ranks) {
    r.host = numeric_address(r.host);
  }
  for (auto const& r : ranks) {
    if (loopback(r.host) && !std::all_of(ranks.begin(), ranks.end(), [&](rank const& other) {
          return loopback(other.host);
        })) {
      throw InvalidInputException(
        "rank %s's exchange address %s is a loopback address, which the other ranks cannot reach",
        r.name,
        r.host);
    }
  }
  std::string peers;
  for (auto& r : ranks) {
    r.catalog = "sirius_rank_" + std::to_string(r.index);
    peers += (r.index ? "," : "") + r.host + ":" + std::to_string(r.exchange_port);
  }
  try {
    {
      std::lock_guard lock{state_mutex_};
      ranks_   = std::move(ranks);
      options_ = options;
      for (auto& r : ranks_) {
        if (!r.registered) { spawn_rank(options_, token, r); }
      }
    }
    ranks_cv_.notify_all();
    // Each join may wait minutes for a rank to start, so they run in parallel. The first error
    // ends the other joins' retries, and is the one reported.
    std::atomic<bool> give_up{false};
    std::exception_ptr first;
    rank const* first_failed = nullptr;
    std::vector<std::future<void>> joins;
    for (auto& r : ranks_) {
      joins.push_back(std::async(std::launch::async, [&, this] {
        try {
          connect_rank(r, options_, 600s, true, give_up);
        } catch (...) {
          if (give_up.exchange(true)) { return; }
          first        = std::current_exception();
          first_failed = &r;
        }
      }));
    }
    for (auto& join : joins) {
      while (join.wait_for(100ms) != std::future_status::ready) {
        if (context.IsInterrupted()) { give_up = true; }
      }
    }
    if (first) {
      try {
        std::rethrow_exception(first);
      } catch (std::exception& e) {
        std::lock_guard lock{state_mutex_};
        last_errors_[first_failed->name] = duckdb::ErrorData(e).RawMessage();
        throw;
      }
    }
    if (give_up) { throw duckdb::InterruptException(); }

    // Each session's setup. With Sirius, SET sirius_exchange_rank connects NIXL and waits for
    // every peer, so the ranks run it in parallel. Shared-memory staging reaches only processes
    // on one host, and 'auto' can take a GPU on another host for one here, so ranks on several
    // hosts never stage.
    auto const several_hosts = std::any_of(
      ranks_.begin(), ranks_.end(), [&](rank const& r) { return r.host != ranks_[0].host; });
    std::vector<std::vector<std::string>> setup(ranks_.size());
    for (auto& r : ranks_) {
      auto& sql = setup[r.index];
      if (!options_.sirius_extension.empty()) {
        sql.push_back("SET enable_duckdb_fallback = false");
        if (several_hosts) { sql.push_back("SET sirius_exchange_host_staging = 'never'"); }
        sql.push_back("SET sirius_exchange_peers = " + KeywordHelper::WriteQuoted(peers));
        sql.push_back("SET sirius_exchange_rank = " + std::to_string(r.index));
      }
      if (!options_.rank_init.empty()) { sql.push_back(options_.rank_init); }
    }
    auto const setup_done = fan_out(context, setup, client_policy(context));
    for (uint32_t i = 0; i < setup_done.outcomes.size(); ++i) {
      if (setup_done.outcomes[i].error) {
        auto const what = "rank " + std::to_string(i) + " failed its session setup: " +
                          setup_done.outcomes[i].error->RawMessage();
        std::lock_guard lock{state_mutex_};
        last_errors_[ranks_[i].name] = what;
        throw IOException(what);
      }
    }
    std::lock_guard lock{state_mutex_};
    state_ = cluster_state::ready;
  } catch (std::exception& e) {
    stop_locked();
    std::lock_guard lock{state_mutex_};
    state_  = cluster_state::failed;
    reason_ = duckdb::ErrorData(e).RawMessage();
    throw;
  }
}

void gateway_state::connect_rank(rank& r,
                                 start_options const& options,
                                 std::chrono::seconds patience,
                                 bool join,
                                 std::atomic<bool> const& give_up)
{
  Connection admin{db_};
  auto attach = "ATTACH " + KeywordHelper::WriteQuoted(r.uri) + " AS " + r.catalog + " (TOKEN " +
                KeywordHelper::WriteQuoted(r.token);
  if (!r.fingerprint.empty()) {
    attach += ", SSL_FINGERPRINT " + KeywordHelper::WriteQuoted(r.fingerprint);
    // Quack's TLS client reads http_timeout (30 s by default) for every call, so a longer
    // statement would fail; its plain client waits a day. The statement timeout bounds both.
    admin.Query("SET GLOBAL http_timeout = 86400");
  }
  attach += ")";
  auto const deadline = steady_clock::now() + patience;
  for (;;) {
    {
      std::lock_guard lock{state_mutex_};
      if (rank_exited(r)) {
        throw IOException(
          "rank %d exited during start; see %s", r.index, log_path(options, r.index));
      }
    }
    auto channel  = std::make_shared<Connection>(db_);
    auto attached = admin.Query(attach);
    auto result =
      attached->HasError() ? std::move(attached) : channel->Query("CONNECT " + r.catalog);
    if (!result->HasError()) { result = channel->Query("SELECT 1"); }
    // A registered rank's exchange peers are fixed once it joins. While it still runs the
    // process the gateway last saw, and that process has joined, ask it to exit; its launcher
    // starts a new one. A process that never joined serves as it is.
    std::string retiring;
    if (!result->HasError() && join && r.registered) {
      auto boot = channel->Query(options.boot_id_sql);
      std::string id;
      if (boot->HasError()) {
        result = std::move(boot);
      } else {
        auto chunk = boot->Fetch();
        id         = chunk && chunk->size() > 0 ? chunk->GetValue(0, 0).ToString() : "";
        std::lock_guard lock{state_mutex_};
        auto* spec = find_registered(r.name);
        if (spec && spec->boot_id == id) {
          retiring = id;
        } else if (spec) {
          spec->boot_id = id;
        }
      }
      if (!retiring.empty() && !options.joined_sql.empty()) {
        auto joined = channel->Query(options.joined_sql);
        if (joined->HasError()) {
          result = std::move(joined);
          retiring.clear();
        } else {
          auto chunk = joined->Fetch();
          if (!chunk || chunk->size() == 0 || !chunk->GetValue(0, 0).GetValue<bool>()) {
            retiring.clear();
          }
        }
      }
      if (!retiring.empty()) { channel->Query(options.exit_sql); }
    }
    if (!result->HasError() && retiring.empty()) {
      auto id =
        admin.Query("SELECT quack_connection_id(" + KeywordHelper::WriteQuoted(r.catalog) + ")");
      if (id->HasError()) { throw IOException("rank %d: %s", r.index, id->GetError()); }
      std::lock_guard lock{state_mutex_};
      r.session = id->Fetch()->GetValue(0, 0).ToString();
      r.channel = std::move(channel);
      return;
    }
    if (give_up || steady_clock::now() >= deadline) {
      channel.reset();
      admin.Query("DETACH DATABASE IF EXISTS " + r.catalog);
      if (!retiring.empty()) {
        throw IOException("rank %d (%s) still runs the process the gateway asked to exit (%s)",
                          r.index,
                          r.name,
                          retiring);
      }
      throw IOException(
        "rank %d (%s) did not answer at %s: %s", r.index, r.name, r.uri, result->GetError());
    }
    channel.reset();
    admin.Query("DETACH DATABASE IF EXISTS " + r.catalog);
    std::this_thread::sleep_for(200ms);
  }
}

rank_spec* gateway_state::find_registered(std::string const& name)
{
  auto it = std::find_if(
    registered_.begin(), registered_.end(), [&](auto const& s) { return s.name == name; });
  return it == registered_.end() ? nullptr : &*it;
}

void gateway_state::add_rank(ClientContext& context, rank_spec spec)
{
  // A node registers its rank again after every restart: the same rank is no change, and does
  // not wait for the turn, which a running statement holds.
  auto const known_as_is = [&] {
    std::lock_guard lock{state_mutex_};
    auto const* known = find_registered(spec.name);
    return known && known->quack_host == spec.quack_host && known->quack_port == spec.quack_port &&
           known->exchange_host == spec.exchange_host &&
           known->exchange_port == spec.exchange_port && known->token == spec.token &&
           known->fingerprint == spec.fingerprint;
  };
  if (known_as_is()) { return; }
  turn t{*this, context};
  if (known_as_is()) { return; }
  // Fail now, with Quack's own error, on a wrong address, token or fingerprint. The probe
  // records the rank's boot id: the first join uses that process, unless it has already joined.
  start_options options;
  read_settings(context, options);
  rank probe;
  probe.name        = spec.name;
  probe.registered  = true;
  probe.uri         = "quack:" + spec.quack_host + ":" + std::to_string(spec.quack_port);
  probe.catalog     = "sirius_probe";
  probe.token       = spec.token;
  probe.fingerprint = spec.fingerprint;
  connect_rank(probe, options, 10s, false, std::atomic<bool>{false});
  auto boot = probe.channel->Query(options.boot_id_sql);
  probe.channel.reset();
  Connection{db_}.Query("DETACH DATABASE IF EXISTS sirius_probe");
  if (boot->HasError()) {
    throw IOException("rank %s: %s: %s", spec.name, options.boot_id_sql, boot->GetError());
  }
  auto chunk   = boot->Fetch();
  spec.boot_id = chunk && chunk->size() > 0 ? chunk->GetValue(0, 0).ToString() : "";

  {
    std::lock_guard lock{state_mutex_};
    auto* known         = find_registered(spec.name);
    auto const replaced = known != nullptr;
    if (replaced) {
      *known = spec;
    } else {
      registered_.push_back(spec);
    }
    if (state_ != cluster_state::stopped) {
      fail_locked("membership changed: " + std::string(replaced ? "replaced " : "added ") +
                  spec.name + ", so the gateway restarts the ranks before the next statement");
    }
  }
  ranks_cv_.notify_all();
}

void gateway_state::drop_rank(ClientContext& context, std::string const& name)
{
  turn t{*this, context};
  {
    std::lock_guard lock{state_mutex_};
    auto* spec = find_registered(name);
    if (!spec) { throw InvalidInputException("no rank named %s was added", name); }
    registered_.erase(registered_.begin() + (spec - registered_.data()));
    if (state_ != cluster_state::stopped) {
      fail_locked("membership changed: dropped " + name +
                  ", so the gateway restarts the ranks before the next statement");
    }
  }
  ranks_cv_.notify_all();
}

void gateway_state::stop_locked()
{
  Connection admin{db_};
  std::lock_guard lock{state_mutex_};
  for (auto& r : ranks_) {
    r.channel.reset();
    admin.Query("DETACH DATABASE IF EXISTS " + r.catalog);
    end_rank(r);
  }
  ranks_.clear();
}

void gateway_state::recover_locked(ClientContext& context)
{
  std::vector<std::string> log;
  {
    std::lock_guard lock{state_mutex_};
    log = ddl_log_;
  }
  // A gateway that restarted has an empty log: its ranks replay the one it saved.
  auto const file      = ddl_file(context);
  bool const from_file = log.empty() && !file.empty() && std::filesystem::exists(file);
  if (from_file) {
    log = load_ddl_log(file);
    std::lock_guard lock{state_mutex_};
    ddl_log_ = log;
  }
  start_locked(context, options_);
  std::vector<std::string> kept;
  std::string skipped;
  for (std::size_t i = 0; i < log.size(); ++i) {
    try {
      run_locked(context, log[i], true);
      kept.push_back(log[i]);
    } catch (std::exception& e) {
      auto const error = duckdb::ErrorData(e).RawMessage();
      std::lock_guard lock{state_mutex_};
      // A saved entry can fail for a reason outside the log: a file deleted since, or a secret,
      // which is never saved. Failing on it would fail every later statement too, including the
      // one that fixes it, so it is dropped, provided it failed alike on every rank (the cluster
      // is still ready) and not because the client cancelled it.
      bool const cancelled =
        context.IsInterrupted() || error.find("cancelled on every rank") != std::string::npos;
      if (from_file && state_ == cluster_state::ready && !cancelled) {
        skipped += (skipped.empty() ? "" : "; ") + std::string{"skipped saved DDL entry "} +
                   std::to_string(i + 1) + ": " + error;
        continue;
      }
      state_  = cluster_state::failed;
      reason_ = "replaying \"" + log[i] + "\" failed: " + error;
      throw IOException(reason_);
    }
  }
  if (skipped.empty()) { return; }
  std::fprintf(stderr, "sirius_gateway: %s\n", skipped.c_str());
  auto const save_error = try_save_ddl_log(file, kept);
  std::lock_guard lock{state_mutex_};
  ddl_log_ = std::move(kept);
  // Shown in sirius_gateway_status's error column until the next state change or start.
  reason_ = skipped + (save_error.empty() ? "" : "; " + save_error);
}

void gateway_state::auto_start_locked(ClientContext& context)
{
  std::size_t registered = 0;
  {
    std::lock_guard lock{state_mutex_};
    if (!auto_start_) { return; }
    registered = registered_.size();
  }
  if (registered == 0) { return; }
  // wait_for_ranks gave up: the ranks that are missing register through this same turn.
  uint64_t const min_ranks = std::stoull(setting(context, "sirius_gateway_min_ranks", "1"));
  if (registered < min_ranks) {
    throw IOException(
      "the gateway starts once %d ranks are registered (sirius_gateway_min_ranks); %d are",
      min_ranks,
      static_cast<uint64_t>(registered));
  }
  start_options options;
  read_settings(context, options);
  {
    std::lock_guard lock{state_mutex_};
    options_ = std::move(options);
  }
  // Also after a start that failed, which leaves no ranks: replay what ran before it.
  recover_locked(context);
}

std::shared_ptr<rank_result> gateway_state::run_locked(ClientContext& context,
                                                       std::string const& sql,
                                                       bool replay)
{
  bool touches = false;
  if (ranks_.empty() && !replay && reads_only_catalog(sql, &touches) && touches) {
    // No ranks run: the gateway answers a client's ATTACH from its own catalog, so that a node
    // can attach to register its rank, and attaching starts no ranks.
    Connection local{db_};
    auto read = local.Query(sql);
    if (read->HasError()) { read->ThrowError(); }
    return collect(*read);
  }
  if (ranks_.empty() && !replay) { auto_start_locked(context); }
  if (ranks_.empty()) {
    std::lock_guard lock{state_mutex_};
    throw IOException(reason_.empty()
                        ? "the gateway has no ranks: CALL sirius_gateway_start(<ranks>), "
                          "or register ranks with sirius_gateway_add_rank"
                        : reason_);
  }
  // A registered rank has no process here: one that died since the last statement is found by
  // its port, so that the ranks restart before this statement rather than fail it.
  bool ready = false;
  {
    std::lock_guard lock{state_mutex_};
    ready = state_ == cluster_state::ready;
  }
  auto const dead = ready && !replay ? unreachable_registered(2s) : std::vector<uint32_t>{};
  bool failed     = false;
  {
    std::lock_guard lock{state_mutex_};
    for (auto const i : dead) {
      auto const& r        = ranks_[i];
      last_errors_[r.name] = "lost: " + r.uri + " took no connection";
      fail_locked("rank " + std::to_string(i) + " (" + r.name + ") does not answer at " + r.uri +
                  ", so the gateway restarts the ranks");
    }
    for (auto& r : ranks_) {
      if (rank_exited(r)) {
        auto const what =
          "rank " + std::to_string(r.index) + " exited; see " + log_path(options_, r.index);
        last_errors_[r.name] = what;
        fail_locked(what);
      }
    }
    failed = state_ == cluster_state::failed;
  }
  if (failed) {
    if (replay) {
      std::lock_guard lock{state_mutex_};
      throw IOException(reason_);
    }
    recover_locked(context);
  }
  uint64_t query = 0;
  {
    std::lock_guard lock{state_mutex_};
    query             = ++query_id_;
    running_query_id_ = query;
  }
  struct not_running {
    gateway_state& gateway;
    ~not_running()
    {
      std::lock_guard lock{gateway.state_mutex_};
      gateway.running_query_id_ = 0;
    }
  } clear_running{*this};

  // Catalog reads run on DuckDB alone: Sirius cannot plan some of them, and with fallback off
  // that would fail every client's ATTACH. The ranks' catalogs are the same, so rank 0's answer
  // is still everyone's.
  bool const on_cpu = !options_.sirius_extension.empty() && reads_only_catalog(sql);
  auto const id_setting =
    setting(context, "sirius_gateway_query_id_setting", "sirius_distributed_query_id");
  std::vector<std::vector<std::string>> statements(ranks_.size());
  for (auto& sql_of_rank : statements) {
    if (!id_setting.empty()) {
      sql_of_rank.push_back("SET " + id_setting + " = " + std::to_string(query));
    }
    if (on_cpu) { sql_of_rank.push_back("SET gpu_execution = false"); }
    sql_of_rank.push_back(sql);
  }
  auto const policy   = client_policy(context);
  auto ran            = fan_out(context, statements, policy);
  auto const prefix   = "query " + std::to_string(query) + ": ";
  auto const restarts = ", so the gateway restarts the ranks before the next statement";
  if (on_cpu) {
    // Runs even after a cancel: the client's interrupt must not leave GPU execution off.
    auto const restored =
      fan_out(context,
              std::vector<std::vector<std::string>>(ranks_.size(), {"SET gpu_execution = true"}),
              fan_out_policy{false, policy.grace_s, policy.grace_s});
    for (uint32_t i = 0; i < restored.outcomes.size(); ++i) {
      if (!restored.outcomes[i].error) { continue; }
      std::lock_guard lock{state_mutex_};
      fail_locked("rank " + std::to_string(i) + " could not turn gpu_execution back on: " +
                  restored.outcomes[i].error->RawMessage());
    }
  }

  auto& outcomes = ran.outcomes;
  std::vector<std::optional<duckdb::ErrorData>> errors(outcomes.size());
  std::size_t failures = 0;
  std::string failed_ranks;
  std::set<uint32_t> lost;
  bool transport = false;
  for (uint32_t i = 0; i < outcomes.size(); ++i) {
    if (!outcomes[i].error) { continue; }
    errors[i] = outcomes[i].error;
    ++failures;
    auto const& message = outcomes[i].error->RawMessage();
    transport           = transport || transport_failed(message);
    if (rank_unreachable(message)) { lost.insert(i); }
    failed_ranks += (failed_ranks.empty() ? "" : "; ") + std::string("rank ") + std::to_string(i) +
                    ": " + message;
  }
  {
    std::lock_guard lock{state_mutex_};
    for (auto& r : ranks_) {
      if (rank_exited(r)) { lost.insert(r.index); }
    }
    for (auto const i : lost) {
      last_errors_[ranks_[i].name] =
        "lost in query " + std::to_string(query) + ": " +
        (errors[i] ? errors[i]->RawMessage() : std::string{"the process exited"});
    }
  }
  // Fail-stop, in order: a killed rank, a lost one, a broken exchange, then a state change only
  // some ranks made. Each leaves the ranks out of step.
  std::string fatal;
  if (!ran.killed.empty()) {
    std::string killed;
    for (auto const i : ran.killed) {
      killed += (killed.empty() ? "" : ", ") + std::to_string(i);
    }
    fatal = prefix + ran.cancelled + "; rank " + killed + " did not stop within " +
            std::to_string(policy.grace_s) + " s of the cancel";
  } else if (!lost.empty()) {
    std::string names;
    for (auto const i : lost) {
      names += (names.empty() ? "" : ", ") + std::to_string(i);
    }
    fatal = prefix + "lost rank " + names + " (" + failed_ranks + ")";
  } else if (transport) {
    fatal = prefix + "the exchange between the ranks failed (" + failed_ranks + ")";
  } else if (failures != 0 && failures != outcomes.size() && (replay || changes_state(sql))) {
    fatal = prefix + "the ranks disagree on a statement that changes state (" + failed_ranks + ")";
  }
  if (!fatal.empty()) {
    std::lock_guard lock{state_mutex_};
    fail_locked(fatal + restarts);
    throw IOException(reason_);
  }
  if (!ran.cancelled.empty() && failures != 0) {
    throw IOException("%s; cancelled on every rank", ran.cancelled);
  }
  if (failures == outcomes.size()) { root_cause(errors).Throw(); }
  if (failures != 0) {
    // A read changes nothing, and every rank has retired the query: the ranks keep serving.
    throw IOException("%sthe ranks disagree (%s)", prefix, failed_ranks);
  }
  std::vector<std::string> log;
  {
    std::lock_guard lock{state_mutex_};
    if (state_ == cluster_state::failed) { throw IOException(reason_); }
    if (!replay && changes_state(sql)) {
      ddl_log_.push_back(sql);
      log = ddl_log_;
    }
  }
  if (!log.empty()) {
    // The statement took effect on every rank, so it succeeds even if the log cannot be saved.
    // The error (or, once saved, none) replaces an earlier note of a skipped entry or a save
    // that failed.
    auto const save_error = try_save_ddl_log(ddl_file(context), log);
    std::lock_guard lock{state_mutex_};
    if (state_ == cluster_state::ready) { reason_ = save_error; }
  }

  return collect(*outcomes[0].result);
}

gateway_state::fan_out_policy gateway_state::client_policy(ClientContext& context)
{
  return fan_out_policy{true,
                        std::stoull(setting(context, "sirius_gateway_statement_timeout", "0")),
                        std::stoull(setting(context, "sirius_gateway_cancel_grace", "30"))};
}

gateway_state::fan_out_result gateway_state::fan_out(
  ClientContext& context,
  std::vector<std::vector<std::string>> const& statements,
  fan_out_policy policy)
{
  auto shared = std::make_shared<fan_out_shared>(ranks_.size());
  std::vector<std::thread> threads;
  for (std::size_t i = 0; i < ranks_.size(); ++i) {
    shared->calls[i].channel    = ranks_[i].channel;
    shared->calls[i].statements = statements[i];
  }
  for (std::size_t i = 0; i < ranks_.size(); ++i) {
    threads.emplace_back([shared, i] { run_call(*shared, i); });
  }

  fan_out_result out;
  std::shared_ptr<canceller> cancelling;
  auto const started = steady_clock::now();
  std::optional<steady_clock::time_point> cancelled_at;
  std::optional<steady_clock::time_point> killed_at;
  for (;;) {
    {
      std::unique_lock lock{shared->mutex};
      if (shared->changed.wait_for(lock, 100ms, [&] { return shared->remaining == 0; })) { break; }
    }
    auto const now = steady_clock::now();
    if (!cancelled_at) {
      if (policy.watch_client && context.IsInterrupted()) {
        out.cancelled = "interrupted";
      } else if (policy.timeout_s != 0 && now - started > std::chrono::seconds(policy.timeout_s)) {
        out.cancelled = "timed out after " + std::to_string(policy.timeout_s) + " s";
      } else {
        continue;
      }
      cancelled_at       = now;
      shared->cancel     = true;
      cancelling         = std::make_shared<canceller>();
      cancelling->shared = shared;
      for (auto const& r : ranks_) {
        cancelling->targets.emplace_back(r.catalog, r.session);
      }
      std::thread(run_canceller, std::ref(db_), cancelling).detach();
    } else if (!killed_at && now - *cancelled_at > std::chrono::seconds(policy.grace_s)) {
      // A spawned rank's process is the gateway's own: killing it ends the call. A registered
      // one is left to the cluster's failure, which the next statement's restart handles.
      killed_at = now;
      std::lock_guard state_lock{state_mutex_};
      std::lock_guard lock{shared->mutex};
      for (uint32_t i = 0; i < ranks_.size(); ++i) {
        if (shared->calls[i].done) { continue; }
        out.killed.push_back(i);
        if (ranks_[i].pid > 0) { kill(ranks_[i].pid, SIGKILL); }
      }
    } else if (killed_at && now - *killed_at > 10s) {
      break;  // the calls still stuck are abandoned
    }
  }

  if (cancelling) {
    cancelling->stop = true;
    std::unique_lock lock{cancelling->mutex};
    cancelling->finished_cv.wait_for(lock, 5s, [&] { return cancelling->finished; });
  }
  std::lock_guard lock{shared->mutex};
  out.outcomes.resize(ranks_.size());
  for (std::size_t i = 0; i < ranks_.size(); ++i) {
    auto& call = shared->calls[i];
    if (call.done) {
      threads[i].join();
      out.outcomes[i].result = std::move(call.result);
      out.outcomes[i].error  = std::move(call.error);
    } else {
      threads[i].detach();
      out.outcomes[i].error = duckdb::ErrorData(
        duckdb::ExceptionType::IO, "rank " + std::to_string(i) + " did not answer; abandoned");
    }
  }
  return out;
}

std::vector<uint32_t> gateway_state::unreachable_registered(std::chrono::milliseconds patience)
{
  std::vector<std::pair<uint32_t, std::future<bool>>> probes;
  {
    std::lock_guard lock{state_mutex_};
    for (auto const& r : ranks_) {
      if (!r.registered) { continue; }
      probes.emplace_back(
        r.index,
        std::async(std::launch::async, [host = r.quack_host, port = r.quack_port, patience] {
          return accepts_connections(host, port, patience);
        }));
    }
  }
  std::vector<uint32_t> dead;
  for (auto& [index, open] : probes) {
    if (!open.get()) { dead.push_back(index); }
  }
  return dead;
}

void gateway_state::signal_rank(uint32_t index, int signal)
{
  std::lock_guard lock{state_mutex_};
  if (index >= ranks_.size()) { throw InvalidInputException("no running rank %d", index); }
  auto* process = &ranks_[index];
  if (process->registered) {
    auto const server = std::find_if(test_servers_.begin(),
                                     test_servers_.end(),
                                     [&](rank const& s) { return s.uri == process->uri; });
    process           = server == test_servers_.end() ? nullptr : &*server;
  }
  if (!process || process->pid <= 0) { throw InvalidInputException("no running rank %d", index); }
  kill(process->pid, signal);
  // A killed rank is gone before this returns, so the next statement finds it gone.
  for (int i = 0; signal == SIGKILL && i < 500 && !rank_exited(*process); ++i) {
    std::this_thread::sleep_for(10ms);
  }
}

void gateway_state::test_serve(start_options const& options,
                               uint32_t port,
                               std::string const& token,
                               std::string const& init_sql)
{
  std::filesystem::create_directories(options.log_dir);
  rank server;
  server.index = port;
  server.uri   = "quack:127.0.0.1:" + std::to_string(port);
  spawn_rank(options, token, server, init_sql);
  std::lock_guard lock{state_mutex_};
  test_servers_.push_back(std::move(server));
}

uint64_t gateway_state::running_query_id()
{
  std::lock_guard lock{state_mutex_};
  return running_query_id_;
}

std::vector<gateway_state::rank_status> gateway_state::status()
{
  std::lock_guard lock{state_mutex_};
  auto const reason     = join_waiters_ > 0 && ranks_.empty()
                            ? "waiting for " + std::to_string(join_min_ranks_) +
                            " ranks (sirius_gateway_min_ranks); " +
                            std::to_string(registered_.size()) + " are registered"
                            : reason_;
  auto const last_error = [&](std::string const& name) {
    auto const it = last_errors_.find(name);
    return it == last_errors_.end() ? std::string{} : it->second;
  };
  std::vector<rank_status> rows;
  for (auto const& r : ranks_) {
    auto const up = r.registered ? r.channel != nullptr : r.pid >= 0;
    rows.push_back({r.index,
                    r.name,
                    r.registered,
                    r.pid,
                    r.gpu,
                    r.host,
                    r.uri,
                    up ? "up" : "exited",
                    state_,
                    reason,
                    last_error(r.name)});
  }
  // Added since the ranks last started: they join on the next statement.
  for (auto const& spec : registered_) {
    if (std::none_of(
          ranks_.begin(), ranks_.end(), [&](rank const& r) { return r.name == spec.name; })) {
      rows.push_back({0,
                      spec.name,
                      true,
                      -1,
                      "",
                      spec.exchange_host,
                      "quack:" + spec.quack_host + ":" + std::to_string(spec.quack_port),
                      "registered",
                      state_,
                      reason,
                      last_error(spec.name)});
    }
  }
  if (rows.empty() && state_ != cluster_state::ready) {
    rows.push_back({0, "", false, -1, "", "", "", to_string(state_), state_, reason, ""});
  }
  return rows;
}

uint64_t gateway_state::last_query_id()
{
  std::lock_guard lock{state_mutex_};
  return query_id_;
}

uint64_t gateway_state::waiting()
{
  std::lock_guard lock{queue_mutex_};
  return queue_.size();
}

std::vector<std::string> gateway_state::ddl_log()
{
  std::lock_guard lock{state_mutex_};
  return ddl_log_;
}

}  // namespace sirius_gateway
