#include "gateway_state.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/parser/keyword_helper.hpp"
#include "duckdb/parser/parsed_expression_iterator.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/basetableref.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <functional>
#include <future>
#include <random>
#include <set>
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

/// Whether @p sql only reads the catalog: SELECTs whose every table, at any depth, is a
/// `duckdb_*()` or `pragma_*()` table function or one of the statement's top-level CTEs. Quack
/// sends such reads on every client's ATTACH.
bool reads_only_catalog(std::string const& sql)
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
          only = only && (StringUtil::StartsWith(name, "duckdb_") ||
                          StringUtil::StartsWith(name, "pragma_"));
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

/// The error to report when every rank failed: the first one not caused by a peer's abort.
duckdb::ErrorData const& root_cause(std::vector<std::optional<duckdb::ErrorData>> const& errors)
{
  for (auto const& error : errors) {
    if (!duckdb::StringUtil::Contains(duckdb::StringUtil::Lower(error->RawMessage()), "abort")) {
      return *error;
    }
  }
  return *errors.front();
}

}  // namespace

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

void gateway_state::start(ClientContext& context, start_options options)
{
  turn t{*this, context};
  {
    std::lock_guard lock{state_mutex_};
    ddl_log_.clear();
  }
  start_locked(context, std::move(options));
}

void gateway_state::stop(ClientContext& context)
{
  turn t{*this, context};
  stop_locked();
  std::lock_guard lock{state_mutex_};
  broken_ = "stopped: CALL sirius_gateway_start(<ranks>) or sirius_gateway_restart()";
}

void gateway_state::restart(ClientContext& context)
{
  turn t{*this, context};
  recover_locked(context);
}

std::shared_ptr<rank_result> gateway_state::run(ClientContext& context, std::string const& sql)
{
  turn t{*this, context};
  return run_locked(context, sql, false);
}

void gateway_state::start_locked(ClientContext& context, start_options options)
{
  stop_locked();
  if (options.ranks == 0) {
    throw InvalidInputException("sirius_gateway_start needs at least one rank");
  }
  if (!options.gpus.empty() && options.gpus.size() != options.ranks) {
    throw InvalidInputException(
      "gpus lists %d GPUs for %d ranks", options.gpus.size(), options.ranks);
  }
  if (!options.hosts.empty() && options.hosts.size() != options.ranks) {
    throw InvalidInputException(
      "hosts lists %d hosts for %d ranks", options.hosts.size(), options.ranks);
  }
  if (options.duckdb_binary.empty()) {
    throw InvalidInputException(
      "SET sirius_gateway_duckdb_binary to the DuckDB that runs each rank");
  }
  std::filesystem::create_directories(options.log_dir);

  std::vector<rank> ranks(options.ranks);
  std::string peers;
  for (uint32_t i = 0; i < options.ranks; ++i) {
    auto& r  = ranks[i];
    r.index  = i;
    r.gpu    = options.gpus.empty() ? std::to_string(i) : options.gpus[i];
    r.host   = options.hosts.empty() ? "127.0.0.1" : options.hosts[i];
    r.remote = !is_local_host(r.host);
    if (r.remote && options.rank_launcher.empty()) {
      throw InvalidInputException(
        "rank %d's host %s is not this machine: SET sirius_gateway_rank_launcher to the program "
        "that starts ranks there",
        i,
        r.host);
    }
    // A remote rank's launcher forwards this port to the rank's own 127.0.0.1.
    r.uri     = "quack:127.0.0.1:" + std::to_string(options.quack_base_port + i);
    r.catalog = "sirius_rank_" + std::to_string(i);
    peers += (i ? "," : "") + r.host + ":" + std::to_string(options.exchange_base_port + i);
  }
  auto const token = random_token();
  try {
    {
      std::lock_guard lock{state_mutex_};
      ranks_   = std::move(ranks);
      options_ = options;
      broken_.clear();
      for (auto& r : ranks_) {
        spawn_rank(options_, token, r);
      }
    }
    for (auto& r : ranks_) {
      connect_rank(r, token);
    }

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
    std::string cancelled;
    auto const outcomes = fan_out(context, setup, cancelled);
    for (uint32_t i = 0; i < outcomes.size(); ++i) {
      if (outcomes[i].error) {
        throw IOException(
          "rank %d failed its session setup: %s", i, outcomes[i].error->RawMessage());
      }
    }
  } catch (std::exception& e) {
    stop_locked();
    std::lock_guard lock{state_mutex_};
    broken_ = duckdb::ErrorData(e).RawMessage();
    throw;
  }
}

void gateway_state::connect_rank(rank& r, std::string const& token)
{
  Connection admin{db_};
  auto const attach = "ATTACH " + KeywordHelper::WriteQuoted(r.uri) + " AS " + r.catalog +
                      " (TOKEN " + KeywordHelper::WriteQuoted(token) + ")";
  auto const deadline = steady_clock::now() + 600s;
  for (;;) {
    {
      std::lock_guard lock{state_mutex_};
      if (rank_exited(r)) {
        throw IOException(
          "rank %d exited during start; see %s", r.index, log_path(options_, r.index));
      }
    }
    auto channel  = std::make_unique<Connection>(db_);
    auto attached = admin.Query(attach);
    auto result =
      attached->HasError() ? std::move(attached) : channel->Query("CONNECT " + r.catalog);
    if (!result->HasError()) { result = channel->Query("SELECT 1"); }
    if (!result->HasError()) {
      auto id =
        admin.Query("SELECT quack_connection_id(" + KeywordHelper::WriteQuoted(r.catalog) + ")");
      if (id->HasError()) { throw IOException("rank %d: %s", r.index, id->GetError()); }
      r.session = id->Fetch()->GetValue(0, 0).ToString();
      r.channel = std::move(channel);
      return;
    }
    if (steady_clock::now() > deadline) {
      throw IOException("rank %d did not answer at %s: %s", r.index, r.uri, result->GetError());
    }
    channel.reset();
    admin.Query("DETACH DATABASE IF EXISTS " + r.catalog);
    std::this_thread::sleep_for(200ms);
  }
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
  start_locked(context, options_);
  std::vector<std::string> log;
  {
    std::lock_guard lock{state_mutex_};
    log = ddl_log_;
  }
  for (auto const& entry : log) {
    try {
      run_locked(context, entry, true);
    } catch (std::exception& e) {
      std::lock_guard lock{state_mutex_};
      broken_ = "replaying \"" + entry + "\" failed: " + duckdb::ErrorData(e).RawMessage();
      throw IOException(broken_);
    }
  }
}

std::shared_ptr<rank_result> gateway_state::run_locked(ClientContext& context,
                                                       std::string const& sql,
                                                       bool replay)
{
  if (ranks_.empty()) {
    throw IOException(broken_.empty()
                        ? "the gateway has no ranks: CALL sirius_gateway_start(<ranks>) first"
                        : broken_);
  }
  {
    std::lock_guard lock{state_mutex_};
    for (auto& r : ranks_) {
      if (rank_exited(r) && broken_.empty()) {
        broken_ = "rank " + std::to_string(r.index) + " exited; see " + log_path(options_, r.index);
      }
    }
  }
  if (!broken_.empty()) {
    if (replay) { throw IOException(broken_); }
    recover_locked(context);
  }
  uint64_t query = 0;
  {
    std::lock_guard lock{state_mutex_};
    query = ++query_id_;
  }

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
  std::string cancelled;
  auto outcomes = fan_out(context, statements, cancelled);
  if (on_cpu) {
    std::string ignored;
    auto const restored =
      fan_out(context,
              std::vector<std::vector<std::string>>(ranks_.size(), {"SET gpu_execution = true"}),
              ignored);
    for (uint32_t i = 0; i < restored.size(); ++i) {
      if (!restored[i].error) { continue; }
      std::lock_guard lock{state_mutex_};
      if (broken_.empty()) {
        broken_ = "rank " + std::to_string(i) +
                  " could not turn gpu_execution back on: " + restored[i].error->RawMessage();
      }
    }
  }

  std::vector<std::optional<duckdb::ErrorData>> errors;
  std::string failed_ranks;
  for (uint32_t i = 0; i < outcomes.size(); ++i) {
    if (!outcomes[i].error) { continue; }
    errors.push_back(outcomes[i].error);
    failed_ranks += (failed_ranks.empty() ? "" : "; ") + std::string("rank ") + std::to_string(i) +
                    ": " + outcomes[i].error->RawMessage();
  }
  if (errors.size() == outcomes.size()) {
    if (!cancelled.empty()) { throw IOException("%s; cancelled on every rank", cancelled); }
    root_cause(errors).Throw();
  }
  if (!errors.empty()) {
    std::lock_guard lock{state_mutex_};
    broken_ = "the ranks disagree on query " + std::to_string(query) +
              ", so the gateway restarts them before the next statement (" + failed_ranks + ")";
    throw IOException(broken_);
  }
  if (!replay && changes_state(sql)) {
    std::lock_guard lock{state_mutex_};
    ddl_log_.push_back(sql);
  }

  auto result   = std::make_shared<rank_result>();
  auto& rank0   = *outcomes[0].result;
  result->types = rank0.GetTypes();
  for (auto const& name : rank0.GetNames()) {
    result->names.push_back(name.GetIdentifierName());
  }
  while (auto chunk = rank0.Fetch()) {
    if (chunk->size() > 0) { result->chunks.push_back(std::move(chunk)); }
  }
  return result;
}

std::vector<gateway_state::outcome> gateway_state::fan_out(
  ClientContext& context,
  std::vector<std::vector<std::string>> const& statements,
  std::string& cancelled)
{
  std::vector<outcome> outcomes(ranks_.size());
  std::vector<std::future<void>> done;
  for (uint32_t i = 0; i < ranks_.size(); ++i) {
    done.push_back(std::async(std::launch::async, [this, &statements, &outcomes, i] {
      auto& out = outcomes[i];
      try {
        for (auto const& sql : statements[i]) {
          out.result = ranks_[i].channel->Query(sql);
          if (out.result->HasError()) {
            out.error = out.result->GetErrorObject();
            return;
          }
        }
      } catch (std::exception& e) {
        out.error = duckdb::ErrorData(e);
      }
    }));
  }
  auto const timeout = std::stoull(setting(context, "sirius_gateway_statement_timeout", "0"));
  auto const started = steady_clock::now();
  for (auto& f : done) {
    while (f.wait_for(100ms) != std::future_status::ready) {
      if (!cancelled.empty()) { continue; }
      if (context.IsInterrupted()) {
        cancelled = "interrupted";
      } else if (timeout != 0 && steady_clock::now() - started > std::chrono::seconds(timeout)) {
        cancelled = "timed out after " + std::to_string(timeout) + " s";
      } else {
        continue;
      }
      cancel_all();
    }
  }
  return outcomes;
}

void gateway_state::cancel_all()
{
  Connection control{db_};
  for (auto const& r : ranks_) {
    control.Query("FROM quack_cancel(" + KeywordHelper::WriteQuoted(r.catalog) + ", " +
                  KeywordHelper::WriteQuoted(r.session) + ")");
  }
}

std::vector<gateway_state::rank_status> gateway_state::status()
{
  std::lock_guard lock{state_mutex_};
  std::vector<rank_status> rows;
  for (auto const& r : ranks_) {
    auto const state = r.pid < 0 ? "exited" : !broken_.empty() ? "broken" : "up";
    rows.push_back({r.index, r.pid, r.gpu, r.host, r.uri, state, broken_});
  }
  if (rows.empty() && !broken_.empty()) { rows.push_back({0, -1, "", "", "", "stopped", broken_}); }
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
