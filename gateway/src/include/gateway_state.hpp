#pragma once

#include "duckdb.hpp"

#include <sys/types.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace sirius_gateway {

/// Where the ranks stand as a whole. Fail-stop: anything that may leave the ranks out of step
/// (a lost or killed rank, a broken exchange, a state change some ranks missed) is `failed`,
/// and the next statement restarts every rank and replays the DDL log first.
enum class cluster_state { stopped, starting, ready, failed };
char const* to_string(cluster_state state);

/// How to start the ranks; filled from `sirius_gateway_start`'s arguments and the
/// `sirius_gateway_*` settings.
struct start_options {
  uint32_t ranks = 0;
  /// `CUDA_VISIBLE_DEVICES` per rank; empty: rank i on GPU i.
  std::vector<std::string> gpus;
  /// Exchange address per rank, as the other ranks reach it; empty: every rank on 127.0.0.1.
  std::vector<std::string> hosts;
  /// Program that starts a rank whose host is not this one (`sirius_gateway_rank_launcher`).
  std::string rank_launcher;
  uint32_t exchange_base_port = 27000;
  uint32_t quack_base_port    = 9500;
  std::string duckdb_binary;
  /// Empty: CPU ranks without Sirius, for tests.
  std::string sirius_extension;
  /// SQL every rank's session runs after the exchange settings.
  std::string rank_init;
  std::string log_dir;
  /// How a registered rank names its process, and how it is asked to exit for its launcher to
  /// restart it (`sirius_gateway_rank_boot_id_sql`, `sirius_gateway_rank_exit_sql`).
  std::string boot_id_sql;
  std::string exit_sql;
};

/// A rank started outside the gateway, by a launcher that restarts it whenever it exits;
/// added with `sirius_gateway_add_rank`.
struct rank_spec {
  std::string name;
  std::string quack_host;
  uint32_t quack_port = 0;
  std::string exchange_host;
  uint32_t exchange_port = 0;
  std::string token;
  /// The rank's TLS certificate, pinned; empty: plain TCP, for a rank on this host.
  std::string fingerprint;
  /// The process the gateway last saw: the next join asks it to exit, and waits for a new one.
  std::string boot_id;
};

/// One rank process and the gateway's one session on it.
struct rank {
  uint32_t index = 0;
  std::string name;
  /// Started outside the gateway (a rank_spec); otherwise spawned by sirius_gateway_start.
  bool registered        = false;
  uint32_t quack_port    = 0;
  uint32_t exchange_port = 0;
  std::string token;
  std::string fingerprint;
  pid_t pid = -1;
  /// The rank reads its stdin until EOF, so closing this ends it.
  int stdin_fd = -1;
  std::string gpu;
  std::string host;
  /// Started through the rank launcher, which forwards `uri` to the rank's host.
  bool remote = false;
  std::string uri;
  std::string catalog;
  /// The only user of `catalog`, CONNECTed to it: the rank's persistent session. Shared with
  /// the call running on it, which may outlive the statement if the gateway gives up on it.
  std::shared_ptr<duckdb::Connection> channel;
  /// That session's Quack connection id, which `quack_cancel` takes.
  std::string session;
};

/// Rank 0's result of one statement.
struct rank_result {
  duckdb::vector<duckdb::LogicalType> types;
  std::vector<std::string> names;
  std::vector<std::unique_ptr<duckdb::DataChunk>> chunks;
};

/// Per-database gateway: the ranks, the FIFO of statements, the query number K and the DDL log.
/// Every statement, and every start, stop and restart, takes the one turn in arrival order.
class gateway_state {
 public:
  explicit gateway_state(duckdb::DatabaseInstance& db) : db_(db) {}

  void start(duckdb::ClientContext& context, start_options options);
  /// Registers a rank started outside the gateway; the next statement rebuilds the ranks with it.
  void add_rank(duckdb::ClientContext& context, rank_spec spec);
  /// Unregisters a rank added with add_rank; the next statement rebuilds the ranks without it.
  void drop_rank(duckdb::ClientContext& context, std::string const& name);
  void stop(duckdb::ClientContext& context);
  void restart(duckdb::ClientContext& context);
  /// Runs @p sql on every rank and returns rank 0's result.
  std::shared_ptr<rank_result> run(duckdb::ClientContext& context, std::string const& sql);

  struct rank_status {
    uint32_t index;
    std::string name;
    bool registered;
    int64_t pid;
    std::string gpu;
    std::string host;
    std::string uri;
    /// The rank's process: up or exited.
    std::string state;
    cluster_state cluster;
    /// Why the cluster is stopped or failed.
    std::string error;
  };
  std::vector<rank_status> status();
  uint64_t last_query_id();
  /// The query number of the statement running now, or 0.
  uint64_t running_query_id();
  /// Test hook (`sirius_gateway_test_signal`): sends @p signal to rank @p index's process.
  void signal_rank(uint32_t index, int signal);
  /// Test hook (`sirius_gateway_test_serve`): starts a DuckDB that runs @p init_sql, then serves
  /// Quack on 127.0.0.1:@p port, for tests to register as a rank. It ends with the gateway.
  void test_serve(start_options const& options,
                  uint32_t port,
                  std::string const& token,
                  std::string const& init_sql);
  uint64_t waiting();
  std::vector<std::string> ddl_log();

 private:
  class turn;
  struct outcome {
    std::unique_ptr<duckdb::QueryResult> result;
    std::optional<duckdb::ErrorData> error;
  };
  /// When a fan-out cancels its calls: the client interrupting, and the statement timeout.
  struct fan_out_policy {
    bool watch_client  = true;
    uint64_t timeout_s = 0;
    uint64_t grace_s   = 30;
  };
  struct fan_out_result {
    std::vector<outcome> outcomes;
    /// Why the calls were cancelled; empty if they were not.
    std::string cancelled;
    /// Ranks killed because they did not stop within the grace period after the cancel.
    std::vector<uint32_t> killed;
  };

  void start_locked(duckdb::ClientContext& context, start_options options);
  /// Attaches to @p r and opens its session, retrying until @p patience runs out. A registered
  /// rank still running the process the gateway last saw is asked to exit, and its replacement
  /// waited for.
  void connect_rank(rank& r, std::chrono::seconds patience);
  /// Under state_mutex_: the registered rank named @p name, or null.
  rank_spec* find_registered(std::string const& name);
  void stop_locked();
  /// @param replay Whether @p sql comes from the DDL log, so is neither logged nor recovered.
  std::shared_ptr<rank_result> run_locked(duckdb::ClientContext& context,
                                          std::string const& sql,
                                          bool replay);
  /// Runs @p statements on every rank, each rank's in order, ranks in parallel. If @p context is
  /// interrupted or the timeout passes, no rank starts another statement, and quack_cancel is
  /// sent every 100 ms to each rank still running; a rank still running after the grace period
  /// is killed, and a call still stuck after that is abandoned.
  fan_out_result fan_out(duckdb::ClientContext& context,
                         std::vector<std::vector<std::string>> const& statements,
                         fan_out_policy policy);
  fan_out_policy client_policy(duckdb::ClientContext& context);
  /// Under state_mutex_: the cluster is failed for @p reason, unless it already was.
  void fail_locked(std::string reason);
  /// Starts fresh ranks with the last options and replays the DDL log on them.
  void recover_locked(duckdb::ClientContext& context);

  duckdb::DatabaseInstance& db_;

  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::deque<uint64_t> queue_;
  uint64_t next_ticket_ = 0;

  /// Guards the fields below for readers that do not hold the turn.
  std::mutex state_mutex_;
  std::vector<rank> ranks_;
  std::vector<rank_spec> registered_;
  std::vector<rank> test_servers_;
  start_options options_;
  cluster_state state_ = cluster_state::stopped;
  std::string reason_;
  uint64_t query_id_         = 0;
  uint64_t running_query_id_ = 0;
  std::vector<std::string> ddl_log_;
};

/// Rank process lifecycle; `rank_supervisor.cpp`.
/// A local rank runs the DuckDB binary itself; a remote one runs
/// `<rank_launcher> <host> <gpu> <quack port> <duckdb binary> <init file>`.
/// @p init_sql runs before the rank serves Quack.
void spawn_rank(start_options const& options,
                std::string const& token,
                rank& r,
                std::string const& init_sql = {});
/// Whether @p host names an address of one of this machine's interfaces.
bool is_local_host(std::string const& host);
/// @p host as an IPv4 address: NIXL takes only numeric peer addresses.
std::string numeric_address(std::string const& host);
/// Closes the rank's stdin, waits briefly for it to exit, then kills it.
void end_rank(rank& r);
/// Whether the rank's process is gone; reaps it. A registered rank has no process here, and is
/// never gone by this test.
bool rank_exited(rank& r);
std::string log_path(start_options const& options, uint32_t index);

}  // namespace sirius_gateway
