#pragma once

#include "duckdb.hpp"

#include <sys/types.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
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
  uint32_t exchange_base_port = 27000;
  uint32_t quack_base_port    = 9500;
  std::string duckdb_binary;
  /// Empty: CPU ranks without Sirius, for tests.
  std::string sirius_extension;
  /// SQL every rank's session runs after the exchange settings.
  std::string rank_init;
  /// The exchange settings every rank takes before it connects (`sirius_gateway_exchange_*`).
  std::string mnnvl    = "auto";
  uint64_t frame_bytes = 16ull << 20;
  uint32_t send_frames = 32;
  uint32_t recv_frames = 8;
  std::string log_dir;
  /// How a registered rank names its process, whether that process already joined a set of
  /// ranks, and how it is asked to exit for its launcher to restart it
  /// (`sirius_gateway_rank_boot_id_sql`, `sirius_gateway_rank_joined_sql`,
  /// `sirius_gateway_rank_exit_sql`). An empty joined_sql counts every process as joined.
  std::string boot_id_sql;
  std::string joined_sql;
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
  /// The process the gateway last saw: if it has joined, the next join asks it to exit, and
  /// waits for a new one.
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
  /// Exchange address, numeric.
  std::string host;
  std::string quack_host;
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

  /// Starts the ranks afresh. Without @p replay_saved, clears the DDL log and, once the ranks run,
  /// deletes the saved one; with it, keeps the log and replays it as recover_locked does.
  void start(duckdb::ClientContext& context, start_options options, bool replay_saved = false);
  /// Registers a rank started outside the gateway; the next statement rebuilds the ranks with it.
  /// Registering the same rank again, with the same addresses and credentials, changes nothing,
  /// without waiting for the turn; with others, it replaces the rank of that name.
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
    /// Why the cluster is stopped, failed, or waiting for ranks; while it is ready, a saved DDL
    /// entry skipped on replay or a failure to save the DDL log.
    std::string error;
    /// The rank's last failure to join, to answer, or to keep running; empty if none.
    std::string last_error;
  };
  std::vector<rank_status> status();
  uint64_t last_query_id();
  /// The query number of the statement running now, or 0.
  uint64_t running_query_id();
  /// Test hook (`sirius_gateway_test_signal`): sends @p signal to rank @p index's process, or for
  /// a registered rank, to the sirius_gateway_test_serve process that serves it.
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

  /// Before a statement takes its turn, while the registered ranks wait to start: waits until
  /// `sirius_gateway_min_ranks` are registered, for at most `sirius_gateway_join_wait` seconds,
  /// or until the client interrupts. Registering takes the turn, so the wait must not hold it.
  void wait_for_ranks(duckdb::ClientContext& context, std::string const& sql);
  void start_locked(duckdb::ClientContext& context, start_options options);
  /// Attaches to @p r and opens its session, retrying until @p patience runs out or @p give_up
  /// is set. With @p join, a registered rank still running the process the gateway last saw is
  /// asked to exit if that process has joined, and its replacement waited for.
  void connect_rank(rank& r,
                    start_options const& options,
                    std::chrono::seconds patience,
                    bool join,
                    std::atomic<bool> const& give_up);
  /// Ranks whose Quack port takes no connection within @p patience; probed in parallel, outside
  /// state_mutex_. Spawned ranks are left to rank_exited.
  std::vector<uint32_t> unreachable_registered(std::chrono::milliseconds patience);
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
  /// Starts fresh ranks with the last options and replays the DDL log on them; an empty log is
  /// first loaded from `sirius_gateway_state_dir`, as a gateway that restarted has. An entry
  /// from the in-memory log that fails fails the cluster; one loaded from the file that fails on
  /// every rank is dropped from the log and the file, and named in the status error column.
  void recover_locked(duckdb::ClientContext& context);
  /// Before any sirius_gateway_start or sirius_gateway_stop, and with no ranks running: starts
  /// the registered ranks with the current settings and replays the DDL log, as
  /// sirius_gateway_start(0) followed by a restart would. Throws while fewer than
  /// `sirius_gateway_min_ranks` ranks are registered, once wait_for_ranks gave up.
  void auto_start_locked(duckdb::ClientContext& context);

  duckdb::DatabaseInstance& db_;

  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::deque<uint64_t> queue_;
  uint64_t next_ticket_ = 0;

  /// Guards the fields below for readers that do not hold the turn.
  std::mutex state_mutex_;
  /// Signalled when ranks are registered or dropped, and when auto start ends.
  std::condition_variable ranks_cv_;
  /// Statements in wait_for_ranks, and the sirius_gateway_min_ranks they wait for.
  std::size_t join_waiters_ = 0;
  uint64_t join_min_ranks_  = 0;
  /// Per rank name: its last failure, for sirius_gateway_status.
  std::map<std::string, std::string> last_errors_;
  std::vector<rank> ranks_;
  std::vector<rank_spec> registered_;
  std::vector<rank> test_servers_;
  start_options options_;
  cluster_state state_ = cluster_state::stopped;
  /// Whether a statement with no ranks running starts the registered ones itself; an explicit
  /// sirius_gateway_start or sirius_gateway_stop clears it.
  bool auto_start_ = true;
  std::string reason_;
  uint64_t query_id_         = 0;
  uint64_t running_query_id_ = 0;
  std::vector<std::string> ddl_log_;
};

/// Fills @p options from the `sirius_gateway_*` settings of @p context.
void read_settings(duckdb::ClientContext& context, start_options& options);

/// Rank process lifecycle; `rank_supervisor.cpp`.
/// Runs the DuckDB binary on this host; @p init_sql runs before the rank serves Quack.
void spawn_rank(start_options const& options,
                std::string const& token,
                rank& r,
                std::string const& init_sql = {});
/// @p host as an IPv4 address: NIXL takes only numeric peer addresses.
std::string numeric_address(std::string const& host);
/// Whether a TCP connection to @p host:@p port opens within @p patience.
bool accepts_connections(std::string const& host,
                         uint32_t port,
                         std::chrono::milliseconds patience);
/// Closes the rank's stdin, waits briefly for it to exit, then kills it.
void end_rank(rank& r);
/// Whether the rank's process is gone; reaps it. A registered rank has no process here, and is
/// never gone by this test.
bool rank_exited(rank& r);
std::string log_path(start_options const& options, uint32_t index);

}  // namespace sirius_gateway
