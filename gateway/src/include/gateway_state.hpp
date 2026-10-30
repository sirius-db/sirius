#pragma once

#include "duckdb.hpp"

#include <sys/types.h>

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace sirius_gateway {

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
};

/// One rank process and the gateway's one session on it.
struct rank {
  uint32_t index = 0;
  pid_t pid      = -1;
  /// The rank reads its stdin until EOF, so closing this ends it.
  int stdin_fd = -1;
  std::string gpu;
  std::string host;
  /// Started through the rank launcher, which forwards `uri` to the rank's host.
  bool remote = false;
  std::string uri;
  std::string catalog;
  /// The only user of `catalog`, CONNECTed to it: the rank's persistent session.
  std::unique_ptr<duckdb::Connection> channel;
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
  void stop(duckdb::ClientContext& context);
  void restart(duckdb::ClientContext& context);
  /// Runs @p sql on every rank and returns rank 0's result.
  std::shared_ptr<rank_result> run(duckdb::ClientContext& context, std::string const& sql);

  struct rank_status {
    uint32_t index;
    int64_t pid;
    std::string gpu;
    std::string host;
    std::string uri;
    std::string state;
    std::string error;
  };
  std::vector<rank_status> status();
  uint64_t last_query_id();
  uint64_t waiting();
  std::vector<std::string> ddl_log();

 private:
  class turn;
  struct outcome {
    std::unique_ptr<duckdb::QueryResult> result;
    std::optional<duckdb::ErrorData> error;
  };

  void start_locked(duckdb::ClientContext& context, start_options options);
  void connect_rank(rank& r, std::string const& token);
  void stop_locked();
  /// @param replay Whether @p sql comes from the DDL log, so is neither logged nor recovered.
  std::shared_ptr<rank_result> run_locked(duckdb::ClientContext& context,
                                          std::string const& sql,
                                          bool replay);
  /// Runs @p statements on every rank, each rank's in order, ranks in parallel. Cancels them on
  /// every rank if @p context is interrupted or the statement timeout passes, and then sets
  /// @p cancelled to the reason.
  std::vector<outcome> fan_out(duckdb::ClientContext& context,
                               std::vector<std::vector<std::string>> const& statements,
                               std::string& cancelled);
  void cancel_all();
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
  start_options options_;
  std::string broken_;
  uint64_t query_id_ = 0;
  std::vector<std::string> ddl_log_;
};

/// Rank process lifecycle; `rank_supervisor.cpp`.
/// A local rank runs the DuckDB binary itself; a remote one runs
/// `<rank_launcher> <host> <gpu> <quack port> <duckdb binary> <init file>`.
void spawn_rank(start_options const& options, std::string const& token, rank& r);
/// Whether @p host names an address of one of this machine's interfaces.
bool is_local_host(std::string const& host);
/// Closes the rank's stdin, waits briefly for it to exit, then kills it.
void end_rank(rank& r);
/// Whether the rank's process is gone; reaps it.
bool rank_exited(rank& r);
std::string log_path(start_options const& options, uint32_t index);

}  // namespace sirius_gateway
