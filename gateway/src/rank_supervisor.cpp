#include "duckdb/parser/keyword_helper.hpp"
#include "gateway_state.hpp"

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <thread>

extern char** environ;

namespace sirius_gateway {

using duckdb::IOException;
using duckdb::KeywordHelper;

std::string log_path(start_options const& options, uint32_t index)
{
  return options.log_dir + "/rank_" + std::to_string(index) + ".log";
}

void spawn_rank(start_options const& options, std::string const& token, rank& r)
{
  auto const init = options.log_dir + "/rank_" + std::to_string(r.index) + ".init.sql";
  {
    std::ofstream out(init, std::ios::trunc);
    if (!options.sirius_extension.empty()) {
      out << "LOAD " << KeywordHelper::WriteQuoted(options.sirius_extension) << ";\n";
    }
    out << "CALL quack_serve(" << KeywordHelper::WriteQuoted(r.uri)
        << ", token := " << KeywordHelper::WriteQuoted(token) << ");\n";
    if (!out) { throw IOException("cannot write %s", init); }
  }

  int in[2];
  if (pipe2(in, O_CLOEXEC) != 0) { throw IOException("pipe: %s", std::strerror(errno)); }
  int const log =
    open(log_path(options, r.index).c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (log < 0) {
    close(in[0]);
    close(in[1]);
    throw IOException("cannot open %s: %s", log_path(options, r.index), std::strerror(errno));
  }
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, in[0], STDIN_FILENO);
  posix_spawn_file_actions_adddup2(&actions, log, STDOUT_FILENO);
  posix_spawn_file_actions_adddup2(&actions, log, STDERR_FILENO);

  std::vector<std::string> env;
  for (char** e = environ; *e != nullptr; ++e) {
    if (std::strncmp(*e, "CUDA_VISIBLE_DEVICES=", 21) != 0) { env.emplace_back(*e); }
  }
  env.push_back("CUDA_VISIBLE_DEVICES=" + r.gpu);
  std::vector<std::string> args{options.duckdb_binary, "-unsigned", "-init", init};
  auto pointers = [](std::vector<std::string>& strings) {
    std::vector<char*> result;
    for (auto& s : strings) {
      result.push_back(s.data());
    }
    result.push_back(nullptr);
    return result;
  };
  auto argv = pointers(args);
  auto envp = pointers(env);
  int const rc =
    posix_spawn(&r.pid, options.duckdb_binary.c_str(), &actions, nullptr, argv.data(), envp.data());
  posix_spawn_file_actions_destroy(&actions);
  close(in[0]);
  close(log);
  if (rc != 0) {
    close(in[1]);
    r.pid = -1;
    throw IOException(
      "cannot start rank %d with %s: %s", r.index, options.duckdb_binary, std::strerror(rc));
  }
  r.stdin_fd = in[1];
}

bool rank_exited(rank& r)
{
  if (r.pid < 0) { return true; }
  int status    = 0;
  auto const rc = waitpid(r.pid, &status, WNOHANG);
  if (rc == 0) { return false; }
  r.pid = -1;
  return true;
}

void end_rank(rank& r)
{
  if (r.stdin_fd >= 0) {
    close(r.stdin_fd);
    r.stdin_fd = -1;
  }
  for (int i = 0; i < 50 && !rank_exited(r); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (r.pid >= 0) {
    kill(r.pid, SIGKILL);
    waitpid(r.pid, nullptr, 0);
    r.pid = -1;
  }
}

}  // namespace sirius_gateway
