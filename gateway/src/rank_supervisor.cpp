#include "duckdb/parser/keyword_helper.hpp"
#include "gateway_state.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstring>
#include <filesystem>
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

void spawn_rank(start_options const& options,
                std::string const& token,
                rank& r,
                std::string const& init_sql)
{
  auto const init = options.log_dir + "/rank_" + std::to_string(r.index) + ".init.sql";
  {
    std::ofstream out(init, std::ios::trunc);
    if (!options.sirius_extension.empty()) {
      out << "LOAD " << KeywordHelper::WriteQuoted(options.sirius_extension) << ";\n";
    }
    if (!init_sql.empty()) { out << init_sql << ";\n"; }
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
  // The rank keeps none of the gateway's other open files: a launch script's lock (fd 9 in
  // sirius_gateway.sh) held by a rank that outlives the gateway would block the next gateway.
  // The sysroot's glibc predates posix_spawn_file_actions_addclosefrom_np; a close action on an
  // fd that is closed by the time of the spawn is ignored.
  std::error_code no_proc;  // without /proc, the rank inherits them
  for (auto const& entry : std::filesystem::directory_iterator("/proc/self/fd", no_proc)) {
    int fd          = -1;
    auto const name = entry.path().filename().string();
    if (std::from_chars(name.data(), name.data() + name.size(), fd).ec == std::errc{} &&
        fd > STDERR_FILENO) {
      posix_spawn_file_actions_addclose(&actions, fd);
    }
  }

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
    posix_spawn(&r.pid, args.front().c_str(), &actions, nullptr, argv.data(), envp.data());
  posix_spawn_file_actions_destroy(&actions);
  close(in[0]);
  close(log);
  if (rc != 0) {
    close(in[1]);
    r.pid = -1;
    throw IOException("cannot start rank %d with %s: %s", r.index, args.front(), std::strerror(rc));
  }
  r.stdin_fd = in[1];
}

std::string numeric_address(std::string const& host)
{
  addrinfo hints{};
  hints.ai_family   = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* found   = nullptr;
  if (getaddrinfo(host.c_str(), nullptr, &hints, &found) != 0 || found == nullptr) {
    throw IOException("cannot resolve the exchange address %s", host);
  }
  char address[INET_ADDRSTRLEN];
  inet_ntop(AF_INET,
            &reinterpret_cast<sockaddr_in const*>(found->ai_addr)->sin_addr,
            address,
            sizeof(address));
  freeaddrinfo(found);
  return address;
}

bool accepts_connections(std::string const& host, uint32_t port, std::chrono::milliseconds patience)
{
  addrinfo hints{};
  hints.ai_family   = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* found   = nullptr;
  if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &found) != 0) {
    return false;
  }
  bool open = false;
  for (auto* a = found; a != nullptr && !open; a = a->ai_next) {
    int const fd = socket(a->ai_family, a->ai_socktype | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) { continue; }
    if (connect(fd, a->ai_addr, a->ai_addrlen) == 0) {
      open = true;
    } else if (errno == EINPROGRESS) {
      pollfd p{fd, POLLOUT, 0};
      int error       = 0;
      socklen_t bytes = sizeof(error);
      open            = poll(&p, 1, static_cast<int>(patience.count())) == 1 &&
             getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &bytes) == 0 && error == 0;
    }
    close(fd);
  }
  freeaddrinfo(found);
  return open;
}

bool rank_exited(rank& r)
{
  if (r.registered) { return false; }
  if (r.pid < 0) { return true; }
  int status    = 0;
  auto const rc = waitpid(r.pid, &status, WNOHANG);
  if (rc == 0) { return false; }
  r.pid = -1;
  return true;
}

void end_rank(rank& r)
{
  if (r.registered) { return; }  // its launcher owns its process
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
