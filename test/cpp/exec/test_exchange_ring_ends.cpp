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

// The ring exchange's engine ends, end to end in one process: a batch is exported, staged frame
// by frame into the send ring, each frame copied into the receive ring as a peer's NIXL WRITE
// would, copied out to wherever it landed, sealed, and taken from the exchange inbox.

#include "operator/operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/hashing.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <compression/compressed_representation.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/data/disk_io_backend.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <data/convertible_data_batch.hpp>
#include <data/sirius_converter_registry.hpp>
#include <exec/exchange_inbox.hpp>
#include <exec/exchange_ring_ends.hpp>
#include <fcntl.h>
#include <memory/sirius_memory_reservation_manager.hpp>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

namespace utils = sirius::test::operator_utils;
using cucascade::memory::Tier;
using sirius::exec::exchange_inbox;
using sirius::exec::ring_exchange;
using sirius::exec::ring_no_room;
using sirius::exec::ring::batch_kind;

constexpr std::size_t MiB = std::size_t{1} << 20;

/// GPU, host and disk spaces with the converters registered, and an exchange inbox on the GPU.
struct ring_env {
  fs::path dir = fs::temp_directory_path() / "sirius_exchange_ring_test";
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu  = nullptr;
  cucascade::memory::memory_space* host = nullptr;
  cucascade::memory::memory_space* disk = nullptr;
  std::shared_ptr<exchange_inbox> exchange;

  ring_env()
  {
    fs::create_directories(dir);
    sirius::converter_registry::reset_for_testing();
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_gpu_usage_limit(1ull << 30)
      .set_reservation_fraction_per_gpu(0.9)
      .set_per_numa_region_capacity(1ull << 30)
      .use_gpu_id_as_host_id()
      .set_reservation_fraction_per_numa_region(0.9)
      .set_disk_mounting_point(0, 4ull << 30, dir.string());
    mgr = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
    sirius::converter_registry::initialize();
    gpu  = mgr->get_memory_space(Tier::GPU, 0);
    host = mgr->get_memory_space(Tier::HOST, 0);
    disk = const_cast<cucascade::memory::memory_space*>(
      mgr->get_memory_spaces_for_tier(Tier::DISK).front());
    exchange = std::make_shared<exchange_inbox>(*gpu);
  }
  ~ring_env()
  {
    exchange->close();
    fs::remove_all(dir);
  }
};

ring_env& env()
{
  static ring_env e;
  // Another suite in this process may have shut the global registry down since.
  sirius::converter_registry::initialize();
  return e;
}

/// 1 MiB frames, so a batch of a few MiB takes several.
std::unique_ptr<ring_exchange> make_ring(std::size_t gpu_bytes  = 256 * MiB,
                                         std::size_t host_bytes = 256 * MiB,
                                         bool host              = true,
                                         bool disk              = true)
{
  ring_exchange::options opts;
  opts.frame_bytes = MiB;
  opts.send_bytes  = 4 * MiB;
  opts.recv_bytes  = 4 * MiB;
  auto ring        = std::make_unique<ring_exchange>(0, opts);
  ring->attach(env().exchange,
               {host ? env().host : nullptr, disk ? env().disk : nullptr, gpu_bytes, host_bytes});
  return ring;
}

void null_every_seventh(cudf::column& column, ::cuda::stream_ref stream)
{
  auto const rows = column.size();
  std::vector<std::uint32_t> words(
    cudf::bitmask_allocation_size_bytes(rows) / sizeof(std::uint32_t), 0);
  cudf::size_type nulls = 0;
  for (cudf::size_type i = 0; i < rows; ++i) {
    if (i % 7 == 0) {
      ++nulls;
    } else {
      words[static_cast<std::size_t>(i) / 32] |= 1u << (i % 32);
    }
  }
  rmm::device_buffer mask(words.data(), words.size() * sizeof(std::uint32_t), stream);
  column.set_null_mask(std::move(mask), nulls);
}

template <typename T>
std::unique_ptr<cudf::column> numeric_column(cudf::data_type type, std::vector<T> const& values)
{
  auto stream = utils::default_stream();
  auto column = cudf::make_fixed_width_column(type,
                                              static_cast<cudf::size_type>(values.size()),
                                              cudf::mask_state::UNALLOCATED,
                                              stream,
                                              utils::get_resource_ref(*env().gpu));
  cudaMemcpy(column->mutable_view().head(),
             values.data(),
             values.size() * sizeof(T),
             cudaMemcpyHostToDevice);
  return column;
}

/// About 7 MiB over 200k rows: INT64s, nullable INT32s, DECIMAL(38,2) and strings.
std::unique_ptr<cudf::table> sample_table(std::size_t rows = 200'000)
{
  auto stream = utils::default_stream();
  std::vector<std::int64_t> keys(rows);
  std::iota(keys.begin(), keys.end(), std::int64_t{1'000'000});
  std::vector<std::int32_t> small(rows);
  std::vector<__int128> amounts(rows);
  std::vector<std::string> flags(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    small[i]   = static_cast<std::int32_t>(i % 1000);
    amounts[i] = static_cast<__int128>(i % 50'000) * 100 + 25;
    flags[i]   = (i % 3 == 0) ? "RETURNED" : (i % 3 == 1 ? "NONE" : "DELIVER IN PERSON");
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(numeric_column(cudf::data_type{cudf::type_id::INT64}, keys));
  columns.push_back(numeric_column(cudf::data_type{cudf::type_id::INT32}, small));
  null_every_seventh(*columns.back(), stream);
  columns.push_back(numeric_column(cudf::data_type{cudf::type_id::DECIMAL128, -2}, amounts));
  columns.push_back(utils::make_string_column(flags, stream, utils::get_resource_ref(*env().gpu)));
  return std::make_unique<cudf::table>(std::move(columns));
}

std::shared_ptr<cucascade::data_batch> gpu_batch(std::unique_ptr<cudf::table> table)
{
  auto stream = utils::default_stream();
  auto batch  = sirius::make_data_batch(
    std::move(table), *env().gpu, stream, sirius::telemetry::batch_telemetry_info{});
  batch->to_mutable().get_data()->record_writer_event(stream);
  return batch;
}

struct table_digest {
  std::vector<std::uint32_t> row_hashes;
  std::vector<cudf::size_type> null_counts;
  bool operator==(table_digest const&) const = default;
};

table_digest digest(cudf::table_view const& view)
{
  auto stream = utils::default_stream();
  auto hashes = cudf::hashing::murmurhash3_x86_32(view, cudf::DEFAULT_HASH_SEED, stream);
  table_digest out;
  out.row_hashes = utils::copy_column_to_host<std::uint32_t>(hashes->view(), stream);
  for (auto const& column : view) {
    out.null_counts.push_back(column.null_count());
  }
  return out;
}

/// The digest of @p batch, brought back to the GPU first.
table_digest digest(cucascade::data_batch& batch)
{
  env().exchange->bring_to_gpu(batch);
  auto ro = batch.to_read_only();
  if (cudaEvent_t writer = ro.get_writer_event()) { cudaEventSynchronize(writer); }
  return digest(ro.get_data()->cast<cucascade::gpu_table_representation>().get_table_view());
}

void move_to(std::shared_ptr<cucascade::data_batch> const& batch,
             cucascade::memory::memory_space* space)
{
  rmm::cuda_stream stream;
  sirius::convertible_data_batch convertible(batch);
  REQUIRE(convertible.convert({space}, stream.view(), *env().mgr, true).has_value());
}

template <typename Representation>
bool holds(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return dynamic_cast<Representation const*>(ro.get_data()) != nullptr;
}

void wait(ring_exchange& ends, std::uint64_t ticket)
{
  while (!ends.done(ticket)) {
    std::this_thread::yield();
  }
}

/// Ships @p job from @p from to @p to as the transport would, frame by frame, through the two
/// rings; returns the receiver's landing and its exchange-inbox token.
std::pair<ring_exchange::landing, std::uint64_t> ship(ring_exchange& from,
                                                      ring_exchange& to,
                                                      ring_exchange::send_job const& job)
{
  auto const landing = to.begin(job.header);
  for (std::uint32_t k = 0; k < landing.frames; ++k) {
    auto const send = from.send_ring().lease();
    auto const recv = to.recv_ring().lease();
    REQUIRE(send.has_value());
    REQUIRE(recv.has_value());
    wait(from, from.stage(job.token, k, *send));
    auto const len = std::min<std::uint64_t>(
      from.frame_bytes(), job.stream_bytes - std::uint64_t{k} * from.frame_bytes());
    // The peer's WRITE.
    REQUIRE(cudaMemcpy(reinterpret_cast<void*>(to.recv_ring().frame_address(*recv)),
                       reinterpret_cast<void const*>(from.send_ring().frame_address(*send)),
                       len,
                       cudaMemcpyDeviceToDevice) == cudaSuccess);
    // A device-to-device cudaMemcpy may return before it is done, and the ring's streams do not
    // wait for the legacy stream; NIXL's notification comes after its WRITE is.
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    from.send_ring().release(*send);
    wait(to, to.copy_out(landing.id, k, *recv, len));
    // I2: the frame is free once its copy-out is done, whatever happens to the batch; a later
    // WRITE into it must not change what landed.
    REQUIRE(cudaMemset(reinterpret_cast<void*>(to.recv_ring().frame_address(*recv)), 0xab, len) ==
            cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    to.recv_ring().release(*recv);
  }
  from.release_job(job.token);
  return {landing, to.seal(landing.id)};
}

}  // namespace

TEST_CASE("a GPU batch lands on the GPU through the rings", "[exchange_ring][ends]")
{
  auto sender         = make_ring();
  auto receiver       = make_ring();
  auto batch          = gpu_batch(sample_table());
  auto const expected = digest(*batch);

  auto const job = sender->export_batch(batch);
  REQUIRE(job.has_value());
  CHECK(job->kind == batch_kind::gpu_columns);
  CHECK(job->rows == 200'000);
  CHECK(job->stream_bytes > 5 * MiB);
  batch.reset();

  auto const [landing, token] = ship(*sender, *receiver, *job);
  CHECK(landing.tier == Tier::GPU);
  CHECK(landing.frames == (job->stream_bytes + MiB - 1) / MiB);
  CHECK(receiver->stats().gpu_held > 0);
  auto received = env().exchange->take_batch(token);
  CHECK(holds<cucascade::gpu_table_representation>(*received));
  CHECK(digest(*received) == expected);
  // Taking the batch frees its landing budget.
  CHECK(receiver->stats().gpu_held == 0);
  CHECK(receiver->stats().landed_batches[0] == 1);
}

TEST_CASE("a batch lands on host when the GPU budget is spent", "[exchange_ring][ends]")
{
  auto sender         = make_ring();
  auto receiver       = make_ring(/*gpu_bytes=*/0);
  auto batch          = gpu_batch(sample_table());
  auto const expected = digest(*batch);
  auto const job      = sender->export_batch(batch);
  REQUIRE(job.has_value());

  auto const [landing, token] = ship(*sender, *receiver, *job);
  CHECK(landing.tier == Tier::HOST);
  CHECK(receiver->stats().host_held == job->stream_bytes);
  auto received = env().exchange->take_batch(token);
  // A standard host batch: the stream is laid out like cuCascade's host tables.
  CHECK(holds<cucascade::host_data_representation>(*received));
  CHECK(receiver->stats().host_held == 0);
  CHECK(digest(*received) == expected);
}

TEST_CASE("a batch lands on disk when the GPU and host shares are spent", "[exchange_ring][ends]")
{
  auto sender         = make_ring();
  auto receiver       = make_ring(/*gpu_bytes=*/0, /*host_bytes=*/0);
  auto batch          = gpu_batch(sample_table());
  auto const expected = digest(*batch);
  auto const job      = sender->export_batch(batch);
  REQUIRE(job.has_value());

  auto const [landing, token] = ship(*sender, *receiver, *job);
  CHECK(landing.tier == Tier::DISK);
  auto received = env().exchange->take_batch(token);
  CHECK(holds<cucascade::disk_data_representation>(*received));
  CHECK(digest(*received) == expected);
}

TEST_CASE("a spilled sealed landing frees its GPU charge only with return_spilled",
          "[exchange_ring][ends]")
{
  for (bool const return_spilled : {false, true}) {
    auto sender = make_ring();
    ring_exchange::options opts;
    opts.frame_bytes    = MiB;
    opts.send_bytes     = 4 * MiB;
    opts.recv_bytes     = 4 * MiB;
    opts.return_spilled = return_spilled;
    ring_exchange receiver(0, opts);
    receiver.attach(env().exchange, {env().host, env().disk, 256 * MiB, 256 * MiB});
    auto batch          = gpu_batch(sample_table());
    auto const expected = digest(*batch);
    auto const job      = sender->export_batch(batch);
    REQUIRE(job.has_value());
    batch.reset();

    auto const [landing, token] = ship(*sender, receiver, *job);
    REQUIRE(landing.tier == Tier::GPU);
    auto const charged = receiver.stats().gpu_held;
    CHECK(charged > 0);
    // The downgrade executor spills it before its receiver takes it.
    move_to(env().exchange->peek_batch(token), env().host);
    CHECK(receiver.stats().gpu_held == (return_spilled ? 0 : charged));
    auto received = env().exchange->take_batch(token);
    CHECK(receiver.stats().gpu_held == 0);  // given back once
    CHECK(digest(*received) == expected);
  }
}

TEST_CASE("a throw in seal() gives back the landing's charge", "[exchange_ring][ends]")
{
  auto sender   = make_ring();
  auto receiver = make_ring();
  // adopt() refuses a batch once the exchange is closed.
  auto closed = std::make_shared<exchange_inbox>(*env().gpu);
  receiver->attach(closed, {env().host, env().disk, 256 * MiB, 256 * MiB});
  closed->close();
  auto const job = sender->export_batch(gpu_batch(sample_table()));
  REQUIRE(job.has_value());
  CHECK_THROWS_AS(ship(*sender, *receiver, *job), sirius::invalid_input_exception);
  CHECK(receiver->stats().gpu_held == 0);
}

/// A disk whose every write takes 20 ms more: a slow file system.
class slow_disk final : public cucascade::idisk_io_backend {
 public:
  static constexpr auto delay = std::chrono::milliseconds{20};

  void write(fs::path const& path,
             void const* dev_ptr,
             std::size_t size,
             std::size_t file_offset,
             ::cuda::stream_ref stream) override
  {
    std::vector<char> host(size);
    REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);
    REQUIRE(cudaMemcpy(host.data(), dev_ptr, size, cudaMemcpyDeviceToHost) == cudaSuccess);
    write(path, host.data(), size, file_offset);
  }

  void read(fs::path const& path,
            void* dev_ptr,
            std::size_t size,
            std::size_t file_offset,
            ::cuda::stream_ref) override
  {
    std::vector<char> host(size);
    read(path, host.data(), size, file_offset);
    REQUIRE(cudaMemcpy(dev_ptr, host.data(), size, cudaMemcpyHostToDevice) == cudaSuccess);
  }

  void write(fs::path const& path,
             void const* host_ptr,
             std::size_t size,
             std::size_t file_offset) override
  {
    std::this_thread::sleep_for(delay);
    int const fd = ::open(path.c_str(), O_CREAT | O_WRONLY, 0644);
    auto const n = ::pwrite(fd, host_ptr, size, static_cast<off_t>(file_offset));
    ::close(fd);
    if (n != static_cast<ssize_t>(size)) { throw std::runtime_error("slow_disk: short write"); }
  }

  void read(fs::path const& path,
            void* host_ptr,
            std::size_t size,
            std::size_t file_offset) override
  {
    int const fd = ::open(path.c_str(), O_RDONLY);
    auto const n = ::pread(fd, host_ptr, size, static_cast<off_t>(file_offset));
    ::close(fd);
    if (n != static_cast<ssize_t>(size)) { throw std::runtime_error("slow_disk: short read"); }
  }
};

TEST_CASE("disk landings written by several threads overlap", "[exchange_ring][ends]")
{
  auto const dir = env().dir / "slow";
  fs::create_directories(dir);
  cucascade::memory::memory_space disk(
    cucascade::memory::disk_memory_space_config{1, 4ull << 30, dir.string()},
    std::make_shared<slow_disk>());
  auto const table    = sample_table();
  auto const expected = digest(table->view());

  // Every frame of one batch reaches the receive ring before any copy-out, as from a sender
  // that runs ahead: its disk writes are all queued at once. Returns how long they took.
  auto const land = [&](std::size_t writers) {
    ring_exchange::options opts;
    opts.frame_bytes  = MiB;
    opts.send_bytes   = 16 * MiB;
    opts.recv_bytes   = 16 * MiB;
    opts.disk_writers = writers;
    ring_exchange sender(0, opts);
    ring_exchange receiver(0, opts);
    sender.attach(env().exchange, {env().host, env().disk, 256 * MiB, 256 * MiB});
    receiver.attach(env().exchange, {nullptr, &disk, 0, 0});

    auto const job = sender.export_batch(gpu_batch(std::make_unique<cudf::table>(*table)));
    REQUIRE(job.has_value());
    auto const landing = receiver.begin(job->header);
    REQUIRE(landing.tier == Tier::DISK);
    REQUIRE(landing.frames <= receiver.recv_ring().frames());
    std::vector<std::uint32_t> frames;
    for (std::uint32_t k = 0; k < landing.frames; ++k) {
      auto const send = sender.send_ring().lease();
      auto const recv = receiver.recv_ring().lease();
      REQUIRE(send.has_value());
      REQUIRE(recv.has_value());
      wait(sender, sender.stage(job->token, k, *send));
      auto const len = std::min<std::uint64_t>(MiB, job->stream_bytes - std::uint64_t{k} * MiB);
      REQUIRE(cudaMemcpy(reinterpret_cast<void*>(receiver.recv_ring().frame_address(*recv)),
                         reinterpret_cast<void const*>(sender.send_ring().frame_address(*send)),
                         len,
                         cudaMemcpyDeviceToDevice) == cudaSuccess);
      REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
      sender.send_ring().release(*send);
      frames.push_back(*recv);
    }
    sender.release_job(job->token);

    auto const start = std::chrono::steady_clock::now();
    std::vector<std::uint64_t> tickets;
    for (std::uint32_t k = 0; k < landing.frames; ++k) {
      auto const len = std::min<std::uint64_t>(MiB, job->stream_bytes - std::uint64_t{k} * MiB);
      tickets.push_back(receiver.copy_out(landing.id, k, frames[k], len));
    }
    // I1: a copy-out queues its write and returns; the transport never waits on the disk.
    CHECK(std::chrono::steady_clock::now() - start < slow_disk::delay);
    for (auto const ticket : tickets) {
      wait(receiver, ticket);
    }
    auto const took = std::chrono::steady_clock::now() - start;
    for (auto const frame : frames) {
      receiver.recv_ring().release(frame);
    }
    auto received = env().exchange->take_batch(receiver.seal(landing.id));
    CHECK(holds<cucascade::disk_data_representation>(*received));
    CHECK(digest(*received) == expected);
    return took;
  };

  auto const serial   = land(1);
  auto const parallel = land(4);
  CHECK(serial >= 7 * slow_disk::delay);
  CHECK(parallel * 2 < serial);
  fs::remove_all(dir);
}

TEST_CASE("I5: no room in any tier is the one failure, with every tier's numbers",
          "[exchange_ring][ends]")
{
  auto sender    = make_ring();
  auto receiver  = make_ring(/*gpu_bytes=*/0, /*host_bytes=*/0, /*host=*/true, /*disk=*/false);
  auto const job = sender->export_batch(gpu_batch(sample_table()));
  REQUIRE(job.has_value());
  CHECK_THROWS_MATCHES(
    receiver->begin(job->header),
    ring_no_room,
    Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring("no tier has room") &&
                                    Catch::Matchers::ContainsSubstring("GPU needs") &&
                                    Catch::Matchers::ContainsSubstring("host ") &&
                                    Catch::Matchers::ContainsSubstring("disk none")));
  CHECK(receiver->stats().refused == 1);
  sender->release_job(job->token);
}

TEST_CASE("I1: landing never waits on GPU memory", "[exchange_ring][ends]")
{
  auto sender    = make_ring();
  auto receiver  = make_ring();
  auto const job = sender->export_batch(gpu_batch(sample_table()));
  REQUIRE(job.has_value());
  // Someone else holds almost the whole GPU space: the landing goes to host at once.
  std::unique_ptr<cucascade::memory::reservation> held;
  for (auto bytes = env().gpu->get_available_memory(); !held && bytes > 64 * MiB; bytes -= MiB) {
    held = env().gpu->make_reservation_or_null(bytes);
  }
  REQUIRE(held);
  REQUIRE_FALSE(env().gpu->make_reservation_or_null(8 * MiB));
  auto const started = std::chrono::steady_clock::now();
  auto const landing = receiver->begin(job->header);
  CHECK(std::chrono::steady_clock::now() - started < std::chrono::seconds(1));
  CHECK(landing.tier == Tier::HOST);
  receiver->abort(landing.id);
  CHECK(receiver->stats().host_held == 0);
  sender->release_job(job->token);
}

TEST_CASE("a batch spilled to host ships raw from host", "[exchange_ring][ends]")
{
  auto sender         = make_ring();
  auto receiver       = make_ring();
  auto batch          = gpu_batch(sample_table());
  auto const expected = digest(*batch);
  move_to(batch, env().host);
  REQUIRE(holds<cucascade::host_data_representation>(*batch));
  auto const job = sender->export_batch(batch);
  REQUIRE(job.has_value());
  CHECK(job->kind == batch_kind::host_columns);
  // The sender did not bring it back to the GPU.
  CHECK(holds<cucascade::host_data_representation>(*batch));
  batch.reset();

  auto const [landing, token] = ship(*sender, *receiver, *job);
  CHECK(landing.tier == Tier::GPU);
  auto received = env().exchange->take_batch(token);
  CHECK(digest(*received) == expected);
}

TEST_CASE("a landing refuses frames out of order of its stream", "[exchange_ring][ends]")
{
  auto sender    = make_ring();
  auto receiver  = make_ring();
  auto const job = sender->export_batch(gpu_batch(sample_table()));
  REQUIRE(job.has_value());
  auto const landing = receiver->begin(job->header);
  auto const frame   = receiver->recv_ring().lease();
  REQUIRE(frame.has_value());
  CHECK_THROWS_WITH(receiver->copy_out(landing.id, landing.frames, *frame, MiB),
                    Catch::Matchers::ContainsSubstring("out of range"));
  CHECK_THROWS_WITH(receiver->copy_out(landing.id, 0, *frame, 5),
                    Catch::Matchers::ContainsSubstring("the stream needs"));
  wait(*receiver, receiver->copy_out(landing.id, 0, *frame, MiB));
  CHECK_THROWS_WITH(receiver->copy_out(landing.id, 0, *frame, MiB),
                    Catch::Matchers::ContainsSubstring("already copied"));
  CHECK_THROWS_WITH(receiver->seal(landing.id), Catch::Matchers::ContainsSubstring("frames"));
  receiver->abort(landing.id);
  receiver->recv_ring().release(*frame);
  sender->release_job(job->token);
  CHECK(receiver->stats().gpu_held == 0);
}

TEST_CASE("a batch a ring job holds is not spilled under it", "[exchange_ring][ends]")
{
  auto sender = make_ring();
  auto batch  = gpu_batch(sample_table());
  move_to(batch, env().host);
  REQUIRE(holds<cucascade::host_data_representation>(*batch));
  // A spill candidate snapshotted while the batch was still in its repository.
  sirius::convertible_data_batch stale(batch, /*skip_subscribed=*/true);
  auto const job = sender->export_batch(batch);
  REQUIRE(job.has_value());
  CHECK(batch->get_subscriber_count() == 1);
  rmm::cuda_stream stream;
  CHECK_FALSE(stale.convert({env().disk}, stream.view(), *env().mgr, true).has_value());
  CHECK(holds<cucascade::host_data_representation>(*batch));
  sender->release_job(job->token);
  CHECK(batch->get_subscriber_count() == 0);
  // Released, it spills as before.
  CHECK(stale.convert({env().disk}, stream.view(), *env().mgr, true).has_value());
  CHECK(holds<cucascade::disk_data_representation>(*batch));
}

TEST_CASE("a batch spilled to disk ships from its file", "[exchange_ring][ends]")
{
  auto sender         = make_ring();
  auto receiver       = make_ring();
  auto batch          = gpu_batch(sample_table());
  auto const expected = digest(*batch);
  move_to(batch, env().disk);
  REQUIRE(holds<cucascade::disk_data_representation>(*batch));
  auto const job = sender->export_batch(batch);
  REQUIRE(job.has_value());
  CHECK(job->kind == batch_kind::host_columns);
  // Not brought back to the GPU, nor to host.
  CHECK(holds<cucascade::disk_data_representation>(*batch));
  batch.reset();

  auto const [landing, token] = ship(*sender, *receiver, *job);
  CHECK(landing.tier == Tier::GPU);
  auto received = env().exchange->take_batch(token);
  CHECK(digest(*received) == expected);
}
