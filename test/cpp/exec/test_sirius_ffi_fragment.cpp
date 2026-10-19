/*
 * Copyright 2025, Sirius Contributors.
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

// Regression coverage for the public FFI Context and Fragment lifecycle (src/sirius_ffi.cpp).
// A build() failure must roll back its open transaction, not commit it.
//
// These tests declare a column type name that TransformStringToLogicalType() can never resolve,
// which fails build() inside resolve_inputs() before `substrait_plan` is parsed, and check what
// the public API can observe: the rollback leaves the connection able to start and fail a
// second, independent Fragment cleanly. Valid fragments are covered in
// test_sirius_ffi_embedder.cpp.

#include "config.hpp"
#include "log/sink.hpp"
#include "sirius/ffi.hpp"

#include <catch.hpp>
#include <duckdb/common/arrow/arrow.hpp>
#include <duckdb/common/exception/transaction_exception.hpp>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <source_location>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Small 2GB-GPU/4GB-host config shared with other [isolated_context] tests.
fs::path isolated_memory_config_path()
{
  std::source_location loc = std::source_location::current();
  return fs::path(loc.file_name()).parent_path().parent_path() / "scan" / "memory.yaml";
}

std::optional<std::string> saved_env(const char* name)
{
  if (auto const* value = std::getenv(name)) { return value; }
  return std::nullopt;
}

void restore_env(const char* name, std::optional<std::string> const& value)
{
  if (value) {
    setenv(name, value->c_str(), 1);
  } else {
    unsetenv(name);
  }
}

struct logging_state_guard {
  explicit logging_state_guard(fs::path root) : test_root(std::move(root)) {}

  ~logging_state_guard()
  {
    duckdb::Config::LOG_BACKEND.swap(backend);
    duckdb::Config::LOG_DIR.swap(log_dir);
    duckdb::Config::LOG_LEVEL.swap(level);
    if (sirius::log::get_sink() != sink) { sirius::log::set_sink(sink); }
    restore_env("SIRIUS_LOG_BACKEND", env_backend);
    restore_env("SIRIUS_LOG_DIR", env_log_dir);
    restore_env("SIRIUS_LOG_LEVEL", env_level);
    std::error_code error;
    fs::remove_all(test_root, error);
  }

  std::string backend{duckdb::Config::LOG_BACKEND};
  std::string log_dir{duckdb::Config::LOG_DIR};
  std::string level{duckdb::Config::LOG_LEVEL};
  std::shared_ptr<sirius::log::sink> sink{sirius::log::get_sink()};
  std::optional<std::string> env_backend{saved_env("SIRIUS_LOG_BACKEND")};
  std::optional<std::string> env_log_dir{saved_env("SIRIUS_LOG_DIR")};
  std::optional<std::string> env_level{saved_env("SIRIUS_LOG_LEVEL")};
  fs::path test_root;
};

void declare_unresolvable_column(sirius::ffi::Fragment& fragment, const std::string& type_name)
{
  fragment.declare_input_column(0, "a", type_name);
}

// Both TransactionContext::Commit() and ::Rollback() clear current_transaction before doing any
// work that can throw, so "does BeginTransaction() work afterward" cannot tell the old
// commit-on-failure bug apart from the fix. What both bugs share is that a broken or skipped
// rollback would leave the transaction open, and the next BeginTransaction() would then
// throw TransactionException("cannot start a transaction within a transaction") — that's the
// one thing worth asserting against here.
void require_build_fails_without_transaction_exception(sirius::ffi::Fragment& fragment)
{
  bool threw_transaction_exception = false;
  bool threw_other                 = false;
  try {
    fragment.build("");
  } catch (const duckdb::TransactionException&) {
    threw_transaction_exception = true;
  } catch (...) {
    threw_other = true;
  }
  REQUIRE_FALSE(threw_transaction_exception);
  REQUIRE(threw_other);
}

// --- push_arrow mid-stream failure ---------------------------------------------------------------

// A Substrait JSON plan that reads the single nullable BIGINT column `a` of input stream 0.
std::string bigint_stream_read_json()
{
  return R"({"relations":[{"root":{"input":{"read":{"baseSchema":{"names":["a"],"struct":{)"
         R"("types":[{"i64":{"nullability":"NULLABILITY_NULLABLE"}}],)"
         R"("nullability":"NULLABILITY_REQUIRED"}},"namedTable":{"names":[")" +
         *sirius::ffi::stream_view_name(0) + R"("]}}},"names":["a"]}}]})";
}

// Releases an Arrow struct whose memory the parent owns.
template <typename T>
void release_borrowed(T* self)
{
  self->release = nullptr;
}

// Schema `struct<a: int64>`.
struct bigint_schema {
  ArrowSchema child{};
  ArrowSchema* children[1]{&child};

  static void release(ArrowSchema* self)
  {
    delete static_cast<bigint_schema*>(self->private_data);
    self->release = nullptr;
  }

  static void export_to(ArrowSchema* out)
  {
    auto* holder          = new bigint_schema;
    holder->child.format  = "l";
    holder->child.name    = "a";
    holder->child.flags   = ARROW_FLAG_NULLABLE;
    holder->child.release = release_borrowed<ArrowSchema>;
    *out                  = ArrowSchema{};
    out->format           = "+s";
    out->name             = "";
    out->n_children       = 1;
    out->children         = holder->children;
    out->release          = release;
    out->private_data     = holder;
  }
};

// One batch `struct<a: int64>` with no nulls.
struct bigint_batch {
  std::vector<std::int64_t> values;
  const void* child_buffers[2]{nullptr, nullptr};
  const void* parent_buffers[1]{nullptr};
  ArrowArray child{};
  ArrowArray* children[1]{&child};

  static void release(ArrowArray* self)
  {
    delete static_cast<bigint_batch*>(self->private_data);
    self->release = nullptr;
  }

  static void export_to(std::vector<std::int64_t> values, ArrowArray* out)
  {
    auto* holder             = new bigint_batch;
    holder->values           = std::move(values);
    holder->child_buffers[1] = holder->values.data();
    auto const length        = static_cast<std::int64_t>(holder->values.size());
    holder->child.length     = length;
    holder->child.n_buffers  = 2;
    holder->child.buffers    = holder->child_buffers;
    holder->child.release    = release_borrowed<ArrowArray>;
    *out                     = ArrowArray{};
    out->length              = length;
    out->n_buffers           = 1;
    out->buffers             = holder->parent_buffers;
    out->n_children          = 1;
    out->children            = holder->children;
    out->release             = release;
    out->private_data        = holder;
  }
};

// A `struct<a: int64>` stream that yields `good_batches` batches, then fails get_next.
struct failing_stream {
  int good_batches{0};
  int served{0};

  static int get_schema(ArrowArrayStream*, ArrowSchema* out)
  {
    bigint_schema::export_to(out);
    return 0;
  }

  static int get_next(ArrowArrayStream* self, ArrowArray* out)
  {
    auto* state = static_cast<failing_stream*>(self->private_data);
    if (state->served == state->good_batches) { return EIO; }
    ++state->served;
    bigint_batch::export_to({1, 2, 3}, out);
    return 0;
  }

  static const char* get_last_error(ArrowArrayStream*) { return "injected get_next failure"; }

  static void release(ArrowArrayStream* self)
  {
    delete static_cast<failing_stream*>(self->private_data);
    self->release = nullptr;
  }

  static ArrowArrayStream make(int good_batches)
  {
    ArrowArrayStream out{};
    out.get_schema     = get_schema;
    out.get_next       = get_next;
    out.get_last_error = get_last_error;
    out.release        = release;
    out.private_data   = new failing_stream{good_batches, 0};
    return out;
  }
};

std::size_t push(sirius::ffi::Fragment& fragment, ArrowArrayStream& stream)
{
  return fragment.push_arrow(0, reinterpret_cast<std::uintptr_t>(&stream), 0);
}

}  // namespace

TEST_CASE("Fragment::build() failure during resolve_inputs() rolls back cleanly",
          "[isolated_context][sirius_ffi]")
{
  auto context = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());

  auto first = sirius::ffi::make_fragment(*context);
  declare_unresolvable_column(*first, "not_a_real_type_xyz");
  REQUIRE_THROWS(first->build(""));

  auto second = sirius::ffi::make_fragment(*context);
  declare_unresolvable_column(*second, "also_not_a_real_type_xyz");
  require_build_fails_without_transaction_exception(*second);
}

TEST_CASE("Fragment destroyed after a failed build() leaves the Context usable",
          "[isolated_context][sirius_ffi]")
{
  auto context = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());

  // The failed fragment goes out of scope with no further use.
  {
    auto first = sirius::ffi::make_fragment(*context);
    declare_unresolvable_column(*first, "not_a_real_type_xyz");
    REQUIRE_THROWS(first->build(""));
  }

  auto second = sirius::ffi::make_fragment(*context);
  declare_unresolvable_column(*second, "also_not_a_real_type_xyz");
  require_build_fails_without_transaction_exception(*second);
}

TEST_CASE("FFI context restores logging settings when sink construction fails",
          "[isolated_context][sirius_ffi]")
{
  auto const test_root = fs::temp_directory_path() / "sirius-ffi-log-dir-rollback-test";
  logging_state_guard restore{test_root};
  fs::remove_all(test_root);
  fs::create_directories(test_root);
  auto const blocker = test_root / "not-a-directory";
  std::ofstream(blocker) << "file";
  REQUIRE(fs::is_regular_file(blocker));
  auto const invalid_dir = blocker / "child";

  duckdb::Config::LOG_BACKEND = "noop";
  duckdb::Config::LOG_DIR     = (test_root / "previous").string();
  duckdb::Config::LOG_LEVEL   = "warn";
  setenv("SIRIUS_LOG_BACKEND", "spdlog", 1);
  setenv("SIRIUS_LOG_DIR", invalid_dir.string().c_str(), 1);
  setenv("SIRIUS_LOG_LEVEL", "debug", 1);

  REQUIRE_THROWS(sirius::ffi::make_context_from_config(isolated_memory_config_path().string()));
  REQUIRE(duckdb::Config::LOG_BACKEND == "noop");
  REQUIRE(duckdb::Config::LOG_DIR == (test_root / "previous").string());
  REQUIRE(duckdb::Config::LOG_LEVEL == "warn");
  REQUIRE(sirius::log::get_sink() == restore.sink);
}

TEST_CASE("FFI push_arrow failing mid-stream poisons the Fragment",
          "[isolated_context][sirius_ffi]")
{
  auto context = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());

  SECTION("a failure after a pushed batch: close_input, relay_from, push_arrow and run throw")
  {
    auto fragment = sirius::ffi::make_fragment(*context);
    fragment->declare_input_column(0, "a", "BIGINT");
    fragment->build_json(bigint_stream_read_json());

    auto stream = failing_stream::make(1);
    REQUIRE_THROWS_AS(push(*fragment, stream), std::runtime_error);
    REQUIRE(stream.release == nullptr);

    auto const names_failure = Catch::Matchers::ContainsSubstring("injected get_next failure");
    REQUIRE_THROWS_AS(fragment->close_input(0, 0), std::logic_error);
    REQUIRE_THROWS_WITH(fragment->close_input(0, 0), names_failure);
    REQUIRE_THROWS_AS(fragment->run(), std::logic_error);
    REQUIRE_THROWS_WITH(fragment->run(), names_failure);
    auto retry = failing_stream::make(1);
    REQUIRE_THROWS_AS(push(*fragment, retry), std::logic_error);
    REQUIRE(retry.release == nullptr);

    // The failure is reported before the source is looked at.
    auto source = sirius::ffi::make_fragment(*context);
    REQUIRE_THROWS_AS(fragment->relay_from(*source, 1, 0, 0), std::logic_error);
  }

  SECTION("a failure before the first batch pushes nothing and leaves the Fragment usable")
  {
    auto fragment = sirius::ffi::make_fragment(*context);
    fragment->declare_input_column(0, "a", "BIGINT");
    fragment->build_json(bigint_stream_read_json());

    auto stream = failing_stream::make(0);
    REQUIRE_THROWS_AS(push(*fragment, stream), std::runtime_error);
    REQUIRE_NOTHROW(fragment->close_input(0, 0));
    REQUIRE_NOTHROW(fragment->run());
  }
}
