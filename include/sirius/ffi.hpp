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

/*
 * Public C++ surface for embedding Sirius (`Context` and `Fragment`).
 * The embedder is the process that links it: Rust `sirius-sys` or C++ tests.
 *
 * Intentionally lightweight — a small RAII wrapper that
 * forward-declares the heavy internal type — so consumers bind it without
 * pulling in sirius_context.hpp (and its cudf/rmm/duckdb includes).
 *
 * Symbols are exported with default visibility for both libsirius and the
 * loadable extension. This header is the public C++ API installed by libsirius.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#ifndef SIRIUS_FFI_EXPORT
#define SIRIUS_FFI_EXPORT __attribute__((visibility("default")))
#endif

namespace sirius::ffi {

class Fragment;

/// Immutable object-store (S3 API) settings of a `Context`: the origin, region, static
/// credentials, transport, signing form and trust anchor used by every remote read of an
/// execution that captured this snapshot. Aggregate — construct with designated initializers,
/// share by pointer, derive a snapshot with new credentials through `with_credentials()`.
///
/// Supported: path-style `s3://` URIs only (`s3a://` and other schemes are refused); static
/// credentials (IAM roles and instance profiles are resolved by the caller); TLS peer
/// verification is always on; no proxy, no client certificates, no virtual-hosted addressing,
/// no automatic refresh. Members are appended only.
struct SIRIUS_FFI_EXPORT ObjectStoreConfig {
  /// Required. `"https://host[:port]"` or `"http://host[:port]"`; no path, query or fragment.
  const std::string endpoint;
  /// Required. SigV4 signing region.
  const std::string region;
  /// Required.
  const std::string access_key;
  /// Required. Never logged and never echoed in error texts.
  const std::string secret_key;
  /// STS temporary-credential session token; empty for long-lived keys.
  const std::string session_token;
  /// `"auto"` | `"http"` | `"rdma"`. `"rdma"` requires a build that registers an S3-over-RDMA
  /// backend and is refused otherwise.
  const std::string transport{"auto"};
  /// `"presigned"` | `"header"`: SigV4 auth in the URL query (default; works everywhere AWS
  /// does) or in the `Authorization` header (for gateways that prefer header auth).
  const std::string signing_mode{"presigned"};
  /// PEM CA bundle used to verify the endpoint's TLS certificate; empty uses the system trust
  /// store.
  const std::string ca_bundle_path;
  /// Namespace this snapshot may read: empty = any `s3://` object at the endpoint;
  /// `"s3://bucket/"` or `"s3://bucket/prefix/"` limits reads to that prefix.
  /// With a non-empty prefix, an item outside it, an empty object key, or a key containing an
  /// empty, `.` or `..` segment or a percent byte is refused before binding (`object_store_scope`).
  const std::string uri_prefix;

  /// A new snapshot with these credentials and every other field copied from `*this`, which is
  /// left unchanged. No validation, no I/O. Use an empty token for long-lived keys.
  [[nodiscard]] std::shared_ptr<ObjectStoreConfig> with_credentials(
    const std::string& new_access_key,
    const std::string& new_secret_key,
    const std::string& new_session_token) const;
};

/// RAII handle to a Sirius engine context.
///
/// Constructing a `Context` brings up an initialized engine (a
/// `duckdb::SiriusContext`) and an embedded in-process DuckDB whose connection
/// has that engine registered as the `sirius_state` so the GPU executor can find
/// it. DuckDB is used only to lower a Substrait plan to a DuckDB
/// `LogicalOperator` (the translation step) and to own the catalog. Execution
/// runs directly on the Sirius engine, not through DuckDB's query pipeline.
///
/// Held from Rust via `cxx::UniquePtr`; created by `make_context()` /
/// `make_context_from_config()` and freed when the `UniquePtr` drops. The
/// constructors can throw (bad config, GPU bring-up failure); the `make_*`
/// factories are bound as fallible so failures surface as errors.
class SIRIUS_FFI_EXPORT Context {
 public:
  Context();
  explicit Context(const std::string& config_path);
  ~Context();

  Context(const Context&)            = delete;
  Context& operator=(const Context&) = delete;

  /// Executes a serialized Substrait plan on the GPU, writing the results to the
  /// Arrow C Data Interface stream at `out_stream_addr` (one schema, a sequence
  /// of record batches). `out_stream_addr` is the address of a caller-owned
  /// `ArrowArrayStream` that the caller releases per the Arrow ABI. Throws on
  /// translation or execution failure. The stream keeps the database context alive until
  /// released; its last release can tear down the engine on the releasing thread.
  void execute_substrait(const std::string& plan, std::uintptr_t out_stream_addr);

  /// The current snapshot when exactly one is published (whatever its `uri_prefix`), otherwise
  /// the default-scope snapshot (empty `uri_prefix`) or nullptr when none. Thread-safe. A
  /// returned snapshot stays valid and unchanged after a later replacement.
  [[nodiscard]] std::shared_ptr<ObjectStoreConfig> object_store_config() const;

  /// The published snapshot that would serve `uri`: the one with the longest `uri_prefix` the
  /// URI starts with (scheme compared case-insensitively, the rest byte-exact; the default scope
  /// matches any URI — which schemes a route supports is decided by the route), or nullptr when
  /// none matches. Administrative observation only: executions resolve against the set they
  /// captured when their binding began. Thread-safe.
  /// @throws std::runtime_error (`object_store_scope:`) when the selected non-default scope
  /// cannot serve the key: an empty, `.` or `..` segment or a `%` byte.
  [[nodiscard]] std::shared_ptr<ObjectStoreConfig> object_store_config_for(
    std::string_view uri) const;

  /// Atomically replace every published snapshot with `cfg` alone (its `uri_prefix` names the
  /// scope); nullptr clears all scopes. Checks `cfg`
  /// (required fields non-empty, endpoint shape, `transport`/`signing_mode` value sets,
  /// `ca_bundle_path` readable, `uri_prefix` shape) and that the configured IO backend honors
  /// every option — an option the backend would ignore is refused, nothing is silently dropped.
  /// No network access and no IO resources are created here; access failures surface when the
  /// snapshot is used. `execute_substrait()` and `Fragment::build()` capture the snapshot
  /// current when their binding begins and keep it for that execution; a replacement affects
  /// later executions only. Thread-safe; may be called while queries run (logically const: the
  /// context identity is unchanged).
  /// @throws std::invalid_argument naming the offending field; the message never carries a
  ///         field value.
  void set_object_store(std::shared_ptr<ObjectStoreConfig> cfg) const;

  /// Publish `cfg` for the scope named by its `uri_prefix` (exact bytes; empty = the default
  /// scope), atomically replacing a snapshot already published for that key and leaving other
  /// keys unchanged. Same checks as set_object_store(). Executions that already captured their
  /// set are unaffected. Thread-safe.
  /// @throws std::invalid_argument for a null `cfg` or an offending field (never a value).
  void put_object_store(std::shared_ptr<ObjectStoreConfig> cfg) const;

  /// Remove the snapshot published for `uri_prefix` (exact bytes; empty = the default scope);
  /// no-op when none is published for it. Thread-safe.
  /// @throws std::invalid_argument for a malformed prefix (never a value); the set is unchanged.
  void drop_object_store(std::string_view uri_prefix) const;

 private:
  struct Impl;
  explicit Context(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;

  friend class Fragment;
  friend SIRIUS_FFI_EXPORT std::unique_ptr<Fragment> make_fragment(Context& context);
  friend SIRIUS_FFI_EXPORT std::unique_ptr<Context> make_context_from_config(
    const std::string& config_path, std::shared_ptr<ObjectStoreConfig> cfg);
};

/// One plan fragment of a multi-fragment query, executed on this process's [`Context`].
///
/// A fragment is either **intermediate** (declares output streams, rooted in a streaming sink)
/// or a **result** fragment (no output streams, produces Arrow). Both kinds may declare input
/// streams fed by other fragments without copying.
///
/// Usage order: declare inputs/outputs → build → relay_from every sender → run →
/// drain via relay_from or result_to_arrow.
///
/// Any number of fragments may be built before any runs; run them in any order where each
/// source runs before its receiver's relay_from. build(), run() and Context::execute_substrait
/// execute one at a time per Context: a concurrent call waits for the one in progress. run() and
/// destruction may happen on a thread other than build()'s.
class SIRIUS_FFI_EXPORT Fragment {
 public:
  ~Fragment();

  Fragment(const Fragment&)            = delete;
  Fragment& operator=(const Fragment&) = delete;

  /// Declare one column of input stream `stream_id` (in plan order). `type` is a DuckDB type
  /// name (`BIGINT`, `DECIMAL(15,2)`, `DATE`, …).
  /// @throws after build().
  void declare_input_column(std::uint64_t stream_id,
                            const std::string& name,
                            const std::string& type);

  /// Declare a sender that must close input stream `stream_id` before it ends. With none
  /// declared the stream expects single sender 0.
  /// @throws after build().
  void declare_input_sender(std::uint64_t stream_id, std::uint32_t sender_id);

  /// Declare the row count of input stream `stream_id`, summed over its senders, so the
  /// optimizer can pick a join's build side. Undeclared streams plan as 1 row. Last call wins.
  /// @throws after build().
  void declare_input_cardinality(std::uint64_t stream_id, std::uint64_t rows);

  /// Declare an output stream. A fragment with no output stream is a result fragment; two or
  /// more need declare_output_broadcast() or declare_output_hash_key(), or build() throws.
  /// @throws after build() or on duplicate id.
  void declare_output(std::uint64_t stream_id);

  /// Every output receives the full fragment output (broadcast sink). Requires at least two
  /// declared outputs: build() rejects a partition mode declared on 0 or 1 outputs rather than
  /// silently ignoring it. Mutually exclusive with declare_output_hash_key.
  /// @throws after build() or after declare_output_hash_key(), or from build() itself when
  /// fewer than two outputs are declared.
  void declare_output_broadcast();

  /// Declare one hash-partition key column for a multi-output sink. Call once per key in
  /// partition-expression order. Requires at least two declared outputs, same as
  /// declare_output_broadcast(). Mutually exclusive with declare_output_broadcast.
  /// @throws after build() or after declare_output_broadcast(), or from build() itself when
  /// fewer than two outputs are declared or the key column is out of range or of an
  /// unsupported type (integer, boolean, varchar, and decimal keys are supported).
  void declare_output_hash_key(std::uint32_t column_index);

  /// Lower and plan `substrait_plan` against the declared streams.
  /// Creates a view `sirius_stream_<id>` for each declared input stream. A failed build() rolls
  /// back and leaves the Fragment unbuilt, so it may be called again.
  /// @throws if already built, on an unknown input type name, a declared input the plan never
  /// reads, an invalid output partitioning (see declare_output*), or translation/planning
  /// failure.
  void build(const std::string& substrait_plan);

  /// Move every batch on `source`'s output stream `source_stream_id` into this fragment's
  /// input stream `input_stream_id`, then close `sender_id` on it. Must be called after
  /// source.run() and before this->run(). Every check below runs before any data moves, except
  /// an input that already ended: that throws on the first batch, which is lost.
  /// @return number of batches moved.
  /// @throws before either fragment's build(), when the source has not run or its run() failed,
  /// after this->run(), when the source is a result fragment or on another Context, on an
  /// undeclared input or unknown stream id, on a sender outside the declared set, on a schema
  /// mismatch, or when the input already ended.
  std::size_t relay_from(Fragment& source,
                         std::uint64_t source_stream_id,
                         std::uint64_t input_stream_id,
                         std::uint32_t sender_id);

  /// Close sender `sender_id` on input stream `stream_id`. EOS mirror for remote senders
  /// (relay_from closes its own sender). Idempotent per sender.
  /// @throws before build() or on unknown stream/sender.
  void close_input(std::uint64_t stream_id, std::uint32_t sender_id);

  /// Execute the fragment and block until pipelines finish. Every input must be closed first
  /// (relay_from and close_input close their sender). Runs once; after a failure, build a new
  /// fragment.
  /// @throws before build(), while an input is open (the fragment stays runnable), on a second
  /// call, or on execution failure.
  void run();

  /// Write this result fragment's rows into the caller-owned ArrowArrayStream at
  /// `out_stream_addr` (Arrow C Data Interface). Same contract as Context::execute_substrait.
  /// Callable once.
  /// @throws on an intermediate fragment, before a successful run(), or when already called.
  void result_to_arrow(std::uintptr_t out_stream_addr);

  /// Batches currently parked on output stream `stream_id`. For diagnostics. 0 before build().
  /// @throws after build() on an unknown id, including any id on a result fragment.
  [[nodiscard]] std::size_t output_batch_count(std::uint64_t stream_id) const;

  /// Total rows parked on output stream `stream_id`, without draining it. A throw means the
  /// count is unknown: callers feeding declare_input_cardinality() leave the stream undeclared.
  /// @throws before build(), on an unknown id, or on a parked batch that is not GPU-resident
  ///         (a spilled batch's row count is not readable without materializing it).
  [[nodiscard]] std::uint64_t output_row_count(std::uint64_t stream_id) const;

  /// DuckDB type-name strings for each output column. Matches what declare_input_column accepts.
  /// @throws before build() or on a result fragment.
  [[nodiscard]] std::unique_ptr<std::vector<std::string>> output_types() const;

 private:
  struct Impl;
  explicit Fragment(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;

  friend SIRIUS_FFI_EXPORT std::unique_ptr<Fragment> make_fragment(Context& context);
};

/// Create a [`Context`] configured from built-in defaults.
SIRIUS_FFI_EXPORT std::unique_ptr<Context> make_context();

/// Create a [`Context`] configured from the YAML file at `config_path`.
SIRIUS_FFI_EXPORT std::unique_ptr<Context> make_context_from_config(const std::string& config_path);

/// Create a [`Context`] configured from the YAML file at `config_path` (empty = built-in
/// defaults) with `cfg` as its object-store snapshot (nullptr = no object store). `cfg` is
/// checked as by `Context::set_object_store()` before the engine is brought up, so a rejected
/// snapshot leaves no engine state. Throws like the other factories (unreadable config file,
/// engine bring-up failure) and `std::invalid_argument` for a rejected `cfg`.
SIRIUS_FFI_EXPORT std::unique_ptr<Context> make_context_from_config(
  const std::string& config_path, std::shared_ptr<ObjectStoreConfig> cfg);

/// Create a [`Fragment`] on `context`. The context must outlive it.
SIRIUS_FFI_EXPORT std::unique_ptr<Fragment> make_fragment(Context& context);

/// DuckDB view name a plan must read to consume input stream `stream_id`.
/// Fragment::build() creates this view; the plan emits a read of this name where a file scan
/// would otherwise appear.
SIRIUS_FFI_EXPORT std::unique_ptr<std::string> stream_view_name(std::uint64_t stream_id);

}  // namespace sirius::ffi
