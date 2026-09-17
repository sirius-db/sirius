// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "codegen/jit/kernel_cache.hpp"
#include "codegen/plan/plan_interpreter.hpp"

#include <cudf/scalar/scalar.hpp>

#include <cstddef>
#include <functional>
#include <list>
#include <optional>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace simpatico {

class decode_frame;
class decode_session;

/** A non-owning handle to a stable, preregistered column owner. Adoption is single-assignment. */
class decode_column_slot final {
 public:
  void adopt(std::unique_ptr<cudf::column> column) const noexcept;
  [[nodiscard]] cudf::column& get() const;
  [[nodiscard]] cudf::column* operator->() const { return &get(); }
  [[nodiscard]] cudf::column_view view() const { return get().view(); }
  [[nodiscard]] explicit operator bool() const noexcept { return owner_ && *owner_; }

 private:
  friend class decode_frame;
  explicit decode_column_slot(std::unique_ptr<cudf::column>& owner) : owner_(&owner) {}
  std::unique_ptr<cudf::column>* owner_;
};

struct mask_destination {
  std::uint32_t* words;
  std::int64_t num_rows;
};

struct value_result {
  std::optional<cudf::data_type> stored_type;
};

struct predicate_result {
  decode_predicate predicate;
  std::optional<mask_destination> ballot;
};

/** A validated copy of the public selection descriptor; borrowed device storage outlives finish. */
class validated_selection final {
 public:
  explicit validated_selection(PlanTree const& plan, decode_selection const& selection);
  [[nodiscard]] decode_selection const& get() const noexcept { return selection_; }

 private:
  decode_selection selection_;
};

using decode_source =
  std::variant<std::reference_wrapper<PlanTree const>,
               std::reference_wrapper<standalone_compressed_representation const>>;

struct column_decode_request {
  decode_source source;
  std::variant<value_result, predicate_result> result = value_result{};
  std::optional<validated_selection> selection;
};

struct membership_source {
  decltype(sirius::codegen::membership_filter_directive::probe) probe;
  cudf::data_type stored_type;
};

struct mask_decode_request {
  PlanTree const& plan;
  std::variant<sirius::codegen::range_predicate, membership_source> source;
  mask_destination destination;
};

enum class mask_source_status { accepted, declined };

/**
 * Per-request storage for the single decode walk and its leaves. All helpers run on the submitting
 * CPU thread. Device and host dependencies are registered before enqueue and remain owned until the
 * enclosing session proves completion. Adopting an already-created owner drains the stream if host
 * bookkeeping throws before ownership transfers.
 *
 * `retained_device_bytes()` is a pressure estimate combining direct RMM buffer capacities, owned
 * column `alloc_size()` values (including representation-owned columns) and supplied scalar byte
 * estimates. It excludes hidden column capacity beyond buffer sizes, allocator/pool overhead,
 * borrowed input and terminal output storage. It is not a physical-memory bound.
 */
class decode_frame final {
 public:
  ~decode_frame();
  decode_frame(decode_frame const&)            = delete;
  decode_frame& operator=(decode_frame const&) = delete;

  [[nodiscard]] rmm::cuda_stream_view stream() const noexcept { return stream_; }
  [[nodiscard]] rmm::device_async_resource_ref mr() const noexcept { return mr_; }

  decode_column_slot make_column();
  decode_column_slot keep_column(std::unique_ptr<cudf::column> column);
  decode_column_slot memo_column(std::uint64_t key);
  [[nodiscard]] cudf::column* find_memo(std::uint64_t key) const;
  [[nodiscard]] bool contains_memo(std::uint64_t key) const;
  void terminal_memo(std::uint64_t key) noexcept { terminal_memo_ = key; }
  std::unique_ptr<cudf::column> release(decode_column_slot slot) noexcept;
  std::unique_ptr<cudf::column> release_memo(std::uint64_t key);
  decode_column_slot output() noexcept { return decode_column_slot{output_}; }

  rmm::device_buffer& allocate_buffer(std::size_t bytes);
  rmm::device_buffer& allocate_output_buffer(std::size_t bytes);
  rmm::device_buffer& keep_buffer(rmm::device_buffer buffer);
  compressed_representation& keep_representation(std::unique_ptr<compressed_representation> rep);
  cudf::scalar& keep_scalar(std::unique_ptr<cudf::scalar> scalar, std::size_t device_bytes);
  void keep_kernel(std::shared_ptr<codegen::jit::CompiledKernel const> kernel);

  /** Uninitialized staging storage; initialize every consumed or uploaded byte before its first
   * read or device copy. */
  template <typename T>
    requires(std::is_trivially_copyable_v<T> && alignof(T) <= alignof(std::max_align_t))
  std::span<T> host_array(std::size_t count)
  {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::length_error("decode host upload size overflow");
    }
    return {reinterpret_cast<T*>(host_bytes(count * sizeof(T)).data()), count};
  }

  void read_bytes(void* destination, void const* source, std::size_t bytes);
  template <typename T>
    requires std::is_trivially_copyable_v<T>
  T read_scalar(T const* source)
  {
    auto storage = host_array<T>(1);
    read_bytes(storage.data(), source, sizeof(T));
    return storage.front();
  }

  [[nodiscard]] std::size_t retained_device_bytes() const;
  [[nodiscard]] std::size_t retained_host_bytes() const noexcept;
  std::unordered_map<std::uint64_t, std::size_t> remaining_consumers;

 private:
  friend class decode_session;
  decode_frame(rmm::cuda_stream_view stream,
               rmm::device_async_resource_ref mr,
               decode_session& session);
  std::span<std::byte> host_bytes(std::size_t bytes);
  rmm::cuda_stream_view stream_;
  rmm::device_async_resource_ref mr_;
  decode_session& session_;
  std::unordered_map<std::uint64_t, std::unique_ptr<cudf::column>> memo_;
  std::optional<std::uint64_t> terminal_memo_;
  std::list<std::unique_ptr<cudf::column>> columns_;
  std::list<rmm::device_buffer> buffers_;
  std::list<rmm::device_buffer> output_buffers_;
  std::vector<std::unique_ptr<compressed_representation>> representations_;
  std::vector<std::unique_ptr<cudf::scalar>> scalars_;
  std::size_t scalar_bytes_ = 0;
  struct host_upload {
    std::unique_ptr<std::byte[]> bytes;
    std::size_t capacity;
  };
  static_assert(std::is_nothrow_move_constructible_v<host_upload>);
  std::vector<host_upload> uploads_;
  std::unique_ptr<cudf::column> output_;
};

/** Diagnostic peaks of `decode_frame` retained-state estimates, not allocator high-water marks. */
struct decode_session_stats {
  std::size_t peak_frames                = 0;
  std::size_t peak_retained_device_bytes = 0;
  std::size_t peak_retained_host_bytes   = 0;
  std::size_t retirement_stream_queries  = 0;
  std::size_t pressure_waits             = 0;
};

/**
 * Scoped decode submission and completion on borrowed streams and an explicit MR. Requests borrow
 * their compressed/device inputs through finish or destruction. Construction performs no CUDA work;
 * append may leave work pending, and only finish publishes completed outputs. All calls and
 * destruction remain on the constructing CPU thread/current device. Under retained-state pressure,
 * append can wait for a supplied stream's entire tail; queued dependencies must progress
 * independently of later calls on the submitting CPU thread.
 */
class decode_session final {
 public:
  decode_session(std::span<rmm::cuda_stream_view const> streams, rmm::device_async_resource_ref mr);
  ~decode_session() noexcept;
  decode_session(decode_session const&)            = delete;
  decode_session& operator=(decode_session const&) = delete;
  decode_session(decode_session&&)                 = delete;
  decode_session& operator=(decode_session&&)      = delete;

  void append(column_decode_request const& request);
  mask_source_status append(mask_decode_request const& request);
  std::vector<std::unique_ptr<cudf::column>> finish();
  [[nodiscard]] decode_session_stats const& stats() const noexcept;

 private:
  friend class decode_frame;
  friend struct decode_session_test_access;
  decode_frame& register_test_frame();
  void allocation_checkpoint(std::size_t device_bytes, std::size_t host_bytes);
  void keep_kernel(std::shared_ptr<codegen::jit::CompiledKernel const> kernel);
  struct impl;
  std::unique_ptr<impl> state_;
};

void decode_request(column_decode_request const& request, decode_frame& frame);
mask_source_status decode_request(mask_decode_request const& request, decode_frame& frame);
void decode_standalone(compressed_representation const& rep,
                       decode_frame& frame,
                       decode_column_slot output);

compressed_representation& reconstruct_decode_representation(
  std::string const& compressor_name,
  std::vector<std::string> const& output_names,
  std::span<decode_column_slot const> outputs,
  leaf_meta_v const& meta,
  decode_frame& frame);

}  // namespace simpatico
