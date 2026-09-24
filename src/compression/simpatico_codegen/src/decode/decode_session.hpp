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

/**
 * @brief A handle to a column owner reserved in a decode_frame.
 *
 * Adding other owners to the frame does not invalidate the handle.
 * The handle does not extend the frame's lifetime.
 */
class decode_column_slot final {
 public:
  /**
   * @brief Transfer a column into an empty slot without allocating or waiting.
   *
   * Adopting into an occupied slot terminates the process.
   */
  void adopt(std::unique_ptr<cudf::column> column) const noexcept;
  /**
   * @brief Access the owned column without waiting for its data to be ready.
   *
   * @throw std::logic_error if the slot is empty
   */
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
  std::uint32_t* words;  ///< Caller-owned device storage for selection bitmask.
  std::int64_t num_rows;
};

/**
 * @brief The result of a column_decode_request is either a value_result or a predicate_result.
 *
 * A value_result represents a column of the requested type, optionally restored from a stored type.
 * A predicate_result represents a BOOL8 selection mask, optionally written into a caller-owned BITmask.
 */
struct value_result {
  /// Restore this logical type without converting bytes when the fixed-width sizes match.
  std::optional<cudf::data_type> stored_type;
};
struct predicate_result {
  decode_predicate predicate;
  /// Also write selection bits; this requires a null-free predicate result.
  std::optional<mask_destination> ballot;
};

/**
 * @brief Check a row selection's metadata against a plan before submitting decode work.
 *
 * Copies the descriptor, not the referenced mask, row set, or index data. Those must remain valid
 * until session completion. Validation does not inspect device data.
 */
class validated_selection final {
 public:
  /**
   * @throw std::invalid_argument if the selection's route, counts, or index metadata are invalid
   */
  explicit validated_selection(PlanTree const& plan, decode_selection const& selection);
  [[nodiscard]] decode_selection const& get() const noexcept { return selection_; }

 private:
  decode_selection selection_;
};

using decode_source =
  std::variant<std::reference_wrapper<PlanTree const>,
               std::reference_wrapper<standalone_compressed_representation const>>;

/**
 * @brief Request a decoded column or BOOL8 predicate result, optionally for selected rows.
 *
 * The source is borrowed through session completion. Standalone representations support only values
 * without row selection; plans also support predicates and selections.
 */
struct column_decode_request {
  decode_source source;
  std::variant<value_result, predicate_result> result = value_result{};
  std::optional<validated_selection> selection;
};

struct membership_source {
  decltype(sirius::codegen::membership_filter_directive::probe) probe;
  cudf::data_type stored_type;
};

/**
 * @brief Write filter results into a caller-owned mask without returning a column.
 *
 * The plan and destination remain valid through session completion. The destination must hold
 * `selection_mask::AllocWordsFor(num_rows)` words. A declined membership probe writes all ones,
 * leaving every row selected.
 */
struct mask_decode_request {
  PlanTree const& plan;
  std::variant<sirius::codegen::range_predicate, membership_source> source;
  mask_destination destination;
};

enum class mask_source_status { ACCEPTED, DECLINED };

/**
 * @brief Hold temporary data for one decode request until its GPU work completes.
 *
 * The decoder and codec helpers use the frame's stream and memory resource. Register owners before
 * queuing work that uses them. Adding owners does not invalidate existing slots or storage
 * references. All helpers run on the session's CPU thread.
 *
 * If recording a column, buffer, representation, or scalar owner fails, the frame waits for its
 * stream before releasing that owner. Transferring an owner out of the frame does not wait for GPU
 * completion; its new owner must preserve its lifetime.
 */
class decode_frame final {
 public:
  ~decode_frame();
  decode_frame(decode_frame const&)            = delete;
  decode_frame& operator=(decode_frame const&) = delete;

  [[nodiscard]] rmm::cuda_stream_view stream() const noexcept { return stream_; }
  [[nodiscard]] rmm::device_async_resource_ref mr() const noexcept { return mr_; }

  /**
   * @brief Reserve an empty column slot before creating or submitting work for the column.
   */
  decode_column_slot make_column();
  decode_column_slot keep_column(std::unique_ptr<cudf::column> column);
  /**
   * @brief Find or reserve a column slot shared by plan steps using the same key.
   */
  decode_column_slot memo_column(std::uint64_t key);
  [[nodiscard]] cudf::column* find_memo(std::uint64_t key) const;
  /**
   * @brief Check whether a key was registered, even if its column has since been released.
   */
  [[nodiscard]] bool contains_memo(std::uint64_t key) const;
  void terminal_memo(std::uint64_t key) noexcept { terminal_memo_ = key; }
  /**
   * @brief Transfer ownership out of a slot, leaving it empty without waiting.
   */
  std::unique_ptr<cudf::column> release(decode_column_slot slot) noexcept;
  std::unique_ptr<cudf::column> release_memo(std::uint64_t key);
  decode_column_slot output() noexcept { return decode_column_slot{output_}; }

  rmm::device_buffer& allocate_buffer(std::size_t bytes);
  /**
   * @brief Allocate final-output storage, excluded from the temporary-memory estimate.
   */
  rmm::device_buffer& allocate_output_buffer(std::size_t bytes);
  rmm::device_buffer& keep_buffer(rmm::device_buffer buffer);
  compressed_representation& keep_representation(std::unique_ptr<compressed_representation> rep);
  cudf::scalar& keep_scalar(std::unique_ptr<cudf::scalar> scalar, std::size_t device_bytes);
  /**
   * @brief Keep a compiled kernel loaded until the session completes, even if its cache is cleared.
   */
  void keep_kernel(std::shared_ptr<codegen::jit::CompiledKernel const> kernel);

  /**
   * @brief Allocate host storage that stays valid while the request is pending.
   *
   * Storage is uninitialized; initialize every byte before reading or copying it to the device.
   *
   * @throw std::length_error if the requested byte count overflows
   */
  template <typename T>
    requires(std::is_trivially_copyable_v<T> && alignof(T) <= alignof(std::max_align_t))
  std::span<T> host_array(std::size_t count)
  {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::length_error("decode host upload size overflow");
    }
    return {reinterpret_cast<T*>(host_bytes(count * sizeof(T)).data()), count};
  }

  /**
   * @brief Copy device bytes to host storage and wait for the frame's stream before returning.
   */
  void read_bytes(void* destination, void const* source, std::size_t bytes);
  template <typename T>
    requires std::is_trivially_copyable_v<T>
  T read_scalar(T const* source)
  {
    auto storage = host_array<T>(1);
    read_bytes(storage.data(), source, sizeof(T));
    return storage.front();
  }

  /**
   * @brief Estimate temporary device storage retained by this request.
   *
   * Counts buffer capacities, owned columns' `alloc_size()` (including representation-owned
   * columns), and supplied scalar sizes. Excludes borrowed input, final output, hidden column
   * capacity, and allocator overhead; this is not a physical-memory bound.
   */
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
 * @brief A single-threaded, multi-stream decode session (coordinator and owner for a group of
 * decode requests).
 *
 * The decode_session has 3 main responsibilities:
 *  1. Submission: assign requests to the supplied CUDA streams and invoke the decoder (append()).
 *  2. Lifetime management: keep temporary buffers, request metadata, and pending outputs alive as
 * long as needed by the GPU.
 *  3. Completion: wait for all streams to finish and return the final outputs (finish())
 */
class decode_session final {
 public:
  /**
   * @brief Create a session without submitting GPU work.
   *
   * All calls and destruction must use the constructing CPU thread and current CUDA device. Inputs
   * must be ready on their assigned streams and remain valid until completion or session
   * destruction. The resource and streams must outlive allocations that use them, including
   * returned columns.
   *
   * @param streams Nonempty list of borrowed streams, assigned to requests in rotation
   * @param mr Borrowed resource for device allocations
   */
  decode_session(std::span<rmm::cuda_stream_view const> streams, rmm::device_async_resource_ref mr);
  /**
   * @brief Attempt to complete pending work before releasing owners; log cleanup errors without
   * throwing.
   */
  ~decode_session() noexcept;
  decode_session(decode_session const&)            = delete;
  decode_session& operator=(decode_session const&) = delete;
  decode_session(decode_session&&)                 = delete;
  decode_session& operator=(decode_session&&)      = delete;

  /**
   * @brief Copy a request and submit its decode work, retaining the result until finish().
   *
   * May wait for required host readbacks or to release temporary storage. Such waits can include
   * other work already queued on a supplied stream, so that work must not depend on a later call
   * from this CPU thread. Submission failures attempt to complete pending work, close the session,
   * and rethrow the original exception.
   */
  void append(column_decode_request const& request);
  /**
   * @brief Submit a mask request with the same lifetime, waiting, and failure rules as column
   * requests.
   *
   * @return Whether the filter was accepted, not whether GPU work has completed; no returned-column
   * slot is added
   */
  mask_source_status append(mask_decode_request const& request);
  /**
   * @brief Complete submitted work and transfer decoded columns to the caller in request order.
   *
   * Once any request has been submitted, completion includes other work queued on the supplied
   * streams. An empty session performs no CUDA work. Success or failure closes the session: neither
   * append() nor finish() may be called again. A failed call returns no partial results.
   *
   * @return Completed columns, excluding mask-only requests
   */
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

/**
 * @brief Rebuild a codec representation from decoded channels owned by the frame.
 *
 * May transfer columns out of the supplied slots. The returned representation and any remaining
 * channel storage stay owned by the frame while GPU work is pending.
 */
compressed_representation& reconstruct_decode_representation(
  std::string const& compressor_name,
  std::vector<std::string> const& output_names,
  std::span<decode_column_slot const> outputs,
  leaf_meta_v const& meta,
  decode_frame& frame);

}  // namespace simpatico
