// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "codegen/jit/kernel_cache.hpp"
#include "codegen/plan/plan_interpreter.hpp"

#include <cuda/stream>

#include <array>
#include <bit>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace simpatico {

struct mask_destination {
  std::uint32_t* words;  ///< Caller-owned device storage for selection bitmask.
  std::int64_t num_rows;
};

/**
 * @brief The result of a column_decode_request is either a value_result or a predicate_result.
 *
 * A value_result represents a column of the requested type, optionally restored from a stored type.
 * A predicate_result represents a BOOL8 selection mask, optionally written into a caller-owned
 * BITmask.
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

/**
 * @brief A membership probe over the plan's decoded key column.
 *
 * `prior_mask_words` is handed to `probe` as its optional prior keep mask, letting rows it already
 * excludes skip the lookup; it must stay valid until the probe's work on the request's stream
 * completes. The destination receives only the probe's result, so the caller ANDs it with the
 * prior.
 */
struct membership_source {
  decltype(sirius::codegen::membership_filter_directive::probe) probe;
  cudf::data_type stored_type;
  std::uint32_t const* prior_mask_words = nullptr;
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
 * @brief The context of one decode request: its stream, its memory resource, and the host state
 * that its queued GPU work may still read.
 *
 * The frame owns no device memory. Decoders hold device temporaries in RAII owners allocated on
 * stream() and let them release at scope end, even while work that reads them is still queued. This
 * is safe because deallocation through a `device_async_resource_ref` is stream-ordered: the memory
 * is reused only after earlier work on the same stream. A temporary must therefore never be read or
 * written by work queued on another stream.
 *
 * Storage from host_array() and kernels passed to keep_kernel() stay valid until the owning
 * decode_session has drained its streams. All calls run on the session's CPU thread.
 */
class decode_frame final {
 public:
  decode_frame(decode_frame const&)            = delete;
  decode_frame& operator=(decode_frame const&) = delete;

  [[nodiscard]] ::cuda::stream_ref stream() const noexcept { return stream_; }
  [[nodiscard]] rmm::device_async_resource_ref mr() const noexcept { return mr_; }

  /**
   * @brief Allocate host storage for the source of an asynchronous upload.
   *
   * An asynchronous copy from pageable memory may read its source after the call returns, so this
   * storage stays valid until session completion. Storage is uninitialized; initialize every byte
   * before reading or copying it to the device.
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
   * @brief Copy device bytes to host storage with `read_device_bytes_completed`
   * (`util/host_observation.hpp`) on the frame's stream.
   */
  void read_bytes(void* destination, void const* source, std::size_t bytes);

  /**
   * @brief Read one value from device memory with read_bytes(); @p source must be readable by work
   * on the frame's stream.
   */
  template <typename T>
    requires std::is_trivially_copyable_v<T>
  T read_scalar(T const* source)
  {
    std::array<std::byte, sizeof(T)> bytes{};
    read_bytes(bytes.data(), source, sizeof(T));
    return std::bit_cast<T>(bytes);
  }

  /**
   * @brief Keep a compiled kernel loaded until the session completes, even if its cache is cleared.
   */
  void keep_kernel(std::shared_ptr<codegen::jit::CompiledKernel const> kernel);

 private:
  friend class decode_session;
  decode_frame(::cuda::stream_ref stream, rmm::device_async_resource_ref mr) noexcept;
  std::span<std::byte> host_bytes(std::size_t bytes);
  ::cuda::stream_ref stream_;
  rmm::device_async_resource_ref mr_;
  std::vector<std::unique_ptr<std::byte[]>> uploads_;
  // Dropping the final module handle can synchronize the CUDA context.
  std::vector<std::shared_ptr<codegen::jit::CompiledKernel const>> kernels_;
};

/**
 * @brief A single-threaded, multi-stream decode session (coordinator and owner for a group of
 * decode requests).
 *
 * The decode_session has 3 main responsibilities:
 *  1. Submission: assign requests to the supplied CUDA streams and invoke the decoder (append()).
 *  2. Lifetime management: keep each request's copy, its decode_frame host state, and pending
 * outputs alive until the streams drain. Device temporaries are released during append() in stream
 * order.
 *  3. Completion: wait for the streams that received a request and return the final outputs
 * (finish())
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
   * @param streams Nonempty list of borrowed streams, assigned to requests in rotation (see
   * next_stream())
   * @param mr Borrowed resource for device allocations; it must honor the stream-ordered
   * deallocation contract of `device_async_resource_ref` (a synchronous resource may block on
   * release)
   */
  decode_session(std::span<const ::cuda::stream_ref> streams, rmm::device_async_resource_ref mr);
  /**
   * @brief Attempt to complete pending work before releasing results and host state; log cleanup
   * errors without throwing.
   */
  ~decode_session() noexcept;
  decode_session(decode_session const&)            = delete;
  decode_session& operator=(decode_session const&) = delete;
  decode_session(decode_session&&)                 = delete;
  decode_session& operator=(decode_session&&)      = delete;

  /**
   * @brief Copy a request and submit its decode work, retaining the result until finish().
   *
   * May wait for host readbacks and cuDF calls on the request's stream, such as constructing a
   * generic predicate needle. Device temporaries release in stream order; allocation or release may
   * still block depending on the memory resource. Such waits can include other work already queued
   * on that stream, so that work must not depend on a later call from this CPU thread. Submission
   * failures attempt to complete pending work, close the session, and rethrow the original
   * exception.
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
   * Waits once for each distinct stream that received a request, which also covers other work
   * queued there; a supplied stream that received no request is not waited for. An empty session
   * performs no CUDA work. Success or failure closes the session: neither append() nor finish() may
   * be called again. A failed call returns no partial results.
   *
   * @return Completed columns, excluding mask-only requests
   */
  std::vector<std::unique_ptr<cudf::column>> finish();

  /**
   * @brief The stream the next append() assigns its request to, so a caller can allocate storage
   * that request writes, or order it after other work, on that stream.
   */
  [[nodiscard]] ::cuda::stream_ref next_stream() const noexcept;

 private:
  friend struct decode_session_test_access;
  decode_frame& register_test_frame();
  [[nodiscard]] std::size_t retained_host_uploads() const noexcept;
  struct impl;
  std::unique_ptr<impl> state_;
};

/**
 * @brief Decode one request through a single-stream session and return its completed result.
 *
 * Validation and execution failures propagate as from decode_session::append() and finish().
 */
[[nodiscard]] std::unique_ptr<cudf::column> decode_one(column_decode_request const& request,
                                                       ::cuda::stream_ref stream,
                                                       rmm::device_async_resource_ref mr);
mask_source_status decode_one(mask_decode_request const& request,
                              ::cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr);

/**
 * @brief Decode one column request on the frame's stream.
 *
 * @return An owning column whose writes may still be pending on the frame's stream
 */
[[nodiscard]] std::unique_ptr<cudf::column> decode_request(column_decode_request const& request,
                                                           decode_frame& frame);
mask_source_status decode_request(mask_decode_request const& request, decode_frame& frame);

/**
 * @brief Thrown by a selected decode whose full-width value column is null-masked. Row selection
 * has no null model, so a caller may decline the selection instead of failing.
 */
class unsupported_nullable_selection final : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};
[[nodiscard]] std::unique_ptr<cudf::column> decode_standalone(compressed_representation const& rep,
                                                              decode_frame& frame);

/**
 * @brief Rebuild @p node's codec representation from channels decoded on the frame's stream,
 * publishing `PlanNode::dictionary_key_width_hint` on a dictionary.
 *
 * Consumes @p channels. The result owns them and releases them on that stream, so it may be
 * destroyed as soon as the work that decodes from it has been queued.
 *
 * @throw std::invalid_argument if the channel names, count, or types do not match the codec, or if
 * the hint is below -1 or a positive hint disagrees with the total key character size
 */
[[nodiscard]] std::unique_ptr<compressed_representation> reconstruct_decode_representation(
  PlanNode const& node,
  std::vector<std::string> const& output_names,
  std::vector<std::unique_ptr<cudf::column>> channels,
  decode_frame& frame);

}  // namespace simpatico
