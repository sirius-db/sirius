/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade data_batch — ROCm stub.
//! The most-used cuCascade type. data_batch / read_only_data_batch / mutable_data_batch.

#pragma once
#include "cucascade/data/common.hpp"
#include "cucascade/data/representation_converter.hpp"
#include "cucascade/memory/common.hpp"
#include <rmm/cuda_stream.hpp>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace cucascade::memory { class memory_space; }

namespace cucascade {

/// Read-only view of a data batch (copyable, holds a shared reference).
class read_only_data_batch {
 public:
  read_only_data_batch() = default;
  read_only_data_batch(read_only_data_batch const&) = default;
  read_only_data_batch& operator=(read_only_data_batch const&) = default;
  read_only_data_batch(read_only_data_batch&&) noexcept = default;
  read_only_data_batch& operator=(read_only_data_batch&&) noexcept = default;

  read_only_data_batch(std::shared_ptr<idata_representation> data,
                       memory::memory_space* space, uint64_t batch_id)
    : data_(std::move(data)), space_(space), batch_id_(batch_id) {}

  idata_representation const* get_data() const { return data_.get(); }
  memory::memory_space* get_memory_space() const { return space_; }
  memory::Tier get_current_tier() const { return memory::Tier::GPU; }
  uint64_t get_batch_id() const { return batch_id_; }
  cudaEvent_t get_writer_event() const { return nullptr; }

  template <typename TargetRepr>
  std::shared_ptr<class data_batch> clone_to(representation_converter_registry& /*registry*/,
                                             uint64_t /*new_batch_id*/,
                                             memory::memory_space const* /*target_space*/,
                                             rmm::cuda_stream_view /*stream*/,
                                             std::unique_ptr<idata_batch_probe> /*probe*/ = {}) {
    throw std::runtime_error("cuCascade stub: read_only_data_batch::clone_to");
  }

  std::shared_ptr<class data_batch> clone(uint64_t new_batch_id,
                                          rmm::cuda_stream_view /*stream*/,
                                          std::unique_ptr<idata_batch_probe> /*probe*/ = {}) {
    throw std::runtime_error("cuCascade stub: read_only_data_batch::clone");
  }

 protected:
  std::shared_ptr<idata_representation> data_;
  memory::memory_space* space_{nullptr};
  uint64_t batch_id_{0};
};

/// Mutable view of a data batch (move-only, holds exclusive access).
class mutable_data_batch {
 public:
  mutable_data_batch() = default;
  mutable_data_batch(mutable_data_batch&&) noexcept = default;
  mutable_data_batch& operator=(mutable_data_batch&&) noexcept = default;

  mutable_data_batch(std::shared_ptr<idata_representation> data,
                     memory::memory_space* space, uint64_t batch_id)
    : data_(std::move(data)), space_(space), batch_id_(batch_id) {}

  idata_representation const* get_data() const { return data_.get(); }
  memory::memory_space* get_memory_space() const { return space_; }
  uint64_t get_batch_id() const { return batch_id_; }

  void set_data(std::unique_ptr<idata_representation> /*new_data*/) {
    throw std::runtime_error("cuCascade stub: mutable_data_batch::set_data");
  }

  void rebind_stream(rmm::cuda_stream_view) {}

  template <typename TargetRepr>
  std::unique_ptr<TargetRepr> convert_to(representation_converter_registry&,
                                         memory::memory_space const*,
                                         rmm::cuda_stream_view) {
    throw std::runtime_error("cuCascade stub: mutable_data_batch::convert_to");
  }

  template <typename TargetRepr>
  std::shared_ptr<class data_batch> clone_to(representation_converter_registry&,
                                             uint64_t,
                                             memory::memory_space const*,
                                             rmm::cuda_stream_view,
                                             std::unique_ptr<idata_batch_probe> = {}) {
    throw std::runtime_error("cuCascade stub: mutable_data_batch::clone_to");
  }

 private:
  std::shared_ptr<idata_representation> data_;
  memory::memory_space* space_{nullptr};
  uint64_t batch_id_{0};
};

/// A data batch: owns a data representation with a lifecycle (idle → mutable → read-only).
class data_batch {
 public:
  static std::shared_ptr<data_batch> make(uint64_t batch_id,
                                          std::unique_ptr<idata_representation> data,
                                          std::unique_ptr<idata_batch_probe> probe = {},
                                          memory::memory_space* space = nullptr) {
    return std::shared_ptr<data_batch>(new data_batch(batch_id, std::move(data), std::move(probe), space));
  }

  batch_state get_state() const { return state_; }
  uint64_t get_batch_id() const { return batch_id_; }

  read_only_data_batch to_read_only() {
    if (state_ != batch_state::mutable_locked && state_ != batch_state::idle) {
      throw std::runtime_error("cuCascade: to_read_only from invalid state");
    }
    state_ = batch_state::read_only;
    return read_only_data_batch(data_, space_, batch_id_);
  }
  mutable_data_batch to_mutable() {
    if (state_ != batch_state::read_only && state_ != batch_state::idle) {
      throw std::runtime_error("cuCascade: to_mutable from invalid state");
    }
    state_ = batch_state::mutable_locked;
    return mutable_data_batch(data_, space_, batch_id_);
  }
  bool try_to_read_only() {
    if (state_ != batch_state::mutable_locked && state_ != batch_state::idle) return false;
    state_ = batch_state::read_only;
    return true;
  }
  bool try_to_mutable() {
    if (state_ != batch_state::read_only && state_ != batch_state::idle) return false;
    state_ = batch_state::mutable_locked;
    return true;
  }

  static read_only_data_batch to_idle(read_only_data_batch&& ro) {
    return std::move(ro);  // state transitions handled by caller
  }
  static mutable_data_batch to_idle(mutable_data_batch&& mut) {
    return std::move(mut);
  }
  static mutable_data_batch readonly_to_mutable(read_only_data_batch&& ro) {
    auto data = ro.get_data();
    auto space = ro.get_memory_space();
    auto id = ro.get_batch_id();
    return mutable_data_batch(std::shared_ptr<idata_representation>(
      const_cast<idata_representation*>(data)), space, id);
  }
  static read_only_data_batch mutable_to_readonly(mutable_data_batch&& mut) {
    auto data = mut.get_data();
    auto space = mut.get_memory_space();
    auto id = mut.get_batch_id();
    return read_only_data_batch(std::shared_ptr<idata_representation>(
      const_cast<idata_representation*>(data)), space, id);
  }

  void subscribe() { ++subscriber_count_; }
  void unsubscribe() { if (subscriber_count_ > 0) --subscriber_count_; }
  std::size_t get_subscriber_count() const { return subscriber_count_; }

 private:
  data_batch(uint64_t batch_id, std::unique_ptr<idata_representation> data,
             std::unique_ptr<idata_batch_probe> /*probe*/, memory::memory_space* space)
    : batch_id_(batch_id), data_(std::move(data)), space_(space) {}

  uint64_t batch_id_{0};
  batch_state state_{batch_state::idle};
  std::shared_ptr<idata_representation> data_;
  memory::memory_space* space_{nullptr};
  std::size_t subscriber_count_{0};
};

}  // namespace cucascade
