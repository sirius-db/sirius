/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade representation_converter_registry — ROCm stub.

#pragma once
#include "cucascade/data/common.hpp"
#include "cucascade/memory/common.hpp"
#include <rmm/cuda_stream.hpp>
#include <functional>
#include <memory>
#include <stdexcept>
#include <typeindex>
#include <unordered_map>

namespace cucascade::memory { class memory_space; }

namespace cucascade {

class representation_converter_registry {
 public:
  template <typename Src, typename Target, typename Callable>
  void register_converter(Callable&& /*fn*/) {}

  template <typename Src, typename Target>
  bool has_converter() const { return false; }

  bool has_converter_for(idata_representation const& /*source*/,
                         std::type_index /*target*/) const { return false; }

  template <typename TargetType>
  std::unique_ptr<TargetType> convert(idata_representation const& /*source*/,
                                      memory::memory_space const* /*target_space*/,
                                      rmm::cuda_stream_view /*stream*/) {
    throw std::runtime_error("cuCascade stub: convert<TargetType>");
  }

  std::unique_ptr<idata_representation> convert(idata_representation const& /*source*/,
                                                std::type_index /*target*/,
                                                memory::memory_space const* /*target_space*/,
                                                rmm::cuda_stream_view /*stream*/) {
    throw std::runtime_error("cuCascade stub: convert(type_index)");
  }

  template <typename Src, typename Target>
  void unregister_converter() {}

  void clear() {}
};

}  // namespace cucascade
