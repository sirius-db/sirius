/*
 * Copyright 2026, rocm-cuda-compat contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade representation_converter_registry — ROCm real registry.
//! The registry stores converter callables keyed by (src_type_index, target_type_index).
//! convert() looks up and invokes the callable. The callable bodies (which do
//! hipMemcpy etc.) are registered by the application via register_builtin_converters.

#pragma once
#include "cucascade/data/common.hpp"
#include "cucascade/memory/common.hpp"
#include <rmm/cuda_stream.hpp>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <utility>

namespace cucascade::memory { class memory_space; }

namespace cucascade {

class representation_converter_registry {
 public:
  using converter_fn = std::function<std::unique_ptr<idata_representation>(
    idata_representation const&, memory::memory_space const*, rmm::cuda_stream_view)>;

  using key_type = std::pair<std::type_index, std::type_index>;

  struct key_hash {
    std::size_t operator()(key_type const& k) const {
      return std::hash<std::type_index>{}(k.first) ^ std::hash<std::type_index>{}(k.second);
    }
  };

  template <typename Src, typename Target, typename Callable>
  void register_converter(Callable&& fn) {
    converters_[{std::type_index(typeid(Src)), std::type_index(typeid(Target))}] =
      [f = std::forward<Callable>(fn)](idata_representation const& src,
                                       memory::memory_space const* space,
                                       rmm::cuda_stream_view stream)
      -> std::unique_ptr<idata_representation> {
        auto result = f(src, space, stream);
        return std::unique_ptr<idata_representation>(result.release());
      };
  }

  template <typename Src, typename Target>
  bool has_converter() const {
    return converters_.find({std::type_index(typeid(Src)), std::type_index(typeid(Target))}) != converters_.end();
  }

  bool has_converter_for(idata_representation const& source, std::type_index target) const {
    auto src_ti = std::type_index(typeid(source));
    return converters_.find({src_ti, target}) != converters_.end();
  }

  template <typename TargetType>
  std::unique_ptr<TargetType> convert(idata_representation const& source,
                                      memory::memory_space const* target_space,
                                      rmm::cuda_stream_view stream) {
    auto it = converters_.find({std::type_index(typeid(source)), std::type_index(typeid(TargetType))});
    if (it == converters_.end()) {
      throw std::runtime_error("cuCascade: no converter registered for " +
        std::string(typeid(source).name()) + " -> " + std::string(typeid(TargetType).name()));
    }
    auto result = it->second(source, target_space, stream);
    return std::unique_ptr<TargetType>(dynamic_cast<TargetType*>(result.release()));
  }

  std::unique_ptr<idata_representation> convert(idata_representation const& source,
                                                std::type_index target,
                                                memory::memory_space const* target_space,
                                                rmm::cuda_stream_view stream) {
    auto it = converters_.find({std::type_index(typeid(source)), target});
    if (it == converters_.end()) {
      throw std::runtime_error("cuCascade: no converter registered for " +
        std::string(typeid(source).name()) + " -> " + std::string(target.name()));
    }
    return it->second(source, target_space, stream);
  }

  template <typename Src, typename Target>
  void unregister_converter() {
    converters_.erase({std::type_index(typeid(Src)), std::type_index(typeid(Target))});
  }

  void clear() { converters_.clear(); }

 private:
  std::unordered_map<key_type, converter_fn, key_hash> converters_;
};

}  // namespace cucascade
