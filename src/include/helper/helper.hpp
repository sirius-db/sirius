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

 #pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <atomic>
#include <thread>
#include <iostream>
#include <nvtx3/nvtx3.hpp>

namespace sirius {

// Smart pointer wrappers
template<typename T>
using unique_ptr = std::unique_ptr<T>;

template<typename T>
using shared_ptr = std::shared_ptr<T>;

// Smart pointer factory functions
template<typename T, typename... Args>
constexpr auto make_unique(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

template<typename T, typename... Args>
auto make_shared(Args&&... args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}

// Container wrappers
template<typename T, typename Allocator = std::allocator<T>>
using vector = std::vector<T, Allocator>;

template<typename Key, typename T, typename Hash = std::hash<Key>, 
            typename KeyEqual = std::equal_to<Key>, 
            typename Allocator = std::allocator<std::pair<const Key, T>>>
using unordered_map = std::unordered_map<Key, T, Hash, KeyEqual, Allocator>;

template<typename T, typename Container = std::deque<T>>
using queue = std::queue<T, Container>;

// Threading wrappers
using mutex = std::mutex;

template<typename Mutex>
using lock_guard = std::lock_guard<Mutex>;

template<typename T>
using atomic = std::atomic<T>;

using thread = std::thread;

template<class T, class SRC>
void DynamicCastCheck(const SRC *source) {
#ifndef __APPLE__
    // Actual check is on the fact that dynamic_cast and reinterpret_cast are equivalent
    reinterpret_cast<const T *>(source) == dynamic_cast<const T *>(source);
#endif
}

}