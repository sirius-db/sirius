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

#pragma once

#include "cudf/cudf_utils.hpp"
#include "duckdb/common/enums/join_type.hpp"

#include <memory>

namespace sirius::op {

using join_index_pair = std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                                  std::unique_ptr<rmm::device_uvector<cudf::size_type>>>;

/// Wrapper that unifies cudf::hash_join and cudf::distinct_hash_join behind a single API.
/// When unique_keys is true (and join type supports it), uses the faster distinct_hash_join.
class build_hash_table {
 public:
  build_hash_table()  = default;
  ~build_hash_table() = default;

  build_hash_table(const build_hash_table&)            = delete;
  build_hash_table& operator=(const build_hash_table&) = delete;
  build_hash_table(build_hash_table&&)                 = default;
  build_hash_table& operator=(build_hash_table&&)      = default;

  /// Build the hash table from the given keys.
  /// Uses distinct_hash_join when unique_keys is true and join_type supports it (INNER, LEFT).
  void build(cudf::table_view build_keys,
             bool unique_keys,
             duckdb::JoinType join_type,
             cudf::null_equality null_eq,
             rmm::cuda_stream_view stream);

  /// Probe for INNER join.
  join_index_pair inner_join(cudf::table_view probe_keys, rmm::cuda_stream_view stream) const;

  /// Probe for LEFT join.
  join_index_pair left_join(cudf::table_view probe_keys, rmm::cuda_stream_view stream) const;

  /// Probe for FULL OUTER join (only available with generic hash_join).
  join_index_pair full_join(cudf::table_view probe_keys, rmm::cuda_stream_view stream) const;

  void reset();

  bool is_built() const;

  bool is_distinct() const;

 private:
  std::unique_ptr<cudf::hash_join> _generic;
  std::unique_ptr<cudf::distinct_hash_join> _distinct;
};

}  // namespace sirius::op
