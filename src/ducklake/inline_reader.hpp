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
#include <memory>
namespace duckdb {
class DataChunk;
}
namespace sirius::ducklake {
class binding_handle;
// Independent cursor, created/used/destroyed only on the captured query owner.
// A returned chunk is borrowed until next()/destruction; detach before publication.
class inline_reader {
 public:
  ~inline_reader();
  duckdb::DataChunk const* next();  // nullptr is EOS; a zero-row chunk is not EOS.
 private:
  friend class binding_handle;
  struct impl;
  explicit inline_reader(std::unique_ptr<impl>);
  std::unique_ptr<impl> impl_;
};
}  // namespace sirius::ducklake
