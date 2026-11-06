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

#include "io/s3/duckdb_secret_config.hpp"
#include "transparent/plan_source_policy.hpp"

#include <cucascade/io/uri_parser.hpp>

#include <string>
#include <string_view>

namespace sirius::ducklake {
// Scan every dependency even after an unknown URI so a later S3 dependency
// retains the existing S3 veto. Classification adds no new non-S3 replay veto.
class dependency_source_policy {
 public:
  void observe(std::string const& path)
  {
    if (path.empty()) return;
    // Match the shared S3 policy even if the remainder of the URI is malformed.
    if (io::s3::is_s3_path(path)) {
      s3_ = true;
      return;
    }
    try {
      auto uri = cucascade::io::parse(path);
      if (uri.scheme != "file" || !uri.host.empty()) unknown_ = true;
    } catch (std::invalid_argument const&) {
      unknown_ = true;
    }
  }
  transparent::scan_source_policy finish() const
  {
    using transparent::byte_source_class;
    if (s3_)
      return {"ducklake_scan",
              byte_source_class::sirius_owned_s3,
              false,
              "S3 CPU fallback is not supported"};
    if (unknown_) return {"ducklake_scan", byte_source_class::unclassified, true, {}};
    return {"ducklake_scan", byte_source_class::local_file, true, {}};
  }

 private:
  bool s3_      = false;
  bool unknown_ = false;
};
}  // namespace sirius::ducklake
