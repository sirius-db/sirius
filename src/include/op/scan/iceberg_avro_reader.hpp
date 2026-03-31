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

#include <string>
#include <utility>
#include <vector>

namespace sirius::op::scan {

/**
 * @brief Read an Iceberg manifest-list Avro file.
 *
 * Parses the Avro container format, extracts the embedded JSON schema to
 * locate the `manifest_path` (string) and `content` (int) fields, and
 * returns one entry per manifest with its content type:
 *   0 = DATA, 1 = POSITION_DELETES, 2 = EQUALITY_DELETES
 *
 * Only the "null" (uncompressed) Avro codec is supported; deflate/snappy
 * tables will throw.
 *
 * @param path  Filesystem path to the manifest-list .avro file.
 * @return      Vector of (manifest_path, content) pairs.
 * @throws      std::runtime_error on I/O or parse errors.
 */
std::vector<std::pair<std::string, int>> read_iceberg_manifest_list(const std::string& path);

/**
 * @brief Read an Iceberg manifest Avro file and return delete-file paths.
 *
 * Parses the `data_file` record inside each manifest entry.  Returns the
 * `data_file.file_path` for every entry whose `data_file.content` matches
 * @p target_content.
 *
 * Content values: 0 = DATA, 1 = POSITION_DELETES, 2 = EQUALITY_DELETES
 *
 * @param path            Filesystem path to the manifest .avro file.
 * @param target_content  Only include entries with this content type.
 * @return                File paths matching the target content type.
 * @throws                std::runtime_error on I/O or parse errors.
 */
std::vector<std::string> read_iceberg_manifest_delete_files(const std::string& path,
                                                            int target_content);

}  // namespace sirius::op::scan
