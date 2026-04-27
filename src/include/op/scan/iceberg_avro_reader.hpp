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

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op::scan {

/**
 * @brief Metadata for a single file entry from an Iceberg manifest.
 *
 * Used for data files (content=0), positional/equality delete files (content=1,2),
 * and V3 deletion vectors (file_format="puffin" with offset info).
 */
struct IcebergDeleteFileEntry {
  std::string file_path;
  int content;                        // 0=DATA, 1=POSITION_DELETES, 2=EQUALITY_DELETES
  std::string file_format;            // "parquet" or "puffin" (always lowercase)
  std::string referenced_data_file;   // data file this DV applies to (V3, empty if absent)
  int64_t content_offset{-1};         // byte offset in Puffin file (V3, -1 if absent)
  int64_t content_size_in_bytes{-1};  // byte length of DV blob (V3, -1 if absent)
  int64_t sequence_number{0};         // manifest entry sequence number (for eq delete filtering)

  /// True if this entry is a V3 deletion vector (Puffin format with offset info).
  /// file_format is normalized to lowercase at Avro parse time.
  [[nodiscard]] bool is_deletion_vector() const
  {
    return file_format == "puffin" && content_offset >= 0 && content_size_in_bytes > 0;
  }
};

/**
 * @brief Read an Iceberg manifest-list Avro file.
 *
 * Parses the Avro container format, extracts the embedded JSON schema to
 * locate the `manifest_path` (string) and `content` (int) fields, and
 * returns one entry per manifest with its content type:
 *   0 = data manifest, 1 = delete manifest (any delete type)
 *
 * Supports "null" (uncompressed) and "deflate" Avro codecs.
 *
 * @param path  Filesystem path to the manifest-list .avro file.
 * @return      Vector of (manifest_path, content) pairs.
 * @throws      std::runtime_error on I/O or parse errors.
 */
std::vector<std::pair<std::string, int>> read_iceberg_manifest_list(const std::string& path);

/**
 * @brief Read an Iceberg manifest Avro file and return file entries.
 *
 * Parses the `data_file` record inside each manifest entry.  Returns a
 * full IcebergDeleteFileEntry for every entry whose `data_file.content`
 * matches @p target_content (or all entries if target_content < 0).
 * Includes V3 fields when present.
 *
 * data_file.content values: 0 = DATA, 1 = POSITION_DELETES, 2 = EQUALITY_DELETES
 * (Not to be confused with manifest list content: 0 = data manifest, 1 = delete manifest)
 *
 * @param path            Filesystem path to the manifest .avro file.
 * @param target_content  Only include entries with this content type, or -1 for all.
 * @return                Manifest entries matching the target content type.
 * @throws                std::runtime_error on I/O or parse errors.
 */
std::vector<IcebergDeleteFileEntry> read_iceberg_manifest_entries(const std::string& path,
                                                                  int target_content);

}  // namespace sirius::op::scan
