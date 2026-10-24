/*
 * Copyright (c) 2026, Alibaba Group Holding Limited
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <stdint.h>

#define PAIMON_READ_BRIDGE_MAGIC   UINT64_C(0x5041494d4f4e5244)
#define PAIMON_READ_BRIDGE_VERSION 1

enum PaimonReadStatus {
  PAIMON_READ_OK               = 0,
  PAIMON_READ_INVALID_ARGUMENT = 1,
  PAIMON_READ_UNSUPPORTED      = 2,
  PAIMON_READ_OUT_OF_MEMORY    = 3,
  PAIMON_READ_INTERNAL_ERROR   = 4,
  PAIMON_READ_EOF              = 5,
  PAIMON_READ_RESOURCE_REFUSED = 6,
  PAIMON_READ_CANCELLED        = 7
};

struct PaimonReadString {
  const char* data;
  uint64_t size;
};

struct PaimonReadOption {
  struct PaimonReadString key;
  struct PaimonReadString value;
};

/* All views remain valid until their descriptor is released. Never free them. */
struct PaimonBoundReadViewV1 {
  struct PaimonReadString warehouse;
  struct PaimonReadString database;
  struct PaimonReadString table;
  struct PaimonReadString location;
  struct PaimonReadString branch;
  /* Native schema JSON includes stable field IDs and bound field types. */
  struct PaimonReadString schema_json;
  int64_t schema_id;
  int64_t snapshot_id;
  /* Zero means empty at bind time, NOT "resolve latest". */
  uint32_t has_snapshot;
  uint64_t option_count;
  const struct PaimonReadOption* options;
};

/* Descriptor-only interface. Exact V1 does not authorize planning, reading or replay. */
struct PaimonReadBridgeV1 {
  uint64_t magic;
  uint32_t version;
  uint32_t struct_size;
  /* bind_data must be NULL or a live DuckDB FunctionData on the matching host. */
  int32_t (*accepts_bind)(const void* bind_data);
  /* On failure, *descriptor is NULL. Diagnostics use a caller-owned buffer. */
  int32_t (*retain)(const void* bind_data, void** descriptor, char* error, uint64_t error_capacity);
  /* Only a live descriptor from this provider may be passed to describe/release. */
  int32_t (*describe)(const void* descriptor,
                      struct PaimonBoundReadViewV1* view,
                      uint64_t view_size);
  /* Release once; NULL is permitted. No concurrent release and describe. */
  void (*release)(void* descriptor);
};

/* V2 preserves the descriptor prefix and adds experimental native reads.
 * V1-only consumers must reject V2 rather than assume binary compatibility. */
#define PAIMON_READ_BRIDGE_NATIVE_VERSION 2
struct ArrowArray;
struct ArrowSchema;

/* Borrowed only for the duration of a call. Callback must be thread-safe and
 * must not throw or re-enter the bridge. Cancellation is checked at whole-call
 * boundaries; blocking native I/O and planning are not interruptible. */
struct PaimonReadCancellationV1 {
  const void* state;
  int32_t (*is_cancelled)(const void* state);
};

struct PaimonPlanOptionsV1 {
  uint64_t struct_size;
  /* Finite limits are refused BEFORE native planning. UINT64_MAX explicitly
   * opts into unbounded planning for qualification, not production admission. */
  uint64_t planning_byte_limit;
};

struct PaimonReadBridgeV2 {
  struct PaimonReadBridgeV1 descriptor;
  int32_t (*create_plan)(const void* descriptor,
                         const struct PaimonPlanOptionsV1* options,
                         const struct PaimonReadCancellationV1* cancel,
                         void** plan,
                         uint64_t* split_count,
                         char* error,
                         uint64_t error_capacity);
  void (*release_plan)(void* plan);
  int32_t (*get_split)(const void* plan, uint64_t index, void** split);
  void (*release_split)(void* split);
  int32_t (*open_reader)(const void* split,
                         const struct PaimonReadCancellationV1* cancel,
                         void** reader,
                         char* error,
                         uint64_t error_capacity);
  void (*release_reader)(void* reader);
  /* At most one outstanding batch per reader. A live batch refuses advancement.
   * Returned complete roots are borrowed, including _VALUE_KIND as column 0.
   * Never move, release, or modify roots/children. Release only via release_batch.
   * A batch retains reader/pool/plan owners, even after other handles release.
   * Error/cancellation terminates the reader; subsequent calls repeat status. */
  int32_t (*next_batch)(void* reader,
                        const struct PaimonReadCancellationV1* cancel,
                        void** batch,
                        const struct ArrowArray** array,
                        const struct ArrowSchema** schema,
                        char* error,
                        uint64_t error_capacity);
  void (*release_batch)(void* batch);
};
/* Every handle is provider-owned; release exactly once, NULL permitted. Only
 * live handles of the correct type from this provider are valid inputs.
 * Keep the authenticated producer module and matching host loaded until ALL
 * handles/roots release. Serialize operations/release on the same handle.
 * Independent readers may run concurrently; one reader serializes native work.
 * No replay, predicate, projection or hard memory-limit guarantee is supplied. */
