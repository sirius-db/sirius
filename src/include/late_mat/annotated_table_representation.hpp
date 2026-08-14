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

// Origin-annotated GPU table representation (env gate: SIRIUS_EXP_LATE_MAT).
//
// v1 late-mat carrier: fully materialized data (a normal GPU table) with an
// IMMUTABLE origin annotation riding alongside — "these columns came from
// pinned entry E (columns C...), and these rows are selection S of global span
// R". Downstream code that knows nothing about late materialization keeps
// working unchanged (every existing consumer reaches the data through
// cast<gpu_table_representation> / dynamic_cast-to-base, both upcasts);
// late-mat consumers dynamic_cast to this type to read the annotation.
//
// Constructed ONLY by gate-on paths (the scan operator's gated output attach);
// gate off => this type never exists anywhere and behavior is byte-identical.
//
// LIFETIME / BOUNDARY SEMANTICS (v1):
//  - clone(): inherited => returns the BASE type; the annotation is dropped.
//  - downgrade/spill: the converter registry dispatches on EXACT typeid, so
//    this header registers delegating converters for (annotated -> host) and
//    (annotated -> gpu). They forward to the base-pair builtin converters —
//    i.e. the annotation is DROPPED at the spill boundary (the v1
//    force-materialize policy; the data is already materialized, so dropping
//    metadata is correct by construction). Consumers must always treat
//    "no annotation" as the fallback.

#include "late_mat/column_origin.hpp"

#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <memory>
#include <stdexcept>
#include <utility>

namespace sirius::late_mat {

/**
 * @brief Immutable per-batch late-mat annotation: per-column origins + the
 *        batch's row selection over its global pin-order span.
 */
struct batch_annotation {
  scan_batch_origin origin;  ///< per-column origins + global range + chunk index
  row_selection selection;   ///< which rows of origin.range are live (v1: dense)
};

/**
 * @brief GPU table representation carrying a late-mat origin annotation.
 *
 * Exactly a gpu_table_representation plus one shared_ptr of metadata; all
 * data-path behavior is inherited.
 */
class origin_annotated_gpu_table_representation final
  : public ::cucascade::gpu_table_representation {
 public:
  origin_annotated_gpu_table_representation(std::unique_ptr<cudf::table> table,
                                            ::cucascade::memory::memory_space& memory_space,
                                            rmm::cuda_stream_view writer_stream,
                                            std::shared_ptr<const batch_annotation> annotation)
    : ::cucascade::gpu_table_representation(std::move(table), memory_space, writer_stream),
      _annotation(std::move(annotation))
  {
    if (!_annotation) {
      throw std::invalid_argument(
        "[origin_annotated_gpu_table_representation] annotation must be non-null — use the "
        "plain gpu_table_representation when there is nothing to annotate");
    }
  }

  [[nodiscard]] const std::shared_ptr<const batch_annotation>& annotation() const noexcept
  {
    return _annotation;
  }

 private:
  std::shared_ptr<const batch_annotation> _annotation;
};

namespace detail {

/// Build a converter that forwards an annotated source to the base-pair
/// (gpu_table_representation -> TargetType) converter of @p registry, dropping
/// the annotation. The temporary base-typed view keeps stream lineage intact:
/// the conversion stream first waits on the source's writer event, then the
/// temp records its own writer event on that stream.
template <typename TargetType>
::cucascade::representation_converter_fn make_annotation_dropping_converter(
  ::cucascade::representation_converter_registry& registry)
{
  return [&registry](::cucascade::idata_representation& source,
                     const ::cucascade::memory::memory_space* target_memory_space,
                     rmm::cuda_stream_view stream,
                     ::cucascade::memory::reservation* reservation)
           -> std::unique_ptr<::cucascade::idata_representation> {
    // Base-class methods are all this needs, so one delegate serves every
    // Sirius gpu-representation subclass registered below.
    auto& annotated = source.cast<::cucascade::gpu_table_representation>();
    // Order the conversion stream after the source's writes, then hand the
    // base-pair converter a base-typed view whose writer event is recorded on
    // that ordered stream. `source` outlives this call (the caller owns it
    // until the conversion result replaces it), so no owner is needed beyond
    // an inert placeholder.
    // NOTE: deliberately the raw runtime call, NOT cucascade/cuda/event.hpp —
    // that header declares namespace cucascade::cuda, which from inside
    // namespace cucascade shadows ::cuda and breaks cucascade's own
    // `cuda::stream_ref` references in any TU that includes both.
    if (cudaEvent_t const ev = annotated.get_writer_event(); ev != nullptr) {
      if (auto const st = cudaStreamWaitEvent(stream.value(), ev, 0); st != cudaSuccess) {
        throw std::runtime_error(
          std::string("[late_mat annotation-dropping converter] cudaStreamWaitEvent failed: ") +
          cudaGetErrorString(st));
      }
    }
    ::cucascade::gpu_table_representation base_view(annotated.get_table_view(),
                                                    /*owner=*/int{0},
                                                    annotated.get_size_in_bytes(),
                                                    annotated.get_memory_space(),
                                                    stream);
    if (reservation != nullptr) {
      return registry.convert<TargetType>(base_view, *reservation, stream);
    }
    return registry.convert<TargetType>(base_view, target_memory_space, stream);
  };
}

}  // namespace detail

/**
 * @brief Register the late-mat converter pairs on @p registry.
 *
 * Registered UNCONDITIONALLY at startup (registration is one-time and free on
 * the hot path; it keeps gate-on/off binaries structurally identical):
 *  - annotated -> host_data_representation  (downgrade/spill boundary)
 *  - annotated -> gpu_table_representation  (cross-device moves)
 * Both drop the annotation (v1 force-materialize-at-boundary policy).
 */
inline void register_late_mat_converters(::cucascade::representation_converter_registry& registry)
{
  registry.register_converter<origin_annotated_gpu_table_representation,
                              ::cucascade::host_data_representation>(
    detail::make_annotation_dropping_converter<::cucascade::host_data_representation>(registry));
  registry.register_converter<origin_annotated_gpu_table_representation,
                              ::cucascade::gpu_table_representation>(
    detail::make_annotation_dropping_converter<::cucascade::gpu_table_representation>(registry));
}

}  // namespace sirius::late_mat
