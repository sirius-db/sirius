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

namespace sirius::pipeline {

//! Lightweight context for plan-time pipeline construction.
//! Replaces the sirius_engine& dependency so that pipelines can be built
//! without an engine instance (e.g. at optimizer/bind time).
struct pipeline_build_context {
  //! Whether query results must preserve insertion order
  //! (from DuckDB's PreserveInsertionOrderSetting)
  bool preserve_insertion_order = true;
};

}  // namespace sirius::pipeline
