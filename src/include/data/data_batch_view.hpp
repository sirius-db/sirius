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
#include <variant>
#include <memory>
#include <cudf/table/table.hpp>

#include "helper/helper.hpp"
#include "data/common.hpp"

namespace sirius {

using sirius::memory::Tier;

class DataBatchView : public cudf::table_view {
public:
    DataBatchView(DataBatch* batch, std::vector<cudf::column_view> const& cols)
        : cudf::table_view(cols), batch_(batch) {
        batch_->increment_refcount();
    }

    DataBatchView(const DataBatchView& other)
        : cudf::table_view(other), batch_(other.batch_) {
        batch_->increment_refcount();
    }

    DataBatchView& operator=(const DataBatchView& other) {
        if (this != &other) {
            cudf::table_view::operator=(other);
            batch_ = other.batch_;
            batch_->increment_refcount();
        }
        return *this;
    }

    ~DataBatchView() {
        batch_->decrement_refcount();
    }

private:
    DataBatch* batch_;
};

} // namespace sirius