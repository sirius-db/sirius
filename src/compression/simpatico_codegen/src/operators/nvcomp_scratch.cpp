// SPDX-License-Identifier: Apache-2.0
#include "codegen/util/nvcomp_scratch.hpp"

namespace simpatico {

namespace detail {
void release_ans_manager_scratch();
void release_bitcomp_manager_scratch();
void release_cascaded_manager_scratch();
void release_snappy_manager_scratch();
void release_lz4_manager_scratch();
void release_deflate_manager_scratch();
}  // namespace detail

void release_nvcomp_manager_scratch()
{
  detail::release_ans_manager_scratch();
  detail::release_bitcomp_manager_scratch();
  detail::release_cascaded_manager_scratch();
  detail::release_snappy_manager_scratch();
  detail::release_lz4_manager_scratch();
  detail::release_deflate_manager_scratch();
}

}  // namespace simpatico
