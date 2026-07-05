// SPDX-License-Identifier: Apache-2.0
//
// Release the GPU scratch nvcomp's Manager objects cache thread-locally.
#pragma once

namespace simpatico {

/// Frees the GPU scratch buffers cached inside the thread-local nvcomp
/// Manager objects (ans/bitcomp/cascaded/snappy/lz4/deflate). Each Manager's
/// scratch only grows to the high-water mark of every compress/decompress
/// call it has served — nvcomp's `ManagerBase::allocate_gpu_scratch` never
/// shrinks it back down — so code that tries many operators against many
/// differently-sized columns on one thread (e.g. `explore`'s beam search)
/// can end up permanently pinning hundreds of MB to GBs of GPU memory per
/// codec that outlives the call that needed it. This memory is allocated
/// through nvcomp's own default allocator, not RMM, so it is invisible to
/// (and not reclaimable by) the RMM pool used everywhere else. Safe to call
/// any time: the next compress/decompress on this thread simply reallocates
/// on demand.
void release_nvcomp_manager_scratch();

}  // namespace simpatico
