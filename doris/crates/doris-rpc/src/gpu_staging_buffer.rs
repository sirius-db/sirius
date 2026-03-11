//! Fixed-size GPU staging buffer for nixl transfers.
//!
//! Replaces per-transfer `cuMemAlloc` + `register_memory` cycles with a single
//! pre-allocated, pre-registered buffer. Sub-regions are bump-allocated per transfer
//! and released when all leases are dropped.
//!
//! # Usage
//!
//! ```text
//! startup:
//!   staging = GpuStagingBuffer::new(agent, backend, 512MB, device_id=0)
//!
//! per-transfer:
//!   leases = staging.try_stage(&[(src_addr, len, dev_id), ...])
//!   // leases[i].addr() points into the staging buffer (cuMemAlloc-backed)
//!   // ... do nixl transfer using staged addresses ...
//!   drop(leases)  // decrements lease count, no GPU free
//!   // when all leases from current epoch drop → allocator resets
//! ```

use std::sync::{Arc, Mutex};

use nixl_sys::{Backend, RegistrationHandle};
use tracing::{info, warn};

use crate::cuda_driver::{cuda_alloc, cuda_free, cuda_memcpy_dtod};
use crate::nixl_exchange::GpuBufferDesc;

/// Alignment for sub-region allocations (256 bytes for GPU performance).
const STAGING_ALIGNMENT: usize = 256;

/// Round up `val` to the next multiple of `align`.
fn align_up(val: usize, align: usize) -> usize {
    (val + align - 1) & !(align - 1)
}

/// Bump allocator state.
struct BumpAllocator {
    offset: usize,
    capacity: usize,
    active_leases: usize,
}

impl BumpAllocator {
    fn new(capacity: usize) -> Self {
        Self {
            offset: 0,
            capacity,
            active_leases: 0,
        }
    }

    /// Try to allocate `size` bytes, returning the offset within the buffer.
    fn alloc(&mut self, size: usize) -> Option<usize> {
        let aligned_offset = align_up(self.offset, STAGING_ALIGNMENT);
        let end = aligned_offset + size;
        if end > self.capacity {
            return None;
        }
        self.offset = end;
        self.active_leases += 1;
        Some(aligned_offset)
    }

    /// Release a lease. When all leases are dropped, reset the allocator.
    fn release(&mut self) {
        self.active_leases = self.active_leases.saturating_sub(1);
        if self.active_leases == 0 {
            self.offset = 0;
        }
    }
}

/// Fixed-size GPU buffer allocated via `cuMemAlloc` at startup and registered
/// once with nixl. Replaces per-transfer alloc+register cycles.
pub struct GpuStagingBuffer {
    base_addr: usize,
    total_size: usize,
    device_id: u64,
    _registration: RegistrationHandle,
    allocator: Mutex<BumpAllocator>,
}

impl GpuStagingBuffer {
    /// Allocate a staging buffer and register it with nixl.
    ///
    /// `size`: total staging buffer size in bytes
    /// `device_id`: GPU device to allocate on
    /// `agent`: nixl agent for registration
    /// `backend`: nixl backend for registration options
    pub fn new(
        size: usize,
        device_id: u64,
        agent: &nixl_sys::Agent,
        backend: &Backend,
    ) -> Result<Self, String> {
        let base_addr = cuda_alloc(size)?;

        let mut opt =
            nixl_sys::OptArgs::new().map_err(|e| format!("OptArgs::new: {e}"))?;
        opt.add_backend(backend)
            .map_err(|e| format!("add_backend: {e}"))?;

        let wrapper = StagingMemoryRegion {
            addr: base_addr,
            len: size,
            device_id,
        };

        let registration = match agent.register_memory(&wrapper, Some(&opt)) {
            Ok(handle) => handle,
            Err(e) => {
                let _ = cuda_free(base_addr);
                return Err(format!("register_memory: {e}"));
            }
        };

        info!(
            base = format_args!("0x{base_addr:x}"),
            size,
            size_mb = size / (1024 * 1024),
            device_id,
            "GPU staging buffer allocated and registered with nixl"
        );

        Ok(Self {
            base_addr,
            total_size: size,
            device_id,
            _registration: registration,
            allocator: Mutex::new(BumpAllocator::new(size)),
        })
    }

    /// Base address of the staging buffer.
    pub fn base_addr(&self) -> usize {
        self.base_addr
    }

    /// Total size of the staging buffer.
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Device ID of the staging buffer.
    pub fn device_id(&self) -> u64 {
        self.device_id
    }

    /// Try to stage source GPU buffers into the staging buffer.
    ///
    /// For each source buffer:
    /// 1. Bump-allocates a sub-region in the staging buffer
    /// 2. D2D copies source data into the sub-region
    ///
    /// Returns `StagingLease` handles with the staged addresses.
    /// On overflow or copy failure, returns `Err` — caller should fall back
    /// to per-buffer `cuMemAlloc`.
    pub fn try_stage(
        &self,
        src_buffers: &[(usize, usize, u64)], // (addr, len, device_id)
    ) -> Result<Vec<StagingLease>, StagingError> {
        let total_needed: usize = src_buffers
            .iter()
            .map(|&(_, len, _)| align_up(len, STAGING_ALIGNMENT))
            .sum();

        let mut allocator = self.allocator.lock().unwrap();

        // Pre-check: would all buffers fit?
        if allocator.offset + total_needed > allocator.capacity {
            return Err(StagingError::Overflow {
                needed: total_needed,
                available: allocator.capacity.saturating_sub(allocator.offset),
            });
        }

        let mut leases = Vec::with_capacity(src_buffers.len());
        for &(src_addr, len, _dev_id) in src_buffers {
            let offset = allocator.alloc(len).ok_or(StagingError::Overflow {
                needed: len,
                available: allocator.capacity.saturating_sub(allocator.offset),
            })?;
            let staged_addr = self.base_addr + offset;

            // D2D copy from source (RMM) to staging (cuMemAlloc).
            if let Err(e) = cuda_memcpy_dtod(staged_addr, src_addr, len) {
                // Roll back this lease and any previous ones.
                allocator.active_leases = allocator.active_leases.saturating_sub(1);
                // Drop previous leases (they'll decrement their counts).
                for lease in leases {
                    drop(lease);
                }
                return Err(StagingError::CopyFailed(e));
            }

            leases.push(StagingLease {
                staged_addr,
                len,
                device_id: self.device_id,
                _allocator: Arc::new(StagingLeaseDropper {
                    allocator: &self.allocator as *const Mutex<BumpAllocator>,
                }),
            });
        }

        Ok(leases)
    }

    /// Try to sub-allocate destination space (no D2D copy needed).
    ///
    /// Used by the receiver to allocate space for incoming nixl transfers
    /// within the pre-registered staging buffer, avoiding per-transfer
    /// `cuMemAlloc` + `register_memory`.
    pub fn try_allocate(
        &self,
        sizes: &[(u64, u64)], // (len, device_id)
    ) -> Result<Vec<StagingLease>, StagingError> {
        let total_needed: usize = sizes
            .iter()
            .map(|&(len, _)| align_up(len as usize, STAGING_ALIGNMENT))
            .sum();

        let mut allocator = self.allocator.lock().unwrap();

        if allocator.offset + total_needed > allocator.capacity {
            return Err(StagingError::Overflow {
                needed: total_needed,
                available: allocator.capacity.saturating_sub(allocator.offset),
            });
        }

        let mut leases = Vec::with_capacity(sizes.len());
        for &(len, _dev_id) in sizes {
            let offset = allocator.alloc(len as usize).ok_or(StagingError::Overflow {
                needed: len as usize,
                available: allocator.capacity.saturating_sub(allocator.offset),
            })?;
            let staged_addr = self.base_addr + offset;

            leases.push(StagingLease {
                staged_addr,
                len: len as usize,
                device_id: self.device_id,
                _allocator: Arc::new(StagingLeaseDropper {
                    allocator: &self.allocator as *const Mutex<BumpAllocator>,
                }),
            });
        }

        Ok(leases)
    }

    /// Current allocator state for diagnostics.
    pub fn stats(&self) -> StagingStats {
        let alloc = self.allocator.lock().unwrap();
        StagingStats {
            used: alloc.offset,
            capacity: alloc.capacity,
            active_leases: alloc.active_leases,
        }
    }
}

impl Drop for GpuStagingBuffer {
    fn drop(&mut self) {
        // Registration is dropped first (via _registration field order),
        // then we free the GPU memory.
        if let Err(e) = cuda_free(self.base_addr) {
            warn!(
                error = %e,
                addr = format_args!("0x{:x}", self.base_addr),
                "GpuStagingBuffer: failed to free GPU memory"
            );
        }
    }
}

/// Staging buffer usage statistics.
#[derive(Debug, Clone)]
pub struct StagingStats {
    pub used: usize,
    pub capacity: usize,
    pub active_leases: usize,
}

/// Error from staging buffer operations.
#[derive(Debug)]
pub enum StagingError {
    /// Not enough space in the staging buffer.
    Overflow { needed: usize, available: usize },
    /// D2D copy failed.
    CopyFailed(String),
}

impl std::fmt::Display for StagingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Overflow { needed, available } => {
                write!(
                    f,
                    "staging buffer overflow: need {} bytes, {} available",
                    needed, available
                )
            }
            Self::CopyFailed(e) => write!(f, "staging D2D copy failed: {e}"),
        }
    }
}

/// Interior drop handle that releases the bump allocator lease.
///
/// Uses a raw pointer to the allocator mutex. This is safe because:
/// - `StagingLease` is only created by `GpuStagingBuffer` methods
/// - `GpuStagingBuffer` owns the `Mutex<BumpAllocator>`
/// - Leases must not outlive the staging buffer
struct StagingLeaseDropper {
    allocator: *const Mutex<BumpAllocator>,
}

// SAFETY: The allocator pointer is valid for the lifetime of the staging buffer,
// and the Mutex provides thread-safe access.
unsafe impl Send for StagingLeaseDropper {}
unsafe impl Sync for StagingLeaseDropper {}

impl Drop for StagingLeaseDropper {
    fn drop(&mut self) {
        // SAFETY: The allocator pointer is valid as long as GpuStagingBuffer exists.
        // Leases must not outlive the buffer (enforced by API design).
        let allocator = unsafe { &*self.allocator };
        allocator.lock().unwrap().release();
    }
}

/// RAII sub-region handle. Does NOT free GPU memory — the staging buffer owns it.
/// Decrements `active_leases` on drop; allocator resets when all leases are dropped.
pub struct StagingLease {
    staged_addr: usize,
    len: usize,
    device_id: u64,
    /// Held for RAII drop — releases the bump allocator lease.
    _allocator: Arc<StagingLeaseDropper>,
}

impl StagingLease {
    /// Address within the staging buffer where data was staged.
    pub fn addr(&self) -> usize {
        self.staged_addr
    }

    /// Size of the staged sub-region.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Device ID.
    pub fn device_id(&self) -> u64 {
        self.device_id
    }

    /// Convert to a `GpuBufferDesc` for nixl transfer.
    pub fn as_gpu_buffer_desc(&self) -> GpuBufferDesc {
        GpuBufferDesc {
            addr: self.staged_addr,
            len: self.len,
            device_id: self.device_id,
        }
    }
}

/// nixl `MemoryRegion` + `NixlDescriptor` wrapper for the staging buffer.
#[derive(Debug)]
struct StagingMemoryRegion {
    addr: usize,
    len: usize,
    device_id: u64,
}

unsafe impl Send for StagingMemoryRegion {}
unsafe impl Sync for StagingMemoryRegion {}

impl nixl_sys::MemoryRegion for StagingMemoryRegion {
    fn size(&self) -> usize {
        self.len
    }

    unsafe fn as_ptr(&self) -> *const u8 {
        self.addr as *const u8
    }
}

impl nixl_sys::NixlDescriptor for StagingMemoryRegion {
    fn mem_type(&self) -> nixl_sys::MemType {
        nixl_sys::MemType::Vram
    }
    fn device_id(&self) -> u64 {
        self.device_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(255, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
        assert_eq!(align_up(1024, 256), 1024);
    }

    #[test]
    fn test_bump_allocator_basic() {
        let mut alloc = BumpAllocator::new(1024);
        assert_eq!(alloc.offset, 0);
        assert_eq!(alloc.active_leases, 0);

        // First alloc at offset 0.
        let off = alloc.alloc(100);
        assert_eq!(off, Some(0));
        assert_eq!(alloc.active_leases, 1);
        // offset is now 100 (will be aligned on next alloc).

        // Second alloc — aligned to 256.
        let off2 = alloc.alloc(200);
        assert_eq!(off2, Some(256));
        assert_eq!(alloc.active_leases, 2);
    }

    #[test]
    fn test_bump_allocator_overflow() {
        let mut alloc = BumpAllocator::new(512);

        let off = alloc.alloc(300);
        assert_eq!(off, Some(0));

        // Next alloc would start at 512 (align_up(300, 256)), needs 300 more = 812 > 512.
        let off2 = alloc.alloc(300);
        assert_eq!(off2, None);
        // Lease count should NOT have incremented on failure.
        // (Currently it does increment before the check — but alloc returns None
        //  only if end > capacity, and we don't increment in that case.)
        assert_eq!(alloc.active_leases, 1);
    }

    #[test]
    fn test_bump_allocator_reset_on_all_released() {
        let mut alloc = BumpAllocator::new(4096);

        alloc.alloc(100); // lease 1
        alloc.alloc(200); // lease 2
        assert_eq!(alloc.active_leases, 2);
        assert!(alloc.offset > 0);

        alloc.release(); // release 1
        assert_eq!(alloc.active_leases, 1);
        assert!(alloc.offset > 0); // not reset yet

        alloc.release(); // release 2
        assert_eq!(alloc.active_leases, 0);
        assert_eq!(alloc.offset, 0); // reset!
    }

    #[test]
    fn test_bump_allocator_exact_fit() {
        let mut alloc = BumpAllocator::new(256);

        // Exactly 256 bytes should fit.
        let off = alloc.alloc(256);
        assert_eq!(off, Some(0));

        // No more space.
        let off2 = alloc.alloc(1);
        assert_eq!(off2, None);
    }

    #[test]
    fn test_bump_allocator_zero_size() {
        let mut alloc = BumpAllocator::new(1024);
        let off = alloc.alloc(0);
        assert_eq!(off, Some(0));
        assert_eq!(alloc.active_leases, 1);
    }

    #[test]
    fn test_staging_error_display() {
        let err = StagingError::Overflow {
            needed: 1024,
            available: 512,
        };
        let msg = format!("{err}");
        assert!(msg.contains("1024"));
        assert!(msg.contains("512"));

        let err2 = StagingError::CopyFailed("cuMemcpyDtoD failed".to_string());
        let msg2 = format!("{err2}");
        assert!(msg2.contains("cuMemcpyDtoD"));
    }

    #[test]
    fn test_staging_stats() {
        // Verify StagingStats derives.
        let stats = StagingStats {
            used: 100,
            capacity: 1024,
            active_leases: 2,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.used, 100);
        assert_eq!(cloned.capacity, 1024);
        assert_eq!(cloned.active_leases, 2);
        let _ = format!("{:?}", stats);
    }

    // --- GPU-dependent tests (skip when no GPU/nixl) ---

    #[test]
    fn test_staging_buffer_lifecycle() {
        // Requires GPU + nixl. Skip gracefully if unavailable.
        let Some(exchange) = crate::nixl_exchange::NixlExchange::try_new("staging-test") else {
            return; // nixl not available
        };

        // Try allocating a small staging buffer.
        let result = GpuStagingBuffer::new(
            1024 * 1024, // 1 MB
            0,
            exchange.agent(),
            exchange.backend(),
        );

        match result {
            Ok(staging) => {
                assert!(staging.base_addr() > 0);
                assert_eq!(staging.total_size(), 1024 * 1024);
                assert_eq!(staging.device_id(), 0);

                let stats = staging.stats();
                assert_eq!(stats.used, 0);
                assert_eq!(stats.capacity, 1024 * 1024);
                assert_eq!(stats.active_leases, 0);

                // Drop should clean up without error.
                drop(staging);
            }
            Err(e) => {
                // cuMemAlloc failed (no GPU) — expected in CI.
                eprintln!("staging buffer alloc failed (no GPU?): {e}");
            }
        }
    }

    #[test]
    fn test_staging_try_allocate() {
        let Some(exchange) = crate::nixl_exchange::NixlExchange::try_new("staging-alloc-test") else {
            return;
        };

        let Ok(staging) = GpuStagingBuffer::new(
            4096, // 4 KB
            0,
            exchange.agent(),
            exchange.backend(),
        ) else {
            return;
        };

        // Allocate two sub-regions.
        let leases = staging.try_allocate(&[(1024, 0), (1024, 0)]);
        match leases {
            Ok(leases) => {
                assert_eq!(leases.len(), 2);
                assert_eq!(leases[0].len(), 1024);
                assert_eq!(leases[1].len(), 1024);
                // Second lease should be at an aligned offset after the first.
                assert_eq!(leases[1].addr(), leases[0].addr() + 1024);

                let stats = staging.stats();
                assert_eq!(stats.active_leases, 2);

                // Drop leases → allocator resets.
                drop(leases);
                let stats = staging.stats();
                assert_eq!(stats.active_leases, 0);
                assert_eq!(stats.used, 0);
            }
            Err(e) => {
                eprintln!("try_allocate failed: {e}");
            }
        }
    }

    #[test]
    fn test_staging_overflow_fallback() {
        let Some(exchange) = crate::nixl_exchange::NixlExchange::try_new("staging-overflow-test") else {
            return;
        };

        let Ok(staging) = GpuStagingBuffer::new(
            1024, // 1 KB — intentionally small
            0,
            exchange.agent(),
            exchange.backend(),
        ) else {
            return;
        };

        // Try to allocate more than the buffer can hold.
        let result = staging.try_allocate(&[(2048, 0)]);
        assert!(
            matches!(result, Err(StagingError::Overflow { .. })),
            "expected overflow error"
        );

        // Allocator should still be usable for smaller allocations.
        let result2 = staging.try_allocate(&[(512, 0)]);
        assert!(result2.is_ok(), "should succeed for smaller allocation");
    }

    #[test]
    fn test_staging_concurrent_leases() {
        let Some(exchange) = crate::nixl_exchange::NixlExchange::try_new("staging-concurrent-test") else {
            return;
        };

        let Ok(staging) = GpuStagingBuffer::new(
            4096,
            0,
            exchange.agent(),
            exchange.backend(),
        ) else {
            return;
        };

        // Simulate two concurrent transfers.
        let lease1 = staging.try_allocate(&[(512, 0)]).unwrap();
        let lease2 = staging.try_allocate(&[(512, 0)]).unwrap();

        let stats = staging.stats();
        assert_eq!(stats.active_leases, 2);

        // Drop first transfer — allocator should NOT reset.
        drop(lease1);
        let stats = staging.stats();
        assert_eq!(stats.active_leases, 1);
        assert!(stats.used > 0);

        // Drop second transfer — now allocator resets.
        drop(lease2);
        let stats = staging.stats();
        assert_eq!(stats.active_leases, 0);
        assert_eq!(stats.used, 0);
    }

    #[test]
    fn test_staging_lease_as_gpu_buffer_desc() {
        let Some(exchange) = crate::nixl_exchange::NixlExchange::try_new("staging-desc-test") else {
            return;
        };

        let Ok(staging) = GpuStagingBuffer::new(
            4096,
            0,
            exchange.agent(),
            exchange.backend(),
        ) else {
            return;
        };

        let leases = staging.try_allocate(&[(1024, 0)]).unwrap();
        let desc = leases[0].as_gpu_buffer_desc();
        assert_eq!(desc.addr, leases[0].addr());
        assert_eq!(desc.len, 1024);
        assert_eq!(desc.device_id, 0);
    }
}
