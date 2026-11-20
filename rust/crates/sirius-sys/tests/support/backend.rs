//! Rust-owned stand-ins for native handles, used only by Miri test executables.
//! These model ownership and output slots, not YAML parsing or engine behavior.
//!
//! SAFETY: the unmangled symbols here implement the test C ABI. The Miri build
//! skips native linking and includes exactly one copy of this backend.

use crate::c_api::*;
use std::{cell::RefCell, ffi::c_char, ptr, sync::Arc};

thread_local! {
    static FAILURE: RefCell<Option<(u32, Option<Vec<u8>>)>> = const { RefCell::new(None) };
    static PATH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

pub fn fail_next(status: u32, message: Option<&[u8]>) {
    FAILURE.with(|f| *f.borrow_mut() = Some((status, message.map(Vec::from))));
}

pub fn last_path() -> Vec<u8> {
    PATH.with(|p| p.borrow().clone())
}

// Each successful factory transfers one Arc reference through an opaque pointer.
// Reading the payload makes Miri check that borrowed handles remain live.
#[derive(Default)]
struct Builder(u8);
#[derive(Default)]
struct Config([u64; 2]);

unsafe fn create<T, P: Default>(out: *mut *mut T, error: *mut *mut SiriusError) -> u32 {
    if !error.is_null() {
        // SAFETY: a supplied diagnostic slot is writable for this call.
        unsafe { *error = ptr::null_mut() };
    }
    if out.is_null() {
        return SIRIUS_INVALID_ARGUMENT;
    }
    // SAFETY: the caller supplies a writable handle slot, checked above for null.
    unsafe { *out = ptr::null_mut() };
    if let Some((status, message)) = FAILURE.with(|f| f.borrow_mut().take()) {
        if let Some(mut message) = message.filter(|_| !error.is_null()) {
            message.push(0);
            // SAFETY: the non-null slot receives sole ownership of this boxed diagnostic.
            unsafe { *error = Box::into_raw(Box::new(message)).cast() };
        }
        return status;
    }
    // SAFETY: the output slot receives one strong reference to the new payload.
    unsafe { *out = Arc::into_raw(Arc::new(P::default())).cast_mut().cast() };
    SIRIUS_SUCCESS
}

unsafe fn retain<T, P>(value: *mut T) {
    if !value.is_null() {
        // SAFETY: value came from Arc::<P>::into_raw and the caller holds a live reference.
        unsafe { Arc::increment_strong_count(value.cast::<P>()) };
    }
}
unsafe fn release<T, P>(value: *mut T) {
    if !value.is_null() {
        // SAFETY: the caller transfers one outstanding Arc<P> reference for release.
        unsafe { Arc::decrement_strong_count(value.cast::<P>()) };
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn sirius_context_config_builder_create(
    out: *mut *mut SiriusContextConfigBuilder,
    error: *mut *mut SiriusError,
) -> u32 {
    // SAFETY: the caller provides writable, non-overlapping output slots or null.
    unsafe { create::<_, Builder>(out, error) }
}
#[unsafe(no_mangle)]
unsafe extern "C" fn sirius_context_config_builder_from_yaml(
    path: *const c_char,
    size: usize,
    out: *mut *mut SiriusContextConfigBuilder,
    error: *mut *mut SiriusError,
) -> u32 {
    // SAFETY: the calling wrapper borrows a non-null path buffer of size bytes for this call.
    let bytes = unsafe { std::slice::from_raw_parts(path.cast::<u8>(), size) };
    PATH.with(|p| *p.borrow_mut() = bytes.to_vec());
    if bytes.contains(&0) {
        fail_next(SIRIUS_INVALID_ARGUMENT, Some(b"path contains NUL"));
    }
    // SAFETY: the caller provides writable, non-overlapping output slots or null.
    unsafe { create::<_, Builder>(out, error) }
}
#[unsafe(no_mangle)]
unsafe extern "C" fn sirius_context_config_builder_build(
    builder: *const SiriusContextConfigBuilder,
    out: *mut *mut SiriusContextConfig,
    error: *mut *mut SiriusError,
) -> u32 {
    // SAFETY: the caller borrows a live Builder allocated by this backend.
    let builder = unsafe { &*builder.cast::<Builder>() };
    assert_eq!(builder.0, 0);
    // SAFETY: the caller provides writable, non-overlapping output slots or null.
    unsafe { create::<_, Config>(out, error) }
}
#[unsafe(no_mangle)]
unsafe extern "C" fn sirius_context_config_builder_retain(value: *mut SiriusContextConfigBuilder) {
    // SAFETY: a non-null handle borrows a live Arc<Builder> reference.
    unsafe { retain::<_, Builder>(value) }
}
#[unsafe(no_mangle)]
unsafe extern "C" fn sirius_context_config_builder_release(value: *mut SiriusContextConfigBuilder) {
    // SAFETY: a non-null handle surrenders an owned Arc<Builder> reference.
    unsafe { release::<_, Builder>(value) }
}
#[unsafe(no_mangle)]
unsafe extern "C" fn sirius_context_config_retain(value: *mut SiriusContextConfig) {
    if !value.is_null() {
        // SAFETY: non-null configuration handles point to a live Config in this backend.
        let config = unsafe { &*value.cast::<Config>() };
        assert_eq!(config.0, [0; 2]);
    }
    // SAFETY: a non-null handle borrows a live Arc<Config> reference.
    unsafe { retain::<_, Config>(value) }
}
#[unsafe(no_mangle)]
unsafe extern "C" fn sirius_context_config_release(value: *mut SiriusContextConfig) {
    // SAFETY: a non-null handle surrenders an owned Arc<Config> reference.
    unsafe { release::<_, Config>(value) }
}
#[unsafe(no_mangle)]
unsafe extern "C" fn sirius_error_message(error: *const SiriusError) -> *const c_char {
    if error.is_null() {
        c"".as_ptr()
    } else {
        // SAFETY: the caller keeps this boxed, NUL-terminated message live while borrowing it.
        unsafe { (*error.cast::<Vec<u8>>()).as_ptr().cast() }
    }
}
#[unsafe(no_mangle)]
unsafe extern "C" fn sirius_error_message_size(error: *const SiriusError) -> usize {
    if error.is_null() {
        0
    } else {
        // SAFETY: error points to a live message whose final byte is the appended terminator.
        unsafe { (*error.cast::<Vec<u8>>()).len() - 1 }
    }
}
#[unsafe(no_mangle)]
unsafe extern "C" fn sirius_error_destroy(error: *mut SiriusError) {
    if !error.is_null() {
        // SAFETY: create transferred the Box with into_raw and the caller returns it exactly once.
        drop(unsafe { Box::from_raw(error.cast::<Vec<u8>>()) });
    }
}
#[unsafe(no_mangle)]
extern "C" fn sirius_abi_version() -> u32 {
    SIRIUS_ABI_VERSION
}
