//! Rust-owned stand-ins for native handles, used only by Miri test executables.
//! These model ownership and output slots, not YAML parsing or engine behavior.
#![allow(unsafe_op_in_unsafe_fn)]

use sirius_sys::c_api::*;
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
        *error = ptr::null_mut();
    }
    if out.is_null() {
        return SIRIUS_INVALID_ARGUMENT;
    }
    *out = ptr::null_mut();
    if let Some((status, message)) = FAILURE.with(|f| f.borrow_mut().take()) {
        if let Some(mut message) = message.filter(|_| !error.is_null()) {
            message.push(0);
            *error = Box::into_raw(Box::new(message)).cast();
        }
        return status;
    }
    *out = Arc::into_raw(Arc::new(P::default())).cast_mut().cast();
    SIRIUS_SUCCESS
}

unsafe fn retain<T, P>(value: *mut T) {
    if !value.is_null() {
        Arc::increment_strong_count(value.cast::<P>());
    }
}
unsafe fn release<T, P>(value: *mut T) {
    if !value.is_null() {
        Arc::decrement_strong_count(value.cast::<P>());
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn sirius_context_config_builder_create(
    out: *mut *mut SiriusContextConfigBuilder,
    error: *mut *mut SiriusError,
) -> u32 {
    create::<_, Builder>(out, error)
}
#[unsafe(no_mangle)]
unsafe extern "C" fn sirius_context_config_builder_from_yaml(
    path: *const c_char,
    size: usize,
    out: *mut *mut SiriusContextConfigBuilder,
    error: *mut *mut SiriusError,
) -> u32 {
    let bytes = std::slice::from_raw_parts(path.cast::<u8>(), size);
    PATH.with(|p| *p.borrow_mut() = bytes.to_vec());
    if bytes.contains(&0) {
        fail_next(SIRIUS_INVALID_ARGUMENT, Some(b"path contains NUL"));
    }
    create::<_, Builder>(out, error)
}
#[unsafe(no_mangle)]
unsafe extern "C" fn sirius_context_config_builder_build(
    builder: *const SiriusContextConfigBuilder,
    out: *mut *mut SiriusContextConfig,
    error: *mut *mut SiriusError,
) -> u32 {
    assert_eq!((*builder.cast::<Builder>()).0, 0);
    create::<_, Config>(out, error)
}
#[unsafe(no_mangle)]
unsafe extern "C" fn sirius_context_config_builder_retain(value: *mut SiriusContextConfigBuilder) {
    retain::<_, Builder>(value)
}
#[unsafe(no_mangle)]
unsafe extern "C" fn sirius_context_config_builder_release(value: *mut SiriusContextConfigBuilder) {
    release::<_, Builder>(value)
}
#[unsafe(no_mangle)]
unsafe extern "C" fn sirius_context_config_retain(value: *mut SiriusContextConfig) {
    if !value.is_null() {
        assert_eq!((*value.cast::<Config>()).0, [0; 2]);
    }
    retain::<_, Config>(value)
}
#[unsafe(no_mangle)]
unsafe extern "C" fn sirius_context_config_release(value: *mut SiriusContextConfig) {
    release::<_, Config>(value)
}
#[unsafe(no_mangle)]
unsafe extern "C" fn sirius_error_message(error: *const SiriusError) -> *const c_char {
    if error.is_null() {
        c"".as_ptr()
    } else {
        (*error.cast::<Vec<u8>>()).as_ptr().cast()
    }
}
#[unsafe(no_mangle)]
unsafe extern "C" fn sirius_error_message_size(error: *const SiriusError) -> usize {
    if error.is_null() {
        0
    } else {
        (*error.cast::<Vec<u8>>()).len() - 1
    }
}
#[unsafe(no_mangle)]
unsafe extern "C" fn sirius_error_destroy(error: *mut SiriusError) {
    if !error.is_null() {
        drop(Box::from_raw(error.cast::<Vec<u8>>()));
    }
}
#[unsafe(no_mangle)]
extern "C" fn sirius_abi_version() -> u32 {
    SIRIUS_ABI_VERSION
}
