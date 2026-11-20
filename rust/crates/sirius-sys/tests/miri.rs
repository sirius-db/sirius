//! Check raw pointer usage against Rust-owned C ABI stand-ins under Miri.
#![cfg(miri)]
#[path = "support/backend.rs"]
mod backend;

use sirius_sys::c_api::*;
use std::ptr;

#[test]
fn raw_handles_and_diagnostics() {
    let version: extern "C" fn() -> u32 = sirius_abi_version;
    assert_eq!(version(), SIRIUS_ABI_VERSION);
    // SAFETY: slots are writable, each owning reference is released once, and
    // diagnostic bytes are read only while their owner is live.
    unsafe {
        let mut builder = ptr::null_mut();
        assert_eq!(
            sirius_context_config_builder_create(&mut builder, ptr::null_mut()),
            SIRIUS_SUCCESS
        );
        sirius_context_config_builder_retain(builder);
        sirius_context_config_builder_release(builder);
        let mut config = ptr::null_mut();
        assert_eq!(
            sirius_context_config_builder_build(builder, &mut config, ptr::null_mut()),
            SIRIUS_SUCCESS
        );
        sirius_context_config_builder_release(builder);
        sirius_context_config_retain(config);
        sirius_context_config_release(config);
        sirius_context_config_release(config);

        backend::fail_next(SIRIUS_CONFIGURATION_IO, Some(b"bad\0path"));
        let mut error = ptr::null_mut();
        let path = b"file.yaml";
        assert_eq!(
            sirius_context_config_builder_from_yaml(
                path.as_ptr().cast(),
                path.len(),
                &mut builder,
                &mut error
            ),
            SIRIUS_CONFIGURATION_IO
        );
        assert!(builder.is_null());
        assert_eq!(backend::last_path(), path);
        assert_eq!(
            std::slice::from_raw_parts(
                sirius_error_message(error).cast::<u8>(),
                sirius_error_message_size(error)
            ),
            b"bad\0path"
        );
        sirius_error_destroy(error);
        assert!(!sirius_error_message(ptr::null()).is_null());
        assert_eq!(sirius_error_message_size(ptr::null()), 0);
        sirius_error_destroy(ptr::null_mut());
        sirius_context_config_release(ptr::null_mut());
        sirius_context_config_builder_release(ptr::null_mut());
    }
}
