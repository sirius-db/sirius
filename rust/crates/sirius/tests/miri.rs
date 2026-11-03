//! Exercise safe wrappers against Rust-owned C ABI stand-ins under Miri.
#![cfg(miri)]
#[path = "../../sirius-sys/tests/support/backend.rs"]
mod backend;

use sirius::{ConfigError, ContextConfigBuilder};
use sirius_sys::c_api::*;

#[test]
fn independent_handles_and_repeated_builds() {
    let builder = ContextConfigBuilder::new().unwrap();
    let copy = builder.try_clone().unwrap();
    drop(builder);
    let first = copy.build().unwrap();
    let second = copy.build().unwrap();
    drop(copy);
    let retained = first.try_clone().unwrap();
    drop(first);
    drop(second);
    drop(retained);
}

#[test]
fn diagnostics_are_copied_before_release() {
    for (status, message) in [
        (SIRIUS_CONFIGURATION_IO, b"a\0b\xff".as_slice()),
        (SIRIUS_MALFORMED_YAML, b"malformed".as_slice()),
        (SIRIUS_INVALID_CONFIGURATION, b"invalid".as_slice()),
        (999, b"unexpected".as_slice()),
    ] {
        backend::fail_next(status, Some(message));
        let error = ContextConfigBuilder::from_yaml("config.yaml")
            .err()
            .unwrap();
        let text = match error {
            ConfigError::Io(s)
            | ConfigError::MalformedYaml(s)
            | ConfigError::InvalidConfiguration(s) => s,
            ConfigError::Unexpected {
                status: 999,
                message,
            } => message,
            other => panic!("unexpected error: {other}"),
        };
        assert_eq!(text, String::from_utf8_lossy(message));
    }
    backend::fail_next(999, None);
    assert!(matches!(ContextConfigBuilder::new(),
        Err(ConfigError::Unexpected { message, .. }) if message.is_empty()));
}

#[test]
fn failed_factories_leave_no_owned_handle() {
    backend::fail_next(SIRIUS_ALLOCATION_FAILURE, None);
    assert!(matches!(
        ContextConfigBuilder::new(),
        Err(ConfigError::AllocationFailure)
    ));
    let builder = ContextConfigBuilder::new().unwrap();
    backend::fail_next(SIRIUS_ALLOCATION_FAILURE, Some(b"allocation"));
    assert!(matches!(
        builder.build(),
        Err(ConfigError::AllocationFailure)
    ));
    builder.build().unwrap();
    assert!(matches!(
        ContextConfigBuilder::from_yaml("bad\0path"),
        Err(ConfigError::InvalidPath)
    ));
}

#[test]
fn path_bytes_are_borrowed_for_the_call() {
    use std::os::unix::ffi::OsStrExt;
    let bytes = b"config-\xff.yaml";
    let path = std::ffi::OsStr::from_bytes(bytes).to_os_string();
    let builder = ContextConfigBuilder::from_yaml(&path).unwrap();
    drop(path);
    assert_eq!(backend::last_path(), bytes);
    builder.build().unwrap();
}
