//! Exercise safe wrappers against Rust-owned C ABI stand-ins under Miri.
#![cfg(miri)]
use sirius_sys::miri_test_backend as backend;

use sirius::{ConfigError, ContextConfigBuilder};
use sirius_sys::c_api::*;

#[test]
fn independent_handles_and_repeated_builds() {
    let builder = ContextConfigBuilder::new().unwrap();
    let copy = builder.clone();
    drop(builder);
    let first = copy.build().unwrap();
    let second = copy.build().unwrap();
    drop(copy);
    let retained = first.clone();
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
        let error = ContextConfigBuilder::from_yaml("config.yaml").unwrap_err();
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
    for (status, expected) in [
        (SIRIUS_CONFIGURATION_IO, "configuration I/O error"),
        (SIRIUS_MALFORMED_YAML, "malformed YAML"),
        (SIRIUS_INVALID_CONFIGURATION, "invalid configuration"),
    ] {
        backend::fail_next(status, None);
        let error = ContextConfigBuilder::from_yaml("config.yaml").unwrap_err();
        assert_eq!(error.to_string(), expected);
    }
    backend::fail_next(999, None);
    let error = ContextConfigBuilder::new().unwrap_err();
    assert_eq!(error.to_string(), "Sirius status 999");
    assert!(matches!(error,
        ConfigError::Unexpected { status: 999, message } if message.is_empty()));
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

#[test]
fn configuration_handles_can_move_and_be_shared_between_threads() {
    let builder = ContextConfigBuilder::new().unwrap();
    let config = builder.build().unwrap();
    let owned_builder = builder.clone();
    let owned_config = config.clone();
    std::thread::scope(|scope| {
        let moved = scope.spawn(move || {
            drop(owned_config.clone());
            drop(owned_config);
            owned_builder.build().unwrap()
        });
        let borrowed = scope.spawn(|| {
            drop(config.clone());
            builder.clone().build().unwrap()
        });
        drop(moved.join().unwrap());
        drop(borrowed.join().unwrap());
    });
    builder.build().unwrap();
    let barrier = std::sync::Barrier::new(2);
    let owners = [(builder.clone(), config.clone()), (builder, config)];
    std::thread::scope(|scope| {
        for (builder, config) in owners {
            let barrier = &barrier;
            scope.spawn(move || {
                barrier.wait();
                drop(builder);
                drop(config);
            });
        }
    });
}

#[test]
fn expression_final_release_on_another_thread() {
    let original = sirius::Expression::literal_int64(42).unwrap();
    let shared = original.clone();
    drop(original);
    std::thread::spawn(move || drop(shared)).join().unwrap();
}

#[test]
fn field_view_borrows_from_retained_schema() {
    use sirius::{DataType, Field, Schema};
    let name = String::from("field");
    let schema = Schema::new(&[Field {
        name: &name,
        data_type: DataType::Int64,
        nullable: true,
    }])
    .unwrap();
    drop(name);
    let retained = schema.clone();
    let view = retained.field(0).unwrap();
    drop(schema);
    assert_eq!(view.name, "field");
}

#[test]
fn allocation_error_preserves_available_diagnostic() {
    backend::fail_next(SIRIUS_ALLOCATION_FAILURE, Some(b"\xffout of memory\0"));
    let result = sirius::Expression::literal_int64(10);
    assert!(
        matches!(result, Err(sirius::PlanError::AllocationFailure(message)) if message == "\u{fffd}out of memory\0")
    );
}
