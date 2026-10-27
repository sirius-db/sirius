// Exercises only the public cxx bridge. Running this test requires one GPU; CI's
// cargo test --no-run checks that the bridge and the selected Sirius artifact link.
use cxx::{SharedPtr, let_cxx_string};
use sirius_sys::{
    ObjectStoreConfigArgs, make_context_from_config_with_object_store, make_object_store_config,
};

#[test]
fn object_store_snapshot_round_trip() {
    let args = ObjectStoreConfigArgs {
        endpoint: "https://192.0.2.1:9".into(),
        region: "fixture-region".into(),
        access_key: "AKIA_SENTINEL_7f3a".into(),
        secret_key: "SECRET_SENTINEL_9b42".into(),
        session_token: "TOKEN_SENTINEL_c681".into(),
        ..Default::default()
    };
    let snapshot = make_object_store_config(&args);
    assert!(!snapshot.is_null());
    drop(args);
    let config =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../test/cpp/scan/memory.yaml");
    let_cxx_string!(config_path = config.to_string_lossy().as_ref());
    let context = make_context_from_config_with_object_store(&config_path, snapshot.clone())
        .expect("construct a context with the public snapshot");
    assert_eq!(context.object_store_config().as_ptr(), snapshot.as_ptr());
    let retained = context.object_store_config();

    let_cxx_string!(access_key = "AKIA_REPLACEMENT_71b2");
    let_cxx_string!(secret_key = "SECRET_REPLACEMENT_83c4");
    let_cxx_string!(token = "TOKEN_REPLACEMENT_95d6");
    let replacement = retained.with_credentials(&access_key, &secret_key, &token);
    assert!(!replacement.is_null());
    assert_ne!(replacement.as_ptr(), retained.as_ptr());
    context
        .set_object_store(replacement.clone())
        .expect("publish replacement credentials");
    assert_eq!(context.object_store_config().as_ptr(), replacement.as_ptr());

    // The bridge also preserves the distinction between pure derivation and set-time
    // validation. Complete field-value equality is checked by the C++ public API test.
    let_cxx_string!(empty = "");
    let invalid = retained.with_credentials(&empty, &secret_key, &token);
    assert!(!invalid.is_null());
    let error = context
        .set_object_store(invalid)
        .expect_err("reject an empty access key without replacing the current snapshot");
    assert!(error.what().starts_with("ObjectStoreConfig.access_key:"));
    for sentinel in [
        "AKIA_SENTINEL_7f3a",
        "SECRET_SENTINEL_9b42",
        "TOKEN_SENTINEL_c681",
        "SECRET_REPLACEMENT_83c4",
        "TOKEN_REPLACEMENT_95d6",
    ] {
        assert!(!error.what().contains(sentinel));
    }
    assert_eq!(context.object_store_config().as_ptr(), replacement.as_ptr());

    context
        .set_object_store(SharedPtr::null())
        .expect("clear the current snapshot");
    assert!(context.object_store_config().is_null());
    drop(context);
    assert_eq!(retained.as_ptr(), snapshot.as_ptr());
    assert!(
        !retained
            .with_credentials(&access_key, &secret_key, &empty)
            .is_null()
    );
}
