// Exercises only the public cxx bridge. The snapshot test requires one GPU; CI's
// cargo test --no-run checks that the bridge and the selected Sirius artifact link.
use cxx::{SharedPtr, let_cxx_string};
use sirius_sys::{
    ObjectStoreConfigArgs, drop_object_store_prefix, make_context_from_config_with_object_store,
    make_object_store_config, object_store_config_for_uri,
};

#[test]
fn object_store_debug_redacts_credentials() {
    let args = ObjectStoreConfigArgs {
        endpoint: "https://debug-endpoint.invalid:9443".into(),
        region: "debug-region".into(),
        access_key: "AKIA_SENTINEL_7f3a".into(),
        secret_key: "SECRET_SENTINEL_9b42".into(),
        session_token: "TOKEN_SENTINEL_c681".into(),
        transport: "rdma".into(),
        signing_mode: "header".into(),
        ca_bundle_path: "/debug-ca-bundle.pem".into(),
        uri_prefix: "s3://debug-bucket/allowed/".into(),
    };
    for rendered in [format!("{args:?}"), format!("{args:#?}")] {
        for credential in [&args.access_key, &args.secret_key, &args.session_token] {
            assert!(!rendered.contains(credential.as_str()));
        }
        assert_eq!(rendered.matches("<redacted>").count(), 3);
        for value in [
            &args.endpoint,
            &args.region,
            &args.transport,
            &args.signing_mode,
            &args.ca_bundle_path,
            &args.uri_prefix,
        ] {
            assert!(rendered.contains(value.as_str()));
        }
    }
}

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

    let scoped = make_object_store_config(&ObjectStoreConfigArgs {
        endpoint: "https://192.0.2.1:9".into(),
        region: "scoped-region".into(),
        access_key: "AKIA_SENTINEL_7f3a".into(),
        secret_key: "SECRET_SENTINEL_9b42".into(),
        session_token: "TOKEN_SENTINEL_c681".into(),
        uri_prefix: "s3://bucket/allowed/".into(),
        ..Default::default()
    });
    context
        .put_object_store(scoped.clone())
        .expect("publish a scope");
    let null_error = context
        .put_object_store(SharedPtr::null())
        .expect_err("put requires a snapshot");
    assert!(null_error.what().starts_with("object_store:"));
    assert_eq!(context.object_store_config().as_ptr(), scoped.as_ptr());

    let_cxx_string!(uri = "s3://bucket/allowed/f");
    let retained_scope =
        object_store_config_for_uri(&context, &uri).expect("look up the published scope");
    assert_eq!(retained_scope.as_ptr(), scoped.as_ptr());
    let_cxx_string!(bad = "s3://bucket/allowed/%2f");
    let lookup_error = match object_store_config_for_uri(&context, &bad) {
        Err(error) => error,
        Ok(_) => panic!("lookup must reject an invalid selected scoped key"),
    };
    assert!(lookup_error.what().starts_with("object_store_scope:"));

    let_cxx_string!(malformed = "s3://bucket/allowed");
    let drop_error = drop_object_store_prefix(&context, &malformed)
        .expect_err("malformed drop preserves the scope");
    assert!(
        drop_error
            .what()
            .starts_with("ObjectStoreConfig.uri_prefix:")
    );
    assert_eq!(context.object_store_config().as_ptr(), scoped.as_ptr());
    let_cxx_string!(prefix = "s3://bucket/allowed/");
    drop_object_store_prefix(&context, &prefix).expect("drop the published scope");
    assert!(context.object_store_config().is_null());
    let scoped_pointer = scoped.as_ptr();
    drop(scoped);
    drop(context);
    assert_eq!(retained_scope.as_ptr(), scoped_pointer);
    let derived = retained_scope.with_credentials(&access_key, &secret_key, &empty);
    assert!(!derived.is_null());
    assert_ne!(derived.as_ptr(), retained_scope.as_ptr());
    assert_eq!(retained.as_ptr(), snapshot.as_ptr());
    assert!(
        !retained
            .with_credentials(&access_key, &secret_key, &empty)
            .is_null()
    );
}
