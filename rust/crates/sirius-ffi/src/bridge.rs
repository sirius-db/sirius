//! Low-level bindings to `include/sirius/ffi.hpp`.

// The `# Safety` docs on the unsafe bridge fns live on the declarations below;
// cxx's macro expansion hides them from clippy's `missing_safety_doc`, so allow
// it for the generated module.
#[allow(clippy::missing_safety_doc)]
#[cxx::bridge(namespace = "sirius::ffi")]
mod ffi {
    /// Wire transport of an object-store snapshot (mirrors `sirius::ffi::Transport`).
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    #[repr(u8)]
    enum Transport {
        #[default]
        kAuto,
        kHttp,
        kRdma,
    }
    /// SigV4 placement (mirrors `sirius::ffi::SigningMode`).
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    #[repr(u8)]
    enum SigningMode {
        #[default]
        kPresigned,
        kHeader,
    }

    /// Named fields for building an [`ObjectStoreConfig`] snapshot; see the C++ struct for
    /// the meaning of each field. `Debug` is implemented by hand and redacts the three
    /// credential fields.
    #[derive(Clone, Default)]
    struct ObjectStoreConfigArgs {
        endpoint: String,
        region: String,
        access_key: String,
        secret_key: String,
        session_token: String,
        transport: Transport,
        signing_mode: SigningMode,
        ca_bundle_path: String,
        uri_prefix: String,
    }

    unsafe extern "C++" {
        include!("sirius/ffi.hpp");
        include!("sirius-ffi/src/shim.hpp");

        // The enums are defined by the C++ header; these declarations make CXX check the shared
        // declarations above against them instead of emitting a second definition.
        type Transport;
        type SigningMode;

        /// RAII handle to an initialized Sirius engine context.
        type Context;

        /// Construct an initialized [`Context`] from built-in defaults, owned by
        /// the returned `UniquePtr`.
        fn make_context() -> Result<UniquePtr<Context>>;

        /// Construct an initialized [`Context`] from the YAML config file at
        /// `config_path`, owned by the returned `UniquePtr`. `config_path` binds
        /// to the C++ `const std::string&` parameter.
        fn make_context_from_config(config_path: &CxxString) -> Result<UniquePtr<Context>>;

        /// Execute a serialized Substrait plan on the GPU, writing the results
        /// into the Arrow C Data Interface stream at `out_stream_addr` — the
        /// address (as `usize`) of a caller-owned `ArrowArrayStream` the caller
        /// releases per the Arrow ABI. `plan` binds to the C++ `const
        /// std::string&` and carries the protobuf-encoded `substrait::Plan`
        /// bytes. Bound as fallible: translation or execution failure surfaces as
        /// `Err(cxx::Exception)`.
        ///
        /// # Safety
        /// `out_stream_addr` must be the address of a valid, writable
        /// `ArrowArrayStream` that outlives this call; C++ writes the result
        /// stream through it. The safe [`crate::SiriusContext`] wrapper upholds this.
        unsafe fn execute_substrait(
            self: Pin<&mut Context>,
            plan: &CxxString,
            out_stream_addr: usize,
        ) -> Result<()>;

        /// Immutable object-store (S3 API) settings of a [`Context`]; built once from
        /// [`ObjectStoreConfigArgs`] and shared by pointer. Field semantics are documented on
        /// the C++ struct.
        type ObjectStoreConfig;

        /// Build a snapshot from named fields. No validation and no I/O here; the snapshot is
        /// checked when it is given to a context.
        fn make_object_store_config(args: &ObjectStoreConfigArgs) -> SharedPtr<ObjectStoreConfig>;

        /// A new snapshot with these credentials and every other field copied from `self`,
        /// which is left unchanged. Use an empty token for long-lived keys.
        fn with_credentials(
            self: &ObjectStoreConfig,
            new_access_key: &CxxString,
            new_secret_key: &CxxString,
            new_session_token: &CxxString,
        ) -> SharedPtr<ObjectStoreConfig>;

        /// Publish `cfg` for the scope named by its `uri_prefix`, replacing the snapshot
        /// published for exactly that prefix (the empty prefix is the default scope). Other
        /// scopes are untouched. Fails (leaving the set unchanged) when a required field is
        /// empty, the endpoint is malformed, the CA bundle is not a readable regular file, the
        /// prefix is malformed, or the configured IO backend does not honor an option. No network
        /// access; executions that already captured a set keep it. Thread-safe; a null pointer
        /// is rejected.
        fn put_object_store(self: &Context, cfg: SharedPtr<ObjectStoreConfig>) -> Result<()>;

        /// Remove the snapshot published for exactly `uri_prefix`; removing an absent prefix
        /// is a no-op. Fails on a malformed prefix without changing the published set.
        fn drop_object_store_prefix(ctx: &Context, uri_prefix: &CxxString) -> Result<()>;

        /// The snapshot that would serve `uri`: the longest matching published prefix, then
        /// the default scope; null when nothing is published or no scope matches. Whenever at
        /// least one snapshot is published, a key that cannot be matched byte-for-byte (an
        /// empty, `.` or `..` segment, or a raw `%` byte) fails before any scope is selected,
        /// default scope included, as an execution would.
        fn object_store_config_for_uri(
            ctx: &Context,
            uri: &CxxString,
        ) -> Result<SharedPtr<ObjectStoreConfig>>;

        /// Construct an initialized [`Context`] from the YAML config file at `config_path`
        /// (empty = built-in defaults) with `cfg` as its object-store snapshot (null = none).
        /// `cfg` is checked before the engine is brought up; a rejected snapshot leaves no
        /// engine state.
        fn make_context_from_config_with_object_store(
            config_path: &CxxString,
            cfg: SharedPtr<ObjectStoreConfig>,
        ) -> Result<UniquePtr<Context>>;
    }
}

impl std::fmt::Debug for ffi::ObjectStoreConfigArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectStoreConfigArgs")
            .field("endpoint", &self.endpoint)
            .field("region", &self.region)
            .field("access_key", &"<redacted>")
            .field("secret_key", &"<redacted>")
            .field("session_token", &"<redacted>")
            .field("transport", &self.transport)
            .field("signing_mode", &self.signing_mode)
            .field("ca_bundle_path", &self.ca_bundle_path)
            .field("uri_prefix", &self.uri_prefix)
            .finish()
    }
}

pub use ffi::{
    Context, ObjectStoreConfig, ObjectStoreConfigArgs, SigningMode, Transport,
    drop_object_store_prefix, make_context, make_context_from_config,
    make_context_from_config_with_object_store, make_object_store_config,
    object_store_config_for_uri,
};
