use std::{env, path::PathBuf};

use heck::{ToShoutySnakeCase, ToUpperCamelCase};
use proc_macro2::TokenStream;
use prost_build::{Service, ServiceGenerator};
use quote::{format_ident, quote};

const BRPC_PROTOS: &[&str] = &[
    "src/brpc/options.proto",
    "src/brpc/policy/baidu_rpc_meta.proto",
    "src/brpc/streaming_rpc_meta.proto",
];
const STARROCKS_PROTOS: &[&str] = &[
    "binlog.proto",
    "data.proto",
    "descriptors.proto",
    "internal_service.proto",
    "lake_types.proto",
    "olap_common.proto",
    "olap_file.proto",
    "status.proto",
    "tablet_schema.proto",
    "types.proto",
];

/// Generates StarRocks protobuf bindings, BRPC metadata bindings, and the BRPC service facade.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let brpc_dir = manifest_dir.join("brpc");
    let proto_dir = manifest_dir.join("starrocks/gensrc/proto");

    println!("cargo:rerun-if-changed={}", brpc_dir.display());
    println!("cargo:rerun-if-changed={}", proto_dir.display());

    require_exchange_proto_patch(&manifest_dir, &proto_dir)?;

    let mut protos = BRPC_PROTOS
        .iter()
        .map(|proto| {
            let path = brpc_dir.join(proto);
            println!("cargo:rerun-if-changed={}", path.display());
            path
        })
        .collect::<Vec<_>>();
    protos.extend(STARROCKS_PROTOS.iter().map(|proto| {
        let path = proto_dir.join(proto);
        println!("cargo:rerun-if-changed={}", path.display());
        path
    }));

    let mut config = prost_build::Config::new();
    config.disable_comments(["."]);
    config.service_generator(Box::new(BrpcServiceGenerator));
    config.compile_protos(&protos, &[brpc_dir.join("src"), proto_dir])?;

    Ok(())
}

/// Fails the build, with the remedy in the message, when the `starrocks` submodule is missing the
/// Sirius exchange RPCs.
///
/// Those RPCs (`PExchangeNixlMd`, `PStagingLeaseRequest`, `PTransmitPackedParams`, ...) are
/// Sirius-only extensions that live in `patches/nixl-exchange-proto.patch`, and nothing applies
/// the patch automatically: `git submodule update` resets the working tree to upstream. Without
/// the patch `prost` generates a `PInternalService` that simply lacks three methods, and the
/// handlers that implement them (the CN-to-CN exchange transport) fail far from the cause with
/// `unresolved imports crate::proto::starrocks::PExchangeNixlMd` and `method exchange_nixl_md is
/// not a member of trait PInternalService`. Those errors name the symptom and give a fresh clone
/// no way to guess the fix.
///
/// The check lives here, not in the pixi task or the CI step, because build.rs is the one place
/// every build path (plain cargo, pixi, CI) goes through: fail at the earliest point that can
/// name the fix.
fn require_exchange_proto_patch(
    manifest_dir: &std::path::Path,
    proto_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let internal_service = proto_dir.join("internal_service.proto");
    // A missing submodule is a different failure with a different fix, so let prost report it.
    let Ok(contents) = std::fs::read_to_string(&internal_service) else {
        return Ok(());
    };
    // One representative message rather than all of them: the patch is applied whole or not at
    // all, and matching on a single name keeps this from breaking when the patch grows.
    if contents.contains("message PExchangeNixlMd") {
        return Ok(());
    }
    let patch = manifest_dir.join("patches/nixl-exchange-proto.patch");
    let script = manifest_dir.join("scripts/apply-starrocks-patches.sh");
    let submodule = proto_dir
        .parent()
        .and_then(|gensrc| gensrc.parent())
        .unwrap_or(proto_dir);
    // The guidance goes to stderr and the returned error stays one line: cargo renders a build
    // script's error with `Debug`, which would print embedded newlines as literal `\n`.
    eprintln!(
        "\n\
         {} does not define the Sirius exchange RPCs, so the CN cannot be built.\n\
         Apply the patch to the starrocks submodule:\n\
         \n    git -C {} apply {}\n\n\
         or run the idempotent script that applies every patch under patches/:\n\
         \n    bash {}\n\n\
         The patch is not applied automatically and is not part of upstream StarRocks. Re-apply\n\
         it after any `git submodule update` that resets the submodule working tree.\n",
        internal_service.display(),
        submodule.display(),
        patch.display(),
        script.display(),
    );
    Err("the starrocks submodule is missing patches/nixl-exchange-proto.patch (see above)".into())
}

/// Emits an async BRPC/Tower service facade from StarRocks' protobuf service
/// descriptor. This plays the role tonic-build would normally play for gRPC,
/// while keeping the generated API independent of HTTP/2 and gRPC framing.
struct BrpcServiceGenerator;

impl ServiceGenerator for BrpcServiceGenerator {
    /// Generates the StarRocks PInternalService interface and request router.
    fn generate(&mut self, service: Service, output: &mut String) {
        if service.package != "starrocks" || service.proto_name != "PInternalService" {
            return;
        }

        for method in &service.methods {
            // Streaming would need a different transport than the one-request/one-response
            // PRPC framing this generator targets; fail the build loudly rather than emit a
            // facade that silently drops the streaming contract.
            assert!(
                !(method.client_streaming || method.server_streaming),
                "BRPC service generator does not support streaming method {}.{}",
                service.proto_name,
                method.proto_name,
            );
        }

        let service_ident = format_ident!("{}", service.name);
        let router_ident = format_ident!("{}Router", service.name);
        let service_name = service.proto_name.as_str();

        let method_consts = service.methods.iter().map(|method| {
            let const_ident = format_ident!("{}", method.proto_name.to_shouty_snake_case());
            let proto_name = method.proto_name.as_str();
            quote! { pub const #const_ident: &str = #proto_name; }
        });

        let variants = service.methods.iter().map(|method| {
            let variant = format_ident!("{}", method.proto_name.to_upper_camel_case());
            quote! { #variant, }
        });

        let from_proto_arms = service.methods.iter().map(|method| {
            let variant = format_ident!("{}", method.proto_name.to_upper_camel_case());
            let proto_name = method.proto_name.as_str();
            quote! { #proto_name => Some(Self::#variant), }
        });

        let proto_name_arms = service.methods.iter().map(|method| {
            let variant = format_ident!("{}", method.proto_name.to_upper_camel_case());
            let proto_name = method.proto_name.as_str();
            quote! { Self::#variant => #proto_name, }
        });

        let trait_methods = service.methods.iter().map(|method| {
            let method_ident = format_ident!("{}", method.name);
            let const_ident = format_ident!("{}", method.proto_name.to_shouty_snake_case());
            let input_type: syn::Type =
                syn::parse_str(&method.input_type).expect("method input type should parse");
            let output_type: syn::Type =
                syn::parse_str(&method.output_type).expect("method output type should parse");
            // `+ Send` keeps the dispatch future Send so connections can be spawned; an `async fn`
            // impl satisfies this return-position-impl-trait signature. The reply is wrapped in
            // `prpc::Reply` so a handler may attach bytes (e.g. `fetch_data`'s `TResultBatch`).
            quote! {
                fn #method_ident(
                    &self,
                    _request: #input_type,
                    _attachment: Vec<u8>,
                ) -> impl Future<Output = Result<crate::prpc::Reply<#output_type>, crate::prpc::Error>> + Send {
                    async move {
                        Err(crate::prpc::Error::method_not_implemented(methods::#const_ident))
                    }
                }
            }
        });

        let dispatch_arms = service.methods.iter().map(|method| {
            let variant = format_ident!("{}", method.proto_name.to_upper_camel_case());
            let call_ident = format_ident!("call_{}", method.name);
            quote! {
                Method::#variant => Self::#call_ident(inner, request.body, request.attachment).await,
            }
        });

        let call_fns = service.methods.iter().map(|method| {
            let method_ident = format_ident!("{}", method.name);
            let call_ident = format_ident!("call_{}", method.name);
            let const_ident = format_ident!("{}", method.proto_name.to_shouty_snake_case());
            let input_type: syn::Type =
                syn::parse_str(&method.input_type).expect("method input type should parse");
            quote! {
                async fn #call_ident(
                    inner: Arc<T>,
                    request_bytes: Vec<u8>,
                    attachment: Vec<u8>,
                ) -> Result<crate::prpc::Response, crate::prpc::Error> {
                    let request = #input_type::decode(request_bytes.as_slice())
                        .map_err(|err| crate::prpc::Error::invalid_request(methods::#const_ident, err))?;
                    let reply = inner.#method_ident(request, attachment).await?;
                    Ok(crate::prpc::Response::with_attachment(
                        reply.message.encode_to_vec(),
                        reply.attachment,
                    ))
                }
            }
        });

        let tokens: TokenStream = quote! {
            pub mod p_internal_service_brpc {
                use prost::Message as _;
                use std::{future::Future, pin::Pin, sync::Arc, task::{Context, Poll}};
                use super::*;

                pub const SERVICE_NAME: &str = #service_name;

                pub mod methods {
                    #(#method_consts)*
                }

                #[derive(Clone, Copy, Debug, Eq, PartialEq)]
                pub enum Method {
                    #(#variants)*
                }

                impl Method {
                    pub fn from_proto_name(name: &str) -> Option<Self> {
                        match name {
                            #(#from_proto_arms)*
                            _ => None,
                        }
                    }

                    pub const fn proto_name(self) -> &'static str {
                        match self {
                            #(#proto_name_arms)*
                        }
                    }
                }

                /// Generated StarRocks PInternalService interface; unimplemented methods
                /// default to a PRPC "method not implemented" error.
                pub(crate) trait #service_ident: Send + Sync {
                    #(#trait_methods)*
                }

                /// Tower service that decodes PRPC requests and dispatches them to `T`.
                #[derive(Clone, Debug)]
                pub(crate) struct #router_ident<T> {
                    /// Shared concrete service implementation.
                    inner: Arc<T>,
                }

                impl<T> #router_ident<T> {
                    pub(crate) fn new(inner: T) -> Self {
                        Self { inner: Arc::new(inner) }
                    }
                }

                impl<T> tower::Service<crate::prpc::Request> for #router_ident<T>
                where
                    T: #service_ident + 'static,
                {
                    type Response = crate::prpc::Response;
                    type Error = crate::prpc::Error;
                    type Future =
                        Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

                    fn poll_ready(
                        &mut self,
                        _context: &mut Context<'_>,
                    ) -> Poll<Result<(), Self::Error>> {
                        Poll::Ready(Ok(()))
                    }

                    fn call(&mut self, request: crate::prpc::Request) -> Self::Future {
                        let inner = Arc::clone(&self.inner);
                        Box::pin(async move { Self::dispatch(inner, request).await })
                    }
                }

                impl<T> #router_ident<T>
                where
                    T: #service_ident + 'static,
                {
                    async fn dispatch(
                        inner: Arc<T>,
                        request: crate::prpc::Request,
                    ) -> Result<crate::prpc::Response, crate::prpc::Error> {
                        if request.service_name != SERVICE_NAME {
                            return Err(crate::prpc::Error::service_not_found(format!(
                                "service name '{}' not found",
                                request.service_name
                            )));
                        }
                        let Some(method) = Method::from_proto_name(&request.method_name) else {
                            return Err(crate::prpc::Error::method_not_found(format!(
                                "method '{}' not found",
                                request.method_name
                            )));
                        };
                        match method {
                            #(#dispatch_arms)*
                        }
                    }

                    #(#call_fns)*
                }
            }
        };

        let file: syn::File =
            syn::parse2(tokens).expect("generated BRPC service facade should parse");
        output.push_str(&prettyplease::unparse(&file));
    }
}
