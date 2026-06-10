use std::{env, path::PathBuf};

use prost_build::{Method, Service, ServiceGenerator};

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let proto_dir = manifest_dir.join("starrocks/gensrc/proto");

    println!("cargo:rerun-if-changed={}", proto_dir.display());

    let protos = STARROCKS_PROTOS
        .iter()
        .map(|proto| {
            let path = proto_dir.join(proto);
            println!("cargo:rerun-if-changed={}", path.display());
            path
        })
        .collect::<Vec<_>>();

    let mut config = prost_build::Config::new();
    config.disable_comments(["."]);
    config.service_generator(Box::new(BrpcServiceGenerator));
    config.compile_protos(&protos, &[proto_dir])?;

    Ok(())
}

struct BrpcServiceGenerator;

impl ServiceGenerator for BrpcServiceGenerator {
    fn generate(&mut self, service: Service, output: &mut String) {
        if service.package != "starrocks" || service.proto_name != "PInternalService" {
            return;
        }

        output.push_str("pub mod p_internal_service_brpc {\n");
        output.push_str("    use prost::Message as _;\n");
        output.push_str("    use std::{future::{Ready, ready}, task::{Context, Poll}};\n");
        output.push_str("    use super::*;\n");
        output.push_str(&format!(
            "    pub const SERVICE_NAME: &str = {:?};\n",
            service.proto_name
        ));
        output.push_str("    pub mod methods {\n");
        for method in &service.methods {
            output.push_str(&format!(
                "        pub const {}: &str = {:?};\n",
                const_name(&method.proto_name),
                method.proto_name
            ));
        }
        output.push_str("    }\n");

        output.push_str("    #[derive(Clone, Copy, Debug, Eq, PartialEq)]\n");
        output.push_str("    pub enum Method {\n");
        for method in &service.methods {
            output.push_str(&format!("        {},\n", method_variant(method)));
        }
        output.push_str("    }\n");

        output.push_str("    impl Method {\n");
        output.push_str("        pub fn from_proto_name(name: &str) -> Option<Self> {\n");
        output.push_str("            match name {\n");
        for method in &service.methods {
            output.push_str(&format!(
                "                {:?} => Some(Self::{}),\n",
                method.proto_name,
                method_variant(method)
            ));
        }
        output.push_str("                _ => None,\n");
        output.push_str("            }\n");
        output.push_str("        }\n");

        output.push_str("        pub const fn proto_name(self) -> &'static str {\n");
        output.push_str("            match self {\n");
        for method in &service.methods {
            output.push_str(&format!(
                "                Self::{} => {:?},\n",
                method_variant(method),
                method.proto_name
            ));
        }
        output.push_str("            }\n");
        output.push_str("        }\n");
        output.push_str("    }\n");

        output.push_str(&format!("    pub(crate) trait {} {{\n", service.name));
        for method in &service.methods {
            if method.client_streaming || method.server_streaming {
                panic!(
                    "BRPC service generator does not support streaming method {}.{}",
                    service.proto_name, method.proto_name
                );
            }
            output.push_str(&format!(
                "        fn {}(&self, _request: {}, _attachment: &[u8]) -> Result<{}, crate::prpc::Error> {{\n",
                method.name, method.input_type, method.output_type
            ));
            output.push_str(&format!(
                "            Err(crate::prpc::Error::method_not_implemented(methods::{}))\n",
                const_name(&method.proto_name)
            ));
            output.push_str("        }\n");
        }
        output.push_str("    }\n");

        output.push_str("    #[derive(Clone, Debug)]\n");
        output.push_str(&format!(
            "    pub(crate) struct {}Router<T> {{\n",
            service.name
        ));
        output.push_str("        inner: T,\n");
        output.push_str("    }\n");

        output.push_str(&format!("    impl<T> {}Router<T> {{\n", service.name));
        output.push_str("        pub(crate) fn new(inner: T) -> Self {\n");
        output.push_str("            Self { inner }\n");
        output.push_str("        }\n");
        output.push_str("    }\n");

        output.push_str(&format!(
            "    impl<T> tower::Service<crate::prpc::Request> for {}Router<T>\n",
            service.name
        ));
        output.push_str(&format!("    where\n        T: {},\n", service.name));
        output.push_str("    {\n");
        output.push_str("        type Response = crate::prpc::Response;\n");
        output.push_str("        type Error = crate::prpc::Error;\n");
        output.push_str("        type Future = Ready<Result<Self::Response, Self::Error>>;\n\n");
        output.push_str(
            "        fn poll_ready(&mut self, _context: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {\n",
        );
        output.push_str("            Poll::Ready(Ok(()))\n");
        output.push_str("        }\n\n");
        output.push_str(
            "        fn call(&mut self, request: crate::prpc::Request) -> Self::Future {\n",
        );
        output.push_str("            ready(self.dispatch(request))\n");
        output.push_str("        }\n");
        output.push_str("    }\n");

        output.push_str(&format!("    impl<T> {}Router<T>\n", service.name));
        output.push_str(&format!("    where\n        T: {},\n", service.name));
        output.push_str("    {\n");
        output.push_str("        fn dispatch(&self, request: crate::prpc::Request) -> Result<crate::prpc::Response, crate::prpc::Error> {\n");
        output.push_str("            if request.service_name != SERVICE_NAME {\n");
        output.push_str(
            "                return Err(crate::prpc::Error::service_not_found(format!(\n",
        );
        output.push_str("                    \"service name '{}' not found\",\n");
        output.push_str("                    request.service_name\n");
        output.push_str("                )));\n");
        output.push_str("            }\n\n");
        output.push_str(
            "            let Some(method) = Method::from_proto_name(&request.method_name) else {\n",
        );
        output
            .push_str("                return Err(crate::prpc::Error::method_not_found(format!(\n");
        output.push_str("                    \"method '{}' not found\",\n");
        output.push_str("                    request.method_name\n");
        output.push_str("                )));\n");
        output.push_str("            };\n\n");
        output.push_str("            match method {\n");
        for method in &service.methods {
            output.push_str(&format!(
                "                Method::{} => self.call_{}(&request.body, &request.attachment),\n",
                method_variant(method),
                method.name
            ));
        }
        output.push_str("            }\n");
        output.push_str("        }\n\n");

        for method in &service.methods {
            output.push_str(&format!(
                "        fn call_{}(&self, request_bytes: &[u8], attachment: &[u8]) -> Result<crate::prpc::Response, crate::prpc::Error> {{\n",
                method.name
            ));
            output.push_str(&format!(
                "            let request = {}::decode(request_bytes)\n",
                method.input_type
            ));
            output.push_str(&format!(
                "                .map_err(|err| crate::prpc::Error::invalid_request(methods::{}, err))?;\n",
                const_name(&method.proto_name)
            ));
            output.push_str(&format!(
                "            let response = self.inner.{}(request, attachment)?;\n",
                method.name
            ));
            output
                .push_str("            Ok(crate::prpc::Response::new(response.encode_to_vec()))\n");
            output.push_str("        }\n");
        }
        output.push_str("    }\n");
        output.push_str("}\n");
    }
}

fn const_name(proto_name: &str) -> String {
    proto_name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn method_variant(method: &Method) -> String {
    upper_camel(&method.proto_name)
}

fn upper_camel(value: &str) -> String {
    value
        .split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            let Some(first) = chars.next() else {
                return String::new();
            };
            let mut result = String::new();
            result.push(first.to_ascii_uppercase());
            result.extend(chars);
            result
        })
        .collect()
}
