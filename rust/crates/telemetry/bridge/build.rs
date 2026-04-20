use quent_codegen::CxxOptions;

fn main() {
    let builder = instrumentation_model::SiriusModel::build("Sirius");

    let options = CxxOptions {
        crate_name: "telemetry-bridge".into(),
        instrumentation_crate: "instrumentation_model".into(),
        ..Default::default()
    };

    let files = quent_codegen::emit_cxx(&builder, &options);
    let bridge_files = quent_codegen::write_bridge_files(&files, &options);

    cxx_build::bridges(bridge_files)
        .std("c++20")
        .compile("telemetry_bridge");

    quent_codegen::copy_cxx_headers();
}
