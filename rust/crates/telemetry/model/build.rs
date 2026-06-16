// Captures this crate's git into QUENT_SOURCE_* so the model.qmi sidecar records
// Sirius as the model source instead of falling back to quent's build info.
fn main() {
    quent_build_info::emit_source();
}
