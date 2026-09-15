#[allow(unused, clippy::all)]
mod bridge {
    include!(concat!(env!("OUT_DIR"), "/bridge_mod.rs"));
}

// Keep Quent's static NVTX injection object in this Rust archive. The final
// native link supplies the public trampoline and retains this archive whole.
#[used]
static NVTX_INJECTION_LINK_ANCHOR: extern "C" fn() = nvtx_injection_link_anchor;

extern "C" fn nvtx_injection_link_anchor() {
    // SAFETY: Quent rejects a missing export-table callback before using it.
    let _ = unsafe { nvtx_injection::InitializeInjectionNvtx2(None) };
}
