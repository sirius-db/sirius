#[allow(unused, clippy::all)]
mod bridge {
    include!(concat!(env!("OUT_DIR"), "/bridge_mod.rs"));
}

// Keep the static NVTX initializer in the bridge archive for the C++ trampoline.
extern crate nvtx_injection;
