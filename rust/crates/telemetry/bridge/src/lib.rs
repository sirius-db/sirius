#[allow(unused, clippy::all)]
mod bridge {
    include!(concat!(env!("OUT_DIR"), "/bridge_mod.rs"));
}

// Link the static NVTX initializer for the C++ trampoline.
extern crate nvtx_injection;
