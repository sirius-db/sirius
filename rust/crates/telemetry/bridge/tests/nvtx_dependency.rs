#[test]
fn nvtx_event_is_linked() {
    // Generated observers need the NVTX event crate as a direct dependency.
    assert!(std::mem::size_of::<quent_nvtx_events::NvtxEvent>() > 0);
}
