use sirius_sys::c_api;

/// Own the optional native diagnostic until its bytes have been copied.
pub(crate) struct Diagnostic(pub(crate) *mut c_api::SiriusError);

impl Diagnostic {
    pub(crate) fn message(&self) -> String {
        // SAFETY: the C factory supplied this live diagnostic (or null); it is
        // borrowed only until the copy completes and released by Drop. The C API
        // guarantees a non-null message pointer, even for a null diagnostic.
        unsafe {
            let ptr = c_api::sirius_error_message(self.0);
            let len = c_api::sirius_error_message_size(self.0);
            String::from_utf8_lossy(std::slice::from_raw_parts(ptr.cast(), len)).into_owned()
        }
    }
}
impl Drop for Diagnostic {
    fn drop(&mut self) {
        // SAFETY: this is the unique owner, and no borrowed bytes remain.
        unsafe { c_api::sirius_error_destroy(self.0) }
    }
}
