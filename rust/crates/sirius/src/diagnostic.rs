use sirius_sys::c_api;

/// Own the optional native diagnostic until its bytes have been copied.
pub(crate) struct Diagnostic(pub(crate) *mut c_api::SiriusError);

impl Diagnostic {
    fn bytes(&self) -> &[u8] {
        // SAFETY: the C factory supplied this live diagnostic (or null); it is
        // borrowed only until the copy completes and released by Drop. The C API
        // guarantees a non-null message pointer, even for a null diagnostic.
        unsafe {
            let ptr = c_api::sirius_error_message(self.0);
            let len = c_api::sirius_error_message_size(self.0);
            std::slice::from_raw_parts(ptr.cast(), len)
        }
    }

    pub(crate) fn message(&self) -> String {
        String::from_utf8_lossy(self.bytes()).into_owned()
    }

    /// Copy a possibly non-UTF-8 diagnostic without aborting on allocation failure.
    pub(crate) fn best_effort_message(&self) -> String {
        let mut message = String::new();
        for chunk in self.bytes().utf8_chunks() {
            let replacement = if chunk.invalid().is_empty() {
                ""
            } else {
                "�"
            };
            if message
                .try_reserve(chunk.valid().len().saturating_add(replacement.len()))
                .is_err()
            {
                return String::new();
            }
            message.push_str(chunk.valid());
            message.push_str(replacement);
        }
        message
    }
}
impl Drop for Diagnostic {
    fn drop(&mut self) {
        // SAFETY: this is the unique owner, and no borrowed bytes remain.
        unsafe { c_api::sirius_error_destroy(self.0) }
    }
}
