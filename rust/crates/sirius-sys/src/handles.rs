//! Owning Rust handles for the public Sirius C API.
//!
//! Cloning an immutable handle retains its native reference. Dropping the last
//! reference releases the object. Calls borrow their inputs for their duration.
//!
//! ```no_run
//! use sirius_sys::handles::{ConfigBuilderHandle, Error};
//!
//! let builder = ConfigBuilderHandle::new()?;
//! let config = builder.build()?;
//! drop(builder); // The snapshot owns its settings.
//! # Ok::<(), Error>(())
//! ```

use crate::c_api;
use std::ptr::{self, NonNull};

/// Native status and optional diagnostic from a failed C API call.
#[derive(Debug)]
pub struct Error {
    /// Status returned by the C API.
    pub status: u32,
    /// UTF-8 diagnostic, with invalid bytes replaced. Empty if unavailable or if copying it fails to allocate.
    pub message: String,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sirius status {}", self.status)?;
        if !self.message.is_empty() {
            write!(f, ": {}", self.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {}

struct Diagnostic(*mut c_api::SiriusError);

impl Diagnostic {
    fn message(&self) -> String {
        // SAFETY: the C API returns a non-null byte buffer even for a null
        // diagnostic. This owner keeps the bytes alive throughout the copy.
        let bytes = unsafe {
            let ptr = c_api::sirius_error_message(self.0);
            let len = c_api::sirius_error_message_size(self.0);
            std::slice::from_raw_parts(ptr.cast(), len)
        };
        let mut message = String::new();
        for chunk in bytes.utf8_chunks() {
            let replacement = if chunk.invalid().is_empty() {
                ""
            } else {
                "�"
            };
            if message
                .try_reserve(chunk.valid().len() + replacement.len())
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
        // SAFETY: the exclusive diagnostic slot owns this reference or is null.
        unsafe { c_api::sirius_error_destroy(self.0) }
    }
}

fn checked(call: impl FnOnce(*mut *mut c_api::SiriusError) -> u32) -> Result<(), Error> {
    let mut diagnostic = Diagnostic(ptr::null_mut());
    let status = call(&mut diagnostic.0);
    if status == c_api::SIRIUS_SUCCESS {
        Ok(())
    } else {
        Err(Error {
            status,
            message: diagnostic.message(),
        })
    }
}

fn factory<T>(
    call: impl FnOnce(*mut *mut T, *mut *mut c_api::SiriusError) -> u32,
) -> Result<NonNull<T>, Error> {
    let mut value = ptr::null_mut();
    checked(|error| call(&mut value, error))?;
    Ok(NonNull::new(value).expect("Sirius factory returned success with a null handle"))
}

// Use only for immutable C objects whose contract permits concurrent use,
// retain/release, and final release on a different thread.
macro_rules! immutable_handle {
    ($(#[$attr:meta])* $name:ident, $raw:ty, $retain:path, $release:path) => {
        $(#[$attr])*
        pub struct $name(NonNull<$raw>);

        impl Clone for $name {
            fn clone(&self) -> Self {
                // SAFETY: self owns a live reference throughout the retain.
                unsafe { $retain(self.0.as_ptr()) };
                Self(self.0)
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                // SAFETY: each owner releases exactly one native reference.
                unsafe { $release(self.0.as_ptr()) }
            }
        }

        // SAFETY: ownership and final release may move to another thread.
        unsafe impl Send for $name {}
        // SAFETY: shared methods borrow a live, immutable, thread-safe handle.
        unsafe impl Sync for $name {}
    };
}

immutable_handle!(
    /// An immutable configuration snapshot. Each clone can outlive the builder.
    ConfigHandle,
    c_api::SiriusContextConfig,
    c_api::sirius_context_config_retain,
    c_api::sirius_context_config_release
);

immutable_handle!(
    /// Immutable configuration settings that can be built concurrently.
    ConfigBuilderHandle,
    c_api::SiriusContextConfigBuilder,
    c_api::sirius_context_config_builder_retain,
    c_api::sirius_context_config_builder_release
);

impl ConfigBuilderHandle {
    /// Creates a builder with built-in defaults.
    ///
    /// # Errors
    ///
    /// Returns the native status and diagnostic if creating the builder fails.
    ///
    /// # Panics
    ///
    /// Panics if the linked library reports success with a null builder handle.
    pub fn new() -> Result<Self, Error> {
        // SAFETY: factory supplies exclusive, empty output slots.
        factory(|out, error| unsafe { c_api::sirius_context_config_builder_create(out, error) })
            .map(Self)
    }

    /// Loads the settings from the given path bytes. The path is borrowed only
    /// during this call.
    ///
    /// # Errors
    ///
    /// Returns the native status and diagnostic for invalid paths, unreadable files,
    /// malformed YAML, invalid settings, or native allocation failures.
    ///
    /// # Panics
    ///
    /// Panics if the linked library reports success with a null builder handle.
    pub fn from_yaml(path: &[u8]) -> Result<Self, Error> {
        // Native string constructors need a valid range even for an empty path.
        let data = if path.is_empty() {
            c"".as_ptr()
        } else {
            path.as_ptr().cast()
        };
        // SAFETY: data is readable for path.len() and factory owns both output slots.
        factory(|out, error| unsafe {
            c_api::sirius_context_config_builder_from_yaml(data, path.len(), out, error)
        })
        .map(Self)
    }

    /// Builds an independently owned configuration.
    ///
    /// # Errors
    ///
    /// Returns the native status and diagnostic if creating the snapshot fails.
    ///
    /// # Panics
    ///
    /// Panics if the linked library reports success with a null configuration handle.
    pub fn build(&self) -> Result<ConfigHandle, Error> {
        // SAFETY: self keeps the builder alive and factory owns both output slots.
        factory(|out, error| unsafe {
            c_api::sirius_context_config_builder_build(self.0.as_ptr(), out, error)
        })
        .map(ConfigHandle)
    }
}

pub mod plan;

#[cfg(all(test, not(miri)))]
mod tests {
    #[test]
    fn failed_call_without_native_diagnostic_preserves_status() {
        let error = super::checked(|_| 999).unwrap_err();
        assert_eq!(error.status, 999);
        assert!(error.message.is_empty());
    }
}
