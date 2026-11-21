//! Immutable configuration construction without GPU access.

use std::path::Path;

use crate::diagnostic::Diagnostic;
use sirius_sys::c_api;
use std::ptr::NonNull;

/// A failure while constructing a configuration.
#[derive(Debug)]
#[non_exhaustive]
pub enum ConfigError {
    /// The native library reported allocation failure.
    AllocationFailure,
    /// A configuration file could not be opened or read.
    Io(String),
    /// The input could not be parsed as YAML.
    MalformedYaml(String),
    /// Settings are unknown, invalid, or conflicting.
    InvalidConfiguration(String),
    /// An unexpected native status was returned; diagnostics may be empty.
    Unexpected { status: u32, message: String },
    /// The path contains a NUL byte or cannot be represented on the platform.
    InvalidPath,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AllocationFailure => f.write_str("allocation failed"),
            Self::Io(message) if message.is_empty() => f.write_str("configuration I/O error"),
            Self::MalformedYaml(message) if message.is_empty() => f.write_str("malformed YAML"),
            Self::InvalidConfiguration(message) if message.is_empty() => {
                f.write_str("invalid configuration")
            }
            Self::Io(message)
            | Self::MalformedYaml(message)
            | Self::InvalidConfiguration(message) => f.write_str(message),
            Self::Unexpected { status, message } if message.is_empty() => {
                write!(f, "Sirius status {status}")
            }
            Self::Unexpected { status, message } => write!(f, "Sirius status {status}: {message}"),
            Self::InvalidPath if cfg!(unix) => {
                f.write_str("configuration path contains a NUL byte")
            }
            Self::InvalidPath => f.write_str("configuration path cannot be represented"),
        }
    }
}

impl std::error::Error for ConfigError {}

fn configuration_error(status: u32, diagnostic: Diagnostic) -> ConfigError {
    match status {
        c_api::SIRIUS_ALLOCATION_FAILURE => ConfigError::AllocationFailure,
        c_api::SIRIUS_CONFIGURATION_IO => ConfigError::Io(diagnostic.message()),
        c_api::SIRIUS_MALFORMED_YAML => ConfigError::MalformedYaml(diagnostic.message()),
        c_api::SIRIUS_INVALID_CONFIGURATION => {
            ConfigError::InvalidConfiguration(diagnostic.message())
        }
        _ => ConfigError::Unexpected {
            status,
            message: diagnostic.message(),
        },
    }
}

/// An immutable configuration whose settings have been validated without accessing GPUs.
///
/// Owns its native snapshot independently of its builder or source file. Hardware
/// availability and capacity are checked when the engine is constructed.
/// This type is `Send` and `Sync`: it may be moved or shared between threads.
pub struct ContextConfig {
    inner: NonNull<c_api::SiriusContextConfig>,
}

// SAFETY: the C contract permits transferring owning references, including final release.
unsafe impl Send for ContextConfig {}
// SAFETY: settings are immutable; concurrent retain is allowed by the C contract.
// Atomic reference counts and Rust borrows keep the handle live during use.
unsafe impl Sync for ContextConfig {}

impl Clone for ContextConfig {
    /// Share the immutable settings through a new owned handle.
    ///
    /// ```no_run
    /// # use sirius::{ConfigError, ContextConfigBuilder};
    /// # fn example() -> Result<(), ConfigError> {
    /// let config = ContextConfigBuilder::new()?.build()?;
    /// let copy = config.clone();
    /// # Ok(())
    /// # }
    /// ```
    fn clone(&self) -> Self {
        // SAFETY: self keeps the immutable handle live; the new owner releases the added reference.
        unsafe { c_api::sirius_context_config_retain(self.inner.as_ptr()) };
        Self { inner: self.inner }
    }
}

/// Construct configurations from built-in defaults or a YAML file.
///
/// Parsing and validation are performed through the public C ABI. Loading and
/// building do not initialize CUDA, discover hardware, or allocate engine resources.
/// This type is `Send` and `Sync`: it may be moved or shared between threads.
/// Building and cloning through shared references may run concurrently.
pub struct ContextConfigBuilder {
    inner: NonNull<c_api::SiriusContextConfigBuilder>,
}

// SAFETY: the C contract permits transferring owning references, including final release.
unsafe impl Send for ContextConfigBuilder {}
// SAFETY: settings are immutable; shared operations retain/build concurrently under the
// C contract. Atomic reference counts and Rust borrows keep each handle live during use.
unsafe impl Sync for ContextConfigBuilder {}

impl ContextConfigBuilder {
    /// Start with built-in defaults. Native allocation failures are returned as
    /// [`ConfigError::AllocationFailure`].
    ///
    /// ```no_run
    /// # use sirius::{ConfigError, ContextConfigBuilder};
    /// # fn example() -> Result<(), ConfigError> {
    /// let config = ContextConfigBuilder::new()?.build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new() -> Result<Self, ConfigError> {
        let mut value = std::ptr::null_mut();
        let mut diagnostic = Diagnostic(std::ptr::null_mut());
        // SAFETY: both output slots are writable and empty.
        let status =
            unsafe { c_api::sirius_context_config_builder_create(&mut value, &mut diagnostic.0) };
        if status != c_api::SIRIUS_SUCCESS {
            return Err(configuration_error(status, diagnostic));
        }
        Ok(Self {
            inner: NonNull::new(value).expect("successful builder creation"),
        })
    }

    /// Read and validate a YAML file, retaining its settings for later builds.
    ///
    /// Subsequent edits or deletion of the file do not affect this builder.
    /// Uses the [Sirius YAML schema](https://github.com/sirius-db/sirius/blob/main/docs/super-sirius/configuration.md).
    /// Unix path bytes are preserved, including non-UTF-8 names. NUL bytes are rejected.
    ///
    /// ```no_run
    /// # use sirius::{ConfigError, ContextConfigBuilder};
    /// # fn example() -> Result<(), ConfigError> {
    /// let config = ContextConfigBuilder::from_yaml("sirius.yaml")?.build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_yaml(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        #[cfg(unix)]
        let bytes = {
            use std::os::unix::ffi::OsStrExt;
            path.as_os_str().as_bytes()
        };
        #[cfg(not(unix))]
        let bytes = path.to_str().ok_or(ConfigError::InvalidPath)?.as_bytes();
        let mut value = std::ptr::null_mut();
        let mut diagnostic = Diagnostic(std::ptr::null_mut());
        // SAFETY: the path slice remains live, and output slots are writable and empty.
        let status = unsafe {
            c_api::sirius_context_config_builder_from_yaml(
                bytes.as_ptr().cast(),
                bytes.len(),
                &mut value,
                &mut diagnostic.0,
            )
        };
        if status == c_api::SIRIUS_INVALID_ARGUMENT {
            return Err(ConfigError::InvalidPath);
        }
        if status != c_api::SIRIUS_SUCCESS {
            return Err(configuration_error(status, diagnostic));
        }
        Ok(Self {
            inner: NonNull::new(value).expect("successful YAML loading"),
        })
    }

    /// Produce an immutable configuration without rereading the file or accessing GPUs.
    ///
    /// ```no_run
    /// # use sirius::{ConfigError, ContextConfigBuilder};
    /// # fn example() -> Result<(), ConfigError> {
    /// let builder = ContextConfigBuilder::from_yaml("sirius.yaml")?;
    /// let config = builder.build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(&self) -> Result<ContextConfig, ConfigError> {
        let mut value = std::ptr::null_mut();
        let mut diagnostic = Diagnostic(std::ptr::null_mut());
        // SAFETY: self keeps the builder live; output slots are writable and empty.
        let status = unsafe {
            c_api::sirius_context_config_builder_build(
                self.inner.as_ptr(),
                &mut value,
                &mut diagnostic.0,
            )
        };
        if status != c_api::SIRIUS_SUCCESS {
            return Err(configuration_error(status, diagnostic));
        }
        Ok(ContextConfig {
            inner: NonNull::new(value).expect("successful configuration build"),
        })
    }
}

impl Clone for ContextConfigBuilder {
    /// Share the loaded settings through an independently owned builder handle.
    ///
    /// ```no_run
    /// # use sirius::{ConfigError, ContextConfigBuilder};
    /// # fn example() -> Result<(), ConfigError> {
    /// let builder = ContextConfigBuilder::from_yaml("sirius.yaml")?;
    /// let copy = builder.clone();
    /// drop(builder);
    /// let config = copy.build()?;
    /// # Ok(())
    /// # }
    /// ```
    fn clone(&self) -> Self {
        // SAFETY: self keeps the immutable handle live; the new owner releases the added reference.
        unsafe { c_api::sirius_context_config_builder_retain(self.inner.as_ptr()) };
        Self { inner: self.inner }
    }
}

impl std::fmt::Debug for ContextConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextConfig").finish_non_exhaustive()
    }
}

impl std::fmt::Debug for ContextConfigBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextConfigBuilder")
            .finish_non_exhaustive()
    }
}

impl Drop for ContextConfig {
    fn drop(&mut self) {
        // SAFETY: this owner holds one live reference, with no outstanding Rust borrows.
        unsafe { c_api::sirius_context_config_release(self.inner.as_ptr()) }
    }
}
impl Drop for ContextConfigBuilder {
    fn drop(&mut self) {
        // SAFETY: this owner holds one live reference, with no outstanding Rust borrows.
        unsafe { c_api::sirius_context_config_builder_release(self.inner.as_ptr()) }
    }
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;

    #[test]
    fn configuration_handles_can_move_and_be_shared_between_threads() {
        let builder = ContextConfigBuilder::new().unwrap();
        let config = builder.build().unwrap();
        let owned_builder = builder.clone();
        let owned_config = config.clone();
        std::thread::scope(|scope| {
            let moved = scope.spawn(move || {
                drop(owned_config.clone());
                drop(owned_config);
                owned_builder.build().unwrap()
            });
            let borrowed = scope.spawn(|| {
                drop(config.clone());
                builder.clone().build().unwrap()
            });
            drop(moved.join().unwrap());
            drop(borrowed.join().unwrap());
        });
        builder.build().unwrap();
        let barrier = std::sync::Barrier::new(2);
        let owners = [(builder.clone(), config.clone()), (builder, config)];
        std::thread::scope(|scope| {
            for (builder, config) in owners {
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    drop(builder);
                    drop(config);
                });
            }
        });
    }

    #[test]
    fn defaults_and_copies_own_their_handles() {
        assert_eq!(c_api::sirius_abi_version(), c_api::SIRIUS_ABI_VERSION);
        let builder = ContextConfigBuilder::new().unwrap();
        let copy = builder.clone();
        drop(builder);
        let config = copy.build().unwrap();
        let config_copy = config.clone();
        drop(copy);
        drop(config);
        drop(config_copy);
    }

    #[test]
    fn yaml_snapshot_outlives_its_file() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), "sirius: {}\n").unwrap();
        let builder = ContextConfigBuilder::from_yaml(file.path()).unwrap();
        drop(file);
        let config = builder.build().unwrap();
        drop(builder);
        drop(config);
    }

    #[test]
    fn errors_keep_their_categories() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.yaml");
        assert!(matches!(
            ContextConfigBuilder::from_yaml(&path),
            Err(ConfigError::Io(_))
        ));
        std::fs::write(&path, "sirius: [").unwrap();
        assert!(matches!(
            ContextConfigBuilder::from_yaml(&path),
            Err(ConfigError::MalformedYaml(_))
        ));
        std::fs::write(&path, "sirius:\n  unknown_setting: true\n").unwrap();
        assert!(matches!(
            ContextConfigBuilder::from_yaml(&path),
            Err(ConfigError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            ContextConfigBuilder::from_yaml("bad\0path"),
            Err(ConfigError::InvalidPath)
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn preserves_non_utf8_path_bytes() {
        use std::os::unix::ffi::OsStrExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir
            .path()
            .join(std::ffi::OsStr::from_bytes(b"config-\xff.yaml"));
        std::fs::write(&path, "sirius: {}\n").unwrap();
        ContextConfigBuilder::from_yaml(&path)
            .unwrap()
            .build()
            .unwrap();
    }

    #[test]
    fn missing_diagnostics_and_unknown_statuses_are_supported() {
        assert!(matches!(
            configuration_error(
                c_api::SIRIUS_ALLOCATION_FAILURE,
                Diagnostic(std::ptr::null_mut())
            ),
            ConfigError::AllocationFailure
        ));
        assert!(
            matches!(configuration_error(999, Diagnostic(std::ptr::null_mut())),
            ConfigError::Unexpected { status: 999, message } if message.is_empty())
        );
    }
}
