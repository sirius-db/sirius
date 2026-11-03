//! Immutable configuration construction without GPU access.

use std::path::Path;

use crate::diagnostic::Diagnostic;
use sirius_sys::c_api;
use std::ptr::NonNull;

/// A failure while constructing or copying a configuration.
#[derive(Debug)]
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
    /// The platform path cannot be represented by the native bridge.
    InvalidPath,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AllocationFailure => f.write_str("allocation failed"),
            Self::Io(message)
            | Self::MalformedYaml(message)
            | Self::InvalidConfiguration(message) => f.write_str(message),
            Self::Unexpected { status, message } => write!(f, "Sirius status {status}: {message}"),
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
/// No thread-safety guarantees are exposed yet: this type is neither `Send` nor `Sync`.
pub struct ContextConfig {
    inner: NonNull<c_api::SiriusContextConfig>,
}

impl ContextConfig {
    /// Share the immutable settings through a new owned handle.
    ///
    /// ```no_run
    /// # use sirius::{ConfigError, ContextConfigBuilder};
    /// # fn example() -> Result<(), ConfigError> {
    /// let config = ContextConfigBuilder::new()?.build()?;
    /// let copy = config.try_clone()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_clone(&self) -> Result<Self, ConfigError> {
        // SAFETY: self keeps the immutable handle live; the new owner releases the added reference.
        unsafe { c_api::sirius_context_config_retain(self.inner.as_ptr()) };
        Ok(Self { inner: self.inner })
    }
}

/// Construct configurations from built-in defaults or a YAML file.
///
/// Parsing and validation are performed through the public C ABI. Loading and
/// building do not initialize CUDA, discover hardware, or allocate engine resources.
/// This type is neither `Send` nor `Sync`.
pub struct ContextConfigBuilder {
    inner: NonNull<c_api::SiriusContextConfigBuilder>,
}

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

    /// Share the loaded settings through an independently owned builder handle.
    ///
    /// ```no_run
    /// # use sirius::{ConfigError, ContextConfigBuilder};
    /// # fn example() -> Result<(), ConfigError> {
    /// let builder = ContextConfigBuilder::from_yaml("sirius.yaml")?;
    /// let copy = builder.try_clone()?;
    /// drop(builder);
    /// let config = copy.build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_clone(&self) -> Result<Self, ConfigError> {
        // SAFETY: self keeps the immutable handle live; the new owner releases the added reference.
        unsafe { c_api::sirius_context_config_builder_retain(self.inner.as_ptr()) };
        Ok(Self { inner: self.inner })
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
    fn defaults_and_copies_own_their_handles() {
        assert_eq!(c_api::sirius_abi_version(), c_api::SIRIUS_ABI_VERSION);
        let builder = ContextConfigBuilder::new().unwrap();
        let copy = builder.try_clone().unwrap();
        drop(builder);
        let config = copy.build().unwrap();
        let config_copy = config.try_clone().unwrap();
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
