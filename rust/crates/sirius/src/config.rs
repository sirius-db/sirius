//! Immutable configuration construction without GPU access.

use std::path::Path;

use sirius_sys::c_api;
use sirius_sys::handles::{ConfigBuilderHandle, ConfigHandle, Error};

/// A failure while loading settings or building a [`ContextConfig`].
///
/// Match variants to decide how to recover. Use [`Display`](std::fmt::Display)
/// to show a diagnostic even when the native message is empty.
///
/// ```no_run
/// use sirius::{ConfigError, ContextConfigBuilder};
///
/// let builder = match ContextConfigBuilder::from_yaml("sirius.yaml") {
///     Ok(builder) => builder,
///     Err(ConfigError::Io(_)) => ContextConfigBuilder::new()?,
///     Err(error) => return Err(error),
/// };
/// let config = builder.build()?;
/// # Ok::<(), ConfigError>(())
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum ConfigError {
    /// The native library reported allocation failure.
    AllocationFailure,
    /// The file could not be opened or read. Carries the native I/O diagnostic.
    Io(String),
    /// The file could not be parsed as YAML. Carries the parser diagnostic.
    MalformedYaml(String),
    /// The YAML parsed successfully, but its settings are unknown, invalid, or conflicting.
    InvalidConfiguration(String),
    /// The library returned a status outside the recognized configuration errors.
    Unexpected {
        /// Numeric status returned by the Sirius C API.
        status: u32,
        /// Native diagnostic text, or an empty string when unavailable.
        message: String,
    },
    /// The path contains a NUL byte or cannot be represented on the platform.
    InvalidPath,
}

/// Formats a diagnostic for display, including its numeric status for unexpected errors.
///
/// ```
/// use sirius::ConfigError;
///
/// let error = ConfigError::Io(String::new());
/// assert_eq!(error.to_string(), "configuration I/O error");
/// ```
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

fn configuration_error(error: Error) -> ConfigError {
    match error.status {
        c_api::SIRIUS_ALLOCATION_FAILURE => ConfigError::AllocationFailure,
        c_api::SIRIUS_CONFIGURATION_IO => ConfigError::Io(error.message),
        c_api::SIRIUS_MALFORMED_YAML => ConfigError::MalformedYaml(error.message),
        c_api::SIRIUS_INVALID_CONFIGURATION => ConfigError::InvalidConfiguration(error.message),
        status => ConfigError::Unexpected {
            status,
            message: error.message,
        },
    }
}

/// An immutable configuration whose settings have been validated without accessing GPUs.
///
/// Owns its native snapshot independently of its builder or source file. Hardware
/// availability and capacity are checked when the engine is constructed.
/// You can share it between threads and drop it on any thread.
///
/// ```
/// use sirius::{ConfigError, ContextConfig, ContextConfigBuilder};
///
/// let config: ContextConfig = ContextConfigBuilder::new()?.build()?;
/// let worker_config = config.clone();
/// drop(config); // The clone still owns the settings.
/// # Ok::<(), ConfigError>(())
/// ```
pub struct ContextConfig {
    inner: ConfigHandle,
}

impl Clone for ContextConfig {
    /// Acquires another reference to these settings without copying or allocating.
    ///
    /// ```
    /// use sirius::{ConfigError, ContextConfigBuilder};
    ///
    /// let config = ContextConfigBuilder::new()?.build()?;
    /// let copy = config.clone();
    /// # Ok::<(), ConfigError>(())
    /// ```
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// Construct configurations from built-in defaults or a YAML file.
///
/// Parsing and validation are performed through the public C ABI. Loading and
/// building do not initialize CUDA, discover hardware, or allocate engine resources.
/// You can share a builder across threads and call [`build`](Self::build)
/// concurrently. Each configuration owns its settings independently of the builder.
///
/// ```no_run
/// use sirius::{ConfigError, ContextConfigBuilder};
///
/// let settings = ContextConfigBuilder::from_yaml("sirius.yaml")?;
/// let config = settings.build()?;
/// # Ok::<(), ConfigError>(())
/// ```
pub struct ContextConfigBuilder {
    inner: ConfigBuilderHandle,
}

impl ContextConfigBuilder {
    /// Starts a builder with Sirius's built-in defaults.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::AllocationFailure`] if Sirius cannot allocate the
    /// builder, or [`ConfigError::Unexpected`] for an unrecognized status.
    ///
    /// # Panics
    ///
    /// Panics if the linked library reports success with a null builder handle.
    ///
    /// # Examples
    ///
    /// ```
    /// use sirius::{ConfigError, ContextConfigBuilder};
    ///
    /// let config = ContextConfigBuilder::new()?.build()?;
    /// # Ok::<(), ConfigError>(())
    /// ```
    pub fn new() -> Result<Self, ConfigError> {
        ConfigBuilderHandle::new()
            .map(|inner| Self { inner })
            .map_err(configuration_error)
    }

    /// Read and validate a YAML file, retaining its settings for later builds.
    ///
    /// Subsequent edits or deletion of the file do not affect this builder.
    /// Uses the [Sirius YAML schema](https://github.com/sirius-db/sirius/blob/main/docs/super-sirius/configuration.md).
    /// Relative paths resolve against the current working directory. On Unix the
    /// path may contain non-UTF-8 bytes.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::InvalidPath`] for an unrepresentable path (such as
    /// one containing a NUL byte), [`ConfigError::Io`] for a read failure,
    /// [`ConfigError::MalformedYaml`] for a syntax error, or
    /// [`ConfigError::InvalidConfiguration`] for invalid settings. Native
    /// allocation failures and unexpected statuses also surface as [`ConfigError`].
    ///
    /// # Panics
    ///
    /// Panics if the linked library reports success with a null builder handle.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use sirius::{ConfigError, ContextConfigBuilder};
    ///
    /// let config = ContextConfigBuilder::from_yaml("sirius.yaml")?.build()?;
    /// # Ok::<(), ConfigError>(())
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
        ConfigBuilderHandle::from_yaml(bytes)
            .map(|inner| Self { inner })
            .map_err(|error| {
                if error.status == c_api::SIRIUS_INVALID_ARGUMENT {
                    ConfigError::InvalidPath
                } else {
                    configuration_error(error)
                }
            })
    }

    /// Produces an immutable snapshot using only the builder's retained settings.
    /// You can call this method repeatedly on the same builder.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::AllocationFailure`] if Sirius cannot allocate the
    /// snapshot, or [`ConfigError::Unexpected`] for an unrecognized status.
    ///
    /// # Panics
    ///
    /// Panics if the linked library reports success with a null configuration handle.
    ///
    /// # Examples
    ///
    /// ```
    /// use sirius::{ConfigError, ContextConfigBuilder};
    ///
    /// let builder = ContextConfigBuilder::new()?;
    /// let config = builder.build()?;
    /// # Ok::<(), ConfigError>(())
    /// ```
    pub fn build(&self) -> Result<ContextConfig, ConfigError> {
        self.inner
            .build()
            .map(|inner| ContextConfig { inner })
            .map_err(configuration_error)
    }
}

impl Clone for ContextConfigBuilder {
    /// Acquires an independently owned reference to the same settings without allocating.
    ///
    /// ```
    /// use sirius::{ConfigError, ContextConfigBuilder};
    ///
    /// let builder = ContextConfigBuilder::new()?;
    /// let copy = builder.clone();
    /// drop(builder);
    /// let config = copy.build()?;
    /// # Ok::<(), ConfigError>(())
    /// ```
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
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

// Keep the public Drop implementations. Each owning field performs the native release.
impl Drop for ContextConfig {
    fn drop(&mut self) {}
}

impl Drop for ContextConfigBuilder {
    fn drop(&mut self) {}
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
            configuration_error(Error {
                status: c_api::SIRIUS_ALLOCATION_FAILURE,
                message: String::new()
            }),
            ConfigError::AllocationFailure
        ));
        assert!(
            matches!(configuration_error(Error { status: 999, message: String::new() }),
            ConfigError::Unexpected { status: 999, message } if message.is_empty())
        );
    }
}
