//! Resolved Sirius engine bring-up settings and the derived-config YAML they may carry.
//!
//! The CLI exposes coarse GPU knobs (`--gpu-memory-limit`, `--gpu-memory-fraction`,
//! `--host-memory-limit`) so an operator can carve out a slice of a shared device without
//! writing a full Sirius YAML config. This module turns those knobs into the exact YAML the
//! C++ config loader (`sirius_config.cpp`) understands. Byte values are passed through
//! verbatim: the authoritative parser is the C++ `parse_bytes`, so nothing is converted here.

use std::path::{Path, PathBuf};

/// Engine bring-up settings after CLI resolution: which config file to load (an operator-supplied
/// one or a derived carve-out config), where engine artifacts live, and which CUDA device to pin.
#[derive(Clone, Debug)]
pub struct EngineSettings {
    /// Sirius YAML config path (built-in defaults when `None`).
    pub config: Option<PathBuf>,
    /// Directory for engine artifacts: derived config, logs, telemetry.
    pub engine_dir: PathBuf,
    /// CUDA device ordinal to export as `CUDA_VISIBLE_DEVICES` before engine bring-up.
    pub gpu_device: Option<u32>,
}

/// Renders the derived Sirius config for the given memory carve-out flags, or `None` when no
/// memory flag is set (only `--gpu-device`/`--engine-dir` do not need a config file).
///
/// `reservation_limit_fraction` is pinned to 1.0 so the carve-out itself is the whole budget:
/// the engine may reserve up to 100% of the configured limit, not 100% of the device — that is
/// what lets two CNs coexist on one GPU.
pub fn derive_sirius_config_yaml(
    gpu_memory_limit: Option<&str>,
    gpu_memory_fraction: Option<f64>,
    host_memory_limit: Option<&str>,
    engine_dir: &Path,
) -> Option<String> {
    assert!(
        gpu_memory_limit.is_none() || gpu_memory_fraction.is_none(),
        "gpu_memory_limit and gpu_memory_fraction are mutually exclusive (clap enforces this)"
    );
    if gpu_memory_limit.is_none() && gpu_memory_fraction.is_none() && host_memory_limit.is_none() {
        return None;
    }

    let mut yaml = String::from("sirius:\n  topology:\n    num_gpus: 1\n  memory:\n");
    // The gpu mapping is only emitted when a GPU limit is requested: emitting the reservation
    // override alone would let the engine reserve the whole device it was not asked to cap.
    if let Some(limit) = gpu_memory_limit {
        yaml.push_str("    gpu:\n");
        yaml.push_str(&format!(
            "      usage_limit_bytes: \"{}\"\n",
            yaml_escape(limit)
        ));
        yaml.push_str("      reservation_limit_fraction: 1.0\n");
    } else if let Some(fraction) = gpu_memory_fraction {
        yaml.push_str("    gpu:\n");
        yaml.push_str(&format!("      usage_limit_fraction: {fraction:?}\n"));
        yaml.push_str("      reservation_limit_fraction: 1.0\n");
    }
    if let Some(capacity) = host_memory_limit {
        yaml.push_str("    host:\n");
        yaml.push_str(&format!(
            "      capacity_bytes: \"{}\"\n",
            yaml_escape(capacity)
        ));
    }
    yaml.push_str("  telemetry:\n");
    yaml.push_str(&format!(
        "    output_directory: \"{}\"\n",
        yaml_escape(&engine_dir.join("telemetry").display().to_string())
    ));
    Some(yaml)
}

/// Escapes a value for a double-quoted YAML scalar.
fn yaml_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `--gpu-memory-limit` value must land verbatim in `usage_limit_bytes`; the C++
    /// `parse_bytes` is the authoritative parser.
    #[test]
    fn gpu_limit_passes_through_verbatim() {
        let yaml =
            derive_sirius_config_yaml(Some("8GiB"), None, None, Path::new(".cn1")).unwrap();
        assert!(yaml.contains("usage_limit_bytes: \"8GiB\"\n"), "{yaml}");
        assert!(!yaml.contains("usage_limit_fraction"), "{yaml}");
    }

    /// The fraction variant emits `usage_limit_fraction` instead of bytes.
    #[test]
    fn gpu_fraction_variant() {
        let yaml = derive_sirius_config_yaml(None, Some(0.5), None, Path::new(".cn1")).unwrap();
        assert!(yaml.contains("usage_limit_fraction: 0.5\n"), "{yaml}");
        assert!(!yaml.contains("usage_limit_bytes"), "{yaml}");
    }

    /// The host mapping appears exactly when `--host-memory-limit` is set.
    #[test]
    fn host_key_presence_and_absence() {
        let with_host =
            derive_sirius_config_yaml(Some("8GiB"), None, Some("12GiB"), Path::new(".cn1"))
                .unwrap();
        assert!(with_host.contains("    host:\n"), "{with_host}");
        assert!(
            with_host.contains("capacity_bytes: \"12GiB\"\n"),
            "{with_host}"
        );

        let without_host =
            derive_sirius_config_yaml(Some("8GiB"), None, None, Path::new(".cn1")).unwrap();
        assert!(!without_host.contains("host:"), "{without_host}");
        assert!(!without_host.contains("capacity_bytes"), "{without_host}");
    }

    /// Every GPU carve-out pins `reservation_limit_fraction` to 1.0 so the limit is the budget.
    #[test]
    fn reservation_limit_fraction_is_always_one() {
        for yaml in [
            derive_sirius_config_yaml(Some("8GiB"), None, None, Path::new(".cn1")).unwrap(),
            derive_sirius_config_yaml(None, Some(0.25), None, Path::new(".cn1")).unwrap(),
        ] {
            assert!(
                yaml.contains("reservation_limit_fraction: 1.0\n"),
                "{yaml}"
            );
        }
    }

    /// Telemetry lands under the engine directory.
    #[test]
    fn telemetry_directory_is_under_engine_dir() {
        let yaml =
            derive_sirius_config_yaml(Some("8GiB"), None, None, Path::new(".cn2")).unwrap();
        assert!(
            yaml.contains("output_directory: \".cn2/telemetry\"\n"),
            "{yaml}"
        );
    }

    /// `--gpu-device`/`--engine-dir` alone need no config file at all.
    #[test]
    fn no_yaml_when_no_memory_flag_is_set() {
        assert_eq!(
            derive_sirius_config_yaml(None, None, None, Path::new(".cn1")),
            None
        );
    }

    /// A host-only carve-out must not emit a gpu mapping (its reservation override without a
    /// usage limit would claim the whole device).
    #[test]
    fn host_only_omits_gpu_mapping() {
        let yaml =
            derive_sirius_config_yaml(None, None, Some("12GiB"), Path::new(".cn1")).unwrap();
        assert!(!yaml.contains("gpu:"), "{yaml}");
        assert!(yaml.contains("capacity_bytes: \"12GiB\"\n"), "{yaml}");
    }

    /// Full-document snapshot of the cluster2 shape, pinned against the C++ schema
    /// (`sirius_config.cpp` rejects unknown keys).
    #[test]
    fn full_document_snapshot() {
        let yaml =
            derive_sirius_config_yaml(Some("8GiB"), None, Some("12GiB"), Path::new(".cn1"))
                .unwrap();
        assert_eq!(
            yaml,
            concat!(
                "sirius:\n",
                "  topology:\n",
                "    num_gpus: 1\n",
                "  memory:\n",
                "    gpu:\n",
                "      usage_limit_bytes: \"8GiB\"\n",
                "      reservation_limit_fraction: 1.0\n",
                "    host:\n",
                "      capacity_bytes: \"12GiB\"\n",
                "  telemetry:\n",
                "    output_directory: \".cn1/telemetry\"\n",
            )
        );
    }
}
