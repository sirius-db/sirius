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
///
/// `cpu_affinity` confines the engine's unpinned thread pools to one socket (see
/// [`crate::gpu_affinity`]). It only *decorates* a config that a memory flag already called for:
/// on its own it must not conjure one, because a derived config takes precedence over an
/// operator-supplied `--sirius-config` at the call site.
pub fn derive_sirius_config_yaml(
    gpu_memory_limit: Option<&str>,
    gpu_memory_fraction: Option<f64>,
    host_memory_limit: Option<&str>,
    engine_dir: &Path,
    cpu_affinity: Option<&[u32]>,
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
    // `sirius.memory.host` is the HIGH-LEVEL host config: it runs the reservation-manager
    // configurator, whose `use_host_per_numa()` policy already emits one host space per distinct
    // NUMA node of the configured GPUs and binds its arena there with `numa_alloc_onnode`. With
    // `num_gpus: 1` that is exactly one host space on this CN's own socket — already correct.
    // Do NOT "fix" this by moving to the low-level `sirius.space.host` sequence to set `numa_id`
    // by hand: declaring any explicit space disables the configurator wholesale, taking
    // per-host capacity and the portable-allocation heuristic with it.
    if let Some(capacity) = host_memory_limit {
        yaml.push_str("    host:\n");
        yaml.push_str(&format!(
            "      capacity_bytes: \"{}\"\n",
            yaml_escape(capacity)
        ));
    }
    // Scan datasource backend. The engine defaults to `true` (the uring sirius_datasource);
    // setting SIRIUS_CN_USE_SIRIUS_DATASOURCE=false selects the kvikio/cudf datasource instead.
    // That path is rejected for multi-GPU topologies (sirius_scan_manager.cpp), but each CN pins
    // exactly one device via CUDA_VISIBLE_DEVICES and this YAML declares num_gpus: 1, so the
    // guard does not apply per-CN. Measured on Q06/SF100 standalone: uring ~4.9 s vs cudf ~0.23 s.
    let datasource = std::env::var_os("SIRIUS_CN_USE_SIRIUS_DATASOURCE").map(|v| {
        let v = v.to_string_lossy().to_ascii_lowercase();
        !matches!(v.as_str(), "false" | "0" | "no" | "off")
    });
    if datasource.is_some() || cpu_affinity.is_some() {
        yaml.push_str("  executor:\n");
        yaml.push_str("    scan_manager:\n");
        if let Some(use_sirius) = datasource {
            yaml.push_str(&format!("      use_sirius_datasource: {use_sirius}\n"));
        }
        // The three pools the engine never pins itself. The GPU pipeline pool is deliberately
        // absent: `task_scheduler.cpp` overwrites its `cpu_affinity_list` from the discovered
        // GPU topology, so a YAML value there would be silently discarded.
        if let Some(cpus) = cpu_affinity {
            let cpus = cpu_affinity_sequence(cpus);
            yaml.push_str(&format!("      cpu_affinity: {cpus}\n"));
            yaml.push_str("    task_creator:\n");
            yaml.push_str(&format!("      cpu_affinity: {cpus}\n"));
            yaml.push_str("    downgrade:\n");
            yaml.push_str(&format!("      cpu_affinity: {cpus}\n"));
        }
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

/// Renders CPU ordinals as a YAML flow sequence. The engine's reader requires a sequence for
/// `cpu_affinity` (`yaml_reader.hpp` rejects a scalar), and it has no range syntax, so a whole
/// 72-core socket is emitted core by core.
fn cpu_affinity_sequence(cpus: &[u32]) -> String {
    let items: Vec<String> = cpus.iter().map(u32::to_string).collect();
    format!("[{}]", items.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `--gpu-memory-limit` value must land verbatim in `usage_limit_bytes`; the C++
    /// `parse_bytes` is the authoritative parser.
    #[test]
    fn gpu_limit_passes_through_verbatim() {
        let yaml =
            derive_sirius_config_yaml(Some("8GiB"), None, None, Path::new(".cn1"), None).unwrap();
        assert!(yaml.contains("usage_limit_bytes: \"8GiB\"\n"), "{yaml}");
        assert!(!yaml.contains("usage_limit_fraction"), "{yaml}");
    }

    /// The fraction variant emits `usage_limit_fraction` instead of bytes.
    #[test]
    fn gpu_fraction_variant() {
        let yaml =
            derive_sirius_config_yaml(None, Some(0.5), None, Path::new(".cn1"), None).unwrap();
        assert!(yaml.contains("usage_limit_fraction: 0.5\n"), "{yaml}");
        assert!(!yaml.contains("usage_limit_bytes"), "{yaml}");
    }

    /// The host mapping appears exactly when `--host-memory-limit` is set.
    #[test]
    fn host_key_presence_and_absence() {
        let with_host =
            derive_sirius_config_yaml(Some("8GiB"), None, Some("12GiB"), Path::new(".cn1"), None)
                .unwrap();
        assert!(with_host.contains("    host:\n"), "{with_host}");
        assert!(
            with_host.contains("capacity_bytes: \"12GiB\"\n"),
            "{with_host}"
        );

        let without_host =
            derive_sirius_config_yaml(Some("8GiB"), None, None, Path::new(".cn1"), None).unwrap();
        assert!(!without_host.contains("host:"), "{without_host}");
        assert!(!without_host.contains("capacity_bytes"), "{without_host}");
    }

    /// Every GPU carve-out pins `reservation_limit_fraction` to 1.0 so the limit is the budget.
    #[test]
    fn reservation_limit_fraction_is_always_one() {
        for yaml in [
            derive_sirius_config_yaml(Some("8GiB"), None, None, Path::new(".cn1"), None).unwrap(),
            derive_sirius_config_yaml(None, Some(0.25), None, Path::new(".cn1"), None).unwrap(),
        ] {
            assert!(yaml.contains("reservation_limit_fraction: 1.0\n"), "{yaml}");
        }
    }

    /// Telemetry lands under the engine directory.
    #[test]
    fn telemetry_directory_is_under_engine_dir() {
        let yaml =
            derive_sirius_config_yaml(Some("8GiB"), None, None, Path::new(".cn2"), None).unwrap();
        assert!(
            yaml.contains("output_directory: \".cn2/telemetry\"\n"),
            "{yaml}"
        );
    }

    /// `--gpu-device`/`--engine-dir` alone need no config file at all.
    #[test]
    fn no_yaml_when_no_memory_flag_is_set() {
        assert_eq!(
            derive_sirius_config_yaml(None, None, None, Path::new(".cn1"), None),
            None
        );
    }

    /// A host-only carve-out must not emit a gpu mapping (its reservation override without a
    /// usage limit would claim the whole device).
    #[test]
    fn host_only_omits_gpu_mapping() {
        let yaml =
            derive_sirius_config_yaml(None, None, Some("12GiB"), Path::new(".cn1"), None).unwrap();
        assert!(!yaml.contains("gpu:"), "{yaml}");
        assert!(yaml.contains("capacity_bytes: \"12GiB\"\n"), "{yaml}");
    }

    /// Full-document snapshot of the cluster2 shape, pinned against the C++ schema
    /// (`sirius_config.cpp` rejects unknown keys).
    #[test]
    fn full_document_snapshot() {
        let yaml =
            derive_sirius_config_yaml(Some("8GiB"), None, Some("12GiB"), Path::new(".cn1"), None)
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

    /// Full-document snapshot with a socket pinning, pinned against the C++ schema: the keys are
    /// `sirius.executor.{scan_manager,task_creator,downgrade}.cpu_affinity`, each a *sequence*
    /// (`sirius_config.cpp` reads them into `thread_pool_config::cpu_affinity_list`, and
    /// `yaml_reader.hpp` throws "expected a sequence" for anything else).
    #[test]
    fn full_document_snapshot_with_cpu_affinity() {
        let yaml = derive_sirius_config_yaml(
            Some("8GiB"),
            None,
            Some("12GiB"),
            Path::new(".cn1"),
            Some(&[0, 1, 2, 3]),
        )
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
                "  executor:\n",
                "    scan_manager:\n",
                "      cpu_affinity: [0, 1, 2, 3]\n",
                "    task_creator:\n",
                "      cpu_affinity: [0, 1, 2, 3]\n",
                "    downgrade:\n",
                "      cpu_affinity: [0, 1, 2, 3]\n",
                "  telemetry:\n",
                "    output_directory: \".cn1/telemetry\"\n",
            )
        );
    }

    /// No affinity means no `executor:` block at all — the pre-existing shape, byte for byte.
    #[test]
    fn absent_affinity_emits_no_executor_block() {
        let yaml =
            derive_sirius_config_yaml(Some("8GiB"), None, Some("12GiB"), Path::new(".cn1"), None)
                .unwrap();
        assert!(!yaml.contains("executor:"), "{yaml}");
        assert!(!yaml.contains("cpu_affinity"), "{yaml}");
    }

    /// The GPU pipeline pool must never be emitted: `task_scheduler.cpp` overwrites its affinity
    /// from the discovered GPU topology, so a YAML value there would be dead config.
    #[test]
    fn pipeline_pool_is_not_pinned_from_yaml() {
        let yaml = derive_sirius_config_yaml(
            Some("8GiB"),
            None,
            None,
            Path::new(".cn1"),
            Some(&[0, 1, 2, 3]),
        )
        .unwrap();
        assert!(!yaml.contains("pipeline:"), "{yaml}");
    }

    /// A whole 72-core socket renders as one flow sequence; the engine's reader has no range
    /// syntax, so every core is listed.
    #[test]
    fn full_socket_renders_every_core() {
        let cpus: Vec<u32> = (0..=71).collect();
        let yaml =
            derive_sirius_config_yaml(Some("8GiB"), None, None, Path::new(".cn1"), Some(&cpus))
                .unwrap();
        assert!(yaml.contains("      cpu_affinity: [0, 1, 2,"), "{yaml}");
        assert!(yaml.contains(", 70, 71]\n"), "{yaml}");
        assert_eq!(yaml.matches("cpu_affinity: ").count(), 3, "{yaml}");
    }

    /// A CPU affinity alone must not conjure a config file: the caller treats `Some(yaml)` as
    /// "use the derived config", which would silently shadow an operator's `--sirius-config`.
    #[test]
    fn cpu_affinity_alone_does_not_conjure_a_config() {
        assert_eq!(
            derive_sirius_config_yaml(None, None, None, Path::new(".cn1"), Some(&[0, 1])),
            None
        );
    }
}
