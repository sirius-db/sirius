//! GPU → CPU-socket affinity, discovered from sysfs.
//!
//! Each CN pins exactly one GPU, and the engine already hard-binds that CN's pinned host arena
//! to the GPU's NUMA node (`numa_alloc_onnode`, via the `bind_cpu_to_gpu_numa` host policy).
//! What the engine does *not* do is place most of its threads: only the GPU pipeline pool is
//! pinned (`task_scheduler.cpp` copies the GPU's `cpu_cores` into its thread-pool config). The
//! `scan_manager`, `task_creator` and `downgrade` pools are pinned only from YAML
//! (`sirius_config.cpp` `cpu_affinity`), and the CN emitted none — so ~20 threads per CN floated
//! across every CPU on the box while their memory sat on one socket.
//!
//! This module resolves the CPU list to put in that YAML. It reads the same sysfs attributes the
//! engine's own topology discovery uses, so the two agree by construction.

use std::path::{Path, PathBuf};

use tracing::{info, warn};

/// Overrides the derived affinity: a cpulist (`"0-71"`, `"0-3,8"`) to force, or an off switch
/// (`off`/`0`/`false`/`no`/`none`) to emit no affinity at all.
const AFFINITY_ENV: &str = "SIRIUS_CN_CPU_AFFINITY";

const PCI_DEVICES: &str = "/sys/bus/pci/devices";
const NVIDIA_VENDOR: &str = "0x10de";

/// The CPU socket a GPU hangs off: its NUMA node and that node's CPU ordinals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuSocket {
    /// NUMA node of the GPU's PCI device — always a CPU-bearing node, never an HBM domain
    /// (a GPU's `numa_node` attribute names the host bridge's socket, not its own memory).
    pub numa_node: u32,
    /// The node's CPU ordinals, from the device's `local_cpulist`.
    pub cpus: Vec<u32>,
}

/// Resolves the CPU cores this CN's threads should be confined to, or `None` when the answer
/// cannot be established — in which case the caller must emit no affinity and leave the pools
/// free-floating, which is the historical behaviour.
///
/// Order of precedence: the `SIRIUS_CN_CPU_AFFINITY` override, then sysfs discovery from the
/// effective GPU ordinal. A wrong pinning is worse than none, so every step that cannot be
/// resolved with certainty gives up rather than guessing.
pub fn cpu_affinity_for_gpu(gpu_device: Option<u32>) -> Option<Vec<u32>> {
    if let Some(raw) = std::env::var_os(AFFINITY_ENV) {
        let raw = raw.to_string_lossy().trim().to_ascii_lowercase();
        if matches!(raw.as_str(), "off" | "0" | "false" | "no" | "none") {
            info!("{AFFINITY_ENV}={raw}: emitting no CPU affinity for the engine thread pools");
            return None;
        }
        match parse_cpulist(&raw) {
            Some(cpus) if !cpus.is_empty() => {
                info!(
                    cpus = cpus.len(),
                    "{AFFINITY_ENV} override: pinning to '{raw}'"
                );
                return Some(cpus);
            }
            _ => {
                warn!("{AFFINITY_ENV}='{raw}' is not a cpulist or an off switch; ignoring it");
            }
        }
    }

    let ordinal = effective_gpu_ordinal(gpu_device)?;
    match gpu_socket(ordinal) {
        Some(socket) => {
            info!(
                gpu = ordinal,
                numa_node = socket.numa_node,
                cpus = socket.cpus.len(),
                "pinning the engine thread pools to the GPU's socket"
            );
            Some(socket.cpus)
        }
        None => {
            warn!(
                gpu = ordinal,
                "could not resolve the GPU's NUMA socket from {PCI_DEVICES}; \
                 leaving the engine thread pools unpinned"
            );
            None
        }
    }
}

/// The physical GPU ordinal this process will actually run on.
///
/// Mirrors `engine.rs`: `--gpu-device` is exported as `CUDA_VISIBLE_DEVICES` only when that
/// variable is unset, so a pre-exported value wins. A non-numeric entry (`GPU-<uuid>`,
/// `MIG-<uuid>`) is not resolvable to a PCI slot here, so it yields `None`.
fn effective_gpu_ordinal(gpu_device: Option<u32>) -> Option<u32> {
    match std::env::var("CUDA_VISIBLE_DEVICES") {
        Ok(visible) => visible.split(',').next()?.trim().parse::<u32>().ok(),
        Err(_) => gpu_device,
    }
}

/// Looks up the socket of the `ordinal`-th NVIDIA GPU on the system.
pub fn gpu_socket(ordinal: u32) -> Option<GpuSocket> {
    gpu_socket_in(Path::new(PCI_DEVICES), ordinal)
}

fn gpu_socket_in(pci_root: &Path, ordinal: u32) -> Option<GpuSocket> {
    let dir = pci_root.join(nvidia_gpu_bdfs(pci_root).get(ordinal as usize)?);

    // `numa_node` is -1 when the platform exposes no proximity domain for the device.
    let numa_node = u32::try_from(read_attr(&dir, "numa_node")?.parse::<i32>().ok()?).ok()?;
    let cpus = parse_cpulist(&read_attr(&dir, "local_cpulist")?)?;
    // A cpuless proximity domain (on this box: the GPU-HBM nodes) has an empty cpulist. Binding
    // threads or memory there would be catastrophic, so treat it as "unknown" instead.
    if cpus.is_empty() {
        return None;
    }
    Some(GpuSocket { numa_node, cpus })
}

/// The PCI addresses of every NVIDIA display/3D device, ordered the way CUDA enumerates them.
///
/// CUDA orders devices by `CUDA_DEVICE_ORDER`, whose default (`FASTEST_FIRST`) breaks ties by
/// PCI bus id — so on a homogeneous box like this one it is PCI order, same as `PCI_BUS_ID`.
/// Sorting the addresses as strings is sorting them numerically: sysfs always renders them
/// fixed-width as `dddd:bb:dd.f`.
fn nvidia_gpu_bdfs(pci_root: &Path) -> Vec<PathBuf> {
    let mut bdfs: Vec<PathBuf> = std::fs::read_dir(pci_root)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| {
            let dir = entry.path();
            if read_attr(&dir, "vendor").as_deref() != Some(NVIDIA_VENDOR) {
                return false;
            }
            // Class 0x0300xx (VGA) / 0x0302xx (3D controller) are the GPUs; the NVIDIA-vendored
            // host bridges and switches on this platform are 0x0604xx and must not be counted.
            read_attr(&dir, "class")
                .is_some_and(|class| class.starts_with("0x0300") || class.starts_with("0x0302"))
        })
        .map(|entry| PathBuf::from(entry.file_name()))
        .collect();
    bdfs.sort();
    bdfs
}

fn read_attr(dir: &Path, name: &str) -> Option<String> {
    std::fs::read_to_string(dir.join(name))
        .ok()
        .map(|s| s.trim().to_string())
}

/// Parses a Linux cpulist (`"0-71"`, `"0-3,8,12-15"`, `""`) into CPU ordinals.
///
/// Returns `None` on anything malformed rather than a partial list: a truncated affinity mask
/// would silently over-constrain the pools.
fn parse_cpulist(list: &str) -> Option<Vec<u32>> {
    let mut cpus = Vec::new();
    for range in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match range.split_once('-') {
            Some((lo, hi)) => {
                let (lo, hi) = (
                    lo.trim().parse::<u32>().ok()?,
                    hi.trim().parse::<u32>().ok()?,
                );
                if lo > hi {
                    return None;
                }
                cpus.extend(lo..=hi);
            }
            None => cpus.push(range.parse::<u32>().ok()?),
        }
    }
    Some(cpus)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a fake `/sys/bus/pci/devices` tree: `(bdf, vendor, class, numa_node, cpulist)`.
    fn fake_pci_root(devices: &[(&str, &str, &str, &str, &str)]) -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        for (bdf, vendor, class, numa_node, cpulist) in devices {
            let dir = root.path().join(bdf);
            std::fs::create_dir_all(&dir).unwrap();
            for (name, value) in [
                ("vendor", *vendor),
                ("class", *class),
                ("numa_node", *numa_node),
                ("local_cpulist", *cpulist),
            ] {
                std::fs::write(dir.join(name), format!("{value}\n")).unwrap();
            }
        }
        root
    }

    /// The 4x GB200 layout of this box: GPU0/1 on socket 0, GPU2/3 on socket 1, with the
    /// NVIDIA-vendored host bridges interleaved so the class filter is exercised.
    fn gb200_root() -> tempfile::TempDir {
        fake_pci_root(&[
            ("0008:00:00.0", "0x10de", "0x060400", "0", "0-71"),
            ("0008:01:00.0", "0x10de", "0x030200", "0", "0-71"),
            ("0009:00:00.0", "0x10de", "0x060400", "0", "0-71"),
            ("0009:01:00.0", "0x10de", "0x030200", "0", "0-71"),
            ("0018:00:00.0", "0x10de", "0x060400", "1", "72-143"),
            ("0018:01:00.0", "0x10de", "0x030200", "1", "72-143"),
            ("0019:00:00.0", "0x10de", "0x060400", "1", "72-143"),
            ("0019:01:00.0", "0x10de", "0x030200", "1", "72-143"),
        ])
    }

    /// GPU ordinals map to sockets exactly as `topo.md` records them: 0,1 → node 0 (CPUs 0-71),
    /// 2,3 → node 1 (CPUs 72-143). The NVIDIA host bridges must not shift the ordinals.
    #[test]
    fn gb200_gpu_ordinals_map_to_their_socket() {
        let root = gb200_root();
        for ordinal in [0, 1] {
            let socket = gpu_socket_in(root.path(), ordinal).unwrap();
            assert_eq!(socket.numa_node, 0, "gpu {ordinal}");
            assert_eq!(socket.cpus, (0..=71).collect::<Vec<_>>(), "gpu {ordinal}");
        }
        for ordinal in [2, 3] {
            let socket = gpu_socket_in(root.path(), ordinal).unwrap();
            assert_eq!(socket.numa_node, 1, "gpu {ordinal}");
            assert_eq!(socket.cpus, (72..=143).collect::<Vec<_>>(), "gpu {ordinal}");
        }
    }

    /// An ordinal past the last GPU resolves to nothing rather than wrapping onto another GPU.
    #[test]
    fn ordinal_out_of_range_is_unresolved() {
        let root = gb200_root();
        assert_eq!(gpu_socket_in(root.path(), 4), None);
    }

    /// A cpuless proximity domain — the shape of this box's four GPU-HBM NUMA nodes — must never
    /// be turned into an affinity list.
    #[test]
    fn cpuless_domain_is_rejected() {
        let root = fake_pci_root(&[("0008:01:00.0", "0x10de", "0x030200", "2", "")]);
        assert_eq!(gpu_socket_in(root.path(), 0), None);
    }

    /// `numa_node = -1` (no proximity domain exposed) is unresolved, not node 4294967295.
    #[test]
    fn unknown_numa_node_is_rejected() {
        let root = fake_pci_root(&[("0008:01:00.0", "0x10de", "0x030200", "-1", "0-71")]);
        assert_eq!(gpu_socket_in(root.path(), 0), None);
    }

    /// Non-NVIDIA devices are skipped even when they sit at a lower PCI address.
    #[test]
    fn other_vendors_do_not_shift_ordinals() {
        let root = fake_pci_root(&[
            ("0001:01:00.0", "0x1000", "0x030200", "1", "72-143"),
            ("0008:01:00.0", "0x10de", "0x030200", "0", "0-71"),
        ]);
        let socket = gpu_socket_in(root.path(), 0).unwrap();
        assert_eq!(socket.numa_node, 0);
    }

    #[test]
    fn cpulist_forms() {
        assert_eq!(parse_cpulist("0-3"), Some(vec![0, 1, 2, 3]));
        assert_eq!(parse_cpulist("5"), Some(vec![5]));
        assert_eq!(parse_cpulist("0-1,4,6-7"), Some(vec![0, 1, 4, 6, 7]));
        assert_eq!(parse_cpulist(""), Some(vec![]));
        assert_eq!(parse_cpulist("3-1"), None);
        assert_eq!(parse_cpulist("0-x"), None);
        assert_eq!(parse_cpulist("garbage"), None);
    }
}
