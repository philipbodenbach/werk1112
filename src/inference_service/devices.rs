//! Physical inventory is independent of CUDA's process-local ordinal space.
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, process::Command};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Device {
    pub id: String,
    pub physical_index: u32,
    pub vendor: String,
    pub name: String,
    pub architecture: Option<String>,
    pub runtime: String,
    pub pci_bus_id: String,
    pub total_bytes: u64,
    pub free_bytes: Option<u64>,
    pub numa_node: Option<i32>,
    pub pcie_link: Option<String>,
    pub visible_index: Option<usize>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Inventory {
    pub devices: Vec<Device>,
    pub host_available_bytes: Option<u64>,
    #[serde(default)]
    pub memory_topology: Option<crate::inference::MemoryTopology>,
    /// Raw evidence only. A topology entry is not proof of CUDA peer access.
    pub topology: Option<String>,
    pub warnings: Vec<String>,
}

impl Inventory {
    pub fn detect() -> Result<Self> {
        let resources = super::detect_host_resources();
        let host_available_bytes = resources.host_memory_bytes;
        let query = [
            "--query-gpu=index,uuid,name,pci.bus_id,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ];
        let output = ["nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi"]
            .iter()
            .find_map(|program| {
                Command::new(program)
                    .args(query)
                    .output()
                    .ok()
                    .filter(|o| o.status.success())
            });
        let Some(output) = output else {
            return Ok(Self { host_available_bytes, warnings: vec!["NVIDIA inventory unavailable; no CUDA placement may be inferred. ROCm/Vulkan discovery remains on existing backend paths.".into()], ..Self::default() });
        };
        let mut inventory = Self::from_csv(&String::from_utf8(output.stdout)?)?;
        inventory.host_available_bytes = host_available_bytes;
        inventory.memory_topology = resources.memory_topology;
        if let Some(output) = Command::new("nvidia-smi")
            .args([
                "--query-gpu=uuid,compute_cap",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()
            .filter(|o| o.status.success())
        {
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                if let Some((uuid, architecture)) = line.split_once(',') {
                    if let Some(device) = inventory.devices.iter_mut().find(|d| d.id == uuid.trim())
                    {
                        device.architecture =
                            Some(format!("sm_{}", architecture.trim().replace('.', "")));
                    }
                }
            }
        }
        // Never equate nvidia-smi numbering with CUDA ordinals. Multi-device
        // numeric masks are rejected until a driver-level enumeration is present.
        inventory.apply_visibility(
            std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref(),
            std::env::var("NVIDIA_VISIBLE_DEVICES").ok().as_deref(),
        )?;
        inventory.topology = Command::new("nvidia-smi")
            .args(["topo", "-m"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned());
        for device in &mut inventory.devices {
            let pci = device.pci_bus_id.to_ascii_lowercase();
            let pci = if pci.starts_with("00000000:") {
                format!("0000:{}", &pci[9..])
            } else {
                pci
            };
            let base = format!("/sys/bus/pci/devices/{pci}");
            device.numa_node = std::fs::read_to_string(format!("{base}/numa_node"))
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .filter(|n| *n >= 0);
            device.pcie_link = std::fs::read_to_string(format!("{base}/current_link_speed"))
                .ok()
                .map(|s| s.trim().to_owned());
        }
        Ok(inventory)
    }

    pub fn from_csv(csv: &str) -> Result<Self> {
        let mut devices = Vec::new();
        let mut ids = BTreeSet::new();
        for line in csv.lines().filter(|l| !l.trim().is_empty()) {
            let fields: Vec<_> = line.split(',').map(str::trim).collect();
            if fields.len() != 6 || !fields[1].starts_with("GPU-") {
                bail!("unrecognized NVIDIA inventory row: {line}");
            }
            if !ids.insert(fields[1].to_string()) {
                bail!("duplicate GPU UUID in inventory");
            }
            let mib = |s: &str| -> Result<u64> {
                s.parse::<u64>()?
                    .checked_mul(1024 * 1024)
                    .context("GPU memory overflow")
            };
            let total_bytes = mib(fields[4])?;
            let free_bytes = mib(fields[5]).ok();
            if total_bytes == 0 || free_bytes.is_some_and(|free| free > total_bytes) {
                bail!("invalid GPU memory measurement");
            }
            devices.push(Device {
                id: fields[1].into(),
                physical_index: fields[0].parse()?,
                vendor: "nvidia".into(),
                name: fields[2].into(),
                architecture: None,
                runtime: "cuda".into(),
                pci_bus_id: fields[3].into(),
                total_bytes,
                free_bytes,
                numa_node: None,
                pcie_link: None,
                visible_index: Some(devices.len()),
            });
        }
        Ok(Self {
            devices,
            ..Self::default()
        })
    }

    pub fn apply_visibility(&mut self, cuda: Option<&str>, container: Option<&str>) -> Result<()> {
        let resolve = |mask: Option<&str>, container_mask: bool| -> Result<Vec<String>> {
            let Some(mask) = mask else {
                return Ok(self.devices.iter().map(|d| d.id.clone()).collect());
            };
            if mask == "all" && container_mask {
                return Ok(self.devices.iter().map(|d| d.id.clone()).collect());
            }
            if matches!(mask.trim(), "" | "-1" | "none" | "void") {
                return Ok(Vec::new());
            }
            let mut ids = Vec::new();
            for token in mask.split(',').map(str::trim) {
                if token.starts_with("MIG-") {
                    bail!(
                        "MIG visibility is not supported by deployment profiles; no parent GPU will be substituted"
                    );
                }
                let matches: Vec<_> = self
                    .devices
                    .iter()
                    .filter(|d| {
                        if token.starts_with("GPU-") {
                            d.id.starts_with(token)
                        } else if container_mask {
                            token.parse::<u32>().ok() == Some(d.physical_index)
                        } else {
                            self.devices.len() == 1 && token == "0"
                        }
                    })
                    .collect();
                if matches.len() != 1 {
                    bail!(
                        "cannot unambiguously resolve visibility '{token}'; use full GPU UUIDs (CUDA numeric ordinals are not nvidia-smi indices)"
                    );
                }
                let id = matches[0].id.clone();
                if ids.contains(&id) {
                    bail!("duplicate device in visibility mask");
                }
                ids.push(id);
            }
            Ok(ids)
        };
        let container_ids = resolve(container, true)?;
        let mut cuda_ids = resolve(cuda, false)?;
        cuda_ids.retain(|id| container_ids.contains(id));
        for device in &mut self.devices {
            device.visible_index = cuda_ids.iter().position(|id| id == &device.id);
        }
        Ok(())
    }

    pub fn visible(&self) -> Vec<&Device> {
        let mut devices: Vec<_> = self
            .devices
            .iter()
            .filter(|d| d.visible_index.is_some())
            .collect();
        devices.sort_by_key(|d| d.visible_index);
        devices
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn two() -> Inventory {
        Inventory::from_csv(
            "0, GPU-aaa, A, 0000:01:00.0, 24000, 23000\n1, GPU-bbb, B, 0000:02:00.0, 12000, 10000",
        )
        .unwrap()
    }
    #[test]
    fn empty_and_unknown_topology() {
        assert!(Inventory::from_csv("").unwrap().devices.is_empty());
        assert!(two().topology.is_none());
    }
    #[test]
    fn visibility_reorders_without_expansion() {
        let mut i = two();
        i.apply_visibility(Some("GPU-bbb,GPU-aaa"), None).unwrap();
        assert_eq!(i.visible()[0].id, "GPU-bbb");
        i.apply_visibility(Some("GPU-bbb,GPU-aaa"), Some("GPU-aaa"))
            .unwrap();
        assert_eq!(i.visible().len(), 1);
        assert_eq!(i.visible()[0].id, "GPU-aaa");
    }
    #[test]
    fn numeric_and_mig_fail_closed() {
        assert!(two().apply_visibility(Some("1,0"), None).is_err());
        assert!(two().apply_visibility(Some("MIG-abc"), None).is_err());
        assert!(two().apply_visibility(Some("GPU-a,GPU-aaa"), None).is_err());
    }
    #[test]
    fn disabled_is_empty() {
        let mut i = two();
        i.apply_visibility(Some(""), None).unwrap();
        assert!(i.visible().is_empty());
    }
    #[test]
    fn unequal_memory_is_per_device() {
        let i = two();
        assert_eq!(i.devices[1].total_bytes, 12000 * 1024 * 1024);
    }
}
