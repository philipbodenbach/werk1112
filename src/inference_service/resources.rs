#[cfg(not(target_os = "macos"))]
use std::path::Path;
use std::{env, fs};

use crate::inference::{HostResources, RuntimeAccelerator};

#[cfg(any(not(target_os = "macos"), test))]
const CUDA_DEVICE_PATHS: [&str; 3] = ["/dev/nvidiactl", "/dev/nvidia0", "/dev/dxg"];

pub fn detect_host_resources() -> HostResources {
    let host_memory_bytes = fs::read_to_string("/proc/meminfo").ok().and_then(|data| {
        data.lines()
            .find(|line| line.starts_with("MemAvailable:"))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|value| value.parse::<u64>().ok())
            .map(|kib| kib.saturating_mul(1024))
    });
    let accelerator_memory_bytes = env::var("WERK_ACCELERATOR_MEMORY_BYTES")
        .ok()
        .and_then(|value| value.parse().ok());
    HostResources {
        host_memory_bytes,
        accelerator_memory_bytes,
        accelerator: Some(format!("{:?}", detected_accelerator()).to_ascii_lowercase()),
    }
}

pub(super) fn detected_accelerator() -> RuntimeAccelerator {
    if let Some(accelerator) = configured_media_accelerator() {
        return accelerator;
    }
    #[cfg(target_os = "macos")]
    {
        return RuntimeAccelerator::Mps;
    }
    #[cfg(not(target_os = "macos"))]
    {
        let cuda = CUDA_DEVICE_PATHS
            .iter()
            .any(|path| Path::new(path).exists())
            || accelerator_env_is_enabled("CUDA_VISIBLE_DEVICES")
            || accelerator_env_is_enabled("NVIDIA_VISIBLE_DEVICES");
        let rocm =
            Path::new("/dev/kfd").exists() || accelerator_env_is_enabled("ROCR_VISIBLE_DEVICES");
        accelerator_from_hardware_signals(cuda, rocm)
    }
}

pub(super) fn configured_media_accelerator() -> Option<RuntimeAccelerator> {
    let value = env::var("WERK_MEDIA_ACCELERATOR").ok()?;
    if value.trim().is_empty() {
        return None;
    }
    Some(match value.trim().to_ascii_lowercase().as_str() {
        "cuda" => RuntimeAccelerator::Cuda,
        "rocm" | "hip" => RuntimeAccelerator::Rocm,
        "mps" | "metal" => RuntimeAccelerator::Mps,
        "mlx" => RuntimeAccelerator::Mlx,
        "cpu" => RuntimeAccelerator::Cpu,
        _ => RuntimeAccelerator::Other,
    })
}

fn accelerator_env_is_enabled(name: &str) -> bool {
    env::var(name).is_ok_and(|value| {
        let value = value.trim();
        !value.is_empty()
            && !matches!(
                value.to_ascii_lowercase().as_str(),
                "-1" | "none" | "disabled" | "void"
            )
    })
}

#[cfg(any(not(target_os = "macos"), test))]
fn accelerator_from_hardware_signals(
    cuda_available: bool,
    rocm_available: bool,
) -> RuntimeAccelerator {
    if cuda_available {
        RuntimeAccelerator::Cuda
    } else if rocm_available {
        RuntimeAccelerator::Rocm
    } else {
        RuntimeAccelerator::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wsl_dxg_cuda_signal_selects_cuda() {
        assert!(CUDA_DEVICE_PATHS.contains(&"/dev/dxg"));
        assert_eq!(
            accelerator_from_hardware_signals(true, false),
            RuntimeAccelerator::Cuda
        );
    }
}
