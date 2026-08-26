#[cfg(target_os = "windows")]
use std::path::PathBuf;
use std::{env, fs};
#[cfg(not(target_os = "macos"))]
use std::{path::Path, process::Command, sync::OnceLock};

use crate::inference::{HostResources, MemoryTopology, RuntimeAccelerator};

#[cfg(any(not(target_os = "macos"), test))]
const CUDA_DEVICE_PATHS: [&str; 3] = ["/dev/nvidiactl", "/dev/nvidia0", "/dev/dxg"];

#[cfg(any(not(target_os = "macos"), test))]
const MEBIBYTE_BYTES: u64 = 1024 * 1024;

#[cfg(not(target_os = "macos"))]
static NVIDIA_MEMORY_BYTES: OnceLock<Option<u64>> = OnceLock::new();

#[cfg(target_os = "linux")]
static DGX_SPARK: OnceLock<bool> = OnceLock::new();

pub fn detect_host_resources() -> HostResources {
    let host_memory_bytes = fs::read_to_string("/proc/meminfo").ok().and_then(|data| {
        data.lines()
            .find(|line| line.starts_with("MemAvailable:"))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|value| value.parse::<u64>().ok())
            .map(|kib| kib.saturating_mul(1024))
    });
    let accelerator = detected_accelerator();
    let memory_topology = detected_memory_topology();
    let configured_accelerator_memory_bytes = env::var("WERK_ACCELERATOR_MEMORY_BYTES")
        .ok()
        .and_then(|value| value.parse().ok());
    let accelerator_memory_bytes = select_accelerator_memory_bytes(
        accelerator,
        configured_accelerator_memory_bytes,
        memory_topology,
        || detected_accelerator_memory_bytes(accelerator),
    );
    HostResources {
        host_memory_bytes,
        accelerator_memory_bytes,
        accelerator: Some(format!("{accelerator:?}").to_ascii_lowercase()),
        memory_topology,
    }
}

fn select_accelerator_memory_bytes<F>(
    accelerator: RuntimeAccelerator,
    configured: Option<u64>,
    memory_topology: Option<MemoryTopology>,
    detect: F,
) -> Option<u64>
where
    F: FnOnce() -> Option<u64>,
{
    if accelerator == RuntimeAccelerator::Cpu {
        return None;
    }
    configured.or_else(|| {
        (memory_topology != Some(MemoryTopology::Unified))
            .then(detect)
            .flatten()
    })
}

#[cfg(target_os = "linux")]
fn detected_memory_topology() -> Option<MemoryTopology> {
    is_dgx_spark_environment().then_some(MemoryTopology::Unified)
}

#[cfg(not(target_os = "linux"))]
fn detected_memory_topology() -> Option<MemoryTopology> {
    None
}

#[cfg(target_os = "linux")]
fn is_dgx_spark_environment() -> bool {
    *DGX_SPARK.get_or_init(probe_dgx_spark_environment)
}

#[cfg(target_os = "linux")]
fn probe_dgx_spark_environment() -> bool {
    if !matches!(env::consts::ARCH, "aarch64" | "arm64") {
        return false;
    }

    let device_tree_model = [
        "/proc/device-tree/model",
        "/sys/firmware/devicetree/base/model",
    ]
    .into_iter()
    .find_map(|path| {
        fs::read(path)
            .ok()
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
    });
    if dgx_spark_signals(env::consts::ARCH, device_tree_model.as_deref(), None) {
        return true;
    }

    let nvidia_smi = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned());
    dgx_spark_signals(
        env::consts::ARCH,
        device_tree_model.as_deref(),
        nvidia_smi.as_deref(),
    )
}

#[cfg(any(target_os = "linux", test))]
fn dgx_spark_signals(
    architecture: &str,
    device_tree_model: Option<&str>,
    nvidia_smi: Option<&str>,
) -> bool {
    if !matches!(architecture, "aarch64" | "arm64") {
        return false;
    }
    [device_tree_model, nvidia_smi]
        .into_iter()
        .flatten()
        .any(signal_identifies_dgx_spark)
}

#[cfg(any(target_os = "linux", test))]
fn signal_identifies_dgx_spark(signal: &str) -> bool {
    let lower = signal.to_ascii_lowercase();
    lower.contains("dgx spark")
        || lower.contains("nvidia spark")
        || lower
            .split(|character: char| !character.is_ascii_alphanumeric())
            .any(|token| token == "gb10")
}

#[cfg(not(target_os = "macos"))]
fn detected_accelerator_memory_bytes(accelerator: RuntimeAccelerator) -> Option<u64> {
    if accelerator == RuntimeAccelerator::Cuda {
        return nvidia_smi_memory_bytes();
    }
    None
}

#[cfg(target_os = "macos")]
fn detected_accelerator_memory_bytes(_accelerator: RuntimeAccelerator) -> Option<u64> {
    None
}

#[cfg(not(target_os = "macos"))]
fn nvidia_smi_memory_bytes() -> Option<u64> {
    if cuda_visibility_disables_accelerator() {
        return None;
    }
    *NVIDIA_MEMORY_BYTES.get_or_init(probe_nvidia_smi_memory_bytes)
}

#[cfg(not(target_os = "macos"))]
fn cuda_visibility_disables_accelerator() -> bool {
    if let Ok(value) = env::var("CUDA_VISIBLE_DEVICES") {
        return visibility_value_disables_accelerator(&value);
    }
    env::var("NVIDIA_VISIBLE_DEVICES")
        .ok()
        .is_some_and(|value| visibility_value_disables_accelerator(&value))
}

#[cfg(any(not(target_os = "macos"), test))]
fn visibility_value_disables_accelerator(value: &str) -> bool {
    let value = value.trim();
    let first = value.split(',').next().unwrap_or_default().trim();
    first.is_empty()
        || matches!(
            first.to_ascii_lowercase().as_str(),
            "-1" | "none" | "disabled" | "void"
        )
}

#[cfg(not(target_os = "macos"))]
fn probe_nvidia_smi_memory_bytes() -> Option<u64> {
    let cuda_visible = env::var("CUDA_VISIBLE_DEVICES").ok();
    let nvidia_visible = env::var("NVIDIA_VISIBLE_DEVICES").ok();
    let device = selected_nvidia_device(cuda_visible.as_deref(), nvidia_visible.as_deref());

    if let Some(bytes) = query_nvidia_smi_memory(Path::new("nvidia-smi"), device) {
        return Some(bytes);
    }
    #[cfg(target_os = "linux")]
    if let Some(bytes) = query_nvidia_smi_memory(Path::new("/usr/lib/wsl/lib/nvidia-smi"), device) {
        return Some(bytes);
    }
    #[cfg(target_os = "windows")]
    if let Some(system_root) = env::var_os("SystemRoot") {
        let executable = PathBuf::from(system_root)
            .join("System32")
            .join("nvidia-smi.exe");
        if let Some(bytes) = query_nvidia_smi_memory(&executable, device) {
            return Some(bytes);
        }
    }
    None
}

#[cfg(not(target_os = "macos"))]
fn query_nvidia_smi_memory(program: &Path, device: Option<&str>) -> Option<u64> {
    let mut command = Command::new(program);
    // CUDA's logical numeric device order can differ from nvidia-smi's
    // physical indices. UUIDs are stable across both APIs; numeric selections
    // are only used indirectly when every reported device has the same
    // capacity (or only one device is visible).
    if let Some(device) = device.filter(|device| stable_nvidia_device_id(device)) {
        command.arg(format!("--id={device}"));
    }
    let output = command
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    unambiguous_nvidia_smi_memory_bytes(&String::from_utf8_lossy(&output.stdout))
}

#[cfg(any(not(target_os = "macos"), test))]
fn stable_nvidia_device_id(device: &str) -> bool {
    let upper = device.trim().to_ascii_uppercase();
    upper.starts_with("GPU-") || upper.starts_with("MIG-")
}

#[cfg(any(not(target_os = "macos"), test))]
fn selected_nvidia_device<'a>(
    cuda_visible: Option<&'a str>,
    nvidia_visible: Option<&'a str>,
) -> Option<&'a str> {
    let value = cuda_visible.or(nvidia_visible)?;
    let device = value.split(',').next()?.trim();
    if device.is_empty()
        || matches!(
            device.to_ascii_lowercase().as_str(),
            "-1" | "all" | "none" | "disabled" | "void"
        )
    {
        None
    } else {
        Some(device)
    }
}

#[cfg(test)]
fn parse_nvidia_smi_memory_bytes(output: &str) -> Option<u64> {
    parsed_nvidia_smi_memory_bytes(output).into_iter().next()
}

#[cfg(any(not(target_os = "macos"), test))]
fn unambiguous_nvidia_smi_memory_bytes(output: &str) -> Option<u64> {
    let values = output
        .lines()
        .map(|line| line.trim().trim_start_matches('\u{feff}'))
        .filter(|line| !line.is_empty())
        .map(|line| {
            let value = line.split_whitespace().next()?.parse::<u64>().ok()?;
            (value > 0).then(|| value.saturating_mul(MEBIBYTE_BYTES))
        })
        .collect::<Option<Vec<_>>>()?;
    let first = values.first().copied()?;
    values.iter().all(|value| *value == first).then_some(first)
}

#[cfg(test)]
fn parsed_nvidia_smi_memory_bytes(output: &str) -> Vec<u64> {
    output
        .lines()
        .filter_map(|line| {
            let value = line
                .trim()
                .trim_start_matches('\u{feff}')
                .split_whitespace()
                .next()?
                .parse::<u64>()
                .ok()?;
            (value > 0).then(|| value.saturating_mul(MEBIBYTE_BYTES))
        })
        .collect()
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
        let cuda = !cuda_visibility_disables_accelerator()
            && (CUDA_DEVICE_PATHS
                .iter()
                .any(|path| Path::new(path).exists())
                || accelerator_env_is_enabled("CUDA_VISIBLE_DEVICES")
                || accelerator_env_is_enabled("NVIDIA_VISIBLE_DEVICES")
                || nvidia_smi_memory_bytes().is_some());
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

#[cfg(not(target_os = "macos"))]
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
    fn spark_detection_requires_arm64_and_an_exact_spark_or_gb10_signal() {
        assert!(dgx_spark_signals(
            "aarch64",
            Some("NVIDIA DGX Spark\0"),
            None,
        ));
        assert!(dgx_spark_signals("arm64", None, Some("NVIDIA GB10\n"),));
        assert!(!dgx_spark_signals(
            "x86_64",
            Some("NVIDIA DGX Spark"),
            Some("NVIDIA GB10"),
        ));
        assert!(!dgx_spark_signals("aarch64", None, Some("NVIDIA GB100"),));
    }

    #[test]
    fn unified_memory_skips_automatic_vram_but_keeps_explicit_override() {
        let probe_called = std::cell::Cell::new(false);
        let automatic = select_accelerator_memory_bytes(
            RuntimeAccelerator::Cuda,
            None,
            Some(MemoryTopology::Unified),
            || {
                probe_called.set(true);
                Some(128 * 1024 * 1024 * 1024)
            },
        );
        assert_eq!(automatic, None);
        assert!(!probe_called.get());

        let configured = select_accelerator_memory_bytes(
            RuntimeAccelerator::Cuda,
            Some(96 * 1024 * 1024 * 1024),
            Some(MemoryTopology::Unified),
            || panic!("explicit override must skip automatic probing"),
        );
        assert_eq!(configured, Some(96 * 1024 * 1024 * 1024));
    }

    #[test]
    fn non_unified_accelerators_keep_existing_memory_probe_behavior() {
        assert_eq!(
            select_accelerator_memory_bytes(
                RuntimeAccelerator::Cuda,
                None,
                Some(MemoryTopology::Discrete),
                || Some(24 * 1024 * 1024 * 1024),
            ),
            Some(24 * 1024 * 1024 * 1024)
        );
        assert_eq!(
            select_accelerator_memory_bytes(
                RuntimeAccelerator::Cpu,
                Some(24 * 1024 * 1024 * 1024),
                None,
                || panic!("CPU resources must not probe accelerator memory"),
            ),
            None
        );
    }

    #[test]
    fn wsl_dxg_cuda_signal_selects_cuda() {
        assert!(CUDA_DEVICE_PATHS.contains(&"/dev/dxg"));
        assert_eq!(
            accelerator_from_hardware_signals(true, false),
            RuntimeAccelerator::Cuda
        );
    }

    #[test]
    fn parses_nvidia_smi_mebibytes_with_or_without_units() {
        assert_eq!(
            parse_nvidia_smi_memory_bytes("24576\n"),
            Some(24_576 * MEBIBYTE_BYTES)
        );
        assert_eq!(
            parse_nvidia_smi_memory_bytes("memory.total [MiB]\n24576 MiB\n"),
            Some(24_576 * MEBIBYTE_BYTES)
        );
    }

    #[test]
    fn ignores_invalid_nvidia_smi_memory_rows() {
        assert_eq!(parse_nvidia_smi_memory_bytes("N/A\n0\n"), None);
        assert_eq!(
            parse_nvidia_smi_memory_bytes("N/A\n12288\n24576\n"),
            Some(12_288 * MEBIBYTE_BYTES)
        );
    }

    #[test]
    fn accepts_only_unambiguous_multi_gpu_memory_without_a_stable_id() {
        assert_eq!(
            unambiguous_nvidia_smi_memory_bytes("24576\n24576\n"),
            Some(24_576 * MEBIBYTE_BYTES)
        );
        assert_eq!(unambiguous_nvidia_smi_memory_bytes("24576\n12288\n"), None);
        assert_eq!(unambiguous_nvidia_smi_memory_bytes("24576\nN/A\n"), None);
        assert!(stable_nvidia_device_id("GPU-abc"));
        assert!(stable_nvidia_device_id("MIG-GPU-abc/1/0"));
        assert!(!stable_nvidia_device_id("0"));
    }

    #[test]
    fn selects_first_cuda_visible_device_for_memory_query() {
        assert_eq!(
            selected_nvidia_device(Some("2,0"), Some("GPU-fallback")),
            Some("2")
        );
        assert_eq!(
            selected_nvidia_device(None, Some("GPU-abc,GPU-def")),
            Some("GPU-abc")
        );
        assert_eq!(selected_nvidia_device(Some("all"), Some("GPU-abc")), None);
    }

    #[test]
    fn explicit_empty_or_disabled_visibility_suppresses_cuda_probe() {
        for value in ["", " ", "-1", "-1,0", "none", "none,0", "disabled", "void"] {
            assert!(visibility_value_disables_accelerator(value));
        }
        for value in ["all", "0", "0,-1", "2,0", "GPU-abc"] {
            assert!(!visibility_value_disables_accelerator(value));
        }
    }
}
