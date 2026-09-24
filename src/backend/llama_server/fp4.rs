//! Pinned standalone NVFP4 runtime. No Python, vLLM or alternate KV-cache owner.
use super::*;
use crate::backend::Fp4Kernel;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub(super) const REVISION: &str = "fc343a84bbd925b37dde3219de35ea0bed50d630";
const SOURCE: &str = "https://github.com/ggml-org/llama.cpp";
const KERNEL_ENV: &str = "GGML_CUDA_NVFP4_KERNEL";
const PATCH: &str = include_str!("../../../runtime/llama-nvfp4/patch.diff");

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Device {
    index: u32,
    compute_capability: u32,
    compiled_arch: u32,
    native_nvfp4: bool,
    marlin_nvfp4: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Capabilities {
    schema: u32,
    revision: String,
    marlin: bool,
    native: bool,
    devices: Vec<Device>,
}

/// Capture once before spawn. Persistent state must follow the child's actual
/// execution policy, not the parent environment observed at snapshot time.
#[derive(Debug, Clone, Default, Serialize)]
pub(super) struct Execution {
    environment: Vec<(String, String)>,
    fp4: Option<Capabilities>,
    build_fingerprint: Option<String>,
    policy: Option<Fp4Kernel>,
}
impl Execution {
    #[cfg(test)]
    pub(super) fn test_identity(policy: Fp4Kernel, build_fingerprint: &str) -> Self {
        Self {
            environment: vec![(KERNEL_ENV.into(), policy.label().into())],
            fp4: Some(Capabilities {
                schema: 1,
                revision: REVISION.into(),
                marlin: true,
                native: true,
                devices: vec![Device {
                    index: 0,
                    compute_capability: 120,
                    compiled_arch: 120,
                    native_nvfp4: true,
                    marlin_nvfp4: true,
                }],
            }),
            build_fingerprint: Some(build_fingerprint.into()),
            policy: Some(policy),
        }
    }
    pub(super) fn apply(&self, command: &mut Command) {
        command.env_remove(KERNEL_ENV);
        command.envs(self.environment.iter().map(|(k, v)| (k, v)));
    }
    pub(super) fn diagnostic(&self) -> Option<String> {
        let capabilities = self.fp4.as_ref()?;
        let policy = self.policy.unwrap_or_default();
        let devices = capabilities
            .devices
            .iter()
            .map(|device| {
                let route = match policy {
                    Fp4Kernel::Auto if device.native_nvfp4 => {
                        "native Blackwell (where supported), then Marlin, then GGML"
                    }
                    Fp4Kernel::Auto if device.marlin_nvfp4 => "Marlin (where supported), then GGML",
                    Fp4Kernel::Auto | Fp4Kernel::Ggml => "GGML dispatch",
                    Fp4Kernel::Native => "native Blackwell",
                    Fp4Kernel::Marlin => "Marlin W4A16",
                };
                format!(
                    "CUDA {} / SM {}: {route}",
                    device.index, device.compute_capability
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        Some(format!("NVFP4 policy {}: {devices}", policy.label()))
    }
}

fn runtime_variable(key: &str) -> bool {
    key.starts_with("LLAMA_ARG_")
        || key.starts_with("GGML_")
        || matches!(
            key,
            "CUDA_VISIBLE_DEVICES" | "CUDA_DEVICE_ORDER" | "LD_LIBRARY_PATH" | "LD_PRELOAD"
        )
}

fn resolve_policy(
    requested: Option<Fp4Kernel>,
    inherited: Option<&str>,
) -> Result<Option<Fp4Kernel>> {
    if requested.is_some() {
        return Ok(requested);
    }
    inherited
        .map(|value| {
            serde_json::from_value::<Fp4Kernel>(Value::String(value.to_owned()))
                .with_context(|| format!("invalid {KERNEL_ENV}; use auto, native, marlin or ggml"))
        })
        .transpose()
}

pub(super) fn prepare(
    executable: &Path,
    mode: LlamaCppMode,
    requested: Option<Fp4Kernel>,
) -> Result<Execution> {
    let mut environment = env::vars()
        .filter(|(key, _)| runtime_variable(key))
        .collect::<std::collections::BTreeMap<_, _>>();
    let policy = resolve_policy(requested, environment.get(KERNEL_ENV).map(String::as_str))?;
    if mode != LlamaCppMode::Cuda {
        if policy.is_some_and(|policy| !matches!(policy, Fp4Kernel::Auto)) {
            bail!("--fp4-kernel native/marlin/ggml requires the llama.cpp CUDA runtime");
        }
        return Ok(Execution {
            environment: environment.into_iter().collect(),
            ..Default::default()
        });
    }
    let probe = probe_path(executable);
    if !probe.is_file() {
        if policy.is_some_and(|policy| !matches!(policy, Fp4Kernel::Auto | Fp4Kernel::Ggml)) {
            bail!(
                "the selected llama-server has no verified NVFP4 kernel controls; install `werk backend install llama-cuda-nvfp4`"
            );
        }
        // Stock GGML already owns its own dispatch. Never advertise Marlin for it.
        environment.remove(KERNEL_ENV);
        return Ok(Execution {
            environment: environment.into_iter().collect(),
            ..Default::default()
        });
    }
    let (capabilities, build_fingerprint) = verified_profile(executable)?;
    let policy = policy.unwrap_or_default();
    validate_policy(&capabilities, policy)?;
    environment.insert(KERNEL_ENV.into(), policy.label().into());
    Ok(Execution {
        environment: environment.into_iter().collect(),
        fp4: Some(capabilities),
        build_fingerprint: Some(build_fingerprint),
        policy: Some(policy),
    })
}

fn validate_policy(capabilities: &Capabilities, policy: Fp4Kernel) -> Result<()> {
    if capabilities.schema != 1 || capabilities.revision != REVISION {
        bail!(
            "NVFP4 runtime capability schema or source revision differs from the supported profile"
        );
    }
    if capabilities.devices.is_empty() {
        bail!("the NVFP4 CUDA runtime reports no visible CUDA devices");
    }
    for device in &capabilities.devices {
        if device.compiled_arch == 0 {
            bail!(
                "the NVFP4 CUDA build contains no runnable kernel image for device {} (SM {}); rebuild for this GPU",
                device.index,
                device.compute_capability
            );
        }
        match policy {
            Fp4Kernel::Native if !capabilities.native || !device.native_nvfp4 => bail!(
                "native NVFP4 is unavailable on CUDA device {} (SM {}); it requires a Blackwell GPU and architecture-specific CUDA kernels",
                device.index,
                device.compute_capability
            ),
            Fp4Kernel::Marlin if !capabilities.marlin || !device.marlin_nvfp4 => bail!(
                "Marlin NVFP4 is unavailable on CUDA device {} (SM {}) in this build",
                device.index,
                device.compute_capability
            ),
            _ => {}
        }
    }
    Ok(())
}

fn probe_path(executable: &Path) -> PathBuf {
    executable.with_file_name(if cfg!(windows) {
        "werk-fp4-probe.exe"
    } else {
        "werk-fp4-probe"
    })
}

fn probe_capabilities(executable: &Path) -> Result<Capabilities> {
    Ok(verified_profile(executable)?.0)
}

// Routing is evaluated on every request. Hash large CUDA libraries only when
// their file identities change; querying CUDA must not add per-token overhead.
fn verified_profile(executable: &Path) -> Result<(Capabilities, String)> {
    type Cache = HashMap<String, (String, Capabilities, String)>;
    static PROFILES: OnceLock<Mutex<Cache>> = OnceLock::new();
    let mut files = vec![
        executable.to_path_buf(),
        probe_path(executable),
        executable.with_file_name("werk-fp4-build.json"),
    ];
    files.extend(cuda_library_paths(executable)?);
    let mut stamps = Vec::new();
    for path in files {
        let metadata = fs::metadata(&path).with_context(|| {
            format!(
                "missing NVFP4 runtime component {}; run `werk backend install llama-cuda-nvfp4`",
                path.display()
            )
        })?;
        #[cfg(unix)]
        let native_identity = {
            use std::os::unix::fs::MetadataExt;
            json!([
                metadata.dev(),
                metadata.ino(),
                metadata.ctime(),
                metadata.ctime_nsec()
            ])
        };
        #[cfg(not(unix))]
        let native_identity = Value::Null;
        stamps.push(json!([
            path,
            metadata.len(),
            metadata
                .modified()?
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
                .to_string(),
            native_identity
        ]));
    }
    let key = serde_json::to_string(&json!([
        executable,
        env::var_os("CUDA_VISIBLE_DEVICES"),
        env::var_os("CUDA_DEVICE_ORDER"),
        env::var_os("LD_LIBRARY_PATH"),
        env::var_os("LD_PRELOAD")
    ]))?;
    let stamp = serde_json::to_string(&stamps)?;
    let mut cache = PROFILES
        .get_or_init(Mutex::default)
        .lock()
        .map_err(|_| anyhow!("NVFP4 probe cache poisoned"))?;
    if let Some((previous, capabilities, build)) = cache.get(&key) {
        if previous == &stamp {
            return Ok((capabilities.clone(), build.clone()));
        }
    }
    let build = verify_receipt(executable)?;
    let capabilities = query_capabilities(executable)?;
    if cache.len() >= 16 {
        cache.clear();
    }
    cache.insert(key, (stamp, capabilities.clone(), build.clone()));
    Ok((capabilities, build))
}

fn query_capabilities(executable: &Path) -> Result<Capabilities> {
    let output = Command::new(probe_path(executable)).output()
        .context("cannot query the installed NVFP4 CUDA runtime; install `werk backend install llama-cuda-nvfp4`")?;
    if !output.status.success() {
        bail!(
            "NVFP4 capability probe failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let capabilities: Capabilities = serde_json::from_slice(&output.stdout)
        .context("NVFP4 runtime returned invalid capability metadata")?;
    validate_policy(&capabilities, Fp4Kernel::Auto)?;
    Ok(capabilities)
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
struct BuildReceipt {
    patch_sha256: String,
    server_sha256: String,
    probe_sha256: String,
    cuda_libraries: Vec<(String, String)>,
}
fn digest_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let bytes = file.read(&mut buffer)?;
        if bytes == 0 {
            break;
        }
        hash.update(&buffer[..bytes]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn cuda_library_paths(executable: &Path) -> Result<Vec<PathBuf>> {
    let directory = executable
        .parent()
        .context("NVFP4 executable has no parent")?;
    let mut libraries = Vec::new();
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if (name.starts_with("libggml-cuda") || name.starts_with("ggml-cuda")) && path.is_file() {
            libraries.push(path);
        }
    }
    libraries.sort();
    Ok(libraries)
}
fn build_receipt(executable: &Path) -> Result<BuildReceipt> {
    let cuda_libraries = cuda_library_paths(executable)?
        .into_iter()
        .map(|path| {
            Ok((
                path.file_name()
                    .context("CUDA library has no filename")?
                    .to_string_lossy()
                    .into_owned(),
                digest_file(&path)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(BuildReceipt {
        patch_sha256: format!("{:x}", Sha256::digest(PATCH.as_bytes())),
        server_sha256: digest_file(executable)?,
        probe_sha256: digest_file(&probe_path(executable))?,
        cuda_libraries,
    })
}
/// Called only after a successful pinned source build and capability check.
pub(super) fn register_runtime(executable: &Path) -> Result<()> {
    let capabilities = query_capabilities(executable)?;
    if !capabilities.marlin {
        bail!("NVFP4 profile was built without Marlin kernels");
    }
    fs::write(
        executable.with_file_name("werk-fp4-build.json"),
        serde_json::to_vec_pretty(&build_receipt(executable)?)?,
    )?;
    Ok(())
}
fn verify_receipt(executable: &Path) -> Result<String> {
    let file = fs::File::open(executable.with_file_name("werk-fp4-build.json")).context(
        "NVFP4 runtime has no matching build receipt; run `werk backend install llama-cuda-nvfp4`",
    )?;
    let receipt: BuildReceipt = serde_json::from_reader(file.take(64 * 1024))
        .context("invalid NVFP4 runtime build receipt")?;
    if receipt != build_receipt(executable)? {
        bail!(
            "NVFP4 server, CUDA kernels, probe or patch changed since installation; run `werk backend install llama-cuda-nvfp4`"
        );
    }
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&receipt)?)
    ))
}

pub(super) fn validate_runtime(executable: &Path) -> Result<()> {
    let capabilities = probe_capabilities(executable)?;
    if !capabilities.marlin {
        bail!("NVFP4 CUDA profile was built without its Marlin kernels");
    }
    Ok(())
}

pub(super) fn is_nvfp4_model(store: &ModelStore, manifest: &ModelManifest) -> Result<bool> {
    if manifest.format != ModelFormat::Gguf {
        return Ok(false);
    }
    Ok(store
        .quantization_profile(manifest)?
        .is_some_and(|profile| profile.is_authoritative() && profile.has_nvfp4()))
}

// Keep Linux's system GCC/ld/OpenMP together when Homebrew shadows ld. Explicit
// compiler choices retain precedence. Only build children receive this PATH.
fn repair_system_toolchain() -> bool {
    cfg!(target_os = "linux")
        && ["CC", "CXX", "CUDAHOSTCXX", "WERK_LLAMA_CUDA_HOST_COMPILER"]
            .iter()
            .all(|key| env::var_os(key).is_none())
        && Path::new("/usr/bin/g++").is_file()
        && find_in_path("ld").is_some_and(|path| path.to_string_lossy().contains("linuxbrew"))
}

pub(super) fn build_command() -> Command {
    let mut command = Command::new("cmake");
    if repair_system_toolchain() {
        let mut paths = vec![PathBuf::from("/usr/bin"), PathBuf::from("/bin")];
        if let Some(nvcc) = cuda_compiler().and_then(|path| path.parent().map(Path::to_path_buf)) {
            paths.insert(0, nvcc);
        }
        paths.extend(env::split_paths(&env::var_os("PATH").unwrap_or_default()));
        if let Ok(path) = env::join_paths(paths) {
            command.env("PATH", path);
        }
    }
    command
}

pub(super) fn configure_build(command: &mut Command) -> Result<()> {
    let nccl = env::var("WERK_LLAMA_CUDA_NCCL")
        .ok()
        .map(|value| {
            parse_boolean_setting(&value).context("WERK_LLAMA_CUDA_NCCL must be true or false")
        })
        .transpose()?
        .unwrap_or(false);
    // NCCL is optional; the standalone profile does not require a separately
    // installed collective-communication runtime. Users may explicitly enable it.
    command.arg(format!(
        "-DGGML_CUDA_NCCL={}",
        if nccl { "ON" } else { "OFF" }
    ));
    if repair_system_toolchain() {
        command.args([
            "-DCMAKE_C_COMPILER=/usr/bin/gcc",
            "-DCMAKE_CXX_COMPILER=/usr/bin/g++",
            "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++",
        ]);
        let output = Command::new("/usr/bin/g++")
            .arg("-print-file-name=libgomp.so")
            .output()?;
        let library = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
        if output.status.success() && library.is_absolute() && library.is_file() {
            command.arg(format!("-DOpenMP_gomp_LIBRARY={}", library.display()));
        }
    }
    Ok(())
}

fn prepare_checkout(directory: &Path, source: &str, revision: &str, verbose: bool) -> Result<()> {
    if !directory.join(".git").is_dir() {
        if directory.exists() {
            bail!(
                "NVFP4 source path already exists without a Git checkout: {}",
                directory.display()
            );
        }
        run_command(
            Command::new("git").arg("init").arg(directory),
            "cannot initialize NVFP4 runtime source",
            verbose,
        )?;
    }
    let head = || {
        Command::new("git")
            .arg("-C")
            .arg(directory)
            .args(["rev-parse", "--verify", "HEAD"])
            .output()
    };
    let output = head()?;
    if output.status.success() {
        if String::from_utf8_lossy(&output.stdout).trim() != revision {
            bail!("NVFP4 source revision differs from the tested pin; refusing to overwrite it");
        }
        return Ok(());
    }
    // An interrupted first fetch leaves an initialized repository without HEAD.
    // Retry that state while still refusing to overwrite another checked-out pin.
    run_command(
        Command::new("git")
            .arg("-C")
            .arg(directory)
            .args(["fetch", "--depth", "1", source, revision]),
        "cannot fetch pinned NVFP4 runtime",
        verbose,
    )?;
    run_command(
        Command::new("git")
            .arg("-C")
            .arg(directory)
            .args(["checkout", "--detach", revision]),
        "cannot select pinned NVFP4 runtime",
        verbose,
    )?;
    let output = head()?;
    if !output.status.success() || String::from_utf8_lossy(&output.stdout).trim() != revision {
        bail!("NVFP4 source checkout does not match the required revision");
    }
    Ok(())
}

pub(super) fn prepare_source(directory: &Path, verbose: bool) -> Result<()> {
    prepare_checkout(directory, SOURCE, REVISION, verbose)?;
    let patch = directory
        .parent()
        .context("NVFP4 source has no parent")?
        .join("werk-nvfp4.patch");
    fs::write(&patch, PATCH)?;
    let check = |reverse: bool| -> Result<bool> {
        let mut command = Command::new("git");
        command.arg("-C").arg(directory).args(["apply", "--check"]);
        if reverse {
            command.arg("--reverse");
        }
        Ok(command.arg(&patch).output()?.status.success())
    };
    if check(true)? {
        return Ok(());
    }
    if !check(false)? {
        bail!(
            "NVFP4 source does not match the supported kernel patch; refusing to overwrite local changes"
        );
    }
    run_command(
        Command::new("git")
            .arg("-C")
            .arg(directory)
            .arg("apply")
            .arg(&patch),
        "cannot apply standalone NVFP4 kernels",
        verbose,
    )
}

/// Keep automatic host detection stable for one process, with explicit build
/// architecture overrides still taking precedence. A new GPU on the next run
/// gets a separate build instead of reusing an older Marlin-only binary.
pub(super) fn build_architecture() -> Option<String> {
    for key in ["WERK_LLAMA_CUDA_ARCH", "CUDAARCHS"] {
        if let Ok(value) = env::var(key)
            && !value.trim().is_empty()
        {
            return Some(value.trim().to_owned());
        }
    }
    static ARCHITECTURE: OnceLock<Option<String>> = OnceLock::new();
    ARCHITECTURE.get_or_init(cuda_architecture).clone()
}

fn profile_directory_for_architecture(architecture: Option<&str>) -> String {
    let patch = format!("{:x}", Sha256::digest(PATCH.as_bytes()));
    let arch = format!(
        "{:x}",
        Sha256::digest(architecture.unwrap_or("cmake-default").as_bytes())
    );
    format!("nvfp4-{REVISION}-{}-{}", &patch[..12], &arch[..12])
}

pub(super) fn profile_directory() -> String {
    profile_directory_for_architecture(build_architecture().as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn capabilities(sm: u32, native: bool) -> Capabilities {
        Capabilities {
            schema: 1,
            revision: REVISION.into(),
            native,
            marlin: true,
            devices: vec![Device {
                index: 0,
                compute_capability: sm,
                compiled_arch: sm,
                native_nvfp4: native,
                marlin_nvfp4: sm >= 80,
            }],
        }
    }
    #[test]
    fn checkout_recovers_after_interrupted_initial_fetch_and_preserves_other_heads() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        let target = temp.path().join("target");
        let git = |path: &Path, args: &[&str]| {
            let output = Command::new("git")
                .arg("-C")
                .arg(path)
                .args(args)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            String::from_utf8_lossy(&output.stdout).trim().to_owned()
        };
        fs::create_dir(&source).unwrap();
        git(&source, &["init", "-q"]);
        fs::write(source.join("fixture"), "original").unwrap();
        git(&source, &["add", "fixture"]);
        git(
            &source,
            &[
                "-c",
                "user.name=Werk Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "fixture",
            ],
        );
        let revision = git(&source, &["rev-parse", "HEAD"]);
        fs::create_dir(&target).unwrap();
        git(&target, &["init", "-q"]);
        prepare_checkout(&target, source.to_str().unwrap(), &revision, false).unwrap();
        assert_eq!(
            fs::read_to_string(target.join("fixture")).unwrap(),
            "original"
        );
        fs::write(target.join("fixture"), "user edit").unwrap();
        prepare_checkout(&target, source.to_str().unwrap(), &revision, false).unwrap();
        assert_eq!(
            fs::read_to_string(target.join("fixture")).unwrap(),
            "user edit"
        );
        assert!(
            prepare_checkout(
                &target,
                source.to_str().unwrap(),
                "0000000000000000000000000000000000000000",
                false
            )
            .is_err()
        );
        assert_eq!(
            fs::read_to_string(target.join("fixture")).unwrap(),
            "user edit"
        );
    }

    #[test]
    fn explicit_policy_takes_precedence_over_invalid_inherited_setting() {
        assert_eq!(
            resolve_policy(Some(Fp4Kernel::Ggml), Some("invalid")).unwrap(),
            Some(Fp4Kernel::Ggml)
        );
        assert!(resolve_policy(None, Some("invalid")).is_err());
        assert_eq!(
            resolve_policy(None, Some("marlin")).unwrap(),
            Some(Fp4Kernel::Marlin)
        );
    }

    #[test]
    fn architecture_change_selects_a_new_native_profile() {
        assert_ne!(
            profile_directory_for_architecture(Some("86")),
            profile_directory_for_architecture(Some("120a-real"))
        );
        let mut unavailable = capabilities(120, false);
        unavailable.devices[0].compiled_arch = 0;
        assert!(validate_policy(&unavailable, Fp4Kernel::Auto).is_err());
    }

    #[test]
    fn upstream_override_works_without_a_marlin_profile() {
        let directory = tempfile::tempdir().unwrap();
        let executable = directory.path().join("llama-server");
        assert!(prepare(&executable, LlamaCppMode::Cuda, Some(Fp4Kernel::Ggml)).is_ok());
        assert!(prepare(&executable, LlamaCppMode::Cuda, Some(Fp4Kernel::Marlin)).is_err());
        assert!(prepare(&executable, LlamaCppMode::Cuda, Some(Fp4Kernel::Native)).is_err());
    }

    #[test]
    fn policy_requires_real_device_and_build_support() {
        let ampere = capabilities(86, false);
        assert!(validate_policy(&ampere, Fp4Kernel::Auto).is_ok());
        assert!(validate_policy(&ampere, Fp4Kernel::Marlin).is_ok());
        assert!(validate_policy(&ampere, Fp4Kernel::Native).is_err());
        let blackwell = capabilities(120, true);
        assert!(validate_policy(&blackwell, Fp4Kernel::Native).is_ok());
        let uncompiled = capabilities(120, false);
        assert!(validate_policy(&uncompiled, Fp4Kernel::Native).is_err());
        assert!(validate_policy(&capabilities(89, false), Fp4Kernel::Native).is_err());
    }
    #[test]
    fn explicit_policy_checks_every_visible_gpu() {
        let mut mixed = capabilities(120, true);
        mixed
            .devices
            .push(capabilities(86, false).devices.remove(0));
        assert!(validate_policy(&mixed, Fp4Kernel::Native).is_err());
        assert!(validate_policy(&mixed, Fp4Kernel::Auto).is_ok());
        mixed.schema = 2;
        assert!(validate_policy(&mixed, Fp4Kernel::Auto).is_err());
    }
    #[test]
    fn build_receipt_rejects_changed_cuda_library_or_server() {
        let root = tempfile::tempdir().unwrap();
        let executable = root.path().join("llama-server");
        let library = root.path().join("libggml-cuda.so");
        fs::write(&executable, b"server").unwrap();
        fs::write(probe_path(&executable), b"probe").unwrap();
        fs::write(&library, b"original-kernel").unwrap();
        fs::write(
            executable.with_file_name("werk-fp4-build.json"),
            serde_json::to_vec(&build_receipt(&executable).unwrap()).unwrap(),
        )
        .unwrap();
        assert!(verify_receipt(&executable).is_ok());
        fs::write(&library, b"different-kernel").unwrap();
        assert!(verify_receipt(&executable).is_err());
    }
    #[test]
    fn child_policy_overrides_inherited_kernel_setting() {
        let mut command = Command::new("unused-test-command");
        command.env(KERNEL_ENV, "native");
        let execution = Execution {
            environment: vec![(KERNEL_ENV.into(), "marlin".into())],
            ..Default::default()
        };
        execution.apply(&mut command);
        assert!(
            command
                .get_envs()
                .any(|(key, value)| key == KERNEL_ENV
                    && value == Some(std::ffi::OsStr::new("marlin")))
        );
        Execution::default().apply(&mut command);
        assert!(
            command
                .get_envs()
                .any(|(key, value)| key == KERNEL_ENV && value.is_none())
        );
    }

    #[test]
    fn execution_identity_distinguishes_kernel_and_activation_paths() {
        let execution = |policy| Execution {
            build_fingerprint: None,
            policy: Some(policy),
            fp4: Some(capabilities(120, true)),
            environment: vec![(KERNEL_ENV.into(), policy.label().into())],
        };
        assert_ne!(
            serde_json::to_value(execution(Fp4Kernel::Marlin)).unwrap(),
            serde_json::to_value(execution(Fp4Kernel::Native)).unwrap()
        );
        assert!(
            execution(Fp4Kernel::Auto)
                .diagnostic()
                .unwrap()
                .contains("native Blackwell")
        );
    }
}
