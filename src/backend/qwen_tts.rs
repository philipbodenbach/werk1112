//! Managed, isolated Python runtime discovery for Qwen3-TTS.
//!
//! `qwen-tts` pins versions of shared Python packages such as Transformers.
//! Keeping it in a dedicated virtual environment prevents those constraints
//! from changing the Python environment used by Werk's general media
//! companion.

use anyhow::{Context, Result, anyhow, bail};
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

pub const QWEN_TTS_VERSION: &str = "0.1.1";
pub const QWEN_TTS_PACKAGE: &str = "qwen-tts==0.1.1";

fn qwen_tts_import_check() -> String {
    format!(
        "from importlib.metadata import version; \
from qwen_tts import Qwen3TTSModel; \
assert Qwen3TTSModel is not None, 'qwen_tts.Qwen3TTSModel is unavailable'; \
installed = version('qwen-tts'); \
assert installed == {expected:?}, f'expected qwen-tts {expected}, found {{installed}}'; \
print(installed)",
        expected = QWEN_TTS_VERSION,
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenTtsDiscovery {
    pub python: Option<PathBuf>,
    pub source: String,
    pub attempts: Vec<QwenTtsDiscoveryAttempt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenTtsDiscoveryAttempt {
    pub label: String,
    pub path: PathBuf,
    pub exists: bool,
    pub usable: bool,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenTtsPythonStatus {
    pub usable: bool,
    pub version: Option<String>,
    pub detail: String,
}

/// Root of Werk's isolated Qwen-TTS runtime.
pub fn managed_qwen_tts_dir(store: &crate::model_store::ModelStore) -> PathBuf {
    store.home().join("backends").join("qwen-tts")
}

/// Python executable inside Werk's isolated Qwen-TTS virtual environment.
pub fn managed_qwen_tts_python(store: &crate::model_store::ModelStore) -> PathBuf {
    virtualenv_python(&managed_qwen_tts_dir(store).join("venv"))
}

/// Discovers a validated Qwen-TTS Python interpreter.
///
/// The explicit override wins. If it is missing or unusable, discovery records
/// the failure and checks Werk's managed environment. Arbitrary PATH
/// environments are deliberately not considered, preserving dependency
/// isolation.
pub fn discover_qwen_tts(store: &crate::model_store::ModelStore) -> QwenTtsDiscovery {
    discover_qwen_tts_with_override(
        store,
        env::var_os("WERK_QWEN_TTS_PYTHON").map(PathBuf::from),
    )
}

/// Returns a usable interpreter or a diagnostic with installation guidance.
pub fn require_qwen_tts_python(store: &crate::model_store::ModelStore) -> Result<PathBuf> {
    let discovery = discover_qwen_tts(store);
    discovery
        .python
        .clone()
        .ok_or_else(|| anyhow!(missing_qwen_tts_message(&discovery)))
}

/// Installs the supported Qwen-TTS package into Werk's isolated environment.
///
/// This function never modifies the Python environment running the general
/// media companion. FlashAttention remains an optional, separately managed
/// optimization and is intentionally not installed here.
pub fn install_managed_qwen_tts(store: &crate::model_store::ModelStore) -> Result<PathBuf> {
    let root = managed_qwen_tts_dir(store);
    let venv = root.join("venv");
    let venv_python = virtualenv_python(&venv);

    fs::create_dir_all(&root).with_context(|| {
        format!(
            "failed to create Qwen-TTS backend directory {}",
            root.display()
        )
    })?;

    if venv_python.is_file() && qwen_tts_python_status(&venv_python).usable {
        return Ok(venv_python);
    }

    let bootstrap = find_bootstrap_python().ok_or_else(|| {
        anyhow!(
            "no Python interpreter found for Qwen-TTS; install Python 3.9 or newer, \
             or set WERK_QWEN_TTS_PYTHON to a Python executable"
        )
    })?;
    validate_bootstrap_python(&bootstrap)?;

    if !venv_python.is_file() {
        eprintln!("Creating Qwen-TTS virtualenv at {}", venv.display());
        run_command(
            Command::new(&bootstrap).arg("-m").arg("venv").arg(&venv),
            "failed to create the managed Qwen-TTS virtualenv",
        )?;
    }

    if !venv_python.is_file() {
        bail!(
            "managed Qwen-TTS virtualenv was created without a Python executable at {}; \
             ensure the Python venv module is installed",
            venv_python.display()
        );
    }

    eprintln!("Installing {QWEN_TTS_PACKAGE} into {}", venv.display());
    run_command(
        Command::new(&venv_python)
            .arg("-m")
            .arg("pip")
            .arg("install")
            .arg("--upgrade")
            .arg("pip"),
        "failed to upgrade pip in the managed Qwen-TTS virtualenv",
    )?;
    run_command(
        Command::new(&venv_python)
            .arg("-m")
            .arg("pip")
            .arg("install")
            .arg("--upgrade")
            .arg(QWEN_TTS_PACKAGE),
        "failed to install qwen-tts==0.1.1; check network access, free disk space, \
         and PyTorch/torchaudio wheel availability for this platform",
    )?;

    validate_qwen_tts_python(&venv_python).with_context(|| {
        format!(
            "managed Qwen-TTS installation at {} is unusable; rerun `werk backend install \
             qwen-tts` after resolving the reported Python dependency error",
            venv.display()
        )
    })?;
    Ok(venv_python)
}

/// Validates the exact supported package and its public model wrapper.
pub fn validate_qwen_tts_python(path: &Path) -> Result<()> {
    let status = qwen_tts_python_status(path);
    if status.usable {
        Ok(())
    } else {
        bail!(
            "Qwen-TTS Python validation failed for {}: {}",
            path.display(),
            status.detail
        )
    }
}

/// Returns import and version status without mutating the Python environment.
pub fn qwen_tts_python_status(path: &Path) -> QwenTtsPythonStatus {
    if !path.is_file() {
        return QwenTtsPythonStatus {
            usable: false,
            version: None,
            detail: "Python path does not exist or is not a file".to_string(),
        };
    }

    match Command::new(path)
        .arg("-c")
        .arg(qwen_tts_import_check())
        .output()
    {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            QwenTtsPythonStatus {
                usable: true,
                version: Some(version.clone()),
                detail: format!("Qwen-TTS {version} model wrapper import ok"),
            }
        }
        Ok(output) => QwenTtsPythonStatus {
            usable: false,
            version: None,
            detail: command_failure_detail(
                "Python cannot import the supported qwen_tts.Qwen3TTSModel",
                &output,
            ),
        },
        Err(error) => QwenTtsPythonStatus {
            usable: false,
            version: None,
            detail: format!("failed to run Python: {error}"),
        },
    }
}

pub fn missing_qwen_tts_message(discovery: &QwenTtsDiscovery) -> String {
    let mut message = "No usable isolated Qwen-TTS runtime found.\n\nTried:".to_string();
    for attempt in &discovery.attempts {
        let exists = if attempt.exists { "exists" } else { "missing" };
        let usable = if attempt.usable {
            "usable"
        } else {
            "not usable"
        };
        message.push_str(&format!(
            "\n- {}: {} ({exists}, {usable}): {}",
            attempt.label,
            attempt.path.display(),
            attempt.detail
        ));
    }
    message.push_str("\n\nFix:");
    message.push_str("\n- run: werk backend install qwen-tts");
    message.push_str("\n- or set WERK_QWEN_TTS_PYTHON=/path/to/python-with-qwen-tts-0.1.1");
    message.push_str(
        "\nQwen-TTS is isolated because its pinned Transformers dependencies must not be \
         installed into Werk's general media environment.",
    );
    message
}

fn discover_qwen_tts_with_override(
    store: &crate::model_store::ModelStore,
    override_python: Option<PathBuf>,
) -> QwenTtsDiscovery {
    let mut attempts = Vec::new();

    if let Some(path) = override_python {
        let status = qwen_tts_python_status(&path);
        attempts.push(QwenTtsDiscoveryAttempt {
            label: "WERK_QWEN_TTS_PYTHON".to_string(),
            path: path.clone(),
            exists: path.is_file(),
            usable: status.usable,
            detail: status.detail,
        });
        if attempts.last().is_some_and(|attempt| attempt.usable) {
            return QwenTtsDiscovery {
                python: Some(path),
                source: "env WERK_QWEN_TTS_PYTHON".to_string(),
                attempts,
            };
        }
    }

    let path = managed_qwen_tts_python(store);
    let status = qwen_tts_python_status(&path);
    attempts.push(QwenTtsDiscoveryAttempt {
        label: "managed venv".to_string(),
        path: path.clone(),
        exists: path.is_file(),
        usable: status.usable,
        detail: status.detail,
    });
    if attempts.last().is_some_and(|attempt| attempt.usable) {
        return QwenTtsDiscovery {
            python: Some(path),
            source: "managed venv".to_string(),
            attempts,
        };
    }

    QwenTtsDiscovery {
        python: None,
        source: "missing".to_string(),
        attempts,
    }
}

fn virtualenv_python(venv: &Path) -> PathBuf {
    if cfg!(windows) {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python")
    }
}

fn find_bootstrap_python() -> Option<PathBuf> {
    env::var_os("WERK_QWEN_TTS_PYTHON")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| {
            bootstrap_python_names()
                .iter()
                .find_map(|name| find_in_path(name))
        })
}

fn bootstrap_python_names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["python.exe", "python3.exe", "py.exe"]
    } else {
        &["python3", "python"]
    }
}

fn find_in_path(name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(name);
    if path.components().count() > 1 && path.is_file() {
        return Some(path);
    }
    let path_var = env::var_os("PATH")?;
    env::split_paths(&path_var)
        .map(|directory| directory.join(name))
        .find(|candidate| candidate.is_file())
}

fn validate_bootstrap_python(path: &Path) -> Result<()> {
    let output = Command::new(path)
        .arg("-c")
        .arg(
            "import sys; assert sys.version_info >= (3, 9), \
             f'Python 3.9 or newer is required, found {sys.version.split()[0]}'; \
             print(sys.version.split()[0])",
        )
        .output()
        .with_context(|| format!("failed to run bootstrap Python {}", path.display()))?;
    if !output.status.success() {
        bail!(
            "Python 3.9 or newer is required to install Qwen-TTS with {}: {}",
            path.display(),
            command_failure_detail("bootstrap Python is unsupported", &output)
        );
    }
    Ok(())
}

fn run_command(command: &mut Command, context: &str) -> Result<()> {
    let status = command.status().with_context(|| context.to_string())?;
    if !status.success() {
        bail!("{context}; command exited with {status}");
    }
    Ok(())
}

fn command_failure_detail(prefix: &str, output: &Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let detail = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        output.status.to_string()
    };
    format!("{prefix}: {}", tail_chars(&detail, 4_096))
}

fn tail_chars(value: &str, max_chars: usize) -> String {
    let count = value.chars().count();
    if count <= max_chars {
        value.to_string()
    } else {
        format!(
            "...{}",
            value
                .chars()
                .skip(count.saturating_sub(max_chars))
                .collect::<String>()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_store::ModelStore;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(name: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock")
                .as_nanos();
            let path = env::temp_dir().join(format!(
                "werk-qwen-tts-{name}-{}-{unique}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("create test directory");
            Self(path)
        }

        fn store(&self) -> ModelStore {
            ModelStore::resolve(Some(self.0.clone())).expect("resolve test store")
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn managed_paths_are_backend_scoped() {
        let directory = TestDirectory::new("managed-path");
        let store = directory.store();
        let root = directory.0.join("backends").join("qwen-tts");

        assert_eq!(managed_qwen_tts_dir(&store), root);
        if cfg!(windows) {
            assert_eq!(
                managed_qwen_tts_python(&store),
                root.join("venv").join("Scripts").join("python.exe")
            );
        } else {
            assert_eq!(
                managed_qwen_tts_python(&store),
                root.join("venv").join("bin").join("python")
            );
        }
    }

    #[test]
    fn missing_python_has_actionable_status() {
        let directory = TestDirectory::new("missing-python");
        let path = directory.0.join("absent-python");

        let status = qwen_tts_python_status(&path);

        assert!(!status.usable);
        assert_eq!(status.version, None);
        assert!(status.detail.contains("does not exist"));
    }

    #[test]
    fn missing_message_lists_install_and_override_fixes() {
        let discovery = QwenTtsDiscovery {
            python: None,
            source: "missing".to_string(),
            attempts: vec![QwenTtsDiscoveryAttempt {
                label: "managed venv".to_string(),
                path: PathBuf::from("/example/qwen-tts/python"),
                exists: false,
                usable: false,
                detail: "Python path does not exist".to_string(),
            }],
        };

        let message = missing_qwen_tts_message(&discovery);

        assert!(message.contains("werk backend install qwen-tts"));
        assert!(message.contains("WERK_QWEN_TTS_PYTHON"));
        assert!(message.contains("/example/qwen-tts/python"));
    }

    #[cfg(unix)]
    fn fake_python(directory: &Path, name: &str, version: &str, success: bool) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let path = directory.join(name);
        let body = if success {
            format!("#!/bin/sh\nprintf '%s\\n' '{version}'\n")
        } else {
            "#!/bin/sh\nprintf '%s\\n' 'simulated import failure' >&2\nexit 1\n".to_string()
        };
        fs::write(&path, body).expect("write fake Python");
        let mut permissions = fs::metadata(&path)
            .expect("fake Python metadata")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).expect("make fake Python executable");
        path
    }

    #[cfg(unix)]
    #[test]
    fn explicit_override_wins_when_usable() {
        let directory = TestDirectory::new("override");
        let store = directory.store();
        let override_python = fake_python(&directory.0, "override-python", QWEN_TTS_VERSION, true);

        let discovery = discover_qwen_tts_with_override(&store, Some(override_python.clone()));

        assert_eq!(discovery.python, Some(override_python));
        assert_eq!(discovery.source, "env WERK_QWEN_TTS_PYTHON");
        assert_eq!(discovery.attempts.len(), 1);
        assert!(discovery.attempts[0].usable);
    }

    #[cfg(unix)]
    #[test]
    fn invalid_override_falls_back_to_managed_environment() {
        let directory = TestDirectory::new("override-fallback");
        let store = directory.store();
        let managed = managed_qwen_tts_python(&store);
        fs::create_dir_all(managed.parent().expect("managed Python parent"))
            .expect("create managed venv directory");
        let installed = fake_python(
            managed.parent().expect("managed Python parent"),
            managed
                .file_name()
                .expect("managed Python name")
                .to_str()
                .unwrap(),
            QWEN_TTS_VERSION,
            true,
        );
        let invalid_override = directory.0.join("missing-override");

        let discovery = discover_qwen_tts_with_override(&store, Some(invalid_override));

        assert_eq!(discovery.python, Some(installed));
        assert_eq!(discovery.source, "managed venv");
        assert_eq!(discovery.attempts.len(), 2);
        assert!(!discovery.attempts[0].usable);
        assert!(discovery.attempts[1].usable);
    }

    #[cfg(unix)]
    #[test]
    fn failed_import_is_reported_by_validation() {
        let directory = TestDirectory::new("failed-import");
        let python = fake_python(&directory.0, "broken-python", QWEN_TTS_VERSION, false);

        let error = validate_qwen_tts_python(&python).expect_err("validation must fail");

        assert!(error.to_string().contains("simulated import failure"));
        assert!(error.to_string().contains(&python.display().to_string()));
    }
}
