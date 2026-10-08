//! Explicit provisioning of established Python runtimes. Discovery only reads
//! completed installations and never invokes pip or downloads model weights.
use crate::model_store::ModelStore;
use anyhow::{Context, Result, ensure};
use std::{env, fs, path::PathBuf, process::Command};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PythonBackend {
    Mlx,
    MlxVlm,
    Transformers,
    Media,
    OnnxCpu,
}

impl PythonBackend {
    pub fn target(self) -> &'static str {
        match self {
            Self::Mlx => "mlx",
            Self::MlxVlm => "mlx-vlm",
            Self::Transformers => "transformers",
            Self::Media => "media",
            Self::OnnxCpu => "onnx-cpu",
        }
    }

    pub fn override_env(self) -> &'static str {
        match self {
            Self::Mlx => "WERK_MLX_PYTHON",
            Self::MlxVlm => "WERK_MLX_VLM_PYTHON",
            Self::Transformers => "WERK_TRANSFORMERS_PYTHON",
            Self::Media => "WERK_MEDIA_PYTHON",
            Self::OnnxCpu => "WERK_ONNX_GENAI_PYTHON",
        }
    }

    fn packages(self) -> &'static [&'static str] {
        match self {
            Self::OnnxCpu => &["onnxruntime-genai"],
            Self::Mlx => &["mlx-lm"],
            Self::MlxVlm => &["mlx-vlm"],
            Self::Transformers => &[
                "torch>=2.6,<3",
                "transformers>=5.19,<6",
                "accelerate",
                "sentencepiece",
                "protobuf",
            ],
            Self::Media => &[
                "torch>=2.6,<3",
                "diffusers[torch]",
                "transformers>=5.19,<6",
                "sentencepiece",
                "protobuf",
                "Pillow",
                "numpy",
                "soundfile",
                "av",
                "scipy",
                "librosa",
            ],
        }
    }

    fn validation(self) -> &'static str {
        match self {
            Self::OnnxCpu => {
                "from onnxruntime_genai import Model, Tokenizer, Generator, GeneratorParams; print('ONNX GenAI ready (CPU); models require genai_config.json')"
            }
            Self::Mlx => {
                "import mlx.core as mx; import mlx_lm.generate; assert mx.metal.is_available(), 'Metal unavailable'; print('MLX-LM ready (Metal)')"
            }
            Self::MlxVlm => {
                "import mlx.core as mx; import mlx_vlm.generate; assert mx.metal.is_available(), 'Metal unavailable'; print('MLX-VLM ready (Metal)')"
            }
            Self::Transformers => {
                "import torch, accelerate, sentencepiece; from transformers import AutoModelForCausalLM, AutoTokenizer; print('Transformers ready; CUDA/ROCm:', torch.cuda.is_available(), 'MPS:', torch.backends.mps.is_available())"
            }
            Self::Media => {
                "import torch, accelerate, transformers, numpy, PIL, soundfile, av, scipy, librosa; from diffusers import DiffusionPipeline; print('Media ready; CUDA/ROCm:', torch.cuda.is_available(), 'MPS:', torch.backends.mps.is_available())"
            }
        }
    }

    fn root(self, store: &ModelStore) -> PathBuf {
        store.home().join("backends").join(self.target())
    }

    fn python(self, store: &ModelStore) -> PathBuf {
        self.root(store).join("venv").join(if cfg!(windows) {
            "Scripts/python.exe"
        } else {
            "bin/python"
        })
    }

    pub fn managed_python(self, store: &ModelStore) -> Option<PathBuf> {
        let python = self.python(store);
        (python.is_file()
            && fs::read_to_string(self.root(store).join("installed"))
                .is_ok_and(|value| value == self.packages().join("\n")))
        .then_some(python)
    }

    pub fn discover_python(self, store: &ModelStore) -> PathBuf {
        self.select_python(store, env::var_os(self.override_env()).map(PathBuf::from))
    }

    fn select_python(self, store: &ModelStore, explicit: Option<PathBuf>) -> PathBuf {
        // Invalid explicit overrides fail at the probe; never silently replace them.
        explicit
            .or_else(|| self.managed_python(store))
            .or_else(|| {
                env::current_exe()
                    .ok()
                    .and_then(|exe| {
                        exe.parent().map(|dir| {
                            dir.join(if cfg!(windows) {
                                "python.exe"
                            } else {
                                "python3"
                            })
                        })
                    })
                    .filter(|path| path.is_file())
            })
            .unwrap_or_else(|| PathBuf::from(if cfg!(windows) { "python" } else { "python3" }))
    }

    pub fn hint(self) -> String {
        platform_rejection(self.target(), env::consts::OS, env::consts::ARCH)
            .map(str::to_string)
            .unwrap_or_else(|| format!("Run `werk backend install {}` and retry, or set {} to a compatible Python environment", self.target(), self.override_env()))
    }

    pub fn probe(self, store: &ModelStore) -> Result<String> {
        ensure_install_platform(self.target())?;
        let python = self.discover_python(store);
        let output = Command::new(&python)
            .args(["-I", "-B", "-c", self.validation()])
            .env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1")
            .output()
            .with_context(|| format!("Cannot execute {}; {}", python.display(), self.hint()))?;
        ensure!(
            output.status.success(),
            "{}: {}. {}",
            python.display(),
            String::from_utf8_lossy(&output.stderr).trim(),
            self.hint()
        );
        Ok(format!(
            "{} via {}",
            String::from_utf8_lossy(&output.stdout).trim(),
            python.display()
        ))
    }

    pub fn install(self, store: &ModelStore) -> Result<PathBuf> {
        ensure_install_platform(self.target())?;
        // Conservative shared wheel range; avoid choosing an unsupported newest Python.
        let check = if matches!(self, Self::Mlx | Self::MlxVlm) {
            "import sys,platform; assert (3,11) <= sys.version_info[:2] < (3,14); assert platform.machine() == 'arm64'; assert int(platform.mac_ver()[0].split('.')[0] or 0) >= 14"
        } else {
            "import sys; assert (3,11) <= sys.version_info[:2] < (3,14)"
        };
        let candidates: &[(&str, &[&str])] = if cfg!(windows) {
            &[
                ("python", &[]),
                ("python3", &[]),
                ("py", &["-3.12"]),
                ("py", &["-3.11"]),
                ("py", &["-3.13"]),
            ]
        } else {
            &[
                ("python3.12", &[]),
                ("python3.11", &[]),
                ("python3.13", &[]),
                ("python3", &[]),
            ]
        };
        let (bootstrap, bootstrap_args) = candidates.iter().find(|(candidate, args)| Command::new(candidate)
            .args(*args).args(["-I", "-c", check]).output().is_ok_and(|o| o.status.success()))
            .with_context(|| format!("Install Python 3.11–3.13 with venv on PATH, then retry werk backend install {}. MLX additionally requires native arm64 Python and macOS 14+.", self.target()))?;
        let python = self.install_with(store, bootstrap, bootstrap_args, |cmd| {
            let status = crate::terminal::command_status(cmd)?;
            ensure!(status.success(), "installer command failed");
            Ok(())
        }).with_context(|| format!("{} installation failed. Check the command output and platform wheel availability, then retry `werk backend install {}`; alternatively configure {}", self.target(), self.target(), self.override_env()))?;
        if env::var_os(self.override_env()).is_some() {
            crate::ui_eprintln!(
                "{} still overrides the managed runtime; unset it to use this installation.",
                self.override_env()
            );
        }
        if matches!(self, Self::Transformers | Self::Media) {
            crate::ui_println!(
                "PyTorch accelerator availability is reported above. For another CUDA/ROCm build, use the official PyTorch installer (https://pytorch.org/get-started/locally/) in a separate environment and select it with {}. Package installation does not install GPU drivers.",
                self.override_env()
            );
        }
        Ok(python)
    }

    fn install_with(
        self,
        store: &ModelStore,
        bootstrap: &str,
        bootstrap_args: &[&str],
        mut run: impl FnMut(&mut Command) -> Result<()>,
    ) -> Result<PathBuf> {
        let root = self.root(store);
        fs::create_dir_all(&root)?;
        let mut options = fs::OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let lock = options.open(root.join("install.lock"))?;
        fs2::FileExt::try_lock_exclusive(&lock)
            .context("installer locked; check permissions or wait for the other installation")?;
        let _lock = InstallLock(lock);
        let marker = root.join("installed");
        match fs::remove_file(&marker) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        let python = self.python(store);
        if !python.is_file() {
            run(Command::new(bootstrap)
                .args(bootstrap_args)
                .args(["-I", "-m", "venv"])
                .arg(root.join("venv")))?;
        }
        run(Command::new(&python).args(["-I", "-m", "pip", "install", "--upgrade", "pip"]))?;
        run(Command::new(&python)
            .args(["-I", "-m", "pip", "install", "--upgrade"])
            .args(self.packages()))?;
        run(Command::new(&python).args(["-I", "-m", "pip", "check"]))?;
        run(Command::new(&python).args(["-I", "-c", self.validation()]))?;
        ensure!(
            python.is_file(),
            "venv Python is missing after installation"
        );
        fs::write(marker, self.packages().join("\n"))?;
        Ok(python)
    }
}

struct InstallLock(fs::File);
impl Drop for InstallLock {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.0);
    }
}

/// Werk's supported adapter routes, not every experimental upstream port.
pub fn platform_rejection(target: &str, os: &str, arch: &str) -> Option<&'static str> {
    match target {
        "mlx" | "mlx-vlm" | "omlx" if os != "macos" || arch != "aarch64" => Some(
            "This managed MLX route requires macOS on Apple Silicon. Use werk backend install llama-cpu, llama-vulkan or transformers on supported hardware instead.",
        ),
        "llama-metal" if os != "macos" => Some(
            "Metal requires macOS. Use werk backend install llama-vulkan or llama-cpu instead.",
        ),
        "llama-cuda" | "llama-cuda-nvfp4" | "llama-cuda-offload" | "onnx-cuda" if os == "macos" => {
            Some("CUDA is not supported on macOS. Use werk backend install llama-metal instead.")
        }
        "llama-rocm" | "onnx-rocm" if os == "macos" => {
            Some("ROCm is not supported on macOS. Use werk backend install llama-metal instead.")
        }
        "onnx-rocm" if os != "linux" => Some(
            "The ONNX ROCm runner requires Linux. Use werk backend install onnx-cpu with a compatible runner bundle, or llama-vulkan for GGUF models.",
        ),
        "vllm" if os != "linux" || arch != "x86_64" => Some(
            "The managed vLLM installer requires Linux x86_64 (including WSL2). Use an upstream-supported external environment or a compatible backend from werk backend list; no third-party Windows/macOS port is installed.",
        ),
        "transformers" | "media" | "text-analysis" | "qwen-tts"
            if !matches!(
                (os, arch),
                ("linux", "x86_64" | "aarch64") | ("windows", "x86_64") | ("macos", "aarch64")
            ) =>
        {
            Some(
                "Managed Python ML packages require Linux x86_64/ARM64, Windows x86_64, or Apple Silicon with compatible upstream wheels. Use an external Python environment if upstream supports your host, or werk backend install llama-cpu.",
            )
        }
        _ => None,
    }
}

pub fn ensure_install_platform(target: &str) -> Result<()> {
    if let Some(reason) = platform_rejection(target, env::consts::OS, env::consts::ARCH) {
        anyhow::bail!("{reason}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_install_platform_matrix() {
        for (os, arch) in [
            ("linux", "x86_64"),
            ("linux", "aarch64"),
            ("windows", "x86_64"),
            ("macos", "aarch64"),
            ("macos", "x86_64"),
        ] {
            for target in ["mlx", "mlx-vlm", "omlx"] {
                assert_eq!(
                    platform_rejection(target, os, arch).is_none(),
                    os == "macos" && arch == "aarch64"
                );
            }
            assert!(platform_rejection("llama-cpu", os, arch).is_none());
            assert_eq!(
                platform_rejection("vllm", os, arch).is_none(),
                os == "linux" && arch == "x86_64"
            );
            assert_eq!(
                platform_rejection("llama-metal", os, arch).is_none(),
                os == "macos"
            );
        }
        assert!(platform_rejection("onnx-rocm", "windows", "x86_64").is_some());
        assert!(platform_rejection("transformers", "macos", "x86_64").is_some());
    }

    #[test]
    fn backend_install_discovery_failure_retry_and_override() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::resolve(Some(dir.path().join("home"))).unwrap();
        for backend in [
            PythonBackend::Mlx,
            PythonBackend::MlxVlm,
            PythonBackend::Transformers,
            PythonBackend::Media,
            PythonBackend::OnnxCpu,
        ] {
            assert!(backend.managed_python(&store).is_none());
            assert!(!backend.root(&store).exists());
            let python = backend.python(&store);
            let mut calls = 0;
            let failed = backend.install_with(&store, "bootstrap", &[], |_| {
                calls += 1;
                fs::create_dir_all(python.parent().unwrap())?;
                fs::write(&python, "fake interpreter")?;
                if calls == 3 {
                    anyhow::bail!("pip failed");
                }
                Ok(())
            });
            assert!(failed.is_err());
            assert!(backend.managed_python(&store).is_none());
            backend
                .install_with(&store, "bootstrap", &[], |command| {
                    assert_eq!(command.get_program(), python.as_os_str());
                    Ok(())
                })
                .unwrap();
            assert_eq!(backend.managed_python(&store), Some(python.clone()));
            assert_eq!(backend.select_python(&store, None), python);
            let override_path = PathBuf::from("nonexistent explicit interpreter");
            assert_eq!(
                backend.select_python(&store, Some(override_path.clone())),
                override_path
            );
            assert!(
                backend
                    .install_with(&store, "bootstrap", &[], |_| anyhow::bail!(
                        "validation failed"
                    ))
                    .is_err()
            );
            assert!(backend.managed_python(&store).is_none());
        }
    }
}
