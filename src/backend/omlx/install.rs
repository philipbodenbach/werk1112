//! Explicit, isolated oMLX provisioning. Discovery never calls the installer.
use super::*;
use anyhow::ensure;

pub(super) const VERSION: &str = "0.7.0";
const PACKAGE: &str = "git+https://github.com/jundot/omlx.git@v0.7.0";
pub(super) const INSTALL_HINT: &str = "oMLX is not installed. Run `werk backend install omlx`, then retry; alternatively set WERK_OMLX_BIN to an existing oMLX Python CLI launcher";
const PYTHON_CHECK: &str = "import sys, platform; sys.exit(0 if (3,11) <= sys.version_info[:2] < (3,14) and platform.machine() == 'arm64' and int(platform.mac_ver()[0].split('.')[0] or 0) >= 15 else 1)";
const VALIDATE: &str = "import importlib.metadata as m; import mlx.core as mx; import mlx_lm; from omlx.cli import main; assert m.version('omlx') == '0.7.0'; assert mx.metal.is_available(), 'Metal is unavailable'; print('oMLX ' + m.version('omlx') + ' ready (Metal)')";

fn root(store: &ModelStore) -> PathBuf {
    store.home().join("backends/omlx")
}
pub(super) fn python(store: &ModelStore) -> PathBuf {
    root(store).join("venv/bin/python")
}
pub(super) fn executable(store: &ModelStore) -> PathBuf {
    root(store).join("venv/bin/omlx")
}
fn marker(store: &ModelStore) -> PathBuf {
    root(store).join("installed-version")
}
pub(super) fn installed(store: &ModelStore) -> bool {
    fs::read_to_string(marker(store)).is_ok_and(|v| v.trim() == VERSION)
        && python(store).is_file()
        && executable(store).is_file()
}

pub fn install(store: &ModelStore) -> Result<PathBuf> {
    ensure!(
        cfg!(all(target_os = "macos", target_arch = "aarch64")),
        "oMLX installation requires macOS 15+ on Apple Silicon; use a compatible backend from werk backend list on this platform"
    );
    let bootstrap = ["python3.13", "python3.12", "python3.11", "python3"].into_iter()
        .filter_map(find_program)
        .find(|path| Command::new(path).args(["-I", "-c", PYTHON_CHECK]).output()
            .is_ok_and(|output| output.status.success()))
        .context("oMLX needs native arm64 Python 3.11–3.13 with venv on macOS 15+. Install a compatible Python on PATH, then retry werk backend install omlx")?;
    ensure!(
        Command::new("git")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success()),
        "oMLX installation needs Git for its pinned dependencies. Install Git / Xcode Command Line Tools, then retry werk backend install omlx"
    );
    install_with(store, &bootstrap, |command, detail| {
        let status =
            crate::terminal::command_status(command).with_context(|| detail.to_string())?;
        ensure!(
            status.success(),
            "{detail}. Check the installer output, network access and Python/Metal requirements, then retry werk backend install omlx"
        );
        Ok(())
    })
}

struct InstallLock(fs::File);
impl Drop for InstallLock {
    fn drop(&mut self) {
        // Release explicitly: another thread may briefly fork with a duplicate
        // descriptor before exec closes it, outliving this File's drop.
        let _ = fs2::FileExt::unlock(&self.0);
    }
}

fn install_with(
    store: &ModelStore,
    bootstrap: &Path,
    mut run: impl FnMut(&mut Command, &str) -> Result<()>,
) -> Result<PathBuf> {
    let root = root(store);
    fs::create_dir_all(&root)?;
    let mut options = fs::OpenOptions::new();
    options.create(true).read(true).write(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let lock = options.open(root.join("install.lock"))?;
    fs2::FileExt::try_lock_exclusive(&lock)
        .context("cannot lock the oMLX installer; check directory permissions or wait for another installation to finish")?;
    let _lock = InstallLock(lock);
    // A failed update must not be discovered as a complete managed installation.
    match fs::remove_file(marker(store)) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    let python = python(store);
    if !python.is_file() {
        crate::ui_eprintln!(
            "Creating oMLX environment at {}",
            root.join("venv").display()
        );
        run(
            Command::new(bootstrap)
                .args(["-I", "-m", "venv"])
                .arg(root.join("venv")),
            "Could not create the oMLX virtualenv",
        )?;
    }
    run(
        Command::new(&python).args(["-I", "-c", PYTHON_CHECK]),
        "The managed oMLX Python is incompatible; use native arm64 Python 3.11–3.13 on macOS 15+",
    )?;
    crate::ui_eprintln!(
        "Installing oMLX {VERSION} into {}",
        root.join("venv").display()
    );
    run(
        Command::new(&python).args(["-I", "-m", "pip", "install", "--upgrade", "pip"]),
        "Could not update pip in the oMLX environment",
    )?;
    // Upstream's versioned source declares the compatible MLX dependency pins.
    run(
        Command::new(&python).args(["-I", "-m", "pip", "install", PACKAGE]),
        "Could not install oMLX",
    )?;
    run(
        Command::new(&python).args(["-I", "-c", VALIDATE]),
        "oMLX import/Metal validation failed",
    )?;
    let launcher = executable(store);
    ensure!(
        python.is_file() && launcher.is_file(),
        "oMLX installation did not create its Python CLI; retry werk backend install omlx"
    );
    fs::write(marker(store), VERSION)?;
    Ok(launcher)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn omlx_install_failure_is_not_discoverable_and_retry_uses_managed_environment() {
        let temp = tempfile::tempdir().unwrap();
        let store = ModelStore::resolve(Some(temp.path().join("store with spaces"))).unwrap();
        let mut commands = vec![];
        let mut fake = |command: &mut Command, _: &str| -> Result<()> {
            let args = command
                .get_args()
                .map(|s| s.to_string_lossy().into_owned())
                .collect::<Vec<_>>();
            commands.push((command.get_program().to_owned(), args.clone()));
            if args.iter().any(|a| a == "venv") {
                fs::create_dir_all(python(&store).parent().unwrap())?;
                fs::write(python(&store), "python fixture")?;
            }
            if args.iter().any(|a| a == PACKAGE) {
                fs::write(executable(&store), "console fixture")?;
                bail!("simulated pip failure");
            }
            Ok(())
        };
        assert!(install_with(&store, Path::new("python-bootstrap"), &mut fake).is_err());
        assert!(!installed(&store));
        assert!(commands.iter().any(|(_, a)| a.iter().any(|v| v == PACKAGE)));
        commands.clear();
        let path = install_with(&store, Path::new("python-bootstrap"), |command, _| {
            commands.push((
                command.get_program().to_owned(),
                command
                    .get_args()
                    .map(|s| s.to_string_lossy().into_owned())
                    .collect::<Vec<_>>(),
            ));
            Ok(())
        })
        .unwrap();
        assert_eq!(path, executable(&store));
        assert!(installed(&store));
        assert!(
            commands
                .iter()
                .all(|(p, _)| p == python(&store).as_os_str())
        );
        assert!(commands.last().unwrap().1.iter().any(|a| a == VALIDATE));
        assert!(
            install_with(&store, Path::new("python-bootstrap"), |command, _| {
                if command.get_args().any(|arg| arg == VALIDATE) {
                    bail!("simulated Metal validation failure");
                }
                Ok(())
            })
            .is_err()
        );
        assert!(!installed(&store));
    }
    #[test]
    fn omlx_install_rejects_other_platforms_before_creating_files() {
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let store = ModelStore::resolve(Some(temp.path().join("untouched"))).unwrap();
        assert!(
            install(&store)
                .unwrap_err()
                .to_string()
                .contains("Apple Silicon")
        );
        assert!(!store.home().exists());
    }
}
