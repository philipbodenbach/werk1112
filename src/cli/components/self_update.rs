//! Update the Werk executable using verified official release files.
use super::*;
const REPOSITORY: &str = "philipbodenbach/werk1112";

fn version(value: &str) -> Option<[u64; 3]> {
    value
        .trim_start_matches('v')
        .split('.')
        .map(str::parse)
        .collect::<Result<Vec<u64>, _>>()
        .ok()?
        .try_into()
        .ok()
}

fn supported() -> bool {
    matches!(
        (env::consts::OS, env::consts::ARCH),
        ("linux", "x86_64") | ("macos", "aarch64")
    ) || cfg!(feature = "release-linux-aarch64")
}

pub(super) fn component(offline: bool) -> Component {
    let mut row = Component {
        id: "werk".into(),
        name: "werk1112".into(),
        kind: "application".into(),
        status: "installed / unchecked".into(),
        installed: Some(env!("CARGO_PKG_VERSION").into()),
        available: None,
        detail: format!(
            "Current executable: {}. Self-update verifies the official archive SHA-256 and retains the previous executable. Restart Werk after updating.",
            env::current_exe().unwrap_or_default().display()
        ),
        required_by: vec![],
        action: None,
    };
    if !supported() {
        row.detail.push_str(" In-place replacement is unavailable on this platform; use the official installer or rebuild from source.");
        return row;
    }
    if offline {
        return row;
    }
    let result = (|| -> Result<String> {
        let client = reqwest::blocking::Client::builder()
            .user_agent("werk-component-manager")
            .timeout(Duration::from_secs(15))
            .build()?;
        let response = client
            .get(format!(
                "https://api.github.com/repos/{REPOSITORY}/releases/latest"
            ))
            .send()?
            .error_for_status()?;
        let mut bytes = Vec::new();
        response.take(2 * 1024 * 1024).read_to_end(&mut bytes)?;
        let data: Value = serde_json::from_slice(&bytes)?;
        let tag = data["tag_name"]
            .as_str()
            .context("Release has no version")?;
        anyhow::ensure!(
            version(tag).is_some() && data["prerelease"] == false && data["draft"] == false,
            "No stable release version found"
        );
        Ok(tag.trim_start_matches('v').into())
    })();
    match result {
        Ok(latest) => {
            row.available = Some(latest.clone());
            if version(&latest) > version(env!("CARGO_PKG_VERSION")) {
                row.status = "update available".into();
                row.action = Some(Action::UpdateWerk(latest));
            } else {
                row.status = "up to date".into();
            }
        }
        Err(e) => {
            row.status = "scan error".into();
            row.detail.push_str(&format!("\nRelease check failed: {e}"));
        }
    }
    row
}

pub(super) fn install(release: &str) -> Result<()> {
    anyhow::ensure!(
        supported()
            && version(release).is_some()
            && version(release) > version(env!("CARGO_PKG_VERSION")),
        "Not a newer supported stable Werk release"
    );
    let executable = env::current_exe()?;
    let parent = executable
        .parent()
        .context("Executable has no installation directory")?;
    let staging = tempfile::Builder::new()
        .prefix(".werk-update-")
        .tempdir_in(parent)
        .context("The Werk installation directory is not writable")?;
    let script = staging.path().join("install.sh");
    fs::write(&script, include_str!("../../../scripts/install.sh"))?;
    anyhow::ensure!(
        crate::terminal::command_status(
            Command::new("sh")
                .arg(&script)
                .env("WERK_VERSION", release)
                .env("WERK_REPO", REPOSITORY)
                .env("WERK_INSTALL_DIR", staging.path())
                .env("TMPDIR", staging.path())
        )?
        .success(),
        "Werk release download or checksum verification failed"
    );
    let binary = staging.path().join("werk");
    let (output, _) = updates::command_output(
        Command::new(&binary).arg("--version"),
        Duration::from_secs(10),
    )?;
    anyhow::ensure!(
        String::from_utf8_lossy(&output)
            .split_whitespace()
            .any(|s| s == release),
        "Downloaded executable reports a different version; current Werk retained"
    );
    let backup = tempfile::Builder::new()
        .prefix(".werk-previous-")
        .tempdir_in(parent)?
        .keep();
    let previous = backup.join(
        executable
            .file_name()
            .context("Invalid executable filename")?,
    );
    fs::copy(&executable, &previous).context("Could not preserve the previous executable")?;
    fs::rename(&binary, &executable)
        .context("Cannot replace the executable; previous version remains installed")?;
    crate::ui_println!(
        "Werk {release} installed. Previous executable: {}. Restart Werk.",
        previous.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stable_release_comparison_is_numeric_and_rejects_paths() {
        assert!(version("1.10.0") > version("1.9.9"));
        for bad in ["../1.2.3", "1.2.3-rc1", "1.2", "1.2.3.4"] {
            assert!(version(bad).is_none());
        }
    }
}
