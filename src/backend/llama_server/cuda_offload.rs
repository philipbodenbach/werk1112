//! Optional native runtime; stock CUDA installs keep using upstream llama.cpp.

use super::*;

pub(super) const REVISION: &str = "907a73da9a149faa8c42ccde890f1d575586810f";
const SOURCE: &str = "https://github.com/GenerelSchwerz/llama.cpp";
const PREFETCH: &str = "ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);";
const DEMAND: &str = "ml.init_mappings(params.moe_expert_cache_slots == 0, use_mlock ? &pimpl->mlock_mmaps : nullptr);";

pub(super) fn prepare_source(directory: &Path, verbose: bool) -> Result<()> {
    if !directory.join(".git").is_dir() {
        if directory.exists() {
            bail!(
                "offload source path already exists without a Git checkout: {}",
                directory.display()
            );
        }
        run_command(
            Command::new("git").arg("init").arg(directory),
            "cannot initialize CUDA offload source",
            verbose,
        )?;
        run_command(
            Command::new("git")
                .arg("-C")
                .arg(directory)
                .args(["fetch", "--depth", "1", SOURCE, REVISION]),
            "cannot fetch pinned CUDA offload runtime",
            verbose,
        )?;
        run_command(
            Command::new("git")
                .arg("-C")
                .arg(directory)
                .args(["checkout", "--detach", REVISION]),
            "cannot select pinned CUDA offload runtime",
            verbose,
        )?;
    }
    let output = Command::new("git")
        .arg("-C")
        .arg(directory)
        .args(["rev-parse", "HEAD"])
        .output()?;
    if !output.status.success() || String::from_utf8_lossy(&output.stdout).trim() != REVISION {
        bail!("CUDA offload source revision differs from the tested pin; refusing to overwrite it");
    }
    let loader = directory.join("src/llama-model.cpp");
    let source = fs::read_to_string(&loader)?;
    if source.contains(DEMAND) {
        return Ok(());
    }
    if source.matches(PREFETCH).count() != 1 {
        bail!("CUDA offload source does not match the on-demand loading patch");
    }
    // MAP_POPULATE otherwise reads every expert at startup, even with a bounded GPU cache.
    fs::write(loader, source.replacen(PREFETCH, DEMAND, 1))?;
    Ok(())
}

pub(super) fn validate_runtime(executable: &Path) -> Result<()> {
    let output = Command::new(executable).arg("--help").output()?;
    let help = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !output.status.success()
        || ![
            "--moe-expert-cache-size",
            "--moe-expert-cache-host-pinned-mb",
            "--lazy-mode",
            "--slot-save-path",
        ]
        .iter()
        .all(|flag| help_has_exact_option(&help, flag))
    {
        bail!(
            "built CUDA runtime does not advertise the required expert, lazy-row and snapshot controls"
        );
    }
    Ok(())
}
