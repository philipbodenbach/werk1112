use super::{action, media};
use crate::{model_store::ModelStore, model_support};
use anyhow::{Result, ensure};
use clap::Subcommand;
use std::path::PathBuf;

#[derive(Debug, Clone, Subcommand)]
pub enum SupportCommands {
    /// List upstream identity, implemented capabilities and honest verification status.
    List,
    /// Register an official hosted model; credentials remain in environment variables.
    Register {
        model: String,
        #[arg(long)]
        name: Option<String>,
    },
    /// Register a downloaded DROID policy package without copying weights.
    RegisterAction {
        name: String,
        #[arg(long)]
        checkpoint: PathBuf,
    },
    /// Generate/edit with FLUX 3 Image; request uses the documented BFL fields.
    Image {
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Generate Eleven v4 dialogue audio; request requires text/voice_id pairs.
    Speech {
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Predict DROID actions from synchronized base64 camera images and joint state.
    Action {
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}
fn save(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    file.write_all(bytes)?;
    Ok(())
}
pub fn run(store: &ModelStore, command: SupportCommands) -> Result<()> {
    match command {
        SupportCommands::List => {
            println!("{}", serde_json::to_string_pretty(model_support::CATALOG)?)
        }
        SupportCommands::Register { model, name } => {
            let m = model_support::register_remote(store, &model, name.as_deref())?;
            println!(
                "Registered {} ({model}); inference status IMPLEMENTED_UNTESTED",
                m.id
            );
        }
        SupportCommands::RegisterAction { name, checkpoint } => {
            ensure!(
                !name.is_empty()
                    && name
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || "-_".contains(c)),
                "invalid model name"
            );
            let checkpoint = checkpoint.canonicalize()?;
            ensure!(
                checkpoint.join("config.json").is_file()
                    && checkpoint.join("model.safetensors").is_file(),
                "unavailable_weights: expected a complete DROID package (config.json, model.safetensors and saved manifest/encoder references)"
            );
            ensure!(
                !store.model_dir(&name).join("manifest.json").exists(),
                "model already registered"
            );
            let mut manifest = store.link_path(&checkpoint, &name)?;
            manifest.architecture = Some("flux3_action_droid".into());
            manifest.backend = "flux-action".into();
            manifest.metadata = Default::default();
            manifest.metadata.family = Some("flux3-action-droid".into());
            store.write_manifest(&manifest)?;
            println!(
                "Registered {}; action inference status IMPLEMENTED_UNTESTED",
                manifest.id
            );
        }
        SupportCommands::Image { request, output } => {
            ensure!(!output.exists(), "output already exists");
            let request = serde_json::from_slice(&std::fs::read(request)?)?;
            let result = media::image(&request)?;
            save(&output, &result.bytes)?;
            println!("{}", result.metadata);
        }
        SupportCommands::Speech { request, output } => {
            ensure!(!output.exists(), "output already exists");
            let request = serde_json::from_slice(&std::fs::read(request)?)?;
            let result = media::speech(&request)?;
            save(&output, &result.bytes)?;
            println!("{}", result.metadata);
        }
        SupportCommands::Action { request, output } => {
            ensure!(!output.exists(), "output already exists");
            let request = serde_json::from_slice(&std::fs::read(request)?)?;
            let result = action::ActionBackend::default().execute(store, request)?;
            save(&output, &serde_json::to_vec_pretty(&result)?)?;
        }
    }
    Ok(())
}
