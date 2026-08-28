use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

use crate::{
    capabilities::InferenceTask,
    inference::{InferenceRequest, ParameterValue},
};

use super::compatibility::{harmless_default, is_meaningful_name};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(in crate::api) struct Txt2ImgRequest {
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    negative_prompt: String,
    #[serde(default = "default_seed")]
    seed: i64,
    #[serde(default = "default_subseed")]
    subseed: i64,
    #[serde(default)]
    subseed_strength: f64,
    #[serde(default = "default_seed_resize")]
    seed_resize_from_h: i64,
    #[serde(default = "default_seed_resize")]
    seed_resize_from_w: i64,
    #[serde(default = "default_true")]
    seed_enable_extras: bool,
    #[serde(default)]
    sampler_name: Option<String>,
    #[serde(default)]
    sampler_index: Option<String>,
    #[serde(default)]
    scheduler: Option<String>,
    #[serde(default = "default_one")]
    batch_size: u32,
    #[serde(default = "default_one")]
    n_iter: u32,
    #[serde(default = "default_steps")]
    steps: u32,
    #[serde(default = "default_cfg_scale")]
    cfg_scale: f64,
    #[serde(default = "default_dimension")]
    width: u32,
    #[serde(default = "default_dimension")]
    height: u32,
    #[serde(default)]
    restore_faces: Option<bool>,
    #[serde(default)]
    tiling: Option<bool>,
    #[serde(default)]
    styles: Option<Vec<String>>,
    #[serde(default)]
    eta: Option<f64>,
    #[serde(default)]
    ddim_discretize: Option<String>,
    #[serde(default = "default_s_noise")]
    s_noise: Option<f64>,
    #[serde(default = "default_denoising_strength")]
    denoising_strength: Option<f64>,
    #[serde(default = "default_override_settings")]
    override_settings: Option<BTreeMap<String, Value>>,
    #[serde(default = "default_true")]
    override_settings_restore_afterwards: bool,
    #[serde(default)]
    refiner_checkpoint: Option<String>,
    #[serde(default)]
    refiner_switch_at: Option<f64>,
    #[serde(default)]
    enable_hr: bool,
    #[serde(default = "default_hr_scale")]
    hr_scale: f64,
    #[serde(default)]
    hr_upscaler: Option<String>,
    #[serde(default)]
    hr_second_pass_steps: u32,
    #[serde(default)]
    hr_resize_x: u32,
    #[serde(default)]
    hr_resize_y: u32,
    #[serde(default)]
    hr_checkpoint_name: Option<String>,
    #[serde(default)]
    hr_sampler_name: Option<String>,
    #[serde(default)]
    hr_scheduler: Option<String>,
    #[serde(default)]
    hr_prompt: String,
    #[serde(default)]
    hr_negative_prompt: String,
    #[serde(default)]
    script_name: Option<String>,
    #[serde(default = "default_script_args")]
    script_args: Option<Vec<Value>>,
    #[serde(default = "default_alwayson_scripts")]
    alwayson_scripts: Option<BTreeMap<String, Value>>,
    #[serde(default = "default_true")]
    send_images: bool,
    #[serde(default)]
    save_images: bool,
    #[serde(default)]
    force_task_id: Option<String>,
    #[serde(default)]
    infotext: Option<String>,
    #[serde(default, flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
pub(in crate::api) struct ProgressQuery {
    #[serde(default)]
    pub(super) skip_current_image: bool,
}

impl Txt2ImgRequest {
    pub(super) fn validate(&self) -> Result<(), String> {
        if self.seed < -1 {
            return Err("seed must be -1 or a non-negative integer".to_string());
        }
        if self.batch_size == 0 || self.n_iter == 0 {
            return Err("batch_size and n_iter must be greater than zero".to_string());
        }
        self.batch_size
            .checked_mul(self.n_iter)
            .ok_or_else(|| "batch_size multiplied by n_iter is too large".to_string())?;
        if self.enable_hr {
            return Err(
                "enable_hr=true is not supported by the Werk AUTOMATIC1111 adapter".to_string(),
            );
        }
        if self.restore_faces == Some(true) {
            return Err(
                "restore_faces=true is not supported by the selected local adapter".to_string(),
            );
        }
        if self.tiling == Some(true) {
            return Err(
                "tiling=true requests seamless generation, which Werk does not currently expose"
                    .to_string(),
            );
        }
        if self
            .styles
            .as_ref()
            .is_some_and(|styles| !styles.is_empty())
        {
            return Err("named AUTOMATIC1111 prompt styles are not supported".to_string());
        }
        if self
            .refiner_checkpoint
            .as_deref()
            .is_some_and(is_meaningful_name)
        {
            return Err(
                "refiner_checkpoint is not supported by this compatibility adapter".to_string(),
            );
        }
        if self
            .script_name
            .as_deref()
            .is_some_and(|name| !name.trim().is_empty())
            || self
                .script_args
                .as_ref()
                .is_some_and(|arguments| !arguments.is_empty())
            || self
                .alwayson_scripts
                .as_ref()
                .is_some_and(|scripts| !scripts.is_empty())
        {
            return Err("AUTOMATIC1111 scripts and alwayson_scripts are not supported".to_string());
        }
        if self
            .infotext
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
        {
            return Err("infotext parameter restoration is not supported".to_string());
        }
        if self.save_images {
            return Err(
                "save_images=true is not supported; embedded results are returned without retaining a managed output directory"
                    .to_string(),
            );
        }
        if !self.override_settings_restore_afterwards {
            return Err(
                "override_settings_restore_afterwards=false is not supported; use POST /sdapi/v1/options for a process-wide checkpoint selection"
                    .to_string(),
            );
        }
        if self.subseed_strength != 0.0 {
            return Err(
                "subseed_strength is not supported by the selected local adapter".to_string(),
            );
        }
        if !matches!(self.seed_resize_from_h, -1 | 0) || !matches!(self.seed_resize_from_w, -1 | 0)
        {
            return Err(
                "seed_resize_from_h/seed_resize_from_w are only accepted at their disabled sentinel values (-1 or 0)"
                    .to_string(),
            );
        }
        if self.s_noise.is_some_and(|value| value != 1.0) {
            return Err(
                "s_noise is only accepted at the AUTOMATIC1111 default value 1.0".to_string(),
            );
        }
        if self
            .ddim_discretize
            .as_deref()
            .is_some_and(|value| !value.eq_ignore_ascii_case("uniform"))
        {
            return Err(
                "ddim_discretize is only accepted at the AUTOMATIC1111 default value 'uniform'"
                    .to_string(),
            );
        }
        if let Some(settings) = &self.override_settings {
            for (name, value) in settings {
                match name.as_str() {
                    "sd_model_checkpoint" => {
                        if !value.is_string() {
                            return Err("override_settings.sd_model_checkpoint must be a string"
                                .to_string());
                        }
                    }
                    "CLIP_stop_at_last_layers" if value.as_u64() == Some(1) => {}
                    _ if harmless_default(value) => {}
                    _ => {
                        return Err(format!(
                            "override_settings.{name} is not supported by the Werk AUTOMATIC1111 adapter"
                        ));
                    }
                }
            }
        }
        for (name, value) in &self.extra {
            if !harmless_default(value) {
                return Err(format!(
                    "AUTOMATIC1111 parameter '{name}' is not supported by the selected local adapter"
                ));
            }
        }
        Ok(())
    }

    pub(super) fn checkpoint_override(&self) -> Option<&str> {
        self.override_settings
            .as_ref()?
            .get("sd_model_checkpoint")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
    }

    pub(super) fn compatibility_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if self.sampler_name.is_some() || self.sampler_index.is_some() {
            warnings.push(
                "sampler_name/sampler_index was accepted for protocol compatibility; Werk selected the executable runtime sampler"
                    .to_string(),
            );
        }
        if self
            .scheduler
            .as_deref()
            .is_some_and(|value| !value.eq_ignore_ascii_case("automatic"))
        {
            warnings.push(
                "scheduler was accepted for protocol compatibility; Werk selected the executable runtime scheduler"
                    .to_string(),
            );
        }
        if self.eta.is_some_and(|value| value != 0.0) {
            warnings.push(
                "eta was accepted for protocol compatibility but is not applied by the selected local adapter"
                    .to_string(),
            );
        }
        warnings
    }

    pub(super) fn normalized_parameters(&self) -> Value {
        let mut value = serde_json::to_value(self).unwrap_or_else(|_| json!({}));
        if let Some(parameters) = value.as_object_mut()
            && parameters.get("sampler_index").is_none_or(Value::is_null)
        {
            parameters.insert("sampler_index".to_string(), json!("Euler"));
        }
        value
    }

    pub(super) fn seed(&self) -> i64 {
        self.seed
    }

    pub(super) fn send_images(&self) -> bool {
        self.send_images
    }

    pub(super) fn steps(&self) -> u32 {
        self.steps
    }

    pub(super) fn batch_size(&self) -> u32 {
        self.batch_size
    }

    pub(super) fn image_count(&self) -> u32 {
        self.batch_size.saturating_mul(self.n_iter)
    }

    pub(super) fn into_inference(self, model: String, seed: u64) -> InferenceRequest {
        let image_count = self.batch_size.saturating_mul(self.n_iter);
        let mut request = InferenceRequest::new(model, InferenceTask::ImageGeneration);
        request.prompt = Some(self.prompt);
        request.negative_prompt =
            (!self.negative_prompt.is_empty()).then_some(self.negative_prompt);
        request
            .parameters
            .insert("image.width".to_string(), self.width.into());
        request
            .parameters
            .insert("image.height".to_string(), self.height.into());
        request
            .parameters
            .insert("image.steps".to_string(), self.steps.into());
        request
            .parameters
            .insert("image.guidance".to_string(), self.cfg_scale.into());
        request
            .parameters
            .insert("image.seed".to_string(), seed.into());
        request
            .parameters
            .insert("image.num_images".to_string(), image_count.into());
        request.parameters.insert(
            "image.output_format".to_string(),
            ParameterValue::from("png"),
        );
        request
    }
}

fn default_seed() -> i64 {
    -1
}

fn default_subseed() -> i64 {
    -1
}

fn default_seed_resize() -> i64 {
    -1
}

fn default_one() -> u32 {
    1
}

fn default_steps() -> u32 {
    50
}

fn default_cfg_scale() -> f64 {
    7.0
}

fn default_dimension() -> u32 {
    512
}

fn default_s_noise() -> Option<f64> {
    Some(1.0)
}

fn default_denoising_strength() -> Option<f64> {
    Some(0.75)
}

fn default_override_settings() -> Option<BTreeMap<String, Value>> {
    Some(BTreeMap::new())
}

fn default_script_args() -> Option<Vec<Value>> {
    Some(Vec::new())
}

fn default_alwayson_scripts() -> Option<BTreeMap<String, Value>> {
    Some(BTreeMap::new())
}

fn default_hr_scale() -> f64 {
    2.0
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn request_rejects_meaningful_unsupported_features() {
        let request: Txt2ImgRequest = serde_json::from_value(json!({
            "prompt": "robot",
            "enable_hr": true
        }))
        .unwrap();
        assert!(request.validate().unwrap_err().contains("enable_hr"));

        let request: Txt2ImgRequest = serde_json::from_value(json!({
            "prompt": "robot",
            "enable_hr": false,
            "alwayson_scripts": {},
            "script_args": [],
            "unknown_false_default": false
        }))
        .unwrap();
        request.validate().unwrap();
    }
}
