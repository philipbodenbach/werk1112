use super::http::{self, ProviderClient};
use crate::model_support::Provider;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    time::{Duration, Instant},
};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluxImageRequest {
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_tolerance: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grounding: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}
impl FluxImageRequest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.prompt.trim().is_empty(),
            "invalid_request: prompt must not be blank"
        );
        if let Some(images) = &self.images {
            ensure!(
                (1..=10).contains(&images.len()) && images.iter().all(|s| !s.trim().is_empty()),
                "invalid_request: images requires 1..10 URL/base64 references"
            );
        }
        if let Some(ratio) = &self.aspect_ratio {
            ensure!(
                [
                    "auto", "21:9", "2:1", "16:9", "3:2", "7:5", "4:3", "5:4", "1:1", "4:5", "3:4",
                    "5:7", "2:3", "9:16", "1:2", "9:21"
                ]
                .contains(&ratio.as_str()),
                "invalid_request: unsupported aspect_ratio"
            );
        }
        if let Some(resolution) = &self.resolution {
            ensure!(
                ["768sq", "1k", "1.5k", "2k", "4k"].contains(&resolution.as_str()),
                "invalid_request: unsupported resolution"
            );
        }
        ensure!(
            self.safety_tolerance.is_none_or(|s| s <= 4),
            "invalid_request: safety_tolerance must be 0..4"
        );
        ensure!(
            self.version.as_deref().is_none_or(|s| s == "latest"),
            "invalid_request: only version latest is documented"
        );
        Ok(())
    }
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DialogueInput {
    pub text: String,
    pub voice_id: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DialogueSettings {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub similarity: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stability: Option<f64>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeechRequest {
    pub inputs: Vec<DialogueInput>,
    #[serde(default = "default_audio_format")]
    pub output_format: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub settings: Option<DialogueSettings>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub apply_text_normalization: Option<String>,
}
fn default_audio_format() -> String {
    "mp3_44100_128".into()
}
impl SpeechRequest {
    pub fn validate(&self) -> Result<&'static str> {
        ensure!(
            !self.inputs.is_empty()
                && self
                    .inputs
                    .iter()
                    .all(|i| !i.text.trim().is_empty() && !i.voice_id.trim().is_empty()),
            "invalid_request: nonempty text and voice_id are required"
        );
        ensure!(
            self.inputs
                .iter()
                .map(|i| &i.voice_id)
                .collect::<HashSet<_>>()
                .len()
                <= 10,
            "invalid_request: at most ten voices"
        );
        // The dialogue endpoint documents a stricter reliable limit than the model card.
        ensure!(
            self.inputs
                .iter()
                .map(|i| i.text.chars().count())
                .sum::<usize>()
                <= 2000,
            "invalid_request: dialogue requests are limited to 2000 characters; split longer scripts"
        );
        if let Some(settings) = &self.settings {
            ensure!(
                settings
                    .similarity
                    .is_none_or(|s| s.is_finite() && (0.0..=1.0).contains(&s))
                    && settings
                        .stability
                        .is_none_or(|s| s.is_finite() && (0.0..=1.0).contains(&s)),
                "invalid_request: stability must be 0..1"
            );
        }
        ensure!(
            self.apply_text_normalization
                .as_deref()
                .is_none_or(|s| ["auto", "on", "off"].contains(&s)),
            "invalid_request: normalization must be auto, on or off"
        );
        ensure!(
            self.language_code.as_deref().is_none_or(
                |s| (2..=3).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_lowercase())
            ),
            "invalid_request: language_code must be an ISO language code"
        );
        audio_mime(&self.output_format)
    }
    pub fn body(&self) -> Result<Value> {
        self.validate()?;
        let mut body = serde_json::to_value(self)?;
        body.as_object_mut().unwrap().remove("output_format");
        body.as_object_mut().unwrap().remove("stream");
        body["model_id"] = json!("eleven_v4");
        Ok(body)
    }
}
pub fn audio_mime(format: &str) -> Result<&'static str> {
    let formats = [
        "mp3_22050_32",
        "mp3_24000_48",
        "mp3_44100_32",
        "mp3_44100_64",
        "mp3_44100_96",
        "mp3_44100_128",
        "mp3_44100_192",
        "pcm_8000",
        "pcm_16000",
        "pcm_22050",
        "pcm_24000",
        "pcm_32000",
        "pcm_44100",
        "pcm_48000",
        "wav_8000",
        "wav_16000",
        "wav_22050",
        "wav_24000",
        "wav_32000",
        "wav_44100",
        "wav_48000",
        "ulaw_8000",
        "alaw_8000",
        "opus_48000_32",
        "opus_48000_64",
        "opus_48000_96",
        "opus_48000_128",
        "opus_48000_192",
    ];
    ensure!(
        formats.contains(&format),
        "invalid_request: unsupported output_format"
    );
    Ok(if format.starts_with("mp3_") {
        "audio/mpeg"
    } else if format.starts_with("wav_") {
        "audio/wav"
    } else if format.starts_with("opus_") {
        "audio/ogg"
    } else if format.starts_with("ulaw_") {
        "audio/basic"
    } else if format.starts_with("alaw_") {
        "audio/pcma"
    } else {
        "audio/pcm"
    })
}
pub struct MediaOutput {
    pub bytes: Vec<u8>,
    pub mime: String,
    pub metadata: Value,
}
pub fn image(request: &FluxImageRequest) -> Result<MediaOutput> {
    image_with(http::client(Provider::Bfl)?.as_ref(), request)
}
pub(crate) fn image_with(
    client: &ProviderClient,
    request: &FluxImageRequest,
) -> Result<MediaOutput> {
    request.validate()?;
    let submission =
        http::read_json(client.post("/v1/flux-3-image", &serde_json::to_value(request)?)?)?;
    let polling = submission["polling_url"]
        .as_str()
        .ok_or_else(|| http::error("upstream_schema", "missing polling_url"))?;
    client.validate_poll_url(polling)?;
    let started = Instant::now();
    loop {
        ensure!(
            started.elapsed() < Duration::from_secs(600),
            "upstream_timeout: image polling exceeded ten minutes; submission is not replayed"
        );
        let result = http::read_json(client.get(polling, true)?)?;
        match result["status"].as_str() {
            Some("Pending" | "Reasoning" | "Generating") => {
                std::thread::sleep(Duration::from_millis(1500))
            }
            Some("Ready") => {
                let sample = result["result"]["sample"]
                    .as_str()
                    .ok_or_else(|| http::error("upstream_schema", "missing image sample"))?;
                let url = reqwest::Url::parse(sample)
                    .map_err(|_| http::error("upstream_schema", "invalid sample URL"))?;
                let secure_download = url.scheme() == "https";
                #[cfg(test)]
                let secure_download = secure_download
                    || (url.scheme() == "http"
                        && sample.starts_with(&format!("{}/", client.origin)));
                ensure!(
                    secure_download && url.username().is_empty() && url.password().is_none(),
                    "upstream_schema: expected HTTPS image download"
                );
                // Signed storage URL: deliberately never attach provider credentials.
                let response = client.get(sample, false)?;
                let bytes = http::read_bytes(response, 64 * 1024 * 1024)?;
                let mime = if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
                    "image/png"
                } else if bytes.starts_with(b"\xff\xd8\xff") {
                    "image/jpeg"
                } else {
                    return Err(http::error(
                        "upstream_schema",
                        "provider did not return a PNG/JPEG image",
                    ));
                };
                return Ok(MediaOutput {
                    bytes,
                    mime: mime.into(),
                    metadata: json!({"id": submission["id"], "cost": submission["cost"], "status":"Ready", "duration": result["result"]["duration"]}),
                });
            }
            Some("Request Moderated" | "Content Moderated") => {
                return Err(http::error(
                    "content_moderated",
                    "BFL moderated the request or output",
                ));
            }
            _ => {
                return Err(http::error(
                    "upstream_error",
                    "BFL image task failed or returned an unknown status",
                ));
            }
        }
    }
}
pub fn speech_response(request: &SpeechRequest) -> Result<reqwest::blocking::Response> {
    speech_response_with(http::client(Provider::ElevenLabs)?.as_ref(), request)
}
pub(crate) fn speech_response_with(
    client: &ProviderClient,
    request: &SpeechRequest,
) -> Result<reqwest::blocking::Response> {
    let body = request.body()?;
    client.post(
        &format!(
            "/v1/text-to-dialogue{}?output_format={}",
            if request.stream { "/stream" } else { "" },
            request.output_format
        ),
        &body,
    )
}
pub fn speech(request: &SpeechRequest) -> Result<MediaOutput> {
    let mime = request.validate()?.to_string();
    let bytes = http::read_bytes(speech_response(request)?, 64 * 1024 * 1024)?;
    ensure!(!bytes.is_empty(), "upstream_schema: empty audio response");
    Ok(MediaOutput {
        bytes,
        mime,
        metadata: json!({"model":"eleven_v4","output_format":request.output_format}),
    })
}
